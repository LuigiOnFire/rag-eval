#!/usr/bin/env python3
"""
Train decoder model using SFT with special action tokens.

This script trains a decoder model on trajectories formatted with special tokens
that trigger retrieval, generation, and other actions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import argparse
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from action_tokens import add_special_tokens_to_tokenizer, ALL_ACTION_TOKENS


def load_sft_examples(filepath: str) -> list:
    """Load SFT examples from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get('examples', [])


def prepare_dataset(examples: list, tokenizer, max_length: int = 2048):
    """
    Prepare dataset for SFT training.
    
    Args:
        examples: List of dicts with 'input' and 'output' keys
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace Dataset
    """
    # Format as input-output pairs
    texts = []
    for ex in examples:
        # Format: <input>\n<output>
        text = f"{ex['input']}\n{ex['output']}"
        texts.append(text)
    
    # Create dataset
    dataset = Dataset.from_dict({'text': texts})
    
    # Tokenize
    def tokenize_function(examples):
        # Tokenize with padding and truncation
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        # For language modeling, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train decoder with SFT")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Base model to fine-tune")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to SFT examples JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"models/decoder_sft_{timestamp}"
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training Configuration:")
    print(f"  Base model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    # Load tokenizer and model
    print(f"Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Add special tokens
    print(f"Adding {len(ALL_ACTION_TOKENS)} special tokens...")
    num_added = add_special_tokens_to_tokenizer(tokenizer)
    print(f"  Added {num_added} new tokens")
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Load data
    print(f"\nLoading training data...")
    examples = load_sft_examples(args.data)
    print(f"  Loaded {len(examples)} examples")
    
    # Prepare dataset
    print(f"Preparing dataset...")
    dataset = prepare_dataset(examples, tokenizer, args.max_length)
    print(f"  Tokenized {len(dataset)} examples")
    
    # Split train/validation (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))
    
    # Save config
    config = {
        'base_model': args.model,
        'data_file': args.data,
        'num_examples': len(examples),
        'special_tokens': ALL_ACTION_TOKENS,
        'training_args': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_length': args.max_length,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {output_dir / 'final'}")
    print(f"✓ Config saved to: {output_dir / 'training_config.json'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
