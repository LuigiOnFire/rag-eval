#!/usr/bin/env python3
"""
Train controller with class weighting and test at various epochs.

This script:
1. Adds class weights to handle imbalanced data
2. Trains without early stopping to see full epoch progression
3. Tests the controller at each checkpoint

Usage:
    python scripts/train_and_test_epochs.py --trajectories results/trajectories_100.json --epochs 15
"""

import argparse
import json
import logging
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.behavior_cloning import Controller, Action, ACTION_NAMES, ControllerConfig
from src.green_search import load_cost_table
from src.retriever import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeightedTrajectoryDataset(Dataset):
    """PyTorch Dataset for (state, action) pairs."""
    
    def __init__(
        self, 
        states: List[str], 
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.states = states
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.states[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    """Trainer with class weights for imbalanced data."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: List[int], num_classes: int = 8) -> List[float]:
    """Compute inverse frequency class weights."""
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1
    
    # Inverse frequency weighting
    total = sum(counts)
    weights = []
    for count in counts:
        if count > 0:
            weights.append(total / (num_classes * count))
        else:
            weights.append(1.0)  # Default weight for unseen classes
    
    # Normalize so mean weight is 1
    mean_weight = sum(weights) / len(weights)
    weights = [w / mean_weight for w in weights]
    
    return weights


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0))
    }


def test_controller_quick(
    controller: Controller,
    retriever: BM25Retriever,
    queries: List[Dict],
    cost_table: Dict,
    max_steps: int = 10
) -> Dict:
    """Quick test of controller on queries - runs FULL trajectories."""
    import ollama
    
    results = []
    action_counts = {}
    total_energy = 0.0
    correct = 0
    
    for q in queries:
        query = q["question"]
        expected = q["answer"]
        
        # Run full trajectory
        observations = []
        retrieved_docs = []
        trajectory_actions = []
        trajectory_energy = 0.0
        answer = None
        
        for step in range(max_steps):
            # Build state with observations
            state_parts = [f"[CLS] {query}"]
            for obs in observations[-3:]:  # Last 3 observations
                state_parts.append(f"[SEP] {obs[:200]}")
            state = " ".join(state_parts)
            
            # Get controller prediction
            action, probs = controller.predict(state)
            action_name = action.name
            trajectory_actions.append(action_name)
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            # Track energy
            cost = cost_table.get(action_name, 20.0)
            trajectory_energy += cost
            
            # Execute action
            if "RETRIEVE" in action_name:
                # Retrieval - returns (passages, scores)
                passages, scores = retriever.retrieve(query, top_k=3)
                obs = " | ".join([r["text"][:150] for r in passages])
                observations.append(f"Retrieved: {obs}")
                retrieved_docs.extend([r["text"] for r in passages])
                
            elif "DECOMPOSE" in action_name:
                # Decompose query
                model = "llama3:8b" if "LLM" in action_name else "mistral:latest"
                prompt = f"Break this into 2 simpler questions:\n{query}\nSub-questions:"
                response = ollama.generate(model=model, prompt=prompt)
                observations.append(f"Sub-questions: {response['response'].strip()[:200]}")
                
            elif "GENERATE" in action_name:
                # Generate answer - this is terminal
                model = "llama3:8b" if "LLM" in action_name else "mistral:latest"
                context = "\n".join(retrieved_docs[-3:]) if retrieved_docs else ""
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely:"
                response = ollama.generate(model=model, prompt=prompt)
                answer = response['response'].strip()
                break  # Terminal action
                
            elif "REASON" in action_name:
                # Reasoning step
                model = "llama3:8b" if "LLM" in action_name else "mistral:latest"
                context = "\n".join(observations[-2:]) if observations else ""
                prompt = f"Given: {context}\n\nReason about: {query}"
                response = ollama.generate(model=model, prompt=prompt)
                observations.append(f"Reasoning: {response['response'].strip()[:200]}")
        
        total_energy += trajectory_energy
        
        # Check correctness
        if expected and answer:
            is_correct = (
                expected.lower() in answer.lower() or
                answer.lower() in expected.lower()
            )
            if is_correct:
                correct += 1
        
        results.append({
            "query": query,
            "actions": trajectory_actions,
            "answer": answer[:100] if answer else None,
            "expected": expected,
            "steps": len(trajectory_actions)
        })
    
    return {
        "accuracy": correct / len(queries) if queries else 0,
        "action_distribution": action_counts,
        "avg_energy": total_energy / len(queries) if queries else 0,
        "avg_steps": sum(r["steps"] for r in results) / len(results) if results else 0,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Train with class weighting and test epochs")
    parser.add_argument("--trajectories", type=str, required=True,
                       help="Path to trajectories JSON")
    parser.add_argument("--output", type=str, default="models/controller_weighted",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--test_queries", type=int, default=10,
                       help="Number of queries for testing each checkpoint")
    parser.add_argument("--use_weights", action="store_true", default=True,
                       help="Use class weighting")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    training_path = Path(str(args.trajectories).replace('.json', '.training.json'))
    if training_path.exists():
        logger.info(f"Loading from {training_path}")
        with open(training_path) as f:
            data = json.load(f)
        states = [item["text"] for item in data]
        labels = [item["label"] for item in data]
    else:
        logger.error(f"Training file not found: {training_path}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(states)} training pairs")
    
    # Compute class weights
    class_weights = compute_class_weights(labels)
    logger.info(f"Class weights: {dict(zip(ACTION_NAMES, [f'{w:.2f}' for w in class_weights]))}")
    
    # Split data
    train_states, test_states, train_labels, test_labels = train_test_split(
        states, labels, test_size=0.2, random_state=42
    )
    train_states, val_states, train_labels, val_labels = train_test_split(
        train_states, train_labels, test_size=0.125, random_state=42
    )
    
    logger.info(f"Train: {len(train_states)}, Val: {len(val_states)}, Test: {len(test_states)}")
    
    # Load tokenizer
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = WeightedTrajectoryDataset(train_states, train_labels, tokenizer)
    val_dataset = WeightedTrajectoryDataset(val_states, val_labels, tokenizer)
    test_dataset = WeightedTrajectoryDataset(test_states, test_labels, tokenizer)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=8,
        problem_type="single_label_classification"
    )
    
    # Training arguments - save every epoch, no early stopping
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,  # Don't auto-load best
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        seed=42,
        report_to="none",
    )
    
    # Create trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights if args.use_weights else None,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Saved final model to {final_path}")
    
    # Load retriever and cost table for testing
    logger.info("Loading retriever for testing...")
    retriever = BM25Retriever()
    retriever.load_index(
        str(project_root / "data" / "indexes" / "faiss.index"),
        str(project_root / "data" / "processed" / "passages.json")
    )
    cost_table = load_cost_table()
    
    # Load test queries
    traj_path = Path(args.trajectories)
    with open(traj_path) as f:
        traj_data = json.load(f)
    
    test_queries = []
    for traj in traj_data.get("trajectories", []):
        test_queries.append({
            "question": traj["query"],
            "answer": traj["ground_truth"]
        })
    random.seed(123)  # Different seed for test queries
    test_queries = random.sample(test_queries, min(args.test_queries, len(test_queries)))
    
    # Test each checkpoint
    print(f"\n{'='*70}")
    print("CHECKPOINT EVALUATION")
    print(f"{'='*70}\n")
    
    epoch_results = []
    
    # Find all checkpoints
    checkpoints = sorted(output_dir.glob("checkpoint-*"), 
                        key=lambda x: int(x.name.split("-")[1]))
    checkpoints.append(final_path)
    
    for ckpt_path in checkpoints:
        if ckpt_path.name == "final":
            epoch_num = args.epochs
        else:
            step = int(ckpt_path.name.split("-")[1])
            steps_per_epoch = len(train_dataset) // 16
            epoch_num = step // steps_per_epoch if steps_per_epoch > 0 else step
        
        print(f"\n--- Checkpoint: {ckpt_path.name} (Epoch ~{epoch_num}) ---")
        
        # Load controller from checkpoint
        controller = Controller(str(ckpt_path))
        
        # Test
        test_result = test_controller_quick(controller, retriever, test_queries, cost_table)
        
        print(f"Accuracy: {test_result['accuracy']*100:.1f}%")
        print(f"Avg Steps: {test_result['avg_steps']:.1f}")
        print(f"Action Distribution: {test_result['action_distribution']}")
        print(f"Avg Energy: {test_result['avg_energy']:.1f} mWh")
        
        epoch_results.append({
            "checkpoint": ckpt_path.name,
            "epoch": epoch_num,
            "accuracy": test_result["accuracy"],
            "avg_steps": test_result["avg_steps"],
            "action_distribution": test_result["action_distribution"],
            "avg_energy": test_result["avg_energy"]
        })
    
    # Save epoch results
    results_path = output_dir / "epoch_results.json"
    with open(results_path, 'w') as f:
        json.dump(epoch_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Results saved to: {results_path}")
    
    # Print progression
    print("\nAccuracy Progression:")
    for r in epoch_results:
        bar = "█" * int(r["accuracy"] * 20)
        print(f"  Epoch {r['epoch']:2d}: {r['accuracy']*100:5.1f}% {bar}")


if __name__ == "__main__":
    main()
