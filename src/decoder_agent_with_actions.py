#!/usr/bin/env python3
"""
Decoder agent with action token execution.

This module provides an inference wrapper that:
1. Generates text from the decoder model
2. Parses special action tokens as they are generated
3. Executes actions (retrieve, extract) when tokens are detected
4. Continues generation with context updated
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import re
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from action_tokens import ACTION_TOKENS, parse_action_tokens


class ActionExecutor:
    """Executes actions triggered by special tokens."""
    
    def __init__(self, retriever, generator):
        """
        Args:
            retriever: Retriever instance (e.g., BM25Retriever)
            generator: Generator instance for extraction/synthesis
        """
        self.retriever = retriever
        self.generator = generator
    
    def execute_retrieve(self, query: str) -> str:
        """Execute retrieval action and return context."""
        docs = self.retriever.retrieve(query, top_k=3)
        context = '\n\n'.join([
            f"Document {i+1}: {doc.content}"
            for i, doc in enumerate(docs)
        ])
        return f"\n\nRetrieved documents:\n{context}"
    
    def execute_extract(self, text: str) -> str:
        """Execute extraction (no-op, just return the text)."""
        # Extraction is already done by the model
        return ""
    
    def execute_generate(self, text: str) -> str:
        """Execute generation (no-op, this is the final answer)."""
        return ""
    
    def execute_decompose(self, text: str) -> str:
        """Execute decomposition (no-op, handled by model)."""
        return ""
    
    def execute_reason(self, text: str) -> str:
        """Execute reasoning (no-op, handled by model)."""
        return ""


class DecoderAgentWithActions:
    """
    Decoder agent that executes actions based on special tokens.
    
    During generation, the model may output action tokens like:
    <|retrieve|>query<|/retrieve|>
    
    This wrapper intercepts these tokens, executes the action (e.g., retrieval),
    and injects the result back into the context before continuing generation.
    """
    
    def __init__(
        self,
        model_path: str,
        retriever,
        generator,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_path: Path to fine-tuned decoder model
            retriever: Retriever instance
            generator: Generator instance
            device: Device to run model on
        """
        self.device = device
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Action executor
        self.executor = ActionExecutor(retriever, generator)
        
        # Build token IDs for fast detection
        self.action_token_ids = {}
        for key, token in ACTION_TOKENS.items():
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            self.action_token_ids[key] = token_id
    
    def generate_with_actions(
        self,
        question: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[str, List[dict]]:
        """
        Generate answer with action execution.
        
        Args:
            question: Input question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            (final_text, actions_executed)
            - final_text: Complete generated text with action results
            - actions_executed: List of dicts with action details
        """
        # Format input
        prompt = f"Question: {question}\n\n"
        
        # Encode
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Track actions
        actions_executed = []
        
        # Generation loop with action execution
        current_input = input_ids
        full_text = prompt
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Generate next chunk
            with torch.no_grad():
                outputs = self.model.generate(
                    current_input,
                    max_new_tokens=min(100, max_new_tokens - tokens_generated),
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode new tokens
            new_text = self.tokenizer.decode(
                outputs[0][current_input.shape[1]:],
                skip_special_tokens=False
            )
            full_text += new_text
            tokens_generated += outputs.shape[1] - current_input.shape[1]
            
            # Parse for action tokens
            actions = parse_action_tokens(full_text)
            
            # Execute new actions
            new_actions = actions[len(actions_executed):]
            
            if new_actions:
                # Execute the first new action
                action_type, content = new_actions[0]
                
                print(f"[ACTION] {action_type}: {content[:100]}...")
                
                # Execute action
                if action_type == 'retrieve':
                    result = self.executor.execute_retrieve(content)
                elif action_type == 'extract':
                    result = self.executor.execute_extract(content)
                elif action_type == 'generate':
                    result = self.executor.execute_generate(content)
                    # Generate is final, stop here
                    actions_executed.append({
                        'type': action_type,
                        'content': content,
                        'result': result
                    })
                    break
                elif action_type == 'decompose':
                    result = self.executor.execute_decompose(content)
                elif action_type == 'reason':
                    result = self.executor.execute_reason(content)
                else:
                    result = ""
                
                # Add to actions
                actions_executed.append({
                    'type': action_type,
                    'content': content,
                    'result': result
                })
                
                # Inject result into context
                if result:
                    full_text += result
                
                # Re-encode for next generation
                current_input = self.tokenizer.encode(
                    full_text,
                    return_tensors='pt'
                ).to(self.device)
            else:
                # No action, continue generation
                current_input = outputs
            
            # Check for stop conditions
            if self.tokenizer.eos_token_id in outputs[0]:
                break
        
        return full_text, actions_executed
    
    def run(self, question: str) -> dict:
        """
        Run full inference on a question.
        
        Args:
            question: Input question
            
        Returns:
            Dict with:
            - question: Input question
            - generated_text: Full generated output
            - actions: List of executed actions
            - final_answer: Extracted final answer
        """
        generated_text, actions = self.generate_with_actions(question)
        
        # Extract final answer from generate action
        final_answer = None
        for action in actions:
            if action['type'] == 'generate':
                final_answer = action['content']
                break
        
        return {
            'question': question,
            'generated_text': generated_text,
            'actions': actions,
            'final_answer': final_answer or "No answer generated",
        }


def main():
    """Example usage."""
    import argparse
    from retriever import BM25Retriever
    from generator import OllamaGenerator
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained decoder model")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to answer")
    parser.add_argument("--passages", type=str,
                        default="data/processed/passages.json",
                        help="Path to passages file")
    args = parser.parse_args()
    
    # Setup retriever and generator
    retriever = BM25Retriever(args.passages)
    generator = OllamaGenerator(model_name="mistral:latest")
    
    # Create agent
    agent = DecoderAgentWithActions(
        model_path=args.model,
        retriever=retriever,
        generator=generator
    )
    
    # Run
    print(f"\nQuestion: {args.question}\n")
    result = agent.run(args.question)
    
    print("\n=== Actions Executed ===")
    for i, action in enumerate(result['actions'], 1):
        print(f"{i}. {action['type']}: {action['content'][:100]}...")
    
    print(f"\n=== Final Answer ===")
    print(result['final_answer'])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
