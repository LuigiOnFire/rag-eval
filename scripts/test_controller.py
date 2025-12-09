#!/usr/bin/env python3
"""
Test the trained controller in a real RAG pipeline.

This script:
1. Loads a trained controller checkpoint
2. Runs it on test queries using controller-guided search
3. Measures accuracy, energy, and action distribution

Usage:
    python scripts/test_controller.py --model models/controller/final --num_queries 20
    
    # Test a specific checkpoint
    python scripts/test_controller.py --model models/controller/checkpoint-21 --num_queries 10
"""

import argparse
import json
import logging
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.behavior_cloning import Controller, Action, ACTION_NAMES
from src.green_search import GreenSearch, load_cost_table
from src.retriever import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ControllerSearchConfig:
    """Config for controller-guided search."""
    max_steps: int = 10
    temperature: float = 0.0  # 0 = greedy, >0 = sampling
    fallback_to_search: bool = True  # Fall back to GreenSearch on failure


class ControllerGuidedPipeline:
    """
    RAG pipeline guided by a trained controller.
    
    Instead of searching, the controller directly predicts the next action.
    """
    
    def __init__(
        self,
        controller: Controller,
        retriever: BM25Retriever,
        llm_model: str = "llama3:8b",
        slm_model: str = "mistral:latest",
        config: Optional[ControllerSearchConfig] = None
    ):
        self.controller = controller
        self.retriever = retriever
        self.llm_model = llm_model
        self.slm_model = slm_model
        self.config = config or ControllerSearchConfig()
        self.cost_table = load_cost_table()
        
        # Import ollama for generation
        import ollama
        self.ollama = ollama
    
    def _build_state(
        self, 
        query: str, 
        observations: List[str],
        current_answer: Optional[str] = None
    ) -> str:
        """Build state string for controller input."""
        parts = [f"[CLS] {query}"]
        for obs in observations[-3:]:  # Last 3 observations
            parts.append(f"[SEP] {obs[:200]}")  # Truncate long observations
        if current_answer:
            parts.append(f"[ANS] {current_answer}")
        return " ".join(parts)
    
    def _execute_action(
        self,
        action: Action,
        query: str,
        observations: List[str],
        retrieved_docs: List[str]
    ) -> Tuple[str, float]:
        """
        Execute an action and return (result, energy_cost).
        """
        cost = self.cost_table.get(action.name, 20.0)
        
        if action == Action.RETRIEVE_KEYWORD:
            # BM25 retrieval - returns (passages, scores)
            passages, scores = self.retriever.retrieve(query, top_k=3)
            result = " | ".join([r["text"][:200] for r in passages])
            retrieved_docs.extend([r["text"] for r in passages])
            return f"Retrieved: {result}", cost
            
        elif action == Action.RETRIEVE_DENSE:
            # For now, same as keyword (we don't have dense retriever)
            passages, scores = self.retriever.retrieve(query, top_k=3)
            result = " | ".join([r["text"][:200] for r in passages])
            retrieved_docs.extend([r["text"] for r in passages])
            return f"Retrieved: {result}", cost
            
        elif action in [Action.GENERATE_LLM, Action.GENERATE_SLM]:
            model = self.llm_model if action == Action.GENERATE_LLM else self.slm_model
            
            # Build prompt with context
            context = "\n".join(retrieved_docs[-3:]) if retrieved_docs else ""
            prompt = f"""Answer the question based on the context provided.

Context:
{context}

Question: {query}

Answer (be concise):"""
            
            response = self.ollama.generate(model=model, prompt=prompt)
            answer = response['response'].strip()
            return answer, cost
            
        elif action in [Action.DECOMPOSE_LLM, Action.DECOMPOSE_SLM]:
            model = self.llm_model if action == Action.DECOMPOSE_LLM else self.slm_model
            
            prompt = f"""Break this question into 2 simpler sub-questions:

Question: {query}

Sub-questions (one per line):"""
            
            response = self.ollama.generate(model=model, prompt=prompt)
            return f"Sub-questions: {response['response'].strip()}", cost
            
        elif action in [Action.REASON_LLM, Action.REASON_SLM]:
            model = self.llm_model if action == Action.REASON_LLM else self.slm_model
            
            context = "\n".join(observations[-3:]) if observations else ""
            prompt = f"""Given this information, reason step-by-step about the question.

Information:
{context}

Question: {query}

Reasoning:"""
            
            response = self.ollama.generate(model=model, prompt=prompt)
            return f"Reasoning: {response['response'].strip()}", cost
        
        return "Unknown action", cost
    
    def run(
        self,
        query: str,
        expected_answer: Optional[str] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Run the controller-guided pipeline on a query.
        
        Returns dict with:
        - answer: Final answer
        - correct: Whether it matches expected (if provided)
        - energy: Total energy used
        - steps: Number of steps taken
        - actions: List of actions taken
        """
        observations = []
        retrieved_docs = []
        actions_taken = []
        total_energy = 0.0
        answer = None
        
        for step in range(self.config.max_steps):
            # Build current state
            state = self._build_state(query, observations, answer)
            
            # Get controller prediction
            action, probs = self.controller.predict(state)
            actions_taken.append(action.name)
            
            if verbose:
                logger.info(f"Step {step+1}: {action.name} (conf: {probs[action.value]:.2f})")
            
            # Execute action
            result, cost = self._execute_action(
                action, query, observations, retrieved_docs
            )
            total_energy += cost
            observations.append(result)
            
            # Check if this is a terminal action (GENERATE)
            if action in [Action.GENERATE_LLM, Action.GENERATE_SLM]:
                answer = result
                break
        
        # Check correctness
        correct = None
        if expected_answer and answer:
            # Simple substring match
            correct = (
                expected_answer.lower() in answer.lower() or
                answer.lower() in expected_answer.lower()
            )
        
        return {
            "query": query,
            "answer": answer,
            "expected": expected_answer,
            "correct": correct,
            "energy_mwh": total_energy,
            "steps": len(actions_taken),
            "actions": actions_taken
        }


def load_test_queries(n: int = 20) -> List[Dict]:
    """Load test queries from HotPotQA."""
    # Primary: load from questions.json (has ground truth answers)
    questions_path = project_root / "data" / "processed" / "questions.json"
    
    if questions_path.exists():
        with open(questions_path) as f:
            data = json.load(f)
        
        # Data is already in the right format
        questions = [{"question": item["question"], "answer": item["answer"]} for item in data]
        
        if questions:
            random.seed(42)
            return random.sample(questions, min(n, len(questions)))
    
    # Fallback: load from trajectories file
    traj_path = project_root / "results" / "trajectories_100.json"
    if traj_path.exists():
        with open(traj_path) as f:
            data = json.load(f)
        
        questions = []
        for traj in data.get("trajectories", []):
            if traj.get("ground_truth"):  # Only if ground truth exists
                questions.append({
                    "question": traj["query"],
                    "answer": traj["ground_truth"]
                })
        
        if questions:
            random.seed(42)
            return random.sample(questions, min(n, len(questions)))
    
    # Ultimate fallback: hardcoded test questions
    return [
        {"question": "Who directed Sinister?", "answer": "Scott Derrickson"},
        {"question": "What is the capital of France?", "answer": "Paris"},
    ][:n]


def main():
    parser = argparse.ArgumentParser(description="Test trained controller")
    parser.add_argument("--model", type=str, default="models/controller/final",
                       help="Path to controller model")
    parser.add_argument("--num_queries", type=int, default=20,
                       help="Number of test queries")
    parser.add_argument("--max_steps", type=int, default=10,
                       help="Max steps per query")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    args = parser.parse_args()
    
    # Load controller
    model_path = project_root / args.model
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    logger.info(f"Loading controller from {model_path}")
    controller = Controller(str(model_path))
    
    # Load retriever
    logger.info("Loading retriever...")
    retriever = BM25Retriever()
    retriever.load_index(
        str(project_root / "data" / "indexes" / "faiss.index"),
        str(project_root / "data" / "processed" / "passages.json")
    )
    
    # Create pipeline
    config = ControllerSearchConfig(max_steps=args.max_steps)
    pipeline = ControllerGuidedPipeline(controller, retriever, config=config)
    
    # Load test queries
    logger.info(f"Loading {args.num_queries} test queries...")
    queries = load_test_queries(args.num_queries)
    
    # Run tests
    results = []
    correct_count = 0
    total_energy = 0.0
    action_counts = {}
    
    print(f"\n{'='*60}")
    print(f"CONTROLLER TEST: {args.model}")
    print(f"{'='*60}\n")
    
    for i, q in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {q['question'][:50]}...")
        
        result = pipeline.run(
            q["question"],
            expected_answer=q["answer"],
            verbose=args.verbose
        )
        results.append(result)
        
        if result["correct"]:
            correct_count += 1
            status = "✓"
        elif result["correct"] is False:
            status = "✗"
        else:
            status = "?"
        
        total_energy += result["energy_mwh"]
        
        for action in result["actions"]:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"  {status} Answer: {(result['answer'] or 'None')[:50]}...")
        print(f"    Expected: {q['answer']}")
        print(f"    Actions: {' -> '.join(result['actions'])}")
        print(f"    Energy: {result['energy_mwh']:.1f} mWh\n")
    
    # Summary
    accuracy = correct_count / len(queries) if queries else 0
    avg_energy = total_energy / len(queries) if queries else 0
    avg_steps = sum(r["steps"] for r in results) / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Accuracy: {correct_count}/{len(queries)} ({accuracy*100:.1f}%)")
    print(f"Avg Energy: {avg_energy:.1f} mWh/query")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count} ({count/total_actions*100:.1f}%)")
    
    # Save results
    if args.output:
        output_path = project_root / args.output
        with open(output_path, 'w') as f:
            json.dump({
                "model": args.model,
                "num_queries": len(queries),
                "accuracy": accuracy,
                "avg_energy_mwh": avg_energy,
                "avg_steps": avg_steps,
                "action_counts": action_counts,
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
