#!/usr/bin/env python3
"""
Trajectory Generation Pipeline for SFT Training Data (Decoder Agent)

Pipeline:
1. Run decoder agent on HotPotQA questions
2. Verify correctness
3. For correct: optionally simplify (remove redundant steps)
4. For incorrect: retry with supporting_facts hints (if available)
5. Save trajectories with metadata
"""

import sys
sys.path.insert(0, '.')

import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datasets import load_dataset

from src.decoder_agent import DecoderAgent, ActionType
from src.retriever import BM25Retriever
from src.generator import OllamaGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Result of a single trajectory generation."""
    question: str
    ground_truth: str
    question_type: str  # 'comparison' or 'bridge'
    supporting_facts: List[List[str]]  # [[title, sent_idx], ...]
    
    # Trajectory data
    steps: List[Dict]
    answer: Optional[str]
    is_correct: bool
    total_energy_wh: float
    
    # Refinement metadata
    attempt: int  # 1 = first try, 2 = retry with hints
    was_simplified: bool
    original_steps: Optional[int]  # Before simplification


class LLMWrapper:
    """Wrap OllamaGenerator for decoder agent interface."""
    def __init__(self, gen: OllamaGenerator):
        self.gen = gen
        
    def generate(self, prompt: str) -> str:
        import requests
        response = requests.post(
            f"{self.gen.base_url}/api/chat",
            json={
                "model": self.gen.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": self.gen.temperature,
                    "top_p": self.gen.top_p,
                    "num_predict": 500
                }
            },
            timeout=self.gen.timeout
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


class RetrieverWrapper:
    """Wrap retriever for decoder agent interface. Works with BM25, Dense, or Hybrid."""
    def __init__(self, ret):
        self.ret = ret
        
    def search(self, query: str, top_k: int = 5):
        passages, scores = self.ret.retrieve(query, top_k=top_k)
        return [{'title': p.get('title', ''), 'text': p.get('text', '')} for p in passages]


class TrajectoryGenerator:
    """Generate and refine trajectories for SFT."""
    
    def __init__(
        self,
        agent: DecoderAgent,
        enable_simplification: bool = True,
        enable_retry: bool = True,
        max_retry_attempts: int = 1,
    ):
        self.agent = agent
        self.enable_simplification = enable_simplification
        self.enable_retry = enable_retry
        self.max_retry_attempts = max_retry_attempts
        
    def generate_trajectory(
        self,
        question: str,
        ground_truth: str,
        question_type: str = "unknown",
        supporting_facts: Optional[List] = None,
    ) -> TrajectoryResult:
        """Generate a trajectory, with optional retry and simplification."""
        
        # Attempt 1: Normal generation
        result = self.agent.run(question, ground_truth=ground_truth)
        
        trajectory = TrajectoryResult(
            question=question,
            ground_truth=ground_truth,
            question_type=question_type,
            supporting_facts=supporting_facts or [],
            steps=result['steps'],
            answer=result['answer'],
            is_correct=result['is_correct'],
            total_energy_wh=result['total_energy_wh'],
            attempt=1,
            was_simplified=False,
            original_steps=None,
        )
        
        # If correct, try to simplify
        if trajectory.is_correct and self.enable_simplification:
            simplified = self._try_simplify(trajectory)
            if simplified:
                trajectory = simplified
                
        # If incorrect, retry with hints
        elif not trajectory.is_correct and self.enable_retry and supporting_facts:
            retry_result = self._retry_with_hints(
                question, ground_truth, question_type, supporting_facts
            )
            if retry_result and retry_result.is_correct:
                trajectory = retry_result
                
        return trajectory
    
    def _try_simplify(self, trajectory: TrajectoryResult) -> Optional[TrajectoryResult]:
        """
        Try to simplify a correct trajectory.
        
        Simplification rules:
        1. Remove consecutive duplicate retrieval actions
        2. Check if LLM actions could be SLM (for simple questions)
        """
        steps = trajectory.steps
        original_count = len(steps)
        
        # Rule 1: Remove duplicate consecutive retrievals
        simplified_steps = []
        seen_retrievals = set()
        
        for step in steps:
            action = step['action']
            params = step.get('parameters', '')
            
            if action in ('RETRIEVE_KEYWORD', 'RETRIEVE_DENSE'):
                key = f"{action}:{params}"
                if key in seen_retrievals:
                    continue  # Skip duplicate
                seen_retrievals.add(key)
            
            simplified_steps.append(step)
        
        # Only return if we actually simplified
        if len(simplified_steps) < original_count:
            return TrajectoryResult(
                question=trajectory.question,
                ground_truth=trajectory.ground_truth,
                question_type=trajectory.question_type,
                supporting_facts=trajectory.supporting_facts,
                steps=simplified_steps,
                answer=trajectory.answer,
                is_correct=trajectory.is_correct,
                total_energy_wh=trajectory.total_energy_wh,  # Keep original energy (actual cost)
                attempt=trajectory.attempt,
                was_simplified=True,
                original_steps=original_count,
            )
        
        return None
    
    def _retry_with_hints(
        self,
        question: str,
        ground_truth: str,
        question_type: str,
        supporting_facts: List,
    ) -> Optional[TrajectoryResult]:
        """
        Retry with hints from supporting_facts.
        
        We inject the entity names from supporting_facts into the prompt.
        """
        # Extract titles from supporting facts
        hint_titles = list(set(sf[0] for sf in supporting_facts))
        hint_str = ", ".join(hint_titles[:3])  # Max 3 hints
        
        # Modify question with hint
        hinted_question = f"{question}\n\n[Hint: Look for information about: {hint_str}]"
        
        result = self.agent.run(hinted_question, ground_truth=ground_truth)
        
        if result['is_correct']:
            return TrajectoryResult(
                question=question,  # Original question (not hinted)
                ground_truth=ground_truth,
                question_type=question_type,
                supporting_facts=supporting_facts,
                steps=result['steps'],
                answer=result['answer'],
                is_correct=True,
                total_energy_wh=result['total_energy_wh'],
                attempt=2,
                was_simplified=False,
                original_steps=None,
            )
        
        return None


def load_hotpotqa_sample(split: str = "validation", n_samples: int = 100) -> List[Dict]:
    """Load HotPotQA samples."""
    logger.info(f"Loading HotPotQA {split} split...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        samples.append({
            'question': item['question'],
            'answer': item['answer'],
            'type': item['type'],  # 'comparison' or 'bridge'
            'supporting_facts': list(zip(
                item['supporting_facts']['title'],
                item['supporting_facts']['sent_id']
            )),
        })
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--no-simplify", action="store_true", help="Disable simplification")
    parser.add_argument("--no-retry", action="store_true", help="Disable retry with hints")
    parser.add_argument("--passages", type=str, default="data/processed/passages.json")
    parser.add_argument("--retriever", type=str, default="bm25", 
                        choices=["bm25", "dense", "hybrid"],
                        help="Retriever type: bm25, dense, or hybrid")
    parser.add_argument("--faiss-index", type=str, default="data/indexes/faiss.index",
                        help="Path to FAISS index file (for dense/hybrid)")
    args = parser.parse_args()
    
    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/trajectories/trajectories_{timestamp}.json"
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load passages and build retriever
    logger.info("Loading passages...")
    with open(args.passages) as f:
        passages = json.load(f)
    logger.info(f"Loaded {len(passages)} passages")
    
    # Build retriever based on type
    if args.retriever == "bm25":
        logger.info("Building BM25 index...")
        retriever = BM25Retriever()
        retriever.build_index(passages)
    elif args.retriever == "dense":
        logger.info("Loading dense retriever with FAISS...")
        from src.retriever import FaissRetriever
        retriever = FaissRetriever()
        retriever.load_index(args.faiss_index, args.passages)
    elif args.retriever == "hybrid":
        logger.info("Building hybrid retriever (BM25 + Dense)...")
        from src.retriever import FaissRetriever, HybridRetriever
        
        # Build BM25
        bm25_ret = BM25Retriever()
        bm25_ret.build_index(passages)
        
        # Load dense
        dense_ret = FaissRetriever()
        dense_ret.load_index(args.faiss_index, args.passages)
        
        # Combine
        retriever = HybridRetriever(bm25_ret, dense_ret, bm25_weight=0.4)
        logger.info("Hybrid retriever ready")
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever}")
    
    # Load LLMs
    logger.info("Loading LLMs...")
    llm = OllamaGenerator(model_name="llama3:8b")
    slm = OllamaGenerator(model_name="mistral:latest")
    
    # Create agent
    agent = DecoderAgent(
        llm=LLMWrapper(llm),
        slm=LLMWrapper(slm),
        retriever=RetrieverWrapper(retriever),
        max_steps=8,
    )
    
    # Create trajectory generator
    generator = TrajectoryGenerator(
        agent=agent,
        enable_simplification=not args.no_simplify,
        enable_retry=not args.no_retry,
    )
    
    # Load HotPotQA samples
    samples = load_hotpotqa_sample(n_samples=args.n_samples)
    
    # Generate trajectories
    trajectories = []
    stats = {
        'total': 0,
        'correct_first_try': 0,
        'correct_after_retry': 0,
        'incorrect': 0,
        'simplified': 0,
        'total_steps': 0,
        'total_energy_wh': 0.0,
    }
    
    for i, sample in enumerate(samples):
        logger.info(f"\n[{i+1}/{len(samples)}] Processing: {sample['question'][:60]}...")
        
        try:
            result = generator.generate_trajectory(
                question=sample['question'],
                ground_truth=sample['answer'],
                question_type=sample['type'],
                supporting_facts=sample['supporting_facts'],
            )
            
            trajectories.append(asdict(result))
            
            # Update stats
            stats['total'] += 1
            stats['total_steps'] += len(result.steps)
            stats['total_energy_wh'] += result.total_energy_wh
            
            if result.is_correct:
                if result.attempt == 1:
                    stats['correct_first_try'] += 1
                else:
                    stats['correct_after_retry'] += 1
                if result.was_simplified:
                    stats['simplified'] += 1
            else:
                stats['incorrect'] += 1
            
            # Log progress
            correct = stats['correct_first_try'] + stats['correct_after_retry']
            logger.info(
                f"  Result: {'✓' if result.is_correct else '✗'} "
                f"(attempt {result.attempt}, {len(result.steps)} steps) "
                f"Running: {correct}/{stats['total']} correct ({100*correct/stats['total']:.1f}%)"
            )
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate final stats
    correct_total = stats['correct_first_try'] + stats['correct_after_retry']
    stats['accuracy'] = correct_total / stats['total'] if stats['total'] > 0 else 0
    stats['avg_steps'] = stats['total_steps'] / stats['total'] if stats['total'] > 0 else 0
    stats['avg_energy_wh'] = stats['total_energy_wh'] / stats['total'] if stats['total'] > 0 else 0
    
    # Save results
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_samples': args.n_samples,
            'simplification_enabled': not args.no_simplify,
            'retry_enabled': not args.no_retry,
        },
        'stats': stats,
        'trajectories': trajectories,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAJECTORY GENERATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"Correct (first try): {stats['correct_first_try']} ({100*stats['correct_first_try']/stats['total']:.1f}%)")
    logger.info(f"Correct (after retry): {stats['correct_after_retry']} ({100*stats['correct_after_retry']/stats['total']:.1f}%)")
    logger.info(f"Total correct: {correct_total} ({100*stats['accuracy']:.1f}%)")
    logger.info(f"Simplified: {stats['simplified']}")
    logger.info(f"Avg steps: {stats['avg_steps']:.2f}")
    logger.info(f"Avg energy: {stats['avg_energy_wh']:.4f} Wh")


if __name__ == "__main__":
    main()
