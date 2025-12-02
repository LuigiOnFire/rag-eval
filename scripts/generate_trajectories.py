#!/usr/bin/env python3
"""
Generate training trajectories using GreenTreeSearch.

This script runs Phase 1 of the Green-DeepRAG training pipeline:
1. Loads the cost table (from benchmark_costs.py)
2. Loads HotPotQA questions
3. Runs GreenTreeSearch to find cheapest correct trajectories
4. Saves trajectories for behavior cloning (Phase 2)

Usage:
    python scripts/generate_trajectories.py --config config_local.yaml --num_samples 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from datasets import load_dataset

from src.generator import OllamaGenerator
from src.retriever import BM25Retriever
from src.green_tree_search import (
    GreenTreeSearch, 
    load_cost_table, 
    create_simple_judge,
    Action
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_hotpotqa_samples(num_samples: int, split: str = "validation") -> list:
    """
    Load HotPotQA samples.
    
    Args:
        num_samples: Number of samples to load
        split: Dataset split to use
        
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    logger.info(f"Loading {num_samples} samples from HotPotQA {split} split...")
    
    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append({
            "question": item["question"],
            "answer": item["answer"],
            "id": item.get("id", str(i))
        })
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate training trajectories with GreenTreeSearch")
    parser.add_argument("--config", type=str, default="config_local.yaml",
                       help="Path to configuration file")
    parser.add_argument("--cost_table", type=str, default="results/cost_table.json",
                       help="Path to cost table JSON")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of HotPotQA samples to process")
    parser.add_argument("--output", type=str, default="results/trajectories.json",
                       help="Output path for trajectories")
    parser.add_argument("--max_steps", type=int, default=5,
                       help="Maximum steps per trajectory")
    parser.add_argument("--split", type=str, default="validation",
                       help="HotPotQA split to use")
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Load cost table
    logger.info(f"Loading cost table from {args.cost_table}")
    cost_table = load_cost_table(args.cost_table)
    logger.info(f"Cost table (Wh): {json.dumps({Action(k).name: f'{v:.6f}' for k, v in cost_table.items()}, indent=2)}")
    
    # Initialize SLM
    slm_config = config.get('slm', config.get('generator', {}))
    logger.info(f"Initializing SLM: {slm_config.get('model_name', 'unknown')}")
    slm = OllamaGenerator(
        model_name=slm_config.get('model_name', 'mistral'),
        temperature=slm_config.get('temperature', 0.0),
        max_tokens=slm_config.get('max_tokens', 256),
        timeout=slm_config.get('timeout', 60)
    )
    
    # Initialize LLM
    llm_config = config.get('llm', config.get('generator', {}))
    logger.info(f"Initializing LLM: {llm_config.get('model_name', 'unknown')}")
    llm = OllamaGenerator(
        model_name=llm_config.get('model_name', 'llama3:8b'),
        temperature=llm_config.get('temperature', 0.0),
        max_tokens=llm_config.get('max_tokens', 256),
        timeout=llm_config.get('timeout', 120)
    )
    
    # Initialize retriever
    retriever_config = config.get('retriever', {})
    logger.info("Initializing BM25 retriever...")
    retriever = BM25Retriever()
    
    # Load index
    index_path = retriever_config.get('bm25_index_path', './data/indexes/faiss.bm25.pkl')
    passages_path = retriever_config.get('passages_path', './data/processed/passages.json')
    
    if Path(passages_path).exists():
        retriever.load_index(index_path, passages_path)
    else:
        logger.warning(f"Passages file not found at {passages_path}. Retrieval may fail.")
    
    # Create judge (substring match for HotPotQA)
    judge = create_simple_judge(exact_match=False)
    
    # Initialize GreenTreeSearch
    logger.info("Initializing GreenTreeSearch...")
    searcher = GreenTreeSearch(
        slm=slm,
        llm=llm,
        retriever=retriever,
        judge=judge,
        cost_table=cost_table,
        max_steps=args.max_steps
    )
    
    # Log cost order
    cost_order = searcher.get_cost_ordered_actions()
    logger.info(f"Action cost order: {[a.name for a in cost_order]}")
    
    # Load HotPotQA samples
    samples = load_hotpotqa_samples(args.num_samples, args.split)
    
    # Extract queries and ground truths
    queries = [s["question"] for s in samples]
    ground_truths = [s["answer"] for s in samples]
    
    # Run batch search
    logger.info(f"Starting trajectory generation for {len(queries)} queries...")
    trajectories = searcher.batch_search(
        queries=queries,
        ground_truths=ground_truths,
        save_path=args.output
    )
    
    # Extract training data
    training_data = searcher.extract_training_data(trajectories, only_correct=True)
    
    # Save training data separately
    training_path = Path(args.output).with_suffix('.training.json')
    with open(training_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    logger.info(f"Saved {len(training_data)} training pairs to {training_path}")
    
    # Print summary
    correct = sum(1 for t in trajectories if t.correct)
    total_energy = sum(t.total_energy_wh for t in trajectories)
    avg_steps = sum(len(t.steps) for t in trajectories) / len(trajectories) if trajectories else 0
    
    print("\n" + "="*60)
    print("TRAJECTORY GENERATION SUMMARY")
    print("="*60)
    print(f"Total queries:     {len(trajectories)}")
    print(f"Correct answers:   {correct} ({correct/len(trajectories)*100:.1f}%)")
    print(f"Total energy:      {total_energy:.6f} Wh")
    print(f"Avg energy/query:  {total_energy/len(trajectories):.6f} Wh")
    print(f"Avg steps/query:   {avg_steps:.2f}")
    print(f"Training pairs:    {len(training_data)}")
    print(f"Output saved to:   {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
