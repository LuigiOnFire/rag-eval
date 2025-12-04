#!/usr/bin/env python3
"""
Generate training trajectories using GreenSearch (Cost-Priority Search).

This script runs Phase 1 of the Green-DeepRAG training pipeline:
1. Loads the cost table (from benchmark_costs.py output)
2. Loads HotPotQA questions
3. Runs GreenSearch to find cheapest correct trajectories
4. Saves trajectories for behavior cloning (Phase 2)

This is the V2 script that uses the new cost-priority search algorithm
instead of the old strategy-based approach.

Usage:
    python scripts/generate_trajectories_v2.py --config config_local.yaml --num_samples 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.generator import OllamaGenerator, create_generator
from src.retriever import BM25Retriever
from src.green_search import (
    GreenSearch,
    Judge,
    Trajectory,
    load_cost_table,
    ActionType,
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


def trajectory_to_dict(trajectory: Trajectory) -> Dict[str, Any]:
    """Convert Trajectory dataclass to serializable dict."""
    return {
        "query": trajectory.query,
        "ground_truth": trajectory.ground_truth,
        "answer": trajectory.answer,
        "is_correct": trajectory.is_correct,
        "total_cost": trajectory.total_cost,
        "total_energy_wh": trajectory.total_energy_wh,
        "nodes_explored": trajectory.nodes_explored,
        "search_depth": trajectory.search_depth,
        "steps": [
            {
                "state": step.state,
                "action": str(step.action),
                "action_id": step.action_id,
                "action_type": step.action.action_type.name,
                "observation": step.observation,
                "cost": step.cost,
                "cumulative_cost": step.cumulative_cost,
                "energy_wh": step.energy_wh,
                "cumulative_energy_wh": step.cumulative_energy_wh,
            }
            for step in trajectory.steps
        ]
    }


def extract_training_data(trajectories: List[Trajectory], only_correct: bool = True) -> List[Dict[str, Any]]:
    """
    Extract (state, action) training pairs from trajectories.
    
    Args:
        trajectories: List of completed trajectories
        only_correct: Only extract from correct trajectories
        
    Returns:
        List of training pairs for behavior cloning
    """
    training_pairs = []
    
    for traj in trajectories:
        if only_correct and not traj.is_correct:
            continue
        
        pairs = traj.to_training_pairs()
        training_pairs.extend(pairs)
    
    return training_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate training trajectories with GreenSearch")
    parser.add_argument("--config", type=str, default="config_local.yaml",
                       help="Path to configuration file")
    parser.add_argument("--cost_table", type=str, default="results/cost_table.json",
                       help="Path to cost table JSON")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of HotPotQA samples to process")
    parser.add_argument("--output", type=str, default="results/trajectories_v2.json",
                       help="Output path for trajectories")
    parser.add_argument("--max_depth", type=int, default=10,
                       help="Maximum search depth")
    parser.add_argument("--max_nodes", type=int, default=500,
                       help="Maximum nodes to explore per query")
    parser.add_argument("--split", type=str, default="validation",
                       help="HotPotQA split to use")
    parser.add_argument("--use_llm_judge", action="store_true",
                       help="Use LLM for semantic answer judging")
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Load cost table
    logger.info(f"Loading cost table from {args.cost_table}")
    cost_table = load_cost_table(args.cost_table)
    logger.info(f"Cost table (Wh): {json.dumps({f'Action_{k}': f'{v:.6f}' for k, v in cost_table.items()}, indent=2)}")
    
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
    
    # Create judge
    judge = Judge(llm=llm, use_llm_judge=args.use_llm_judge)
    
    # Initialize GreenSearch
    logger.info("Initializing GreenSearch (Cost-Priority Search)...")
    searcher = GreenSearch(
        retriever=retriever,
        slm=slm,
        llm=llm,
        cost_table=cost_table,
        judge=judge,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
    )
    
    # Log cost table
    logger.info(f"Cost table loaded with {len(cost_table)} actions")
    logger.info(f"Max depth: {args.max_depth}, Max nodes: {args.max_nodes}")
    
    # Load HotPotQA samples
    samples = load_hotpotqa_samples(args.num_samples, args.split)
    
    # Run search for each query
    logger.info(f"Starting trajectory generation for {len(samples)} queries...")
    
    trajectories: List[Trajectory] = []
    failed_queries: List[Dict[str, str]] = []
    
    for sample in tqdm(samples, desc="Generating trajectories"):
        query = sample["question"]
        ground_truth = sample["answer"]
        sample_id = sample["id"]
        
        try:
            trajectory = searcher.search(query, ground_truth)
            
            if trajectory:
                trajectories.append(trajectory)
                logger.debug(f"[{sample_id}] Found solution: cost={trajectory.total_cost:.4f}, depth={trajectory.search_depth}")
            else:
                failed_queries.append({
                    "id": sample_id,
                    "question": query,
                    "answer": ground_truth,
                    "reason": "No solution found within limits"
                })
                logger.debug(f"[{sample_id}] No solution found")
                
        except Exception as e:
            failed_queries.append({
                "id": sample_id,
                "question": query,
                "answer": ground_truth,
                "reason": str(e)
            })
            logger.warning(f"[{sample_id}] Error: {e}")
    
    # Convert trajectories to serializable format
    trajectories_data = [trajectory_to_dict(t) for t in trajectories]
    
    # Save trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "metadata": {
                "config": args.config,
                "cost_table": args.cost_table,
                "num_samples": args.num_samples,
                "max_depth": args.max_depth,
                "max_nodes": args.max_nodes,
                "use_llm_judge": args.use_llm_judge,
            },
            "trajectories": trajectories_data,
            "failed_queries": failed_queries,
        }, f, indent=2)
    logger.info(f"Saved {len(trajectories_data)} trajectories to {output_path}")
    
    # Extract training data
    training_data = extract_training_data(trajectories, only_correct=True)
    
    # Save training data separately
    training_path = output_path.with_suffix('.training.json')
    with open(training_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    logger.info(f"Saved {len(training_data)} training pairs to {training_path}")
    
    # Print summary
    correct = sum(1 for t in trajectories if t.is_correct)
    total_energy = sum(t.total_energy_wh for t in trajectories)
    total_nodes = sum(t.nodes_explored for t in trajectories)
    avg_depth = sum(t.search_depth for t in trajectories) / len(trajectories) if trajectories else 0
    
    print("\n" + "="*60)
    print("TRAJECTORY GENERATION SUMMARY (GreenSearch V2)")
    print("="*60)
    print(f"Total queries:     {args.num_samples}")
    print(f"Solutions found:   {len(trajectories)} ({len(trajectories)/args.num_samples*100:.1f}%)")
    print(f"Correct answers:   {correct} ({correct/len(trajectories)*100:.1f}% of solutions)" if trajectories else "Correct answers:   0")
    print(f"Failed queries:    {len(failed_queries)}")
    print(f"Total energy:      {total_energy:.6f} Wh")
    print(f"Avg energy/query:  {total_energy/len(trajectories):.6f} Wh" if trajectories else "Avg energy/query:  N/A")
    print(f"Total nodes:       {total_nodes}")
    print(f"Avg depth:         {avg_depth:.2f}")
    print(f"Training pairs:    {len(training_data)}")
    print(f"Output saved to:   {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
