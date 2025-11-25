"""
Run comparison across all baseline RAG systems.

Evaluates each baseline on the same queries and compares:
- Accuracy metrics (exact match, F1)
- Energy consumption (via CodeCarbon)
- Latency
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from baselines import NaiveRAG, FullKRAG, NoRetrievalRAG, AdaptiveRAG
from evaluation import evaluate, save_results, print_summary
from evaluation.energy import EnergyTracker, CODECARBON_AVAILABLE

logger = logging.getLogger(__name__)


def load_hotpotqa_questions(num_questions: int = 20, seed: int = 42) -> List[Dict[str, str]]:
    """
    Load questions from HotpotQA dataset.
    
    Args:
        num_questions: Number of questions to load
        seed: Random seed for reproducibility
        
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    logger.info(f"Loading {num_questions} questions from HotpotQA...")
    
    dataset = load_dataset(
        "hotpot_qa",
        "fullwiki",
        split="validation",
        trust_remote_code=True
    )
    
    # Shuffle and select
    dataset = dataset.shuffle(seed=seed)
    
    questions = []
    for i, item in enumerate(dataset):
        if i >= num_questions:
            break
        questions.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    logger.info(f"Loaded {len(questions)} questions")
    return questions


def run_comparison(
    config_path: str = "config_local.yaml",
    num_questions: int = 20,
    output_dir: str = "results/comparison",
    skip_baselines: Optional[List[str]] = None,
    track_energy: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run comparison across all baselines.
    
    Args:
        config_path: Path to configuration file
        num_questions: Number of questions to evaluate
        output_dir: Directory to save results
        skip_baselines: List of baseline names to skip
        track_energy: Whether to track energy consumption
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with comparison results
    """
    skip_baselines = skip_baselines or []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load questions
    qa_pairs = load_hotpotqa_questions(num_questions)
    queries = [qa["question"] for qa in qa_pairs]
    ground_truth = [qa["answer"] for qa in qa_pairs]
    
    # Initialize baselines
    baselines = []
    
    if "naive" not in skip_baselines:
        baselines.append(("naive_k5", NaiveRAG(config_path=config_path, top_k=5)))
    
    if "full_k" not in skip_baselines:
        baselines.append(("full_k50", FullKRAG(config_path=config_path, top_k=50)))
    
    if "no_retrieval" not in skip_baselines:
        baselines.append(("no_retrieval", NoRetrievalRAG(config_path=config_path)))
    
    if "adaptive_rule" not in skip_baselines:
        baselines.append(("adaptive_rule", AdaptiveRAG(config_path=config_path, classifier_type="rule_based")))
    
    if "adaptive_llm" not in skip_baselines:
        baselines.append(("adaptive_llm", AdaptiveRAG(config_path=config_path, classifier_type="llm_based")))
    
    logger.info(f"Running comparison with {len(baselines)} baselines on {len(queries)} questions")
    
    # Initialize energy tracker
    energy_tracker = None
    if track_energy and CODECARBON_AVAILABLE:
        energy_tracker = EnergyTracker(
            project_name="rag_baseline_comparison",
            output_dir=str(output_path / "energy")
        )
        logger.info("Energy tracking enabled")
    elif track_energy:
        logger.warning("Energy tracking requested but CodeCarbon not available")
    
    # Run each baseline
    all_results = {}
    
    for baseline_name, baseline in baselines:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running baseline: {baseline_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if energy_tracker:
                # Use energy-tracked evaluation
                energy_result = energy_tracker.measure_batch(
                    rag_system=baseline,
                    queries=queries,
                    experiment_name=baseline_name,
                    return_trace=False
                )
                predictions = energy_result["answers"]
                energy_metrics = energy_result["energy"].to_dict()
            else:
                # Regular evaluation
                predictions = baseline.batch_answer(queries, return_trace=False)
                energy_metrics = None
            
            duration = time.time() - start_time
            
            # Calculate accuracy metrics
            eval_results = evaluate(
                rag_system=baseline,  # Pass for metadata
                queries=queries,
                ground_truth=ground_truth,
                predictions=predictions  # Use pre-computed predictions
            )
            
            # Add energy metrics
            if energy_metrics:
                eval_results["energy_metrics"] = energy_metrics
            
            eval_results["total_duration_sec"] = duration
            eval_results["avg_query_time_sec"] = duration / len(queries)
            
            # Store results
            all_results[baseline_name] = eval_results
            
            # Print summary
            if verbose:
                print_summary(eval_results)
            else:
                logger.info(f"  Exact Match: {eval_results['basic_metrics']['exact_match_rate']:.2%}")
                logger.info(f"  Duration: {duration:.2f}s ({duration/len(queries):.2f}s/query)")
                if energy_metrics:
                    logger.info(f"  Energy: {energy_metrics['energy_kwh']*1000:.4f} Wh")
            
        except Exception as e:
            logger.error(f"Error running baseline {baseline_name}: {e}")
            all_results[baseline_name] = {"error": str(e)}
            import traceback
            traceback.print_exc()
    
    # Build comparison summary
    comparison = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(queries),
            "config_path": config_path,
            "energy_tracking": track_energy and CODECARBON_AVAILABLE
        },
        "baselines": all_results
    }
    
    # Save results
    output_file = output_path / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print comparison table
    print_comparison_table(all_results)
    
    return comparison


def print_comparison_table(results: Dict[str, Any]):
    """Print a comparison table of all baselines."""
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    
    # Header
    print(f"{'Baseline':<20} {'Exact Match':<15} {'Avg Time (s)':<15} {'Energy (Wh)':<15}")
    print("-"*80)
    
    for name, result in results.items():
        if "error" in result:
            print(f"{name:<20} {'ERROR':<15} {'-':<15} {'-':<15}")
            continue
        
        exact_match = result.get("basic_metrics", {}).get("exact_match_rate", 0)
        avg_time = result.get("avg_query_time_sec", 0)
        
        energy = "-"
        if "energy_metrics" in result:
            energy_kwh = result["energy_metrics"].get("energy_kwh", 0)
            energy = f"{energy_kwh * 1000:.4f}"
        
        print(f"{name:<20} {exact_match:<15.2%} {avg_time:<15.2f} {energy:<15}")
    
    print("="*80)


def evaluate_with_predictions(
    rag_system,
    queries: List[str],
    ground_truth: List[str],
    predictions: List[str]
) -> Dict[str, Any]:
    """
    Evaluate using pre-computed predictions.
    
    This is a modified version of evaluate() that accepts predictions directly
    instead of calling the RAG system again.
    """
    from evaluation.harness import compute_exact_match, compute_f1
    
    # Compute metrics
    exact_matches = []
    f1_scores = []
    per_query_results = []
    
    for query, pred, truth in zip(queries, predictions, ground_truth):
        em = compute_exact_match(pred, truth)
        f1 = compute_f1(pred, truth)
        
        exact_matches.append(em)
        f1_scores.append(f1)
        
        per_query_results.append({
            "query": query,
            "prediction": pred,
            "ground_truth": truth,
            "exact_match": em,
            "f1": f1
        })
    
    return {
        "system_name": rag_system.name,
        "num_queries": len(queries),
        "basic_metrics": {
            "exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
            "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        },
        "per_query_results": per_query_results
    }


# Patch evaluate to support pre-computed predictions
def evaluate(rag_system, queries, ground_truth, predictions=None):
    """
    Evaluate a RAG system on a set of queries.
    
    If predictions are provided, uses them directly.
    Otherwise, calls the RAG system to generate them.
    """
    from evaluation.harness import compute_exact_match, compute_f1
    
    if predictions is None:
        predictions = rag_system.batch_answer(queries, return_trace=False)
    
    # Compute metrics
    exact_matches = []
    f1_scores = []
    per_query_results = []
    
    for query, pred, truth in zip(queries, predictions, ground_truth):
        em = compute_exact_match(pred, truth)
        f1 = compute_f1(pred, truth)
        
        exact_matches.append(em)
        f1_scores.append(f1)
        
        per_query_results.append({
            "query": query,
            "prediction": pred,
            "ground_truth": truth,
            "exact_match": em,
            "f1": f1
        })
    
    return {
        "system_name": rag_system.name,
        "num_queries": len(queries),
        "basic_metrics": {
            "exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
            "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        },
        "per_query_results": per_query_results
    }


def main():
    parser = argparse.ArgumentParser(description="Run RAG baseline comparison")
    parser.add_argument("--config", type=str, default="config_local.yaml",
                        help="Path to config file")
    parser.add_argument("--num-questions", type=int, default=20,
                        help="Number of questions to evaluate")
    parser.add_argument("--output-dir", type=str, default="results/comparison",
                        help="Output directory for results")
    parser.add_argument("--skip", type=str, nargs="+", default=[],
                        help="Baselines to skip (naive, full_k, no_retrieval)")
    parser.add_argument("--no-energy", action="store_true",
                        help="Disable energy tracking")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comparison
    run_comparison(
        config_path=args.config,
        num_questions=args.num_questions,
        output_dir=args.output_dir,
        skip_baselines=args.skip,
        track_energy=not args.no_energy,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
