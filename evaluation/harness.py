"""
Minimal evaluation harness for RAG systems.

This module provides a simple interface to evaluate any RAG system
that implements the BaseRAG interface.
"""

import json
import logging
import re
import string
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from base import BaseRAG

logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    """
    Normalize text for exact match comparison.
    
    Follows SQuAD evaluation:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Normalize whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    """
    Compute exact match score.
    
    Returns 1 if normalized prediction equals normalized ground truth, 0 otherwise.
    Also returns 1 if ground truth is contained in prediction (for QA flexibility).
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    # Exact match or containment
    if pred_norm == gt_norm or gt_norm in pred_norm:
        return 1
    return 0


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    Follows SQuAD evaluation.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def evaluate(
    rag_system: BaseRAG,
    queries: List[str],
    ground_truth: List[str],
    return_traces: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a RAG system on a set of queries.
    
    This is the minimal evaluation interface. For full RAGChecker
    evaluation, use RAGEvaluator from evaluator.py.
    
    Args:
        rag_system: Any RAG system implementing BaseRAG
        queries: List of questions
        ground_truth: List of expected answers
        return_traces: Whether to include full traces in results
        
    Returns:
        Dict with:
            - answers: Generated answers
            - traces: Full traces (if return_traces=True)
            - metadata: System info and timing
    """
    logger.info(f"Evaluating {rag_system.name} on {len(queries)} queries...")
    
    start_time = datetime.now()
    
    # Get answers
    if return_traces:
        results = rag_system.batch_answer(queries, return_trace=True)
        answers = [r["answer"] for r in results]
        traces = results
    else:
        answers = rag_system.batch_answer(queries, return_trace=False)
        traces = None
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Basic metrics (exact match)
    exact_matches = sum(
        1 for ans, gt in zip(answers, ground_truth)
        if gt.lower() in ans.lower()
    )
    
    result = {
        "system_name": rag_system.name,
        "num_queries": len(queries),
        "answers": answers,
        "ground_truth": ground_truth,
        "basic_metrics": {
            "exact_match_count": exact_matches,
            "exact_match_rate": exact_matches / len(queries) if queries else 0
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "queries_per_second": len(queries) / duration if duration > 0 else 0
        }
    }
    
    if traces:
        result["traces"] = traces
    
    return result


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved results to {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY: {results['system_name']}")
    print("="*60)
    print(f"Queries: {results['num_queries']}")
    print(f"Duration: {results['metadata']['duration_seconds']:.2f}s")
    print(f"Speed: {results['metadata']['queries_per_second']:.2f} queries/sec")
    print()
    print("Basic Metrics:")
    print(f"  Exact Match Rate: {results['basic_metrics']['exact_match_rate']:.1%}")
    print("="*60)
