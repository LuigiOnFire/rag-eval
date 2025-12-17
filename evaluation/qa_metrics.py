#!/usr/bin/env python3
"""
Standard QA evaluation metrics matching Adaptive-RAG paper.

Implements:
- EM (Exact Match): Normalized string equality
- F1: Token-level precision/recall
- Acc (Answer Accuracy): Soft match - gold answer contained in prediction

These match the evaluation protocol from:
"Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models 
 through Question Complexity" (Jeong et al., NAACL 2024)
"""

import re
import string
from typing import Dict, List, Optional, Tuple


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation.
    
    Lower text, remove punctuation, articles and extra whitespace.
    Standard SQuAD normalization.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match (EM): 1 if normalized strings are identical, 0 otherwise.
    
    This is the strictest metric - requires exact string match after normalization.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score.
    
    Measures overlap between prediction and ground truth tokens.
    More lenient than EM - partial credit for partial matches.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty cases
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    # Count common tokens
    common_tokens = set(pred_tokens) & set(truth_tokens)
    num_same = len(common_tokens)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def answer_accuracy(prediction: str, ground_truth: str) -> float:
    """
    Answer Accuracy (Acc): Soft match checking containment.
    
    Returns 1 if the normalized gold answer is contained in the normalized
    prediction. This is more lenient than EM but stricter than F1.
    
    This matches the "Acc" metric commonly used in Adaptive-RAG and similar papers.
    """
    norm_pred = normalize_answer(prediction)
    norm_truth = normalize_answer(ground_truth)
    
    # Check if gold answer is contained in prediction
    return float(norm_truth in norm_pred)


def compute_all_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute all three metrics at once.
    
    Returns:
        Dict with 'em', 'f1', and 'acc' scores
    """
    if not prediction:
        return {'em': 0.0, 'f1': 0.0, 'acc': 0.0}
    
    return {
        'em': exact_match(prediction, ground_truth),
        'f1': f1_score(prediction, ground_truth),
        'acc': answer_accuracy(prediction, ground_truth)
    }


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        results: List of per-sample metric dicts from compute_all_metrics
        
    Returns:
        Dict with averaged 'em', 'f1', 'acc' scores (as percentages)
    """
    if not results:
        return {'em': 0.0, 'f1': 0.0, 'acc': 0.0}
    
    n = len(results)
    return {
        'em': sum(r['em'] for r in results) / n * 100,
        'f1': sum(r['f1'] for r in results) / n * 100,
        'acc': sum(r['acc'] for r in results) / n * 100
    }


# For convenience, expose the main function
def evaluate_answer(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Alias for compute_all_metrics."""
    return compute_all_metrics(prediction, ground_truth)


if __name__ == "__main__":
    # Quick test
    test_cases = [
        # (prediction, ground_truth)
        ("Kevin Spacey", "Kevin Spacey"),  # Exact match
        ("Kevin Spacey is the actor in American Beauty.", "Kevin Spacey"),  # Acc match
        ("The actor is Spacey.", "Kevin Spacey"),  # Partial F1
        ("I don't know", "Kevin Spacey"),  # No match
        ("yes", "yes"),  # Simple exact
        ("Yes, they are both American.", "yes"),  # Acc match
    ]
    
    print("QA Metrics Test Cases:")
    print("=" * 70)
    for pred, truth in test_cases:
        metrics = compute_all_metrics(pred, truth)
        print(f"Pred: '{pred[:40]}...' | Truth: '{truth}'")
        print(f"  EM: {metrics['em']:.0f}  F1: {metrics['f1']:.2f}  Acc: {metrics['acc']:.0f}")
        print()
