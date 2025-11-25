"""
Baseline RAG implementations.

All baselines implement the BaseRAG interface for uniform evaluation.
"""

from .naive import NaiveRAG
from .full_k import FullKRAG
from .no_retrieval import NoRetrievalRAG
from .adaptive import AdaptiveRAG

__all__ = ["NaiveRAG", "FullKRAG", "NoRetrievalRAG", "AdaptiveRAG"]
