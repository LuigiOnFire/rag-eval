"""Package initialization for RAG baseline pipeline."""

from .corpus import CorpusProcessor
from .retriever import FaissRetriever
from .generator import GeminiGenerator
from .pipeline import RAGPipeline
from .evaluator import RAGEvaluator

__version__ = "0.1.0"

__all__ = [
    "CorpusProcessor",
    "FaissRetriever",
    "GeminiGenerator",
    "RAGPipeline",
    "RAGEvaluator"
]
