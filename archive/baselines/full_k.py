"""
Full-k RAG baseline (k=50).

Upper bound baseline that retrieves many more passages than necessary.
Useful to measure the cost of over-retrieval vs quality gain.
"""

from typing import List, Dict, Optional, Union, Any
import yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import RAGPipeline, load_pipeline_from_config
from base import BaseRAG


class FullKRAG(BaseRAG):
    """
    Full-k RAG baseline: retrieve k=50 passages.
    
    This represents an upper bound on retrieval - more context than typically needed.
    Useful for measuring:
    - Quality ceiling (does more context help?)
    - Energy cost of over-retrieval
    - Generation quality with noisy context
    """
    
    DEFAULT_K = 50
    
    def __init__(
        self,
        config_path: str = "config_local.yaml",
        top_k: int = DEFAULT_K
    ):
        """
        Initialize Full-k RAG.
        
        Args:
            config_path: Path to configuration file
            top_k: Number of passages to retrieve (default 50)
        """
        self.config_path = config_path
        self.top_k = top_k
        self._pipeline: Optional[RAGPipeline] = None
    
    @property
    def pipeline(self) -> RAGPipeline:
        """Lazy-load pipeline on first access."""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self) -> RAGPipeline:
        """Load and configure the pipeline with our settings."""
        # Load base pipeline from config
        pipeline = load_pipeline_from_config(self.config_path)
        
        # Override top_k
        pipeline.top_k = self.top_k
        pipeline._name = self.name
        
        return pipeline
    
    @property
    def name(self) -> str:
        """Return the name of this RAG system."""
        return f"full_k{self.top_k}"
    
    def answer(self, query: str, return_trace: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Answer a question using full-k RAG.
        
        Args:
            query: The question to answer
            return_trace: If True, return dict with answer + retrieval info
        
        Returns:
            str or dict depending on return_trace
        """
        return self.pipeline.answer(query, return_trace=return_trace)
    
    def batch_answer(
        self, 
        queries: List[str], 
        return_trace: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Answer multiple questions.
        
        Args:
            queries: List of questions
            return_trace: If True, include retrieval info in results
        
        Returns:
            List of answers (str or dict)
        """
        return self.pipeline.batch_answer(queries, return_trace=return_trace)


def create_full_k_rag(config_path: str = "config_local.yaml", top_k: int = 50) -> FullKRAG:
    """
    Factory function to create FullKRAG instance.
    
    Args:
        config_path: Path to configuration file
        top_k: Number of passages to retrieve
        
    Returns:
        Initialized FullKRAG
    """
    return FullKRAG(config_path=config_path, top_k=top_k)


if __name__ == "__main__":
    # Quick test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    rag = FullKRAG()
    print(f"System name: {rag.name}")
    
    # Test single query
    result = rag.answer("What is the capital of France?", return_trace=True)
    print(f"\nAnswer: {result['answer']}")
    print(f"Retrieved docs: {len(result['retrieved_docs'])}")
