"""
Base interface for all RAG systems.

All RAG implementations (naive, adaptive, deep, RL-based) must implement
this interface to ensure they can be evaluated uniformly.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, List, Any


class BaseRAG(ABC):
    """Abstract base class for RAG systems."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this RAG system."""
        pass
    
    @abstractmethod
    def answer(self, query: str, return_trace: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Answer a question using the RAG system.
        
        Args:
            query: The question to answer
            return_trace: If False, return only the answer string (fast, for benchmarking)
                         If True, return a dict with answer + retrieved_docs + metadata
        
        Returns:
            If return_trace=False:
                str: Just the answer
            If return_trace=True:
                dict: {
                    "answer": str,
                    "retrieved_docs": List[Dict],  # Retrieved passages
                    "metadata": Dict  # System-specific metadata
                }
        """
        pass
    
    def batch_answer(
        self, 
        queries: List[str], 
        return_trace: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Answer multiple questions.
        
        Default implementation calls answer() sequentially.
        Subclasses can override for batched efficiency.
        
        Args:
            queries: List of questions to answer
            return_trace: Whether to return full traces
            
        Returns:
            List of answers (strings or dicts depending on return_trace)
        """
        return [self.answer(q, return_trace=return_trace) for q in queries]
