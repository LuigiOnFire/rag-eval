"""
Adaptive-RAG baseline.

Routes queries to different retrieval strategies based on predicted complexity:
- simple: No retrieval (LLM only)
- moderate: Standard retrieval (k=5)
- complex: Extended retrieval (k=20)

Inspired by: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language 
Models through Question Complexity" (Jeong et al., NAACL 2024)
"""

import logging
from typing import Dict, List, Optional, Union, Any, Literal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from base import BaseRAG
from .classifiers import RuleBasedClassifier, LLMClassifier
from .naive import NaiveRAG
from .no_retrieval import NoRetrievalRAG

logger = logging.getLogger(__name__)

ClassifierType = Literal["rule_based", "llm_based"]


class AdaptiveRAG(BaseRAG):
    """
    Adaptive RAG that routes queries based on complexity.
    
    Uses a classifier to predict query complexity, then routes to:
    - simple → NoRetrievalRAG (just LLM)
    - moderate → NaiveRAG with k=5
    - complex → NaiveRAG with k=20
    
    This allows the system to save computation on simple queries
    while providing more context for complex ones.
    """
    
    def __init__(
        self,
        config_path: str = "config_local.yaml",
        classifier_type: ClassifierType = "rule_based",
        simple_k: int = 0,      # No retrieval for simple
        moderate_k: int = 5,    # Standard retrieval
        complex_k: int = 20,    # Extended retrieval
    ):
        """
        Initialize Adaptive RAG.
        
        Args:
            config_path: Path to configuration file
            classifier_type: Which classifier to use ('rule_based' or 'llm_based')
            simple_k: Number of passages for simple queries (0 = no retrieval)
            moderate_k: Number of passages for moderate queries
            complex_k: Number of passages for complex queries
        """
        self.config_path = config_path
        self.classifier_type = classifier_type
        self.simple_k = simple_k
        self.moderate_k = moderate_k
        self.complex_k = complex_k
        
        # Initialize classifier
        if classifier_type == "rule_based":
            self.classifier = RuleBasedClassifier()
        elif classifier_type == "llm_based":
            self.classifier = LLMClassifier(config_path=config_path)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Lazy-load strategies
        self._strategies: Dict[str, BaseRAG] = {}
        
        logger.info(
            f"AdaptiveRAG initialized with {classifier_type} classifier "
            f"(simple_k={simple_k}, moderate_k={moderate_k}, complex_k={complex_k})"
        )
    
    def _get_strategy(self, complexity: str) -> BaseRAG:
        """Get or create the appropriate strategy for a complexity level."""
        if complexity not in self._strategies:
            if complexity == "simple":
                if self.simple_k == 0:
                    self._strategies[complexity] = NoRetrievalRAG(
                        config_path=self.config_path
                    )
                else:
                    self._strategies[complexity] = NaiveRAG(
                        config_path=self.config_path,
                        top_k=self.simple_k
                    )
            elif complexity == "moderate":
                self._strategies[complexity] = NaiveRAG(
                    config_path=self.config_path,
                    top_k=self.moderate_k
                )
            elif complexity == "complex":
                self._strategies[complexity] = NaiveRAG(
                    config_path=self.config_path,
                    top_k=self.complex_k
                )
            else:
                raise ValueError(f"Unknown complexity level: {complexity}")
        
        return self._strategies[complexity]
    
    @property
    def name(self) -> str:
        """Return the name of this RAG system."""
        return f"adaptive_{self.classifier_type}"
    
    def answer(self, query: str, return_trace: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Answer a question using adaptive retrieval.
        
        Args:
            query: The question to answer
            return_trace: If True, include routing info in result
        
        Returns:
            str or dict depending on return_trace
        """
        # Classify query complexity
        complexity = self.classifier.classify(query)
        logger.debug(f"Query classified as '{complexity}': {query[:50]}...")
        
        # Get appropriate strategy
        strategy = self._get_strategy(complexity)
        
        # Get answer from strategy
        if return_trace:
            result = strategy.answer(query, return_trace=True)
            # Add routing metadata
            result["metadata"]["complexity"] = complexity
            result["metadata"]["classifier"] = self.classifier.name
            result["metadata"]["strategy"] = strategy.name
            return result
        else:
            return strategy.answer(query, return_trace=False)
    
    def batch_answer(
        self,
        queries: List[str],
        return_trace: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Answer multiple questions with adaptive routing.
        
        Groups queries by complexity and processes each group with
        the appropriate strategy for efficiency.
        """
        # Classify all queries
        complexities = self.classifier.batch_classify(queries)
        
        # Group queries by complexity
        groups: Dict[str, List[tuple]] = {"simple": [], "moderate": [], "complex": []}
        for i, (query, complexity) in enumerate(zip(queries, complexities)):
            groups[complexity].append((i, query))
        
        # Log distribution
        logger.info(
            f"Query distribution: simple={len(groups['simple'])}, "
            f"moderate={len(groups['moderate'])}, complex={len(groups['complex'])}"
        )
        
        # Process each group
        results = [None] * len(queries)
        
        for complexity, items in groups.items():
            if not items:
                continue
            
            indices, group_queries = zip(*items)
            strategy = self._get_strategy(complexity)
            
            logger.info(f"Processing {len(group_queries)} {complexity} queries with {strategy.name}")
            group_results = strategy.batch_answer(list(group_queries), return_trace=return_trace)
            
            # Add routing metadata if tracing
            if return_trace:
                for result in group_results:
                    if isinstance(result, dict):
                        result["metadata"]["complexity"] = complexity
                        result["metadata"]["classifier"] = self.classifier.name
                        result["metadata"]["strategy"] = strategy.name
            
            # Place results back in original order
            for idx, result in zip(indices, group_results):
                results[idx] = result
        
        return results


def create_adaptive_rag(
    config_path: str = "config_local.yaml",
    classifier_type: ClassifierType = "rule_based"
) -> AdaptiveRAG:
    """
    Factory function to create AdaptiveRAG instance.
    
    Args:
        config_path: Path to configuration file
        classifier_type: 'rule_based' or 'llm_based'
        
    Returns:
        Initialized AdaptiveRAG
    """
    return AdaptiveRAG(config_path=config_path, classifier_type=classifier_type)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Test with rule-based classifier (faster)
    rag = AdaptiveRAG(classifier_type="rule_based")
    print(f"System name: {rag.name}")
    
    test_queries = [
        "What is the capital of France?",  # simple
        "How does machine learning work?",  # moderate
        "What is the difference between supervised and unsupervised learning?",  # complex
    ]
    
    print("\nTesting adaptive routing:")
    for query in test_queries:
        result = rag.answer(query, return_trace=True)
        print(f"\nQuery: {query}")
        print(f"  Complexity: {result['metadata']['complexity']}")
        print(f"  Strategy: {result['metadata']['strategy']}")
        print(f"  Answer: {result['answer'][:100]}...")
