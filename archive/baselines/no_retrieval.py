"""
No-retrieval baseline (generator only).

Lower bound baseline that generates answers without any retrieved context.
Measures how well the LLM can answer from parametric knowledge alone.
"""

from typing import List, Dict, Optional, Union, Any
import yaml
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import create_generator
from base import BaseRAG

logger = logging.getLogger(__name__)


class NoRetrievalRAG(BaseRAG):
    """
    No-retrieval baseline: generator only, no context.
    
    This represents a lower bound - purely parametric knowledge from the LLM.
    Useful for measuring:
    - How much RAG actually helps vs pure LLM
    - Energy cost of generation without retrieval overhead
    - LLM's inherent knowledge on the query domain
    """
    
    def __init__(
        self,
        config_path: str = "config_local.yaml"
    ):
        """
        Initialize No-retrieval RAG.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._generator = None
        self._config = None
    
    @property
    def config(self) -> Dict:
        """Lazy-load configuration."""
        if self._config is None:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    @property
    def generator(self):
        """Lazy-load generator on first access."""
        if self._generator is None:
            self._generator = self._load_generator()
        return self._generator
    
    def _load_generator(self):
        """Load generator from config."""
        logger.info("Loading generator for no-retrieval baseline...")
        generator_config = self.config["generator"].copy()
        generator_config["system_prompt"] = self.config["prompt"].get("system")
        return create_generator(generator_config)
    
    @property
    def name(self) -> str:
        """Return the name of this RAG system."""
        return "no_retrieval"
    
    def _format_prompt_no_context(self, question: str) -> str:
        """
        Format prompt without any context.
        
        Args:
            question: The user's question
            
        Returns:
            Formatted prompt for no-context generation
        """
        # Simple prompt without context - just ask the question directly
        return f"""Answer the following question based on your knowledge.
If you're not sure, say "I don't know."

Question: {question}

Answer:"""
    
    def answer(self, query: str, return_trace: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Answer a question without retrieval.
        
        Args:
            query: The question to answer
            return_trace: If True, return dict with answer + metadata
        
        Returns:
            str or dict depending on return_trace
        """
        # Format prompt without context
        prompt = self._format_prompt_no_context(query)
        
        # Call generator directly with empty context
        result = self.generator.generate(
            question=query,
            context_passages=[],  # No context!
            prompt_template=prompt.replace(f"Question: {query}", "Question: {question}")
        )
        
        answer = result.get("answer", "")
        
        if return_trace:
            return {
                "answer": answer,
                "retrieved_docs": [],  # Empty - no retrieval
                "metadata": {
                    "num_retrieved": 0,
                    "model": self.generator.model_name,
                    "baseline_type": "no_retrieval"
                }
            }
        else:
            return answer
    
    def batch_answer(
        self, 
        queries: List[str], 
        return_trace: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Answer multiple questions without retrieval.
        
        Args:
            queries: List of questions
            return_trace: If True, include metadata in results
        
        Returns:
            List of answers (str or dict)
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Generating answer {i}/{len(queries)} (no retrieval)")
            try:
                result = self.answer(query, return_trace=return_trace)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate answer for query {i}: {e}")
                if return_trace:
                    results.append({
                        "answer": "",
                        "retrieved_docs": [],
                        "metadata": {"error": str(e)}
                    })
                else:
                    results.append("")
        
        return results


def create_no_retrieval_rag(config_path: str = "config_local.yaml") -> NoRetrievalRAG:
    """
    Factory function to create NoRetrievalRAG instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized NoRetrievalRAG
    """
    return NoRetrievalRAG(config_path=config_path)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    rag = NoRetrievalRAG()
    print(f"System name: {rag.name}")
    
    # Test single query
    result = rag.answer("What is the capital of France?", return_trace=True)
    print(f"\nAnswer: {result['answer']}")
    print(f"Retrieved docs: {len(result['retrieved_docs'])}")
