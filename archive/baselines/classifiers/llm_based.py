"""
LLM-based query complexity classifier.

Uses the LLM (Ollama/Mistral) to classify query complexity via prompting.
More accurate than rule-based but requires inference.
"""

import logging
from typing import Literal, Optional
import yaml
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from generator import OllamaGenerator

logger = logging.getLogger(__name__)

ComplexityLevel = Literal["simple", "moderate", "complex"]


class LLMClassifier:
    """
    LLM-based query complexity classifier.
    
    Uses Ollama to classify queries into simple/moderate/complex.
    Caches results for efficiency.
    """
    
    CLASSIFICATION_PROMPT = """Classify this question's complexity for a retrieval-augmented system.

Categories:
- A (Simple): Basic factoid question. One entity lookup, no reasoning needed.
  Examples: "Who is the CEO of Apple?", "What year was the Eiffel Tower built?"

- B (Moderate): Requires some context or reasoning, but single-hop retrieval suffices.
  Examples: "What are the main causes of World War I?", "How does photosynthesis work?"

- C (Complex): Multi-hop reasoning, comparisons, or requires combining multiple facts.
  Examples: "What is the difference between RNA and DNA?", "Who was president when both X and Y happened?"

Question: {query}

Respond with ONLY the letter (A, B, or C):"""

    def __init__(
        self,
        config_path: str = "config_local.yaml",
        model_name: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        cache_results: bool = True
    ):
        """
        Initialize LLM classifier.
        
        Args:
            config_path: Path to config file (for model settings)
            model_name: Override model name (default: from config)
            base_url: Ollama API base URL
            cache_results: Whether to cache classification results
        """
        self.config_path = config_path
        self.base_url = base_url.rstrip("/")
        self.cache_results = cache_results
        self._cache: dict[str, ComplexityLevel] = {}
        
        # Load model name from config if not specified
        if model_name is None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            model_name = config["generator"]["model_name"]
        
        # Remove ollama/ prefix if present
        self.model_name = model_name.replace("ollama/", "")
        
        logger.info(f"LLMClassifier initialized with model: {self.model_name}")
    
    @property
    def name(self) -> str:
        return f"llm_based_{self.model_name}"
    
    def _call_ollama(self, prompt: str) -> str:
        """Make a direct call to Ollama API for classification."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Deterministic
                        "num_predict": 5,    # Only need A, B, or C
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return "B"  # Default to moderate on error
    
    def classify(self, query: str) -> ComplexityLevel:
        """
        Classify query complexity using LLM.
        
        Args:
            query: The question to classify
            
        Returns:
            'simple', 'moderate', or 'complex'
        """
        # Check cache
        if self.cache_results and query in self._cache:
            return self._cache[query]
        
        # Build prompt
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)
        
        # Get LLM response
        response = self._call_ollama(prompt)
        
        # Parse response - look for A, B, or C
        response_upper = response.upper().strip()
        
        if 'A' in response_upper[:3]:
            complexity = "simple"
        elif 'C' in response_upper[:3]:
            complexity = "complex"
        else:
            complexity = "moderate"  # Default to B/moderate
        
        # Cache result
        if self.cache_results:
            self._cache[query] = complexity
        
        return complexity
    
    def batch_classify(self, queries: list[str]) -> list[ComplexityLevel]:
        """
        Classify multiple queries.
        
        Note: This calls the LLM sequentially. For large batches,
        consider using async or batching strategies.
        """
        results = []
        for i, query in enumerate(queries):
            logger.debug(f"Classifying query {i+1}/{len(queries)}")
            results.append(self.classify(query))
        return results
    
    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Return number of cached classifications."""
        return len(self._cache)


# Convenience function
def create_llm_classifier(config_path: str = "config_local.yaml") -> LLMClassifier:
    """Factory function to create LLMClassifier."""
    return LLMClassifier(config_path=config_path)


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)
    
    classifier = LLMClassifier()
    
    test_queries = [
        # Should be simple
        "Who is Barack Obama?",
        "What is the capital of France?",
        "When was Einstein born?",
        
        # Should be moderate
        "What are the main causes of climate change?",
        "How does photosynthesis work in plants?",
        
        # Should be complex
        "What is the difference between mitosis and meiosis?",
        "Who was the president when both World War I ended and the League of Nations was founded?",
    ]
    
    print(f"LLM Classifier Test Results (model: {classifier.model_name}):")
    print("=" * 60)
    for query in test_queries:
        complexity = classifier.classify(query)
        print(f"[{complexity:8}] {query[:50]}...")
    
    print(f"\nCache size: {classifier.cache_size}")
