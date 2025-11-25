"""
Rule-based query complexity classifier.

Uses heuristic patterns to classify queries without any model inference.
Fast and free, useful as a baseline classifier.
"""

import re
from typing import Literal

ComplexityLevel = Literal["simple", "moderate", "complex"]


class RuleBasedClassifier:
    """
    Heuristic-based query complexity classifier.
    
    Classification rules:
    - complex: Multi-hop indicators (comparisons, conjunctions, multiple entities)
    - simple: Short factoid questions (who/what/when/where + short)
    - moderate: Everything else
    """
    
    # Patterns indicating complex multi-hop questions
    COMPLEX_PATTERNS = [
        r'\b(and|both|also)\b.*\b(who|what|where|when)\b',  # "X and Y who..."
        r'\bcompare[ds]?\b',  # comparisons
        r'\bdifferen(ce|t)\b',  # differences
        r'\b(same|similar)\s+(as|to)\b',  # similarities
        r'\bbetween\b',  # "between X and Y"
        r'\brelat(ed|ion)\b',  # relationships
        r'\b(before|after)\b.*\b(before|after)\b',  # temporal chains
        r'\bwh(o|at|ich)\b.*\bthat\b.*\bwh(o|at|ich)\b',  # nested clauses
        r'\b(first|then|finally|next)\b',  # sequential reasoning
        r',.*,.*\?',  # Multiple clauses with commas
    ]
    
    # Patterns indicating simple factoid questions
    SIMPLE_PATTERNS = [
        r'^(who|what|when|where|which)\s+(is|was|are|were)\s+\w+\s*\?*$',  # "Who is X?"
        r'^(who|what)\s+(invented|discovered|founded|created|wrote)\s+',  # "Who invented X?"
        r'^(what|which)\s+(year|date|country|city|language)\b',  # "What year..."
        r'^(when|where)\s+(was|is|did)\s+\w+\s+(born|die|found|start)',  # "When was X born?"
    ]
    
    # Question word at start usually simpler
    SIMPLE_STARTERS = ['who is', 'what is', 'when was', 'where is', 'when did']
    
    def __init__(self):
        """Initialize the classifier with compiled regex patterns."""
        self.complex_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS]
        self.simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
    
    @property
    def name(self) -> str:
        return "rule_based"
    
    def classify(self, query: str) -> ComplexityLevel:
        """
        Classify query complexity using heuristic rules.
        
        Args:
            query: The question to classify
            
        Returns:
            'simple', 'moderate', or 'complex'
        """
        query_lower = query.lower().strip()
        word_count = len(query.split())
        
        # Check for complex patterns first
        for pattern in self.complex_patterns:
            if pattern.search(query):
                return "complex"
        
        # Check for multiple question marks (compound questions)
        if query.count('?') > 1:
            return "complex"
        
        # Very long questions are likely complex
        if word_count > 25:
            return "complex"
        
        # Check for simple patterns
        for pattern in self.simple_patterns:
            if pattern.search(query):
                return "simple"
        
        # Simple starters with short length
        if any(query_lower.startswith(starter) for starter in self.SIMPLE_STARTERS):
            if word_count <= 10:
                return "simple"
        
        # Short questions are often simple
        if word_count <= 6:
            return "simple"
        
        # Default to moderate
        return "moderate"
    
    def batch_classify(self, queries: list[str]) -> list[ComplexityLevel]:
        """Classify multiple queries."""
        return [self.classify(q) for q in queries]


# Convenience function
def create_rule_based_classifier() -> RuleBasedClassifier:
    """Factory function to create RuleBasedClassifier."""
    return RuleBasedClassifier()


if __name__ == "__main__":
    # Test the classifier
    classifier = RuleBasedClassifier()
    
    test_queries = [
        # Should be simple
        "Who is Barack Obama?",
        "What is the capital of France?",
        "When was Einstein born?",
        
        # Should be moderate
        "What are the main causes of climate change?",
        "How does photosynthesis work in plants?",
        "What is the significance of the Treaty of Versailles?",
        
        # Should be complex
        "What is the difference between mitosis and meiosis?",
        "Who was the president when both World War I ended and the League of Nations was founded?",
        "Compare the economic policies of Reagan and Clinton.",
        "What happened first, the French Revolution or the American Revolution, and how did one influence the other?",
    ]
    
    print("Rule-Based Classifier Test Results:")
    print("=" * 60)
    for query in test_queries:
        complexity = classifier.classify(query)
        print(f"[{complexity:8}] {query[:50]}...")
