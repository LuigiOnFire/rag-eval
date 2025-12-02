"""
End-to-end RAG pipeline orchestration.

This module handles:
- Loading retriever and generator
- Running retrieval → generation pipeline
- Batch processing queries
- Saving results
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import yaml

from .retriever import FaissRetriever, BM25Retriever, create_retriever
from .generator import create_generator
from .base import BaseRAG

logger = logging.getLogger(__name__)

# Type alias for any retriever
Retriever = FaissRetriever | BM25Retriever


class RAGPipeline(BaseRAG):
    """End-to-end RAG pipeline implementing BaseRAG interface."""
    
    def __init__(
        self,
        retriever: Retriever,
        generator,  # Can be any generator (GeminiGenerator, OllamaGenerator, etc.)
        top_k: int = 5,
        score_threshold: float = 0.0,
        prompt_template: Optional[str] = None,
        name: str = "naive_rag"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Initialized retriever (BM25Retriever or FaissRetriever)
            generator: Initialized generator (GeminiGenerator, OllamaGenerator, etc.)
            top_k: Number of passages to retrieve
            score_threshold: Minimum retrieval score
            prompt_template: Optional prompt template
            name: Name identifier for this RAG system
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.prompt_template = prompt_template
        self._name = name
        
        logger.info("RAG Pipeline initialized")
    
    @property
    def name(self) -> str:
        """Return the name of this RAG system."""
        return self._name
    
    def answer(self, query: str, return_trace: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Answer a question using the RAG pipeline.
        
        Implements BaseRAG.answer().
        
        Args:
            query: The question to answer
            return_trace: If False, return only answer string
                         If True, return dict with answer + retrieved_docs + metadata
        
        Returns:
            str or dict depending on return_trace
        """
        result = self.run(query, verbose=False)
        
        if return_trace:
            return {
                "answer": result["answer"],
                "retrieved_docs": result["retrieved_passages"],
                "metadata": {
                    "retrieval_scores": result["retrieval_scores"],
                    "num_retrieved": result["num_retrieved"],
                    "model": self.generator.model_name
                }
            }
        else:
            return result["answer"]
    
    def batch_answer(
        self, 
        queries: List[str], 
        return_trace: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Answer multiple questions efficiently.
        
        Overrides BaseRAG.batch_answer() for better efficiency.
        """
        results = self.batch_run(queries, verbose=False)
        
        if return_trace:
            return [
                {
                    "answer": r.get("answer", ""),
                    "retrieved_docs": r.get("retrieved_passages", []),
                    "metadata": {
                        "num_retrieved": r.get("num_retrieved", 0),
                        "model": self.generator.model_name
                    }
                }
                for r in results
            ]
        else:
            return [r.get("answer", "") for r in results]
    
    def run(self, question: str, verbose: bool = False) -> Dict:
        """
        Run RAG pipeline for a single question.
        
        Args:
            question: Input question
            verbose: Whether to log intermediate steps
            
        Returns:
            Dictionary with question, answer, retrieved passages, and metadata
        """
        # Step 1: Retrieve relevant passages
        if verbose:
            logger.info(f"Question: {question}")
            logger.info(f"Retrieving top {self.top_k} passages...")
        
        passages, scores = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
            score_threshold=self.score_threshold
        )
        
        if verbose:
            logger.info(f"Retrieved {len(passages)} passages")
            for i, (passage, score) in enumerate(zip(passages, scores), 1):
                logger.info(f"  {i}. [{score:.4f}] {passage['title']}")
        
        # Step 2: Generate answer
        if verbose:
            logger.info("Generating answer...")
        
        result = self.generator.generate(
            question=question,
            context_passages=passages,
            prompt_template=self.prompt_template
        )
        
        # Add retrieval metadata
        result["retrieved_passages"] = passages
        result["retrieval_scores"] = scores
        result["num_retrieved"] = len(passages)
        
        if verbose:
            logger.info(f"Answer: {result['answer']}\n")
        
        return result
    
    def batch_run(
        self,
        questions: List[str],
        verbose: bool = False,
        delay: float = 0.0
    ) -> List[Dict]:
        """
        Run RAG pipeline for multiple questions.
        
        Args:
            questions: List of questions
            verbose: Whether to log progress
            delay: Delay between generation calls (for rate limiting)
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Processing {len(questions)} questions...")
        
        # Step 1: Batch retrieve
        if verbose:
            logger.info("Retrieving passages for all questions...")
        
        all_passages = []
        for question in questions:
            passages, scores = self.retriever.retrieve(
                query=question,
                top_k=self.top_k,
                score_threshold=self.score_threshold
            )
            all_passages.append(passages)
        
        # Step 2: Batch generate
        if verbose:
            logger.info("Generating answers...")
        
        results = self.generator.batch_generate(
            questions=questions,
            context_passages_list=all_passages,
            prompt_template=self.prompt_template,
            delay=delay
        )
        
        # Add retrieval metadata
        for result, passages in zip(results, all_passages):
            if "error" not in result:
                result["retrieved_passages"] = passages
                result["num_retrieved"] = len(passages)
        
        logger.info(f"Processed {len(results)} questions")
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save pipeline results to JSON file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(results),
                "retriever_model": self.retriever.model_name,
                "generator_model": self.generator.model_name,
                "top_k": self.top_k
            },
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results to {output_file}")


def load_pipeline_from_config(config_path: str = "config.yaml") -> RAGPipeline:
    """
    Load RAG pipeline from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Initialized RAGPipeline
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize retriever using factory (supports bm25 and dense)
    logger.info("Loading retriever...")
    retriever_type = config["retriever"].get("type", "bm25")  # Default to BM25
    
    retriever = create_retriever(
        retriever_type=retriever_type,
        passages_path=config["retriever"]["passages_path"],
        # Dense retriever options
        model_name=config["retriever"].get("model_name"),
        embedding_dim=config["retriever"].get("embedding_dim", 384),
        normalize_embeddings=config["retriever"].get("normalize_embeddings", True),
        batch_size=config["retriever"].get("batch_size", 32)
    )
    
    # Load index
    if retriever_type == "bm25":
        index_path = config["retriever"].get("bm25_index_path", "./data/indexes/faiss.bm25.pkl")
    else:
        index_path = config["retriever"]["index_path"]
    
    retriever.load_index(
        index_path=index_path,
        passages_path=config["retriever"]["passages_path"]
    )
    
    # Initialize generator using factory (supports Gemini, Ollama, etc.)
    logger.info("Loading generator...")
    generator = create_generator(config["generator"])
    
    # Create pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=config["retrieval"]["top_k"],
        score_threshold=config["retrieval"]["score_threshold"],
        prompt_template=config["prompt"]["user_template"]
    )
    
    return pipeline


def main():
    """Example usage of RAG pipeline."""
    import sys
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run RAG pipeline')
    parser.add_argument('--config', type=str, default='config_local.yaml',
                        help='Path to config file (default: /config_local.yaml for local models)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=config["logging"]["level"],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load pipeline
    try:
        pipeline = load_pipeline_from_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.info("Make sure you have:")
        logger.info("  1. Run corpus.py to create passages")
        logger.info("  2. Run retriever.py to build the index")
        logger.info("  3. Set GEMINI_API_KEY in .env file")
        sys.exit(1)
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "Who invented the telephone?",
        "What is the capital of France?"
    ]
    
    logger.info("Running RAG pipeline on test questions...\n")
    
    # Run pipeline
    results = pipeline.batch_run(
        questions=test_questions,
        verbose=True,
        delay=1.0  # Rate limiting
    )
    
    # Save results
    output_path = Path(config["evaluation"]["output_dir"]) / "test_results.json"
    pipeline.save_results(results, str(output_path))
    
    # Print summary
    logger.info("\n=== Results Summary ===")
    for i, result in enumerate(results, 1):
        if "error" in result:
            logger.info(f"{i}. ERROR: {result['error']}")
        else:
            logger.info(f"{i}. Q: {result['question']}")
            logger.info(f"   A: {result['answer'][:100]}...")


if __name__ == "__main__":
    main()
