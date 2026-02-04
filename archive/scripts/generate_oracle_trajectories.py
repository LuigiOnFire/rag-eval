#!/usr/bin/env python3
"""
Generate trajectories using oracle-guided search.

The oracle sees FULL context to make decisions, leading to higher success rates.
The trajectories are recorded with COMPRESSED observations for BC training.

Usage:
    python scripts/generate_oracle_trajectories.py --num_samples 100 --output results/oracle_trajectories.json
    
To use a stronger oracle model:
    ollama pull llama3.2:3b  # Download first
    python scripts/generate_oracle_trajectories.py --oracle_model llama3.2:3b
"""

import argparse
import json
import logging
import sys
import os
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oracle_search import OracleGuidedSearch, LLMOracleWrapper, generate_trajectories_with_oracle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRetriever:
    """Simple BM25 retriever wrapper."""
    
    def __init__(self, corpus_path: str, index_path: str = None):
        from rank_bm25 import BM25Okapi
        import numpy as np
        
        logger.info(f"Loading corpus from {corpus_path}...")
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.texts = [p.get("text", p.get("content", "")) for p in self.corpus]
        tokenized = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"Indexed {len(self.corpus)} passages")
        
    def retrieve(self, query: str, k: int = 5):
        """Retrieve top-k documents."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            doc = self.corpus[idx].copy()
            doc["score"] = float(scores[idx])
            results.append(doc)
        
        return results


class OllamaGenerator:
    """LLM wrapper using Ollama."""
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        import requests
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""


def load_hotpotqa(num_samples: int, seed: int = 42):
    """Load HotPotQA questions."""
    from datasets import load_dataset
    
    logger.info("Loading HotPotQA dataset...")
    ds = load_dataset('hotpot_qa', 'distractor', split='validation', trust_remote_code=True)
    
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    
    queries = []
    for idx in indices:
        item = ds[idx]
        queries.append({
            "question": item["question"],
            "answer": item["answer"],
        })
    
    logger.info(f"Loaded {len(queries)} questions")
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate oracle-guided trajectories")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of questions to process")
    parser.add_argument("--output", type=str, default="results/oracle_trajectories.json", help="Output path")
    parser.add_argument("--corpus", type=str, default="data/processed/passages.json", help="Corpus path")
    parser.add_argument("--cost_table", type=str, default="results/cost_table.json", help="Cost table path")
    parser.add_argument("--generator_model", type=str, default="llama3:8b", help="Generator LLM")
    parser.add_argument("--oracle_model", type=str, default="llama3:8b", help="Oracle LLM (can be same or stronger)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load cost table
    logger.info(f"Loading cost table from {args.cost_table}...")
    try:
        with open(args.cost_table) as f:
            cost_table = json.load(f)
    except FileNotFoundError:
        logger.warning("Cost table not found, using defaults")
        cost_table = {
            "GENERATE_LLM": 0.015,
            "DECOMPOSE_LLM": 0.028,
            "RETRIEVE_KEYWORD": 0.009,
            "REASON_LLM": 0.027,
        }
    
    # Initialize components
    logger.info("Initializing retriever...")
    retriever = SimpleRetriever(args.corpus)
    
    logger.info(f"Initializing generator ({args.generator_model})...")
    generator = OllamaGenerator(args.generator_model)
    
    logger.info(f"Initializing oracle ({args.oracle_model})...")
    oracle = LLMOracleWrapper(args.oracle_model)
    
    # Load questions
    queries = load_hotpotqa(args.num_samples, args.seed)
    
    # Generate trajectories
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting oracle-guided trajectory generation")
    logger.info(f"  Questions: {len(queries)}")
    logger.info(f"  Generator: {args.generator_model}")
    logger.info(f"  Oracle: {args.oracle_model}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    trajectories = generate_trajectories_with_oracle(
        queries=queries,
        retriever=retriever,
        generator=generator,
        oracle=oracle,
        cost_table=cost_table,
        output_path=args.output,
        max_trajectories=args.num_samples,
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATION COMPLETE")
    logger.info(f"  Success: {len(trajectories)}/{len(queries)} ({len(trajectories)/len(queries)*100:.1f}%)")
    logger.info(f"  Duration: {duration/60:.1f} minutes")
    logger.info(f"  Output: {args.output}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
