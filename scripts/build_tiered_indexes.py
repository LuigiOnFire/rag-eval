#!/usr/bin/env python3
"""
Build tiered hybrid indexes for the structured agent.

Creates FAST (BM25) + SMART (Dense) + DEEP (Hybrid) retrieval tiers.
"""
import sys
sys.path.insert(0, 'src')

import json
import logging
from pathlib import Path
from tiered_retriever import TieredHybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Build all tiered indexes."""
    # Load passages
    passages_file = "data/processed/passages.json"
    index_dir = "data/indexes_tiered"
    
    logger.info(f"Loading passages from {passages_file}...")
    with open(passages_file) as f:
        passages = json.load(f)
    
    logger.info(f"Loaded {len(passages)} passages")
    
    # Create retriever and build indexes
    logger.info("Initializing tiered retriever...")
    retriever = TieredHybridRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info("Building all indexes (this may take several minutes)...")
    retriever.build_index(passages)
    
    # Save indexes
    logger.info(f"Saving indexes to {index_dir}...")
    retriever.save_indexes(index_dir)
    
    # Test all tiers
    logger.info("Testing all tiers...")
    test_query = "What is the nationality of Scott Derrickson?"
    
    for tier in ["FAST", "SMART", "DEEP"]:
        docs, cost = retriever.search(test_query, tier=tier, top_k=3)
        logger.info(f"{tier}: {len(docs)} docs, {cost:.3f} Wh - {[d.title[:30] for d in docs]}")
    
    logger.info("✅ Tiered indexes built successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())