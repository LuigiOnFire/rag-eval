#!/usr/bin/env python3
"""
Rebuild the Faiss index from passages.

Usage:
    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --passages data/processed/passages.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import FaissRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Rebuild Faiss index from passages")
    parser.add_argument("--passages", type=str, default="data/processed/passages.json",
                        help="Path to passages JSON file")
    parser.add_argument("--index", type=str, default="data/indexes/faiss.index",
                        help="Output path for Faiss index")
    parser.add_argument("--embeddings", type=str, default="data/processed/embeddings.npy",
                        help="Output path for embeddings")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model name")
    parser.add_argument("--embedding-dim", type=int, default=384,
                        help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding")
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    passages_path = project_root / args.passages
    index_path = project_root / args.index
    embeddings_path = project_root / args.embeddings
    
    # Load passages
    logger.info(f"Loading passages from {passages_path}")
    with open(passages_path, 'r') as f:
        passages = json.load(f)
    logger.info(f"Loaded {len(passages)} passages")
    
    # Initialize retriever
    logger.info(f"Initializing retriever with model: {args.model}")
    retriever = FaissRetriever(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        normalize_embeddings=True,
        batch_size=args.batch_size
    )
    
    # Build index
    logger.info("Building index...")
    retriever.build_index(passages, index_type='IndexFlatIP')
    
    # Save index
    logger.info(f"Saving index to {index_path}")
    retriever.save_index(
        index_path=str(index_path),
        passages_path=str(passages_path),
        embeddings_path=str(embeddings_path)
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
