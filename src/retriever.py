"""
Retrieval module using Faiss and SentenceTransformers.

This module handles:
- Embedding passages using SentenceTransformers
- Building and saving Faiss index
- Searching for relevant passages given a query
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FaissRetriever:
    """Retriever using Faiss vector search with SentenceTransformers embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize Faiss retriever.
        
        Args:
            model_name: SentenceTransformer model name
            embedding_dim: Dimension of embeddings
            normalize_embeddings: Whether to normalize embeddings (for cosine similarity)
            batch_size: Batch size for encoding
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=device)
        
        self.index = None
        self.passages = None
        
    def encode_passages(self, passages: List[Dict]) -> np.ndarray:
        """
        Encode passages to embeddings.
        
        Args:
            passages: List of passage dictionaries with 'text' field
            
        Returns:
            Numpy array of embeddings (num_passages, embedding_dim)
        """
        texts = [p["text"] for p in passages]
        
        logger.info(f"Encoding {len(texts)} passages...")
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def build_index(
        self,
        passages: List[Dict],
        index_type: str = "IndexFlatIP"
    ) -> faiss.Index:
        """
        Build Faiss index from passages.
        
        Args:
            passages: List of passage dictionaries
            index_type: Type of Faiss index ('IndexFlatIP' for inner product)
            
        Returns:
            Faiss index
        """
        self.passages = passages
        
        # Encode passages
        embeddings = self.encode_passages(passages)
        
        # Create Faiss index
        if index_type == "IndexFlatIP":
            # Inner product (cosine similarity if embeddings are normalized)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "IndexFlatL2":
            # L2 distance
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        logger.info(f"Building {index_type} index...")
        index.add(embeddings)
        
        logger.info(f"Index built with {index.ntotal} vectors")
        self.index = index
        
        return index
    
    def save_index(
        self,
        index_path: str,
        passages_path: str,
        embeddings_path: Optional[str] = None
    ):
        """
        Save index, passages, and optionally embeddings to disk.
        
        Args:
            index_path: Path to save Faiss index
            passages_path: Path to save passages JSON
            embeddings_path: Optional path to save embeddings numpy array
        """
        index_path = Path(index_path)
        passages_path = Path(passages_path)
        
        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        passages_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved index to {index_path}")
        
        # Save passages
        with open(passages_path, 'w', encoding='utf-8') as f:
            json.dump(self.passages, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved passages to {passages_path}")
        
        # Optionally save embeddings
        if embeddings_path:
            embeddings_path = Path(embeddings_path)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Reconstruct embeddings from index
            embeddings = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
            for i in range(self.index.ntotal):
                embeddings[i] = self.index.reconstruct(i)
            
            np.save(embeddings_path, embeddings)
            logger.info(f"Saved embeddings to {embeddings_path}")
    
    def load_index(self, index_path: str, passages_path: str):
        """
        Load index and passages from disk.
        
        Args:
            index_path: Path to Faiss index file
            passages_path: Path to passages JSON file
        """
        # Load index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded index from {index_path} with {self.index.ntotal} vectors")
        
        # Load passages
        with open(passages_path, 'r', encoding='utf-8') as f:
            self.passages = json.load(f)
        logger.info(f"Loaded {len(self.passages)} passages from {passages_path}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve relevant passages for a query.
        
        Args:
            query: Query string
            top_k: Number of passages to retrieve
            score_threshold: Minimum similarity score (optional filtering)
            
        Returns:
            Tuple of (retrieved_passages, scores)
        """
        if self.index is None or self.passages is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        # Search index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Extract results
        retrieved_passages = []
        retrieved_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold:
                passage = self.passages[idx].copy()
                passage["retrieval_score"] = float(score)
                retrieved_passages.append(passage)
                retrieved_scores.append(float(score))
        
        return retrieved_passages, retrieved_scores
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[List[Dict], List[float]]]:
        """
        Retrieve passages for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of passages to retrieve per query
            score_threshold: Minimum similarity score
            
        Returns:
            List of (retrieved_passages, scores) tuples
        """
        results = []
        for query in tqdm(queries, desc="Retrieving"):
            passages, scores = self.retrieve(query, top_k, score_threshold)
            results.append((passages, scores))
        
        return results


def main():
    """Example usage of FaissRetriever."""
    import yaml
    from corpus import CorpusProcessor
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=config["logging"]["level"],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load or create passages
    passages_path = config["retriever"]["passages_path"]
    if Path(passages_path).exists():
        processor = CorpusProcessor()
        passages = processor.load_passages(passages_path)
    else:
        logger.error(f"Passages not found at {passages_path}. Run corpus.py first.")
        return
    
    # Initialize retriever
    retriever = FaissRetriever(
        model_name=config["retriever"]["model_name"],
        embedding_dim=config["retriever"]["embedding_dim"],
        normalize_embeddings=config["retriever"]["normalize_embeddings"],
        batch_size=config["retriever"]["batch_size"]
    )
    
    # Build index
    retriever.build_index(
        passages,
        index_type=config["retriever"]["index_type"]
    )
    
    # Save index
    retriever.save_index(
        index_path=config["retriever"]["index_path"],
        passages_path=config["retriever"]["passages_path"],
        embeddings_path=config["retriever"]["embeddings_path"]
    )
    
    # Test retrieval
    test_query = "What is machine learning?"
    logger.info(f"\nTest query: {test_query}")
    
    passages, scores = retriever.retrieve(
        test_query,
        top_k=config["retrieval"]["top_k"]
    )
    
    for i, (passage, score) in enumerate(zip(passages, scores), 1):
        logger.info(f"\n{i}. [Score: {score:.4f}] {passage['title']}")
        logger.info(f"   {passage['text'][:200]}...")


if __name__ == "__main__":
    main()
