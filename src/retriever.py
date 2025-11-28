"""
Retrieval module supporting BM25 and dense (Faiss) retrieval.

This module handles:
- BM25 sparse retrieval (default, better for entity matching)
- Dense retrieval using Faiss and SentenceTransformers
- Searching for relevant passages given a query
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
def _import_faiss():
    import faiss
    return faiss

def _import_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def _import_bm25():
    from rank_bm25 import BM25Okapi
    return BM25Okapi


def tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    # Lowercase and split on non-alphanumeric characters
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class BM25Retriever:
    """Retriever using BM25 sparse retrieval (better for entity matching)."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.model_name = "BM25Okapi"  # For compatibility with pipeline metadata
        self.bm25 = None
        self.passages = None
        self.tokenized_corpus = None
        
    def build_index(self, passages: List[Dict]):
        """
        Build BM25 index from passages.
        
        Args:
            passages: List of passage dictionaries with 'text' field
        """
        BM25Okapi = _import_bm25()
        
        self.passages = passages
        
        # Tokenize all passages
        logger.info(f"Tokenizing {len(passages)} passages for BM25...")
        self.tokenized_corpus = [tokenize(p["text"]) for p in passages]
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"BM25 index built with {len(passages)} documents")
    
    def save_index(self, index_path: str, passages_path: str):
        """
        Save BM25 index and passages to disk.
        
        Args:
            index_path: Path to save BM25 index (pickle)
            passages_path: Path to save passages JSON
        """
        index_path_obj = Path(index_path)
        passages_path_obj = Path(passages_path)
        
        # Create directories
        index_path_obj.parent.mkdir(parents=True, exist_ok=True)
        passages_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 index
        bm25_path = index_path_obj.with_suffix('.bm25.pkl')
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        logger.info(f"Saved BM25 index to {bm25_path}")
        
        # Save passages
        with open(passages_path_obj, 'w', encoding='utf-8') as f:
            json.dump(self.passages, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved passages to {passages_path_obj}")
    
    def load_index(self, index_path: str, passages_path: str):
        """
        Load BM25 index and passages from disk.
        
        Args:
            index_path: Path to BM25 index file (or base path)
            passages_path: Path to passages JSON file
        """
        index_path_obj = Path(index_path)
        
        # Determine BM25 index path
        if str(index_path_obj).endswith('.bm25.pkl'):
            bm25_path = index_path_obj
        else:
            bm25_path = index_path_obj.with_suffix('.bm25.pkl')
        
        # Try loading BM25 index
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.tokenized_corpus = data['tokenized_corpus']
            logger.info(f"Loaded BM25 index from {bm25_path}")
        else:
            logger.info(f"No BM25 index found at {bm25_path}, will build from passages")
        
        # Load passages
        with open(passages_path, 'r', encoding='utf-8') as f:
            self.passages = json.load(f)
        logger.info(f"Loaded {len(self.passages)} passages from {passages_path}")
        
        # Build index if not loaded, then save for future use
        if self.bm25 is None:
            self.build_index(self.passages)
            # Save the built index for future runs
            self._save_bm25_index(bm25_path)
    
    def _save_bm25_index(self, bm25_path: Path):
        """Save BM25 index to disk."""
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        logger.info(f"Saved BM25 index to {bm25_path}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve relevant passages for a query using BM25.
        
        Args:
            query: Query string
            top_k: Number of passages to retrieve
            score_threshold: Minimum BM25 score (optional filtering)
            
        Returns:
            Tuple of (retrieved_passages, scores)
        """
        if self.bm25 is None or self.passages is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        # Tokenize query
        tokenized_query = tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Extract results
        retrieved_passages = []
        retrieved_scores = []
        
        for idx in top_indices:
            score = scores[idx]
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
            score_threshold: Minimum BM25 score
            
        Returns:
            List of (retrieved_passages, scores) tuples
        """
        from tqdm import tqdm
        results = []
        for query in tqdm(queries, desc="Retrieving (BM25)"):
            passages, scores = self.retrieve(query, top_k, score_threshold)
            results.append((passages, scores))
        
        return results


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
        
        SentenceTransformer = _import_sentence_transformers()
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
    ):
        """
        Build Faiss index from passages.
        
        Args:
            passages: List of passage dictionaries
            index_type: Type of Faiss index ('IndexFlatIP' for inner product)
            
        Returns:
            Faiss index
        """
        faiss = _import_faiss()
        
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
        index.add(embeddings)  # type: ignore[arg-type]
        
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
        faiss = _import_faiss()
        
        index_path_obj = Path(index_path)
        passages_path_obj = Path(passages_path)
        
        # Create directories
        index_path_obj.parent.mkdir(parents=True, exist_ok=True)
        passages_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(index_path_obj))
        logger.info(f"Saved index to {index_path_obj}")
        
        # Save passages
        with open(passages_path_obj, 'w', encoding='utf-8') as f:
            json.dump(self.passages, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved passages to {passages_path_obj}")
        
        # Optionally save embeddings
        if embeddings_path:
            embeddings_path_obj = Path(embeddings_path)
            embeddings_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Reconstruct embeddings from index (type: ignore for faiss index attributes)
            embeddings = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)  # type: ignore[union-attr]
            for i in range(self.index.ntotal):  # type: ignore[union-attr]
                embeddings[i] = self.index.reconstruct(i)  # type: ignore[union-attr]
            
            np.save(str(embeddings_path_obj), embeddings)
            logger.info(f"Saved embeddings to {embeddings_path_obj}")
    
    def load_index(self, index_path: str, passages_path: str):
        """
        Load index and passages from disk.
        
        Args:
            index_path: Path to Faiss index file
            passages_path: Path to passages JSON file
        """
        faiss = _import_faiss()
        
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
        
        # Search index (type: ignore for faiss SWIG bindings)
        scores, indices = self.index.search(query_embedding, top_k)  # type: ignore[union-attr]
        
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
        from tqdm import tqdm
        results = []
        for query in tqdm(queries, desc="Retrieving (Dense)"):
            passages, scores = self.retrieve(query, top_k, score_threshold)
            results.append((passages, scores))
        
        return results


def create_retriever(
    retriever_type: Literal["bm25", "dense"] = "bm25",
    **kwargs
):
    """
    Factory function to create a retriever.
    
    Args:
        retriever_type: Type of retriever ("bm25" or "dense")
        **kwargs: Additional arguments passed to the retriever constructor
        
    Returns:
        BM25Retriever or FaissRetriever instance
    """
    if retriever_type == "bm25":
        return BM25Retriever()
    elif retriever_type == "dense":
        return FaissRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Use 'bm25' or 'dense'.")


def main():
    """Example usage of retrievers."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description="Test retriever")
    parser.add_argument("--type", choices=["bm25", "dense"], default="bm25",
                        help="Retriever type (default: bm25)")
    parser.add_argument("--query", type=str, 
                        default="Were Scott Derrickson and Ed Wood of the same nationality?",
                        help="Test query")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to return (default: 10)")
    args = parser.parse_args()
    
    # Load config
    with open("config_local.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=config["logging"]["level"],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    passages_path = config["retriever"]["passages_path"]
    index_path = config["retriever"]["index_path"]
    
    # Create retriever based on type
    if args.type == "bm25":
        logger.info("Using BM25 retriever")
        retriever = BM25Retriever()
        retriever.load_index(index_path, passages_path)
    else:
        logger.info("Using Dense (Faiss) retriever")
        retriever = FaissRetriever(
            model_name=config["retriever"]["model_name"],
            embedding_dim=config["retriever"]["embedding_dim"],
            normalize_embeddings=config["retriever"]["normalize_embeddings"],
            batch_size=config["retriever"]["batch_size"]
        )
        retriever.load_index(index_path, passages_path)
    
    # Test retrieval
    test_query = args.query
    logger.info(f"\nTest query: {test_query}")
    
    passages, scores = retriever.retrieve(
        test_query,
        top_k=args.top_k
    )
    
    print(f"\nTop {len(passages)} results:")
    for i, (passage, score) in enumerate(zip(passages, scores), 1):
        print(f"\n{i}. [Score: {score:.4f}] {passage['title']}")
        print(f"   {passage['text'][:150]}...")


if __name__ == "__main__":
    main()
