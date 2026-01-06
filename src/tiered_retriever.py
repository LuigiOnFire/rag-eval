"""
Tiered Hybrid Retriever with Energy-Based Cost Profiles.

Implements Gemini's "Energy Tiers" approach:
- RETRIEVE_FAST: BM25 only (low cost, exact matches)
- RETRIEVE_SMART: Dense only (medium cost, semantic)  
- RETRIEVE_DEEP: Hybrid BM25+Dense+RRF (high cost, nuclear option)
"""
import json
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss
except ImportError as e:
    logging.warning(f"Missing dependency for hybrid retrieval: {e}")

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass 
class Document:
    """A retrieved document with metadata."""
    id: str
    title: str
    text: str
    score: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

logger = logging.getLogger(__name__)


class TieredHybridRetriever:
    """
    Energy-efficient tiered retrieval system.
    
    Cost Profile:
    - FAST: BM25 only (~0.01 Wh)
    - SMART: Dense only (~0.05 Wh)  
    - DEEP: Hybrid + RRF (~0.15 Wh)
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize tiered retriever."""
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        # Indexes
        self.bm25 = None
        self.dense_index = None
        self.passages = None
        self.embeddings = None
        
        # Cost tracking
        self.energy_costs = {
            'FAST': 0.01,    # BM25 only
            'SMART': 0.05,   # Dense only
            'DEEP': 0.15     # Hybrid + RRF
        }
        
    def build_index(self, passages: List[Dict]):
        """Build all indexes (BM25 + Dense)."""
        logger.info(f"Building tiered indexes for {len(passages)} passages...")
        self.passages = passages
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        texts = [p["text"] for p in passages]
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        # Build Dense index
        logger.info(f"Building dense index with {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Encode all passages (batch processing for efficiency)
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings)
        
        # Build FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.dense_index.add(self.embeddings)
        
        logger.info(f"✓ Built BM25 + Dense indexes ({dimension}D embeddings)")
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()
    
    def retrieve_fast(self, query: str, top_k: int = 3) -> Tuple[List[Document], float]:
        """TIER 1: BM25 only - exact matches, lowest cost."""
        if not self.bm25:
            raise ValueError("BM25 index not built")
            
        # BM25 search
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        docs = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only positive scores
                doc = Document(
                    id=str(idx),
                    text=self.passages[idx]["text"],
                    title=self.passages[idx].get("title", f"Doc_{idx}"),
                    score=float(scores[idx])
                )
                docs.append(doc)
        
        return docs, self.energy_costs['FAST']
    
    def retrieve_smart(self, query: str, top_k: int = 5) -> Tuple[List[Document], float]:
        """TIER 2: Dense only - semantic understanding, medium cost."""
        if not self.embedding_model or not self.dense_index:
            raise ValueError("Dense index not built")
            
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search dense index
        scores, indices = self.dense_index.search(query_embedding, top_k)
        
        docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                doc = Document(
                    id=str(idx),
                    text=self.passages[idx]["text"],
                    title=self.passages[idx].get("title", f"Doc_{idx}"),
                    score=float(score)
                )
                docs.append(doc)
        
        return docs, self.energy_costs['SMART']
    
    def retrieve_deep(self, query: str, top_k: int = 8) -> Tuple[List[Document], float]:
        """TIER 3: Hybrid BM25+Dense+RRF - nuclear option, highest cost."""
        # Get results from both retrievers
        bm25_docs, _ = self.retrieve_fast(query, top_k=top_k)
        dense_docs, _ = self.retrieve_smart(query, top_k=top_k)
        
        # Apply Reciprocal Rank Fusion (RRF)
        fused_docs = self._reciprocal_rank_fusion(
            [bm25_docs, dense_docs], 
            k=60  # RRF parameter
        )
        
        # Return top-k after fusion
        return fused_docs[:top_k], self.energy_costs['DEEP']
    
    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
        
        Formula: score = sum(1 / (k + rank)) for each list
        """
        doc_scores = {}
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, 1):
                # Use title as key for deduplication
                key = doc.title
                if key not in doc_scores:
                    doc_scores[key] = {'doc': doc, 'score': 0.0}
                
                # Add RRF score
                doc_scores[key]['score'] += 1.0 / (k + rank)
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Update scores in documents
        fused_docs = []
        for item in sorted_docs:
            doc = item['doc']
            doc.score = item['score']
            fused_docs.append(doc)
        
        return fused_docs
    
    def search(self, query: str, tier: str = "FAST", top_k: int = None) -> Tuple[List[Document], float]:
        """
        Main search interface with tier selection.
        
        Args:
            query: Search query
            tier: "FAST", "SMART", or "DEEP"
            top_k: Number of results (uses tier defaults if None)
            
        Returns:
            (documents, energy_cost)
        """
        if tier == "FAST":
            k = top_k if top_k else 3
            return self.retrieve_fast(query, k)
        elif tier == "SMART":
            k = top_k if top_k else 5
            return self.retrieve_smart(query, k)
        elif tier == "DEEP":
            k = top_k if top_k else 8
            return self.retrieve_deep(query, k)
        else:
            raise ValueError(f"Unknown tier: {tier}. Use 'FAST', 'SMART', or 'DEEP'")
    
    def save_indexes(self, output_dir: str):
        """Save built indexes to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save dense embeddings and passages
        if self.embeddings is not None:
            np.save(output_path / "embeddings.npy", self.embeddings)
        
        if self.passages is not None:
            with open(output_path / "passages.json", "w") as f:
                json.dump(self.passages, f, indent=2)
        
        # Save FAISS index
        if self.dense_index is not None:
            faiss.write_index(self.dense_index, str(output_path / "faiss.index"))
        
        logger.info(f"Saved indexes to {output_path}")
    
    def load_indexes(self, index_dir: str):
        """Load pre-built indexes from disk."""
        index_path = Path(index_dir)
        
        # Load passages
        with open(index_path / "passages.json") as f:
            self.passages = json.load(f)
        
        # Load embeddings
        self.embeddings = np.load(index_path / "embeddings.npy")
        
        # Load FAISS index
        self.dense_index = faiss.read_index(str(index_path / "faiss.index"))
        
        # Rebuild BM25 (quick)
        texts = [p["text"] for p in self.passages]
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        logger.info(f"Loaded tiered indexes from {index_path}")


# Wrapper for structured agent compatibility
class TieredRetrieverWrapper:
    """Wrapper to make TieredHybridRetriever compatible with StructuredAgent."""
    
    def __init__(self, retriever: TieredHybridRetriever, default_tier: str = "FAST"):
        self.retriever = retriever
        self.default_tier = default_tier
        self.last_energy_cost = 0.0
    
    def search(self, query: str, top_k: int = 5, tier: Optional[str] = None) -> List[Dict]:
        """Search with tier selection."""
        use_tier = tier if tier else self.default_tier
        
        docs, cost = self.retriever.search(query, tier=use_tier, top_k=top_k)
        self.last_energy_cost = cost
        
        # Convert to dict format expected by StructuredAgent
        return [
            {
                'title': doc.title,
                'text': doc.content,
                'score': doc.score
            }
            for doc in docs
        ]
    
    def get_last_cost(self) -> float:
        """Get energy cost of last retrieval."""
        return self.last_energy_cost