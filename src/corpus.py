"""
Corpus loading and chunking module for RAG pipeline.

This module handles:
- Loading Wikipedia dataset from HuggingFace
- Sentence-based chunking with configurable size limits
- Saving processed passages to disk
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Iterator, Optional
from datasets import load_dataset
import nltk
from tqdm import tqdm
import yaml


# Download required NLTK data
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

logger = logging.getLogger(__name__)


class CorpusProcessor:
    """Handles corpus loading and chunking."""
    
    def __init__(
        self,
        dataset_name: str = "wikipedia",
        dataset_config: str = "20220301.en",
        num_documents: int = 1000,
        cache_dir: Optional[str] = None,
        max_chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 50,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        skip_first: int = 0,
    ):
        """
        Initialize corpus processor.
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration/version
            num_documents: Number of documents to process
            cache_dir: Directory to cache downloaded dataset
            max_chunk_size: Maximum tokens per chunk
            overlap: Token overlap between consecutive chunks
            min_chunk_size: Minimum tokens to keep a chunk
            shuffle: Whether to shuffle/randomize document selection
            shuffle_seed: Random seed for shuffling (for reproducibility)
            skip_first: Number of documents to skip (for getting different subsets)
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.num_documents = num_documents
        self.cache_dir = cache_dir
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.skip_first = skip_first
        
    def load_dataset(self) -> List[Dict]:
        """
        Load Wikipedia dataset from HuggingFace using streaming to avoid downloading everything.
        
        Returns:
            List of document dictionaries with 'title', 'text', 'url'
        """
        logger.info(
            f"Loading {self.num_documents} documents from "
            f"{self.dataset_name} ({self.dataset_config})..."
        )
        
        # Use streaming=True to avoid downloading the entire dataset
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            streaming=True  # Stream data instead of downloading everything
        )
        
        # Apply shuffling if requested
        if self.shuffle:
            logger.info(f"Shuffling dataset with seed={self.shuffle_seed}")
            dataset = dataset.shuffle(seed=self.shuffle_seed, buffer_size=10000)
        
        # Skip first N documents if requested
        if self.skip_first > 0:
            logger.info(f"Skipping first {self.skip_first} documents")
            dataset = dataset.skip(self.skip_first)
        
        # Select subset of documents from stream
        documents = []
        for idx, example in enumerate(tqdm(
            dataset.take(self.num_documents),
            desc="Loading documents",
            total=self.num_documents
        )):
            documents.append({
                "id": idx,
                "title": example.get("title", ""),
                "text": example.get("text", ""),
                "url": example.get("url", "")
            })
            
            if idx + 1 >= self.num_documents:
                break
            
        logger.info(f"Loaded {len(documents)} documents")
        
        # Log some sample titles to verify diversity
        if documents:
            sample_titles = [doc["title"] for doc in documents[:5]]
            logger.info(f"Sample titles: {sample_titles}")
        
        return documents
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences with token-based size limits.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # Simple word-based token approximation
            sentence_tokens = len(sentence.split())
            
            # If adding this sentence would exceed max size, save current chunk
            if current_size + sentence_tokens > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if current_size >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last few sentences for context
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    sent_size = len(sent.split())
                    if overlap_size + sent_size <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += sent_size
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if current_size >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process documents into chunked passages.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of passage dictionaries with metadata
        """
        logger.info(f"Chunking {len(documents)} documents...")
        
        passages = []
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self.chunk_by_sentences(doc["text"])
            
            for chunk_idx, chunk_text in enumerate(chunks):
                passages.append({
                    "passage_id": f"{doc['id']}_{chunk_idx}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "text": chunk_text,
                    "url": doc["url"],
                    "chunk_index": chunk_idx
                })
        
        logger.info(f"Created {len(passages)} passages from {len(documents)} documents")
        return passages
    
    def save_passages(self, passages: List[Dict], output_path: str):
        """
        Save passages to JSON file.
        
        Args:
            passages: List of passage dictionaries
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(passages, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(passages)} passages to {output_path}")
    
    def load_passages(self, input_path: str) -> List[Dict]:
        """
        Load passages from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            List of passage dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            passages = json.load(f)
        
        logger.info(f"Loaded {len(passages)} passages from {input_path}")
        return passages


def main():
    """Example usage of CorpusProcessor."""    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=config["logging"]["level"],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize processor
    processor = CorpusProcessor(
        dataset_name=config["corpus"]["dataset_name"],
        dataset_config=config["corpus"]["dataset_config"],
        num_documents=config["corpus"]["num_documents"],
        cache_dir=config["corpus"]["cache_dir"],
        max_chunk_size=config["chunking"]["max_chunk_size"],
        overlap=config["chunking"]["overlap"],
        min_chunk_size=config["chunking"]["min_chunk_size"],
        shuffle=config["corpus"].get("shuffle", False),
        shuffle_seed=config["corpus"].get("shuffle_seed"),
        skip_first=config["corpus"].get("skip_first", 0)
    )
    
    # Load and process documents
    documents = processor.load_dataset()
    passages = processor.process_documents(documents)
    
    # Save passages
    processor.save_passages(passages, config["retriever"]["passages_path"])
    
    # Print statistics
    logger.info(f"\nCorpus Statistics:")
    logger.info(f"  Total documents: {len(documents)}")
    logger.info(f"  Total passages: {len(passages)}")
    logger.info(f"  Avg passages per doc: {len(passages) / len(documents):.2f}")
    
    # Sample passage
    logger.info(f"\nSample passage:")
    logger.info(f"  Title: {passages[0]['title']}")
    logger.info(f"  Text: {passages[0]['text'][:200]}...")


if __name__ == "__main__":
    main()
