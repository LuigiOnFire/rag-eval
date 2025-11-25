#!/usr/bin/env python3
"""
Build a corpus of Wikipedia articles that match HotpotQA questions.

HotpotQA provides supporting_facts that tell us exactly which Wikipedia
articles contain the answer. This script extracts those titles and builds
a targeted corpus.

Two approaches:
1. Use HotpotQA's built-in context (already has the relevant paragraphs)
2. Fetch fresh articles from Wikipedia API (for more complete text)

We use approach 1 (HotpotQA context) because:
- It's guaranteed to contain the supporting facts
- No external API calls needed
- Consistent with how HotpotQA was designed
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set
from datasets import load_dataset
from tqdm import tqdm
import nltk

# Download required NLTK data
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_corpus_from_hotpotqa(
    num_questions: int = 100,
    split: str = "validation",
    max_chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_size: int = 50
) -> tuple[List[Dict], Set[str]]:
    """
    Extract Wikipedia articles from HotpotQA context.
    
    HotpotQA fullwiki includes context with multiple Wikipedia articles
    per question. We extract unique articles to build our corpus.
    
    Args:
        num_questions: Number of questions to extract articles from
        split: Dataset split (validation, train)
        max_chunk_size: Max tokens per chunk
        overlap: Token overlap between chunks
        min_chunk_size: Min tokens to keep chunk
        
    Returns:
        Tuple of (passages list, set of article titles)
    """
    logger.info(f"Loading HotpotQA {split} split...")
    
    # Load dataset
    ds = load_dataset("hotpot_qa", "fullwiki", split=split, streaming=True)
    
    # Track unique articles
    articles = {}  # title -> text
    
    logger.info(f"Extracting articles from {num_questions} questions...")
    
    for ex in tqdm(ds.take(num_questions), total=num_questions, desc="Processing questions"):
        # HotpotQA context structure: {"title": [...], "sentences": [[...], ...]}
        context = ex["context"]
        titles = context["title"]
        sentences_list = context["sentences"]
        
        for title, sentences in zip(titles, sentences_list):
            if title not in articles:
                # Join sentences into article text
                text = " ".join(sentences)
                if len(text.strip()) > 0:
                    articles[title] = text
    
    logger.info(f"Extracted {len(articles)} unique articles")
    
    # Chunk articles into passages
    passages = []
    for title, text in tqdm(articles.items(), desc="Chunking articles"):
        chunks = chunk_by_sentences(text, max_chunk_size, overlap, min_chunk_size)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            passages.append({
                "passage_id": f"{len(passages)}",
                "doc_id": title,
                "title": title,
                "text": chunk_text,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "chunk_index": chunk_idx
            })
    
    logger.info(f"Created {len(passages)} passages from {len(articles)} articles")
    
    return passages, set(articles.keys())


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_size: int = 50
) -> List[str]:
    """
    Chunk text by sentences with token-based size limits.
    """
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        if current_size + sentence_tokens > max_chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            if current_size >= min_chunk_size:
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_size = 0
            for sent in reversed(current_chunk):
                sent_size = len(sent.split())
                if overlap_size + sent_size <= overlap:
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
        if current_size >= min_chunk_size:
            chunks.append(chunk_text)
    
    return chunks


def main():
    """Build HotpotQA-matched corpus."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build HotpotQA-matched corpus")
    parser.add_argument("--num-questions", type=int, default=100,
                        help="Number of questions to extract articles from")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split")
    parser.add_argument("--output", type=str, default="data/processed/passages.json",
                        help="Output path for passages")
    parser.add_argument("--max-chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--min-chunk-size", type=int, default=50)
    
    args = parser.parse_args()
    
    # Extract corpus
    passages, titles = extract_corpus_from_hotpotqa(
        num_questions=args.num_questions,
        split=args.split,
        max_chunk_size=args.max_chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk_size
    )
    
    # Save passages
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(passages, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(passages)} passages to {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("CORPUS STATISTICS")
    print("="*60)
    print(f"Questions processed: {args.num_questions}")
    print(f"Unique articles: {len(titles)}")
    print(f"Total passages: {len(passages)}")
    print(f"Avg passages per article: {len(passages) / len(titles):.2f}")
    print()
    print("Sample titles:")
    for title in list(titles)[:10]:
        print(f"  - {title}")
    print()
    print("Sample passage:")
    print(f"  Title: {passages[0]['title']}")
    print(f"  Text: {passages[0]['text'][:200]}...")
    print("="*60)


if __name__ == "__main__":
    main()
