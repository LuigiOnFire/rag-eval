#!/usr/bin/env python3
"""
Build HotpotQA corpus from distractor paragraphs.

Each question in the distractor split has 10 paragraphs:
- 2 gold supporting paragraphs (contain answer evidence)
- 8 distractor paragraphs (related but not answer-containing)

This ensures every validation question has relevant context in the corpus,
enabling meaningful baseline comparisons.

Usage:
    python scripts/build_hotpotqa_distractor_corpus.py
"""
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os


def main():
    print("=" * 60)
    print("Building HotpotQA Distractor Corpus")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/5] Loading HotpotQA validation set (distractor version)...")
    ds = load_dataset(
        "hotpot_qa", 
        "distractor", 
        split="validation", 
        trust_remote_code=True
    )
    # Cast to Dataset for type checker (split= returns Dataset, not DatasetDict)
    from datasets import Dataset
    ds = ds if isinstance(ds, Dataset) else ds  # type: ignore[assignment]
    print(f"      Loaded {len(ds)} questions")  # type: ignore[arg-type]

    # Extract all unique paragraphs (deduplicate by title)
    print("\n[2/5] Extracting unique paragraphs...")
    passages = {}  # title -> passage dict
    
    for item in tqdm(ds, desc="      Processing"):
        titles = item['context']['title']
        sentences_list = item['context']['sentences']
        
        for title, sentences in zip(titles, sentences_list):
            if title not in passages:
                text = " ".join(sentences)
                passages[title] = {
                    "title": title,
                    "text": text,
                    "source": "hotpotqa_distractor"
                }
    
    passages_list = list(passages.values())
    print(f"      Extracted {len(passages_list)} unique paragraphs")

    # Save passages
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    passages_path = output_dir / "passages.json"
    print(f"\n[3/5] Saving passages to {passages_path}...")
    with open(passages_path, 'w') as f:
        json.dump(passages_list, f, indent=2)

    # Generate embeddings
    print("\n[4/5] Generating embeddings...")
    print("      Loading model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    texts = [p["text"] for p in passages_list]
    print(f"      Encoding {len(texts)} passages...")
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=64,
        convert_to_numpy=True
    )
    embeddings = embeddings.astype(np.float32)
    
    # Normalize for cosine similarity (IndexFlatIP)
    faiss.normalize_L2(embeddings)
    
    embeddings_path = output_dir / "embeddings.npy"
    print(f"      Saving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)

    # Build Faiss index
    print("\n[5/5] Building Faiss index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # type: ignore[arg-type]
    
    index_dir = Path("data/indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    print(f"      Saving index to {index_path}...")
    faiss.write_index(index, str(index_path))

    # Summary
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"   Passages:    {len(passages_list):,}")
    print(f"   Embeddings:  {embeddings.shape}")
    print(f"   Index:       {index.ntotal:,} vectors")
    print(f"\n   File sizes:")
    print(f"   - passages.json:  {os.path.getsize(passages_path) / 1e6:.1f} MB")
    print(f"   - embeddings.npy: {os.path.getsize(embeddings_path) / 1e6:.1f} MB")
    print(f"   - faiss.index:    {os.path.getsize(index_path) / 1e6:.1f} MB")
    total_mb = (
        os.path.getsize(passages_path) + 
        os.path.getsize(embeddings_path) + 
        os.path.getsize(index_path)
    ) / 1e6
    print(f"   - TOTAL:          {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
