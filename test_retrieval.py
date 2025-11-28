#!/usr/bin/env python3
"""Quick test of retrieval on HotpotQA question. Basically a debugging script."""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load everything
print('Loading...')
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/indexes/faiss.index")
with open("data/processed/passages.json") as f:
    passages = json.load(f)

# Find Scott Derrickson index in passages
scott_idx = None
for i, p in enumerate(passages):
    if p["title"] == "Scott Derrickson":
        scott_idx = i
        break
print(f"Scott Derrickson is at passage index: {scott_idx}")

# Compute its score
query = "Were Scott Derrickson and Ed Wood of the same nationality?"
q_emb = model.encode([query], normalize_embeddings=True)[0]
scott_emb = index.reconstruct(scott_idx)
score = np.dot(q_emb, scott_emb)
print(f"Scott Derrickson score: {score:.4f}")

# Count how many passages have higher scores
count_higher = 0
for i in range(index.ntotal):
    emb = index.reconstruct(i)
    s = np.dot(q_emb, emb)
    if s > score:
        count_higher += 1
print(f"Passages with higher score: {count_higher}")
print(f"Scott Derrickson rank: {count_higher + 1}")
print(f"To retrieve it, need k >= {count_higher + 1}")
