#!/usr/bin/env python3
"""Test retrieval for problem questions."""
import json
import sys
sys.path.insert(0, '/home/wcrawford/rag_eval')

from src.retriever import BM25Retriever

print("Loading passages...")
with open('/home/wcrawford/rag_eval/data/processed/passages.json') as f:
    passages = json.load(f)
print(f"Loaded {len(passages)} passages")

print("Building index...")
retriever = BM25Retriever()
retriever.build_index(passages)
print("Index built\n")

# Test queries for both problem questions
test_queries = [
    ('Q1-Scott/Ed', 'Scott Derrickson'),
    ('Q1-Scott/Ed', 'Ed Wood'),
    ('Q1-Scott/Ed', 'Ed Wood nationality'),
    ('Q1-Scott/Ed', 'Ed Wood American filmmaker'),
    ('Q2-KissAndTell', 'Kiss and Tell 1945 film'),
    ('Q2-KissAndTell', 'Kiss and Tell Corliss Archer'),
    ('Q2-KissAndTell', 'Corliss Archer'),
    ('Q2-KissAndTell', 'Shirley Temple'),
    ('Q2-KissAndTell', 'Shirley Temple government position'),
    ('Q2-KissAndTell', 'Shirley Temple ambassador diplomat'),
    ('Q2-KissAndTell', 'Shirley Temple Chief of Protocol'),
]

print("=== RETRIEVAL TEST RESULTS ===\n")
for label, q in test_queries:
    results, scores = retriever.retrieve(q, top_k=3)
    titles = [r['title'] for r in results]
    # Check if target is in top 3
    target_found = any(
        ('Scott Derrickson' in t or 'Ed Wood' == t or 'Kiss and Tell' in t or 'Shirley Temple' == t)
        for t in titles
    )
    marker = "✓" if target_found else "✗"
    print(f'{marker} {label} | "{q}"')
    print(f'   Top 3: {titles}')
    print()
