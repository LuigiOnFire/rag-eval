#!/usr/bin/env python
"""Test Ed Wood / Scott Derrickson query with BM25."""

from src.pipeline import load_pipeline_from_config

print("Loading pipeline...")
pipeline = load_pipeline_from_config('config_local.yaml')
print(f"Retriever: {type(pipeline.retriever).__name__}")
print()

query = 'Were Scott Derrickson and Ed Wood of the same nationality?'
print(f"Query: {query}")
print()

# Get retrieved passages
passages, scores = pipeline.retriever.retrieve(query, top_k=5)
print("Retrieved passages:")
for i, (p, s) in enumerate(zip(passages, scores), 1):
    print(f"  {i}. [{s:.2f}] {p['title']}: {p['text'][:80]}...")
print()

# Get answer
print("Generating answer...")
result = pipeline.run(query, verbose=False)
print(f"\nAnswer: {result['answer']}")
print()
print("Ground truth: Yes (both American)")
