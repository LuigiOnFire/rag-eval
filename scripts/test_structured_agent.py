#!/usr/bin/env python3
"""
Test the structured agent on our two problem questions.
"""
import sys
sys.path.insert(0, '/home/wcrawford/rag_eval')

import json
import logging
from src.structured_agent import StructuredAgent
from src.retriever import BM25Retriever
from src.generator import OllamaGenerator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class LLMWrapper:
    """Wrap OllamaGenerator for the agent interface."""
    def __init__(self, gen: OllamaGenerator):
        self.gen = gen
        
    def generate(self, prompt: str) -> str:
        import requests
        response = requests.post(
            f"{self.gen.base_url}/api/chat",
            json={
                "model": self.gen.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


class RetrieverWrapper:
    """Wrap BM25Retriever for the agent interface."""
    def __init__(self, ret: BM25Retriever):
        self.ret = ret
        
    def search(self, query: str, top_k: int = 5):
        passages, scores = self.ret.retrieve(query, top_k=top_k)
        return [{'title': p.get('title', ''), 'text': p.get('text', '')} for p in passages]


def test_questions():
    # Load passages and build retriever
    print("Loading passages...")
    with open('/home/wcrawford/rag_eval/data/processed/passages.json') as f:
        passages = json.load(f)
    print(f"Loaded {len(passages)} passages")
    
    print("Building BM25 index...")
    retriever = BM25Retriever()
    retriever.build_index(passages)
    
    # Load LLM
    print("Loading LLM...")
    llm = OllamaGenerator(model_name="mistral:latest")
    
    # Create agent
    agent = StructuredAgent(
        llm=LLMWrapper(llm),
        retriever=RetrieverWrapper(retriever),
        max_docs_per_query=5
    )
    
    # Load actual HotPotQA validation samples
    from datasets import load_dataset
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    test_cases = []
    for i, item in enumerate(dataset):
        if i >= 10:  # Test on 10 samples
            break
        test_cases.append({
            "question": item['question'],
            "ground_truth": item['answer'],
            "type": item['type'],
            "supporting_facts": item['supporting_facts']
        })
    
    results = []
    
    for i, tc in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {tc['type'].upper()}")
        print(f"Question: {tc['question'][:80]}...")
        print(f"Expected: {tc['ground_truth']}")
        print('='*60)
        
        try:
            result = agent.run(tc['question'])
            
            # Simple correctness check
            gt_lower = tc['ground_truth'].lower().strip()
            ans_lower = result.final_answer.lower().strip()
            is_correct = gt_lower in ans_lower or ans_lower in gt_lower
            
            results.append({
                'question': tc['question'],
                'type': tc['type'],
                'ground_truth': tc['ground_truth'],
                'answer': result.final_answer,
                'correct': is_correct,
                'steps': len(result.steps)
            })
            
            print(f"Final Answer: {result.final_answer}")
            print(f"Correct: {'✓' if is_correct else '✗'}")
            print(f"Steps: {len(result.steps)}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'question': tc['question'],
                'type': tc['type'], 
                'ground_truth': tc['ground_truth'],
                'answer': f"ERROR: {e}",
                'correct': False,
                'steps': 0
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    comparison_correct = sum(1 for r in results if r['type'] == 'comparison' and r['correct'])
    comparison_total = sum(1 for r in results if r['type'] == 'comparison')
    bridge_correct = sum(1 for r in results if r['type'] == 'bridge' and r['correct'])
    bridge_total = sum(1 for r in results if r['type'] == 'bridge')
    
    print(f"Overall: {correct}/{total} ({correct/total*100:.1f}%)")
    if comparison_total > 0:
        print(f"Comparison: {comparison_correct}/{comparison_total} ({comparison_correct/comparison_total*100:.1f}%)")
    if bridge_total > 0:
        print(f"Bridge: {bridge_correct}/{bridge_total} ({bridge_correct/bridge_total*100:.1f}%)")
    
    avg_steps = sum(r['steps'] for r in results) / len(results) if results else 0
    print(f"Avg steps: {avg_steps:.1f}")
    
    print(f"\nErrors:")
    for i, r in enumerate(results):
        if not r['correct']:
            print(f"  {i+1}. {r['question'][:50]}...")
            print(f"     Expected: {r['ground_truth']}")
            print(f"     Got: {r['answer'][:80]}...")
            print()


if __name__ == "__main__":
    test_questions()
