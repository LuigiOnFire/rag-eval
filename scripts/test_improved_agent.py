#!/usr/bin/env python3
"""Test the improved structured agent on previously failing questions."""
import sys
sys.path.insert(0, 'src')
import requests

from structured_agent import StructuredAgent
from retriever import BM25Retriever
from generator import OllamaGenerator


class LLMWrapper:
    """Wrap OllamaGenerator for agent interface."""
    def __init__(self, gen: OllamaGenerator):
        self.gen = gen
        
    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.gen.base_url}/api/chat",
            json={
                "model": self.gen.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 400
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


class RetrieverWrapper:
    """Wrap BM25Retriever for agent interface."""
    def __init__(self, ret: BM25Retriever):
        self.ret = ret
        
    def search(self, query: str, top_k: int = 5):
        passages, scores = self.ret.retrieve(query, top_k=top_k)
        return [{'title': p.get('title', ''), 'text': p.get('text', '')} for p in passages]

# Test cases that previously failed
test_cases = [
    {
        'question': 'What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?',
        'expected': 'Animorphs',
        'type': 'bridge'
    },
    {
        'question': 'Were Scott Derrickson and Ed Wood of the same nationality?',
        'expected': 'yes',
        'type': 'comparison'
    },
    {
        'question': 'What government post did Shirley Temple hold after becoming Chief of Protocol of the United States?',
        'expected': 'United States Ambassador to Ghana',
        'type': 'bridge'
    }
]

def test_agent():
    """Test agent on failing cases."""
    # Load passages and build retriever
    import json
    with open('data/processed/passages.json') as f:
        passages = json.load(f)
    
    retriever = BM25Retriever()
    retriever.build_index(passages)
    
    llm = OllamaGenerator(model_name='mistral:latest')
    
    agent = StructuredAgent(
        llm=LLMWrapper(llm),
        retriever=RetrieverWrapper(retriever),
        max_docs_per_query=5
    )
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_cases)} ({test['type']})")
        print(f"{'='*70}")
        print(f"Q: {test['question'][:80]}...")
        print(f"Expected: {test['expected']}")
        print()
        
        try:
            result = agent.run(test['question'])
            answer = result['answer']
            
            # Simple match check (case-insensitive)
            is_correct = test['expected'].lower() in answer.lower()
            
            print(f"\nAnswer: {answer}")
            print(f"Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            
            # Show key steps
            print(f"\nKey steps:")
            for step in agent.steps[-5:]:  # Last 5 steps
                print(f"  - {step.action}: {step.output[:60]}...")
            
            results.append({
                'question': test['question'],
                'expected': test['expected'],
                'got': answer,
                'correct': is_correct,
                'type': test['type']
            })
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            results.append({
                'question': test['question'],
                'expected': test['expected'],
                'got': f'ERROR: {e}',
                'correct': False,
                'type': test['type']
            })
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    correct = sum(1 for r in results if r['correct'])
    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    # By type
    by_type = {}
    for r in results:
        t = r['type']
        if t not in by_type:
            by_type[t] = {'correct': 0, 'total': 0}
        by_type[t]['total'] += 1
        if r['correct']:
            by_type[t]['correct'] += 1
    
    for t, stats in by_type.items():
        print(f"{t}: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.1f}%)")
    
    return results

if __name__ == '__main__':
    test_agent()
