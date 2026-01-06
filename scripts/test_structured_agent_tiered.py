#!/usr/bin/env python3
"""
Test the structured agent with tiered retrieval on known failure cases.

This tests whether the FAST -> SMART -> DEEP escalation in the agent
can help it find better answers for the cases it previously failed on.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from typing import List, Dict
from structured_agent import StructuredAgent  
from retriever import BM25Retriever
from generator import OllamaGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleLLMWrapper:
    """Simple wrapper to make OllamaGenerator work with StructuredAgent."""
    def __init__(self):
        self.ollama = OllamaGenerator()
    
    def generate(self, prompt: str) -> str:
        """Generate response from prompt only."""
        result = self.ollama.generate(prompt, context_passages=[])
        return result["answer"] if isinstance(result, dict) else result


class BM25RetrieverWrapper:
    """Wrapper to make BM25Retriever work with StructuredAgent."""
    def __init__(self, passages):
        self.retriever = BM25Retriever()
        self.retriever.build_index(passages)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search method expected by StructuredAgent."""
        passages, scores = self.retriever.retrieve(query, top_k=top_k)
        return passages


def main():
    """Test structured agent on known failing cases."""
    
    # Load passages
    with open('data/processed/passages.json', 'r') as f:
        passages = json.load(f)
    
    logger.info(f"Loaded {len(passages)} passages")
    
    # Initialize components
    retriever = BM25RetrieverWrapper(passages)
    llm = SimpleLLMWrapper()
    
    # Create structured agent
    agent = StructuredAgent(
        llm=llm,
        retriever=retriever,
        max_docs_per_query=10
    )
    
    # Test cases that failed before
    test_cases = [
        {
            "query": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "expected": "Chief of Protocol", 
            "case": "Shirley Temple"
        },
        {
            "query": "What science fantasy young adult series, told in first person, features an unknown protagonist?", 
            "expected": "Animorphs",
            "case": "Animorphs"
        },
        {
            "query": "Who was known by stage name Aladin and helped organizations like Disney improve performance?",
            "expected": "Eenasul Fateh",
            "case": "Eenasul Fateh" 
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n{'='*80}")
        print(f"TESTING: {case['case']}")
        print(f"Query: {case['query']}")
        print(f"Expected: {case['expected']}")
        print(f"{'='*80}")
        
        try:
            # Run the structured agent
            result = agent.run(case['query'])
            
            answer = result.final_answer
            steps = result.steps
            sub_queries = result.sub_queries
            
            print(f"\\n🤖 AGENT ANSWER:")
            print(f"\"{answer}\"")
            
            # Check if we got the expected answer
            expected_found = case['expected'].lower() in answer.lower()
            
            print(f"\\n📊 RESULT:")
            if expected_found:
                print(f"✅ SUCCESS: Found expected answer '{case['expected']}'")
                status = "SUCCESS"
            else:
                print(f"❌ FAILURE: Expected '{case['expected']}', got different answer")
                status = "FAILURE"
            
            # Show the agent's reasoning steps
            print(f"\\n🔍 AGENT STEPS:")
            for i, step in enumerate(steps, 1):
                action = step.action if hasattr(step, 'action') else str(step)[:50]
                print(f"  {i}. {action}")
                
            print(f"\\n📝 SUB-QUERIES:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq.question} -> {sq.answer or 'No answer'}")
            
            results.append({
                "case": case['case'],
                "query": case['query'], 
                "expected": case['expected'],
                "answer": answer,
                "status": status,
                "steps": len(steps),
                "sub_queries": len(sub_queries),
                "success": result.success
            })
            
        except Exception as e:
            print(f"\\n❌ ERROR: {e}")
            results.append({
                "case": case['case'],
                "query": case['query'],
                "expected": case['expected'], 
                "answer": "",
                "status": "ERROR",
                "error": str(e),
                "steps": 0
            })
    
    # Summary
    print(f"\\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successes = sum(1 for r in results if r['status'] == 'SUCCESS')
    total = len(results)
    
    print(f"Success Rate: {successes}/{total} ({100*successes/total:.1f}%)")
    
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌" 
        print(f"{status_icon} {result['case']}: {result['status']}")
    
    # Save detailed results
    output_file = "results/tiered_agent_test.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "test_type": "tiered_agent_test", 
            "success_rate": successes/total,
            "results": results
        }, f, indent=2)
    
    print(f"\\nDetailed results saved to {output_file}")
    
    return 0 if successes == total else 1


if __name__ == "__main__":
    sys.exit(main())