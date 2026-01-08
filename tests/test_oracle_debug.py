#!/usr/bin/env python3
"""
Test the 7-tier exploration Oracle with a minimal setup.
This version uses existing indexes and data to avoid setup complexity.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

sys.path.append('/home/wcrawford/rag_eval')
sys.path.append('/home/wcrawford/rag_eval/src')

# Simplified test without full component initialization
class MockAgent:
    """Mock agent for testing Oracle logic."""
    def __init__(self):
        self.llm = MockLLM()
        
    def _retrieve_tier(self, query: str, tier: str) -> List[Dict]:
        """Mock retrieval that returns dummy documents."""
        # Simulate different tier results
        if tier == "FAST":
            return [{"title": f"BM25 result for: {query[:30]}", "text": f"This is a BM25 search result about {query[:50]}..."}]
        elif tier == "SMART":
            return [{"title": f"Dense result for: {query[:30]}", "text": f"This is a dense/embedding search result about {query[:50]}..."}]
        else:
            return [{"title": f"Hybrid result for: {query[:30]}", "text": f"This is a hybrid search result about {query[:50]}..."}]
    
    def run_full_trajectory(self, query: str, force_decompose: bool = False):
        """Mock full trajectory execution."""
        class MockResult:
            final_answer = f"Full decomposition answer for: {query[:50]}..."
            steps = [
                type('obj', (object,), {'action': 'DECOMPOSE', 'input': query, 'output': 'Sub-queries generated'})(),
                type('obj', (object,), {'action': 'RETRIEVE_SMART', 'input': 'sub-query', 'output': 'Retrieved docs'})(),
                type('obj', (object,), {'action': 'GENERATE_FINAL', 'input': query, 'output': f"Answer: {query[:20]}..."})()
            ]
        return MockResult()

class MockLLM:
    """Mock LLM for testing."""
    def generate(self, prompt: str) -> str:
        # Return different responses based on prompt content
        if "rewrite" in prompt.lower() or "optimal for search" in prompt.lower():
            return "Rewritten query with expanded terms and clearer entities"
        elif "select the 3 most relevant" in prompt.lower():
            return "1, 2, 3"
        elif "step 1" in prompt.lower():
            return "What is the first intermediate fact needed?"
        elif "step 2" in prompt.lower():
            return "What is the final answer using the intermediate fact?"
        elif "what is 2" in prompt.lower():
            return "4"
        else:
            return f"Generated response for prompt about: {prompt[:50]}..."

class MockJudge:
    """Mock evaluator with realistic success/failure patterns."""
    def evaluate(self, answer: str, ground_truth: str) -> bool:
        # More realistic evaluation that forces tier progression
        query_type = self._classify_answer_complexity(answer, ground_truth)
        
        # Tier 0 (SLM) should only succeed on very simple questions
        if "Question: What is 2" in answer and "4" in ground_truth:
            return True
        elif "Question:" in answer and len(answer) < 100:
            return False  # Most questions need retrieval
            
        # Tier 1/2 (BM25/Dense) - succeed on entity/concept questions
        if ("scott derrickson" in answer.lower() or "nationality" in answer.lower()) and "american" in ground_truth.lower():
            return True
        elif ("science fantasy" in answer.lower() or "animorphs" in answer.lower()) and "animorphs" in ground_truth.lower():
            return True
            
        # Higher tiers - succeed on complex questions
        if "rewritten query" in answer.lower():
            return True  # Refined queries often work
        elif "filter from" in answer:
            return True  # Filtered results often work
        elif "intermediate fact" in answer.lower():
            return True  # Iterative chains often work
        elif "full decomposition" in answer.lower():
            return True  # Full decomposition usually works
            
        # Default: fail simple mock responses
        return "Generated response for prompt" not in answer
    
    def _classify_answer_complexity(self, answer: str, ground_truth: str) -> str:
        if "2 + 2" in answer or "4" == ground_truth:
            return "simple"
        elif any(term in ground_truth.lower() for term in ["american", "animorphs", "total recall"]):
            return "factual"
        else:
            return "complex"

def test_exploration_oracle():
    """Test the 7-tier Oracle with mock components."""
    
    # Import our Oracle (this should work now with mocks)
    from src.exploration_oracle import ExplorationOracle, CostCalculator
    
    # Initialize with mock components
    mock_agent = MockAgent()
    mock_judge = MockJudge()
    cost_calculator = CostCalculator()
    
    oracle = ExplorationOracle(
        agent=mock_agent,
        judge=mock_judge, 
        cost_calculator=cost_calculator
    )
    
    # Test cases representing different complexity levels
    test_cases = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "expected_tier": "<ASSIGN_SLM>"
        },
        {
            "question": "What is Scott Derrickson's nationality?", 
            "answer": "American",
            "expected_tier": "<RETRIEVE_FAST>"
        },
        {
            "question": "What science fantasy young adult series features enslaved worlds?",
            "answer": "Animorphs", 
            "expected_tier": "<RETRIEVE_SMART>"
        },
        {
            "question": "The actor who played Terminator starred in what other action films?",
            "answer": "Total Recall",
            "expected_tier": "<REFINE_QUERY>"
        },
        {
            "question": "Which of these directors worked on science fiction films in the 1990s?",
            "answer": "James Cameron",
            "expected_tier": "<RETRIEVE_FILTER>"
        }
    ]
    
    print("🚀 Testing 7-Tier Exploration Oracle")
    print("=" * 60)
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {case['question'][:60]}...")
        print(f"Expected Answer: {case['answer']}")
        print(f"Expected Tier: {case['expected_tier']}")
        
        # Run Oracle exploration
        try:
            gold_trajectory = oracle.find_gold_trajectory(
                query=case['question'],
                ground_truth=case['answer']
            )
            
            if gold_trajectory:
                actual_tier = gold_trajectory['gold_action']
                cost = gold_trajectory['gold_cost']
                answer = gold_trajectory['gold_answer']
                
                print(f"✅ SUCCESS: {actual_tier} (cost: ${cost:.2f})")
                print(f"Answer: {answer[:60]}...")
                
                results.append({
                    "question": case['question'],
                    "expected_tier": case['expected_tier'],
                    "actual_tier": actual_tier,
                    "cost": cost,
                    "success": True
                })
                
                # Check if Oracle picked expected tier or something cheaper
                tier_costs = {
                    "<ASSIGN_SLM>": 1, "<RETRIEVE_FAST>": 5, "<RETRIEVE_SMART>": 8,
                    "<REFINE_QUERY>": 12, "<RETRIEVE_FILTER>": 18, "<ITERATE>": 25, "<DECOMPOSE>": 50
                }
                
                expected_cost = tier_costs.get(case['expected_tier'], 100)
                actual_cost = tier_costs.get(actual_tier, 100)
                
                if actual_cost <= expected_cost:
                    print(f"🎯 Good choice: Found cheaper/equal solution!")
                else:
                    print(f"⚠️  More expensive than expected, but still valid")
                    
            else:
                print(f"❌ FAILED: All tiers failed")
                results.append({
                    "question": case['question'],
                    "expected_tier": case['expected_tier'], 
                    "actual_tier": None,
                    "cost": float('inf'),
                    "success": False
                })
                
        except Exception as e:
            print(f"💥 ERROR: {str(e)}")
            results.append({
                "question": case['question'],
                "expected_tier": case['expected_tier'],
                "actual_tier": None, 
                "cost": float('inf'),
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful = [r for r in results if r['success']]
    print(f"\n📊 SUMMARY")
    print(f"Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    if successful:
        print(f"Average Cost: ${sum(r['cost'] for r in successful)/len(successful):.2f}")
        
        tier_distribution = {}
        for r in successful:
            tier = r['actual_tier']
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            
        print("Tier Distribution:")
        for tier, count in sorted(tier_distribution.items()):
            print(f"  {tier}: {count} samples")
    
    print("\n🎉 7-Tier Oracle test complete!")
    return results

if __name__ == "__main__":
    test_exploration_oracle()