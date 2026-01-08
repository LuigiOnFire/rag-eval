#!/usr/bin/env python3
"""
Optimized Oracle that stops after first successful trajectory
"""
import json
import os
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# Add path for Oracle imports
sys.path.append('/home/wcrawford/rag_eval')
sys.path.append('/home/wcrawford/rag_eval/src')

from tests.enhanced_debug_generator import DebugTracker, DebugMockGenerator, DebugMockAgent
from exploration_oracle import ExplorationOracle, CostCalculator

class OptimizedOracle(ExplorationOracle):
    """Oracle that stops after first successful trajectory"""
    
    def find_gold_trajectory(self, query: str, ground_truth: str, stop_on_first_success: bool = True) -> Optional[Dict[str, Any]]:
        """
        Explores tiers until first success (for training) or all tiers (for comparison).
        
        Args:
            query: The question to answer
            ground_truth: Expected answer
            stop_on_first_success: If True, stop after first successful tier
        """
        
        if self.debug_tracker:
            self.debug_tracker.log_step("ORACLE_START", query, f"Starting exploration (stop_on_first: {stop_on_first_success})")
        
        candidates = []
        
        # Define tiers to try in order
        tiers = [
            (0, "Pure SLM", self._try_pure_slm),
            (1, "Fast Retrieval", self._try_fast_retrieval), 
            (2, "Smart Retrieval", self._try_smart_retrieval),
            (3, "Refine Query", self._try_refine_query),
            (4, "Filter Results", self._try_filter_results),
            (5, "Iterative Chain", self._try_iterative_chain),
            (6, "Full Decomposition", self._try_full_decomposition)
        ]
        
        for tier_num, tier_name, tier_func in tiers:
            if self.debug_tracker:
                self.debug_tracker.log_tier_attempt(tier_num, tier_name.upper().replace(" ", "_"), 
                                                  f"Trying {tier_name} for: {query[:50]}...")
            
            print(f"🔍 Tier {tier_num}: {tier_name} ({'~$1' if tier_num == 0 else '~$' + str((tier_num + 1) * 5)})")
            
            result = tier_func(query, ground_truth)
            
            if result.success:
                candidates.append(result)
                print(f"✅ Tier {tier_num} SUCCESS: {result.answer[:100]}")
                
                if stop_on_first_success:
                    print(f"🎯 STOPPING: First successful tier found (Tier {tier_num})")
                    break
            else:
                print(f"❌ Tier {tier_num} failed: {result.error}")
        
        if not candidates:
            print("💀 All tiers failed - no training data generated")
            return None
            
        # Return the first (cheapest) successful candidate
        winner = candidates[0]
        
        print(f"🏆 WINNER: {winner.action} (cost: {winner.cost:.3f})")
        
        # Convert to training format
        return {
            "question": query,
            "ground_truth": ground_truth, 
            "gold_action": winner.action,
            "gold_cost": winner.cost,
            "gold_answer": winner.answer,
            "gold_trace": winner.trace,
            "alternatives": [
                {
                    "action": c.action,
                    "cost": c.cost,
                    "success": c.success
                } for c in candidates
            ]
        }

def test_optimized_oracle():
    """Test the optimized Oracle that stops on first success"""
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/home/wcrawford/rag_eval/results/debug_runs/optimized_test_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Load questions
    questions_file = "/home/wcrawford/rag_eval/data/processed/questions.json"
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    # Test with 3 questions
    questions = all_questions[:3]
    
    print(f"⚡ Optimized Oracle Test (Stop on First Success)")
    print(f"📁 Run directory: {run_dir}")
    print(f"📊 Testing {len(questions)} questions")
    print("=" * 55)
    
    results = []
    
    for i, q_data in enumerate(questions):
        query_id = f"{i:03d}"
        question = q_data["question"] 
        expected = q_data["answer"]
        
        print(f"\n📝 Query {query_id}: {question[:60]}...")
        print(f"🎯 Expected: {expected}")
        
        # Create debug tracker
        debug_tracker = DebugTracker(run_dir, query_id)
        
        # Create components
        generator = DebugMockGenerator(debug_tracker)
        agent = DebugMockAgent(generator, debug_tracker)
        
        class SimpleJudge:
            def evaluate(self, generated: str, expected: str) -> bool:
                return expected.lower() in generated.lower() or generated.lower() in expected.lower()
        
        judge = SimpleJudge()
        cost_calculator = CostCalculator()
        
        # Create optimized Oracle
        oracle = OptimizedOracle(agent, judge, cost_calculator)
        oracle.debug_tracker = debug_tracker
        
        try:
            # Test with stop_on_first_success=True
            result = oracle.find_gold_trajectory(question, expected, stop_on_first_success=True)
            
            if result:
                print(f"✅ SUCCESS: {result['gold_action']} (${result['gold_cost']:.3f})")
                tier = result['gold_action']
                cost = result['gold_cost']
                success = True
            else:
                print("❌ FAILED: All tiers unsuccessful")
                tier = None
                cost = float('inf')
                success = False
            
            # Save debug info
            debug_tracker.save_debug_log()
            
            results.append({
                "query_id": query_id,
                "question": question,
                "expected": expected,
                "success": success,
                "tier": tier,
                "cost": cost
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            debug_tracker.log_step("ERROR", str(e), "Failed")
            debug_tracker.save_debug_log()
    
    # Save summary
    success_count = len([r for r in results if r['success']])
    summary = {
        "timestamp": timestamp,
        "run_type": "optimized_stop_on_first_success",
        "run_directory": run_dir,
        "questions_processed": len(results),
        "success_count": success_count,
        "success_rate": f"{success_count / len(results) * 100:.1f}%",
        "results": results
    }
    
    with open(os.path.join(run_dir, "optimized_test_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Optimized test complete!")
    print(f"📊 Results: {success_count}/{len(results)} successful ({summary['success_rate']})")
    print(f"📁 Debug files: {run_dir}")
    print(f"⚡ Should show fewer tier attempts per query (stopping on first success)")
    
    return run_dir

if __name__ == "__main__":
    run_dir = test_optimized_oracle()
    print(f"\n🔍 Compare with previous runs to see the efficiency improvement!")