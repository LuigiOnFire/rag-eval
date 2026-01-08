#!/usr/bin/env python3
"""
Integrated Oracle Debug Runner

This integrates the detailed debug tracking with the actual 7-tier Oracle execution.
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

class IntegratedDebugOracle(ExplorationOracle):
    """Oracle with integrated debug tracking"""
    
    def __init__(self, agent, judge, cost_calculator, debug_tracker: DebugTracker = None):
        super().__init__(agent, judge, cost_calculator)
        self.debug_tracker = debug_tracker
    
    def find_gold_trajectory(self, query: str, ground_truth: str) -> Optional[Dict[str, Any]]:
        """Enhanced trajectory finding with debug tracking"""
        
        if self.debug_tracker:
            self.debug_tracker.log_step("ORACLE_START", query, "Starting 7-tier exploration")
        
        candidates = []
        
        # === PHASE 1: CHEAP & FAST ===
        
        # Tier 0: Pure SLM
        if self.debug_tracker:
            self.debug_tracker.log_tier_attempt(0, "PURE_SLM", "No retrieval, pure parametric knowledge")
        print("🔍 Tier 0: Pure SLM (no retrieval, cost ~$1)")
        result = self._try_pure_slm(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 0 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 0 failed: {result.error}")
            
        # Continue with other tiers...
        # (I'll implement the full integration step by step)
        
        # For now, let's focus on one tier to show the concept
        return super().find_gold_trajectory(query, ground_truth)

def run_integrated_debug_test():
    """Run Oracle with integrated debug tracking"""
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/home/wcrawford/rag_eval/results/debug_runs/integrated_oracle_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Load questions
    questions_file = "/home/wcrawford/rag_eval/data/processed/questions.json"
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    # Test with 2 questions
    questions = all_questions[:2]
    
    print(f"🎯 Integrated Oracle Debug Test")
    print(f"📁 Run directory: {run_dir}")
    print(f"📊 Processing {len(questions)} questions")
    print("=" * 50)
    
    results = []
    
    for i, q_data in enumerate(questions):
        query_id = f"{i:03d}"
        question = q_data["question"] 
        expected = q_data["answer"]
        
        print(f"\n📝 Query {query_id}: {question[:60]}...")
        print(f"🎯 Expected: {expected}")
        
        # Create debug tracker for this query
        debug_tracker = DebugTracker(run_dir, query_id)
        
        # Create enhanced components with debug tracking
        generator = DebugMockGenerator(debug_tracker)
        agent = DebugMockAgent(generator, debug_tracker)
        
        # Simple judge for testing
        class SimpleJudge:
            def evaluate(self, generated: str, expected: str) -> bool:
                return expected.lower() in generated.lower() or generated.lower() in expected.lower()
        
        judge = SimpleJudge()
        cost_calculator = CostCalculator()
        
        # Create integrated Oracle with debug tracking
        oracle = IntegratedDebugOracle(agent, judge, cost_calculator, debug_tracker)
        
        try:
            # Run Oracle with debug tracking
            result = oracle.find_gold_trajectory(question, expected)
            
            if result:
                print(f"✅ SUCCESS: {result['gold_action']} (${result['gold_cost']:.3f})")
                success = True
                tier = result['gold_action']
                cost = result['gold_cost']
            else:
                print("❌ FAILED: All tiers unsuccessful")
                success = False
                tier = None
                cost = float('inf')
            
            # Save debug information
            debug_tracker.save_debug_log()
            
            results.append({
                "query_id": query_id,
                "question": question,
                "expected": expected,
                "success": success,
                "tier": tier,
                "cost": cost,
                "debug_file": f"query_{query_id}_debug.json"
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            debug_tracker.log_step("ERROR", str(e), "Oracle execution failed")
            debug_tracker.save_debug_log()
    
    # Save run summary
    summary = {
        "timestamp": timestamp,
        "run_directory": run_dir,
        "questions_processed": len(results),
        "success_rate": f"{len([r for r in results if r['success']]) / len(results) * 100:.1f}%",
        "results": results
    }
    
    with open(os.path.join(run_dir, "integrated_run_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Integrated debug test complete!")
    print(f"📁 Results: {run_dir}")
    print(f"📊 Success rate: {summary['success_rate']}")
    
    return run_dir

if __name__ == "__main__":
    run_dir = run_integrated_debug_test()
    print(f"\n🔍 Check detailed debug logs in: {run_dir}")
    print("📋 Each query shows step-by-step Oracle reasoning with state evolution!")