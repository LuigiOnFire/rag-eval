#!/usr/bin/env python3
"""
Small Test Run for Enhanced Debug Trajectory Generation
"""
import json
import os
from datetime import datetime
import sys

# Add path for Oracle imports
sys.path.append('/home/wcrawford/rag_eval')
sys.path.append('/home/wcrawford/rag_eval/src')

from tests.integrated_oracle_debug import run_integrated_debug_test

def run_small_test():
    """Run a small test with just 3 questions"""
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/home/wcrawford/rag_eval/results/debug_runs/small_test_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Load questions - just take the first 3
    questions_file = "/home/wcrawford/rag_eval/data/processed/questions.json"
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    # Small test with 3 questions
    questions = all_questions[:3]
    
    print(f"🔬 Small Debug Test Run")
    print(f"📁 Run directory: {run_dir}")
    print(f"📊 Testing {len(questions)} questions")
    print("=" * 40)
    
    # Process each question individually to show progress
    for i, q_data in enumerate(questions):
        print(f"\n📝 Question {i+1}/{len(questions)}: {q_data['question'][:50]}...")
        print(f"🎯 Expected: {q_data['answer']}")
        
        # Here we would run the integrated debug test
        # For now, let's show what questions we're testing
    
    # Show the questions we'll process
    print(f"\n📋 Questions selected for small test:")
    for i, q_data in enumerate(questions):
        print(f"  {i+1}. {q_data['question'][:70]}...")
        print(f"      Expected: {q_data['answer']}")
        print(f"      Type: {q_data.get('type', 'unknown')}")
    
    print(f"\n🚀 Ready to run detailed debug analysis!")
    return run_dir, questions

if __name__ == "__main__":
    run_dir, questions = run_small_test()
    
    # Ask if user wants to proceed with full debug run
    print(f"\n❓ Proceed with full integrated Oracle debug test on these {len(questions)} questions? (y/n)")
    
    # For now, let's just run it automatically
    print("🔄 Running integrated debug test...")
    
    # This will run the full integrated test with detailed debug tracking
    from tests.integrated_oracle_debug import IntegratedDebugOracle, DebugTracker, DebugMockGenerator, DebugMockAgent
    from exploration_oracle import CostCalculator
    
    # Create components
    results = []
    
    for i, q_data in enumerate(questions):
        query_id = f"{i:03d}"
        question = q_data["question"] 
        expected = q_data["answer"]
        
        print(f"\n📝 Processing Query {query_id}: {question[:60]}...")
        
        # Create debug tracker
        debug_tracker = DebugTracker(run_dir, query_id)
        
        # Create enhanced components
        generator = DebugMockGenerator(debug_tracker)
        agent = DebugMockAgent(generator, debug_tracker)
        
        # Simple judge
        class SimpleJudge:
            def evaluate(self, generated: str, expected: str) -> bool:
                return expected.lower() in generated.lower() or generated.lower() in expected.lower()
        
        judge = SimpleJudge()
        cost_calculator = CostCalculator()
        
        # Create Oracle with debug tracking
        oracle = IntegratedDebugOracle(agent, judge, cost_calculator, debug_tracker)
        
        try:
            result = oracle.find_gold_trajectory(question, expected)
            
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
            print(f"💾 Debug saved: query_{query_id}_debug.json")
            
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
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "run_directory": run_dir,
        "questions_processed": len(results),
        "success_count": success_count,
        "success_rate": f"{success_count / len(results) * 100:.1f}%",
        "results": results
    }
    
    with open(os.path.join(run_dir, "small_test_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Small test complete!")
    print(f"📊 Results: {success_count}/{len(results)} successful ({summary['success_rate']})")
    print(f"📁 Debug files: {run_dir}")
    print(f"🔍 Check individual query debug files for detailed state tracking!")