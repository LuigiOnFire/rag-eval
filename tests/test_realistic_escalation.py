#!/usr/bin/env python3
"""
Test Oracle with realistic failure patterns to see tier escalation
"""
import json
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Mock components with realistic failure patterns
class RealisticMockGenerator:
    def __init__(self):
        # Questions with their tier capability 
        self.question_tiers = {
            "Were Scott Derrickson and Ed Wood": 2,  # Needs Smart retrieval
            "What government position was held": 1,  # Fast retrieval works
            "What science fantasy young adult": 5,   # Needs iterative reasoning
            "Are the Laleli Mosque and Esma": 3,     # Needs query refinement  
            'The director of the romantic comedy "Big Stone Gap"': 6  # Needs decomposition
        }
        
        self.correct_answers = {
            "Were Scott Derrickson and Ed Wood": "yes",
            "What government position was held": "Chief of Protocol", 
            "What science fantasy young adult": "Animorphs",
            "Are the Laleli Mosque and Esma": "no",
            'The director of the romantic comedy "Big Stone Gap"': "Greenwich Village, New York City"
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate answers based on which tier is calling (detected from call stack)"""
        import inspect
        
        # Detect tier from call stack
        tier = 0  # Default
        frame = inspect.currentframe()
        try:
            for i in range(10):  # Look up the call stack
                frame = frame.f_back
                if frame is None:
                    break
                
                func_name = frame.f_code.co_name
                if "_try_pure_slm" in func_name:
                    tier = 0
                    break
                elif "_try_fast_retrieval" in func_name:
                    tier = 1
                    break
                elif "_try_smart_retrieval" in func_name:
                    tier = 2
                    break
                elif "_try_refine_query" in func_name:
                    tier = 3
                    break
                elif "_try_filter_results" in func_name:
                    tier = 4
                    break
                elif "_try_iterative_chain" in func_name:
                    tier = 5
                    break
                elif "_try_full_decomposition" in func_name or "run_full_trajectory" in func_name:
                    tier = 6
                    break
        finally:
            del frame
        
        # Find question from prompt
        question_key = None
        required_tier = 99
        
        for key, tier_need in self.question_tiers.items():
            if key.lower() in prompt.lower():
                question_key = key
                required_tier = tier_need
                break
        
        if question_key and tier >= required_tier:
            # Return correct answer if tier is sufficient
            answer = self.correct_answers[question_key]
            if random.random() < 0.7:  # 70% chance of clean answer
                return answer
            else:
                return f"Based on analysis: {answer}"
        else:
            # Return wrong/incomplete answer if tier insufficient
            wrong_answers = ["I don't have enough information", "Unable to determine", "Context unclear", "Not specified"]
            return random.choice(wrong_answers)

class RealisticMockAgent:
    """Agent with realistic tier-based capabilities"""
    
    def __init__(self, generator):
        self.generator = generator
        self.llm = generator
    
    def _retrieve_tier(self, query: str, tier: str) -> List[Dict]:
        """Mock retrieval with tier-appropriate results"""
        if tier == "FAST":
            tier_num = 1
            retrieval_type = "BM25 keyword search"
        elif tier == "SMART":
            tier_num = 2
            retrieval_type = "Dense embedding search"
        else:
            tier_num = 3  # Hybrid
            retrieval_type = "Hybrid (BM25 + Dense)"
            
        # Generate realistic mock documents based on query content
        docs = []
        for i in range(3):  # Return 3 docs per tier
            if "Scott Derrickson" in query or "Ed Wood" in query:
                docs.append({
                    "title": f"Director Biography {i+1} ({retrieval_type})",
                    "text": f"Scott Derrickson is an American film director known for horror films. Ed Wood was also an American filmmaker, famous for his B-movies in the 1950s. Retrieved via {retrieval_type}."
                })
            elif "government position" in query or "Corliss Archer" in query:
                docs.append({
                    "title": f"Political Figures & Film History {i+1} ({retrieval_type})", 
                    "text": f"Kiss and Tell (1945) featured actress Shirley Temple. Many actresses later moved into government roles, including diplomatic positions like Chief of Protocol. Retrieved via {retrieval_type}."
                })
            elif "science fantasy" in query or "Animorphs" in query:
                docs.append({
                    "title": f"Young Adult Literature {i+1} ({retrieval_type})",
                    "text": f"Animorphs is a science fantasy series written by K.A. Applegate, told in first person narrative. The series features companion books about enslaved alien worlds. Retrieved via {retrieval_type}."
                })
            elif "Laleli Mosque" in query or "Esma Sultan Mansion" in query:
                docs.append({
                    "title": f"Istanbul Architecture {i+1} ({retrieval_type})",
                    "text": f"Laleli Mosque is located in the Laleli district of Istanbul. Esma Sultan Mansion is situated in Ortaköy, a different neighborhood on the Bosphorus. Retrieved via {retrieval_type}."
                })
            elif "Big Stone Gap" in query or "director" in query:
                docs.append({
                    "title": f"Film Directors & Locations {i+1} ({retrieval_type})",
                    "text": f"Big Stone Gap (2014) was directed by Adriana Trigiani. She is based in Greenwich Village, New York City, where many filmmakers reside. Retrieved via {retrieval_type}."
                })
            else:
                docs.append({
                    "title": f"Generic Document {i+1} ({retrieval_type})",
                    "text": f"This is a generic document about {query[:30]}... Retrieved using {retrieval_type} for tier {tier_num}."
                })
        
        # Log retrieved documents 
        print(f"        📋 Retrieved {len(docs)} docs via {retrieval_type}:")
        for i, doc in enumerate(docs):
            title_short = doc["title"][:60] + ("..." if len(doc["title"]) > 60 else "")
            text_short = doc["text"][:80] + ("..." if len(doc["text"]) > 80 else "")
            print(f"          {i+1}. {title_short}")
            print(f"             {text_short}")
        
        return docs
    
    def run_full_trajectory(self, query: str, force_decompose: bool = False):
        """Full decomposition with tier 6 capability"""
        class MockResult:
            def __init__(self, answer, query):
                self.final_answer = answer
                # Create realistic decomposition steps
                self.steps = [
                    type('obj', (object,), {
                        'action': 'DECOMPOSE', 
                        'input': query, 
                        'output': f'Sub-queries: 1) Extract key entities 2) Find relationships 3) Synthesize answer',
                        'retrieved_docs': []
                    })(),
                    type('obj', (object,), {
                        'action': 'RETRIEVE_SMART', 
                        'input': 'Entity extraction query', 
                        'output': 'Retrieved relevant documents',
                        'retrieved_docs': self._generate_decomp_docs(query, "entities")
                    })(),
                    type('obj', (object,), {
                        'action': 'RETRIEVE_SMART', 
                        'input': 'Relationship query', 
                        'output': 'Retrieved relationship documents', 
                        'retrieved_docs': self._generate_decomp_docs(query, "relationships")
                    })(),
                    type('obj', (object,), {
                        'action': 'GENERATE_FINAL', 
                        'input': query, 
                        'output': answer,
                        'retrieved_docs': []
                    })()
                ]
            
            def _generate_decomp_docs(self, query, doc_type):
                """Generate documents for decomposition steps"""
                docs = []
                for i in range(2):
                    if "Scott Derrickson" in query:
                        docs.append({
                            "title": f"Decomposition {doc_type} {i+1}",
                            "text": f"Scott Derrickson: American nationality. Ed Wood: American nationality. Both are American filmmakers from different eras."
                        })
                    elif "government position" in query:
                        docs.append({
                            "title": f"Decomposition {doc_type} {i+1}", 
                            "text": f"Actress from Kiss and Tell: Shirley Temple. Later government role: Chief of Protocol position in State Department."
                        })
                    # Add more decomposition docs for other questions...
                    else:
                        docs.append({
                            "title": f"Decomposition {doc_type} {i+1}",
                            "text": f"Decomposition document {i+1} for {doc_type} related to: {query[:50]}..."
                        })
                return docs
        
        # Use decomposition-level generation (tier 6)
        answer = self.generator.generate(f"Decomposition query: {query}")
        
        print(f"        🔬 Decomposition Steps:")
        print(f"          1. Entity extraction from query")
        print(f"          2. Multi-hop relationship discovery") 
        print(f"          3. Evidence synthesis")
        print(f"          4. Final answer generation")
        
        return MockResult(f"Full decomposition result: {answer}", query)

# Reuse the judge and other components
class SmartAnswerJudge:
    """Improved judge that can handle answer variations"""
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answers for comparison"""
        if not answer:
            return ""
        
        answer = answer.strip().lower()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "based on the context provided:",
            "based on analysis:",
            "full decomposition result:",
            "according to the information:",
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        answer = ' '.join(answer.split())
        return answer
    
    def is_correct(self, generated: str, expected: str) -> bool:
        """Check if generated answer matches expected"""
        if not generated or not expected:
            return False
        
        # Reject clearly wrong answers
        wrong_indicators = ["don't have enough", "unable to determine", "context unclear", "not specified"]
        if any(indicator in generated.lower() for indicator in wrong_indicators):
            return False
        
        gen_norm = self.normalize_answer(generated)
        exp_norm = self.normalize_answer(expected)
        
        # Exact match
        if gen_norm == exp_norm:
            return True
        
        # Check if expected answer is contained in generated
        if exp_norm in gen_norm:
            return True
        
        return False
    
    def evaluate(self, generated: str, expected: str) -> bool:
        """Interface for Oracle"""
        return self.is_correct(generated, expected)

# Import Oracle components
import sys
sys.path.append('/home/wcrawford/rag_eval')
sys.path.append('/home/wcrawford/rag_eval/src')
from exploration_oracle import ExplorationOracle, CostCalculator

def inspect_trajectory(result):
    """Display detailed trajectory information"""
    if not result or "gold_trace" not in result:
        print("    📋 No trajectory data available")
        return
    
    print("    📋 GOLD TRAJECTORY:")
    trace = result["gold_trace"]
    for i, step in enumerate(trace):
        print(f"      Step {i+1}: {step.get('action', 'Unknown')}")
        if 'input' in step:
            print(f"        Input: {step['input'][:80]}...")
        if 'retrieved_docs' in step:
            docs = step['retrieved_docs']
            print(f"        Retrieved {len(docs)} documents:")
            for j, doc in enumerate(docs[:2]):  # Show first 2 docs
                title = doc.get('title', 'No title')[:50]
                text = doc.get('text', 'No text')[:100]
                print(f"          Doc {j+1}: {title} -> {text}...")
        if 'output' in step:
            print(f"        Output: {step['output'][:80]}...")
    
    # Show alternatives that were tried
    if "alternatives" in result:
        print("    🔄 ALTERNATIVES TRIED:")
        for alt in result["alternatives"]:
            status = "✅" if alt.get("success", False) else "❌"
            print(f"      {status} {alt['action']}: ${alt['cost']:.3f}")

def test_realistic_oracle():
    """Test with realistic tier escalation"""
    
    print("🎯 Testing Oracle with Realistic Tier Escalation & Debug Logs")
    print("=" * 70)
    
    # Load real questions
    questions_file = "/home/wcrawford/rag_eval/data/processed/questions.json"
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    questions = all_questions[:5]  # First 5 for testing
    
    # Create realistic components
    generator = RealisticMockGenerator()
    agent = RealisticMockAgent(generator)
    judge = SmartAnswerJudge()
    cost_calculator = CostCalculator()
    
    oracle = ExplorationOracle(
        agent=agent,
        judge=judge,
        cost_calculator=cost_calculator
    )
    
    print(f"📚 Testing {len(questions)} questions with realistic failure patterns")
    print("🎲 Expected tier escalation based on question complexity\n")
    
    results = []
    
    for i, q_data in enumerate(questions):
        question = q_data["question"]
        expected = q_data["answer"]
        q_type = q_data.get("type", "unknown")
        
        print(f"📝 Question {i+1}: {question[:70]}...")
        print(f"🎯 Expected: {expected}")
        print(f"🏷️  Type: {q_type}")
        
        # Find expected tier for this question
        expected_tier = 99
        for key, tier_need in generator.question_tiers.items():
            if key in question:
                expected_tier = tier_need
                break
        print(f"🎚️  Expected Min Tier: {expected_tier}")
        
        try:
            result = oracle.find_gold_trajectory(question, expected)
            
            if result:
                tier = result["gold_action"]
                answer = result["gold_answer"]
                cost = result["gold_cost"]
                
                print(f"✅ SUCCESS: {tier} (${cost:.3f})")
                print(f"📋 Answer: {answer[:80]}...")
                
                # Show detailed trajectory
                inspect_trajectory(result)
                
                success = True
                
                # Store complete result with trajectory
                result_data = {
                    "question": question,
                    "expected": expected,
                    "got": answer,
                    "tier": tier,
                    "cost": cost,
                    "success": success,
                    "expected_min_tier": expected_tier,
                    "gold_trajectory": result.get("gold_trace", []),
                    "alternatives_tried": result.get("alternatives", []),
                    "complete_result": result  # Full Oracle result
                }
            else:
                print("❌ FAILED: All tiers unsuccessful")
                tier, answer, cost = None, None, float('inf')
                success = False
                
                result_data = {
                    "question": question,
                    "expected": expected,
                    "got": answer,
                    "tier": tier,
                    "cost": cost,
                    "success": success,
                    "expected_min_tier": expected_tier,
                    "gold_trajectory": [],
                    "alternatives_tried": [],
                    "complete_result": None
                }
            
            results.append(result_data)
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append({
                "question": question,
                "expected": expected, 
                "got": None,
                "tier": None,
                "cost": float('inf'),
                "success": False,
                "expected_min_tier": expected_tier
            })
        
        print("-" * 50)
    
    # Analyze results
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / len(results) * 100
    avg_cost = sum(r["cost"] for r in successes) / len(successes) if successes else 0
    
    # Tier distribution
    tier_counts = {}
    for result in successes:
        tier = result["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    print(f"\n📊 REALISTIC ESCALATION RESULTS")
    print("=" * 50)
    print(f"Success Rate: {len(successes)}/{len(results)} ({success_rate:.1f}%)")
    print(f"Average Cost: ${avg_cost:.3f}")
    print("Tier Distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} questions")
    
    print("\n🎚️  Tier Escalation Analysis:")
    for result in successes:
        q_short = result["question"][:50] + "..."
        expected_tier = result["expected_min_tier"] 
        actual_tier = result["tier"]
        num_alternatives = len(result.get("alternatives_tried", []))
        trajectory_steps = len(result.get("gold_trajectory", []))
        
        print(f"  {q_short}")
        print(f"    Expected: >= Tier {expected_tier}, Got: {actual_tier}")
        print(f"    Tried {num_alternatives} alternatives, {trajectory_steps} trajectory steps")
        
        # Show which tiers succeeded vs failed
        alternatives = result.get("alternatives_tried", [])
        if alternatives:
            succeeded = [alt['action'] for alt in alternatives if alt.get('success', False)]
            failed = [alt['action'] for alt in alternatives if not alt.get('success', False)]
            if failed:
                print(f"    ❌ Failed: {', '.join(failed[:3])}{'...' if len(failed) > 3 else ''}")
            if succeeded:
                print(f"    ✅ Succeeded: {', '.join(succeeded[:3])}{'...' if len(succeeded) > 3 else ''}")
    
    # Save results with enhanced trajectory data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/wcrawford/rag_eval/results/testing/test_realistic_escalation_{timestamp}.json"
    
    # Create training-ready format
    training_data = {
        "metadata": {
            "timestamp": timestamp,
            "oracle_type": "realistic_tier_escalation_with_trajectories", 
            "n_questions": len(questions),
            "success_rate": f"{success_rate:.1f}%",
            "avg_cost": f"${avg_cost:.3f}",
            "description": "Enhanced Oracle test with detailed trajectories and retrieved documents for training data generation"
        },
        "results": results,
        "training_examples": [
            {
                "id": i,
                "question": r["question"],
                "ground_truth": r["expected"],
                "optimal_strategy": r["tier"],
                "optimal_cost": r["cost"],
                "trajectory": r.get("gold_trajectory", []),
                "failed_attempts": [
                    alt for alt in r.get("alternatives_tried", []) 
                    if not alt.get("success", False)
                ],
                "successful_strategies": [
                    alt for alt in r.get("alternatives_tried", []) 
                    if alt.get("success", False)
                ],
                "cost_analysis": {
                    "cheapest_cost": r["cost"],
                    "most_expensive_alternative": max(
                        [alt["cost"] for alt in r.get("alternatives_tried", [])] + [0]
                    ),
                    "cost_savings": max(
                        [alt["cost"] for alt in r.get("alternatives_tried", [])] + [r["cost"]]
                    ) - r["cost"]
                }
            }
            for i, r in enumerate(results) if r["success"]
        ],
        "tier_analysis": {
            "tier_distribution": {tier: count for tier, count in tier_counts.items()},
            "tier_success_patterns": {
                "most_common_winner": max(tier_counts, key=tier_counts.get) if tier_counts else None,
                "escalation_examples": [
                    {
                        "question_snippet": r["question"][:60] + "...",
                        "expected_min_tier": r["expected_min_tier"],
                        "actual_winner": r["tier"],
                        "escalation_successful": True
                    }
                    for r in successes
                ]
            }
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\\n💾 Enhanced results saved to: {results_file}")
    print(f"📊 Training examples: {len(training_data['training_examples'])} successful trajectories")
    print(f"🎯 Ready for model training with detailed Oracle decision patterns")

if __name__ == "__main__":
    test_realistic_oracle()