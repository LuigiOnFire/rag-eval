#!/usr/bin/env python3
"""
Improved Real Oracle Test with better answer matching
"""
import json
import random
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Mock components for testing
class MockEmbeddingRetriever:
    def __init__(self, passages: List[Dict]):
        self.passages = passages
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Mock retrieval - return random passages for testing"""
        return random.sample(self.passages, min(k, len(self.passages)))

class MockBM25Retriever:
    def __init__(self, passages: List[Dict]):
        self.passages = passages
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Mock BM25 - return random passages for testing"""
        return random.sample(self.passages, min(k, len(self.passages)))

class MockGenerator:
    def __init__(self):
        # Store correct answers for realistic responses
        self.answers = {
            "Were Scott Derrickson and Ed Wood of the same nationality?": "yes",
            "What government position was held by the woman who portrayed Corliss Archer": "Chief of Protocol",
            "What science fantasy young adult series, told in first person": "Animorphs",
            "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?": "no",
            'The director of the romantic comedy "Big Stone Gap"': "Greenwich Village, New York City"
        }
    
    def generate(self, question: str, context: str = "", **kwargs) -> str:
        """Generate realistic answers that might match expected results"""
        # Try to find matching answer
        for key, answer in self.answers.items():
            if key in question:
                # Sometimes return exact answer, sometimes verbose
                if random.random() < 0.3:  # 30% chance of exact answer
                    return answer
                else:
                    return f"Based on the context provided: {answer}"
        
        return f"Based on available information: {random.choice(['yes', 'no', 'unknown'])}"

class MockAgent:
    """Mock agent that can handle the Oracle's component expectations"""
    
    def __init__(self, retriever, bm25_retriever, generator, structured_agent):
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        self.generator = generator
        self.structured_agent = structured_agent
        self.llm = generator  # Alias for compatibility
    
    def _retrieve_tier(self, query: str, tier: str) -> List[Dict]:
        """Retrieve based on tier strategy"""
        if tier == "FAST":
            return self.bm25_retriever.retrieve(query, k=5)
        elif tier == "SMART": 
            return self.retriever.retrieve(query, k=5)
        else:
            # Hybrid: combine both
            fast_docs = self.bm25_retriever.retrieve(query, k=3)
            smart_docs = self.retriever.retrieve(query, k=3)
            return fast_docs + smart_docs
    
    def run_full_trajectory(self, query: str, force_decompose: bool = False):
        """Run full decomposition trajectory"""
        class MockResult:
            def __init__(self, answer):
                self.final_answer = answer
                self.steps = [
                    type('obj', (object,), {'action': 'DECOMPOSE', 'input': query, 'output': 'Sub-queries generated'})(),
                    type('obj', (object,), {'action': 'RETRIEVE_SMART', 'input': 'sub-query', 'output': 'Retrieved docs'})(),
                    type('obj', (object,), {'action': 'GENERATE_FINAL', 'input': query, 'output': answer})()
                ]
        
        # Use structured agent for decomposition
        answer = self.structured_agent.decompose_and_answer(query, [])
        return MockResult(answer)

class MockStructuredAgent:
    def decompose_and_answer(self, question: str, passages: List[Dict]) -> str:
        """Mock structured decomposition"""
        mock_gen = MockGenerator()
        base_answer = mock_gen.generate(question)
        return f"Full decomposition result: {base_answer}"

class SmartAnswerJudge:
    """Improved judge that can handle answer variations"""
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answers for comparison"""
        if not answer:
            return ""
        
        # Extract key content from verbose responses
        answer = answer.strip().lower()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "based on the context provided:",
            "based on available information:",
            "answer based on the provided context:",
            "full decomposition result:",
            "according to the information:",
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Clean up punctuation and spacing
        answer = re.sub(r'[^\w\s]', ' ', answer)
        answer = ' '.join(answer.split())
        
        return answer
    
    def is_correct(self, generated: str, expected: str) -> bool:
        """Check if generated answer matches expected"""
        if not generated or not expected:
            return False
        
        gen_norm = self.normalize_answer(generated)
        exp_norm = self.normalize_answer(expected)
        
        # Exact match
        if gen_norm == exp_norm:
            return True
        
        # Check if expected answer is contained in generated
        if exp_norm in gen_norm:
            return True
        
        # Check for semantic equivalents
        semantic_matches = [
            (["yes", "true", "correct", "same"], ["yes"]),
            (["no", "false", "incorrect", "different"], ["no"]),
        ]
        
        for synonyms, targets in semantic_matches:
            if any(target in exp_norm for target in targets):
                if any(syn in gen_norm for syn in synonyms):
                    return True
        
        return False
    
    def evaluate(self, generated: str, expected: str) -> bool:
        """Check if generated answer matches expected (interface for Oracle)"""
        return self.is_correct(generated, expected)

# Load the actual exploration oracle
import sys
sys.path.append('/home/wcrawford/rag_eval')
sys.path.append('/home/wcrawford/rag_eval/src')
from exploration_oracle import ExplorationOracle, CostCalculator

def load_real_hotpot_questions(limit: int = 5) -> List[Dict]:
    """Load real HotPotQA questions"""
    questions_file = "/home/wcrawford/rag_eval/data/processed/questions.json"
    
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    # Take first N questions for testing
    return all_questions[:limit]

def load_passages() -> List[Dict]:
    """Load processed passages"""
    passages_file = "/home/wcrawford/rag_eval/data/processed/passages.json"
    
    print(f"📖 Loading passages from {passages_file}")
    with open(passages_file, 'r') as f:
        passages = json.load(f)
    
    print(f"✅ Loaded {len(passages)} passages for retrieval")
    return passages

def create_improved_retriever(passages: List[Dict]):
    """Create retriever using real passages"""
    return MockEmbeddingRetriever(passages)

def create_improved_bm25(passages: List[Dict]):
    """Create BM25 retriever using real passages"""  
    return MockBM25Retriever(passages)

@dataclass
class TestResult:
    question: str
    expected: str
    got: Optional[str]
    tier: Optional[str]
    cost: float
    success: bool

def test_improved_oracle():
    """Test Oracle with improved answer matching"""
    
    print("🚀 Initializing Improved Real 7-Tier Oracle")
    print("=" * 50)
    
    # Load data
    passages = load_passages()
    questions = load_real_hotpot_questions()
    
    print(f"📚 Loading HotPotQA questions from /home/wcrawford/rag_eval/data/processed/questions.json")
    print(f"✅ Loaded {len(questions)} real HotPotQA questions")
    
    # Create components
    retriever = create_improved_retriever(passages)
    bm25_retriever = create_improved_bm25(passages)
    generator = MockGenerator()
    structured_agent = MockStructuredAgent()
    
    # Create mock agent that wraps these components
    agent = MockAgent(retriever, bm25_retriever, generator, structured_agent)
    judge = SmartAnswerJudge()
    cost_calculator = CostCalculator()
    
    # Create Oracle
    oracle = ExplorationOracle(
        agent=agent,
        judge=judge,
        cost_calculator=cost_calculator
    )
    
    print(f"\n🔍 Testing Oracle with Real Questions")
    print("=" * 50)
    
    results = []
    
    for i, q_data in enumerate(questions):
        question = q_data["question"]
        expected = q_data["answer"]
        q_type = q_data.get("type", "unknown")
        
        print(f"\n📝 Question {i+1}: {question[:60]}...")
        print(f"🎯 Expected: {expected}")
        print(f"🏷️  Type: {q_type}")
        
        try:
            result = oracle.find_gold_trajectory(question, expected)
            
            if result:
                tier = result["gold_action"]
                answer = result["gold_answer"]
                cost = result["gold_cost"]
                
                print(f"✅ SUCCESS: {tier} (${cost:.3f})")
                print(f"📋 Answer: {answer[:100]}...")
                
                results.append(TestResult(
                    question=question,
                    expected=expected,
                    got=answer,
                    tier=tier,
                    cost=cost,
                    success=True
                ))
            else:
                print("❌ FAILED: All tiers unsuccessful")
                results.append(TestResult(
                    question=question,
                    expected=expected,
                    got=None,
                    tier=None,
                    cost=float('inf'),
                    success=False
                ))
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append(TestResult(
                question=question,
                expected=expected,
                got=None,
                tier=None,
                cost=float('inf'),
                success=False
            ))
    
    # Calculate statistics
    successes = [r for r in results if r.success]
    success_rate = len(successes) / len(results) * 100
    avg_cost = sum(r.cost for r in successes) / len(successes) if successes else 0
    
    # Tier distribution
    tier_counts = {}
    for result in successes:
        tier = result.tier
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    print(f"\n📊 FINAL RESULTS")
    print("=" * 50)
    print(f"Success Rate: {len(successes)}/{len(results)} ({success_rate:.1f}%)")
    print(f"Average Cost: ${avg_cost:.3f}")
    print("Tier Distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} questions")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/wcrawford/rag_eval/results/testing/test_oracle_improved_{timestamp}.json"
    
    print(f"\n💾 Saving results...")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "oracle_type": "7_tier_improved_test",
            "n_questions": len(questions)
        },
        "results": [asdict(r) for r in results]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")

if __name__ == "__main__":
    test_improved_oracle()