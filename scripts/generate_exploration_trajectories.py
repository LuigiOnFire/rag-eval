#!/usr/bin/env python3
"""
Generate training trajectories using the exploration-based Oracle.

This script replaces the old "thinking" Oracle with an "exploring" Oracle
that tries all possible execution paths and picks the cheapest successful one.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

sys.path.append('/home/wcrawford/rag_eval')

from src.exploration_oracle import ExplorationOracle, CostCalculator
from src.structured_agent import StructuredAgent
from src.retriever import HybridRetriever, BM25Retriever, FaissRetriever
from src.generator import OllamaGenerator

class SimpleEvaluator:
    """Simple string-based evaluator for the Oracle."""
    
    def evaluate(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth (case-insensitive, flexible)."""
        if not answer or not ground_truth:
            return False
            
        answer = answer.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # Direct match
        if answer == ground_truth:
            return True
            
        # Ground truth contained in answer
        if ground_truth in answer:
            return True
            
        # Answer contained in ground truth (for longer expected answers)
        if answer in ground_truth:
            return True
            
        return False

def load_hotpot_samples(n_samples: int = 50) -> List[Dict[str, Any]]:
    """Load HotPotQA samples for testing."""
    # For now, create some sample questions to test the Oracle
    samples = [
        {
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "answer": "yes",
            "type": "comparison"
        },
        {
            "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "answer": "Chief of Protocol",
            "type": "bridge"
        },
        {
            "question": "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
            "answer": "3,677 seated",
            "type": "bridge"
        },
        {
            "question": "Are Local H and For Against both from the United States?",
            "answer": "yes", 
            "type": "comparison"
        },
        {
            "question": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
            "answer": "Animorphs",
            "type": "bridge"
        }
    ]
    
    # Duplicate samples to reach n_samples
    while len(samples) < n_samples:
        samples.extend(samples[:min(len(samples), n_samples - len(samples))])
    
    return samples[:n_samples]


class TrainingDataGenerator:
    """Generate training data using exploration-based Oracle."""
    
    def __init__(self):
        print("🚀 Initializing Training Data Generator with 7-Tier Exploration Oracle")
        
        # Initialize components
        print("📚 Loading corpus and pre-built retrieval indexes...")
        
        # Load passages data
        import json
        with open("/home/wcrawford/rag_eval/data/processed/passages.json", 'r') as f:
            passages_data = json.load(f)
            
        # Initialize component retrievers using pre-built indexes
        print("🔧 Loading pre-built BM25 retriever...")
        bm25_retriever = BM25Retriever()
        bm25_retriever.load_index(
            "/home/wcrawford/rag_eval/data/indexes/faiss.bm25.pkl",
            "/home/wcrawford/rag_eval/data/processed/passages.json"
        )
        
        print("🔧 Loading pre-built Dense retriever...")
        faiss_retriever = FaissRetriever()
        faiss_retriever.load_index(
            "/home/wcrawford/rag_eval/data/indexes/faiss.index",
            "/home/wcrawford/rag_eval/data/processed/passages.json"
        )
        
        print("🔧 Building Hybrid retriever...")
        self.retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=faiss_retriever
        )
        
        print("🤖 Initializing generator...")
        self.generator = OllamaGenerator()
        
        print("🧠 Setting up structured agent...")
        self.agent = StructuredAgent(
            llm=self.generator,
            retriever=self.retriever
        )
        
        print("⚖️ Initializing evaluator and cost calculator...")
        self.evaluator = SimpleEvaluator()
        self.cost_calculator = CostCalculator()
        
        # Create the 7-tier exploration Oracle
        print("🔍 Creating 7-tier exploration Oracle...")
        self.oracle = ExplorationOracle(
            agent=self.agent,
            judge=self.evaluator,
            cost_calculator=self.cost_calculator
        )
        
        print("✅ Initialization complete!")
        
    def generate_training_data(self, questions: List[Dict[str, Any]], 
                             output_file: str, n_samples: int = 20) -> Dict[str, Any]:
        """
        Generate training data by exploring all execution paths.
        
        For each question, the Oracle tries:
        1. Pure SLM (no retrieval)
        2. Fast retrieval (BM25) 
        3. Smart retrieval (Dense)
        4. Full decomposition
        
        And picks the cheapest path that succeeds.
        """
        
        print(f"🎯 Generating training data for {n_samples} samples")
        print(f"📊 Using 7-tier exploration Oracle (fills complexity gap)")
        print("🔍 Tiers: SLM($1) -> Fast($5) -> Smart($8) -> Refine($12) -> Filter($18) -> Iterate($25) -> Decompose($50+)")
        
        training_samples = []
        stats = {
            "total_attempted": 0,
            "successful_trajectories": 0,
            "path_distribution": {
                "<ASSIGN_SLM>": 0,
                "<RETRIEVE_FAST>": 0,  
                "<RETRIEVE_SMART>": 0,
                "<REFINE_QUERY>": 0,
                "<RETRIEVE_FILTER>": 0,
                "<ITERATE>": 0,
                "<DECOMPOSE>": 0
            },
            "failed_samples": [],
            "total_cost": 0.0
        }
        
        for i, question_data in enumerate(questions[:n_samples]):
            print(f"\n🔍 Sample {i+1}/{n_samples}")
            print(f"❓ Question: {question_data['question'][:100]}...")
            print(f"🎯 Ground truth: {question_data['answer']}")
            
            stats["total_attempted"] += 1
            
            # Use Oracle to find gold trajectory
            gold_trajectory = self.oracle.find_gold_trajectory(
                query=question_data['question'],
                ground_truth=question_data['answer']
            )
            
            if gold_trajectory:
                # Success! We found a working path
                stats["successful_trajectories"] += 1
                stats["path_distribution"][gold_trajectory["gold_action"]] += 1
                stats["total_cost"] += gold_trajectory["gold_cost"]
                
                training_samples.append({
                    "id": i,
                    "question": question_data['question'],
                    "ground_truth": question_data['answer'],
                    "question_type": question_data.get('type', 'unknown'),
                    **gold_trajectory
                })
                
                print(f"✅ Success: {gold_trajectory['gold_action']} (cost: {gold_trajectory['gold_cost']:.4f})")
                
            else:
                # All paths failed
                stats["failed_samples"].append({
                    "id": i,
                    "question": question_data['question'],
                    "ground_truth": question_data['answer'],
                    "reason": "All exploration paths failed"
                })
                print(f"❌ Failed: All paths unsuccessful")
        
        # Calculate final statistics  
        success_rate = (stats["successful_trajectories"] / stats["total_attempted"]) * 100
        avg_cost = stats["total_cost"] / max(stats["successful_trajectories"], 1)
        
        print(f"\n📈 FINAL STATISTICS")
        print(f"Success Rate: {success_rate:.1f}% ({stats['successful_trajectories']}/{stats['total_attempted']})")
        print(f"Average Cost: {avg_cost:.4f}")
        print(f"Path Distribution:")
        for path, count in stats["path_distribution"].items():
            percentage = (count / max(stats["successful_trajectories"], 1)) * 100
            print(f"  {path}: {count} samples ({percentage:.1f}%)")
        
        # Save results
        results = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_samples": n_samples,
                "oracle_type": "7_tier_exploration_oracle",
                "tiers": [
                    "Tier 0: Pure SLM (~$1)",
                    "Tier 1: Fast BM25 (~$5)", 
                    "Tier 2: Smart Dense (~$8)",
                    "Tier 3: Refine Query + Hybrid (~$12)",
                    "Tier 4: High-Recall + Filter (~$18)",
                    "Tier 5: 2-Step Iterative (~$25)",
                    "Tier 6: Full Decomposition (~$50+)"
                ],
                "success_rate": success_rate,
                "avg_cost_per_sample": avg_cost
            },
            "stats": stats,
            "training_samples": training_samples
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        return results

def main():
    """Main execution function."""
    
    # Load HotPotQA data
    print("📚 Loading test samples...")
    questions = load_hotpot_samples(n_samples=10)  # Start small for debugging
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/wcrawford/rag_eval/data/sft_trajectories/exploration_oracle_{timestamp}.json"
    
    # Generate training data with just 1 sample for debugging
    generator = TrainingDataGenerator()
    results = generator.generate_training_data(
        questions=questions,
        output_file=output_file,
        n_samples=1  # DEBUG: Start with just 1 sample
    )
    
    print("\n🎉 Training data generation complete!")
    print(f"📊 Generated {len(results['training_samples'])} successful training samples")
    
    # Show some examples of what we learned
    if results['training_samples']:
        print(f"\n🏆 Example winning strategies:")
        for sample in results['training_samples'][:3]:
            print(f"  Q: {sample['question'][:60]}...")
            print(f"  Strategy: {sample['gold_action']} (cost: {sample['gold_cost']:.4f})")
            print(f"  Answer: {sample['gold_answer'][:40]}...")
            print()

if __name__ == "__main__":
    main()