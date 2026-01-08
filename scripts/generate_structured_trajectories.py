#!/usr/bin/env python3
"""
Generate training trajectories using the improved structured agent.

This script:
1. Uses the structured agent with Gemini's fixes
2. Generates trajectories on HotPotQA samples
3. Converts successful trajectories to SFT training format
4. Saves results for model training
"""
import sys
sys.path.insert(0, '/home/wcrawford/rag_eval')

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset

from src.structured_agent import StructuredAgent
from src.tiered_retriever import TieredHybridRetriever, TieredRetrieverWrapper
from src.retriever import BM25Retriever
from src.generator import OllamaGenerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class LLMWrapper:
    """Wrap OllamaGenerator for agent interface."""
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


def trajectory_to_sft_examples(trajectory_result, question: str, is_correct: bool) -> List[Dict]:
    """
    Convert a trajectory into SFT training examples.
    
    Each example is: {input: context, output: next_action}
    Only use successful trajectories.
    """
    if not is_correct:
        return []
    
    examples = []
    context = f"Question: {question}\n\n"
    
    for i, step in enumerate(trajectory_result.steps):
        # Build input context (everything up to this step)
        input_context = context
        
        # Add the step as expected output
        if step.action in ['DECOMPOSE', 'REWRITE', 'RETRIEVE', 'EXTRACT', 'GENERATE_FINAL']:
            # Format as instruction following
            if step.action == 'DECOMPOSE':
                output = f"I need to break this question down:\n{step.output}"
            elif step.action == 'REWRITE':
                output = f"Let me rewrite this with context:\n{step.output}" 
            elif step.action == 'RETRIEVE':
                output = f"I'll search for: {step.input}\nRetrieved: {step.output}"
            elif step.action == 'EXTRACT':
                output = f"From the context, I can extract:\n{step.output}"
            elif step.action == 'GENERATE_FINAL':
                output = f"Final answer: {step.output}"
            else:
                output = step.output
                
            examples.append({
                'input': input_context.strip(),
                'output': output.strip(),
                'step_type': step.action,
                'question': question
            })
        
        # Update context for next step
        context += f"Step {i+1} ({step.action}): {step.output}\\n"
        if len(context) > 2000:  # Keep context manageable
            context = context[-2000:]
    
    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate SFT trajectories with structured agent")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of HotPotQA samples")
    parser.add_argument("--output_dir", type=str, default="data/sft_trajectories", help="Output directory")
    parser.add_argument("--passages", type=str, default="data/processed/passages.json")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load passages and build tiered retriever
    logger.info("Loading passages and building tiered retriever...")
    with open(args.passages) as f:
        passages = json.load(f)
    logger.info(f"Loaded {len(passages)} passages")
    
    # Check if tiered indexes exist, otherwise build them
    index_dir = "data/indexes_tiered"
    if Path(index_dir).exists():
        logger.info("Loading existing tiered indexes...")
        retriever = TieredHybridRetriever()
        retriever.load_indexes(index_dir)
    else:
        logger.info("Building tiered indexes (this may take several minutes)...")
        retriever = TieredHybridRetriever()
        retriever.build_index(passages)
        retriever.save_indexes(index_dir)
        logger.info("Tiered indexes built and saved")
    
    # Load LLM
    logger.info("Loading LLM...")
    llm = OllamaGenerator(model_name="mistral:latest")
    
    # Create structured agent with tiered retrieval
    tiered_wrapper = TieredRetrieverWrapper(retriever, default_tier="FAST")
    agent = StructuredAgent(
        llm=LLMWrapper(llm),
        retriever=tiered_wrapper,
        max_docs_per_query=5,
        use_tiered_retrieval=True
    )
    
    # Load HotPotQA validation samples
    logger.info(f"Loading {args.n_samples} HotPotQA samples...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Generate trajectories
    trajectories = []
    sft_examples = []
    stats = {
        'total': 0,
        'correct': 0,
        'comparison_correct': 0,
        'comparison_total': 0,
        'bridge_correct': 0,
        'bridge_total': 0,
        'total_energy_wh': 0.0,
        'avg_energy_wh': 0.0,
        'errors': []
    }
    
    for i, item in enumerate(dataset):
        if i >= args.n_samples:
            break
            
        question = item['question']
        ground_truth = item['answer']
        question_type = item['type']
        
        logger.info(f"\\n[{i+1}/{args.n_samples}] {question_type}: {question[:60]}...")
        
        try:
            # Reset energy tracking
            agent.total_energy_cost = 0.0
            
            # Run agent
            result = agent.run(question)
            
            # Check correctness
            gt_lower = ground_truth.lower().strip()
            ans_lower = result.final_answer.lower().strip()
            is_correct = gt_lower in ans_lower or ans_lower in gt_lower
            
            # Update stats
            energy_cost = getattr(agent, 'total_energy_cost', 0.0)
            stats['total'] += 1
            stats['total_energy_wh'] += energy_cost
            
            if is_correct:
                stats['correct'] += 1
            
            if question_type == 'comparison':
                stats['comparison_total'] += 1
                if is_correct:
                    stats['comparison_correct'] += 1
            elif question_type == 'bridge':
                stats['bridge_total'] += 1
                if is_correct:
                    stats['bridge_correct'] += 1
            
            # Store trajectory
            trajectory = {
                'question': question,
                'ground_truth': ground_truth,
                'answer': result.final_answer,
                'question_type': question_type,
                'correct': is_correct,
                'energy_cost_wh': energy_cost,
                'steps': [
                    {
                        'action': step.action,
                        'input': step.input,
                        'output': step.output
                    }
                    for step in result.steps
                ]
            }
            trajectories.append(trajectory)
            
            # Convert to SFT examples if correct
            if is_correct:
                examples = trajectory_to_sft_examples(result, question, is_correct)
                sft_examples.extend(examples)
                logger.info(f"  ✓ Correct - Generated {len(examples)} SFT examples")
            else:
                logger.info(f"  ✗ Incorrect: {result.final_answer[:50]}...")
                stats['errors'].append({
                    'question': question[:60],
                    'expected': ground_truth,
                    'got': result.final_answer[:60]
                })
        
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            stats['errors'].append({
                'question': question[:60],
                'expected': ground_truth,
                'got': f"ERROR: {e}"
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save trajectories
    trajectories_file = output_dir / f"trajectories_{timestamp}.json"
    with open(trajectories_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'n_samples': args.n_samples,
                'agent_type': 'structured_agent_v2_tiered',
                'retrieval_system': 'tiered_fast_smart_deep'
            },
            'stats': stats,
            'trajectories': trajectories
        }, f, indent=2)
    
    # Save SFT examples
    sft_file = output_dir / f"sft_examples_{timestamp}.json"
    with open(sft_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'n_examples': len(sft_examples),
                'source_samples': args.n_samples,
                'agent_type': 'structured_agent_v2'
            },
            'examples': sft_examples
        }, f, indent=2)
    
    # Print summary
    logger.info(f"\\n{'='*60}")
    logger.info("SUMMARY")
    logger.info('='*60)
    
    accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    stats['avg_energy_wh'] = stats['total_energy_wh'] / stats['total'] if stats['total'] > 0 else 0.0
    
    logger.info(f"Overall accuracy: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    logger.info(f"Average energy per query: {stats['avg_energy_wh']:.4f} Wh")
    logger.info(f"Total energy consumption: {stats['total_energy_wh']:.3f} Wh")
    
    if stats['comparison_total'] > 0:
        comp_acc = stats['comparison_correct'] / stats['comparison_total'] * 100
        logger.info(f"Comparison: {stats['comparison_correct']}/{stats['comparison_total']} ({comp_acc:.1f}%)")
    
    if stats['bridge_total'] > 0:
        bridge_acc = stats['bridge_correct'] / stats['bridge_total'] * 100  
        logger.info(f"Bridge: {stats['bridge_correct']}/{stats['bridge_total']} ({bridge_acc:.1f}%)")
    
    logger.info(f"\\nGenerated {len(sft_examples)} SFT training examples")
    logger.info(f"Trajectories saved: {trajectories_file}")
    logger.info(f"SFT examples saved: {sft_file}")
    
    if accuracy >= 70:
        logger.info(f"\\n🎉 Target achieved! {accuracy:.1f}% >= 70%")
    else:
        logger.info(f"\\n📈 Progress made: {accuracy:.1f}% accuracy")


if __name__ == "__main__":
    main()