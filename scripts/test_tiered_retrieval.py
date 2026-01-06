#!/usr/bin/env python3
"""
Test the tiered retrieval system on known failure cases.

Tests FAST vs SMART vs DEEP on:
- Animorphs (semantic understanding needed)  
- Shirley Temple government (entity disambiguation)
- Ed Wood filmmaker (exact match vs film)
"""
import sys
sys.path.insert(0, 'src')

import json
import logging
from pathlib import Path
from tiered_retriever import TieredHybridRetriever, TieredRetrieverWrapper
from structured_agent import StructuredAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import wrappers
class LLMWrapper:
    """Minimal LLM wrapper for testing."""
    def generate(self, prompt: str) -> str:
        import requests
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral:latest",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


def test_retrieval_tiers():
    """Test different tiers on known failure queries."""
    # Load or build indexes
    index_dir = "data/indexes_tiered"
    
    if Path(index_dir).exists():
        logger.info("Loading existing tiered indexes...")
        retriever = TieredHybridRetriever()
        retriever.load_indexes(index_dir)
    else:
        logger.info("Building tiered indexes (this will take a few minutes)...")
        with open("data/processed/passages.json") as f:
            passages = json.load(f)
        
        retriever = TieredHybridRetriever()
        retriever.build_index(passages)
        retriever.save_indexes(index_dir)
    
    # Test queries that previously failed
    test_cases = [
        {
            'query': 'science fantasy young adult series companion books enslaved worlds',
            'expected_doc': 'Animorphs',
            'case': 'Animorphs (semantic understanding)'
        },
        {
            'query': 'Shirley Temple government position diplomat',
            'expected_doc': 'Shirley Temple',
            'case': 'Shirley Temple (entity + government)'
        },
        {
            'query': 'Ed Wood filmmaker director',  
            'expected_doc': 'Ed Wood',
            'case': 'Ed Wood (entity disambiguation)'
        },
        {
            'query': 'Scott Derrickson nationality director',
            'expected_doc': 'Scott Derrickson',
            'case': 'Scott Derrickson (entity + attribute)'
        }
    ]
    
    logger.info("Testing tiered retrieval on failure cases...")
    logger.info("=" * 80)
    
    for case in test_cases:
        query = case['query']
        expected = case['expected_doc']
        case_name = case['case']
        
        logger.info(f"\n{case_name}")
        logger.info(f"Query: {query}")
        logger.info(f"Looking for: {expected}")
        logger.info("-" * 50)
        
        # Test each tier
        for tier in ["FAST", "SMART", "DEEP"]:
            try:
                docs, cost = retriever.search(query, tier=tier, top_k=3)
                
                # Check if expected doc is in results
                found = any(expected.lower() in doc.title.lower() for doc in docs)
                status = "✓ FOUND" if found else "✗ MISSED"
                
                titles = [doc.title[:40] for doc in docs]
                logger.info(f"  {tier:5s} ({cost:.3f}Wh): {status} - {titles}")
                
            except Exception as e:
                logger.error(f"  {tier:5s}: ERROR - {e}")
    
    logger.info("\n" + "=" * 80)


def test_agent_with_tiers():
    """Test full agent with tiered retrieval."""
    logger.info("\nTesting StructuredAgent with tiered retrieval...")
    
    # Load tiered retriever
    retriever = TieredHybridRetriever()
    retriever.load_indexes("data/indexes_tiered")
    
    # Wrap for agent
    retriever_wrapper = TieredRetrieverWrapper(retriever, default_tier="FAST")
    llm = LLMWrapper()
    
    # Create agent with tiered retrieval
    agent = StructuredAgent(
        llm=llm,
        retriever=retriever_wrapper,
        use_tiered_retrieval=True
    )
    
    # Test on Animorphs question
    question = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    
    logger.info(f"Question: {question}")
    logger.info("Running with tiered escalation...")
    
    try:
        result = agent.run(question)
        
        logger.info(f"\nFinal Answer: {result.final_answer}")
        logger.info(f"Total Energy Cost: {agent.total_energy_cost:.3f} Wh")
        
        logger.info("\nTier escalation steps:")
        for step in agent.steps:
            if any(tier in step.action for tier in ["FAST", "SMART", "DEEP", "ESCALATE"]):
                logger.info(f"  {step.action}: {step.output[:80]}...")
                
    except Exception as e:
        logger.error(f"Agent error: {e}")


def main():
    """Run all tests."""
    print("🔬 Testing Tiered Hybrid Retrieval")
    print("=" * 80)
    
    # Test 1: Retrieval tiers comparison
    test_retrieval_tiers()
    
    # Test 2: Full agent with escalation
    test_agent_with_tiers()
    
    print("\n✅ Testing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())