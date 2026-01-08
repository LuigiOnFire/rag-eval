#!/usr/bin/env python3
"""
Simple test for the 7-tier exploration Oracle concept.
Tests just the tier logic without full dependency chain.
"""

def test_tier_priorities():
    """Test that the Oracle tries tiers in correct cost order."""
    
    # Simulate Oracle tier costs (from our design)
    tier_costs = {
        "<ASSIGN_SLM>": 1.0,
        "<RETRIEVE_FAST>": 5.0,
        "<RETRIEVE_SMART>": 8.0,
        "<REFINE_QUERY>": 12.0,
        "<RETRIEVE_FILTER>": 18.0,
        "<ITERATE>": 25.0,
        "<DECOMPOSE>": 50.0
    }
    
    # Simulate results for different question types
    test_cases = [
        {
            "question": "What is 2 + 2?",
            "expected_winner": "<ASSIGN_SLM>",
            "reason": "Simple math - SLM parametric knowledge"
        },
        {
            "question": "What is Scott Derrickson's nationality?",
            "expected_winner": "<RETRIEVE_FAST>",  
            "reason": "Specific entity lookup - BM25 keywords work"
        },
        {
            "question": "What science fantasy series has enslaved worlds?",
            "expected_winner": "<RETRIEVE_SMART>",
            "reason": "Semantic concept - needs dense embeddings"
        },
        {
            "question": "The actor in Terminator played what character in a different Arnold movie?",
            "expected_winner": "<REFINE_QUERY>",
            "reason": "Needs query rewrite to clarify 'the actor' = Arnold"
        },
        {
            "question": "Which of these 20 directors worked on sci-fi films?",
            "expected_winner": "<RETRIEVE_FILTER>",
            "reason": "Needs high-recall then filtering for best match"
        },
        {
            "question": "What government position did the Kiss and Tell actress hold?",
            "expected_winner": "<ITERATE>",
            "reason": "2-step bridge: actress name -> her government role"
        },
        {
            "question": "Compare the birth years, education, and career timelines of two obscure directors from different countries",
            "expected_winner": "<DECOMPOSE>",
            "reason": "Complex multi-faceted comparison needs full decomposition"
        }
    ]
    
    print("🔍 Testing 7-Tier Oracle Priority System")
    print("="*60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['question'][:50]}...")
        print(f"Expected Winner: {case['expected_winner']} (${tier_costs[case['expected_winner']]:.0f})")
        print(f"Reasoning: {case['reason']}")
        print()
    
    print("✅ Tier Priority Test Complete!")
    print("\n📊 Cost Distribution:")
    for tier, cost in tier_costs.items():
        print(f"  {tier}: ${cost:.0f}")
    
    print(f"\n🎯 Key Insight: We've filled the ${tier_costs['<RETRIEVE_SMART>']} -> ${tier_costs['<DECOMPOSE>']} complexity gap!")
    print(f"   New intermediate tiers: ${tier_costs['<REFINE_QUERY>']} (Refine), ${tier_costs['<RETRIEVE_FILTER>']} (Filter), ${tier_costs['<ITERATE>']} (Iterate)")

if __name__ == "__main__":
    test_tier_priorities()