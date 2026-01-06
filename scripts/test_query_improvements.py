#!/usr/bin/env python3
"""
Test tiered retrieval concept on key failing queries.

This script tests whether better retrieval could fix our failing cases
by comparing BM25 vs semantic search results.
"""
import sys
sys.path.insert(0, 'src')

import json
from retriever import BM25Retriever

# Test different query formulations for our failing cases
failing_cases = [
    {
        'original_question': 'What government position was held by the woman who portrayed Corliss Archer in Kiss and Tell?',
        'agent_queries': [
            'What government position was held by Shirley Temple (who portrayed Corliss Archer in the film Kiss and Tell)?'
        ],
        'better_queries': [
            'Shirley Temple government position',
            'Shirley Temple ambassador',
            'Shirley Temple diplomat',
            'Shirley Temple Chief Protocol',
            'Shirley Temple political career'
        ],
        'expected_answer': 'Chief of Protocol'
    },
    {
        'original_question': 'What science fantasy young adult series, told in first person, has companion books about enslaved worlds?',
        'agent_queries': [
            'Which young adult series is categorized as science fantasy?',
            'In what narrative perspective is the science fantasy young adult series featuring an unknown protagonist presented?'
        ],
        'better_queries': [
            'Animorphs',
            'science fantasy young adult series first person',
            'young adult series enslaved worlds alien species',
            'companion books alien species',
            'first person science fantasy series'
        ],
        'expected_answer': 'Animorphs'
    },
    {
        'original_question': 'Who was known by stage name Aladin and helped organizations as consultant?',
        'agent_queries': [
            'Which organization(s) did Aladin (who is unknown by a stage name) consult to improve their performance?'
        ],
        'better_queries': [
            'Eenasul Fateh',
            'Aladin stage name',
            'magician consultant Aladin',
            'Eenasul Fateh Aladin consultant',
            'stage name Aladin organizations'
        ],
        'expected_answer': 'Eenasul Fateh'
    }
]


def test_query_quality():
    """Test whether better queries would improve retrieval."""
    
    # Load passages and build BM25
    print("Loading passages and building BM25 index...")
    with open('data/processed/passages.json') as f:
        passages = json.load(f)
    
    retriever = BM25Retriever()
    retriever.build_index(passages)
    print(f"Built index with {len(passages)} passages\\n")
    
    for case in failing_cases:
        print(f"{'='*80}")
        print(f"CASE: {case['original_question'][:60]}...")
        print(f"Expected: {case['expected_answer']}")
        print(f"{'='*80}")
        
        print("\\n🤖 AGENT QUERIES (what it actually searched):")
        for query in case['agent_queries']:
            docs, scores = retriever.retrieve(query, top_k=3)
            print(f"\\nQuery: {query}")
            print(f"Results:")
            
            # Check if expected answer appears in results
            found_expected = False
            for i, doc in enumerate(docs, 1):
                title = doc.get('title', 'No title')[:60]
                text_preview = doc.get('text', '')[:100].replace('\\n', ' ')
                print(f"  {i}. [{title}] {text_preview}...")
                
                # Simple check for expected answer
                full_text = (title + ' ' + doc.get('text', '')).lower()
                if case['expected_answer'].lower() in full_text:
                    print(f"     ✅ CONTAINS EXPECTED: {case['expected_answer']}")
                    found_expected = True
                    
            if not found_expected:
                print(f"     ❌ Missing expected answer: {case['expected_answer']}")
                
        print("\\n💡 BETTER QUERIES (improved formulations):")
        for query in case['better_queries']:
            docs, scores = retriever.retrieve(query, top_k=3)
            print(f"\\nQuery: {query}")
            
            # Check if expected answer appears in results
            found_expected = False
            for i, doc in enumerate(docs, 1):
                title = doc.get('title', 'No title')[:60]
                
                # Simple check for expected answer
                full_text = (title + ' ' + doc.get('text', '')).lower()
                if case['expected_answer'].lower() in full_text:
                    print(f"  {i}. [{title}] ✅ CONTAINS: {case['expected_answer']}")
                    found_expected = True
                    break
                else:
                    print(f"  {i}. [{title}]")
                    
            if found_expected:
                print(f"     🎯 SUCCESS: Found {case['expected_answer']} with better query!")
                break
                
        print("\\n")


if __name__ == "__main__":
    test_query_quality()