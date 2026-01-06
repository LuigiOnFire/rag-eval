#!/usr/bin/env python3
"""
Test the decoder-based agent on a few queries.
"""

import sys
sys.path.insert(0, '.')

import json
import logging
from src.decoder_agent import DecoderAgent, ActionParser, Action, ActionType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_action_parser():
    """Test the action parser on various formats."""
    print("=== Testing Action Parser ===\n")
    
    test_cases = [
        # Basic retrieval
        ('Action: RETRIEVE_KEYWORD["Shirley Temple government position"]',
         ActionType.RETRIEVE_KEYWORD, "Shirley Temple government position"),
        
        # Dense retrieval
        ('Action: RETRIEVE_DENSE["actresses who held political office"]',
         ActionType.RETRIEVE_DENSE, "actresses who held political office"),
        
        # Decompose SLM
        ('Thought: Simple decomposition needed.\nAction: DECOMPOSE_SLM',
         ActionType.DECOMPOSE_SLM, None),
        
        # Decompose LLM
        ('Thought: Complex multi-hop question.\nAction: DECOMPOSE_LLM',
         ActionType.DECOMPOSE_LLM, None),
        
        # Reason SLM
        ('Action: REASON_SLM',
         ActionType.REASON_SLM, None),
        
        # Reason LLM
        ('Action: REASON_LLM',
         ActionType.REASON_LLM, None),
        
        # Generate SLM (simple question)
        ('Thought: Simple and confident.\nAction: GENERATE_SLM',
         ActionType.GENERATE_SLM, None),
        
        # Generate LLM (complex question)
        ('Thought: Need quality answer.\nAction: GENERATE_LLM',
         ActionType.GENERATE_LLM, None),
    ]
    
    for text, expected_type, expected_param in test_cases:
        action = ActionParser.parse(text)
        if action:
            status = "✓" if action.action_type == expected_type else "✗"
            print(f"{status} Parsed: {action.action_type.value}")
            if action.parameters:
                print(f"  Parameter: {action.parameters}")
        else:
            print(f"✗ Failed to parse: {text[:50]}...")
        print()


def test_with_mock_llm():
    """Test the agent with a mock LLM that follows a script."""
    print("\n=== Testing Agent with Mock LLM ===\n")
    
    # Mock LLM that returns scripted responses
    class MockLLM:
        def __init__(self):
            self.call_count = 0
            self.responses = [
                # Step 1: Decompose with LLM (complex question)
                '''Thought: This is a multi-hop question. I need to first find who played Aunt May, then find their government position.
Action: DECOMPOSE_LLM''',
                
                # Step 2: Retrieve for sub-question 1
                '''Thought: Let me search for the Aunt May actress.
Action: RETRIEVE_KEYWORD["Aunt May actress Spider-Man 3 cast"]''',
                
                # Step 3: Retrieve for sub-question 2  
                '''Thought: From the context, Rosemary Harris played Aunt May. Now I need to find her government position.
Action: RETRIEVE_KEYWORD["Rosemary Harris government position ambassador"]''',
                
                # Step 4: Generate answer with LLM (complex answer)
                '''Thought: Based on the retrieved documents, I cannot find evidence that Rosemary Harris held a government position.
Action: GENERATE_LLM''',
            ]
        
        def generate(self, prompt: str) -> str:
            response = self.responses[min(self.call_count, len(self.responses) - 1)]
            self.call_count += 1
            return response
    
    # Mock retriever
    class MockRetriever:
        def search(self, query: str, top_k: int = 5):
            if "aunt may" in query.lower() or "spider-man" in query.lower():
                return [
                    {'title': 'Spider-Man 3', 'text': 'Spider-Man 3 is a 2007 film. Cast includes Tobey Maguire as Spider-Man, Kirsten Dunst as Mary Jane, and Rosemary Harris as Aunt May.'},
                    {'title': 'Rosemary Harris', 'text': 'Rosemary Harris is a British actress known for her role as Aunt May in the Spider-Man trilogy.'},
                ]
            elif "rosemary harris" in query.lower():
                return [
                    {'title': 'Rosemary Harris', 'text': 'Rosemary Harris (born 1927) is a British actress. She won a Tony Award and appeared in many films.'},
                ]
            return []
    
    # Create agent
    agent = DecoderAgent(
        llm=MockLLM(),
        slm=MockLLM(),  # Not used in this test
        retriever=MockRetriever(),
        max_steps=10,
    )
    
    # Run query
    query = "What government position was held by the actress who played Aunt May in Spider-Man 3?"
    result = agent.run(query, ground_truth="Chief of Protocol")
    
    print(f"Query: {query}")
    print(f"\nSteps taken: {len(result['steps'])}")
    for step in result['steps']:
        print(f"\n  Step {step['step']}: {step['action']}")
        if step.get('parameters'):
            print(f"    Params: {step['parameters']}")
        print(f"    Result: {step['result'][:100]}...")
    
    print(f"\nFinal Answer: {result['answer']}")
    print(f"Correct: {result['is_correct']}")
    print(f"Energy: {result['total_energy_wh']:.4f} Wh")


def test_with_real_llm():
    """Test with actual Ollama LLM."""
    print("\n=== Testing with Real LLM (Ollama) ===\n")
    
    try:
        from src.generator import OllamaGenerator
        from src.retriever import BM25Retriever
        import json
        
        # Load passages directly
        print("Loading passages...")
        with open("data/processed/passages.json") as f:
            passages = json.load(f)
        print(f"Loaded {len(passages)} passages")
        
        # Create retriever
        print("Building BM25 index...")
        retriever = BM25Retriever()
        retriever.build_index(passages)
        
        # Load LLM
        print("Loading LLM...")
        llm = OllamaGenerator(model_name="llama3:8b")
        slm = OllamaGenerator(model_name="mistral:latest")
        
        # Wrap generator to match expected interface
        class LLMWrapper:
            def __init__(self, gen):
                self.gen = gen
            def generate(self, prompt: str) -> str:
                # Call Ollama API directly with the raw prompt
                import requests
                response = requests.post(
                    f"{self.gen.base_url}/api/chat",
                    json={
                        "model": self.gen.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": self.gen.temperature,
                            "top_p": self.gen.top_p,
                            "num_predict": 500
                        }
                    },
                    timeout=self.gen.timeout
                )
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "").strip()
        
        class RetrieverWrapper:
            def __init__(self, ret):
                self.ret = ret
            def search(self, query: str, top_k: int = 5):
                passages, scores = self.ret.retrieve(query, top_k=top_k)
                return [{'title': p.get('title', ''), 'text': p.get('text', '')} for p in passages]
        
        # Create agent
        agent = DecoderAgent(
            llm=LLMWrapper(llm),
            slm=LLMWrapper(slm),
            retriever=RetrieverWrapper(retriever),
            max_steps=6,
        )
        
        # Test query 1: Comparison question
        query = "Are Local H and For Against both from the United States?"
        print(f"Query: {query}\n")
        
        result = agent.run(query, ground_truth="Yes")
        
        print(f"\nSteps taken: {len(result['steps'])}")
        for step in result['steps']:
            print(f"\n--- Step {step['step']}: {step['action']} ---")
            if step.get('parameters'):
                print(f"Params: {step['parameters']}")
            print(f"LLM Output:\n{step['llm_output'][:300]}...")
        
        print(f"\n=== RESULT ===")
        print(f"Answer: {result['answer']}")
        print(f"Correct: {result['is_correct']}")
        print(f"Energy: {result['total_energy_wh']:.4f} Wh")
        
        # Test query 2: Multi-hop bridge question
        print("\n" + "="*60)
        print("=== TEST 2: Multi-hop Bridge Question ===")
        print("="*60 + "\n")
        
        query2 = "What is the name of the city where the headquarters of the company founded by Elon Musk is located?"
        print(f"Query: {query2}\n")
        
        result2 = agent.run(query2, ground_truth="Palo Alto")  # Tesla HQ
        
        print(f"\nSteps taken: {len(result2['steps'])}")
        for step in result2['steps']:
            print(f"\n--- Step {step['step']}: {step['action']} ---")
            if step.get('parameters'):
                print(f"Params: {step['parameters']}")
            print(f"LLM Output:\n{step['llm_output'][:300]}...")
        
        print(f"\n=== RESULT ===")
        print(f"Answer: {result2['answer']}")
        print(f"Correct: {result2['is_correct']}")
        print(f"Energy: {result2['total_energy_wh']:.4f} Wh")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_action_parser()
    test_with_mock_llm()
    
    # Test with real LLM
    test_with_real_llm()
