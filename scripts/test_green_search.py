#!/usr/bin/env python3
"""
Test script for GreenSearch module.

Tests:
1. Basic data structures (ActionCall, SearchNode)
2. Judge functionality
3. Search algorithm with mock components
4. Integration with real retriever/generators (optional)
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.green_search import (
    GreenSearch, SearchNode, Judge, Trajectory,
    ActionCall, ActionType, Model, RetrievalMethod,
    GENERATE, DECOMPOSE, RETRIEVE, REASON,
)


def test_action_calls():
    """Test ActionCall creation and properties."""
    print("=" * 60)
    print("Testing ActionCall")
    print("=" * 60)
    
    # Test all action types
    actions = [
        GENERATE(Model.SLM),
        GENERATE(Model.LLM),
        DECOMPOSE(Model.SLM),
        DECOMPOSE(Model.LLM),
        RETRIEVE(RetrievalMethod.KEYWORD, query="test query", top_k=3),
        RETRIEVE(RetrievalMethod.KEYWORD, query="test query", top_k=10),
        RETRIEVE(RetrievalMethod.DENSE, query="test query", top_k=5),
        REASON(Model.SLM),
        REASON(Model.LLM),
    ]
    
    print("\nAction -> Legacy ID mapping:")
    for action in actions:
        print(f"  {action} -> legacy_id={action.legacy_action_id}, terminal={action.is_terminal}")
    
    # Test immutability (frozen dataclass)
    try:
        actions[0].model = Model.LLM  # type: ignore[misc]
        print("\n✗ ActionCall should be immutable!")
    except Exception:
        print("\n✓ ActionCall is properly immutable")
    
    print()


def test_search_node():
    """Test SearchNode functionality."""
    print("=" * 60)
    print("Testing SearchNode")
    print("=" * 60)
    
    # Create root node
    root = SearchNode(
        query="Who directed Doctor Strange?",
        original_query="Who directed Doctor Strange?",
    )
    print(f"\nRoot node: {root.node_id}")
    print(f"  Depth: {root.depth}")
    print(f"  Cost: {root.accumulated_cost}")
    
    # Create child node
    child = root.create_child(
        action=RETRIEVE(RetrievalMethod.KEYWORD, query="Doctor Strange director", top_k=3),
        observation="BM25 'Doctor Strange director': Scott Derrickson, Marvel...",
        cost=0.009,
        energy_wh=0.009,
        new_context=[{"title": "Scott Derrickson", "text": "Scott Derrickson is an American director..."}],
    )
    print(f"\nChild node: {child.node_id}")
    print(f"  Depth: {child.depth}")
    print(f"  Cost: {child.accumulated_cost:.4f}")
    print(f"  Context count: {len(child.context)}")
    print(f"  Parent: {child.parent.node_id if child.parent else 'None'}")
    
    # Test compressed state
    print(f"\nCompressed state (BERT-compatible):")
    state = child.get_compressed_state()
    print(f"  Length: {len(state)} chars")
    print(f"  Preview: {state[:200]}...")
    
    # Test can_take_action
    print(f"\nAction validation:")
    print(f"  Can GENERATE? {child.can_take_action(GENERATE())}")
    print(f"  Can RETRIEVE same query? {child.can_take_action(RETRIEVE(query='Doctor Strange director'))}")
    print(f"  Can RETRIEVE new query? {child.can_take_action(RETRIEVE(query='new query'))}")
    
    # Test trajectory extraction
    grandchild = child.create_child(
        action=GENERATE(Model.LLM),
        observation="Generated (LLM): Scott Derrickson",
        cost=0.015,
        energy_wh=0.015,
        answer="Scott Derrickson",
    )
    
    trajectory = grandchild.get_trajectory()
    print(f"\nTrajectory length: {len(trajectory)}")
    for i, node in enumerate(trajectory):
        action_str = str(node.action_taken) if node.action_taken else "ROOT"
        print(f"  Step {i}: {action_str}")
    
    print()


def test_judge():
    """Test Judge functionality."""
    print("=" * 60)
    print("Testing Judge")
    print("=" * 60)
    
    judge = Judge(llm=None, use_llm_judge=False)
    
    test_cases = [
        # (generated, ground_truth, expected)
        ("Scott Derrickson", "Scott Derrickson", True),
        ("The director is Scott Derrickson", "Scott Derrickson", True),
        ("Yes, they are both American", "yes", True),
        ("No, they are different", "yes", False),
        ("I don't have enough information", "yes", False),
        ("I cannot determine the answer", "Scott", False),
        ("", "yes", False),
        ("American", "American", True),
    ]
    
    print("\nJudge test cases:")
    all_passed = True
    for generated, gt, expected in test_cases:
        result = judge(generated, gt)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} '{generated[:40]}...' vs '{gt}' -> {result} (expected {expected})")
    
    if all_passed:
        print("\n✓ All judge tests passed!")
    else:
        print("\n✗ Some judge tests failed!")
    
    print()


class MockModel:
    """Mock LLM/SLM for testing."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses: Dict[str, str] = responses or {}
        self.call_count = 0
    
    def generate(
        self, 
        question: str, 
        context_passages: List[Dict],
        prompt_template: Optional[str] = None
    ) -> Dict:
        self.call_count += 1
        
        # Check for specific responses
        for key, response in self.responses.items():
            if key.lower() in question.lower():
                return {"answer": response}
        
        # Default responses based on prompt type
        if "sub-questions" in question.lower() or "break down" in question.lower():
            return {"answer": "1. Who is X?\n2. What is Y?"}
        elif "analyze" in question.lower():
            return {"answer": "Based on the context, we can see that..."}
        else:
            return {"answer": "I don't have enough information."}


class MockRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, passages: Optional[List[Dict]] = None):
        self.passages: List[Dict] = passages or [
            {"title": "Test Doc 1", "text": "This is test document 1."},
            {"title": "Test Doc 2", "text": "This is test document 2."},
        ]
        self.call_count = 0
    
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Dict], List[float]]:
        self.call_count += 1
        return self.passages[:top_k], [1.0] * min(top_k, len(self.passages))


def test_green_search_basic():
    """Test GreenSearch with mock components."""
    print("=" * 60)
    print("Testing GreenSearch (basic)")
    print("=" * 60)
    
    # Create mocks that will return correct answer
    # The mock checks if the key appears anywhere in the prompt
    # The GreenSearch prompt includes "Question: Who directed Doctor Strange?"
    mock_slm = MockModel(responses={
        "doctor strange": "I don't know the answer.",
    })
    mock_llm = MockModel(responses={
        "doctor strange": "Scott Derrickson",  # Must contain ground truth exactly
    })
    mock_retriever = MockRetriever(passages=[
        {"title": "Scott Derrickson", "text": "Scott Derrickson is an American director known for Doctor Strange."},
    ])
    
    # Create searcher with low limits for fast testing
    searcher = GreenSearch(
        retriever=mock_retriever,
        slm=mock_slm,
        llm=mock_llm,
        max_depth=5,
        max_nodes=50,
    )
    
    # Run search
    print("\nSearching for: 'Who directed Doctor Strange?'")
    trajectory = searcher.search(
        query="Who directed Doctor Strange?",
        ground_truth="Scott Derrickson"
    )
    
    if trajectory:
        print(f"\n✓ Found solution!")
        print(f"  Total cost: {trajectory.total_cost:.4f}")
        print(f"  Search depth: {trajectory.search_depth}")
        print(f"  Nodes explored: {trajectory.nodes_explored}")
        print(f"  Answer: {trajectory.answer[:50]}...")
        print(f"\n  Steps:")
        for i, step in enumerate(trajectory.steps):
            print(f"    {i+1}. {step.action} (cost={step.cost:.4f})")
    else:
        print("\n✗ No solution found!")
    
    print(f"\n  Mock call counts: SLM={mock_slm.call_count}, LLM={mock_llm.call_count}, Retriever={mock_retriever.call_count}")
    print()


def test_cost_ordering():
    """Verify that search finds cheapest solution first."""
    print("=" * 60)
    print("Testing Cost Ordering")
    print("=" * 60)
    
    # LLM gives right answer immediately (cheapest path)
    # SLM gives wrong answer
    # The mock key must match something in the prompt
    
    mock_slm = MockModel(responses={"question": "Wrong answer"})
    mock_llm = MockModel(responses={"question": "Correct"})  # Contains ground truth
    mock_retriever = MockRetriever()
    
    searcher = GreenSearch(
        retriever=mock_retriever,
        slm=mock_slm,
        llm=mock_llm,
        max_depth=3,
        max_nodes=20,
    )
    
    trajectory = searcher.search(
        query="Test question?",
        ground_truth="Correct"
    )
    
    if trajectory:
        first_action = trajectory.steps[0].action if trajectory.steps else None
        print(f"\n  First successful action: {first_action}")
        print(f"  Total cost: {trajectory.total_cost:.4f}")
        
        # Verify it found the cheaper LLM path
        if first_action and first_action.model == Model.LLM:
            print("\n✓ Correctly found cheaper LLM path first!")
        else:
            print("\n  (Found a different path)")
    else:
        print("\n✗ No solution found")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GREENSEARCH TEST SUITE")
    print("=" * 60 + "\n")
    
    test_action_calls()
    test_search_node()
    test_judge()
    test_green_search_basic()
    test_cost_ordering()
    
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
