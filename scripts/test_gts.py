#!/usr/bin/env python3
"""Quick test of GreenTreeSearch module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.green_tree_search import (
    Action, ActionType, Model, RetrievalMethod, ActionCall,
    GENERATE, DECOMPOSE, RETRIEVE, REASON,
    load_cost_table, create_simple_judge
)

print("Testing GreenTreeSearch module...")

# Test legacy Action enum
print("\n1. Legacy Action enum (for neural network):")
for a in Action:
    print(f"  {a.value}: {a.name}")

# Test new parameterized ActionCall
print("\n2. New ActionCall system:")
actions = [
    GENERATE(model="slm"),
    GENERATE(model="llm"),
    DECOMPOSE(model="slm"),
    RETRIEVE(method="keyword", query="Who directed Sinister?"),
    RETRIEVE(method="dense", query="film nationality"),
    REASON(model="llm"),
]
for action in actions:
    print(f"  {action} -> legacy_id={action.legacy_action_id}")

# Test cost table loading
print("\n3. Cost table (sorted by cost):")
cost_table = load_cost_table('results/cost_table.json')
sorted_actions = sorted(cost_table.items(), key=lambda x: x[1])
for action_id, cost in sorted_actions:
    print(f"  {Action(action_id).name}: {cost:.6f} Wh ({cost * 3600:.2f} J)")

# Test judge
print("\n4. Judge tests:")
judge = create_simple_judge(exact_match=False)
print(f"  'The director is Scott Derrickson' contains 'Scott Derrickson': {judge('The director is Scott Derrickson', 'Scott Derrickson')}")
print(f"  'Paris' contains 'London': {judge('Paris', 'London')}")

print("\nAll tests passed!")
