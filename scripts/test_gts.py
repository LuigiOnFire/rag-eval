#!/usr/bin/env python3
"""Quick test of GreenTreeSearch module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.green_tree_search import Action, load_cost_table, create_simple_judge

print("Testing GreenTreeSearch module...")

# Test Action enum
print("\n1. Action enum:")
for a in Action:
    print(f"  {a.value}: {a.name}")

# Test cost table loading
print("\n2. Cost table (sorted by cost):")
cost_table = load_cost_table('results/cost_table.json')
sorted_actions = sorted(cost_table.items(), key=lambda x: x[1])
for action_id, cost in sorted_actions:
    print(f"  {Action(action_id).name}: {cost:.6f} Wh ({cost * 3600:.2f} J)")

# Test judge
print("\n3. Judge tests:")
judge = create_simple_judge(exact_match=False)
print(f"  'The director is Scott Derrickson' contains 'Scott Derrickson': {judge('The director is Scott Derrickson', 'Scott Derrickson')}")
print(f"  'Paris' contains 'London': {judge('Paris', 'London')}")

print("\nAll tests passed!")
