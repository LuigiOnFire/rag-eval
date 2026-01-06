#!/usr/bin/env python3
"""
Test action token parsing and formatting.

Quick test script to verify special token functionality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from action_tokens import (
    format_action_with_tokens,
    parse_action_tokens,
    ACTION_TOKENS,
    ALL_ACTION_TOKENS
)


def test_formatting():
    """Test formatting text with action tokens."""
    print("=" * 60)
    print("TEST: Formatting with action tokens")
    print("=" * 60)
    
    # Test each action type
    test_cases = [
        ('retrieve', 'What is the nationality of Scott Derrickson?'),
        ('extract', 'Scott Derrickson is American'),
        ('generate', 'Yes, both are American'),
        ('decompose', '1. What is X?\n2. What is Y?'),
        ('reason', 'This question requires comparing two facts'),
    ]
    
    for action, content in test_cases:
        formatted = format_action_with_tokens(action, content)
        print(f"\n{action.upper()}:")
        print(f"  Input:  {content[:60]}...")
        print(f"  Output: {formatted[:80]}...")


def test_parsing():
    """Test parsing action tokens from text."""
    print("\n\n" + "=" * 60)
    print("TEST: Parsing action tokens")
    print("=" * 60)
    
    # Test text with multiple actions
    text = """Question: Were Scott Derrickson and Ed Wood of the same nationality?

<|decompose|>
1. What is the nationality of Scott Derrickson?
2. What is the nationality of Ed Wood?
<|/decompose|>

I'll search for: <|retrieve|>What is the nationality of Scott Derrickson?<|/retrieve|>

Retrieved documents:
Document 1: Scott Derrickson is an American director...

<|extract|>Scott Derrickson is American<|/extract|>

Now searching for: <|retrieve|>What is the nationality of Ed Wood?<|/retrieve|>

Retrieved documents:
Document 1: Ed Wood was an American filmmaker...

<|extract|>Ed Wood was American<|/extract|>

Final answer: <|generate|>Yes, both Scott Derrickson and Ed Wood were American<|/generate|>"""
    
    print(f"\nInput text ({len(text)} chars):")
    print(text[:200] + "...")
    
    # Parse
    actions = parse_action_tokens(text)
    
    print(f"\nParsed {len(actions)} actions:")
    for i, (action_type, content) in enumerate(actions, 1):
        print(f"\n{i}. {action_type.upper()}:")
        print(f"   {content[:100]}...")


def test_roundtrip():
    """Test roundtrip: format → parse → verify."""
    print("\n\n" + "=" * 60)
    print("TEST: Roundtrip (format → parse)")
    print("=" * 60)
    
    test_content = "What is the capital of France?"
    action = "retrieve"
    
    # Format
    formatted = format_action_with_tokens(action, test_content)
    print(f"\nOriginal content: {test_content}")
    print(f"Formatted: {formatted}")
    
    # Parse
    parsed = parse_action_tokens(formatted)
    
    # Verify
    assert len(parsed) == 1, f"Expected 1 action, got {len(parsed)}"
    assert parsed[0][0] == action, f"Expected action '{action}', got '{parsed[0][0]}'"
    assert parsed[0][1] == test_content, f"Content mismatch"
    
    print(f"Parsed back: action={parsed[0][0]}, content={parsed[0][1]}")
    print("✓ Roundtrip successful!")


def test_token_list():
    """Test token list is complete."""
    print("\n\n" + "=" * 60)
    print("TEST: Token definitions")
    print("=" * 60)
    
    print(f"\nDefined {len(ACTION_TOKENS)} token pairs:")
    for key, value in ACTION_TOKENS.items():
        print(f"  {key:20s} → {value}")
    
    print(f"\nAll tokens list ({len(ALL_ACTION_TOKENS)} tokens):")
    for token in ALL_ACTION_TOKENS:
        print(f"  {token}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ACTION TOKEN TEST SUITE")
    print("=" * 60)
    
    test_formatting()
    test_parsing()
    test_roundtrip()
    test_token_list()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
