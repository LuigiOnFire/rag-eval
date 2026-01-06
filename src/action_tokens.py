"""
Special tokens configuration for decoder-based RAG agent.

Defines action tokens that trigger retrieval, generation, etc.
"""

# Special tokens for action triggering
ACTION_TOKENS = {
    'retrieve_start': '<|retrieve|>',
    'retrieve_end': '<|/retrieve|>',
    'generate_start': '<|generate|>',
    'generate_end': '<|/generate|>',
    'decompose_start': '<|decompose|>',
    'decompose_end': '<|/decompose|>',
    'reason_start': '<|reason|>',
    'reason_end': '<|/reason|>',
    'extract_start': '<|extract|>',
    'extract_end': '<|/extract|>',
}

# All special tokens as a list
ALL_ACTION_TOKENS = list(ACTION_TOKENS.values())


def add_special_tokens_to_tokenizer(tokenizer):
    """
    Add action special tokens to a tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Number of tokens added
    """
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': ALL_ACTION_TOKENS
    })
    return num_added


def format_action_with_tokens(action: str, content: str = None) -> str:
    """
    Format an action with special tokens.
    
    Args:
        action: Action type (e.g., 'retrieve', 'generate')
        content: Optional content/parameter for the action
        
    Returns:
        Formatted string with special tokens
        
    Examples:
        >>> format_action_with_tokens('retrieve', 'Shirley Temple government')
        '<|retrieve|>Shirley Temple government<|/retrieve|>'
        
        >>> format_action_with_tokens('generate')
        '<|generate|>'
    """
    action_lower = action.lower()
    
    if action_lower not in ['retrieve', 'generate', 'decompose', 'reason', 'extract']:
        raise ValueError(f"Unknown action: {action}")
    
    start_token = ACTION_TOKENS[f'{action_lower}_start']
    end_token = ACTION_TOKENS[f'{action_lower}_end']
    
    if content:
        return f"{start_token}{content}{end_token}"
    else:
        return start_token


def parse_action_tokens(text: str):
    """
    Parse special tokens from text to extract actions.
    
    Args:
        text: Text containing special tokens
        
    Returns:
        List of (action_type, content) tuples
        
    Examples:
        >>> parse_action_tokens('Let me search <|retrieve|>Ed Wood filmmaker<|/retrieve|>')
        [('retrieve', 'Ed Wood filmmaker')]
    """
    import re
    
    actions = []
    
    # Pattern to match any action token pair
    pattern = r'<\|(\w+)\|>(.*?)<\|/\1\|>'
    
    for match in re.finditer(pattern, text, re.DOTALL):
        action_type = match.group(1)
        content = match.group(2).strip()
        actions.append((action_type, content))
    
    return actions


def convert_trajectory_to_token_format(trajectory: dict) -> dict:
    """
    Convert a trajectory from text format to special token format.
    
    Args:
        trajectory: Trajectory dict with steps
        
    Returns:
        Updated trajectory with special tokens
    """
    for step in trajectory.get('steps', []):
        action = step.get('action', '').lower()
        
        # Map old action names to new token format
        action_map = {
            'retrieve': 'retrieve',
            'retrieve_keyword': 'retrieve',
            'retrieve_dense': 'retrieve',
            'generate_slm': 'generate',
            'generate_llm': 'generate',
            'generate_final': 'generate',
            'decompose': 'decompose',
            'decompose_slm': 'decompose',
            'reason': 'reason',
            'reason_slm': 'reason',
            'extract': 'extract',
        }
        
        mapped_action = action_map.get(action)
        if mapped_action:
            # Update output to use special tokens
            original_output = step.get('output', '')
            step['output'] = format_action_with_tokens(mapped_action, original_output)
    
    return trajectory
