# Decoder-Based Agent Architecture

## Summary

We're pivoting from an **encoder classifier** (RoBERTa predicting 1-of-8 action classes) to a **decoder agent** (LLM generating actions with parameters).

## Key Differences

| Aspect | Encoder (Old) | Decoder (New) |
|--------|---------------|---------------|
| **Action Selection** | Classify into 8 fixed classes | Generate action + parameters |
| **Retrieval Query** | Implicit (use original query) | Explicit: `RETRIEVE["specific query"]` |
| **Context Window** | 512 tokens (compressed) | 4K-8K tokens (full context) |
| **Reasoning** | Hidden in compressed obs | Explicit chain-of-thought |
| **Training** | Classification loss | SFT on action sequences |
| **Flexibility** | Fixed action space | Natural language actions |

## Action Space (8 Actions)

```
# Retrieval (parameterized - the key improvement)
Action: RETRIEVE_KEYWORD["query"]     # BM25 search with specific query
Action: RETRIEVE_DENSE["query"]       # Dense retrieval with specific query  

# Decomposition (break multi-hop into sub-questions)
Action: DECOMPOSE_SLM                 # Cheap decomposition (Mistral)
Action: DECOMPOSE_LLM                 # Quality decomposition (Llama-8B)

# Reasoning (synthesize from retrieved context)
Action: REASON_SLM                    # Cheap synthesis
Action: REASON_LLM                    # Quality synthesis

# Generation (terminal - produce final answer)
Action: GENERATE_SLM                  # Cheap final answer
Action: GENERATE_LLM                  # Quality final answer
```

**Design Rationale:**
- **Parameterized retrieval**: Controller generates targeted search queries (decoder advantage)
- **SLM/LLM variants**: Controller learns energy-aware routing (encoder advantage preserved)
- **Clear cost model**: Each action has measurable energy for reward shaping

## State Structure

With larger context, we can structure state richly:

```
=== QUERY ===
{original question}

=== SUB-QUESTIONS ===
Q1: {sub-question 1}
  Status: ANSWERED
  Answer: {answer}
  Evidence: {doc excerpt}

Q2: {sub-question 2}
  Status: SEARCHING
  Attempts: 2

=== RETRIEVED CONTEXT ===
[Doc Title 1]: {content...}
[Doc Title 2]: {content...}

=== REASONING TRACE ===
- Identified multi-hop structure
- Found entity X
- Need to find relationship Y

=== NEXT ACTION ===
```

## Training Pipeline

### Phase 1: Trajectory Generation
1. Run decoder agent with strong LLM (GPT-4 or llama3:8b)
2. Agent sees full context, makes decisions
3. Record successful trajectories: `[(state, action, result), ...]`

### Phase 2: Supervised Fine-Tuning (SFT)
1. Convert trajectories to (input, output) pairs
2. Input: state formatted as prompt
3. Output: `Thought: ... Action: ACTION["param"]`
4. Fine-tune smaller LLM (Phi-3, Qwen-3B) on this data

### Phase 3: PPO Refinement (Optional)
1. Use SFT model as policy
2. Reward: correctness - energy cost
3. Learn to avoid expensive dead-ends

## Benefits of Decoder Approach

1. **Parameterized Retrieval**: Model generates targeted queries instead of searching with original question
   - Old: `RETRIEVE_KEYWORD` → searches "What government position was held by..."
   - New: `RETRIEVE_KEYWORD["Shirley Temple Chief of Protocol"]` → targeted search

2. **Multi-Hop Reasoning**: Natural decomposition with explicit sub-questions
   ```
   Action: DECOMPOSE_LLM
   → Worker returns: ["Who played Aunt May in Spider-Man 3?", "What government position did they hold?"]
   ```

3. **Recovery from Failures**: Can reformulate queries when retrieval fails
   ```
   Thought: Retrieval for "Rosemary Harris government" found nothing. 
   Let me try a different approach.
   Action: RETRIEVE_KEYWORD["Spider-Man Aunt May actress political career"]
   ```

4. **Richer Training Signal**: Each step has reasoning + action, not just class label

## Energy Tracking

Measured cost per action (from our benchmarks):

| Action | Energy (mWh) | Notes |
|--------|-------------|-------|
| RETRIEVE_KEYWORD | ~9 | BM25 search |
| RETRIEVE_DENSE | ~9 | Dense retrieval |
| GENERATE_LLM | ~15 | llama3:8b |
| GENERATE_SLM | ~20 | mistral:latest |
| DECOMPOSE_SLM | ~24 | mistral decomposition |
| REASON_LLM | ~27 | llama3:8b reasoning |
| DECOMPOSE_LLM | ~28 | llama3 decomposition |
| REASON_SLM | ~65 | mistral reasoning |

The SLM/LLM choice lets the model learn when cheap is good enough.

## Files

- [src/decoder_agent.py](src/decoder_agent.py) - Main agent implementation
- [scripts/test_decoder_agent.py](scripts/test_decoder_agent.py) - Tests and examples
- `scripts/generate_decoder_trajectories.py` - (TODO) Trajectory generation
- `scripts/train_decoder_sft.py` - (TODO) SFT training

## Next Steps

1. **Generate trajectories** with the decoder agent on HotPotQA
2. **Analyze success rate** - compare to encoder approach (60% success)
3. **SFT training** on successful trajectories
4. **PPO refinement** for energy-aware optimization
