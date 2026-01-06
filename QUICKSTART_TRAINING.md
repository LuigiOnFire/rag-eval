# Quick Start: Special Token Decoder Training

This guide walks through training a decoder model with special action tokens from scratch.

## Prerequisites

```bash
# Install dependencies
pip install transformers datasets torch accelerate

# Verify data exists
ls data/sft_trajectories/sft_examples_20251218_173907_tokenized.json
```

## Step 1: Test Action Tokens (Optional)

Verify the special token system works:

```bash
python3 scripts/test_action_tokens.py
```

**Expected output:** All tests pass ✓

## Step 2: Review Training Data

Check the tokenized training data:

```bash
python3 -c "
import json
data = json.load(open('data/sft_trajectories/sft_examples_20251218_173907_tokenized.json'))
print(f\"Examples: {len(data['examples'])}\")
print(f\"Format: {data['metadata']['format_version']}\")
print(f\"Tokens: {data['metadata']['special_tokens'][:3]}...\")
"
```

**Expected:**
- 265 examples
- Format: special_tokens_v1
- Tokens: `<|retrieve|>`, `<|/retrieve|>`, ...

## Step 3: Train Model

Train a small decoder model (adjust based on your GPU):

### Option A: Small Model (Llama-3.2-1B) - Recommended for Testing

```bash
python3 scripts/train_decoder_sft.py \
    --model meta-llama/Llama-3.2-1B \
    --data data/sft_trajectories/sft_examples_20251218_173907_tokenized.json \
    --output models/decoder_sft_test \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --lr 2e-5
```

**Resources:**
- GPU Memory: ~10-12GB
- Training Time: ~30-45 minutes (3 epochs)
- Disk Space: ~2.5GB

### Option B: Larger Model (Llama-3.1-8B) - Production

```bash
python3 scripts/train_decoder_sft.py \
    --model meta-llama/Llama-3.1-8B \
    --data data/sft_trajectories/sft_examples_20251218_173907_tokenized.json \
    --output models/decoder_sft_8b \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --lr 1e-5
```

**Resources:**
- GPU Memory: ~24GB (single A100)
- Training Time: ~2-3 hours (3 epochs)
- Disk Space: ~16GB

### Training Output

```
Training Configuration:
  Base model: meta-llama/Llama-3.2-1B
  Data: data/sft_trajectories/...
  Output: models/decoder_sft_test
  Epochs: 3
  ...

Loading model and tokenizer...
Adding 10 special tokens...
  Added 10 new tokens

Loading training data...
  Loaded 265 examples
  
Preparing dataset...
  Tokenized 265 examples
  Train: 238, Eval: 27

Starting training...
[Epoch 1/3] ...
[Epoch 2/3] ...
[Epoch 3/3] ...

✓ Training complete!
✓ Model saved to: models/decoder_sft_test/final
✓ Config saved to: models/decoder_sft_test/training_config.json
```

## Step 4: Test Inference

Test the trained model:

```bash
python3 src/decoder_agent_with_actions.py \
    --model models/decoder_sft_test/final \
    --question "Were Scott Derrickson and Ed Wood of the same nationality?" \
    --passages data/processed/passages.json
```

**Expected behavior:**
1. Model generates decomposition: `<|decompose|>...`
2. Model generates retrieval: `<|retrieve|>Scott Derrickson nationality<|/retrieve|>`
3. Agent executes retrieval, injects documents
4. Model continues with extraction and answer
5. Final answer: "Yes" or "No"

**Example output:**
```
Question: Were Scott Derrickson and Ed Wood of the same nationality?

=== Actions Executed ===
1. decompose: 1. What is the nationality of Scott Derrickson?
2. What is the nationality of Ed Wood?
2. retrieve: What is the nationality of Scott Derrickson?
3. extract: Scott Derrickson is American
4. retrieve: What is the nationality of Ed Wood?
5. extract: Ed Wood was American
6. generate: Yes, both Scott Derrickson and Ed Wood were American

=== Final Answer ===
Yes, both Scott Derrickson and Ed Wood were American
```

## Step 5: Evaluate Performance

Create evaluation script to test on multiple questions:

```python
from decoder_agent_with_actions import DecoderAgentWithActions
from retriever import BM25Retriever
from generator import OllamaGenerator
import json

# Load test questions
with open('data/hotpotqa_validation_100.json') as f:
    questions = json.load(f)

# Setup agent
retriever = BM25Retriever("data/processed/passages.json")
generator = OllamaGenerator(model_name="mistral:latest")
agent = DecoderAgentWithActions(
    model_path="models/decoder_sft_test/final",
    retriever=retriever,
    generator=generator
)

# Evaluate
correct = 0
for q in questions:
    result = agent.run(q['question'])
    if result['final_answer'].lower() == q['answer'].lower():
        correct += 1

accuracy = correct / len(questions)
print(f"Accuracy: {accuracy:.1%} ({correct}/{len(questions)})")
```

## Troubleshooting

### Out of Memory

Reduce batch size or enable gradient checkpointing:

```bash
python3 scripts/train_decoder_sft.py \
    --batch-size 2 \
    --gradient-accumulation 8
```

### Model Not Generating Action Tokens

Check if special tokens were added:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("models/decoder_sft_test/final")
print(tokenizer.special_tokens_map)
print(len(tokenizer))  # Should be base_vocab_size + 10
```

### Retrieval Not Working

Verify BM25 index exists:

```bash
python3 -c "
from retriever import BM25Retriever
r = BM25Retriever('data/processed/passages.json')
docs = r.retrieve('Scott Derrickson', top_k=3)
print(f'Retrieved {len(docs)} documents')
print(docs[0].content[:100])
"
```

## Next Steps

1. **Generate More Data**: Run `scripts/generate_structured_trajectories.py` with `--num-samples 500`
2. **Retrain**: Use larger dataset for better performance
3. **Compare Baselines**: Evaluate against encoder-decoder baseline (42% accuracy target)
4. **Measure Energy**: Use `evaluation/energy.py` to track power consumption
5. **Optimize**: Try different hyperparameters (learning rate, epochs, model size)

## Files Generated

After completing this guide:

- `models/decoder_sft_test/final/` - Trained model (2.5GB)
- `models/decoder_sft_test/training_config.json` - Training config
- `models/decoder_sft_test/logs/` - TensorBoard logs

## Performance Targets

- **Baseline (Structured Agent)**: 42% accuracy, 71.4% on comparison questions
- **Target**: Match or exceed baseline accuracy
- **Stretch Goal**: 70%+ overall accuracy

## References

- Training data: [data/sft_trajectories/sft_examples_20251218_173907_tokenized.json](data/sft_trajectories/sft_examples_20251218_173907_tokenized.json)
- Architecture doc: [SPECIAL_TOKEN_PIPELINE.md](SPECIAL_TOKEN_PIPELINE.md)
- Action tokens: [src/action_tokens.py](src/action_tokens.py)
