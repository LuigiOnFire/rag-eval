# Quick Testing Guide

## Test Mode for Fast Iteration

When debugging RAGChecker integration or testing changes, use test mode to run on only 5 questions instead of 100.

### Running Evaluator in Test Mode

```bash
cd src
python evaluator.py --test
```

With custom sample size:
```bash
python evaluator.py --test --test-samples 3
```

### Running Full Pipeline in Test Mode

```bash
cd experiments
python run_baseline.py --test
```

With custom delay and sample size:
```bash
python run_baseline.py --test --test-samples 3 --delay 2.0
```

### Full Pipeline Options

```bash
# Test mode (5 samples, fast)
python run_baseline.py --test

# Production mode (100 samples from config)
python run_baseline.py

# Rebuild everything and test
python run_baseline.py --test --rebuild-corpus --rebuild-index

# Skip evaluation, just prepare data
python run_baseline.py --skip-eval
```

## Command Summary

| Command | Purpose |
|---------|---------|
| `--test` | Use 5 samples instead of 100 |
| `--test-samples N` | Use N samples in test mode |
| `--delay 2.0` | Wait 2 seconds between API calls |
| `--rebuild-corpus` | Force re-download Wikipedia data |
| `--rebuild-index` | Force rebuild Faiss index |
| `--skip-eval` | Only prepare data, don't run evaluation |

## Typical Workflow

### 1. Initial Setup (One-time)
```bash
# Prepare corpus and index
python run_baseline.py --skip-eval
```

### 2. Quick Testing (Debugging)
```bash
# Test with 3 questions
python run_baseline.py --test --test-samples 3
```

### 3. Full Evaluation (Production)
```bash
# Run full 100 questions
python run_baseline.py
```

## Expected Execution Times

| Mode | Samples | Estimated Time |
|------|---------|----------------|
| Test (3 samples) | 3 | ~30 seconds |
| Test (5 samples) | 5 | ~1 minute |
| Full (100 samples) | 100 | ~20-30 minutes |

*Times assume 15 RPM rate limit with 1s delay between calls*

## Troubleshooting

### Rate Limit Errors
- Increase `--delay` (e.g., `--delay 5.0`)
- Use `--test` mode with fewer samples
- The generator now auto-retries with cooldown periods

### Out of Memory
- Reduce corpus size in `config.yaml`: `num_documents: 500`
- Reduce batch size: `batch_size: 16`
- Use `faiss-cpu` instead of `faiss-gpu`

### All Questions Skipped
- Check API key in `.env` file
- Verify Gemini model name in config: `gemini-2.0-flash`
- Check logs for specific error messages
