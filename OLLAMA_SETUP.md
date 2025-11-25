# Ollama Setup Guide

This guide helps you set up Ollama for local model inference on your L40 GPU.

## Installation

### 1. Install Ollama

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Verify Installation

```bash
ollama --version
```

### 3. Start Ollama Server

```bash
# Start Ollama in the background
ollama serve
```

Or run it in a separate terminal window to see logs.

## Download Models

### Recommended Models for L40 GPU (46GB VRAM)

```bash
# Mistral 7B (~4.1GB) - Fast and efficient
ollama pull mistral

# Llama 3.1 8B (~4.7GB) - Good balance
ollama pull llama3.1:8b

# Llama 3.1 70B (~40GB) - High quality but slower
ollama pull llama3.1:70b
```

### Check Downloaded Models

```bash
ollama list
```

## Test Ollama

```bash
# Test with a simple prompt
ollama run mistral "What is machine learning?"
```

## Running RAG Pipeline with Ollama

### 1. Use Local Config

```bash
cd /home/wcrawford/rag_eval/experiments
python run_baseline.py --config ../config_local.yaml --test --test-samples 3
```

### 2. Configuration Options

Edit `config_local.yaml` to switch models:

```yaml
generator:
  provider: "ollama"
  model_name: "ollama/mistral"      # Use Mistral 7B
  # model_name: "ollama/llama3.1:8b"  # Or Llama 3.1 8B
  # model_name: "ollama/llama3.1:70b" # Or Llama 3.1 70B (slower)
  base_url: "http://localhost:11434"
  timeout: 120  # Higher timeout for larger models
```

## Performance Comparison

| Model | VRAM | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Mistral 7B | ~4GB | Fast | Good | Quick iteration, testing |
| Llama 3.1 8B | ~5GB | Fast | Very Good | Production baseline |
| Llama 3.1 70B | ~40GB | Slow | Excellent | High-quality evaluation |

## Troubleshooting

### Ollama Not Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull the model if missing
ollama pull mistral
```

### Out of Memory

```bash
# Use a smaller model
ollama pull mistral  # Instead of llama3.1:70b
```

### Slow Generation

- Use smaller model (mistral or llama3.1:8b)
- Reduce `max_tokens` in config_local.yaml
- Ensure GPU is being used (check `nvidia-smi` during generation)

## API Reference

Ollama runs on `http://localhost:11434` by default with these endpoints:

- `GET /api/tags` - List models
- `POST /api/generate` - Generate text (streaming)
- `POST /api/chat` - Chat completions (used by our pipeline)
- `POST /api/embeddings` - Generate embeddings

## Next Steps

1. Start Ollama: `ollama serve`
2. Pull a model: `ollama pull mistral`
3. Test pipeline: `cd experiments && python run_baseline.py --config ../config_local.yaml --test`
4. Monitor GPU: `watch -n 1 nvidia-smi`

## Switching Back to Gemini

To switch back to Gemini API:

```bash
python run_baseline.py --config ../config.yaml --test
```

The `--config` flag allows you to easily switch between local and API-based models.
