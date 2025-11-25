# RAG Baseline Pipeline

A modular baseline RAG pipeline for hallucination detection research using:
- **Retriever**: Faiss + SentenceTransformers (all-MiniLM-L6-v2)
- **Generator**: Google Gemini API or Ollama (local models)
- **Evaluation**: Amazon RAGChecker + HotpotQA dataset

## Project Structure

```
rag_eval/
├── src/
│   ├── corpus.py         # Wikipedia corpus loading and sentence-based chunking
│   ├── retriever.py      # Faiss vector search with embeddings
│   ├── generator.py      # Generator supporting Gemini API and Ollama
│   ├── pipeline.py       # End-to-end RAG orchestration
│   └── evaluator.py      # RAGChecker evaluation
├── experiments/
│   ├── run_baseline.py   # Main evaluation script
│   └── logs/             # Evaluation results
├── data/
│   ├── raw/              # Cached datasets
│   ├── processed/        # Chunked passages
│   └── indexes/          # Faiss indexes
├── config.yaml           # Configuration (Gemini API)
├── config_local.yaml     # Configuration (Ollama local models)
├── .env                  # API keys (create from .env.example)
├── OLLAMA_SETUP.md       # Guide for setting up local models
└── requirements.txt      # Dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### 2. Choose Generator Backend

**Option A: Gemini API (Cloud)**

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

**Option B: Ollama (Local GPU)**

See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed instructions:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull a model (Mistral 7B recommended for L40 GPU)
ollama pull mistral
```

**Secure API Key Management:**
- Store keys in `.env` file (never commit to git)
- Add `.env` to `.gitignore`
- Use environment variables only
- For shared environments, use secret management tools (e.g., AWS Secrets Manager)

### 3. Configure Pipeline

**For Gemini API**: Edit `config.yaml`

**For Ollama (local)**: Edit `config_local.yaml`

Both configs allow you to adjust:
- Corpus size (`corpus.num_documents`)
- Retrieval parameters (`retrieval.top_k`)
- Generation settings (`generator.temperature`, `generator.max_tokens`)
- Evaluation dataset and metrics

## Usage

### Quick Start: Run Full Evaluation

**With Gemini API:**
```bash
cd experiments
python run_baseline.py
```

**With Ollama (local models):**
```bash
cd experiments
python run_baseline.py --config ../config_local.yaml
```

This will:
1. Download Wikipedia subset (10k documents)
2. Chunk into passages using sentence-based splitting
3. Embed passages with all-MiniLM-L6-v2
4. Build Faiss index
5. Load HotpotQA evaluation dataset (100 questions)
6. Run retrieval → generation pipeline
7. Evaluate with RAGChecker
8. Save results to `experiments/logs/`

### Step-by-Step Usage

**Prepare corpus only:**
```bash
python run_baseline.py --skip-eval
```

**Rebuild corpus:**
```bash
python run_baseline.py --rebuild-corpus
```

**Rebuild index:**
```bash
python run_baseline.py --rebuild-index
```

**Adjust API rate limiting:**
```bash
python run_baseline.py --delay 2.0  # 2 second delay between calls
```

### Individual Modules

**Process corpus:**
```bash
cd src
python corpus.py
```

**Build index:**
```bash
python retriever.py
```

**Test generation:**
```bash
python generator.py
```

**Test pipeline:**
```bash
python pipeline.py
```

**Run evaluation:**
```bash
python evaluator.py
```

## Configuration

Key settings in `config.yaml`:

```yaml
# Corpus
corpus:
  num_documents: 10000  # Start small, increase for better coverage

# Chunking
chunking:
  max_chunk_size: 512   # Tokens per chunk
  overlap: 50           # Overlap for context

# Retrieval
retrieval:
  top_k: 5              # Number of passages to retrieve

# Generation
generator:
  temperature: 0.0      # Deterministic (0.0) or creative (0.7+)
  max_tokens: 256       # Max answer length

# Evaluation
evaluation:
  num_samples: 100      # Questions to evaluate
```

## Output

Evaluation results saved to `experiments/logs/evaluation_YYYYMMDD_HHMMSS.json`:

```json
{
  "metadata": {
    "timestamp": "2025-11-14T...",
    "num_evaluated": 100,
    "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
    "generator_model": "gemini-1.5-flash"
  },
  "metrics": {
    "precision": 0.XX,
    "recall": 0.XX,
    "f1": 0.XX,
    "hallucination_rate": 0.XX
  },
  "detailed_results": [...]
}
```

## Troubleshooting

**Import errors:**
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

**API key error:**
- Check `.env` file exists and contains `GEMINI_API_KEY=...`
- Verify key is valid at https://aistudio.google.com/app/apikey

**Rate limiting:**
- Increase `--delay` parameter
- Reduce `evaluation.num_samples` in config
- Gemini free tier: 15 requests/minute, 1M tokens/minute

**Memory issues:**
- Reduce `corpus.num_documents`
- Use `faiss-cpu` instead of `faiss-gpu`
- Process in smaller batches

**Index not found:**
- Run `python run_baseline.py --rebuild-index`
- Or run modules individually (corpus.py → retriever.py)

## Next Steps

### Adaptive RAG Extensions

1. **Query Analysis**: Classify query complexity to determine retrieval strategy
2. **Iterative Retrieval**: Multi-hop retrieval for complex questions
3. **Self-Reflection**: LLM evaluates its own answer and retrieves more if uncertain
4. **Reranking**: Add cross-encoder reranker after initial retrieval
5. **Hallucination Detection**: Use RAGChecker metrics to trigger re-retrieval

### Experimentation

- Compare embedding models (all-mpnet-base-v2 vs all-MiniLM-L6-v2)
- Test different chunk sizes and overlap strategies
- Evaluate local LLMs (Mistral, Llama) vs Gemini
- Benchmark on multiple datasets (NQ, TriviaQA, etc.)
- Tune retrieval threshold and top-k

## Citation

If using RAGChecker, cite:
```
@article{ragchecker2024,
  title={RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```
