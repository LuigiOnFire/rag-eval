# RAG Baseline Pipeline - Implementation Plan

## Proposed Project Structure

```
rag_eval/
├── data/
│   ├── raw/              # Wikipedia subset
│   ├── processed/        # Chunked passages
│   └── indexes/          # Faiss indexes
├── src/
│   ├── corpus.py         # Corpus loading and chunking
│   ├── retriever.py      # Embedding + Faiss retrieval
│   ├── generator.py      # LLM generation
│   ├── pipeline.py       # End-to-end RAG pipeline
│   └── evaluator.py      # RAGChecker integration
├── experiments/
│   └── logs/             # Evaluation results
├── notebooks/            # Analysis and debugging
├── requirements.txt
└── config.yaml           # Centralized configuration
```

## Questions Before Implementation

To build the right baseline for you, I need to know:

### 1. Corpus
Do you already have a Wikipedia subset, or should I include code to download one (e.g., via HuggingFace datasets)?

**Options:**
- Download programmatically (e.g., `wikipedia` package, HF `datasets`)
- Point to existing local files
- Start with a small subset (e.g., 10k articles) for testing

### 2. Embedding Model
Which SentenceTransformer model do you prefer?

**Options:**
- `all-MiniLM-L6-v2` - Fast, lightweight (80MB), good for prototyping
- `all-mpnet-base-v2` - Better quality, slower (420MB)
- `sentence-transformers/all-MiniLM-L12-v2` - Middle ground
- Custom model from HuggingFace Hub

### 3. Generator (LLM)
Which free LLM should we use?

**Local Options:**
- `mistralai/Mistral-7B-Instruct-v0.2` - Popular, well-tuned
- `meta-llama/Llama-3.2-3B-Instruct` - Smaller, faster
- `google/flan-t5-large` - Efficient, good for Q&A

**API Options:**
- Google Gemini 1.5 Flash (free tier, 15 RPM, 1M TPM)
- Gemini 1.5 Pro (free tier, 2 RPM, 32k TPM)

### 4. Chunking Strategy
How should we split documents?

**Options:**
- Fixed-size chunks (e.g., 256, 512 tokens) with overlap (e.g., 50 tokens)
- Sentence-based chunking (preserve semantic boundaries)
- Paragraph-based chunking
- Recursive character splitting (LangChain-style)

### 5. Evaluation Dataset
Do you have query-answer pairs, or should we generate them?

**Options:**
- Use existing benchmark (e.g., Natural Questions, MS MARCO)
- Generate synthetic questions from corpus (using LLM)
- Manually create a small test set (10-50 queries)
- Use RAGChecker's built-in benchmark data

### 6. RAGChecker Integration
Are you using the official implementation from Amazon Science, or a pip-installable version?

**Notes:**
- I see `RAGChecker/` folder in your workspace
- Need to confirm if we're using this local version or installing via pip
- Check if it's already set up or needs configuration

## Implementation Phases

### Phase 1: Core Components (Minimal Viable Pipeline)
1. **requirements.txt** - Pinned dependencies
2. **config.yaml** - Centralized configuration
3. **corpus.py** - Load and chunk documents
4. **retriever.py** - Embed + Faiss index + search
5. **generator.py** - LLM wrapper with prompt templates
6. **pipeline.py** - Orchestrate retrieval → generation

### Phase 2: Evaluation
7. **evaluator.py** - RAGChecker integration
8. **experiments/run_baseline.py** - End-to-end evaluation script
9. Logging and metrics tracking

### Phase 3: Utilities & Debugging
10. **notebooks/explore_corpus.ipynb** - Data exploration
11. **notebooks/debug_retrieval.ipynb** - Check retrieval quality
12. **tests/** - Unit tests for core components
13. **scripts/** - Helper scripts (download data, build index, etc.)

## Design Principles

### Modularity
- Each component (retriever, generator, evaluator) is independent
- Clear interfaces between modules
- Easy to swap implementations (e.g., different embedding models)

### Reproducibility
- Fixed random seeds
- Deterministic chunking and indexing
- Version-pinned dependencies
- Configuration file for all hyperparameters

### Debuggability
- Extensive logging (retrieval scores, generated text, metrics)
- Intermediate outputs saved (chunks, embeddings, Faiss index)
- Easy to inspect pipeline at each stage

### Extensibility
- Ready for adaptive RAG experiments
- Pluggable components (e.g., add reranker, query expansion)
- Support for multiple generators/retrievers

## Key Features to Include

### Retriever
- Top-k retrieval (configurable k)
- Score normalization
- Optional metadata filtering
- Cache embeddings to avoid recomputation

### Generator
- Prompt template system (easy to modify)
- Temperature control (deterministic by default)
- Max tokens configuration
- Optional streaming support

### Pipeline
- Batch processing support
- Error handling and retries
- Progress tracking
- Save intermediate results

### Evaluation
- RAGChecker metrics (precision, recall, hallucination)
- Retrieval metrics (MRR, Recall@k, NDCG)
- Generation metrics (BLEU, ROUGE, BERTScore)
- Per-query breakdown for analysis

## Immediate Next Steps

Once you provide answers to the 6 questions above, I will:

1. Create **requirements.txt** with pinned versions
2. Create **config.yaml** with sensible defaults
3. Implement **corpus.py** with your chosen chunking strategy
4. Implement **retriever.py** with Faiss + your embedding model
5. Implement **generator.py** wrapper for your chosen LLM
6. Implement **pipeline.py** orchestrating the full flow
7. Implement **evaluator.py** integrating RAGChecker
8. Create **experiments/run_baseline.py** for end-to-end evaluation

## Alternative: Minimal Working Example First

If you prefer, I can start with a **single-file prototype** (~200 lines) that demonstrates the complete pipeline before building the modular structure. This would let you:

- Test the approach quickly
- Validate choices (embedding model, generator, etc.)
- Then refactor into clean modules

Let me know your preference!
