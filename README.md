# Energy-Aware Adaptive RAG

A research project exploring **compute-driven discovery of optimal RAG strategies** through reinforcement learning. Instead of hand-crafted routing rules, we train a small controller to dynamically decide *when to retrieve*, *which model to use*, and *when to stop* — optimizing for both accuracy and energy efficiency.

## Research Vision

### The Problem
RAG systems face a fundamental cost-quality tradeoff:
- **Retrieve too little** → hallucination
- **Retrieve too much** → slow, expensive, and noisy context
- **Use a big model** → accurate but costly
- **Use a small model** → fast but error-prone

Current solutions rely on **human-designed heuristics** (query complexity rules, confidence thresholds). We take a different approach inspired by the "Bitter Lesson": **let compute discover the optimal policy**.

### The Solution: Green-DeepRAG
A sequential decision agent that learns to route queries through the cheapest successful path:

```
Query → [Controller] → <RETRIEVE>? → <ASSIGN_SLM>/<ASSIGN_LLM>? → <STOP>
              ↓
        Tiny Decoder (1B params)
        Trained via RL to minimize: Energy + Maximize: Accuracy
```

**Key Insight**: Most queries don't need expensive retrieval + large LLM. A small model can handle simple factual questions; retrieval is only needed for knowledge-intensive queries; large models are reserved for complex reasoning.

## Architecture

### Components

| Component | Role | Example |
|-----------|------|---------|
| **Controller** | Tiny decoder that emits control tokens | Qwen-2.5-0.5B, SmolLM-1.7B |
| **SLM Worker** | Fast, cheap generation | Mistral-7B, Llama-3-8B |
| **LLM Worker** | Expensive, accurate generation | Llama-3-70B, GPT-4o |
| **Retriever** | BM25 keyword search | rank_bm25 |
| **Judge** | Validates answer correctness | Exact match + LLM-judge |

### Action Space
The controller generates control tokens to orchestrate the pipeline:
- `<RETRIEVE>` — Call the retriever
- `<ASSIGN_SLM>` — Generate with small model
- `<ASSIGN_LLM>` — Generate with large model  
- `<STOP>` — Emit final answer

### Training Pipeline (3 Phases)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Cost-Ordered Search (Offline Oracle)                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  For each query, try paths in ascending cost order:                         │
│    1. SLM Direct (cost=1) → if correct, save trace                          │
│    2. Retrieve + SLM (cost=6) → if correct, save trace                      │
│    3. LLM Direct (cost=20) → if correct, save trace                         │
│    4. Retrieve + LLM (cost=25) → fallback                                   │
│  Output: Dataset of (query → cheapest successful trajectory)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: Behavior Cloning (Warm Start)                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Supervised fine-tuning of Controller on Phase 1 traces                     │
│  Output: Policy that mimics "cheapest winner" heuristic                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: Cost-Aware PPO (Refinement)                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Online RL with reward: R = α·I(Correct) - β·Energy(trajectory)             │
│  Agent can deviate from greedy path to find better tradeoffs                │
│  Output: Energy-aware adaptive RAG controller                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Baselines & infrastructure (BM25 retrieval, evaluation harness, energy tracking) |
| **Phase 1** | 🔄 In Progress | Cost-ordered search data synthesis |
| **Phase 2** | ❌ Pending | Behavior cloning on generated traces |
| **Phase 3** | ❌ Pending | PPO refinement with energy-aware reward |

### Current Baseline Results (HotpotQA, 100 questions)

| Metric | Dense Retrieval | BM25 Retrieval | Δ |
|--------|-----------------|----------------|---|
| Claim Recall | 39.6% | 51.2% | +11.6% |
| Hallucination | 42.5% | 22.8% | **-19.7%** |
| Faithfulness | 53.5% | 73.2% | **+19.7%** |
| Context Utilization | 30.5% | 40.7% | +10.2% |

## Project Structure

```
rag_eval/
├── src/
│   ├── corpus.py         # Corpus loading and chunking
│   ├── retriever.py      # BM25 and Faiss retrievers
│   ├── generator.py      # LLM generation (Ollama/Gemini)
│   ├── pipeline.py       # RAG orchestration
│   └── base.py           # BaseRAG interface
├── baselines/
│   ├── naive_rag.py      # Standard k=5 retrieval
│   ├── fullk_rag.py      # Exhaustive k=50 retrieval
│   ├── no_retrieval.py   # Generator-only baseline
│   └── adaptive_rag.py   # Rule-based routing (comparison)
├── evaluation/
│   ├── harness.py        # Minimal eval (EM, F1)
│   └── energy.py         # CodeCarbon energy tracking
├── scripts/
│   ├── run_comparison.py # Compare all baselines
│   └── build_hotpotqa_distractor_corpus.py
├── experiments/
│   ├── run_baseline.py   # Main evaluation script
│   └── logs/             # Evaluation results
├── data/
│   ├── processed/        # Chunked passages (66k from HotpotQA)
│   └── indexes/          # BM25 and Faiss indexes
├── config_local.yaml     # Configuration (Ollama)
└── PROJECT_PLAN.md       # Detailed research plan
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### 2. Set Up Ollama (Local LLMs)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start server
ollama serve

# Pull models
ollama pull mistral          # SLM worker (7B)
ollama pull llama3:70b       # LLM worker (optional, requires >40GB VRAM)
```

See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed instructions.

### 3. Configure

Edit `config_local.yaml`:

```yaml
retriever:
  type: "bm25"              # or "dense" for vector search
  index_path: "./data/indexes/faiss.bm25.pkl"

generator:
  type: "ollama"
  model: "mistral"          # SLM worker
  temperature: 0.0
```

## Usage

### Run Baseline Comparison

```bash
python scripts/run_comparison.py --config config_local.yaml --num-questions 50
```

Compares: naive_k5, full_k50, no_retrieval, adaptive_rule

### Run Full Evaluation (with RAGChecker)

```bash
python experiments/run_baseline.py --config config_local.yaml
```

### Build Corpus & Index

```bash
# Build HotpotQA corpus with distractor paragraphs
python scripts/build_hotpotqa_distractor_corpus.py

# Build BM25 index
python src/retriever.py
```

## Baseline Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `naive_k5` | Retrieve top-5, generate once | Standard RAG baseline |
| `full_k50` | Retrieve top-50, generate once | Maximum recall baseline |
| `no_retrieval` | Generate from LLM knowledge only | Lower bound (parametric only) |
| `adaptive_rule` | Route based on query complexity | Heuristic comparison |

## Key Files for Phase 1

To implement the cost-ordered search, you'll need:

```python
# green_tree_search.py (to be implemented)
class GreenTreeSearch:
    def __init__(self, slm, llm, retriever, judge):
        self.costs = {"slm": 1, "retrieve": 5, "llm": 20}
    
    def search(self, query, ground_truth):
        # Try paths in ascending cost order
        # Return cheapest successful trajectory
```

## Dependencies

- **Retrieval**: `rank_bm25`, `faiss-cpu`, `sentence-transformers`
- **Generation**: `ollama`, `google-generativeai`
- **Evaluation**: `ragchecker`, `datasets`
- **Energy**: `codecarbon`
- **RL (Phase 3)**: `trl`, `stable-baselines3`

## References

- [Adaptive-RAG](https://arxiv.org/abs/2403.14403) — Query complexity routing
- [DeepRAG](https://arxiv.org/abs/2502.01142) — Multi-hop retrieval with atomic actions
- [The Bitter Lesson](http://www.incompleteideas.net/IncsIdeas/BitterLesson.html) — Compute > human knowledge
- [RAGChecker](https://arxiv.org/abs/2408.08067) — Fine-grained RAG evaluation

## Citation

```bibtex
@misc{energyawarerag2025,
  title={Energy-Aware Adaptive RAG via Reinforcement Learning},
  author={...},
  year={2025}
}
```