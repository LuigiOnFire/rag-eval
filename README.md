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
An **iterative Manager-Worker agent** where a small encoder (the "Manager") routes tasks to frozen LLM workers, receiving only **compressed observations** back — never raw documents.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MANAGER-WORKER LOOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [CLS] Query [SEP] Step1_Summary [SEP] Step2_Summary [SEP]                 │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │   Controller    │  (RoBERTa-Large)                       │
│                  │   "The Manager" │                                        │
│                  └────────┬────────┘                                        │
│                           │ classifies → Action ID (0-6)                    │
│                           ▼                                                 │
│         ┌─────────────────────────────────────────┐                         │
│         │            WORKER EXECUTION             │                         │
│         │  (SLM/LLM executes, returns <50 token   │                         │
│         │   summary — Manager never sees raw docs) │                         │
│         └─────────────────┬───────────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│              "Found 3 docs on Apple revenue.                                │
│               Missing 2024 data."  (Observation)                            │
│                           │                                                 │
│                           └──────────► Append to state, loop back           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The controller never breaches its 512-token context limit because workers compress all intermediate results into short status updates.

**Why Encoder over Decoder?** We don't need to *generate* text — we only need to *route*. BERT-style encoders provide better bidirectional state understanding per parameter and faster inference than autoregressive decoders.

## Architecture

### Components

| Component | Role | Example |
|-----------|------|---------|
| **Controller** | Encoder that classifies state → action (Manager) | RoBERTa-Large, DeBERTa-v3 |
| **SLM Worker** | Fast, cheap generation + compression | Mistral-7B, Llama-3-8B |
| **LLM Worker** | Expensive, accurate generation + compression | Llama-3-70B, GPT-4o |
| **Retriever** | BM25 or dense search | rank_bm25, Faiss |
| **Judge** | Validates answer correctness | Exact match + LLM-judge |

### Action Space (7 Classes)
The controller outputs a probability distribution over 7 discrete actions:

| ID | Action | Description |
|----|--------|-------------|
| 0 | `Generate_and_End(SLM)` | Final answer with small model |
| 1 | `Generate_and_End(LLM)` | Final answer with large model |
| 2 | `Decompose(SLM)` | Break query into sub-questions |
| 3 | `Decompose(LLM)` | Break query into sub-questions |
| 4 | `Retrieve(Keyword)` | BM25 search |
| 5 | `Retrieve(Dense)` | Vector similarity search |
| 6 | `Reason(LLM)` | Intermediate synthesis/verification |

**Costs are measured via CodeCarbon** on target hardware, not hard-coded. A cost table is pre-computed by benchmarking each action.

**Input format:** `[CLS] Original_Query [SEP] Step_1_Summary [SEP] Step_2_Summary [SEP] ...`  
**Output:** Softmax over 7 action logits

### The Critical Constraint: State Compression

Workers **never pass raw documents** to the controller. Instead:
1. Worker executes action (e.g., retrieves 2000 tokens)
2. Worker generates a **<50 token status update**
3. Status update is appended to controller's state

This ensures the encoder never exceeds 512 tokens while retaining semantic signal.

### Training Pipeline (3 Phases)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Cost-Ordered Search (Offline Oracle)                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  For each query, simulate agent with GreenTreeSearch:                       │
│    - Try actions in ascending cost order                                    │
│    - Generate compressed observations at each step                          │
│    - Record cheapest trajectory that yields correct answer                  │
│  Output: Dataset of (state → action) pairs with compressed observations     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: Behavior Cloning (Classification)                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Train RoBERTa classifier with Cross-Entropy loss on Phase 1 traces         │
│  Input: [CLS] query [SEP] obs_1 [SEP] obs_2 ...  →  Output: action (0-6)    │
│  Output: Policy that mimics "cheapest winner" trajectories                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: Cost-Aware PPO (Refinement)                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Online RL with reward: R = α·I(Correct) - β·Σ Energy(actions)              │
│  Agent learns to balance cheap failures vs expensive successes              │
│  Output: Energy-aware adaptive RAG controller                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Baselines & infrastructure (BM25 retrieval, evaluation harness, energy tracking) |
| **Phase 1** | 🔄 In Progress | Cost-ordered search with compressed observations |
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