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

### Action Space (8 Classes)
The controller outputs a probability distribution over 8 discrete actions:

| ID | Action | Description |
|----|--------|-------------|
| 0 | `Generate_and_End(SLM)` | Final answer with small model |
| 1 | `Generate_and_End(LLM)` | Final answer with large model |
| 2 | `Decompose(SLM)` | Break query into sub-questions |
| 3 | `Decompose(LLM)` | Break query into sub-questions |
| 4 | `Retrieve(Keyword)` | BM25 search |
| 5 | `Retrieve(Dense)` | Vector similarity search |
| 6 | `Reason(SLM)` | Intermediate synthesis (cheap) |
| 7 | `Reason(LLM)` | Intermediate synthesis (expensive) |

**Costs are measured via CodeCarbon** on target hardware, not hard-coded. A cost table is pre-computed by benchmarking each action.

**Input format:** `[CLS] Original_Query [SEP] Step_1_Summary [SEP] Step_2_Summary [SEP] ...`  
**Output:** Softmax over 8 action logits

### The Critical Constraint: State Compression

Workers **never pass raw documents** to the controller. Instead:
1. Worker executes action (e.g., retrieves 2000 tokens)
2. Worker generates a **<50 token status update**
3. Status update is appended to controller's state

This ensures the encoder never exceeds 512 tokens while retaining semantic signal.

### Training Pipeline (3 Phases)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Cost-Priority Search (Offline Oracle)                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  For each query, find the MINIMUM-COST correct path using Uniform Cost      │
│  Search (priority queue ordered by accumulated cost).                       │
│    - First correct solution found is guaranteed to be cheapest              │
│    - Explores parameterized actions (top_k=3,5,10, query variants)          │
│    - Tracks sub-questions for multi-hop decomposition                       │
│  Output: Dataset of (state → action) pairs with compressed observations     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: Behavior Cloning (Classification)                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Train RoBERTa classifier with Cross-Entropy loss on Phase 1 traces         │
│  Input: [CLS] query [SEP] obs_1 [SEP] obs_2 ...  →  Output: action (0-7)    │
│  Output: Policy that mimics optimal (minimum-cost) trajectories             │
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
| **Phase 1** | ✅ Complete | GreenSearch: Cost-Priority Search with optimality guarantee |
| **Phase 2** | ✅ Complete | Behavior cloning module (ready for training) |
| **Phase 3** | ❌ Pending | PPO refinement with energy-aware reward |

### Phase 1: Cost-Priority Search

| Component | Algorithm | Guarantee |
|-----------|-----------|----------|
| `green_search.py` | **Uniform Cost Search** (priority queue) | **First correct = cheapest** |

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
│   ├── green_search.py      # Phase 1: Cost-Priority Search (optimal)
│   ├── behavior_cloning.py  # Phase 2: Controller training
│   ├── generator.py         # LLM interfaces (Ollama/Gemini)
│   ├── retriever.py         # BM25 and Faiss retrievers
│   ├── pipeline.py          # RAG orchestration (legacy)
│   ├── corpus.py            # Corpus loading
│   └── base.py              # BaseRAG interface
├── baselines/
│   ├── naive.py             # Standard k=5 retrieval
│   ├── full_k.py            # Exhaustive k=50 retrieval
│   ├── no_retrieval.py      # Generator-only baseline
│   └── adaptive.py          # Rule-based routing (comparison)
├── evaluation/
│   ├── harness.py           # Minimal eval (EM, F1)
│   └── energy.py            # CodeCarbon energy tracking
├── scripts/
│   ├── generate_trajectories.py  # Run GreenSearch on dataset
│   ├── train_controller.py  # Train behavior-cloned controller
│   ├── benchmark_costs.py   # Measure action energy costs
│   ├── run_comparison.py    # Compare baselines
│   └── build_hotpotqa_*.py  # Corpus preparation
├── models/                  # Trained controller checkpoints
├── results/
│   ├── cost_table.json      # Measured action costs
│   └── trajectories_*.json  # Generated training trajectories
├── data/
│   ├── processed/           # Chunked passages (66k from HotpotQA)
│   └── indexes/             # BM25 and Faiss indexes
├── config_local.yaml        # Configuration (models, paths)
└── PROJECT_PLAN.md          # Detailed research plan
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

## Green-DeepRAG Training Pipeline (Reproduction Guide)

This is the complete workflow for training an energy-aware RAG controller. **Run these steps in order** when switching to new models.

### Step 0: Configure Models

Edit `config_local.yaml` with your SLM and LLM:

```yaml
slm:
  type: "ollama"
  model_name: "mistral:latest"    # Your small model
  temperature: 0.0
  max_tokens: 256

llm:
  type: "ollama"
  model_name: "llama3:70b"        # Your large model (or API-based)
  temperature: 0.0
  max_tokens: 256

retriever:
  type: "bm25"
  bm25_index_path: "./data/indexes/faiss.bm25.pkl"
  passages_path: "./data/processed/passages.json"
```

### Step 1: Benchmark Action Costs

**Why:** The cost table determines the order in which GreenTreeSearch tries actions. Different hardware = different costs.

```bash
python scripts/benchmark_costs.py \
    --config config_local.yaml \
    --n_samples 10 \
    --output results/cost_table.json
```

**Output:** `results/cost_table.json` with energy (Wh) per action:
```json
{
  "0": {"name": "Generate_and_End_SLM", "avg_wh": 0.021, ...},
  "1": {"name": "Generate_and_End_LLM", "avg_wh": 0.017, ...},
  ...
}
```

**Note:** If LLM is cheaper than SLM (can happen with similar-sized models), the search will prefer LLM. This is by design — cost-ordered search finds the *cheapest* correct trajectory.

### Step 2: Generate Training Trajectories

**Why:** Create (state → action) pairs by running Cost-Priority Search on HotpotQA.

```bash
# NEW: Use GreenSearch V2 (Cost-Priority Search with optimality guarantee)
python scripts/generate_trajectories_v2.py \
    --config config_local.yaml \
    --cost_table results/cost_table.json \
    --num_samples 500 \
    --output results/trajectories_v2_500.json

# Legacy: Strategy-based search (kept for comparison)
# python scripts/generate_trajectories.py --num_samples 500 --output results/trajectories_500.json
```

**Output:**
- `results/trajectories_v2_500.json` — Full trajectory data
- `results/trajectories_v2_500.training.json` — Pre-extracted training pairs

**Key Difference:** GreenSearch V2 uses Uniform Cost Search, guaranteeing the first correct solution found is the minimum-cost solution. The legacy approach used fixed strategy templates.

### Step 3: Train Controller via Behavior Cloning

**Why:** Train RoBERTa/DeBERTa to predict the next action given compressed state.

```bash
python scripts/train_controller.py \
    --trajectories results/trajectories_500.training.json \
    --output models/controller_v1 \
    --model roberta-base \
    --epochs 10 \
    --batch_size 16
```

**Output:** `models/controller_v1/final/` with trained model and tokenizer.

### Step 4: (Future) PPO Refinement

Phase 3 will use the behavior-cloned controller as initialization for PPO training with reward:

```
R = α · I(Correct) - β · Σ Energy(actions)
```

---

## Quick Commands Reference

```bash
# Full pipeline (copy-paste ready)
cd /home/wcrawford/rag_eval

# 1. Benchmark (run once per hardware/model change)
python scripts/benchmark_costs.py --config config_local.yaml --n_samples 10 --output results/cost_table.json

# 2. Generate trajectories with Cost-Priority Search (V2 - recommended)
python scripts/generate_trajectories_v2.py --config config_local.yaml --num_samples 500 --output results/trajectories_v2_500.json

# 3. Train controller
python scripts/train_controller.py --trajectories results/trajectories_v2_500.training.json --output models/controller_v1 --epochs 10
```

---

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