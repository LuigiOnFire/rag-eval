# RAG Energy Benchmarking Project Plan

> **Goal:** Compare energy consumption and quality across RAG architectures (naive, adaptive, DeepRAG-inspired, RL-based) to develop an energy-aware RAG system.

---

## Project Structure

```
rag_eval/
├── baselines/
│   ├── naive/              # Current pipeline (k=5)
│   ├── full_k/             # Exhaustive retrieval (k=50) - upper bound
│   ├── adaptive/           # Query complexity routing
│   └── deep/               # "Inspired-by" DeepRAG
│
├── experiments/
│   └── rl_energy/          # RL-based energy-aware system
│
├── evaluation/
│   ├── harness.py          # Minimal: answer() + evaluate()
│   ├── metrics.py          # Quality metrics (RAGChecker wrapper)
│   └── energy.py           # Batch-level CodeCarbon only
│
├── src/                    # Shared components
│   ├── retriever.py
│   ├── generator.py
│   └── corpus.py
│
├── configs/
│   ├── naive.yaml
│   ├── full_k.yaml
│   ├── adaptive.yaml
│   └── deep.yaml
│
├── analysis/               # Notebooks for results
│   ├── energy_profiles.ipynb
│   └── quality_vs_cost.ipynb
│
└── results/
```

---

## Execution Phases

### Phase 1: Minimal Eval Harness + Naive Baseline (1 week)
**Status:** ✅ COMPLETE

**Goal:** Get *any* working baseline through evaluation.

**Deliverables:**
- [x] Fix corpus-query mismatch (HotpotQA-matched corpus)
- [x] Minimal evaluation harness (`evaluation/harness.py`)
- [x] Common `BaseRAG` interface (`src/base.py`)
- [x] Validation test passed (40% exact match, 3s for 5 queries)

**Validation:** Naive RAG runs, quality metrics work, results saved to `results/`.

---

### Phase 2: Add 3 Cheap Baselines (1-2 weeks)
**Status:** ✅ COMPLETE

**Goal:** Establish comparison points before doing anything complex.

**Deliverables:**
- [x] NaiveRAG wrapper (baselines/naive.py) - k=5 retrieval
- [x] FullKRAG baseline (baselines/full_k.py) - k=50 upper bound
- [x] NoRetrievalRAG baseline (baselines/no_retrieval.py) - generator only
- [x] EnergyTracker with CodeCarbon (evaluation/energy.py)
- [x] Comparison script (scripts/run_comparison.py)
- [x] Validation test passed with energy tracking

**Initial results (5 questions, Nov 25):**
| Baseline | Exact Match | Avg Time (s) | Energy (Wh) |
|----------|------------|--------------|-------------|
| naive_k5 | 20% | 1.53 | 0.3502 |
| full_k50 | 20% | 2.01 | 0.6762 |
| no_retrieval | 20% | 1.73 | 0.4666 |

**Key observations:**
- Full_k50 uses ~2x energy of naive_k5 (expected - longer prompts)
- No_retrieval not faster (longer answers without context constraints)
- Energy measurement working via CodeCarbon

**Note:** Adaptive-RAG deferred to Phase 3 (needs paper/repo research).

---

### Phase 3: Adaptive-RAG + DeepRAG-lite (2-3 weeks)
**Status:** IN PROGRESS (3a partial)

**Goal:** Add adaptive strategies that vary retrieval based on query complexity.

#### 3a: Adaptive-RAG
**Paper architecture:**
```
Query → Trained T5 Classifier → [simple: no retrieval | moderate: single-hop | complex: multi-hop] → Answer
```

**Implementation status:**

##### ✅ COMPLETE: Simplified classifiers (for rapid prototyping)
- [x] `RuleBasedClassifier` - Heuristic patterns (free, fast)
- [x] `LLMClassifier` - Ollama-prompted classification
- [x] `AdaptiveRAG` baseline with routing logic
- [x] Integration in `run_comparison.py`

**Initial results (5 questions, Nov 25):**
| Baseline | Exact Match | Avg F1 | Avg Time (s) |
|----------|-------------|--------|--------------|
| naive_k5 | 20% | 0.041 | 1.06 |
| full_k50 | 20% | 0.032 | 1.55 |
| no_retrieval | 20% | 0.060 | 0.26 |
| adaptive_rule | 20% | 0.022 | 1.49 |
| adaptive_llm | 20% | 0.060 | 0.32 |

##### 🔄 TODO: Train classifier (true Adaptive-RAG replication)

The real Adaptive-RAG paper trains a T5 classifier using automatically collected labels:

**Step 1: Generate Silver Labels**
- Run all 3 strategies (no-retrieval, single-hop, multi-hop) on HotpotQA dev set (~500 queries)
- For each query, label it with whichever strategy answered correctly
- Creates training labels based on actual model performance

**Step 2: Create Binary Labels (dataset inductive bias)**
- Single-hop datasets (NQ, TriviaQA, SQuAD) → label as "simple"
- Multi-hop datasets (HotpotQA, MuSiQue) → label as "complex"

**Step 3: Train T5 Classifier**
- Combine silver + binary labels
- Fine-tune T5-small or T5-base on query → complexity mapping
- Output: Trained classifier model

**Step 4: Integrate Trained Classifier**
- Add `TrainedClassifier` to `baselines/classifiers/`
- Use trained model in `AdaptiveRAG` baseline

**Estimated effort:**
| Task | Time |
|------|------|
| Generate predictions (500 queries × 3 strategies) | ~2-3 hrs compute |
| Create silver labels script | 1 hr coding |
| Create binary labels (add NQ/TriviaQA subset) | 2 hrs |
| Train T5 classifier | ~1 hr training |
| Integrate trained classifier | 1 hr coding |
| **Total** | **~1 day** |

**Resources:** 
- [Paper](https://arxiv.org/abs/2403.14403)
- [GitHub](https://github.com/starsuzi/Adaptive-RAG) - See `classifier/` folder

#### 3b: DeepRAG-lite
**Status:** NOT STARTED

**What to implement:**
1. Query complexity classifier (rule-based or small model)
2. Three retrieval modes:
   - Single-hop: k=5, one retrieval
   - Multi-hop: k=5, 2-3 iterations
   - CoT: k=10 + chain-of-thought prompting
3. Controller selects mode based on query

**What NOT to implement:**
- ❌ Full reasoning traces
- ❌ Learned controller (use heuristics first)
- ❌ Exact paper reproduction

**Important:** DeepRAG reproduction will be approximate. Reproduce *architecture patterns*, not exact numbers.

**Validation:** DeepRAG-lite beats naive on complex queries, costs more energy.

---

### Phase 4: RL Energy-Aware System (4-5 weeks)
**Status:** NOT STARTED

**Goal:** Train policy that optimizes quality-energy trade-off.

**Simplified formulation:**
```python
# State: query embedding + retrieval count + current score estimate
# Actions: [STOP, RETRIEVE_MORE, GENERATE]
# Reward: quality_score - lambda * energy_cost
```

**Implementation approach:**
1. PPO with discrete actions
2. Lambda=0.1 (low energy penalty) → Lambda=1.0 (high penalty)
3. Train on HotpotQA subset (1000 queries)
4. Compare Pareto frontier to baselines

**What NOT to implement:**
- ❌ MoE (Mixture of Experts)
- ❌ Decision Transformers
- ❌ Complex state representations

**Validation:** RL policy finds different quality-energy trade-offs than fixed baselines.

---

### Phase 5: Analysis + Paper (1-2 weeks)
**Status:** NOT STARTED

**Goal:** Final comparison and paper-ready results.

**Add MuSiQue dataset here** (not earlier - too expensive for debugging).

**Final comparison:**
```python
systems = ['naive', 'full_k', 'no_retrieval', 'adaptive', 'deep_lite', 'rl_energy']
datasets = ['hotpotqa', 'single_hop', 'musique']

results = benchmark.compare_all(systems, datasets)
```

**Deliverables:**
- Quality vs Energy scatter plot
- Pareto frontier analysis
- Per-dataset breakdown
- Statistical significance tests

---

## Timeline Summary

| Phase | Duration | Risk | Output |
|-------|----------|------|--------|
| 1. Minimal harness + naive | 1 week | Low | Working evaluation |
| 2. Three cheap baselines | 1-2 weeks | Low | 4-system comparison |
| 3. DeepRAG-lite | 2 weeks | Medium | 5-system comparison |
| 4. RL system | 4-5 weeks | High | Novel contribution |
| 5. Analysis + MuSiQue | 1-2 weeks | Low | Paper-ready results |
| **TOTAL** | **9-12 weeks** | | Publishable study |

---

## Common Interface

All RAG systems must implement:

```python
from abc import ABC, abstractmethod

class BaseRAG(ABC):
    @abstractmethod
    def answer(self, query: str, return_trace: bool = False) -> str | dict:
        """
        Args:
            query: The question to answer
            return_trace: If False, return only answer string (fast)
                         If True, return dict with answer + retrieved_docs + metadata
        
        Returns:
            str: Just the answer (return_trace=False)
            dict: Full trace (return_trace=True)
        """
        pass
```

**Key design decision:** `return_trace=False` by default for benchmarking speed.

---

## Energy Measurement Strategy

**Use batch-level measurement, not per-query:**
- Evaluate each system over N=100 queries in a single batch
- Compute mean energy per query from batch total
- This stabilizes variance by ~10x compared to per-query measurement

**Primary tool:** CodeCarbon (lightweight, non-invasive)

**Metrics to collect:**
- Total kWh per batch
- Mean energy per query
- Wall-clock time per batch
- GPU utilization % (NVML, optional)

---

## Key Constraints & Decisions

### DO:
- ✅ Batch-level energy measurement
- ✅ Start with HotpotQA only
- ✅ Use simple single-hop dataset for sanity checks
- ✅ Full-k baseline as upper bound
- ✅ DeepRAG-lite (inspired-by, not exact)
- ✅ PPO for RL
- ✅ Simple state/action spaces

### DON'T:
- ❌ Per-query energy instrumentation (too noisy)
- ❌ MuSiQue until Phase 5 (too expensive)
- ❌ Exact DeepRAG reproduction (incomplete repo, under-specified)
- ❌ MoE or Decision Transformers
- ❌ Over-optimize retrieval quality (good enough is fine)
- ❌ Complex return dicts always (use return_trace flag)

---

## Hardware

- **GPU:** NVIDIA L40 (46GB VRAM)
- **Inference:** Ollama with Mistral 7B (local)
- **Embeddings:** all-MiniLM-L6-v2
- **Vector store:** Faiss IndexFlatIP

---

## Current Status

**Last updated:** November 25, 2025

**Phase 1:** ✅ COMPLETE
- Built HotpotQA-matched corpus (815 passages)
- Created BaseRAG interface (src/base.py)
- Created evaluation harness (evaluation/harness.py)
- Validated end-to-end (40% exact match on 5 queries)

**Phase 2:** ✅ COMPLETE
- Implemented 3 baselines: NaiveRAG, FullKRAG, NoRetrievalRAG
- Added CodeCarbon energy tracking
- Created comparison script with quality + energy metrics
- Validated comparison pipeline

**Phase 3:** IN PROGRESS
- ✅ Implemented simplified classifiers (rule-based, LLM-prompted)
- ✅ Created AdaptiveRAG baseline with routing logic
- ✅ 5-baseline comparison working
- 🔄 TODO: Train T5 classifier for true Adaptive-RAG replication
- ❌ DeepRAG-lite not started

**Phase 4-5:** NOT STARTED

**Next steps:**
1. Generate silver labels (run 3 strategies on 500 HotpotQA queries)
2. Add single-hop dataset (NQ or TriviaQA subset) for binary labels
3. Train T5 classifier on combined labels
4. Integrate trained classifier into AdaptiveRAG

---

## References

- **Adaptive-RAG:** [Paper](https://arxiv.org/abs/2403.14403) - Query complexity routing
- **DeepRAG:** [Paper](https://arxiv.org/abs/2401.08815) - LLM-based retrieval controller
- **RAGChecker:** Amazon's evaluation framework (already integrated)
- **CodeCarbon:** Energy tracking library
