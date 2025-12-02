# Green-DeepRAG Project Plan

> **Goal:** Train an encoder-based controller ("Manager") to dynamically route RAG queries through the cheapest successful trajectory, using compressed observations from worker LLMs.

> **Philosophy:** Compute-driven discovery of optimal policies, not human-designed rules (inspired by "The Bitter Lesson").

---

## Architecture Overview

### The Manager-Worker Iterative Agent

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MANAGER-WORKER LOOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   State: [CLS] Query [SEP] Step1_Summary [SEP] Step2_Summary [SEP] ...      │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │   Controller    │  (RoBERTa-Large / DeBERTa-v3)          │
│                  │   "The Manager" │                                        │
│                  └────────┬────────┘                                        │
│                           │ classifies → Action ID (0-6)                    │
│                           ▼                                                 │
│         ┌─────────────────────────────────────────┐                         │
│         │            WORKER EXECUTION             │                         │
│         │  Worker executes action, then generates │                         │
│         │  a <50 token status update (Observation)│                         │
│         └─────────────────┬───────────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│              "Found 3 docs on Apple revenue.                                │
│               Missing 2024 data."                                           │
│                           │                                                 │
│                           └──────────► Append to state, loop back           │
│                                        (until Generate_and_End action)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Critical Constraint: State Compression

**We strictly forbid passing raw retrieved documents or full reasoning traces to the Controller.**

| What Workers Do | What Gets Passed to Controller |
|-----------------|-------------------------------|
| Retrieve 2000 tokens of docs | "Found 3 docs on X. Missing Y." (<50 tokens) |
| Generate 500-token reasoning | "Computed revenue = $394B. Need verification." |
| Decompose into sub-queries | "Split into: Q1 (director), Q2 (birthplace)." |

**Why:** This ensures the encoder never breaches its 512-token limit while retaining the "semantic gist" needed for routing decisions.

### Components

| Component | Role | Implementation |
|-----------|------|----------------|
| **Controller** | Classifies state → action (the "Manager") | RoBERTa-Large or DeBERTa-v3-Base |
| **SLM Worker** | Fast, cheap execution + compression | Mistral-7B (via Ollama) |
| **LLM Worker** | Expensive, accurate execution + compression | Llama-3-70B or GPT-4o |
| **Retriever** | Keyword or dense search | BM25 (rank_bm25) / Faiss |
| **Judge** | Validates answer correctness | Exact Match + LLM-as-judge |
| **Energy Tracker** | Measures compute cost | CodeCarbon |

### Action Space (7 Classes)

| ID | Action | Description |
|----|--------|-------------|
| 0 | `Generate_and_End(SLM)` | Final answer with small model |
| 1 | `Generate_and_End(LLM)` | Final answer with large model |
| 2 | `Decompose(SLM)` | Break query into sub-questions |
| 3 | `Decompose(LLM)` | Break query into sub-questions |
| 4 | `Retrieve(Keyword)` | BM25 keyword search |
| 5 | `Retrieve(Dense)` | Vector similarity search |
| 6 | `Reason(LLM)` | Intermediate synthesis/verification |

**Costs are measured, not hard-coded.** Before training, we benchmark each action with CodeCarbon to build a cost table (Wh per action). This ensures the reward function reflects actual energy consumption on target hardware.

**Input format:** `[CLS] Original_Query [SEP] Obs_1 [SEP] Obs_2 [SEP] ...`  
**Output:** Linear head → Softmax over 7 logits → argmax for action

### Reward Function

```
R = α · I(Correct) - β · Σ Energy(actions)
```

Where:
- `I(Correct)` = 1 if final answer matches ground truth, 0 otherwise
- `Σ Energy(actions)` = cumulative energy (Wh) from pre-measured cost table
- `α, β` = hyperparameters balancing accuracy vs efficiency

**Cost Table Construction:**
```python
# Pre-compute by running each action N times with CodeCarbon
cost_table = {
    0: measure_avg_energy(slm.generate, n=50),      # Generate_and_End(SLM)
    1: measure_avg_energy(llm.generate, n=50),      # Generate_and_End(LLM)
    2: measure_avg_energy(slm.decompose, n=50),     # Decompose(SLM)
    3: measure_avg_energy(llm.decompose, n=50),     # Decompose(LLM)
    4: measure_avg_energy(retriever.bm25, n=50),    # Retrieve(Keyword)
    5: measure_avg_energy(retriever.dense, n=50),   # Retrieve(Dense)
    6: measure_avg_energy(llm.reason, n=50),        # Reason(LLM)
}
```

---

## Training Pipeline (3 Phases)

### Phase 1: Cost-Ordered Search (Offline Oracle)
**Goal:** Generate trajectories by simulating the agent with a greedy cost-ordered policy.

```python
class GreenTreeSearch:
    """
    Simulate iterative agent, trying cheaper actions first.
    Record the cheapest trajectory that yields a correct answer.
    """
    
    def search(self, query: str, ground_truth: str) -> Trajectory:
        state = f"[CLS] {query} [SEP]"
        trajectory = []
        
        # Try actions in cost order until Generate_and_End succeeds
        for action in self.cost_ordered_actions():
            # Execute action
            result, observation = self.execute(action, state)
            trajectory.append((state, action, observation))
            
            # Update state with compressed observation
            state = f"{state} {observation} [SEP]"
            
            # Check if this is a terminal action
            if action in [GENERATE_END_SLM, GENERATE_END_LLM]:
                if self.judge.is_correct(result, ground_truth):
                    return Trajectory(trajectory, correct=True)
                else:
                    # Try next more expensive action
                    continue
        
        # Fallback: most expensive path
        return Trajectory(trajectory, correct=False)
    
    def execute(self, action: int, state: str) -> tuple[str, str]:
        """Execute action, return (result, compressed_observation)."""
        if action == RETRIEVE_KEYWORD:
            docs = self.retriever.search(query, method="bm25")
            # Worker compresses the retrieval result
            observation = self.slm.summarize(
                f"Summarize in <50 tokens what was found: {docs}"
            )
            return docs, observation
        
        elif action == GENERATE_END_SLM:
            answer = self.slm.generate(query, context=self.context)
            observation = f"Generated answer: {answer[:100]}"
            return answer, observation
        
        # ... etc for other actions
```

**Crucial Detail:** We must simulate the "Worker Observations" during search, using the SLM to compress intermediate results.

**Output:** ~5k trajectories of (state → action → observation) triples

### Phase 2: Behavior Cloning (Classification)
**Goal:** Train RoBERTa to predict the next action given the compressed state.

```python
from transformers import AutoModelForSequenceClassification, Trainer

# Load encoder with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=7  # 7 action classes
)

# Training data: each (state, action) pair from Phase 1 trajectories
training_data = [
    {"text": "[CLS] Who directed Sinister? [SEP]", "label": 4},  # Retrieve(Keyword)
    {"text": "[CLS] Who directed Sinister? [SEP] Found docs on Scott Derrickson. [SEP]", 
     "label": 0},  # Generate_and_End(SLM)
    ...
]

# Train with cross-entropy
trainer = Trainer(
    model=model,
    train_dataset=training_data,
    ...
)
```

**Output:** RoBERTa classifier that mimics the cost-ordered search policy

### Phase 3: Cost-Aware PPO (Refinement)
**Goal:** Online RL to discover better policies than greedy search.

```python
import gym
from stable_baselines3 import PPO

class RAGEnv(gym.Env):
    """
    Custom Gym environment for iterative RAG routing.
    step() returns compressed observations, not raw text.
    """
    
    def __init__(self, slm, llm, retriever, judge):
        self.action_space = gym.spaces.Discrete(7)  # 7 action classes
        self.observation_space = gym.spaces.Box(...)  # RoBERTa embeddings
        self.costs = {0: 1, 1: 20, 2: 1, 3: 20, 4: 5, 5: 5, 6: 20}
        self.max_steps = 5
    
    def reset(self):
        self.query, self.ground_truth = self.sample_question()
        self.state = f"[CLS] {self.query} [SEP]"
        self.trajectory_cost = 0
        self.step_count = 0
        return self.encode(self.state)
    
    def step(self, action: int):
        self.step_count += 1
        self.trajectory_cost += self.costs[action]
        
        # Execute action, get compressed observation
        result, observation = self.execute(action)
        self.state = f"{self.state} {observation} [SEP]"
        
        # Check termination
        done = (action in [0, 1]) or (self.step_count >= self.max_steps)
        
        if done:
            correct = self.judge.is_correct(result, self.ground_truth)
            reward = self.alpha * correct - self.beta * self.trajectory_cost
        else:
            reward = -self.costs[action]  # Step cost
        
        return self.encode(self.state), reward, done, {}

# Train with PPO
model = PPO("MlpPolicy", RAGEnv(...), verbose=1)
model.learn(total_timesteps=50000)
```

**Output:** Energy-aware adaptive RAG controller

---

## Execution Phases

### Phase 0: Infrastructure & Baselines ✅ COMPLETE
**Goal:** Establish evaluation infrastructure and comparison baselines.

**Deliverables:**
- [x] HotpotQA corpus with distractor paragraphs (66k passages)
- [x] BM25 retriever (fixed entity-matching issues from dense retrieval)
- [x] Evaluation harness with RAGChecker integration
- [x] CodeCarbon energy tracking
- [x] Baseline implementations: NaiveRAG, FullKRAG, NoRetrievalRAG, AdaptiveRAG

**Results (HotpotQA, 100 questions):**
| Metric | Dense Retrieval | BM25 Retrieval | Δ |
|--------|-----------------|----------------|---|
| Claim Recall | 39.6% | 51.2% | +11.6% |
| Hallucination | 42.5% | 22.8% | **-19.7%** |
| Faithfulness | 53.5% | 73.2% | **+19.7%** |

---

### Phase 1: Cost-Ordered Search 🔄 IN PROGRESS
**Goal:** Generate trajectories with compressed observations via greedy cost-ordered search.

**Deliverables:**
- [ ] **Cost table benchmark** — Measure each action's energy (Wh) with CodeCarbon
- [ ] `GreenTreeSearch` class implementation
- [ ] Worker observation compression (SLM summarizes to <50 tokens)
- [ ] Ground truth judge (Exact Match + F1)
- [ ] Run search on 500 HotpotQA samples
- [ ] Validate: "Do compressed observations provide enough signal?"

**Key validation metrics:**
- "X% of queries solved with cheapest action"
- "Average trajectory length"
- "Observation quality (do summaries preserve key info?)"

**Estimated effort:** 1-2 weeks

---

### Phase 2: Behavior Cloning ❌ PENDING
**Goal:** Train RoBERTa classifier on (state → action) pairs from Phase 1.

**Deliverables:**
- [ ] Initialize RoBERTa-Large with 7-class classification head
- [ ] Format trajectories: each (state, action) pair is a training example
- [ ] Train with cross-entropy loss
- [ ] Validate classifier accuracy on held-out trajectories

**Estimated effort:** 1 week

---

### Phase 3: PPO Refinement ❌ PENDING
**Goal:** Online RL with iterative environment to improve beyond greedy policy.

**Deliverables:**
- [ ] `RAGEnv` Gym environment (returns compressed observations)
- [ ] PPO training with `stable-baselines3`
- [ ] Hyperparameter sweep (α, β tradeoff)
- [ ] Compare to baselines on Pareto frontier

**Estimated effort:** 2-3 weeks

---

### Phase 4: Evaluation & Paper ❌ PENDING
**Goal:** Final evaluation and paper-ready results.

**Deliverables:**
- [ ] Full HotpotQA evaluation (1000+ questions)
- [ ] Add MuSiQue dataset for generalization
- [ ] Quality vs Energy Pareto plots
- [ ] Statistical significance tests
- [ ] Paper draft

**Estimated effort:** 2 weeks

---

## Timeline Summary

| Phase | Duration | Status | Output |
|-------|----------|--------|--------|
| 0. Infrastructure + Baselines | 3 weeks | ✅ Complete | Working eval, 4 baselines |
| 1. Cost-Ordered Search | 1-2 weeks | 🔄 In Progress | Trajectory dataset with observations |
| 2. Behavior Cloning | 1 week | ❌ Pending | RoBERTa classifier |
| 3. PPO Refinement | 2-3 weeks | ❌ Pending | RL-trained controller |
| 4. Evaluation + Paper | 2 weeks | ❌ Pending | Paper-ready results |
| **TOTAL** | **8-10 weeks** | | Novel RL-based RAG controller |

---

## Key Libraries & Tools

| Purpose | Library |
|---------|---------||
| Controller base model | `transformers` (RoBERTa-Large, DeBERTa-v3) |
| Classification training | `transformers.Trainer` (cross-entropy) |
| RL training | `stable-baselines3` (PPO) |
| RL environment | `gymnasium` |
| Energy tracking | `codecarbon` |
| Retrieval | `rank_bm25`, `faiss` |
| Generation | `ollama` (local), `openai` (API) |
| Evaluation | `ragchecker`, `datasets` |

---

## Previous Work (Archived)

<details>
<summary>Click to expand: Original baseline phases (now complete)</summary>

### Original Phase 1: Minimal Eval Harness + Naive Baseline
**Status:** ✅ COMPLETE

- [x] Fix corpus-query mismatch (HotpotQA-matched corpus)
- [x] Minimal evaluation harness (`evaluation/harness.py`)
- [x] Common `BaseRAG` interface (`src/base.py`)
- [x] Validation test passed (40% exact match, 3s for 5 queries)

### Original Phase 2: Add 3 Cheap Baselines
**Status:** ✅ COMPLETE

- [x] NaiveRAG wrapper (baselines/naive.py) - k=5 retrieval
- [x] FullKRAG baseline (baselines/full_k.py) - k=50 upper bound
- [x] NoRetrievalRAG baseline (baselines/no_retrieval.py) - generator only
- [x] EnergyTracker with CodeCarbon (evaluation/energy.py)

**Initial results (5 questions, Nov 25):**
| Baseline | Exact Match | Avg Time (s) | Energy (Wh) |
|----------|------------|--------------|-------------|
| naive_k5 | 20% | 1.53 | 0.3502 |
| full_k50 | 20% | 2.01 | 0.6762 |
| no_retrieval | 20% | 1.73 | 0.4666 |

### Original Phase 3a: Adaptive-RAG (Simplified)
**Status:** ✅ COMPLETE

- [x] `RuleBasedClassifier` - Heuristic patterns
- [x] `LLMClassifier` - Ollama-prompted classification
- [x] `AdaptiveRAG` baseline with routing logic

**Note:** We are now pivoting away from trained T5 classifiers toward the Green-DeepRAG RL approach.

</details>

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

---

## Hardware

- **GPU:** NVIDIA L40 (46GB VRAM)
- **Inference:** Ollama with Mistral 7B (local)
- **Retrieval:** BM25 (rank_bm25)
- **Embeddings:** all-MiniLM-L6-v2 (for dense retrieval fallback)

---

## Key Design Decisions

### DO:
- ✅ Manager-Worker abstraction (controller never sees raw docs)
- ✅ State compression (<50 token observations from workers)
- ✅ BERT encoder controller (RoBERTa-Large / DeBERTa-v3)
- ✅ Iterative decision loop (not single-shot)
- ✅ 7-class action space (including Decompose, Reason)
- ✅ Cost-ordered search for warm-start data
- ✅ Behavior cloning (cross-entropy) before RL
- ✅ PPO with energy-aware reward
- ✅ BM25 + Dense retrieval options
- ✅ Start with HotpotQA only

### DON'T:
- ❌ Pass raw documents to controller (use compressed observations)
- ❌ Decoder-based controller (we classify, not generate)
- ❌ Human-designed routing rules (let RL discover policy)
- ❌ Per-query energy instrumentation (too noisy)
- ❌ MuSiQue until Phase 4 (too expensive for debugging)
- ❌ Exact DeepRAG reproduction (inspired by, not identical)

---

## References

- **Adaptive-RAG:** [Paper](https://arxiv.org/abs/2403.14403) — Query complexity routing
- **DeepRAG:** [Paper](https://arxiv.org/abs/2502.01142) — Multi-hop retrieval with atomic actions
- **The Bitter Lesson:** [Essay](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Compute > human knowledge
- **RAGChecker:** [Paper](https://arxiv.org/abs/2408.08067) — Fine-grained RAG evaluation
- **CodeCarbon:** Energy tracking library

---

## Current Status

**Last updated:** December 2, 2025

**Phase 0 (Infrastructure):** ✅ COMPLETE
- Built HotpotQA corpus with distractor paragraphs (66k passages)
- Switched from dense to BM25 retrieval (major accuracy improvement)
- Created evaluation harness with RAGChecker integration
- Implemented 4 baselines: NaiveRAG, FullKRAG, NoRetrievalRAG, AdaptiveRAG
- Added CodeCarbon energy tracking

**Phase 1 (Cost-Ordered Search):** 🔄 IN PROGRESS
- Next: **Benchmark cost table** (measure each action's Wh with CodeCarbon)
- Next: Implement `RAGEnv` class (Gym environment with compressed observations)
- Next: Initialize RoBERTa controller with `num_labels=7`
- Next: Implement `GreenTreeSearch` with worker observation compression
- Next: Run on 500 HotpotQA samples to validate approach

**Phase 2-4:** ❌ PENDING
