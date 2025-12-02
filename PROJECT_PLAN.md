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

### Phase 1: Cost-Ordered Search ✅ COMPLETE
**Goal:** Generate trajectories with compressed observations via greedy cost-ordered search.

**Deliverables:**
- [x] **Cost table benchmark** — Measured each action's energy 