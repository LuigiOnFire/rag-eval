# Path Forward: From GreenSearch to Decoder Pre-training

This document outlines the roadmap to implement the "Green-DeepRAG" imitation learning pipeline using the restored `GreenSearch` and `Decoder` architecture.

## Phase 1: Verification & Data Generation
**Goal:** Generate a high-quality "Oracle" dataset where every trajectory is the minimum-cost path to the correct answer.

1.  **Metric Verification (Small Batch)**
    *   **Action:** Run `scripts/generate_trajectories.py` with `num_samples=20`.
    *   **Verify:** 
        *   Are the generated trajectories actually correct? (Check `results/trajectories_v2.json`)
        *   Does the `GreenSearch` prefer cheaper actions (SLM) over expensive ones (LLM) when possible?
        *   ensure the `cost_table` is reasonable.

2.  **Dataset Production (Full Batch)**
    *   **Action:** Run generation on the full HotPotQA training split (e.g., 5,000+ samples).
    *   **Command:** 
        ```bash
        python scripts/generate_trajectories.py \
          --config config_local.yaml \
          --num_samples 5000 \
          --split train \
          --output data/processed/oracle_trajectories_train.json
        ```
    *   **Output:** This will create a dataset of `(Context, Action_Sequence)` pairs optimal for energy efficiency.

## Phase 2: Model Training (Imitation Learning)
**Goal:** Train a "Decoder Agent" (Manager) to predict the optimal action sequence given the context.

1.  **Format Data for SFT**
    *   The generation script already creates a training file (suffix `.training.json`).
    *   Ensure the format matches `scripts/train_decoder_sft.py` expectations (input/output pairs).

2.  **Supervised Fine-Tuning (SFT)**
    *   **Script:** `scripts/train_decoder_sft.py`
    *   **Action:** Fine-tune a base model (e.g., Llama-3-8B or Mistral-7B) on the Oracle trajectories.
    *   **Key Concept:** The model learns to output `[RETRIEVE]`, `[DECOMPOSE]`, etc., token by token.

## Phase 3: Evaluation & Refinement
**Goal:** Prove the trained agent is more efficient than the baseline.

1.  **Run Comparison**
    *   Use `scripts/run_comparison.py` (you may need to adapt it to load your local checkpoint).
    *   Compare:
        *   **Baseline:** Standard RAG (Retrieve-Generate).
        *   **Oracle:** The theoretical upper bound (GreenSearch).
        *   **Decoder Agent:** Your trained model.

2.  **Metric:** "Efficiency Ratio" = (Accuracy / Energy_Wh). We want the Decoder Agent to match Oracle accuracy while using significantly less energy than a naive LLM-only approach.

## Immediate Next Steps
1.  Run the verification batch:
    ```bash
    python scripts/generate_trajectories.py --num_samples 10
    ```
