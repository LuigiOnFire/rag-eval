"""
Benchmark energy costs for RAG-related actions using CodeCarbon.
Outputs a JSON cost table for use in GreenTreeSearch and RAGEnv.
"""
import argparse
import json
import time
from codecarbon import EmissionsTracker
from tqdm import tqdm
import yaml
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import create_generator
from retriever import create_retriever, BM25Retriever


# Sample inputs (define at module level)
SAMPLE_QUERY = "Who directed the 2012 horror film Sinister?"
SAMPLE_CONTEXT_PASSAGES = [
    {"text": "Sinister is a 2012 horror film directed by Scott Derrickson.", "title": "Sinister"}
]
DECOMPOSE_PROMPT = "Break this question into sub-questions: Where was the director of Sinister born?"
REASONING_PROMPT = "Given the retrieved information, synthesize an answer..."


def measure_action(action_fn, n: int) -> dict:
    """
    Run action_fn n times, measure total energy, return stats.

    Returns:
    {
        "total_energy_kwh": float,
        "avg_joules": float,
        "n_samples": int,
        "avg_duration_s": float
    }    
    """
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
    start_time = time.time()
    tracker.start()
    for _ in tqdm(range(n), desc="Measuring action"):
        action_fn()
    emissions_kg_co2 = tracker.stop()  # Returns kg CO2, also populates tracker._total_energy
    duration = time.time() - start_time

    # Access energy consumed (kWh) from tracker's internal state
    energy_kwh = tracker._total_energy.kWh if tracker._total_energy else 0
    
    return {
        "total_energy_kwh": energy_kwh,
        "avg_wh": (energy_kwh * 1000) / n if energy_kwh else 0,
        "avg_joules": (energy_kwh * 3.6e6) / n if energy_kwh else 0,
        "n_samples": n,
        "avg_duration_s": duration / n,
        "total_emissions_kg_co2": emissions_kg_co2,
    }

def build_actions(slm, llm, retriever):
    """Build action dict after components are initialized."""
    return {
        0: ("Generate_and_End_SLM", lambda: slm.generate(SAMPLE_QUERY, SAMPLE_CONTEXT_PASSAGES)),
        1: ("Generate_and_End_LLM", lambda: llm.generate(SAMPLE_QUERY, SAMPLE_CONTEXT_PASSAGES)),
        2: ("Decompose_SLM", lambda: slm.generate(DECOMPOSE_PROMPT, [])),
        3: ("Decompose_LLM", lambda: llm.generate(DECOMPOSE_PROMPT, [])),
        4: ("Retrieve_Keyword", lambda: retriever.retrieve(SAMPLE_QUERY, top_k=5)),
        5: ("Retrieve_Dense", lambda: retriever.retrieve(SAMPLE_QUERY, top_k=5)),  # if you have dense
        6: ("Reason_LLM", lambda: llm.generate(REASONING_PROMPT, SAMPLE_CONTEXT_PASSAGES)),
    }

def main():
    # Parse args (config path, n samples, output path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per action")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON cost table")
    args = parser.parse_args()

    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load SLM and LLM separately
    print("Loading SLM:", config["slm"]["model_name"])
    slm = create_generator(config["slm"])
    
    print("Loading LLM:", config["llm"]["model_name"])
    llm = create_generator(config["llm"])

    # Load retriever with index and passages
    retriever = BM25Retriever()
    retriever.load_index(
        index_path=config["retriever"]["index_path"],
        passages_path=config["retriever"]["passages_path"]
    )

    ACTIONS = build_actions(slm, llm, retriever)

    # Measure samples costs for each action
    cost_table = {}
    for action_id, (name, fn) in ACTIONS.items():
        print(f"Benchmarking {name}...")
        cost_table[action_id] = measure_action(fn, n=args.n_samples)
        cost_table[action_id]["name"] = name

    # Save to JSON
    with open(args.output, "w") as f:
        json.dump(cost_table, f, indent=4)

    # Print summary table
    print("\nBenchmark Results:")
    print(f"{'Action':30} | {'Avg Joules':>12} | {'Avg Duration (s)':>16} | {'Samples':>8}")
    print("-"*80)
    for action_id, stats in cost_table.items(): 
        print(f"{stats['name']:30} | {stats['avg_joules']:12.2f} | {stats['avg_duration_s']:16.2f} | {stats['n_samples']:8d}")

if __name__ == "__main__":
    main()