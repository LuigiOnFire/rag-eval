#!/usr/bin/env python3
"""
Overnight training pipeline:
1. Generate more trajectories (500 samples)
2. Retrain controller with weighted loss
3. Test on 100 queries
4. Save all results

Usage:
    python scripts/overnight_training.py --num_samples 500
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_command(cmd: list, desc: str) -> tuple:
    """Run a command and return (success, output)."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        cmd,
        capture_output=False,  # Show output in real-time
        text=True
    )
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Overnight training pipeline")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of trajectory samples to generate")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--test_queries", type=int, default=100,
                       help="Number of test queries")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip trajectory generation (use existing)")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    python = str(project_root / "venv" / "bin" / "python")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "epochs": args.epochs,
        "test_queries": args.test_queries,
        "stages": {}
    }
    
    # Stage 1: Generate trajectories
    if not args.skip_generation:
        traj_output = f"results/trajectories_{args.num_samples}.json"
        success = run_command([
            python, "scripts/generate_trajectories.py",
            "--config", "config_local.yaml",
            "--num_samples", str(args.num_samples),
            "--output", traj_output,
            "--max_depth", "10",
            "--max_nodes", "50"
        ], f"Stage 1: Generating {args.num_samples} trajectories")
        
        results["stages"]["generation"] = {
            "success": success,
            "output_file": traj_output
        }
        
        if not success:
            print("ERROR: Trajectory generation failed!")
            sys.exit(1)
    else:
        traj_output = f"results/trajectories_{args.num_samples}.json"
        print(f"Skipping generation, using existing: {traj_output}")
    
    # Stage 2: Train controller with weighted loss
    model_output = f"models/controller_weighted_{args.num_samples}"
    success = run_command([
        python, "scripts/train_and_test_epochs.py",
        "--trajectories", traj_output,
        "--output_dir", model_output,
        "--epochs", str(args.epochs),
        "--use_weights"
    ], f"Stage 2: Training controller ({args.epochs} epochs)")
    
    results["stages"]["training"] = {
        "success": success,
        "model_dir": model_output
    }
    
    if not success:
        print("ERROR: Training failed!")
        sys.exit(1)
    
    # Stage 3: Test on queries
    test_output = f"results/controller_test_{args.num_samples}_{timestamp}.json"
    success = run_command([
        python, "scripts/test_controller.py",
        "--model", f"{model_output}/final",
        "--num_queries", str(args.test_queries),
        "--output", test_output
    ], f"Stage 3: Testing on {args.test_queries} queries")
    
    results["stages"]["testing"] = {
        "success": success,
        "output_file": test_output
    }
    
    # Load test results if available
    if Path(test_output).exists():
        with open(test_output, 'r') as f:
            test_data = json.load(f)
            results["final_accuracy"] = test_data.get("accuracy", 0)
            results["final_energy"] = test_data.get("avg_energy_mwh", 0)
    
    # Save summary
    summary_file = f"results/overnight_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OVERNIGHT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Summary saved to: {summary_file}")
    if "final_accuracy" in results:
        print(f"Final Accuracy: {results['final_accuracy']*100:.1f}%")
        print(f"Final Energy: {results['final_energy']:.1f} mWh/query")

if __name__ == "__main__":
    main()
