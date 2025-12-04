#!/usr/bin/env python3
"""
Train controller via behavior cloning on GreenTreeSearch trajectories.

Phase 2 of the Green-DeepRAG training pipeline.

Usage:
    python scripts/train_controller.py --trajectories results/trajectories.json --output models/controller
    
    # With custom model
    python scripts/train_controller.py --trajectories results/trajectories.json --model microsoft/deberta-v3-base
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.behavior_cloning import BehaviorCloning, ControllerConfig, Controller
from src.green_search import Action, ActionCall  # 8-class enum + parameterized actions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train controller via behavior cloning")
    parser.add_argument("--trajectories", type=str, required=True,
                       help="Path to trajectories JSON from GreenTreeSearch")
    parser.add_argument("--output", type=str, default="models/controller",
                       help="Output directory for trained model")
    parser.add_argument("--model", type=str, default="roberta-base",
                       help="Base model name (roberta-base, microsoft/deberta-v3-base)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--only_correct", action="store_true", default=True,
                       help="Only train on correct trajectories")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Fraction of data for test set")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure controller
    config = ControllerConfig(
        model_name=args.model,
        num_labels=8,  # 8 action classes
        max_length=args.max_length,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    logger.info(f"Controller config: {config}")
    
    # Initialize trainer
    bc = BehaviorCloning(config)
    
    # Load training data
    trajectories_path = Path(args.trajectories)
    
    # Check if this is a .training.json file (pre-extracted pairs) or raw trajectories
    if str(trajectories_path).endswith('.training.json'):
        logger.info(f"Loading pre-extracted training pairs from {trajectories_path}")
        states, labels = bc.load_training_data_from_file(str(trajectories_path))
    else:
        # Check if .training.json exists for this trajectory file
        training_path = Path(str(trajectories_path).replace('.json', '.training.json'))
        if training_path.exists():
            logger.info(f"Found pre-extracted training pairs at {training_path}")
            states, labels = bc.load_training_data_from_file(str(training_path))
        else:
            logger.info(f"Loading trajectories from {trajectories_path}")
            states, labels = bc.load_training_data(str(trajectories_path), only_correct=args.only_correct)
    
    if len(states) < 10:
        logger.warning(f"Only {len(states)} training samples. Consider generating more trajectories.")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = bc.prepare_datasets(
        states, labels, test_size=args.test_size
    )
    
    # Train
    logger.info("Starting training...")
    train_metrics = bc.train(train_dataset, val_dataset, str(output_dir))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    eval_results = bc.evaluate(test_dataset)
    
    # Save config
    bc.save_config(str(output_dir / "config.json"))
    
    # Save evaluation results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "train_metrics": train_metrics,
            "test_metrics": eval_results["metrics"],
            "classification_report": eval_results["classification_report"]
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BEHAVIOR CLONING TRAINING COMPLETE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"\nTest Results:")
    print(f"  Accuracy: {eval_results['metrics'].get('test_accuracy', 'N/A'):.4f}")
    print(f"  F1 (macro): {eval_results['metrics'].get('test_f1_macro', 'N/A'):.4f}")
    print(f"  F1 (weighted): {eval_results['metrics'].get('test_f1_weighted', 'N/A'):.4f}")
    print(f"\nModel saved to: {output_dir / 'final'}")
    print("="*60)
    
    # Quick inference test
    print("\n--- Quick Inference Test ---")
    controller = Controller(str(output_dir / "final"))
    
    test_state = "[CLS] Who directed the movie Sinister? [SEP]"
    action, probs = controller.predict(test_state)
    print(f"State: {test_state}")
    print(f"Predicted action: {action.name}")
    print(f"Probabilities: {controller.get_action_distribution(test_state)}")


if __name__ == "__main__":
    main()
