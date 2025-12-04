"""
Behavior Cloning: Train encoder to predict actions from trajectories.

Phase 2 of the Green-DeepRAG training pipeline:
1. Load trajectories from GreenTreeSearch (Phase 1)
2. Convert to (state, action) training pairs
3. Fine-tune RoBERTa/DeBERTa for 8-class classification
4. Save trained controller for inference or Phase 3 (PPO)

The controller learns to mimic the cost-ordered search policy.

Note: The neural network predicts 8 legacy action classes (GENERATE_*_SLM/LLM,
DECOMPOSE_*_SLM/LLM, RETRIEVE_KEYWORD/DENSE, REASON_*_SLM/LLM). The new
parameterized ActionCall system maps to these legacy IDs for compatibility.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .green_tree_search import Action, ActionCall, Trajectory, load_cost_table

logger = logging.getLogger(__name__)


# Legacy action names for reporting (maps to neural network output classes)
ACTION_NAMES = [a.name for a in Action]


@dataclass
class ControllerConfig:
    """Configuration for the controller model."""
    model_name: str = "roberta-base"  # or "microsoft/deberta-v3-base"
    num_labels: int = 8  # 8 legacy action classes
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    seed: int = 42


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for (state, action) pairs."""
    
    def __init__(
        self, 
        states: List[str], 
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.states = states
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.states[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BehaviorCloning:
    """
    Train a controller via behavior cloning on GreenTreeSearch trajectories.
    
    The controller is a transformer encoder (RoBERTa/DeBERTa) with a 
    classification head that predicts the next action given the current state.
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """
        Initialize behavior cloning trainer.
        
        Args:
            config: Controller configuration (uses defaults if None)
        """
        self.config = config or ControllerConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
    
    def load_training_data(
        self, 
        trajectories_path: str,
        only_correct: bool = True
    ) -> Tuple[List[str], List[int]]:
        """
        Load training data from trajectory file.
        
        Args:
            trajectories_path: Path to trajectories JSON from GreenTreeSearch
            only_correct: Only use trajectories that found correct answers
            
        Returns:
            Tuple of (states, labels)
        """
        with open(trajectories_path, 'r') as f:
            data = json.load(f)
        
        states = []
        labels = []
        
        for traj in data["trajectories"]:
            if only_correct and not traj["correct"]:
                continue
            
            for step in traj["steps"]:
                states.append(step["state"])
                labels.append(step["action"])
        
        logger.info(f"Loaded {len(states)} training pairs from {trajectories_path}")
        
        # Log class distribution
        label_counts = {}
        for label in labels:
            action_name = ACTION_NAMES[label]
            label_counts[action_name] = label_counts.get(action_name, 0) + 1
        logger.info(f"Class distribution: {label_counts}")
        
        return states, labels
    
    def load_training_data_from_file(
        self, 
        training_path: str
    ) -> Tuple[List[str], List[int]]:
        """
        Load training data from pre-extracted training pairs file.
        
        Args:
            training_path: Path to .training.json file
            
        Returns:
            Tuple of (states, labels)
        """
        with open(training_path, 'r') as f:
            data = json.load(f)
        
        states = [item["text"] for item in data]
        labels = [item["label"] for item in data]
        
        logger.info(f"Loaded {len(states)} training pairs from {training_path}")
        
        return states, labels
    
    def prepare_datasets(
        self,
        states: List[str],
        labels: List[int],
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
        """
        Split data and create PyTorch datasets.
        
        Args:
            states: List of state strings
            labels: List of action labels
            test_size: Fraction for test set
            val_size: Fraction of remaining for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Check if we have enough samples for stratification
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        min_class_count = min(label_counts.values())
        
        # Use stratification only if all classes have >= 3 samples (enough for train/val/test)
        use_stratify = min_class_count >= 3 and len(states) >= 10
        
        # For very small datasets, use smaller splits
        if len(states) < 10:
            logger.warning(f"Very small dataset ({len(states)} samples). Using minimal splits.")
            # Use at least 1 for test/val, rest for train
            n_test = max(1, int(len(states) * test_size))
            n_val = max(1, int(len(states) * val_size))
            n_train = len(states) - n_test - n_val
            
            if n_train < 1:
                # If too few samples, just duplicate for train
                n_train = 1
                n_test = max(1, (len(states) - 1) // 2)
                n_val = len(states) - n_train - n_test
            
            # Simple index-based split
            train_states = states[:n_train]
            train_labels = labels[:n_train]
            val_states = states[n_train:n_train + n_val]
            val_labels = labels[n_train:n_train + n_val]
            test_states = states[n_train + n_val:]
            test_labels = labels[n_train + n_val:]
        else:
            # Split: first test, then val from remaining
            train_states, test_states, train_labels, test_labels = train_test_split(
                states, labels, test_size=test_size, random_state=self.config.seed, 
                stratify=labels if use_stratify else None
            )
            
            # Check again for val split
            train_label_counts = {}
            for label in train_labels:
                train_label_counts[label] = train_label_counts.get(label, 0) + 1
            min_train_count = min(train_label_counts.values()) if train_label_counts else 0
            use_stratify_val = min_train_count >= 2
            
            train_states, val_states, train_labels, val_labels = train_test_split(
                train_states, train_labels, test_size=val_size/(1-test_size), 
                random_state=self.config.seed, stratify=train_labels if use_stratify_val else None
            )
        
        logger.info(f"Split sizes - Train: {len(train_states)}, Val: {len(val_states)}, Test: {len(test_states)}")
        
        # Create datasets
        train_dataset = TrajectoryDataset(
            train_states, train_labels, self.tokenizer, self.config.max_length
        )
        val_dataset = TrajectoryDataset(
            val_states, val_labels, self.tokenizer, self.config.max_length
        )
        test_dataset = TrajectoryDataset(
            test_states, test_labels, self.tokenizer, self.config.max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0))
        }
    
    def train(
        self,
        train_dataset: TrajectoryDataset,
        val_dataset: TrajectoryDataset,
        output_dir: str = "./models/controller"
    ) -> Dict[str, Any]:
        """
        Train the controller model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model and checkpoints
            
        Returns:
            Training metrics
        """
        # Load model
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            problem_type="single_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            seed=self.config.seed,
            report_to="none",  # Disable wandb/tensorboard for now
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience
            )]
        )
        
        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        self.trainer.save_model(f"{output_dir}/final")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(f"{output_dir}/final")
        
        logger.info(f"Model saved to {output_dir}/final")
        
        return train_result.metrics
    
    def evaluate(
        self, 
        test_dataset: TrajectoryDataset
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on test set.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics and classification report
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        
        if labels is None:
            raise ValueError("No labels found in test dataset")
        
        # Handle tuple case
        if isinstance(labels, tuple):
            labels = labels[0]
        
        # Classification report
        report = classification_report(
            labels, preds, 
            target_names=ACTION_NAMES,
            zero_division=0
        )
        
        logger.info(f"\nClassification Report:\n{report}")
        
        # Convert to list for serialization
        labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        
        return {
            "metrics": predictions.metrics,
            "classification_report": report,
            "predictions": preds.tolist(),
            "labels": labels_list
        }
    
    def save_config(self, path: str):
        """Save controller config to JSON."""
        config_dict = {
            "model_name": self.config.model_name,
            "num_labels": self.config.num_labels,
            "max_length": self.config.max_length,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "action_names": ACTION_NAMES
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config saved to {path}")


class Controller:
    """
    Inference wrapper for the trained controller.
    
    Used during deployment or Phase 3 (PPO).
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Load trained controller for inference.
        
        Args:
            model_path: Path to saved model directory
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load tokenizer and model
        logger.info(f"Loading controller from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Controller loaded on {self.device}")
    
    def predict(self, state: str) -> Tuple[Action, np.ndarray]:
        """
        Predict the next action given current state.
        
        Args:
            state: Current state string (e.g., "[CLS] Query [SEP] Obs1 [SEP] ...")
            
        Returns:
            Tuple of (predicted Action, probability distribution)
        """
        # Tokenize
        inputs = self.tokenizer(
            state,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get predicted action
        action_id = int(np.argmax(probs))
        action = Action(action_id)
        
        return action, probs
    
    def predict_batch(self, states: List[str]) -> List[Tuple[Action, np.ndarray]]:
        """
        Predict actions for a batch of states.
        
        Args:
            states: List of state strings
            
        Returns:
            List of (Action, probabilities) tuples
        """
        # Tokenize batch
        inputs = self.tokenizer(
            states,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Get predictions
        results = []
        for i, prob in enumerate(probs):
            action_id = int(np.argmax(prob))
            action = Action(action_id)
            results.append((action, prob))
        
        return results
    
    def get_action_distribution(self, state: str) -> Dict[str, float]:
        """
        Get human-readable action probability distribution.
        
        Args:
            state: Current state string
            
        Returns:
            Dict mapping action names to probabilities
        """
        _, probs = self.predict(state)
        return {ACTION_NAMES[i]: float(p) for i, p in enumerate(probs)}
