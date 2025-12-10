"""
Training orchestration for time series classification models.

Handles:
- Training loop with progress bars
- Validation and metric computation
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Device management (GPU/CPU)
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import EpochMetrics, MetricsTracker


class Trainer:
    """
    Main training orchestrator for time series classification.

    Features:
    - Automatic device detection (CUDA/CPU)
    - Early stopping with patience
    - Learning rate scheduling
    - Model checkpointing (best and latest)
    - Progress bars via tqdm
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: Optimizer,
        criterion: nn.Module,
        num_classes: int,
        epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        checkpoint_dir: str = "checkpoints",
        use_scheduler: bool = True,
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            test_loader: DataLoader for test/validation data
            optimizer: Optimizer instance
            criterion: Loss function
            num_classes: Number of classification classes
            epochs: Maximum number of training epochs
            patience: Early stopping patience (epochs without improvement)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            checkpoint_dir: Directory to save model checkpoints
            use_scheduler: Whether to use learning rate scheduler
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_classes = num_classes
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Learning rate scheduler
        self.scheduler = None
        if use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5
            )

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.history: list[EpochMetrics] = []

    def train_epoch(self) -> tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        metrics_tracker = MetricsTracker(self.num_classes)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]",
            leave=False,
        )

        for batch_X, batch_y in progress_bar:
            # Move data to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            metrics_tracker.update(outputs.detach(), batch_y)

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = metrics_tracker.get_accuracy()

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """
        Validate on test set.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        metrics_tracker = MetricsTracker(self.num_classes)

        progress_bar = tqdm(
            self.test_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Val]",
            leave=False,
        )

        for batch_X, batch_y in progress_bar:
            # Move data to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            # Track metrics
            total_loss += loss.item()
            metrics_tracker.update(outputs, batch_y)

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.test_loader)
        all_metrics = metrics_tracker.compute()

        return {
            "loss": avg_loss,
            "accuracy": all_metrics["accuracy"],
            "f1_macro": all_metrics["f1_macro"],
            "f1_weighted": all_metrics["f1_weighted"],
            "precision": all_metrics["precision"],
            "recall": all_metrics["recall"],
            "confusion_matrix": all_metrics["confusion_matrix"].tolist(),
        }

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "history": [m.to_dict() for m in self.history],
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def train(self) -> dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dictionary with final metrics and training history
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_accuracy = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Track metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                val_f1=val_metrics["f1_macro"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
            )
            self.history.append(epoch_metrics)

            # Learning rate scheduling
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step(val_metrics["accuracy"])
                new_lr = self.optimizer.param_groups[0]["lr"]
                if new_lr != old_lr:
                    print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

            # Print epoch summary
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            # Check for improvement
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
                self.save_checkpoint("latest.pt", is_best=True)
                print(f"✓ New best model! Accuracy: {self.best_val_accuracy:.4f}")
            else:
                self.epochs_without_improvement += 1
                self.save_checkpoint("latest.pt", is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # Load best model for final evaluation
        self.load_checkpoint("best_model.pt")
        final_metrics = self.validate()

        return {
            "best_val_accuracy": self.best_val_accuracy,
            "final_metrics": final_metrics,
            "history": [m.to_dict() for m in self.history],
            "total_epochs": self.current_epoch + 1,
        }

    def save_training_history(self, filepath: str) -> None:
        """
        Save training history to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        history_data = {
            "epochs": [m.to_dict() for m in self.history],
            "best_val_accuracy": self.best_val_accuracy,
            "total_epochs": self.current_epoch + 1,
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)
