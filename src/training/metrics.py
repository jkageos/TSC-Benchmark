"""
Evaluation metrics for time series classification.

Computes:
- Accuracy
- F1 scores (macro and weighted)
- Precision and recall
- Confusion matrix
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class EpochMetrics:
    """Metrics tracked per epoch during training."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_precision: float
    val_recall: float


class MetricsTracker:
    """Lightweight metrics computation."""

    def __init__(self, num_classes: int, compute_confusion: bool = False):
        """
        Args:
            num_classes: Number of classes
            compute_confusion: Only compute confusion matrix on validation (not training)
        """
        self.num_classes = num_classes
        self.compute_confusion = compute_confusion
        self.reset()

    def reset(self) -> None:
        """Reset the metrics tracker for a new evaluation."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.confusion_mat: torch.Tensor | None = None

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Add batch predictions and targets.

        Args:
            outputs: Model logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        preds = outputs.argmax(dim=1)
        self.preds.append(preds.cpu())
        self.targets.append(targets.cpu())

    def get_accuracy(self) -> float:
        """Compute accuracy from accumulated predictions."""
        if not self.preds:
            return 0.0

        all_preds = torch.cat(self.preds)
        all_targets = torch.cat(self.targets)

        correct = (all_preds == all_targets).sum().item()
        total = len(all_targets)

        return correct / total if total > 0 else 0.0

    def compute(self) -> dict[str, Any]:
        """
        Compute all classification metrics.

        Returns:
            Dict with accuracy, F1 scores, precision, recall, confusion matrix
        """
        if not self.preds:
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "confusion_matrix": np.zeros((self.num_classes, self.num_classes)).tolist(),
            }

        # Concatenate all batches
        y_pred = torch.cat(self.preds).numpy()
        y_true = torch.cat(self.targets).numpy()

        # Compute metrics with zero_division handling for rare classes
        metrics: dict[str, Any] = {
            "accuracy": float((y_pred == y_true).mean()),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        if self.compute_confusion:
            metrics["confusion_matrix"] = confusion_matrix(
                y_true, y_pred, labels=list(range(self.num_classes))
            ).tolist()

        return metrics
