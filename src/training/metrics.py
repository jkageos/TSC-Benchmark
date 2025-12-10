"""
Evaluation metrics for time series classification.

Computes accuracy, F1-score, precision, recall, and confusion matrix.
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsTracker:
    """
    Tracks and computes classification metrics during training.

    Metrics computed:
    - Accuracy
    - F1-score (macro and weighted)
    - Precision (macro)
    - Recall (macro)
    - Confusion matrix
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classification classes
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all tracked predictions and targets."""
        self.all_predictions: list[int] = []
        self.all_targets: list[int] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with new batch of predictions.

        Args:
            predictions: Model predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        """
        # Convert logits to class predictions
        pred_classes = torch.argmax(predictions, dim=1)

        # Move to CPU and convert to numpy
        pred_np = pred_classes.cpu().numpy().tolist()
        target_np = targets.cpu().numpy().tolist()

        self.all_predictions.extend(pred_np)
        self.all_targets.extend(target_np)

    def compute(self) -> dict[str, Any]:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary containing all computed metrics
        """
        if len(self.all_predictions) == 0:
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "confusion_matrix": np.zeros((self.num_classes, self.num_classes)),
            }

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=list(range(self.num_classes))
            ),
        }

        return metrics

    def get_accuracy(self) -> float:
        """Quick accessor for accuracy only."""
        if len(self.all_predictions) == 0:
            return 0.0
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        return float(accuracy_score(y_true, y_pred))


class EpochMetrics:
    """
    Container for metrics from a single epoch.
    """

    def __init__(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float | None = None,
        val_accuracy: float | None = None,
        val_f1: float | None = None,
        val_precision: float | None = None,
        val_recall: float | None = None,
    ):
        self.epoch = epoch
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.val_loss = val_loss
        self.val_accuracy = val_accuracy
        self.val_f1 = val_f1
        self.val_precision = val_precision
        self.val_recall = val_recall

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "val_f1": self.val_f1,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
        }
