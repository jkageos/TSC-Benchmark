"""Model ensemble for improved accuracy."""

import torch
import torch.nn as nn


class ModelEnsemble:
    """Ensemble multiple trained models for better predictions."""

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None):
        """
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

        for model in self.models:
            model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble prediction by weighted averaging of logits.

        Args:
            x: Input tensor

        Returns:
            Ensemble logits
        """
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model(x)
            predictions.append(pred * weight)

        return torch.stack(predictions).sum(dim=0)
