"""Test-time augmentation for improved inference accuracy.

Strategy:
1. Apply multiple random augmentations to the same input
2. Get predictions for each augmented version
3. Average predictions (ensemble effect)

Typically improves accuracy by 0.5-2% with minimal code changes.
"""

import torch
import torch.nn as nn

from src.data.augmentation import TimeSeriesAugmentation


class TestTimeAugmentation:
    """Apply multiple augmentations at test time and average predictions."""

    def __init__(self, model: nn.Module, n_augmentations: int = 5):
        """
        Args:
            model: Trained model for inference
            n_augmentations: Number of augmented copies to generate
        """
        self.model = model
        self.n_augmentations = n_augmentations

        # Lighter augmentation for TTA (avoid distorting too much)
        self.augmenter = TimeSeriesAugmentation(
            jitter_strength=0.01,
            scale_range=(0.95, 1.05),
            prob=0.8,
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions with TTA ensemble.

        Args:
            x: Input batch (batch_size, seq_len) or (batch_size, channels, seq_len)

        Returns:
            Averaged predictions (batch_size, num_classes)
        """
        predictions = []

        # Original prediction (no augmentation)
        predictions.append(self.model(x))

        # Augmented predictions
        for _ in range(self.n_augmentations - 1):
            # Ensure 3D for augmentation
            x_aug = x.unsqueeze(1) if x.dim() == 2 else x
            x_aug = self.augmenter(x_aug)
            predictions.append(self.model(x_aug))

        # Average logits (more stable than averaging probabilities)
        return torch.stack(predictions).mean(dim=0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for predict() to match standard model interface."""
        return self.predict(x)
