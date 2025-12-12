"""Test-time augmentation for improved inference accuracy."""

import torch
import torch.nn as nn

from src.data.augmentation import TimeSeriesAugmentation


class TestTimeAugmentation:
    """Apply multiple augmentations at test time and average predictions."""

    def __init__(self, model: nn.Module, n_augmentations: int = 5):
        self.model = model
        self.n_augmentations = n_augmentations
        self.augmenter = TimeSeriesAugmentation(
            jitter_strength=0.01,  # Lighter augmentation for TTA
            scale_range=(0.95, 1.05),
            prob=0.8,
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with multiple augmentations and average.

        Args:
            x: Input tensor (batch_size, channels, seq_len)

        Returns:
            Averaged logits (batch_size, num_classes)
        """
        self.model.eval()

        # Original prediction
        predictions = [self.model(x)]

        # Augmented predictions
        for _ in range(self.n_augmentations - 1):
            x_aug = self.augmenter(x.clone())
            predictions.append(self.model(x_aug))

        # Average predictions
        return torch.stack(predictions).mean(dim=0)
