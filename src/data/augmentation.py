"""
Time series data augmentation techniques.

Reference: https://github.com/uchidalab/time_series_augmentation
"""

import torch
import torch.nn as nn


class TimeSeriesAugmentation(nn.Module):
    """
    Augmentation strategies for time series data.

    Applies random augmentations during training to improve generalization.
    """

    def __init__(
        self,
        jitter_strength: float = 0.03,
        scale_range: tuple[float, float] = (0.8, 1.2),
        magnitude_warp_strength: float = 0.1,
        time_warp_strength: float = 0.1,
        prob: float = 0.5,
    ):
        """
        Args:
            jitter_strength: Standard deviation for Gaussian noise
            scale_range: Range for random scaling
            magnitude_warp_strength: Strength of magnitude warping
            time_warp_strength: Strength of time warping
            prob: Probability of applying each augmentation
        """
        super().__init__()
        self.jitter_strength = jitter_strength
        self.scale_range = scale_range
        self.magnitude_warp_strength = magnitude_warp_strength
        self.time_warp_strength = time_warp_strength
        self.prob = prob

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        if torch.rand(1).item() > self.prob:
            return x
        noise = torch.randn_like(x) * self.jitter_strength
        return x + noise

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random scaling."""
        if torch.rand(1).item() > self.prob:
            return x
        scale_factor = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(self.scale_range[0], self.scale_range[1])
        return x * scale_factor

    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth magnitude warping using cubic spline."""
        if torch.rand(1).item() > self.prob:
            return x

        batch_size, seq_len = x.shape[0], x.shape[-1]

        # Generate random warp curve
        num_knots = max(4, seq_len // 20)
        warp = torch.randn(batch_size, num_knots, device=x.device) * self.magnitude_warp_strength

        # Interpolate to full sequence length
        warp_full = torch.nn.functional.interpolate(
            warp.unsqueeze(1), size=seq_len, mode="linear", align_corners=True
        ).squeeze(1)

        return x * (1 + warp_full.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations.

        Args:
            x: Input tensor (batch_size, channels, seq_len)

        Returns:
            Augmented tensor
        """
        x = self.jitter(x)
        x = self.scale(x)
        x = self.magnitude_warp(x)
        return x
