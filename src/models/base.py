"""
Abstract base class for all time series classification models.

Defines the standard interface that all models must implement.
Supports both univariate (B, T) and multivariate (B, C, T) inputs.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for time series classification models.

    Contract:
    - All subclasses must implement forward(x: Tensor) -> Tensor
    - Input shape: (batch_size, sequence_length) or (batch_size, channels, sequence_length)
    - Output shape: (batch_size, num_classes) as logits (NOT probabilities)
    - Handles shape normalization in subclasses for consistency
    """

    def __init__(self, num_classes: int, **kwargs):
        """
        Initialize base model.

        Args:
            num_classes: Number of classification classes
            **kwargs: Additional arguments passed by config (model-specific params)
        """
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time series classification.

        Shape contract:
        - Input x: (batch_size, sequence_length) for univariate OR
                   (batch_size, n_channels, sequence_length) for multivariate
        - Output: (batch_size, num_classes) logits for cross-entropy loss

        Subclasses should:
        1. Normalize input shape to (batch, time, channels) internally
        2. Process through model layers
        3. Apply global pooling (temporal dimension reduction)
        4. Return logits (NOT softmax/probabilities)

        Args:
            x: Input tensor (see shape contract above)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        pass
