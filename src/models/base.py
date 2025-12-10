from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all time series classification models.

    All models must implement forward() and define num_classes.
    Supports both univariate and multivariate time series.
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time series classification.

        Args:
            x: Input tensor of shape (batch_size, sequence_length) for univariate
               or (batch_size, n_channels, sequence_length) for multivariate

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        pass
