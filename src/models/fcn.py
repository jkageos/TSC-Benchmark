import torch
import torch.nn as nn

from src.models.base import BaseModel


class FCN(BaseModel):
    """
    Fully Connected Neural Network for time series classification.

    Architecture: Input → Flatten → FC1 → BatchNorm → ReLU → Dropout →
                  FC2 → BatchNorm → ReLU → Dropout → ... → Output

    Args:
        num_classes: Number of classification classes
        input_length: Length of the time series sequence
        input_channels: Number of input channels (default: 1 for univariate)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability (default: 0.5)
        use_batch_norm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        num_classes: int,
        input_length: int,
        input_channels: int = 1,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__(num_classes)

        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_length = input_length
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Calculate flattened input dimension
        layers: list[nn.Module] = []
        prev_dim = input_length * input_channels

        # Build hidden layers with optional batch norm and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Classification head
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) or
               (batch_size, n_channels, sequence_length)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Normalize shape to (batch_size, sequence_length, n_channels)
        if x.dim() == 2:
            # Univariate: (batch_size, seq_len) → (batch_size, seq_len, 1)
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            # Multivariate: (batch_size, channels, seq_len) → (batch_size, seq_len, channels)
            x = x.transpose(1, 2)

        # Flatten all dimensions except batch: (batch_size, seq_len * n_channels)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)

        return self.network(x)
