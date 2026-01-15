import torch
import torch.nn as nn

from src.models.base import BaseModel


class CNN(BaseModel):
    """
    Convolutional Neural Network for time series classification.

    Architecture: Conv1D layers with increasing channels → MaxPooling →
                  Flatten → FC layers → Output

    Captures local temporal patterns using 1D convolutions.

    Args:
        num_classes: Number of classification classes
        input_length: Length of the time series sequence
        input_channels: Number of input channels (default: 1 for univariate)
        kernel_size: Convolution kernel size (default: 3)
        num_filters: List of filter counts for each conv layer (default: [64, 128, 256])
        dropout_rate: Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        num_classes: int,
        input_length: int,
        input_channels: int = 1,
        kernel_size: int = 3,
        num_filters: list[int] | None = None,
        dropout_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(num_classes)

        # Default filter progression
        if num_filters is None:
            num_filters = [64, 128, 256]

        self.input_length = input_length
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Build convolutional layers
        conv_layers: list[nn.Module] = []
        in_channels = input_channels
        current_length = input_length

        for out_channels in num_filters:
            # Conv → BatchNorm → ReLU → MaxPool (halves sequence length)
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,  # Preserve length
                )
            )
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            in_channels = out_channels
            current_length = current_length // 2

        self.conv_layers = nn.Sequential(*conv_layers)

        # REPLACED: Flattening logic with Global Pooling logic
        # flattened_dim = num_filters[-1] * current_length  <-- Remove this strict dependency

        # Classification head receives 'num_filters[-1]' channels regardless of sequence length
        self.gap = nn.AdaptiveAvgPool1d(1)

        fc_layers: list[nn.Module] = [
            nn.Linear(num_filters[-1], 256),  # Input dim is now just number of filters
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        ]

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) or
               (batch_size, n_channels, sequence_length)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Ensure 3D: (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv layers: (B, C_out, L_out)
        x = self.conv_layers(x)

        # Global Average Pooling: (B, C_out, 1) -> (B, C_out)
        x = self.gap(x).squeeze(-1)

        x = self.fc_layers(x)

        return x
