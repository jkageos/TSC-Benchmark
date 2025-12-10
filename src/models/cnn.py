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
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,  # Same padding
                )
            )
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            in_channels = out_channels
            current_length = current_length // 2

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened dimension after conv layers
        flattened_dim = num_filters[-1] * current_length

        # Build fully connected layers
        fc_layers: list[nn.Module] = [
            nn.Linear(flattened_dim, 256),
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
        # Ensure input is 3D: (batch_size, channels, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x
