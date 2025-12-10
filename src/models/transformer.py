import math

import torch
import torch.nn as nn

from src.models.base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to capture temporal positions.

    Uses sinusoidal encoding as in the original transformer paper.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        # Type hint for the buffer to avoid type checker errors
        pe: torch.Tensor = self.pe  # type: ignore
        return x + pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch_size, seq_len, d_model)
        """
        batch_size = query.shape[0]

        # Linear transformations and split into multiple heads
        Q = (
            self.query(query)
            .reshape(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.key(key)
            .reshape(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.value(value)
            .reshape(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        # Final linear transformation
        output = self.fc_out(context)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward.
    """

    def __init__(
        self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class Transformer(BaseModel):
    """
    Transformer model for time series classification with self-attention.

    Includes positional encoding and multiple transformer encoder layers.
    Supports optional cross-attention for multi-view/augmented inputs.

    Args:
        num_classes: Number of classification classes
        input_length: Length of the time series sequence
        input_channels: Number of input channels (default: 1)
        d_model: Embedding dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 4)
        d_ff: Feed-forward network dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        max_seq_len: Maximum sequence length for positional encoding (default: 5000)
    """

    def __init__(
        self,
        num_classes: int,
        input_length: int,
        input_channels: int = 1,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        **kwargs,
    ):
        super().__init__(num_classes)

        self.input_length = input_length
        self.input_channels = input_channels
        self.d_model = d_model

        # Input projection: (batch_size, seq_len, channels) → (batch_size, seq_len, d_model)
        self.input_projection = nn.Linear(input_channels, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Global average pooling + classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) for univariate
               or (batch_size, n_channels, sequence_length) for multivariate
            mask: Optional attention mask

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Handle input shapes
        if x.dim() == 2:
            # Univariate: (batch_size, seq_len) → (batch_size, seq_len, 1)
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            # Multivariate: (batch_size, channels, seq_len) → (batch_size, seq_len, channels)
            x = x.transpose(1, 2)

        # Project input to d_model dimension
        # Input shape: (batch_size, seq_len, input_channels)
        # Output shape: (batch_size, seq_len, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply transformer encoder layers
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # Global average pooling: (batch_size, seq_len, d_model) → (batch_size, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)

        # Classification
        x = self.dropout(x)
        x = self.classifier(x)

        return x
