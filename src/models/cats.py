"""
CATS: Cross-Attention and Temporal Self-Attention for Time Series Classification.

Reference: https://github.com/dongbeank/CATS

Implements dual attention mechanisms:
1. Temporal self-attention: captures dependencies within the time series
2. Cross-attention: optionally attends to auxiliary inputs or augmented views
"""

import math

import torch
import torch.nn as nn

from src.models.base import BaseModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal positions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Precompute positional encodings (fixed, not learned)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        # Sine for even indices, cosine for odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe  # type: ignore
        return x + pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention mechanism."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]

        # Multi-head projection: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = self.query(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Weighted sum of values
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        return self.fc_out(context)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for attending to external context.

    Used when multiple views or augmented versions are available.
    Queries come from primary input, keys/values from auxiliary input.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, cross_input: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Primary input (queries)
            cross_input: Auxiliary input (keys/values). If None, falls back to self-attention
            mask: Optional attention mask
        """
        if cross_input is None:
            cross_input = x

        batch_size = x.shape[0]

        # Queries from x, Keys/Values from cross_input
        Q = self.query(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(cross_input).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(cross_input).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        return self.fc_out(context)


class CATSEncoderLayer(nn.Module):
    """
    CATS encoder layer combining self-attention and cross-attention.

    Architecture:
    1. Temporal self-attention on input
    2. Optional cross-attention to auxiliary input
    3. Feed-forward network
    Each sublayer has residual connection and layer norm.
    """

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Temporal self-attention
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (optional)
        self.cross_attention = CrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cross_input: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        self_attn_out = self.self_attention(x)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Cross-attention with residual (if auxiliary input provided)
        if cross_input is not None:
            cross_attn_out = self.cross_attention(x, cross_input)
            x = self.norm2(x + self.dropout2(cross_attn_out))

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x


class CATS(BaseModel):
    """
    CATS: Cross-Attention and Temporal Self-Attention for Time Series Classification.

    Reference: https://github.com/dongbeank/CATS

    Combines self-attention for temporal modeling with optional cross-attention
    for multi-view or augmented inputs. Falls back to pure self-attention when
    cross input is not provided.
    """

    def __init__(
        self,
        num_classes: int,
        input_length: int,
        input_channels: int = 1,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        **kwargs,
    ):
        super().__init__(num_classes)

        # Project input to model dimension
        self.input_projection = nn.Linear(input_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Stack of CATS encoder layers
        self.encoder_layers = nn.ModuleList(
            [CATSEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Classification head with global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) or
               (batch_size, n_channels, sequence_length)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Handle input shapes - convert to (batch_size, seq_len, channels)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        # Project and add positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Apply CATS encoder layers (no cross input for now)
        for layer in self.encoder_layers:
            x = layer(x, cross_input=None)

        # Global average pooling + classification
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
