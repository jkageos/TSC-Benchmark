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

        # Enable Flash Attention on CUDA
        self.use_flash = torch.cuda.is_available()

        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]

        # Multi-head projection: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = self.query(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 1. OPTIMIZATION: Use Flash Attention if available (2-4x faster)
        if self.use_flash:
            try:
                context = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False
                )
                context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
                return self.fc_out(context)
            except RuntimeError:
                # Fallback to standard attention if hardware/driver issues occur
                pass

        # 2. FALLBACK: Manual Scaled dot-product attention
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


class FeedForward(nn.Module):
    """Point-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CATSEncoderLayer(nn.Module):
    """
    CATS encoder layer combining self-attention and cross-attention.

    Updated to support 'Internal Augmentation' by accepting cross_input.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()

        # Use simple self-attention (inputs: x)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)

        # Use cross-attention (inputs: x, cross_input)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cross_input: torch.Tensor) -> torch.Tensor:
        # 1. Self Attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = residual + self.dropout(x)

        # 2. Cross Attention (attending to auxiliary view)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, cross_input)
        x = residual + self.dropout(x)

        # 3. Feed Forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        return x


class CATS(BaseModel):
    """
    CATS: Cross-Attention and Temporal Self-Attention for Time Series Classification.

    Benchmarking Mode:
    Uses 'Self-Cross-Attention' via internal projections to simulate multi-view
    architecture without external data augmentation.
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

        # Two separate input projections to create "views" from the same raw data
        self.primary_projection = nn.Linear(input_channels, d_model)
        self.auxiliary_projection = nn.Linear(input_channels, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        # CATS Encoder Layers
        self.encoder_layers = nn.ModuleList(
            [CATSEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Classification Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure (B, T, C) layout
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        # 1. Create Primary View
        x_primary = self.primary_projection(x)
        x_primary = self.positional_encoding(x_primary)

        # 2. Create Auxiliary View (Internal "Augmentation" via projection)
        x_aux = self.auxiliary_projection(x)
        x_aux = self.positional_encoding(x_aux)

        # 3. Apply Layers with Cross-Attention between projections
        for layer in self.encoder_layers:
            # Pass our internally generated auxiliary view as 'cross_input'
            x_primary = layer(x_primary, cross_input=x_aux)

        # Global Pooling & Classification
        x = x_primary.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
