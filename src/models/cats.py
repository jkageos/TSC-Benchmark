"""
CATS: Cross-Attention and Temporal Self-Attention for Time Series Classification.

Reference: https://github.com/dongbeank/CATS
Implements cross-attention between augmented views and temporal self-attention.
"""

import math

import torch
import torch.nn as nn

from src.models.base import BaseModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal positions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe  # type: ignore
        return x + pe[:, : x.size(1), :]


class TemporalSelfAttention(nn.Module):
    """Temporal self-attention mechanism with Flash Attention support."""

    def __init__(self, d_model: int, num_heads: int = 8, use_flash: bool = True):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash = False  # Disable globally for reproducibility

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]

        Q = self.query(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Use Flash Attention if available
        if self.use_flash:
            context = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            context = torch.matmul(attention, V)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fc_out(context)

        return output


class CrossAttention(nn.Module):
    """Cross-attention with Flash Attention support."""

    def __init__(self, d_model: int, num_heads: int = 8, use_flash: bool = True):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Disable Flash for CrossAttention (standard attention only)
        self.use_flash = False

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
        batch_size = query.shape[0]

        Q = self.query(query).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(key).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(value).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Standard attention only (no Flash) for stability
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fc_out(context)

        return output


class CATSEncoderLayer(nn.Module):
    """CATS encoder layer combining temporal self-attention and cross-attention."""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()

        self.temporal_self_attn = TemporalSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = CrossAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cross_input: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_output = self.temporal_self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        if cross_input is not None:
            cross_output = self.cross_attn(x, cross_input, cross_input, mask)
            x = self.norm2(x + self.dropout2(cross_output))
        else:
            cross_output = self.temporal_self_attn(x, mask)
            x = self.norm2(x + self.dropout2(cross_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional Flash Attention."""

    def __init__(self, d_model: int, num_heads: int = 8, use_flash: bool = True):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash = False  # Disable globally for reproducibility

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
        batch_size = query.shape[0]

        Q = self.query(query).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(key).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(value).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Use Flash Attention if available (2-4x faster)
        if self.use_flash:
            try:
                context = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False
                )
            except RuntimeError as e:
                print(f"⚠️  Flash Attention failed, falling back to standard attention: {e}")
                self.use_flash = False  # Disable for future calls
                # Compute standard attention as fallback
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float("-inf"))
                attention = torch.softmax(scores, dim=-1)
                context = torch.matmul(attention, V)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            context = torch.matmul(attention, V)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fc_out(context)

        return output


class CATS(BaseModel):
    """
    CATS: Cross-Attention and Temporal Self-Attention for Time Series Classification.

    Reference: https://github.com/dongbeank/CATS
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

        self.input_projection = nn.Linear(input_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList(
            [CATSEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, cross_input=None)

        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
