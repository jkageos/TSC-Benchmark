"""
Autoformer encoder for time series classification.

Key ideas:
- Auto-correlation attention via FFT for efficient long-range dependencies
- Top-k lags selection to focus attention
- Temporal pooling before classification
"""

import math
from typing import cast

import torch
import torch.nn as nn

from src.models.base import BaseModel


class AutoCorrelation(nn.Module):
    """
    Auto-correlation mechanism that captures dependencies across time lags.
    """

    def __init__(self, factor: int = 1):
        super().__init__()
        self.factor = factor

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Auto-correlation attention forward pass.

        Implementation:
        - FFT-based correlation (O(T log T)) instead of O(T^2) attention
        - Select top-k lags for sparse aggregation
        """
        batch_size, n_heads, seq_len, d_k = queries.shape

        # Upcast to float32 for FFT stability (mixed precision safe)
        q32 = queries.float()
        k32 = keys.float()

        # Frequency-domain multiply → inverse FFT gives correlation over lags
        queries_fft = torch.fft.rfft(q32, dim=-2)
        keys_fft = torch.fft.rfft(k32, dim=-2)

        corr = queries_fft * torch.conj(keys_fft)
        corr = torch.fft.irfft(corr, n=seq_len, dim=-2)

        # Top-k lag selection reduces noise and improves efficiency
        topk = max(1, int(seq_len * self.factor))
        weights, indices = torch.topk(corr, topk, dim=-2, largest=True, sorted=True)

        weights = torch.softmax(weights, dim=-2)

        values_expanded = values.unsqueeze(2).expand(-1, -1, topk, -1, -1)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_k)
        values_selected = torch.gather(values_expanded, 2, gather_idx)

        output = torch.sum(weights.unsqueeze(-1) * values_selected, dim=-2)

        return output.to(queries.dtype)


class AutoCorrelationMultiHead(nn.Module):
    """Multi-head auto-correlation attention."""

    def __init__(self, d_model: int, num_heads: int = 8, factor: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.auto_corr = AutoCorrelation(factor=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        Q = self.query(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        context = self.auto_corr(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fc_out(context)

        return output


class Autoformer(BaseModel):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation for Time Series.

    Reference: https://github.com/thuml/Autoformer
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
        factor: int = 1,
        max_seq_len: int = 5000,
        **kwargs,
    ):
        super().__init__(num_classes)

        # Channels → model dim; inputs become (B, T, C)
        self.input_projection = nn.Linear(input_channels, d_model)

        # Register fixed PE as buffer (excluded from optimizer/state_dict params)
        pe = self._get_positional_encoding(d_model, max_seq_len)
        self.register_buffer("pe", pe)

        # Encoder stack: autocorrelation + FFN with residual + norm
        self.encoder_layers: nn.ModuleList = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": AutoCorrelationMultiHead(d_model, num_heads, factor),
                        "norm1": nn.LayerNorm(d_model),
                        "dropout1": nn.Dropout(dropout),
                        "feed_forward": nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)),
                        "norm2": nn.LayerNorm(d_model),
                        "dropout2": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Temporal pooling before final classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def _get_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) or (B, C, T), normalized to (B, T, C)
        """
        # Shape normalization → projection → add PE
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        x = self.input_projection(x)

        pe: torch.Tensor = self.pe  # type: ignore
        x = x + pe[:, : x.size(1), :]

        # Encoder stack: autocorrelation + FFN with residual + norm
        for layer in self.encoder_layers:
            layer_dict = cast(nn.ModuleDict, layer)
            attn_out = cast(AutoCorrelationMultiHead, layer_dict["attention"])(x)
            x = cast(nn.LayerNorm, layer_dict["norm1"])(x + cast(nn.Dropout, layer_dict["dropout1"])(attn_out))

            ff_out = cast(nn.Sequential, layer_dict["feed_forward"])(x)
            x = cast(nn.LayerNorm, layer_dict["norm2"])(x + cast(nn.Dropout, layer_dict["dropout2"])(ff_out))

        # Pool over time, then classify
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
