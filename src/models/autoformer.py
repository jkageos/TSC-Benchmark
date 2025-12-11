"""
Autoformer: Decomposition Transformers with Auto-Correlation for Time Series Forecasting.

Reference: https://github.com/thuml/Autoformer
Implements auto-correlation attention and time series decomposition.
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

    def __init__(self, mask_flag: bool = True, factor: int = 1, scale: float | None = None):
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Auto-correlation attention forward pass.

        Args:
            queries: (batch_size, n_heads, seq_len, d_k)
            keys: (batch_size, n_heads, seq_len, d_k)
            values: (batch_size, n_heads, seq_len, d_k)

        Returns:
            Attention output
        """
        batch_size, n_heads, seq_len, d_k = queries.shape

        # Compute auto-correlation via FFT
        queries_fft = torch.fft.rfft(queries, dim=-2)
        keys_fft = torch.fft.rfft(keys, dim=-2)

        # Multiply in frequency domain
        corr = queries_fft * torch.conj(keys_fft)
        corr = torch.fft.irfft(corr, n=seq_len, dim=-2)

        # Get top-k correlations
        topk = int(seq_len * self.factor)
        weights, indices = torch.topk(corr, topk, dim=-2, largest=True, sorted=True)
        weights = torch.softmax(weights, dim=-2)

        # Apply to values
        values_selected = torch.gather(
            values.unsqueeze(2).expand(-1, -1, topk, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_k),
        )

        output = torch.sum(weights.unsqueeze(-1) * values_selected, dim=-2)

        return output


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

        self.input_projection = nn.Linear(input_channels, d_model)
        self.positional_encoding = self._get_positional_encoding(d_model, max_seq_len)

        # Explicitly type as ModuleList of ModuleDict to satisfy type checkers
        self.encoder_layers: nn.ModuleList = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": AutoCorrelationMultiHead(d_model, num_heads, factor),
                        "norm1": nn.LayerNorm(d_model),
                        "dropout1": nn.Dropout(dropout),
                        "feed_forward": nn.Sequential(
                            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
                        ),
                        "norm2": nn.LayerNorm(d_model),
                        "dropout2": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def _get_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        x = self.input_projection(x)

        pe: torch.Tensor = self._get_positional_encoding(  # type: ignore[call-arg]
            int(x.size(-1)), int(x.size(1))
        ).to(x.device)
        x = x + pe[:, : x.size(1), :]

        for layer in self.encoder_layers:
            layer_dict = cast(nn.ModuleDict, layer)
            attn_out = cast(AutoCorrelationMultiHead, layer_dict["attention"])(x)
            x = cast(nn.LayerNorm, layer_dict["norm1"])(
                x + cast(nn.Dropout, layer_dict["dropout1"])(attn_out)
            )

            ff_out = cast(nn.Sequential, layer_dict["feed_forward"])(x)
            x = cast(nn.LayerNorm, layer_dict["norm2"])(
                x + cast(nn.Dropout, layer_dict["dropout2"])(ff_out)
            )

        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
