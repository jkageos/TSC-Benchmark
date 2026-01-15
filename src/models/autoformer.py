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
    Auto-correlation mechanism with efficient Time Delay Aggregation.
    """

    def __init__(self, factor: int = 1):
        super().__init__()
        self.factor = factor  # Controls top-k selection (c in paper)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Auto-correlation forward pass.

        Args:
            queries: (B, H, L, D)
            keys:    (B, H, L, D)
            values:  (B, H, L, D)
        """
        B, H, L, D = queries.shape

        # 1. Period-based dependencies via FFT
        # (B, H, L, D) -> (B, H, L//2+1, D)
        # FIX: Explicitly cast to float32. cuFFT does not support half precision for
        # non-power-of-two signal sizes, and float32 is better for FFT stability anyway.
        q_fft = torch.fft.rfft(queries.float(), dim=2)
        k_fft = torch.fft.rfft(keys.float(), dim=2)

        # Correlation in frequency domain
        # Result is (B, H, L, D) after irfft
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=2)

        # 2. Time Delay Aggregation
        # Aggregating values based on the period (mean over D dim for top-k selection)
        # corr shape: (B, H, L, D) -> (B, H, L)
        corr_mean = torch.mean(corr, dim=-1)

        # Select top-k lags
        # limits k to avoid OOM on very long sequences
        k = min(int(self.factor * math.log(L)), L)
        # weights: (B, H, k), delays: (B, H, k)
        weights, delays = torch.topk(corr_mean, k, dim=-1)

        # Softmax over selected correlation weights
        weights = torch.softmax(weights, dim=-1)

        # 3. Efficient Roll (Time Delay)
        # We need V(t - tau). We calculate indices: (t - tau) % L

        # (1, 1, L, 1)
        time_indices = torch.arange(L, device=queries.device).reshape(1, 1, L, 1)

        # (B, H, 1, k)
        delays = delays.unsqueeze(2)

        # Broadcast subtract: (B, H, L, k) -> Indices to gather from V
        roll_indices = (time_indices - delays) % L

        # Expand indices for gathered values dimensions: (B, H, L, k, D)
        # We need to gather from 'values' which is (B, H, L, D)
        # We repeat the gather indices across the D dimension
        roll_indices = roll_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)

        # Expand values to accommodate k: (B, H, L, k, D)
        values_expanded = values.unsqueeze(3).expand(-1, -1, -1, k, -1)

        # Gather elements: For every query pos, we pick k delayed values
        # (B, H, L, k, D)
        values_rolled = torch.gather(values_expanded, 2, roll_indices)

        # 4. Weighted Sum
        # weights: (B, H, k) -> (B, H, 1, k, 1) to broadcast
        weights = weights.unsqueeze(2).unsqueeze(-1)

        # (B, H, L, D)
        # Ensure weights match values dtype to maintain mixed precision if active
        output = torch.sum(values_rolled * weights.to(values_rolled.dtype), dim=3)

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
        batch_size, seq_len, _ = x.shape

        # Linear projections
        # (B, L, H, k) -> (B, H, L, k)
        Q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # Use new AutoCorrelation
        context = self.auto_corr(Q, K, V)

        # Reshape back: (B, H, L, k) -> (B, L, H, k) -> (B, L, D_model)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

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
