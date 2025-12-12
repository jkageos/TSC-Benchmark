"""
PatchTST: A Time Series is Worth 16x16 Patches.

Reference: https://github.com/yuqinie98/PatchTST
Implements patch-based tokenization for time series.
"""

import math

import torch
import torch.nn as nn

from src.models.base import BaseModel


class PatchEmbedding(nn.Module):
    """Patch embedding module for time series."""

    def __init__(self, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, channels)

        Returns:
            Patch embeddings (batch_size, num_patches, d_model)
        """
        batch_size, seq_len, channels = x.shape

        # Calculate number of patches
        num_patches = max(1, (seq_len - self.patch_len) // self.stride + 1)

        # Create patches (take first channel for univariate)
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = min(start + self.patch_len, seq_len)  # Prevent overflow

            # Pad if last patch is shorter
            if end - start < self.patch_len:
                patch = x[:, start:end, 0]
                pad_len = self.patch_len - (end - start)
                patch = torch.nn.functional.pad(patch, (0, pad_len), mode="constant", value=0)
            else:
                patch = x[:, start:end, 0]

            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # (batch_size, num_patches, patch_len)
        patch_embeddings = self.projection(patches)

        return patch_embeddings


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with Flash Attention support."""

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


class TransformerBlock(nn.Module):
    """Transformer block for PatchTST."""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class PatchTST(BaseModel):
    """
    PatchTST: A Time Series is Worth 16x16 Patches.

    Reference: https://github.com/yuqinie98/PatchTST
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
        patch_len: int = 16,
        stride: int = 8,
        **kwargs,
    ):
        super().__init__(num_classes)

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Calculate number of patches
        num_patches = max(1, (input_length - patch_len) // stride + 1)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) for univariate
               or (batch_size, n_channels, sequence_length) for multivariate

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Handle input shapes - convert to (batch_size, seq_len, channels)
        if x.dim() == 2:
            # Univariate: (batch_size, seq_len) → (batch_size, seq_len, 1)
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            # Multivariate: (batch_size, channels, seq_len) → (batch_size, seq_len, channels)
            x = x.transpose(1, 2)

        # Create patch embeddings
        x = self.patch_embedding(x)

        # Add class token
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.positional_encoding[:, : x.size(1), :]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Use class token for classification
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.classifier(x)

        return x
