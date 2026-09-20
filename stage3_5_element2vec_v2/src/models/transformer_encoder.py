from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm: str = "pre_ln",
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=(norm == "pre_ln"),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~attention_mask
        return self.encoder(tokens, src_key_padding_mask=key_padding_mask)


def apply_rotary_embedding(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    base: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Rotate adjacent Q/K feature pairs using continuous scalar positions."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected [batch, heads, seq, head_dim], got {tuple(tensor.shape)}")
    if positions.shape != (tensor.shape[0], tensor.shape[2]):
        raise ValueError(
            f"Expected positions shape {(tensor.shape[0], tensor.shape[2])}, got {tuple(positions.shape)}"
        )
    if rotary_dim <= 0 or rotary_dim > tensor.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even number no larger than head_dim")

    work_dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
    pair_indices = torch.arange(0, rotary_dim, 2, device=tensor.device, dtype=work_dtype)
    inv_freq = torch.pow(torch.tensor(float(base), device=tensor.device, dtype=work_dtype), -pair_indices / rotary_dim)
    angles = positions.to(device=tensor.device, dtype=work_dtype).unsqueeze(-1) * float(scale) * inv_freq
    cos = angles.cos().to(dtype=tensor.dtype).unsqueeze(1)
    sin = angles.sin().to(dtype=tensor.dtype).unsqueeze(1)

    rotary = tensor[..., :rotary_dim].reshape(*tensor.shape[:-1], rotary_dim // 2, 2)
    first, second = rotary.unbind(dim=-1)
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
    rotated = rotated.flatten(start_dim=-2)
    if rotary_dim == tensor.shape[-1]:
        return rotated
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class ValueRoPEMultiheadAttention(nn.Module):
    """Multi-head self-attention with continuous-value RoPE on Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rotary_dim: int | None = None,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.rotary_dim = self.head_dim if rotary_dim is None else int(rotary_dim)
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be a positive even number no larger than head_dim")
        self.dropout = float(dropout)
        self.rope_base = float(rope_base)
        self.rope_scale = float(rope_scale)

        # Keep the same parameterization and initialization as nn.MultiheadAttention.
        reference = nn.MultiheadAttention(self.d_model, self.num_heads, dropout=self.dropout, batch_first=True)
        self.in_proj_weight = reference.in_proj_weight
        self.in_proj_bias = reference.in_proj_bias
        self.out_proj = reference.out_proj

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_model:
            raise ValueError(f"Expected tokens [batch, seq, {self.d_model}], got {tuple(tokens.shape)}")
        if attention_mask.shape != tokens.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, seq]")

        batch_size, seq_len, _ = tokens.shape
        qkv = F.linear(tokens, self.in_proj_weight, self.in_proj_bias)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = (split_heads(x) for x in (query, key, value))
        query = apply_rotary_embedding(
            query, rope_positions, self.rotary_dim, base=self.rope_base, scale=self.rope_scale
        )
        key = apply_rotary_embedding(
            key, rope_positions, self.rotary_dim, base=self.rope_base, scale=self.rope_scale
        )
        allowed_keys = attention_mask.to(dtype=torch.bool)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_keys,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attended)


class ValueRoPEEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm: str = "pre_ln",
        rotary_dim: int | None = None,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
    ):
        super().__init__()
        if norm != "pre_ln":
            raise ValueError("The value_rope architecture currently requires norm='pre_ln'")
        self.self_attn = ValueRoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rotary_dim=rotary_dim,
            rope_base=rope_base,
            rope_scale=rope_scale,
        )
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self.dropout1(
            self.self_attn(self.norm1(tokens), attention_mask, rope_positions)
        )
        normalized = self.norm2(tokens)
        feed_forward = self.linear2(self.dropout(self.activation(self.linear1(normalized))))
        return tokens + self.dropout2(feed_forward)


class ValueRoPETransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm: str = "pre_ln",
        rotary_dim: int | None = None,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
    ):
        super().__init__()
        layer = ValueRoPEEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation,
            norm=norm,
            rotary_dim=rotary_dim,
            rope_base=rope_base,
            rope_scale=rope_scale,
        )
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(int(num_layers)))

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, attention_mask, rope_positions)
        return tokens
