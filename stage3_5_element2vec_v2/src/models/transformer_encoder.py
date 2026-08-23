from __future__ import annotations

import torch
from torch import nn


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
