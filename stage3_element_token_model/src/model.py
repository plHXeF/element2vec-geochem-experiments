from __future__ import annotations

import torch
from torch import nn


class ElementTokenTransformer(nn.Module):
    def __init__(
        self,
        n_elements: int,
        n_deposit_types: int,
        element_embedding_dim: int = 32,
        deposit_embedding_dim: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_elements = int(n_elements)
        self.n_deposit_types = int(n_deposit_types)
        self.mask_deposit_index = int(n_deposit_types)
        self.d_model = int(d_model)

        self.element_embedding = nn.Embedding(n_elements, element_embedding_dim)
        self.deposit_embedding = nn.Embedding(n_deposit_types + 1, deposit_embedding_dim)

        self.element_projector = nn.Sequential(
            nn.Linear(element_embedding_dim + 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.deposit_projector = nn.Sequential(
            nn.Linear(deposit_embedding_dim + 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.element_head = nn.Linear(d_model, 1)
        self.deposit_head = nn.Linear(d_model, n_deposit_types)

    def make_tokens(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
        deposit_ids: torch.Tensor,
        label_visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        element_ids = torch.arange(self.n_elements, device=x.device).unsqueeze(0).expand(batch_size, -1)
        elem_emb = self.element_embedding(element_ids)
        elem_input = torch.cat([x.unsqueeze(-1), visible_mask.unsqueeze(-1), elem_emb], dim=-1)
        elem_tokens = self.element_projector(elem_input)

        dep_emb = self.deposit_embedding(deposit_ids)
        zero_value = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        dep_input = torch.cat([zero_value, label_visible_mask.reshape(-1, 1).to(x.dtype), dep_emb], dim=-1)
        dep_token = self.deposit_projector(dep_input).unsqueeze(1)
        return torch.cat([elem_tokens, dep_token], dim=1)

    def encode(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
        deposit_ids: torch.Tensor,
        label_visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.make_tokens(x, visible_mask, deposit_ids, label_visible_mask)
        encoded = self.encoder(tokens)
        return tokens, encoded

    def forward(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
        deposit_ids: torch.Tensor,
        label_visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_tokens, encoded = self.encode(x, visible_mask, deposit_ids, label_visible_mask)
        elem_encoded = encoded[:, : self.n_elements, :]
        deposit_encoded = encoded[:, self.n_elements, :]
        return {
            "input_tokens": input_tokens,
            "encoded": encoded,
            "element_pred": self.element_head(elem_encoded).squeeze(-1),
            "deposit_logits": self.deposit_head(deposit_encoded),
        }


def build_model(config: dict, n_elements: int, n_deposit_types: int) -> ElementTokenTransformer:
    return ElementTokenTransformer(
        n_elements=n_elements,
        n_deposit_types=n_deposit_types,
        element_embedding_dim=int(config.get("element_embedding_dim", 32)),
        deposit_embedding_dim=int(config.get("deposit_embedding_dim", 32)),
        d_model=int(config.get("d_model", 64)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 128)),
        dropout=float(config.get("dropout", 0.1)),
    )
