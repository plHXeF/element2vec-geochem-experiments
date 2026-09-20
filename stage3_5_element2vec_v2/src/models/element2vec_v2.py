from __future__ import annotations

import torch
from torch import nn

from .decoders import Decoders
from .token_projector import TokenProjector
from .transformer_encoder import TransformerEncoder, ValueRoPETransformerEncoder


class Element2VecV2(nn.Module):
    architecture = "mlp_fusion"

    def __init__(
        self,
        num_numeric_tags: int,
        num_pure_label_tokens: int,
        pure_label_class_counts: dict[str, int],
        label_embedding_dim: int = 509,
        d_model: int = 768,
        projector_config: dict | None = None,
        transformer_config: dict | None = None,
    ):
        super().__init__()
        self.num_numeric_tags = int(num_numeric_tags)
        self.mask_numeric_tag_idx = int(num_numeric_tags)
        self.numeric_tag_embedding = nn.Embedding(num_numeric_tags + 1, label_embedding_dim)
        self.pure_label_embedding = nn.Embedding(num_pure_label_tokens, label_embedding_dim)
        projector_config = projector_config or {}
        transformer_config = transformer_config or {}
        self.token_projector = TokenProjector(**projector_config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.transformer_encoder = TransformerEncoder(**transformer_config)
        self.decoders = Decoders(d_model, num_numeric_tags, pure_label_class_counts)

    def make_token_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        kind = batch["kind"]
        num_emb = self.numeric_tag_embedding(batch["numeric_tag_ids"])
        pure_emb = self.pure_label_embedding(batch["pure_label_ids"])
        label_emb = torch.where((kind == 0).unsqueeze(-1), num_emb, pure_emb)
        return torch.cat(
            [
                label_emb,
                batch["values"].unsqueeze(-1),
                batch["mask_tag_flags"].unsqueeze(-1),
                batch["mask_value_flags"].unsqueeze(-1),
            ],
            dim=-1,
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_inputs = self.make_token_inputs(batch)
        projected = self.token_projector(token_inputs)
        bsz = projected.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        transformer_input = torch.cat([cls, projected], dim=1)
        cls_mask = torch.ones(bsz, 1, dtype=torch.bool, device=projected.device)
        attention_mask = torch.cat([cls_mask, batch["attention_mask"]], dim=1)
        encoded = self.transformer_encoder(transformer_input, attention_mask)
        return {
            "token_inputs": token_inputs,
            "projected_tokens": projected,
            "encoded": encoded,
            "attention_mask_with_cls": attention_mask,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.encode(batch)
        positions = batch["mask_positions"] + 1
        masked_hidden = out["encoded"][torch.arange(out["encoded"].shape[0], device=positions.device), positions]
        out["masked_hidden"] = masked_hidden
        out["decoder_outputs"] = self.decoders(masked_hidden)
        if 'numeric_prediction_mask' in batch:
            out['numeric_value_predictions'] = self.decoders.numeric_value_decoder(out['encoded'][:, 1:]).squeeze(-1)
        return out


class Element2VecValueRoPE(nn.Module):
    """Element2Vec variant where values enter only through Q/K rotary positions."""

    architecture = "value_rope"

    def __init__(
        self,
        num_numeric_tags: int,
        num_pure_label_tokens: int,
        pure_label_class_counts: dict[str, int],
        d_model: int = 768,
        transformer_config: dict | None = None,
    ):
        super().__init__()
        self.num_numeric_tags = int(num_numeric_tags)
        self.mask_numeric_tag_idx = int(num_numeric_tags)
        self.numeric_tag_embedding = nn.Embedding(num_numeric_tags + 1, d_model)
        self.pure_label_embedding = nn.Embedding(num_pure_label_tokens, d_model)
        self.numeric_value_mask_embedding = nn.Parameter(torch.empty(d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.numeric_value_mask_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.transformer_encoder = ValueRoPETransformerEncoder(**(transformer_config or {}))
        self.decoders = Decoders(d_model, num_numeric_tags, pure_label_class_counts)

    def make_token_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        kind = batch["kind"]
        numeric = self.numeric_tag_embedding(batch["numeric_tag_ids"])
        pure = self.pure_label_embedding(batch["pure_label_ids"])
        label_embeddings = torch.where((kind == 0).unsqueeze(-1), numeric, pure)
        value_is_masked = (kind == 0) & batch["mask_value_flags"].to(dtype=torch.bool)
        return label_embeddings + value_is_masked.unsqueeze(-1) * self.numeric_value_mask_embedding

    def make_rope_positions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        visible_numeric = (
            (batch["kind"] == 0)
            & batch["attention_mask"].to(dtype=torch.bool)
            & ~batch["mask_value_flags"].to(dtype=torch.bool)
        )
        positions = torch.where(visible_numeric, batch["values"], torch.zeros_like(batch["values"]))
        cls_positions = torch.zeros(
            positions.shape[0], 1, device=positions.device, dtype=positions.dtype
        )
        return torch.cat((cls_positions, positions), dim=1)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_inputs = self.make_token_inputs(batch)
        batch_size = token_inputs.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((cls, token_inputs), dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=token_inputs.device)
        attention_mask = torch.cat((cls_mask, batch["attention_mask"]), dim=1)
        rope_positions = self.make_rope_positions(batch)
        encoded = self.transformer_encoder(transformer_input, attention_mask, rope_positions)
        return {
            "token_inputs": token_inputs,
            # Compatibility alias: no projection is applied in this architecture.
            "projected_tokens": token_inputs,
            "rope_positions": rope_positions,
            "encoded": encoded,
            "attention_mask_with_cls": attention_mask,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.encode(batch)
        positions = batch["mask_positions"] + 1
        masked_hidden = out["encoded"][
            torch.arange(out["encoded"].shape[0], device=positions.device), positions
        ]
        out["masked_hidden"] = masked_hidden
        out["decoder_outputs"] = self.decoders(masked_hidden)
        if 'numeric_prediction_mask' in batch:
            out['numeric_value_predictions'] = self.decoders.numeric_value_decoder(out['encoded'][:, 1:]).squeeze(-1)
        return out


def build_model(config: dict, vocab) -> Element2VecV2 | Element2VecValueRoPE:
    model_cfg = config["model"]
    pure_counts = {field: len(values) for field, values in vocab.pure_vocab["classes"].items()}
    transformer_config = dict(model_cfg["transformer"])
    transformer_config.pop("use_positional_embedding", None)
    architecture = str(model_cfg.get("architecture", "mlp_fusion"))
    if architecture == "value_rope":
        return Element2VecValueRoPE(
            num_numeric_tags=len(vocab.numeric_vocab["numeric_tags"]),
            num_pure_label_tokens=len(vocab.pure_vocab["pure_tokens"]),
            pure_label_class_counts=pure_counts,
            d_model=int(model_cfg["d_model"]),
            transformer_config=transformer_config,
        )
    if architecture != "mlp_fusion":
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return Element2VecV2(
        num_numeric_tags=len(vocab.numeric_vocab["numeric_tags"]),
        num_pure_label_tokens=len(vocab.pure_vocab["pure_tokens"]),
        pure_label_class_counts=pure_counts,
        label_embedding_dim=int(config["tokenization"]["label_embedding_dim"]),
        d_model=int(model_cfg["d_model"]),
        projector_config=model_cfg["token_projector"],
        transformer_config=transformer_config,
    )
