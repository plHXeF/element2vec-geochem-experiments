"""Dynamic-width identities and two-family bounded value RoPE; no scalar fusion."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from ..models.transformer_encoder import ValueRoPEEncoderLayer, ValueRoPEMultiheadAttention
from .schema import FIELDS, LABELS


DIMENSIONS = [8, 16, 32, 64, 128, 256, 512, 1024]


def rotate_families(x, positions):
    if x.shape[-1] % 8 or positions.shape != (x.shape[0], x.shape[2], 2):
        raise ValueError("Two-family RoPE requires head_dim multiple of 8 and positions [B,L,2]")
    pairs = x.shape[-1] // 2
    fam = torch.tensor([0, 0, 1, 1], device=x.device).repeat(pairs // 4)
    freq = torch.tensor([1., .1, 1., .1], device=x.device).repeat(pairs // 4)
    angle = positions.float()[..., fam] * freq
    cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    return torch.stack((even*cos-odd*sin, even*sin+odd*cos), dim=-1).flatten(-2).to(x.dtype)


class FamilyAttention(ValueRoPEMultiheadAttention):
    def forward(self, tokens, attention_mask, rope_positions):
        b, l, _ = tokens.shape
        q, k, v = F.linear(tokens, self.in_proj_weight, self.in_proj_bias).chunk(3, -1)
        q, k, v = [a.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2) for a in (q, k, v)]
        q, k = rotate_families(q, rope_positions), rotate_families(k, rope_positions)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.)
        return self.out_proj(y.transpose(1, 2).contiguous().reshape(b, l, self.d_model))


class E010Model(nn.Module):
    architecture = "e010_family_rope"

    def __init__(self, config, label_counts, active_tags=None):
        super().__init__()
        self.config = copy.deepcopy(config)
        m = config["model"]
        d = int(m["embedding_dim"])
        if d not in DIMENSIONS or m["arm"] not in ("fixed", "native"):
            raise ValueError("Unknown E010 dimension or arm")
        h = int(m.get("fixed_token_dim", 768)) if m["arm"] == "fixed" else d
        heads = h // 64 if m["arm"] == "fixed" else max(1, h // 64)
        if heads < 1 or h % heads or (h // heads) % 8:
            raise ValueError("Invalid head dimensions")
        self.embedding_dim, self.d_model = d, h
        self.label_counts = dict(label_counts)
        self.label_offsets = {}; offset = 0
        for field in LABELS:
            self.label_offsets[field] = offset
            offset += self.label_counts[field]
        self.label_mask_ids = {field: offset+i for i, field in enumerate(LABELS)}
        self.numeric_tag_embedding = nn.Embedding(len(FIELDS)+1, d)
        self.pure_label_embedding = nn.Embedding(offset+len(LABELS), d)
        self.state_embedding = nn.Embedding(8, d)
        self.unit_embedding = nn.Embedding(2, d)
        self.value_mask_embedding = nn.Parameter(torch.zeros(d))
        self.projector = nn.Linear(d, h) if m["arm"] == "fixed" else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
        nn.init.normal_(self.cls_token, std=.02)
        self.layers = nn.ModuleList()
        for _ in range(int(m.get("layers", 12))):
            layer = ValueRoPEEncoderLayer(d_model=h, num_heads=heads, ffn_dim=4*h, dropout=float(m.get("dropout", .1)))
            layer.self_attn = FamilyAttention(h, heads, dropout=float(m.get("dropout", .1)))
            self.layers.append(layer)
        self.numeric_decoder = nn.Linear(h, 2)
        self.tag_decoder = nn.Linear(h, len(FIELDS))
        self.label_decoders = nn.ModuleDict({k: nn.Linear(h, max(1, v)) for k, v in label_counts.items()})
        active = torch.ones(len(FIELDS), dtype=torch.bool) if active_tags is None else torch.as_tensor(active_tags, dtype=torch.bool)
        self.register_buffer("active_tags", active)

    def make_inputs(self, batch):
        numeric = self.numeric_tag_embedding(batch["tag_ids"]) + self.state_embedding(batch["states"]) + self.unit_embedding(batch["families"])
        numeric = numeric + batch["value_mask"].unsqueeze(-1) * self.value_mask_embedding
        pure = self.pure_label_embedding(batch["label_ids"])
        static = torch.where(batch["numeric"].unsqueeze(-1), numeric, pure)
        tokens = self.projector(static)
        angle_kind = self.config.get("angle", "atan")
        z = batch["z"]
        theta = torch.atan(z / 2) if angle_kind == "atan" else (torch.pi/2)*torch.tanh(z/torch.pi)
        visible = batch["numeric"] & batch["mask"] & ~batch["value_mask"] & (batch["states"] >= 1) & (batch["states"] <= 5)
        theta = theta * visible
        positions = F.one_hot(batch["families"], 2) * theta.unsqueeze(-1)
        return static, tokens, positions

    def encode(self, batch, capture=False):
        static, tokens, pos = self.make_inputs(batch)
        b = tokens.shape[0]
        h = torch.cat((self.cls_token.expand(b, -1, -1), tokens), dim=1)
        mask = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=h.device), batch["mask"]), dim=1)
        pos = torch.cat((torch.zeros(b, 1, 2, device=h.device), pos), dim=1)
        snapshots = {}
        if capture:
            first = self.layers[0]
            q, k, _ = F.linear(first.norm1(h), first.self_attn.in_proj_weight, first.self_attn.in_proj_bias).chunk(3, -1)
            attn = first.self_attn
            for name, value in (("rotated_q", q), ("rotated_k", k)):
                value = value.reshape(b, h.shape[1], attn.num_heads, attn.head_dim).transpose(1, 2)
                snapshots[name] = rotate_families(value, pos).transpose(1, 2).flatten(-2)[:, 1:]
        for i, layer in enumerate(self.layers, 1):
            if self.training and self.config["model"].get("gradient_checkpointing", False):
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, mask, pos, use_reentrant=False)
            else:
                h = layer(h, mask, pos)
            if capture and i in (1, 6, len(self.layers)):
                snapshots[f"contextual_{i}"] = h[:, 1:]
        numeric = batch["numeric"] & batch["mask"]
        valid = numeric & (batch["states"] >= 1) & (batch["states"] <= 5)
        def pool(which):
            return (h[:, 1:] * which.unsqueeze(-1)).sum(1) / which.sum(1).clamp_min(1).unsqueeze(-1)
        return {"static": static, "projected": tokens, "positions": pos, "encoded": h,
                "mean": pool(numeric), "valid_mean": pool(valid), "cls": h[:, 0], **snapshots}

    def forward(self, batch):
        out = self.encode(batch)
        h = out["encoded"][:, 1:]
        pred = self.numeric_decoder(h)
        out["mu"] = pred[..., 0]
        out["scale"] = F.softplus(pred[..., 1]) + .05
        out["tag_logits"] = self.tag_decoder(h).masked_fill(~self.active_tags, -1e9)
        positions = batch["label_position"].clamp_min(0)
        selected = h[torch.arange(len(h), device=h.device), positions]
        out["label_logits"] = {k: head(selected) for k, head in self.label_decoders.items()}
        return out

    def parameter_counts(self):
        groups = {"embedding": [self.numeric_tag_embedding, self.pure_label_embedding, self.state_embedding, self.unit_embedding],
                  "projector": [self.projector], "backbone": [self.layers],
                  "heads": [self.numeric_decoder, self.tag_decoder, self.label_decoders]}
        out = {name: sum(p.numel() for m in modules for p in m.parameters()) for name, modules in groups.items()}
        out["special"] = self.cls_token.numel() + self.value_mask_embedding.numel()
        out["total"] = sum(p.numel() for p in self.parameters())
        return out
