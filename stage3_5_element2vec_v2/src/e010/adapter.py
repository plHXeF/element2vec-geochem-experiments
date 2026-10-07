"""E009-compatible explicit adapter for versioned E010 checkpoints."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

from .schema import RAW_COLUMNS, RAW_TAGS, FIELDS, INDEX, convert_frame, hidden_closure
from .training import load_model
from .prepare import digest_file
from .batching import make_batch, to_device


class E010CheckpointRepresentation:
    def __init__(self, spec, features=None, device="cpu", batch_size=32):
        self.spec = spec; self.name = spec["name"]; self.device = device; self.batch_size = batch_size
        self.model, self.checkpoint = load_model(spec["checkpoint"], device)
        self.model.eval(); self.model.requires_grad_(False)
        self.config = self.checkpoint["config"]
        self.fingerprint = digest_file(spec["checkpoint"])
        self.features = features
        if features is not None:
            aliases = dict(zip(RAW_COLUMNS, RAW_TAGS))
            self.selected = {INDEX[aliases.get(f, f)] for f in features}
        else: self.selected = None

    def fit_reference(self, train_records): return self

    @torch.inference_mode()
    def encode(self, records, hidden_tags=()):
        rules = self.checkpoint.get("censor_rules", [])
        frame = records.reindex(columns=RAW_COLUMNS+["canonical_ref_id"]).copy()
        if self.selected is not None:
            for col, tag in zip(RAW_COLUMNS, RAW_TAGS):
                if INDEX[tag] not in self.selected: frame[col] = np.nan
        values, states = convert_frame(frame, rules)
        if self.selected is not None:
            states[:, [j for j in range(len(FIELDS)) if j not in self.selected]] = 0
        data = SimpleNamespace(values=values, states=states, stats=self.checkpoint["statistics"],
                               labels=np.full((len(frame), 3), -1), sample_ids=np.arange(len(frame)),
                               weights=np.ones(len(frame)))
        result = []
        for start in range(0, len(frame), self.batch_size):
            batch = make_batch(data, np.arange(start, min(len(frame), start+self.batch_size)), self.model,
                               training=False, chemistry_only=True, hide=hidden_closure(hidden_tags))
            result.append(self.model.encode(to_device(batch, self.device))["mean"].cpu().numpy())
        return np.concatenate(result) if result else np.empty((0, self.model.d_model))

    def close(self):
        del self.model
        if self.device == "mps": torch.mps.empty_cache()
