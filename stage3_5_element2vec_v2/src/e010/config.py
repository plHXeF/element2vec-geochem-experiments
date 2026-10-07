"""One configuration factory for every E010 comparison; no copied model stages."""
from __future__ import annotations

import copy

from .model import DIMENSIONS


def configuration(arm="fixed", dimension=128, seed=42):
    if arm not in ("fixed", "native") or dimension not in DIMENSIONS:
        raise ValueError("Unsupported comparison")
    return {"e010_version": 1, "prepared_asset": "asset://e010_prepared", "expected_rows": 985996,
            "angle": "atan", "loss": "student_t",
            "model": {"arm": arm, "embedding_dim": dimension, "fixed_token_dim": 768,
                      "layers": 12, "dropout": .1, "gradient_checkpointing": False},
            "training": {"seed": seed, "epochs": 50, "effective_batch": 1024, "micro_batch": 32,
                         "device": "mps", "learning_rate": 2.5e-4, "min_lr": 3e-6, "warmup_steps": 500,
                         "weight_decay": .01, "gradient_clip": 1., "checkpoint_every": 100,
                         "retain_epochs": [1, 10, 25, 50], "log_every": 20,
                         "probe_size": 8192, "probe_every": 500}}


def ablation(name):
    c = copy.deepcopy(configuration())
    c["training"].update(epochs=2, retain_epochs=[1, 2])
    if name == "huber": c["loss"] = "huber"
    elif name == "tanh": c["angle"] = "tanh"
    elif name != "primary": raise ValueError(name)
    return c
