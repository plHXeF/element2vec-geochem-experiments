from __future__ import annotations

from pathlib import Path

import numpy as np


class PyriteArrayDataset:
    """Small array-backed dataset; torch Dataset behavior without importing torch at module import time."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.sample_id = data["sample_id"].astype(str)
        self.deposit_type = data["deposit_type"].astype(str)
        self.x = data["element_values"].astype("float32")
        self.visible = data["visible_mask"].astype("float32")
        self.y = data["label_id"].astype("int64")
        self.elements = data["elements"].astype(str)
        self.deposit_classes = data["deposit_classes"].astype(str)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.visible[idx], self.y[idx]
