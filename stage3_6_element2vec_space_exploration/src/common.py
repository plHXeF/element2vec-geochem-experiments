from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(payload: Any, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def load_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def ensure_directories(config: dict[str, Any]) -> dict[str, Path]:
    paths = {
        key.removesuffix("_dir"): resolve_path(value)
        for key, value in config["project"].items()
        if key.endswith("_dir")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_reproducible_seed(seed: int, num_threads: int = 8) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(max(int(num_threads), 1))
    except ImportError:
        pass


def stratified_sample_indices(
    frame: pd.DataFrame,
    column: str,
    n: int,
    seed: int,
    minimum_per_class: int = 20,
) -> np.ndarray:
    """Return deterministic, approximately proportional stratified indices.

    Rare classes receive up to ``minimum_per_class`` slots before the remaining
    budget is distributed proportionally. Missing labels form an explicit
    stratum so chemistry-only analyses are not restricted to labeled samples.
    """
    if n >= len(frame):
        return np.arange(len(frame), dtype=int)
    rng = np.random.default_rng(seed)
    labels = frame[column].fillna("[MISSING]").astype(str)
    groups = {label: idx.to_numpy(dtype=int) for label, idx in labels.groupby(labels).groups.items()}
    selected: list[int] = []
    remaining_groups: dict[str, np.ndarray] = {}
    for label, indices in groups.items():
        take = min(len(indices), minimum_per_class)
        chosen = rng.choice(indices, size=take, replace=False)
        selected.extend(chosen.tolist())
        remaining_groups[label] = np.setdiff1d(indices, chosen, assume_unique=False)
    budget = max(n - len(selected), 0)
    if budget:
        pool = np.concatenate(list(remaining_groups.values()))
        if len(pool) > budget:
            pool = rng.choice(pool, size=budget, replace=False)
        selected.extend(pool.tolist())
    if len(selected) > n:
        selected = rng.choice(np.asarray(selected), size=n, replace=False).tolist()
    return np.asarray(sorted(selected), dtype=int)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
