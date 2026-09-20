from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def output_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config["project"]["outputs_dir"])


def vocab_dir(config: dict[str, Any]) -> Path:
    configured = config.get("data", {}).get("vocab_dir")
    return resolve_path(configured) if configured else output_dir(config) / "vocab"


def ensure_output_dirs(config: dict[str, Any]) -> None:
    base = output_dir(config)
    for name in ["vocab", "checkpoints", "embeddings", "latents", "metrics", "logs"]:
        (base / name).mkdir(parents=True, exist_ok=True)
