from __future__ import annotations

import ast
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = STAGE_DIR.parent
OUTPUT_DIR = STAGE_DIR / "outputs"


def _strip_inline_comment(text: str) -> str:
    in_quote = False
    quote_char = ""
    for i, ch in enumerate(text):
        if ch in {"'", '"'} and (i == 0 or text[i - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
        if ch == "#" and not in_quote:
            return text[:i].strip()
    return text.strip()


def _parse_scalar(value: str) -> Any:
    value = _strip_inline_comment(value)
    if value == "":
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = _strip_inline_comment(raw_line.strip())
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (STAGE_DIR / path).resolve()


def ensure_output_dirs() -> None:
    for sub in [
        "checkpoints",
        "logs",
        "metrics",
        "latents",
        "embeddings",
        "spaces",
        "figures",
        "report_fig",
        "preprocessed",
    ]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def selected_elements(config: dict[str, Any]) -> list[str]:
    cols = config.get("element_columns")
    if not cols:
        raise ValueError("Config must define element_columns")
    return [str(col) for col in cols]


def metadata_columns(config: dict[str, Any]) -> tuple[str, str]:
    return str(config.get("sample_id_col", "sample_id")), str(config.get("label_col", "type"))


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25
    iqr[~np.isfinite(iqr) | (iqr == 0)] = 1.0
    scaled = (values - med) / iqr
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return scaled.astype(np.float32), med.astype(np.float32), iqr.astype(np.float32)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "not available\n"
    use = df.head(max_rows).copy() if max_rows else df.copy()
    use = use.astype(object).where(pd.notna(use), "")
    headers = [str(c) for c in use.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in use.iterrows():
        rows.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in use.columns) + " |")
    return "\n".join(rows) + "\n"


def torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unavailable"


def config_snapshot(config: dict[str, Any]) -> None:
    ensure_output_dirs()
    write_json(config, OUTPUT_DIR / "logs" / "config_snapshot.json")
    with (OUTPUT_DIR / "logs" / "environment.txt").open("w", encoding="utf-8") as f:
        f.write(f"stage_dir={STAGE_DIR}\n")
        f.write(f"workspace_dir={WORKSPACE_DIR}\n")
        f.write(f"python={os.sys.executable}\n")
        f.write(f"torch_device={torch_device()}\n")
