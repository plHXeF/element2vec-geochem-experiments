from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[3]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.io import load_config, output_dir, resolve_path


def inspect_embedding_file(path: Path, name_col: str, out_path: Path, top_k: int = 10) -> pd.DataFrame:
    df = pd.read_csv(path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    x = df[dim_cols].to_numpy(dtype=float)
    sim = cosine_similarity(x)
    rows = []
    for i, name in enumerate(df[name_col].astype(str)):
        order = np.argsort(-sim[i])
        order = [j for j in order if j != i][:top_k]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "query": name,
                    "neighbor": str(df.iloc[j][name_col]),
                    "rank": rank,
                    "cosine_similarity": float(sim[i, j]),
                }
            )
    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_5_element2vec_v2/configs/v2_base.yaml")
    parser.add_argument("--prefix", default="final", choices=["initial", "final"])
    parser.add_argument("--top-k", type=int, default=10)
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    config = load_config(args.config)
    out = output_dir(config)
    inspect_embedding_file(
        resolve_path(out / "embeddings" / f"{args.prefix}_numeric_tag_embeddings.csv"),
        "numeric_tag",
        out / "metrics" / f"{args.prefix}_numeric_tag_embedding_neighbors.csv",
        args.top_k,
    )
    inspect_embedding_file(
        resolve_path(out / "embeddings" / f"{args.prefix}_pure_label_embeddings.csv"),
        "pure_label_token",
        out / "metrics" / f"{args.prefix}_pure_label_embedding_neighbors.csv",
        args.top_k,
    )
    print("[inspect_embeddings] wrote nearest-neighbor tables under outputs/metrics")


if __name__ == "__main__":
    main()
