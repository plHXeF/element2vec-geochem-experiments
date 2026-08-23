from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from ..utils.io import ensure_output_dirs, load_config, output_dir, resolve_path, save_json
from .transforms import fit_transform_stats


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_vocab(config: dict) -> dict:
    ensure_output_dirs(config)
    out = output_dir(config)
    db_path = resolve_path(config["data"]["sqlite_path"])
    view = config["data"]["view_name"]
    numeric_fields = config["data"]["numeric_fields"]
    pure_fields = config["data"]["pure_label_fields"]
    sample_col = config["data"]["sample_id_column"]
    selected_cols = [sample_col]
    selected_cols += [f["db_column"] for f in numeric_fields]
    selected_cols += list(pure_fields)
    query = f"SELECT {', '.join(_quote(c) for c in selected_cols)} FROM {_quote(view)}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    transform_cfg = config["data"]["value_transform"]
    stats_rows = []
    for item in numeric_fields:
        stats_rows.append(
            fit_transform_stats(
                df[item["db_column"]],
                field_name=item["name"],
                db_column=item["db_column"],
                log_transform=bool(item.get("log_transform", False)),
                eps=float(transform_cfg.get("eps", 1e-9)),
                min_iqr=float(transform_cfg.get("min_iqr", 1e-6)),
            )
        )
    stats = pd.DataFrame(stats_rows)
    stats_path = out / "vocab" / "value_transform_stats.csv"
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    numeric_tags = [item["name"] for item in numeric_fields]
    numeric_vocab = {
        "numeric_tags": numeric_tags,
        "tag_to_idx": {name: i for i, name in enumerate(numeric_tags)},
        "mask_numeric_tag": "[MASK_NUMERIC_TAG]",
        "mask_numeric_tag_idx": len(numeric_tags),
        "fields": numeric_fields,
    }

    pure_token_to_idx: dict[str, int] = {}
    pure_tokens: list[str] = []
    pure_label_classes: dict[str, list[str]] = {}
    for field in pure_fields:
        values = (
            df[field]
            .dropna()
            .astype(str)
            .map(lambda x: x.strip())
            .loc[lambda s: s.ne("")]
            .sort_values()
            .unique()
            .tolist()
        )
        pure_label_classes[field] = values
        tokens = [f"{field}:{value}" for value in values] + [f"{field}:[MASK]"]
        for token in tokens:
            pure_token_to_idx[token] = len(pure_tokens)
            pure_tokens.append(token)

    pure_vocab = {
        "pure_label_fields": pure_fields,
        "pure_tokens": pure_tokens,
        "token_to_idx": pure_token_to_idx,
        "classes": pure_label_classes,
        "class_to_idx": {
            field: {value: i for i, value in enumerate(values)}
            for field, values in pure_label_classes.items()
        },
        "mask_tokens": {field: f"{field}:[MASK]" for field in pure_fields},
    }
    save_json(numeric_vocab, out / "vocab" / "numeric_tag_vocab.json")
    save_json(pure_vocab, out / "vocab" / "pure_label_vocab.json")
    summary = {
        "view_name": view,
        "num_samples": int(len(df)),
        "num_numeric_tags": len(numeric_tags),
        "num_pure_label_tokens": len(pure_tokens),
        "pure_label_class_counts": {k: len(v) for k, v in pure_label_classes.items()},
    }
    save_json(summary, out / "vocab" / "vocab_summary.json")
    print(f"[build_vocab] samples={len(df)} numeric_tags={len(numeric_tags)} pure_tokens={len(pure_tokens)}")
    print(f"[build_vocab] wrote {stats_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_5_element2vec_v2/configs/v2_base.yaml")
    args = parser.parse_args()
    build_vocab(load_config(args.config))


if __name__ == "__main__":
    main()
