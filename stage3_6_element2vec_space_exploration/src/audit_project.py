from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from .common import ensure_directories, load_json, load_yaml, markdown_table, resolve_path, save_json


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def audit(config: dict) -> None:
    paths = ensure_directories(config)
    audit_dir = paths["audit"]
    source_cfg = load_yaml(config["source"]["model_config"])
    source_out = resolve_path(config["source"]["stage3_5_outputs"])
    numeric_vocab = load_json(source_out / "vocab" / "numeric_tag_vocab.json")
    pure_vocab = load_json(source_out / "vocab" / "pure_label_vocab.json")
    stats = pd.read_csv(source_out / "vocab" / "value_transform_stats.csv")
    checkpoint_dir = resolve_path(config["source"]["checkpoint"]).parent
    checkpoints = []
    for path in sorted(checkpoint_dir.glob("*.pt")):
        checkpoints.append({"file": str(path.relative_to(resolve_path("."))), "size_mb": path.stat().st_size / 1024**2})

    view = config["source"]["view_name"]
    db_path = resolve_path(config["source"]["sqlite_path"])
    metadata_fields = [
        "sample_id", "canonical_ref_id", "author", "data_source", "year",
        "latitude", "longitude", "rock_name", "protolith", "material",
        "rock_group", "rock_origin", "rock_facies", "rock_type", "age",
        "mg_number", "asi", "mali", "maficity", "cia", "wip", "spar", "qtzindex",
    ]
    numeric_db_cols = [item["db_column"] for item in source_cfg["data"]["numeric_fields"]]
    with sqlite3.connect(db_path) as conn:
        schema = pd.read_sql_query(f"PRAGMA table_info({_quote(view)})", conn)
        columns = set(schema["name"])
        n_samples = int(pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {_quote(view)}", conn).iloc[0, 0])
        coverage_expr = []
        for col in numeric_db_cols:
            coverage_expr.append(f"AVG(CASE WHEN {_quote(col)} IS NOT NULL THEN 1.0 ELSE 0.0 END) AS {_quote(col)}")
        coverage = pd.read_sql_query(f"SELECT {', '.join(coverage_expr)} FROM {_quote(view)}", conn).iloc[0]
        label_counts = {}
        for field in source_cfg["data"]["pure_label_fields"]:
            label_counts[field] = pd.read_sql_query(
                f"SELECT {_quote(field)} AS value, COUNT(*) AS n FROM {_quote(view)} "
                f"GROUP BY {_quote(field)} ORDER BY n DESC",
                conn,
            ).fillna({"value": "[MISSING]"})

    data_rows = []
    field_cfg = {item["db_column"]: item for item in source_cfg["data"]["numeric_fields"]}
    for col in numeric_db_cols:
        item = field_cfg[col]
        stat = stats.loc[stats["db_column"] == col].iloc[0]
        data_rows.append(
            {
                "token": item["name"],
                "db_column": col,
                "unit": item["unit"],
                "transform": "log10 + robust_z" if item.get("log_transform") else "robust_z",
                "coverage": float(coverage[col]),
                "median_transformed": float(stat["median"]),
                "iqr_transformed": float(stat["iqr"]),
            }
        )
    data_frame = pd.DataFrame(data_rows)
    data_frame.to_csv(audit_dir / "model_input_field_audit.csv", index=False, encoding="utf-8-sig")
    for field, counts in label_counts.items():
        counts.to_csv(audit_dir / f"{field}_counts.csv", index=False, encoding="utf-8-sig")

    available = {
        "checkpoint_files": checkpoints,
        "numeric_tag_embedding": {"available": True, "shape": [len(numeric_vocab["numeric_tags"]) + 1, source_cfg["tokenization"]["label_embedding_dim"]]},
        "pure_label_embedding": {"available": True, "shape": [len(pure_vocab["pure_tokens"]), source_cfg["tokenization"]["label_embedding_dim"]]},
        "projected_token": {"available": True, "shape_per_token": [source_cfg["model"]["d_model"]], "source": "model.encode()['projected_tokens']"},
        "contextual_token": {"available": True, "shape_per_token": [source_cfg["model"]["d_model"]], "source": "model.encode()['encoded'][:,1:]"},
        "sample_cls": {"available": True, "shape_per_sample": [source_cfg["model"]["d_model"]]},
        "sample_mean_numeric": {"available": True, "shape_per_sample": [source_cfg["model"]["d_model"]]},
        "sample_projected_mean_numeric": {"available": True, "shape_per_sample": [source_cfg["model"]["d_model"]]},
        "pooling": ["CLS", "mean_numeric_contextual", "mean_numeric_projected"],
        "input_modes": ["numeric_only", "masked_labels", "visible_labels_control"],
        "unavailable_model_inputs": {
            "isotopes": ["sr87_sr86", "epsilon_nd", "epsilon_hf", "pb206_pb204", "pb207_pb204", "pb208_pb204"],
            "typical_missing_numeric_tags": ["Cs", "Hf", "Ta", "Pr", "Sm", "Eu", "Gd", "Dy", "Er", "Lu", "Mo", "W", "Ho"],
            "pure_label_fields": ["tectonic_setting", "alteration", "province", "suite", "locality"],
        },
    }
    save_json(available, audit_dir / "available_representations.json")

    transformer = source_cfg["model"]["transformer"]
    model_audit = f"""# Element2Vec V2 model audit

## Source artifacts

- Model config: `{config['source']['model_config']}`
- Selected checkpoint: `{config['source']['checkpoint']}`
- Available checkpoints:

{markdown_table(pd.DataFrame(checkpoints))}

## Tokenization and architecture

- Numeric tags: {len(numeric_vocab['numeric_tags'])} observed-only fields plus one masked-tag index.
- Pure-label tokens: {len(pure_vocab['pure_tokens'])} field-value tokens across {', '.join(pure_vocab['pure_label_fields'])}.
- Raw token: {source_cfg['tokenization']['label_embedding_dim']} label dimensions + value + tag-mask flag + value-mask flag = {source_cfg['tokenization']['token_input_dim']} dimensions.
- Token projector: shared MLP from {source_cfg['tokenization']['token_input_dim']} to {source_cfg['model']['d_model']} dimensions.
- Transformer: {transformer['num_layers']} pre-LN layers, {transformer['num_heads']} heads, FFN {transformer['ffn_dim']}, no positional embedding.
- CLS: direct learned {source_cfg['model']['d_model']}-dimensional parameter.
- Training masking: one masked target per sample; numeric value, numeric tag, or one pure label.

## Extractable intermediate representations

The trained model exposes static embeddings, MLP-projected input tokens,
Transformer contextual tokens, CLS, contextual mean-numeric pooling, and an
additional pre-Transformer mean-numeric control. These layers are not treated
as interchangeable coordinate systems.

## Existing validation and visualization

- `train_log.csv` and fixed-train `probe_log.csv` exist.
- Stage 3.5 contains nearest-neighbor inspection and a latent benchmark script.
- Existing latent/metrics directories are empty; no completed V2 latent audit exists.
- Existing probes are not held-out validation and do not establish generalization.

## Leakage-relevant behavior

The checkpoint was trained on all samples with pure labels available during
most numeric masking tasks and as prediction targets during label masking.
Masking or removing labels at inference prevents direct input leakage but does
not make downstream fixed-latent cross-validation an independent generalization
test. All such results must be described as representation decodability.
"""
    (audit_dir / "model_audit.md").write_text(model_audit, encoding="utf-8")

    label_summary = []
    for field, counts in label_counts.items():
        label_summary.append(
            {
                "field": field,
                "classes_nonmissing": int((counts["value"] != "[MISSING]").sum()),
                "missing": int(counts.loc[counts["value"] == "[MISSING]", "n"].sum()),
                "largest_class": str(counts.iloc[0]["value"]),
                "largest_class_n": int(counts.iloc[0]["n"]),
            }
        )
    non_input_available = [c for c in metadata_fields if c in columns and c not in numeric_db_cols]
    data_audit = f"""# Element2Vec V2 data audit

## Data source

- SQLite: `{config['source']['sqlite_path']}`
- View: `{view}`
- Samples: {n_samples:,}
- View columns: {len(columns)}
- Model numeric inputs: {len(numeric_db_cols)}

The view is built from canonicalized references and a cross-publication
deduplication decision table. It requires coordinates and a complete core major
oxide panel. This improves baseline quality but does not eliminate publication,
locality, or suite leakage.

## Model-input coverage and transforms

{markdown_table(data_frame[['token', 'unit', 'transform', 'coverage']])}

## Label hierarchy

{markdown_table(pd.DataFrame(label_summary))}

`rock_group`, `rock_origin`, and `rock_type` are related hierarchical labels,
not independent supervision targets. In particular, `rock_type` comes from the
database computed table and is strongly coupled to major-element chemistry.

## Available external interpretation fields

{', '.join(f'`{field}`' for field in non_input_available)}

The database also exposes isotope columns, but they were not used as V2 tokens.
They may therefore be used as external interpretation targets where coverage is
sufficient. The current view does not expose explicit detection-limit states.
Non-positive trace values are discarded before log transformation, and extreme
robust-z values are clipped to ±{source_cfg['data']['value_transform']['clip_z']}.

## Missingness and units

- Missing numeric fields create no token, so token-set composition directly
  carries reporting and analytical-coverage information.
- Major fields are in weight percent; trace fields are in ppm.
- Major fields use direct robust-z. Trace fields use log10 then robust-z.
- There is no explicit below-detection token or unit embedding.

## Known unavailable targets

The trained vocabulary cannot test a complete REE sequence, Zr–Hf, Nb–Ta,
Rb–Cs–K, Mo–W, or Y–Ho. Tectonic setting, alteration, province, suite, and
locality are also absent as trained pure-label fields. These limitations will be
registered as negative/unavailable experiments rather than silently omitted.
"""
    (audit_dir / "data_audit.md").write_text(data_audit, encoding="utf-8")
    print(f"[audit] wrote {audit_dir / 'model_audit.md'}")
    print(f"[audit] wrote {audit_dir / 'data_audit.md'}")
    print(f"[audit] wrote {audit_dir / 'available_representations.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/exploration.yaml")
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    audit(load_yaml(args.config))


if __name__ == "__main__":
    main()
