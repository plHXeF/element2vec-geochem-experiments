#!/usr/bin/env python3
"""Read-only environment, data-path, checkpoint, and schema diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "checkpoint_manifest.json"
CHECKPOINT_DIR = ROOT / "stage3_5_element2vec_v2" / "outputs" / "checkpoints"
DB_PATH = ROOT / "legacy_whole_rock_geochem" / "geochem.db"
PYRITE_PATH = ROOT / "pyrite_benchmark" / "data" / "raw" / "pyrite.xlsx"

IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "PyYAML": "yaml",
    "PyTorch": "torch",
    "openpyxl": "openpyxl",
    "XGBoost": "xgboost",
    "UMAP": "umap",
    "Plotly": "plotly",
    "SHAP": "shap",
    "HDBSCAN": "hdbscan",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_imports() -> bool:
    ok = True
    print(f"Python: {sys.version.split()[0]}")
    for label, module_name in IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ok] {label}: {version}")
        except Exception as exc:  # diagnostic must continue through optional failures
            ok = False
            print(f"[missing] {label}: {exc}")
    return ok


def check_data(check_schema: bool) -> bool:
    print(f"[{'ok' if DB_PATH.is_file() else 'absent'}] whole-rock DB: {DB_PATH.relative_to(ROOT)}")
    print(f"[{'ok' if PYRITE_PATH.is_file() else 'absent'}] pyrite workbook: {PYRITE_PATH.relative_to(ROOT)}")
    if not check_schema or not DB_PATH.is_file():
        return True

    required = {
        "sample_id", "rock_group", "rock_origin", "rock_type", "sio2", "tio2",
        "al2o3", "fe2o3", "feo", "mgo", "cao", "na2o", "k2o", "p2o5",
        "mno", "loi", "co2", "rb_ppm", "sr_ppm", "ba_ppm", "zr_ppm",
        "nb_ppm", "y_ppm", "v_ppm", "cr_ppm", "ni_ppm", "cu_ppm",
        "zn_ppm", "la_ppm", "ce_ppm", "nd_ppm", "th_ppm", "u_ppm",
        "pb_ppm", "sc_ppm", "co_ppm", "ga_ppm", "yb_ppm",
    }
    uri = f"file:{DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('view','table') AND name=?",
            ("v_high_quality_dataset",),
        ).fetchone()
        if not exists:
            print("[missing] SQLite view: v_high_quality_dataset")
            return False
        columns = {row[1] for row in conn.execute("PRAGMA table_info(v_high_quality_dataset)")}
        count = conn.execute("SELECT COUNT(*) FROM v_high_quality_dataset").fetchone()[0]
    missing = sorted(required - columns)
    print(f"[ok] v_high_quality_dataset rows: {count:,}")
    if missing:
        print(f"[missing] required columns: {', '.join(missing)}")
        return False
    print(f"[ok] required V2 columns: {len(required)}")
    return True


def check_checkpoints() -> bool:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for entry in manifest["files"]:
        path = CHECKPOINT_DIR / entry["name"]
        if not path.is_file():
            ok = False
            print(f"[absent] checkpoint: {path.relative_to(ROOT)}")
            continue
        actual = sha256(path)
        good = actual == entry["sha256"]
        ok &= good
        print(f"[{'ok' if good else 'bad'}] checkpoint: {entry['name']}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-schema", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero when optional packages, datasets, or V2 checkpoints are absent.",
    )
    args = parser.parse_args()

    print("== Environment ==")
    imports_ok = check_imports()
    print("\n== Private inputs ==")
    data_ok = check_data(args.check_schema)
    print("\n== Element2Vec V2 checkpoints ==")
    checkpoints_ok = check_checkpoints()

    if not args.strict:
        print("\nDiagnostic complete. Missing private inputs/checkpoints are expected in a fresh clone.")
        return 0
    return 0 if imports_ok and data_ok and checkpoints_ok and DB_PATH.is_file() and PYRITE_PATH.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
