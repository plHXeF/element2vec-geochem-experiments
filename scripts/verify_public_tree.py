#!/usr/bin/env python3
"""Fail when the public Git tree contains likely data, secrets, or oversized files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_BYTES = 95 * 1024 * 1024

FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet", ".feather"}
FORBIDDEN_PARTS = {
    ".conda", ".venv", "__pycache__", "cache", "preprocessed", "latents", "spaces",
    "fitting_pools", "pca_models", "random_projection_models",
}
FORBIDDEN_EXACT = {
    "pyrite_benchmark/data/raw/pyrite.xlsx",
    "legacy_whole_rock_geochem/geochem.db",
    "stage3_5_element2vec_v2/outputs/checkpoints/full_model.pt",
    "stage3_5_element2vec_v2/outputs/checkpoints/transformer_encoder.pt",
    "stage3_5_element2vec_v2/outputs/checkpoints/token_projector.pt",
    "stage3_5_element2vec_v2/outputs/checkpoints/decoders.pt",
}
SAMPLE_NAME_PATTERNS = (
    "sample_metadata.csv", "sample_input_token_flat.csv", "sample_encoded_token_flat.csv",
    "sample_raw_element_token_flat.csv", "umap3d_coordinates.csv",
)
SECRET_PATTERNS = [
    re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
ABSOLUTE_PATH_PATTERNS = [
    re.compile(rb"/Users/plhxef/"),
    re.compile(rb"[A-Za-z]:\\\\Geochem\\\\"),
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".sh"}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def visible_files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked", action="store_true", help="Audit only files in the Git index.")
    args = parser.parse_args()
    files = tracked_files() if args.tracked else visible_files()
    failures: list[str] = []

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        parts = set(path.relative_to(ROOT).parts)
        if rel in FORBIDDEN_EXACT:
            failures.append(f"forbidden file: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"dataset-like suffix: {rel}")
        if parts & FORBIDDEN_PARTS:
            failures.append(f"forbidden generated directory: {rel}")
        if any(rel.endswith(name) for name in SAMPLE_NAME_PATTERNS):
            failures.append(f"sample-level artifact name: {rel}")
        size = path.stat().st_size
        if size > MAX_GIT_BYTES:
            failures.append(f"oversized Git file ({size} bytes): {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES and size <= 20 * 1024 * 1024:
            data = path.read_bytes()
            for pattern in SECRET_PATTERNS:
                if pattern.search(data):
                    failures.append(f"possible secret: {rel}")
                    break
            # This verifier necessarily contains the forbidden path patterns as
            # literals. Do not make the detector fail on its own definitions.
            if rel != "scripts/verify_public_tree.py":
                for pattern in ABSOLUTE_PATH_PATTERNS:
                    if pattern.search(data):
                        failures.append(f"machine-specific absolute path: {rel}")
                        break

    if failures:
        print("Public-tree audit failed:")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        return 1
    print(f"Public-tree audit passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
