"""Validate transferred inputs and CUDA, then configure the HPC workspace.

Invoked by the batch bootstrap on a compute node; never regenerate input data
or rewrite the frozen Windows preparation run.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import sys

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def transferred_path(root, name):
    """Read Windows/POSIX relative manifests without mutating their bytes."""
    value = PureWindowsPath(name)
    if value.is_absolute() or value.drive or ".." in value.parts:
        raise ValueError(f"Unsafe relative input path: {name}")
    target = root.joinpath(*value.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Input path escapes frozen run: {name}")
    return target


def verify_inputs(root):
    card = json.loads((REPO/"experiments/E010/experiment.json").read_text())
    run_id = card["prepared_input"]["run_id"]
    prep = root/"runs/E010"/run_id
    manifest = json.loads((prep/"manifest.json").read_text())
    if manifest["run_id"] != run_id or not manifest.get("frozen") or manifest.get("status") != "completed":
        raise ValueError("Expected the completed, frozen local preparation run")
    checks = json.loads((prep/"artifact_checksums.json").read_text())
    for name in ("config.logical.yaml", "config.resolved.yaml", "environment.txt", "summary.md", "prepared_data/manifest.json"):
        if name not in {str(k).replace("\\", "/") for k in checks}:
            raise ValueError(f"Incomplete frozen checksum inventory: {name}")
    for name, expected in checks.items():
        if digest(transferred_path(prep, name)) != expected:
            raise ValueError(f"Transferred frozen artifact differs: {name}")
    prepared = prep/"prepared_data"
    if digest(prepared/"manifest.json") != card["prepared_input"]["prepared_fingerprint"]:
        raise ValueError("Prepared-data manifest differs from the registered input version")
    data = json.loads((prepared/"manifest.json").read_text())
    for name, expected in data["files"].items():
        if digest(transferred_path(prepared, name)) != expected:
            raise ValueError(f"Transferred prepared input differs: {name}")
    database = root/"data/whole_rock/whole-rock-v1/geochem.db"
    database_hash = digest(database)
    if database_hash != manifest["inputs"]["whole_rock"]["sha256"] or database_hash != data["source_sha256"]:
        raise ValueError("Database differs from the frozen preparation source")
    print(f"Transferred database and complete frozen preparation {run_id} verified", flush=True)
    return {"whole_rock": database.relative_to(root).as_posix(),
            "e010_prepared": prepared.relative_to(root).as_posix()}


def check_environment():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("HPC environment requires Python 3.11")
    requirements = (REPO/"environments/hpc-cu118/requirements.txt").read_text()
    for line in requirements.splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Dependency version mismatch: {name} requires {expected}")
    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "yaml", "openpyxl", "PIL"):
        importlib.import_module(name)
    import torch
    if torch.__version__ != "2.5.1+cu118" or torch.version.cuda != "11.8":
        raise RuntimeError("Expected torch 2.5.1+cu118; preserve the original environment for resume")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Batch allocation must expose exactly four CUDA GPUs")
    for slot in range(4):
        name = torch.cuda.get_device_name(slot)
        if "RTX A6000" not in name.upper():
            raise RuntimeError(f"Expected RTX A6000, got {name}")
        # Exercise CUDA forward/backward before registering or starting training.
        with torch.cuda.device(slot):
            x = torch.ones((32, 32), device=f"cuda:{slot}", requires_grad=True)
            (x @ x).sum().backward()
            torch.cuda.synchronize()
            del x
            torch.cuda.empty_cache()
        print(f"GPU {slot}: {name}; CUDA forward/backward passed", flush=True)


def configure(root, assets):
    path = root/"workspace.local.json"
    settings = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "assets": {}}
    for name, location in assets.items():
        existing = settings.get("assets", {}).get(name)
        if existing is not None and (root/existing).resolve() != (root/location).resolve():
            raise ValueError(f"Existing asset {name} points elsewhere; check workspace.local.json")
    settings.setdefault("assets", {}).update(assets)
    settings["python"] = sys.executable
    settings.setdefault("python_paths", [])
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(settings, indent=2)+"\n")
    temporary.replace(path)
    print(f"Configured {path}; inputs unchanged", flush=True)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("HPC setup is allowed only inside a Slurm compute allocation")
    root = Path(os.environ["GEOCHEM_WORKSPACE"]).expanduser().resolve()
    assets = verify_inputs(root)
    check_environment()
    configure(root, assets)


if __name__ == "__main__":
    main()
