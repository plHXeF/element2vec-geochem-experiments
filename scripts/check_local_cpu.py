"""Inspect the locked local CPU environment; save only with a registered run."""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geochem_workspace as ws


def inspect():
    definition = ws.REPO_ROOT / "environments/local-cpu"
    expected_python = (definition/".python-version").read_text().strip()
    modules = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
               "scikit-learn": "sklearn", "torch": "torch", "matplotlib": "matplotlib",
               "seaborn": "seaborn", "PyYAML": "yaml", "Pillow": "PIL", "openpyxl": "openpyxl"}
    for name in modules.values(): importlib.import_module(name)
    import torch
    lock = tomllib.loads((definition/"uv.lock").read_text())
    installed = {name: metadata.version(name) for name in modules}
    locked = {p["name"].lower(): p for p in lock["package"]}
    errors = []
    if platform.python_version() != expected_python: errors.append("Python differs from the pinned patch version")
    if Path(sys.prefix).resolve() != (definition/".venv").resolve(): errors.append("Wrong project environment")
    if not torch.__version__.endswith("+cpu") or torch.version.cuda is not None:
        errors.append("Expected CPU-only PyTorch")
    for name, version in installed.items():
        if locked[name.lower()]["version"] != version: errors.append(f"Installed version differs from lock: {name}")
    if locked["torch"]["source"] != {"registry": "https://download.pytorch.org/whl/cpu"}:
        errors.append("PyTorch is not locked to the official CPU index")
    extras = {m: util.find_spec(m) is not None for m in ("xgboost", "umap", "shap", "hdbscan", "plotly")}
    return {"passed": not errors, "errors": errors, "python": platform.python_version(),
            "executable": sys.executable, "base_python": sys.base_prefix, "packages": installed,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "optional_analysis_installed": extras,
            "lock_sha256": hashlib.sha256((definition/"uv.lock").read_bytes()).hexdigest(),
            "data_prepared": False, "real_data_training_started": False}


def main():
    report = inspect()
    if ws.run_dir(): ws.write_json(ws.require_run_dir()/"reports/local_cpu_environment.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
