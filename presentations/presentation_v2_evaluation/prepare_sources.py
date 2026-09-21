"""Prepare archived generators and a private view of their pinned inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geochem_workspace import REPO_ROOT, asset, local_settings, require_run_dir, workspace_root, write_json


def overlay(source, target):
    """Create file links while keeping parent directories writable and private."""
    target.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.name in {"__pycache__", ".git"}:
            continue
        dst = target / child.name
        if child.is_dir():
            overlay(child, dst)
        elif not dst.exists():
            dst.symlink_to(child.resolve())


def main():
    run = require_run_dir()
    runtime = local_settings().get("presentation_runtime", {})
    required = ["artifact_tool_module", "node_modules", "skill_directory", "python", "template"]
    missing = [k for k in required if not runtime.get(k)]
    if missing:
        raise SystemExit("Missing presentation_runtime in workspace.local.json: " + ", ".join(missing))
    paths = {}
    for key in required:
        path = Path(runtime[key]).expanduser()
        path = path if path.is_absolute() else workspace_root() / path
        if not path.exists():
            raise SystemExit(f"Missing presentation dependency {key}: {path}")
        paths[key] = str(path.resolve())
    root = run / "presentation_workspace"
    root.mkdir(exist_ok=False)
    overlay(asset("presentation_pyrite"), root / "presentation_html")
    overlay(asset("latent_reference"), root / "latent_evaluation_benchmark")
    overlay(REPO_ROOT / "stage3_6_element2vec_space_exploration", root / "stage3_6_element2vec_space_exploration")
    overlay(asset("space_audit"), root / "stage3_6_element2vec_space_exploration")
    overlay(asset("subspace_audit"), root / "stage3_6_element2vec_space_exploration/subspace_audit")
    # Old generators reuse filenames. Copy these working inputs so their writes
    # can never follow a symlink back into the frozen delivery.
    shutil.copytree(asset("presentation_v2") / "assets", root / "presentation_v2_evaluation/assets")
    shutil.copytree(asset("presentation_v2") / "output", root / "presentation_v2_evaluation/output")
    (root / "2025-7-4汇报1.pptx").symlink_to(paths["template"])
    values = {"WORKSPACE": str(root), "ROPE_SOURCE": str(REPO_ROOT), "ARTIFACT_TOOL": paths["artifact_tool_module"],
              "NODE_MODULES": paths["node_modules"], "PYTHON": paths["python"], "PRESENTATION_SKILL": paths["skill_directory"]}
    for source in (Path(__file__).parent / "templates").rglob("*.in"):
        relative = str(source.relative_to(Path(__file__).parent / "templates")).replace("generation/", "build/")
        target = root / "presentation_v2_evaluation" / relative[:-3]
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text()
        for key, value in values.items():
            text = text.replace("@@" + key + "@@", value)
        guard = f"import sys, os\nsys.path.insert(0, {str(REPO_ROOT)!r})\nos.environ['GEOCHEM_RUN_DIR'] = {str(run)!r}\nfrom geochem_workspace import require_run_dir\nrequire_run_dir()\n"
        if target.suffix == ".py":
            future = "from __future__ import annotations\n"
            text = text.replace(future, future + guard, 1) if future in text else guard + text
        else:
            text = "import {execFileSync as checkRun} from 'node:child_process';\ncheckRun(" + json.dumps(paths["python"]) + ", ['-c', " + json.dumps(guard) + "]);\n" + text
        target.write_text(text)
    write_json(root / "provenance.json", {"source_run": str(run.relative_to(workspace_root())), "inputs": values,
                                          "note": "Historical generator source, prepared only; no slides regenerated."})
    print(root)


if __name__ == "__main__":
    main()
