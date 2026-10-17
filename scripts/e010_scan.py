"""Explicit user-operated E010 scan. 'plan' prints commands and never trains."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geochem_workspace as ws

PREFIX = "stage3_5_element2vec_v2/configs/e010"
MODULE = "stage3_5_element2vec_v2.src.e010.cli"


def invoke(arguments, capture=False):
    env = os.environ.copy()
    env.pop("GEOCHEM_RUN_DIR", None)
    result = subprocess.run([sys.executable, "scripts/experiment.py", *arguments], cwd=ws.REPO_ROOT,
                            env=env, text=True, stdout=subprocess.PIPE if capture else None, check=True)
    return result.stdout.strip() if capture else None


def scan(root, cfg, resume=False):
    """Each GPU owns one parent registry; children never share a writable run."""
    registry = root/"scan_registry.json"
    if not resume and registry.exists():
        raise FileExistsError("Scan exists; use resume")
    if resume and not registry.exists():
        raise FileNotFoundError("No registered scan to resume")
    configurations = cfg["configurations"]
    if not configurations or len(configurations) != len(set(configurations)):
        raise ValueError("Scan must contain unique configurations")
    state = json.loads(registry.read_text()) if resume else {"configurations": configurations, "runs": {}}
    if state["configurations"] != configurations:
        raise ValueError("Scan matrix changed")
    for config in configurations:
        device = json.loads(ws.resolve_path(config).read_text())["training"]["device"]
        record = state["runs"].get(config)
        if record and record["status"] == "completed": continue
        if record:
            run = ws.workspace_root()/record["path"]
        else:
            run = Path(invoke(["new-run", "E010", "--config", config], capture=True))
            record = {"path": str(run.relative_to(ws.workspace_root())), "status": "ready"}
            state["runs"][config] = record; ws.write_json(registry, state)
        manifest = json.loads((run/"manifest.json").read_text())
        parent_input = json.loads((root/"manifest.json").read_text())["inputs"]["e010_prepared"]
        if manifest["inputs"]["e010_prepared"]["sha256"] != parent_input["sha256"]:
            raise ValueError("Prepared input differs from the parent scan; preserve both runs and restore the pinned asset")
        if manifest.get("frozen"):
            if manifest.get("status") == "completed" and (run/"training_verification.json").exists():
                record["status"] = "completed"; ws.write_json(registry, state); continue
            raise RuntimeError(f"Cannot resume frozen incomplete run: {run}")
        # Only training has a supported durable resume. Other failed steps keep
        # their evidence and require a fresh run, not a rewritten command log.
        for command in manifest.get("commands", []):
            if command.get("exit_code") != 0 and "train" not in command.get("command", []):
                raise RuntimeError(f"A non-training step failed; preserve it and create a new run: {run}")
        if not (run/"preflight.json").exists():
            invoke(["exec", "--run-dir", str(run), "--", "-m", MODULE, "preflight", "--config", config])
        if not (run/"training_complete.json").exists():
            command = ["exec", "--run-dir", str(run), "--", "-m", MODULE, "train", "--config", config]
            if (run/"checkpoints/latest.pt").exists(): command.append("--resume")
            invoke(command)
        invoke(["exec", "--run-dir", str(run), "--", "-m", MODULE, "verify-training", "--config", config])
        invoke(["check", "--run-dir", str(run), "--rehash"])
        invoke(["freeze", "--run-dir", str(run), "--status", "completed", "--outcome", "not_evaluated",
                "--summary", "E010 full-data training passed exposure/checkpoint verification; representation evaluation remains separate.",
                "--algorithms", "Element2Vec E010 two-family RoPE; AdamW; Student-t4 or registered ablation", "--device", device, "--allow-verified-resume"])
        record["status"] = "completed"; ws.write_json(registry, state)
    print("All registered configurations completed. No additional seeds were launched.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "run", "resume"])
    p.add_argument("--config", default=f"{PREFIX}/scan.json")
    args = p.parse_args()
    if args.command == "plan":
        cfg = json.loads(ws.resolve_path(args.config).read_text())
        for path in cfg["configurations"]:
            print(f"new-run E010 --config {path}")
            print(f"  exec --run-dir RUN -- -m {MODULE} preflight --config {path}")
            print(f"  exec --run-dir RUN -- -m {MODULE} train --config {path}")
        print("No run created; no training started.")
        return
    root = ws.require_run_dir(); ws.guard_args(args)
    cfg = ws.load_config_file(args.config)
    scan(root, cfg, resume=args.command == "resume")


if __name__ == "__main__":
    main()
