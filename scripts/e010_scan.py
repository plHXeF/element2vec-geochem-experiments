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
    registry = root/"scan_registry.json"
    if args.command == "run" and registry.exists():
        raise FileExistsError("Scan exists; use resume")
    state = json.loads(registry.read_text()) if registry.exists() else {"configurations": cfg["configurations"], "runs": {}}
    if state["configurations"] != cfg["configurations"]:
        raise ValueError("Scan matrix changed")
    for config in cfg["configurations"]:
        record = state["runs"].get(config)
        if record and record["status"] == "completed": continue
        if record:
            run = ws.workspace_root()/record["path"]
        else:
            run = Path(invoke(["new-run", "E010", "--config", config], capture=True))
            record = {"path": str(run.relative_to(ws.workspace_root())), "status": "ready"}
            state["runs"][config] = record; ws.write_json(registry, state)
        manifest = json.loads((run/"manifest.json").read_text())
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
        invoke(["freeze", "--run-dir", str(run), "--status", "completed", "--outcome", "not_evaluated",
                "--summary", "E010 full-data training passed exposure/checkpoint verification; representation evaluation remains separate.",
                "--algorithms", "Element2Vec E010 two-family RoPE; AdamW; Student-t4 or registered ablation", "--device", "mps", "--allow-verified-resume"])
        record["status"] = "completed"; ws.write_json(registry, state)
    print("All registered configurations completed. No additional seeds were launched.")


if __name__ == "__main__":
    main()
