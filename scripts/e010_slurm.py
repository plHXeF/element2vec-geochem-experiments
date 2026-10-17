"""Launch exactly the last 12 E010 configurations as four independent GPU queues.

The coordinator is run through experiment.py exec. Slurm workers only read its
registry; each worker writes to a separate shard run and separate training runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geochem_workspace as ws
from scripts.e010_scan import invoke, PREFIX

CONFIG = f"{PREFIX}/hpc/scan_last12.json"
SHARDS = [f"{PREFIX}/hpc/scan_gpu{i}.json" for i in range(4)]


def validate_matrix():
    """Reject omissions, repetitions or scientific changes during porting."""
    original = json.loads(ws.resolve_path(f"{PREFIX}/scan.json").read_text())["configurations"][4:]
    expected = [f"{PREFIX}/hpc/{Path(p).name}" for p in original]
    matrix = json.loads(ws.resolve_path(CONFIG).read_text())
    if matrix["configurations"] != expected:
        raise ValueError("HPC matrix must be the original ordered configurations 5 through 16")
    for source, target in zip(original, expected):
        base = json.loads(ws.resolve_path(source).read_text())
        cuda = json.loads(ws.resolve_path(target).read_text())
        base["training"]["device"] = "cuda"
        if cuda != base:
            raise ValueError(f"HPC configuration changes more than the device: {target}")
    shards = [json.loads(ws.resolve_path(p).read_text()) for p in SHARDS]
    flat = [p for shard in shards for p in shard["configurations"]]
    if sorted(flat) != sorted(expected) or any(len(s["configurations"]) != 3 for s in shards):
        raise ValueError("Four disjoint three-configuration queues are required")
    return matrix, shards


def launch(root, resume=False):
    matrix, _ = validate_matrix()
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_NTASKS", 0)) != 4:
        raise RuntimeError("Submit the four-task, four-GPU Slurm script; do not launch on a login node")
    registry = root/"slurm_registry.json"
    if registry.exists() and not resume:
        raise FileExistsError("Launch exists; use resume with this coordinator run")
    if resume and not registry.exists():
        raise FileNotFoundError("No launch registry to resume")
    state = json.loads(registry.read_text()) if resume else {"configurations": matrix["configurations"], "slots": {}}
    if state["configurations"] != matrix["configurations"]:
        raise ValueError("Registered HPC matrix changed")
    # Persist before allocation of children, so a partial setup remains resumable.
    ws.write_json(registry, state)
    parent_input = json.loads((root/"manifest.json").read_text())["inputs"]["e010_prepared"]
    for slot, config in enumerate(SHARDS):
        key = str(slot)
        if key not in state["slots"]:
            run = Path(invoke(["new-run", "E010", "--config", config], capture=True))
            state["slots"][key] = {"path": run.relative_to(ws.workspace_root()).as_posix(), "config": config}
            ws.write_json(registry, state)
        record = state["slots"][key]
        if record["config"] != config:
            raise ValueError("Registered GPU queue changed")
        run = ws.workspace_root()/record["path"]
        manifest = json.loads((run/"manifest.json").read_text())
        if manifest["inputs"]["e010_prepared"]["sha256"] != parent_input["sha256"]:
            raise ValueError("Prepared input differs from the coordinator's pinned asset")
    subprocess.run([
        "srun", "--ntasks=4", "--mpi=none", "--gpus-per-task=1", "--gpu-bind=single:1", "--kill-on-bad-exit=1",
        "--output", str(root/"logs/slurm-%J-slot-%t.log"),
        sys.executable, "-u", "scripts/e010_slurm.py", "worker", "--launch-run", str(root),
    ], cwd=ws.REPO_ROOT, check=True)
    # Only the coordinator writes this receipt; workers never mutate its run.
    devices = [json.loads((ws.workspace_root()/state["slots"][str(slot)]["path"]/"slurm_device.json").read_text())
               for slot in range(4)]
    ws.write_json(root/"slurm_complete.json", {"passed": True, "training_runs": 12,
                  "slurm_job_id": os.environ["SLURM_JOB_ID"], "devices": devices,
                  "parallelism": "one model per GPU"})


def worker(root):
    rank = int(os.environ["SLURM_PROCID"])
    if rank not in range(4): raise ValueError("Unexpected Slurm rank")
    import torch
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("E010 HPC environment requires Python 3.11")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each Slurm task must see exactly one assigned CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    if "RTX A6000" not in gpu.upper():
        raise RuntimeError(f"Expected NVIDIA RTX A6000, got {gpu}")
    state = json.loads((root/"slurm_registry.json").read_text())
    record = state["slots"][str(rank)]
    run = ws.workspace_root()/record["path"]
    mode = "resume" if (run/"scan_registry.json").exists() else "run"
    device = {"slot": rank, "gpu": gpu,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    ws.write_json(run/"slurm_device.json", device)
    print(json.dumps({"slot": rank, "gpu": gpu,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "run": str(run), "mode": mode}), flush=True)
    # invoke clears the coordinator's run environment before binding this shard.
    invoke(["exec", "--run-dir", str(run), "--", "scripts/e010_scan.py", mode, "--config", record["config"]])
    invoke(["check", "--run-dir", str(run), "--rehash"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "resume", "worker"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--launch-run", type=Path)
    args = parser.parse_args()
    matrix, shards = validate_matrix()
    if args.command == "plan":
        for slot, shard in enumerate(shards):
            print(f"GPU queue {slot}: " + ", ".join(Path(p).stem for p in shard["configurations"]))
        print(f"{len(matrix['configurations'])} training runs; no runs created and no training started.")
    elif args.command == "worker":
        if args.launch_run is None: parser.error("worker requires --launch-run")
        worker(args.launch_run)
    else:
        root = ws.require_run_dir(); ws.guard_args(args)
        cfg = ws.load_config_file(args.config)
        if cfg != ws.resolve_config(matrix): raise ValueError("Coordinator must use the registered last-12 matrix")
        launch(root, resume=args.command == "resume")


if __name__ == "__main__":
    main()
