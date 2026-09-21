#!/usr/bin/env python3
"""Register, inspect, execute and freeze local research runs."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# This CLI owns --run-dir parsing; the shared module's legacy bootstrap must not consume it.
_arguments = sys.argv[:]
sys.argv = sys.argv[:1]
import geochem_workspace as ws
sys.argv = _arguments


def git(*args):
    return subprocess.check_output(["git", "-C", str(ws.REPO_ROOT), *args]).decode().strip()


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def run_lock(path):
    lock = path / ".execution.lock"
    try:
        handle = lock.open("x")
    except FileExistsError:
        raise ValueError(f"Run is already in use: {path}. If the process crashed, verify it has stopped before removing {lock.name}.")
    try:
        with handle:
            handle.write(str(os.getpid()) + "\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def cards():
    return [json.loads(p.read_text()) for p in sorted((ws.REPO_ROOT / "experiments").glob("E*/experiment.json"))]


def card_for(id):
    return next((c for c in cards() if c["id"] == id), None) or (_ for _ in ()).throw(ValueError(f"Unknown experiment {id}"))


def read_config(path):
    if not path:
        return {}
    import yaml
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text) or {}


def fingerprint(path):
    path = Path(path)
    if path.is_file():
        return {"sha256": ws.sha256(path), "bytes": path.stat().st_size}
    files = {}
    for p in sorted(path.rglob("*")):
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts:
            files[str(p.relative_to(path))] = {"sha256": ws.sha256(p), "bytes": p.stat().st_size}
    return {"sha256": ws.digest(files), "file_count": len(files), "files": files}


def config_assets(value):
    if isinstance(value, dict):
        return {k for v in value.values() for k in config_assets(v)}
    if isinstance(value, list):
        return {k for v in value for k in config_assets(v)}
    if isinstance(value, str) and value.startswith("asset://"):
        return {value.removeprefix("asset://")}
    return set()


def command_inputs(command, run):
    """Record explicit data/weight overrides in addition to config-declared assets."""
    inputs = {}
    for i, token in enumerate(command):
        if not token.startswith("--") or not any(k in token.split("=", 1)[0] for k in ("checkpoint", "sqlite", "legacy", "baseline", "input", "vocab")):
            continue
        flag, equals, value = token.partition("=")
        if not equals:
            if i + 1 == len(command) or command[i + 1].startswith("--"):
                continue
            value = command[i + 1]
        target = ws.resolve_path(value).resolve()
        if target.exists():
            info = fingerprint(target)
            info["path"] = str(target)
            info["workspace_relative"] = str(target.relative_to(ws.workspace_root())) if target.is_relative_to(ws.workspace_root()) else None
            info["within_run"] = target.is_relative_to(run)
            inputs[flag] = info
    return inputs


def new_run(args):
    card = card_for(args.experiment)
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    if dirty and not args.scratch:
        raise ValueError("Formal runs require a clean working tree. Commit source first, or use --scratch.")
    source = args.config or card["config"]
    source_path = ws.resolve_path(source) if source else None
    config = read_config(source_path)
    seed = args.seed
    if seed is None:
        seed = config.get("training", {}).get("seed", config.get("sampling", {}).get("seed", config.get("subspace_audit", {}).get("seed", config.get("seed", config.get("random_seed", config.get("random_state", 42))))))
    if "training" in config:
        config["training"]["seed"] = seed
    elif "sampling" in config:
        config["sampling"]["seed"] = seed
    elif "random_seed" in config:
        config["random_seed"] = seed
    elif "random_state" in config:
        config["random_state"] = seed
    elif "subspace_audit" in config:
        config["subspace_audit"]["seed"] = seed
    elif config:
        config["seed"] = seed
    # Resolve and hash required inputs before allocating a directory.
    input_keys = sorted(set(card["assets"]) | config_assets(config))
    inputs = {key: fingerprint(ws.asset(key)) for key in input_keys}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-s{seed}-" + secrets.token_hex(3)
    p = ws.workspace_root() / "runs" / card["id"] / run_id
    p.mkdir(parents=True, exist_ok=False)
    for folder in ["logs", "metrics", "checkpoints", "cache", "figures"]:
        (p / folder).mkdir()
    os.environ["GEOCHEM_RUN_DIR"] = str(p)
    source_ref = str(source_path.relative_to(ws.REPO_ROOT)) if source_path and source_path.is_relative_to(ws.REPO_ROOT) else str(source_path) if source_path else None
    manifest = {"schema_version": 1, "experiment_id": card["id"], "run_id": run_id,
                "status": "created", "frozen": False, "kind": "scratch" if args.scratch else "formal",
                "created_at": now(), "git_commit": git("rev-parse", "HEAD"), "git_dirty": dirty,
                "source_config": source_ref, "default_config_entrypoint": card.get("config"), "config_hash": ws.digest(ws.scientific_config(config)),
                "seed": seed, "device": "not executed", "algorithms": "not executed",
                "inputs": inputs, "upstream_experiments": card["upstream_experiments"],
                "upstream_runs": sorted({str(ws.asset(k).relative_to(ws.workspace_root())).split("/")[1] + "/" + str(ws.asset(k).relative_to(ws.workspace_root())).split("/")[2] for k in input_keys if ws.asset(k).is_relative_to(ws.workspace_root() / "runs")}),
                "protocol": card["protocol"], "commands": []}
    ws.write_json(p / "manifest.json", manifest)
    ws.write_json(p / "config.logical.yaml", config)  # JSON is valid YAML, with stable types.
    ws.write_json(p / "config.resolved.yaml", ws.resolve_config(config))
    manifest["resolved_config_sha256"] = ws.sha256(p / "config.resolved.yaml")
    ws.write_json(p / "manifest.json", manifest)
    if card["id"] == "E009":
        ws.write_json(p / "configs/benchmark.json", ws.resolve_config(config))
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "umap-learn", "plotly", "pyarrow", "matplotlib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    (p / "environment.txt").write_text(json.dumps({"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages}, indent=2) + "\n")
    (p / "summary.md").write_text(f"# {card['id']} — {card['title']}\n\nStatus: created\n\nProtocol: {card['protocol']}\n\nResult and limitations: pending.\n")
    if args.scratch:
        (p / "source").mkdir()
        (p / "source/changes.patch").write_bytes(subprocess.check_output(["git", "-C", str(ws.REPO_ROOT), "diff", "--binary", "HEAD"]))
        files = subprocess.check_output(["git", "-C", str(ws.REPO_ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]).decode().split("\0")
        for name in files:
            src = ws.REPO_ROOT / name
            if name and src.is_file():
                dst = p / "source/tree" / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    print(p)


def inspect_run(path, rehash=False):
    p = Path(path).resolve()
    manifest = json.loads((p / "manifest.json").read_text())
    errors = []
    for name in ["config.resolved.yaml", "environment.txt", "summary.md"]:
        if not (p / name).is_file(): errors.append(f"Missing {name}")
    if manifest.get("status") != "historical":
        config = read_config(p / "config.logical.yaml")
        if ws.digest(ws.scientific_config(config)) != manifest["config_hash"]:
            errors.append("Scientific config changed; create a new run")
        if ws.sha256(p / "config.resolved.yaml") != manifest.get("resolved_config_sha256"):
            errors.append("Resolved config snapshot changed; create a new run")
        for key, original in manifest.get("inputs", {}).items():
            try:
                target = ws.asset(key)
                if rehash and fingerprint(target)["sha256"] != original["sha256"]:
                    errors.append(f"Input changed: {key}")
            except (FileNotFoundError, KeyError) as e:
                errors.append(str(e))
        if rehash:
            for step in manifest.get("commands", []):
                for flag, original in step.get("inputs", {}).items():
                    if original.get("within_run"):
                        continue  # A command records the exact same-run intermediate it read.
                    target = ws.workspace_root() / original["workspace_relative"] if original.get("workspace_relative") else Path(original["path"])
                    if not target.exists() or fingerprint(target)["sha256"] != original["sha256"]:
                        errors.append(f"Command input changed: {flag} {target}")
    if rehash and manifest.get("frozen") and (p / "artifact_checksums.json").exists():
        for name, expected in json.loads((p / "artifact_checksums.json").read_text()).items():
            if not (p / name).is_file() or ws.sha256(p / name) != expected:
                errors.append(f"Frozen artifact changed: {name}")
    return {"run_id": manifest["run_id"], "status": manifest["status"], "frozen": manifest.get("frozen", False), "passed": not errors, "errors": errors}


def execute(args):
    os.environ["GEOCHEM_RUN_DIR"] = str(Path(args.run_dir).resolve())
    p = ws.require_run_dir()
    result = inspect_run(p, rehash=True)
    if not result["passed"]: raise ValueError(result["errors"])
    m = json.loads((p / "manifest.json").read_text())
    if m["kind"] == "formal" and (git("rev-parse", "HEAD") != m["git_commit"] or git("status", "--porcelain")):
        raise ValueError("Formal run must execute from its recorded clean commit")
    command = list(args.command)
    if command and command[0] == "--": command.pop(0)
    if not command: raise ValueError("Provide a Python script or -m module after --")
    if any(v == "--run-dir" or v.startswith("--run-dir=") for v in command):
        raise ValueError("exec owns --run-dir; do not pass a different run context to the child command")
    m["status"] = "running"
    entry = {"command": [sys.executable, *command], "started_at": now(), "inputs": command_inputs(command, p)}
    m["commands"].append(entry); ws.write_json(p / "manifest.json", m)
    env = os.environ.copy(); env["PYTHONPATH"] = str(ws.REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLCONFIGDIR"] = str(p / "cache/matplotlib")
    index = len(m["commands"])
    with (p / "logs" / f"step-{index:03d}.log").open("w") as log:
        child = subprocess.Popen([sys.executable, *command], cwd=ws.REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in child.stdout:
                print(line, end="", flush=True); log.write(line); log.flush()
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate(); code = child.wait()
    entry.update(finished_at=now(), exit_code=code)
    m["status"] = "ready" if code == 0 else "interrupted"
    ws.write_json(p / "manifest.json", m)
    raise SystemExit(code)


def freeze(args):
    os.environ["GEOCHEM_RUN_DIR"] = str(Path(args.run_dir).resolve())
    p = ws.require_run_dir(); result = inspect_run(p, rehash=True)
    if not result["passed"]: raise ValueError(result["errors"])
    m = json.loads((p / "manifest.json").read_text())
    if args.status == "completed" and not m["commands"]:
        raise ValueError("No executed step has been recorded")
    if args.status == "completed" and any(c.get("exit_code") != 0 for c in m["commands"]):
        raise ValueError("A step failed; explain and freeze as failed or create a new run")
    if args.status != "completed" and args.outcome != "not_evaluated":
        raise ValueError("Failed/cancelled execution must use outcome not_evaluated; record partial scientific observations in the summary")
    (p / "summary.md").write_text(f"# {m['experiment_id']} / {m['run_id']}\n\n{args.summary}\n\nOutcome: {args.outcome}\n\nAlgorithms: {args.algorithms}\n\nDevice: {args.device}\n")
    excluded = {p / "manifest.json", p / "artifact_checksums.json", p / ".execution.lock"}
    hashes = {str(f.relative_to(p)): ws.sha256(f) for f in sorted(p.rglob("*")) if f.is_file() and not f.is_symlink() and f not in excluded and "__pycache__" not in f.parts}
    ws.write_json(p / "artifact_checksums.json", hashes)
    m.update(status=args.status, frozen=True, finished_at=now(), scientific_outcome=args.outcome, algorithms=args.algorithms, device=args.device)
    ws.write_json(p / "manifest.json", m)
    print(f"Frozen {p}")


def index():
    rows = cards(); out = ws.REPO_ROOT / "experiments"
    with (out / "registry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "module", "execution_status", "scientific_outcome", "report"], lineterminator="\n")
        writer.writeheader(); writer.writerows({k: c[k] for k in writer.fieldnames} for c in rows)
    lines = ["# 实验总表", "", "由实验卡生成。运行路径相对工作区；卡片是实验定义和结论的唯一来源。", "", "| 编号 | 实验 | 状态 / 结论 | 报告 |", "|---|---|---|---|"]
    for c in rows:
        lines.append(f"| [{c['id']}]({c['id']}/experiment.json) | {c['title']} | {c['execution_status']} / {c['scientific_outcome']} | [报告](../{c['report']}) |")
    for c in rows:
        lines += ["", f"## {c['id']} · {c['title']}", "", f"[实验卡]({c['id']}/experiment.json) · [代码](../{c['module']})"]
        if c.get("config"):
            lines[-1] += f" · [配置](../{c['config']})"
        else:
            lines[-1] += " · 配置由代码提交和运行种子固定"
        lines += ["", c.get("conclusion", ""), ""]
        for run in c.get("runs", []):
            lines.append(f"- [历史运行 {run['id']}](../../{run['path']}/summary.md) · [来源与状态](../../{run['path']}/manifest.json) · [逐文件校验](../../{run['path']}/artifact_checksums.json)")
        weights = c.get("artifacts", {}).get("weights", [])
        main_weights = [p for p in weights if p.endswith(("/full_model.pt", "/token_transformer_last.pt"))]
        if main_weights:
            lines.append("- 权重：" + " · ".join(f"[{Path(p).name}](../../{p})" for p in main_weights))
        else:
            lines.append("- 权重：本实验未训练独立模型；依赖见实验卡。")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(out / "README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list"); sub.add_parser("index")
    p = sub.add_parser("new-run"); p.add_argument("experiment"); p.add_argument("--seed", type=int); p.add_argument("--config"); p.add_argument("--scratch", action="store_true")
    p = sub.add_parser("check"); p.add_argument("--run-dir", required=True); p.add_argument("--rehash", action="store_true")
    p = sub.add_parser("exec"); p.add_argument("--run-dir", required=True); p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("freeze"); p.add_argument("--run-dir", required=True); p.add_argument("--status", choices=["completed", "failed", "cancelled"], required=True); p.add_argument("--outcome", choices=["positive", "negative", "mixed", "inconclusive", "not_evaluated"], required=True); p.add_argument("--summary", required=True); p.add_argument("--algorithms", required=True); p.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.action == "list":
        for c in cards():
            print(f"{c['id']}  {c['title']}  [{c['execution_status']} / {c['scientific_outcome']}]")
            for r in sorted((ws.workspace_root() / "runs" / c["id"]).glob("*/manifest.json")):
                m = json.loads(r.read_text()); print(f"  {m['run_id']}  {m['status']}  {r.parent}")
    elif args.action == "index": index()
    elif args.action == "new-run": new_run(args)
    elif args.action in {"exec", "freeze"}:
        p = Path(args.run_dir).resolve()
        os.environ["GEOCHEM_RUN_DIR"] = str(p)
        ws.require_run_dir()
        with run_lock(p):
            if args.action == "exec": execute(args)
            else: freeze(args)
    else:
        result = inspect_run(args.run_dir, args.rehash); print(json.dumps(result, ensure_ascii=False, indent=2)); raise SystemExit(not result["passed"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, PermissionError) as e:
        raise SystemExit(str(e))
