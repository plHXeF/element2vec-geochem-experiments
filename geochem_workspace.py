"""Local workspace paths and run provenance; no scientific transformations live here."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent


def workspace_root() -> Path:
    if os.environ.get("GEOCHEM_WORKSPACE"):
        root = Path(os.environ["GEOCHEM_WORKSPACE"]).expanduser().resolve()
        if not (root / "workspace.local.json").is_file():
            raise FileNotFoundError(f"Missing workspace.local.json in {root}")
        return root
    for root in [REPO_ROOT, *REPO_ROOT.parents]:
        if (root / "workspace.local.json").is_file():
            return root
    raise FileNotFoundError("Set GEOCHEM_WORKSPACE to a workspace containing workspace.local.json")


def local_settings() -> dict:
    return json.loads((workspace_root() / "workspace.local.json").read_text())


def asset(name: str) -> Path:
    key, _, suffix = name.removeprefix("asset://").partition("/")
    settings = local_settings()
    if key not in settings.get("assets", {}):
        raise KeyError(f"Unknown asset {key!r}; register it in workspace.local.json")
    base = Path(settings["assets"][key]).expanduser()
    base = base if base.is_absolute() else workspace_root() / base
    result = (base / suffix).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError("Asset suffix escapes the registered asset")
    if not result.exists():
        raise FileNotFoundError(f"Missing asset {key}: {result}")
    return result


def run_dir(required: bool = False) -> Path | None:
    value = os.environ.get("GEOCHEM_RUN_DIR")
    if not value:
        if required:
            raise ValueError("An explicit --run-dir is required. Create a run with scripts/experiment.py new-run first.")
        return None
    return Path(value).expanduser().resolve()


def bind_cli() -> None:
    """Consume the common flag before legacy argument parsers see it."""
    args = sys.argv
    found = []
    i = 1
    while i < len(args):
        if args[i] == "--run-dir":
            if i + 1 == len(args):
                raise SystemExit("--run-dir requires a path")
            found.append(args[i + 1]); del args[i:i + 2]
        elif args[i].startswith("--run-dir="):
            found.append(args.pop(i).split("=", 1)[1])
        else:
            i += 1
    if len(found) > 1:
        raise SystemExit("Specify --run-dir once")
    if found:
        os.environ["GEOCHEM_RUN_DIR"] = str(Path(found[0]).expanduser().resolve())


def require_run_dir() -> Path:
    bind_cli()
    if "--help" in sys.argv or "-h" in sys.argv:
        # Older scripts without an argument parser would otherwise ignore --help
        # and start work. Stop those here, before any imports or output creation.
        source = Path(sys.argv[0])
        if source.is_file() and "ArgumentParser(" not in source.read_text():
            print("Use scripts/experiment.py exec --run-dir RUN -- " + str(source) + "\nThis legacy entry point has no additional command-line options.")
            raise SystemExit(0)
        return run_dir() or REPO_ROOT / ".run-directory-required"
    p = run_dir(required=True)
    manifest_path = p / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Not a registered run: {p}. Use scripts/experiment.py new-run.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("frozen") or manifest.get("status") in {"completed", "failed", "cancelled", "historical"}:
        raise PermissionError(f"Frozen run cannot be written: {p}")
    if not p.is_relative_to(workspace_root() / "runs"):
        raise ValueError("Run directories must be inside this workspace's runs directory")
    return p


def output_path(subpath: str = "") -> Path:
    # Importing functions for tests never creates output directories.
    base = run_dir() or REPO_ROOT / ".run-directory-required"
    result = (base / subpath).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError("Output path escapes run directory")
    return result


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    value = str(value)
    if value.startswith("asset://"):
        return asset(value)
    if value.startswith("run://"):
        run_dir(required=True)
        return output_path(value[6:])
    if value.startswith("repo://"):
        return (REPO_ROOT / value[7:]).resolve()
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base or REPO_ROOT) / p


def resolve_config(value):
    if isinstance(value, dict):
        return {k: resolve_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_config(v) for v in value]
    if isinstance(value, str) and value.startswith(("asset://", "run://", "repo://")):
        if value.startswith("run://") and run_dir() is None:
            return value  # Model-only imports and unit tests do not allocate a run.
        return str(resolve_path(value))
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def scientific_config(config):
    """Remove machine locations; preserve ordered features and all scientific settings."""
    value = copy.deepcopy(config)
    # Logical asset IDs remain in the scientific signature. Runtime paths do not.
    value.pop("_runtime", None)
    locations = [(str(REPO_ROOT), "repo://")]
    if run_dir(): locations.append((str(run_dir()), "run://"))
    try:
        for name, location in local_settings().get("assets", {}).items():
            p = Path(location)
            locations.append((str((p if p.is_absolute() else workspace_root() / p).resolve()), f"asset://{name}"))
    except FileNotFoundError:
        pass
    locations.sort(key=lambda row: len(row[0]), reverse=True)
    def normalize(obj):
        if isinstance(obj, dict): return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [normalize(v) for v in obj]
        if isinstance(obj, str):
            for physical, logical in locations:
                if obj == physical: return logical + "." if logical.endswith("//") else logical
                if obj.startswith(physical + "/"): return logical.rstrip("/") + "/" + obj[len(physical) + 1:] if not logical.endswith("//") else logical + obj[len(physical) + 1:]
        return obj
    return normalize(value)


def configured_seed(default=42):
    p = run_dir()
    if p and (p / "manifest.json").is_file():
        value = json.loads((p / "manifest.json").read_text()).get("seed", default)
        if isinstance(value, int): return value
    return default


def load_config_file(path: str | Path) -> dict:
    """Use the run's frozen scientific configuration for its primary entry point."""
    import yaml
    source = resolve_path(path).resolve()
    active = run_dir()
    if active and (active / "config.logical.yaml").is_file():
        manifest = json.loads((active / "manifest.json").read_text())
        expected = manifest.get("source_config")
        defaults = manifest.get("default_config_entrypoint")
        primary_sources = {resolve_path(p).resolve() for p in (expected, defaults) if p}
        if source in primary_sources:
            source = active / "config.logical.yaml"
    text = source.read_text()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        config = yaml.safe_load(text) or {}
    return resolve_config(config)


def guard_args(args):
    """Reject output overrides that escape the selected run."""
    root = require_run_dir()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            text = str(value)
            for scheme in ("asset", "run", "repo"):
                if text.startswith(scheme + ":/"):
                    setattr(args, key, resolve_path(scheme + "://" + text.split(":", 1)[1].lstrip("/")))
                    break
    manifest = json.loads((root / "manifest.json").read_text())
    source = manifest.get("source_config")
    selected = getattr(args, "config", None)
    if source and selected:
        accepted = {resolve_path(source).resolve(), root / "config.logical.yaml", root / "config.resolved.yaml", root / "configs/benchmark.json"}
        if manifest.get("default_config_entrypoint"):
            accepted.add(resolve_path(manifest["default_config_entrypoint"]).resolve())
        if resolve_path(selected).resolve() not in accepted:
            raise ValueError("--config differs from the registered run configuration; select the recorded config or create a new run")
    for key in ("output", "outputs_dir", "output_dir", "report", "report_dir"):
        value = getattr(args, key, None)
        if value is not None:
            p = resolve_path(str(value).replace("asset:/", "asset://", 1) if str(value).startswith("asset:/") and not str(value).startswith("asset://") else value).resolve()
            if not p.is_relative_to(root):
                raise ValueError(f"--{key.replace('_', '-')} must be inside --run-dir")
            setattr(args, key, p)
    return args


bind_cli()
try:
    for directory in local_settings().get("python_paths", []):
        p = Path(directory)
        sys.path.insert(0, str(p if p.is_absolute() else workspace_root() / p))
except FileNotFoundError:
    pass  # A fresh clone can still import models and read reports without private data.
