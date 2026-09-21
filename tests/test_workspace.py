"""Integration checks for provenance, isolation, relocation and frozen output protection."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "workspace"
        self.repo = self.root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "experiments/E001").mkdir(parents=True)
        shutil.copy2(SOURCE / "geochem_workspace.py", self.repo)
        shutil.copy2(SOURCE / "scripts/experiment.py", self.repo / "scripts")
        (self.root / "data").mkdir()
        (self.root / "data/input.txt").write_text("fixed input\n")
        (self.root / "workspace.local.json").write_text(json.dumps({"assets": {"toy": "data/input.txt"}}))
        (self.repo / "config.json").write_text(json.dumps({"seed": 7, "input": "asset://toy", "epsilon": 1e-9}))
        card = {"id": "E001", "title": "fixture", "config": "config.json", "assets": ["toy"], "upstream_experiments": [], "protocol": "fixed toy", "module": "fixture", "execution_status": "planned", "scientific_outcome": "not_evaluated", "report": "README.md", "runs": []}
        (self.repo / "experiments/E001/experiment.json").write_text(json.dumps(card))
        (self.repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
        for cmd in [["init", "-q"], ["config", "user.name", "Workspace Test"], ["config", "user.email", "test@example.invalid"], ["add", "."], ["commit", "-qm", "fixture"]]:
            subprocess.run(["git", *cmd], cwd=self.repo, check=True, capture_output=True)

    def cli(self, *args, success=True):
        env = os.environ.copy()
        env.pop("GEOCHEM_RUN_DIR", None)
        env["GEOCHEM_WORKSPACE"] = str(self.root)
        p = subprocess.run([sys.executable, "scripts/experiment.py", *map(str, args)], cwd=self.repo, env=env, capture_output=True, text=True)
        if success: self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        else: self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def create(self):
        return Path(self.cli("new-run", "E001").stdout.strip())

    def test_two_parallel_runs_do_not_overlap(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(lambda _: self.create(), range(2)))
        self.assertNotEqual(*paths)
        for p in paths:
            self.cli("exec", "--run-dir", p, "--", "-c", "from geochem_workspace import require_run_dir; (require_run_dir()/'metrics/result.txt').write_text('ok')")
            self.assertEqual((p / "metrics/result.txt").read_text(), "ok")

    def test_scientific_notation_keeps_numeric_type(self):
        p = self.create()
        self.cli("exec", "--run-dir", p, "--", "-c", "from geochem_workspace import load_config_file; c=load_config_file('config.json'); assert isinstance(c['epsilon'],float) and c['epsilon']==1e-9")

    def test_changed_inputs_and_config_are_rejected(self):
        p = self.create()
        (self.root / "data/input.txt").write_text("changed")
        self.cli("check", "--run-dir", p, "--rehash", success=False)
        (p / "config.logical.yaml").write_text('{"seed": 99}')
        self.cli("check", "--run-dir", p, success=False)

    def test_frozen_run_rejects_writes_and_detects_tampering(self):
        p = self.create()
        self.cli("exec", "--run-dir", p, "--", "-c", "from geochem_workspace import require_run_dir; (require_run_dir()/'metrics/result.txt').write_text('ok')")
        self.cli("freeze", "--run-dir", p, "--status", "completed", "--outcome", "inconclusive", "--summary", "toy only", "--algorithms", "identity", "--device", "cpu")
        self.cli("exec", "--run-dir", p, "--", "-c", "print('should not run')", success=False)
        (p / "metrics/result.txt").write_text("tampered")
        self.cli("check", "--run-dir", p, "--rehash", success=False)

    def test_relocating_workspace_preserves_scientific_signature(self):
        p = self.create(); relative = p.relative_to(self.root)
        signature = json.loads((p / "manifest.json").read_text())["config_hash"]
        new = self.root.with_name("moved")
        self.root.rename(new); self.root = new; self.repo = new / "repo"
        self.cli("check", "--run-dir", new / relative, "--rehash")
        next_run = self.create()
        self.assertEqual(signature, json.loads((next_run / "manifest.json").read_text())["config_hash"])

    def test_formal_dirty_tree_rejected_scratch_source_preserved(self):
        (self.repo / "untracked.py").write_text("important = 1\n")
        self.cli("new-run", "E001", success=False)
        p = Path(self.cli("new-run", "E001", "--scratch").stdout.strip())
        self.assertEqual((p / "source/tree/untracked.py").read_text(), "important = 1\n")

    def test_missing_asset_and_output_escape_are_explicit(self):
        p = self.create()
        self.cli("exec", "--run-dir", p, "--", "-c", "from argparse import Namespace; from geochem_workspace import guard_args; guard_args(Namespace(output='../outside'))", success=False)
        (self.root / "data/input.txt").unlink()
        message = self.cli("new-run", "E001", success=False)
        self.assertIn("Missing asset toy", message.stderr)

    def test_command_checkpoint_override_is_fingerprinted(self):
        checkpoint = self.root / "data/alternate.pt"
        checkpoint.write_text("a different model")
        p = self.create()
        self.cli("exec", "--run-dir", p, "--", "-c", "print('loaded toy')", "--checkpoint", checkpoint)
        m = json.loads((p / "manifest.json").read_text())
        self.assertIn("--checkpoint", m["commands"][0]["inputs"])
        checkpoint.write_text("changed model")
        self.cli("check", "--run-dir", p, "--rehash", success=False)

    def test_failed_execution_is_not_a_scientific_negative(self):
        p = self.create()
        self.cli("freeze", "--run-dir", p, "--status", "failed", "--outcome", "negative", "--summary", "technical failure", "--algorithms", "none", "--device", "cpu", success=False)

    def test_same_run_cannot_execute_concurrently(self):
        p = self.create()
        (p / ".execution.lock").write_text("1234\n")
        result = self.cli("exec", "--run-dir", p, "--", "-c", "print('unsafe')", success=False)
        self.assertIn("already in use", result.stderr)

    def test_config_override_cannot_silently_replace_the_registered_config(self):
        p = self.create()
        self.cli("exec", "--run-dir", p, "--", "-c", "from argparse import Namespace; from geochem_workspace import guard_args; guard_args(Namespace(config='other.json'))", success=False)


if __name__ == "__main__":
    unittest.main()
