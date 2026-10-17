"""Data-free checks of the HPC matrix, GPU isolation and durable run routing."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import e010_scan as scan
from scripts import e010_slurm as hpc


class MatrixTests(unittest.TestCase):
    def test_exact_tail_and_unchanged_scientific_parameters(self):
        matrix, shards = hpc.validate_matrix()
        self.assertEqual(len(matrix["configurations"]), 12)
        self.assertEqual(len(shards), 4)
        self.assertEqual(Path(matrix["configurations"][0]).stem, "fixed_0128")
        self.assertEqual(Path(matrix["configurations"][-1]).stem, "native_1024")

    def test_plan_does_not_allocate_runs_or_invoke_slurm(self):
        with patch.object(sys, "argv", ["e010_slurm.py", "plan"]), patch.object(hpc, "invoke") as invoke, \
                patch.object(hpc.subprocess, "run") as srun:
            hpc.main()
        invoke.assert_not_called(); srun.assert_not_called()

    def test_device_port_rejects_scientific_change(self):
        real = Path.read_text
        def changed(path, *args, **kwargs):
            value = real(path, *args, **kwargs)
            if path.as_posix().endswith("hpc/fixed_0128.json"):
                cfg = json.loads(value); cfg["training"]["effective_batch"] = 4096
                return json.dumps(cfg)
            return value
        with patch.object(Path, "read_text", changed), self.assertRaises(ValueError): hpc.validate_matrix()


class BatchBootstrapTests(unittest.TestCase):
    def bash(self):
        bash = shutil.which("bash")
        if not bash:
            git = shutil.which("git")
            candidate = Path(git).parent.parent/"bin/bash.exe" if git else None
            if candidate and candidate.is_file(): bash = str(candidate)
        if not bash: self.skipTest("Bash is not installed")
        return bash

    def test_batch_script_refuses_execution_outside_allocation(self):
        bash = self.bash()
        env = os.environ.copy(); env.pop("SLURM_JOB_ID", None)
        script = scan.ws.REPO_ROOT/"scripts/slurm/e010_a800.sh"
        result = subprocess.run([bash, str(script), "new"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
        self.assertIn("Run only inside a Slurm allocation", result.stderr)

    def test_new_run_registration_happens_inside_batch_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/"bin").mkdir()
            (root/"conda/etc/profile.d").mkdir(parents=True)
            (root/"conda/etc/profile.d/conda.sh").write_text("")
            scripts = {
                "conda": '#!/bin/bash\nif [[ "$1" == info ]]; then echo "$TEST_CONDA_BASE"; fi\n',
                "nvidia-smi": "#!/bin/bash\nexit 0\n",
                "python": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_LOG"\nif [[ "$2" == new-run ]]; then echo "$TEST_EXPECTED_RUN"; fi\n',
            }
            for name, text in scripts.items():
                path = root/"bin"/name; path.write_bytes(text.encode()); path.chmod(0o755)
            env = os.environ.copy()
            env.update(TEST_TEMP=str(root), TEST_SCRIPT=str(scan.ws.REPO_ROOT/"scripts/slurm/e010_a800.sh"),
                       SLURM_JOB_ID="test-allocation", SLURM_NTASKS="4", SLURM_SUBMIT_DIR=str(scan.ws.REPO_ROOT),
                       GEOCHEM_WORKSPACE=str(root), CONDA_PREFIX=str(root/"conda"))
            bootstrap = '''
task_root="$TEST_TEMP"
if command -v cygpath >/dev/null; then task_root=$(cygpath -u "$task_root"); fi
export PATH="$task_root/bin:$PATH"
export TEST_CONDA_BASE="$task_root/conda" TEST_LOG="$task_root/calls.txt" TEST_EXPECTED_RUN="$task_root/registered-run"
module() { :; }
export -f module
bash "$TEST_SCRIPT" new
'''
            result = subprocess.run([self.bash(), "-c", bootstrap], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            calls = (root/"calls.txt").read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertIn("new-run E010 --config", calls[0])
            self.assertIn("scripts/e010_slurm.py run", calls[1])
            self.assertIn("E010 coordinator run:", result.stdout)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.workspace = Path(tmp.name)
        self.parent = self.workspace/"runs/E010/coordinator"; self.parent.mkdir(parents=True)
        (self.parent/"logs").mkdir()
        self.inputs = {"e010_prepared": {"sha256": "immutable-prepared-input"}}
        self.write_manifest(self.parent)
        self.calls = []; self.children = []

    def write_manifest(self, path, **changes):
        value = {"inputs": self.inputs, "frozen": False, "commands": []}; value.update(changes)
        (path/"manifest.json").write_text(json.dumps(value))

    def invoke(self, args, capture=False):
        self.calls.append(args)
        if args[0] == "new-run":
            path = self.workspace/f"runs/E010/child-{len(self.children)}"; path.mkdir()
            self.write_manifest(path); self.children.append(path)
            return str(path)

    def context(self):
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(scan.ws, "workspace_root", return_value=self.workspace))
        stack.enter_context(patch.object(scan, "invoke", side_effect=self.invoke))
        return stack

    def test_cuda_freeze_and_check_order(self):
        self.context()
        matrix, _ = hpc.validate_matrix()
        cfg = {"configurations": matrix["configurations"][:1]}
        scan.scan(self.parent, cfg)
        commands = [c[0] for c in self.calls]
        self.assertEqual(commands, ["new-run", "exec", "exec", "exec", "check", "freeze"])
        freeze = self.calls[-1]
        self.assertEqual(freeze[freeze.index("--device")+1], "cuda")
        self.assertEqual(freeze[freeze.index("--outcome")+1], "not_evaluated")

    def test_failure_stops_queue_without_allocating_next_model(self):
        self.context()
        matrix, _ = hpc.validate_matrix()
        def fail_train(args, capture=False):
            result = self.invoke(args, capture)
            if "train" in args: raise subprocess.CalledProcessError(1, args)
            return result
        with patch.object(scan, "invoke", side_effect=fail_train), self.assertRaises(subprocess.CalledProcessError):
            scan.scan(self.parent, matrix)
        self.assertEqual(len(self.children), 1)
        self.assertNotIn("freeze", [c[0] for c in self.calls])
        state = json.loads((self.parent/"scan_registry.json").read_text())
        self.assertEqual(len(state["runs"]), 1)

    def test_resume_keeps_completed_runs_and_uses_durable_checkpoint(self):
        self.context()
        paths = hpc.validate_matrix()[0]["configurations"][:2]
        first = Path(self.invoke(["new-run"]))
        second = Path(self.invoke(["new-run"]))
        (second/"checkpoints").mkdir(); (second/"checkpoints/latest.pt").write_text("synthetic marker")
        (second/"preflight.json").write_text("{}")
        failure = {"command": ["python", "-m", scan.MODULE, "train"], "exit_code": 1}
        self.write_manifest(second, commands=[failure])
        state = {"configurations": paths, "runs": {
            paths[0]: {"path": first.relative_to(self.workspace).as_posix(), "status": "completed"},
            paths[1]: {"path": second.relative_to(self.workspace).as_posix(), "status": "ready"}}}
        (self.parent/"scan_registry.json").write_text(json.dumps(state)); self.calls.clear()
        scan.scan(self.parent, {"configurations": paths}, resume=True)
        self.assertFalse(any(c[0] == "new-run" for c in self.calls))
        training = next(c for c in self.calls if "train" in c)
        self.assertIn("--resume", training)
        self.assertEqual(json.loads((second/"manifest.json").read_text())["commands"], [failure])

    def test_parent_and_child_input_versions_must_match(self):
        self.context()
        def drift(args, capture=False):
            result = self.invoke(args, capture)
            if args[0] == "new-run": self.write_manifest(Path(result), inputs={"e010_prepared": {"sha256": "different"}})
            return result
        with patch.object(scan, "invoke", side_effect=drift), self.assertRaises(ValueError):
            scan.scan(self.parent, {"configurations": hpc.validate_matrix()[0]["configurations"][:1]})
        self.assertFalse(any(c[0] == "exec" for c in self.calls))

    def test_nontraining_failure_refuses_resume(self):
        self.context()
        config = hpc.validate_matrix()[0]["configurations"][0]
        child = Path(self.invoke(["new-run"]))
        self.write_manifest(child, commands=[{"command": ["preflight"], "exit_code": 1}])
        state = {"configurations": [config], "runs": {config: {"path": child.relative_to(self.workspace).as_posix(), "status": "ready"}}}
        (self.parent/"scan_registry.json").write_text(json.dumps(state)); self.calls.clear()
        with self.assertRaises(RuntimeError): scan.scan(self.parent, {"configurations": [config]}, resume=True)
        self.assertFalse(self.calls)

    def test_missing_registry_does_not_turn_resume_into_new_training(self):
        self.context()
        with self.assertRaises(FileNotFoundError): scan.scan(self.parent, {"configurations": ["unused"]}, resume=True)
        self.assertFalse(self.calls)

    def test_slurm_uses_four_disjoint_runs_and_scheduler_gpu_binding(self):
        self.context()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-job", "SLURM_NTASKS": "4"}), \
                patch.object(hpc, "invoke", side_effect=self.invoke), patch.object(hpc.subprocess, "run") as srun:
            hpc.launch(self.parent)
            state = json.loads((self.parent/"slurm_registry.json").read_text())
            self.assertEqual(len({v["path"] for v in state["slots"].values()}), 4)
            command = srun.call_args.args[0]
            for flag in ["--ntasks=4", "--gpus-per-task=1", "--gpu-bind=single:1", "--kill-on-bad-exit=1"]:
                self.assertIn(flag, command)
            self.calls.clear()
            hpc.launch(self.parent, resume=True)
            self.assertFalse(self.calls)  # no replacement runs on resubmission

    def test_failed_srun_never_emits_success_receipt(self):
        self.context()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-job", "SLURM_NTASKS": "4"}), \
                patch.object(hpc, "invoke", side_effect=self.invoke), \
                patch.object(hpc.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "srun")), \
                self.assertRaises(subprocess.CalledProcessError):
            hpc.launch(self.parent)
        self.assertFalse((self.parent/"slurm_complete.json").exists())

    def test_login_node_launch_is_rejected_before_writes(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "", "SLURM_NTASKS": "0"}), self.assertRaises(RuntimeError):
            hpc.launch(self.parent)
        self.assertFalse((self.parent/"slurm_registry.json").exists())

    def test_worker_uses_assigned_gpu_and_its_own_parent(self):
        self.context()
        child = Path(self.invoke(["new-run"]))
        config = hpc.SHARDS[2]
        (self.parent/"slurm_registry.json").write_text(json.dumps({"slots": {"2": {
            "path": child.relative_to(self.workspace).as_posix(), "config": config}}}))
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                               get_device_name=lambda i: "NVIDIA A800 80GB PCIe")
        with patch.dict(os.environ, {"SLURM_PROCID": "2", "CUDA_VISIBLE_DEVICES": "scheduler-assigned"}), \
                patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}), \
                patch.object(sys, "version_info", (3, 11, 0)), patch.object(hpc, "invoke") as invoke:
            hpc.worker(self.parent)
            command = invoke.call_args_list[0].args[0]
            self.assertEqual(command[:3], ["exec", "--run-dir", str(child)])
            self.assertEqual(command[-1], config)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "scheduler-assigned")
            cuda.device_count = lambda: 4
            invoke.reset_mock()
            with self.assertRaises(RuntimeError): hpc.worker(self.parent)
            invoke.assert_not_called()


if __name__ == "__main__": unittest.main()
