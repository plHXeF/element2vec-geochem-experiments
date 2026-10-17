"""Synthetic transfer/configuration checks; no CUDA, conda or private data."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import e010_hpc_setup as setup


class SetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prep = self.root/"runs/E010/frozen-fixture"
        self.prepared = self.prep/"prepared_data"; self.prepared.mkdir(parents=True)
        self.database = self.root/"data/whole_rock/whole-rock-v1/geochem.db"
        self.database.parent.mkdir(parents=True); self.database.write_bytes(b"synthetic source")
        (self.prepared/"values.npy").write_bytes(b"synthetic prepared bytes")
        self.write(self.prepared/"manifest.json", {
            "source_sha256": setup.digest(self.database),
            "files": {"values.npy": setup.digest(self.prepared/"values.npy")}})
        self.write(self.prep/"manifest.json", {
            "run_id": "frozen-fixture", "frozen": True, "status": "completed",
            "inputs": {"whole_rock": {"sha256": setup.digest(self.database)}}})
        for name in ("config.logical.yaml", "config.resolved.yaml", "environment.txt", "summary.md"):
            (self.prep/name).write_text("fixture")
        # Emulate Windows-recorded keys on either test host.
        self.write(self.prep/"artifact_checksums.json", {
            str(p.relative_to(self.prep)).replace("/", "\\"): setup.digest(p)
            for p in self.prep.rglob("*") if p.is_file() and p.name != "manifest.json"})
        checks = json.loads((self.prep/"artifact_checksums.json").read_text())
        checks["prepared_data\\manifest.json"] = setup.digest(self.prepared/"manifest.json")
        self.write(self.prep/"artifact_checksums.json", checks)
        self.repo = self.root/"repo"
        (self.repo/"experiments/E010").mkdir(parents=True)
        self.write(self.repo/"experiments/E010/experiment.json", {"prepared_input": {
            "run_id": "frozen-fixture", "prepared_fingerprint": setup.digest(self.prepared/"manifest.json")}})
        self.addCleanup(patch.stopall)
        patch.object(setup, "REPO", self.repo).start()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def test_windows_checksum_paths_verified_without_rewriting_frozen_input(self):
        before = {str(p): p.read_bytes() for p in self.prep.rglob("*") if p.is_file()}
        assets = setup.verify_inputs(self.root)
        self.assertEqual(assets["e010_prepared"], "runs/E010/frozen-fixture/prepared_data")
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.prep.rglob("*") if p.is_file()})

    def test_damaged_upload_is_rejected(self):
        (self.prepared/"values.npy").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "differs"):
            setup.verify_inputs(self.root)
        self.assertFalse((self.root/"workspace.local.json").exists())

    def test_prepared_files_cannot_be_omitted_from_transfer_inventory(self):
        checks_path = self.prep/"artifact_checksums.json"
        checks = json.loads(checks_path.read_text()); del checks["prepared_data\\values.npy"]
        self.write(checks_path, checks)
        (self.prepared/"values.npy").unlink()
        with self.assertRaises(FileNotFoundError): setup.verify_inputs(self.root)

    def test_wrong_database_is_rejected(self):
        self.database.write_bytes(b"wrong database")
        with self.assertRaisesRegex(ValueError, "Database differs"):
            setup.verify_inputs(self.root)

    def test_configuration_preserves_other_assets_and_conflicts_do_not_overwrite(self):
        path = self.root/"workspace.local.json"
        self.write(path, {"schema_version": 1, "assets": {"other": "data/other"}})
        assets = setup.verify_inputs(self.root)
        setup.configure(self.root, assets)
        self.assertEqual(json.loads(path.read_text())["assets"]["other"], "data/other")
        setup.configure(self.root, assets)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "points elsewhere"):
            setup.configure(self.root, {"e010_prepared": "runs/something-else"})
        self.assertEqual(path.read_bytes(), before)

    def test_paths_cannot_escape_the_frozen_run(self):
        for name in ("../outside", "..\\outside", "C:\\private\\input", "/absolute"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                setup.transferred_path(self.prep, name)

    def test_setup_refuses_to_run_without_compute_allocation(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": ""}), \
                patch.object(setup, "verify_inputs") as verify, self.assertRaises(RuntimeError):
            setup.main()
        verify.assert_not_called()

    def test_environment_failure_cannot_write_configuration(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture", "GEOCHEM_WORKSPACE": str(self.root)}), \
                patch.object(setup, "check_environment", side_effect=RuntimeError("CUDA unavailable")), \
                redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            setup.main()
        self.assertFalse((self.root/"workspace.local.json").exists())

    def test_cpu_phase_defers_cuda_execution(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture", "GEOCHEM_WORKSPACE": str(self.root)}), \
                patch.object(setup, "check_environment") as check, redirect_stdout(io.StringIO()):
            setup.main(prepare_only=True)
        check.assert_called_once_with(require_cuda=False)
        self.assertTrue((self.root/"workspace.local.json").exists())


if __name__ == "__main__": unittest.main()
