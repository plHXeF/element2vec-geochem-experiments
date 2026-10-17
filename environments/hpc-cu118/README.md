# E010 Linux CUDA bootstrap

Run `bash scripts/submit_e010.sh new` from the repository root after uploading
the database and complete frozen preparation run. The wrapper only submits two
jobs: `e010_env_cpu.sh` installs/validates the environment and transferred inputs
in `cpu_compute` (4 CPU cores, 16G RAM, no GPUs), then
`e010_a6000_start.sh` runs on four GPUs only after CPU setup succeeds. Slurm
`afterok` and `--kill-on-invalid-dep=yes` prevent GPU execution if setup fails.
The wrapper runs no Python, installation, hashes or data checks on the submission
node. The CPU job creates an isolated conda prefix under the shared workspace's
ignored `.hpc/e010-cu118-v1/` and configures `workspace.local.json`. The GPU job
only reuses that environment, checks CUDA execution and starts the last 12 runs;
it contains no package installation commands.

Python is selected from the 3.11 series; the installed patch is recorded in each
run. Direct scientific dependencies are pinned in `requirements.txt` and torch
is fixed to `2.5.1+cu118` from the official CUDA 11.8 wheel index. This CUDA recipe
is independent of the Windows CPU lock. It uses the installation command listed
by [PyTorch](https://pytorch.org/get-started/previous-versions/); actual driver
compatibility is checked with CUDA forward/backward on all four allocated GPUs.
It neither installs a system NVIDIA driver nor changes an existing CUDA toolkit.

Transitive pip/conda packages are resolved on the initial installation; this is
not a complete dependency lock. The installed pip inventory is saved as
`e010-installed.txt` in the prefix, and normal run records preserve package and
interpreter versions. Later submissions reuse the prefix without installing or
upgrading packages. The recipe marker, exact direct versions and `pip check`
are checked again; retain this prefix for strict training resume.

The CPU compute node must reach conda-forge, PyPI and the official PyTorch wheel
server on first installation. An inaccessible server stops the job before
training. Installation consumes CPU allocation time but requests no GPUs. If networking is
restricted, prepare an offline environment or request the platform's supported
installation route; do not move the installation onto a prohibited management
node. The environment is loaded through `apps/anaconda3/2021.05`, matching the
user-supplied cluster example.

Once setup is complete, submit `sbatch scripts/slurm/e010_a6000_start.sh new`
directly to skip the CPU setup queue. For strict resume, substitute
`resume "$LAUNCH_RUN"` for `new`. Both stages require the repository, environment
and workspace to be on storage shared by CPU and GPU nodes. The initial version
installed packages inside the four-GPU job; that installation path has been
removed. Already-submitted jobs retain their copied batch script: let a running
installation complete, reuse its prefix, and do not update source during active
training. No existing environment or run is deleted or automatically upgraded.

`GEOCHEM_WORKSPACE` defaults to the repository root, matching the documented
upload layout. Set it before submission for a separate shared workspace.
`E010_ENV_PREFIX` can select a different private environment prefix. An existing
unmanaged/incomplete prefix or changed recipe is rejected rather than silently
modified. Preserve it and choose a new prefix after diagnosing the failure.
A setup lock prevents concurrent installers from sharing a workspace; after
forced job termination, first confirm the old job has stopped before handling
any residual lock. No code edits or local paths belong in Git.
