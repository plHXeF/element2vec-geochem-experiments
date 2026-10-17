# E010 Linux CUDA bootstrap

Submit `scripts/slurm/e010_a6000_start.sh` from the repository root after
uploading the database and complete frozen preparation run. This creates an
isolated conda prefix under the workspace's ignored `.hpc/e010-cu118-v1/`,
configures `workspace.local.json` and starts the existing last-12 Slurm launcher.
Every installation, hash and CUDA check runs within the compute allocation.

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

The compute node must reach conda-forge, PyPI and the official PyTorch wheel
server on first installation. An inaccessible server stops the job before
training. Installation consumes time in the GPU allocation. If networking is
restricted, prepare an offline environment or request the platform's supported
installation route; do not move the installation onto a prohibited management
node. The environment is loaded through `apps/anaconda3/2021.05`, matching the
user-supplied cluster example.

`GEOCHEM_WORKSPACE` defaults to the repository root, matching the documented
upload layout. Set it before submission for a separate shared workspace.
`E010_ENV_PREFIX` can select a different private environment prefix. An existing
unmanaged/incomplete prefix or changed recipe is rejected rather than silently
modified. Preserve it and choose a new prefix after diagnosing the failure.
A setup lock prevents concurrent installers from sharing a workspace; after
forced job termination, first confirm the old job has stopped before handling
any residual lock. No code edits or local paths belong in Git.
