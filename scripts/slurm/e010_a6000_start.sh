#!/bin/bash
#SBATCH --job-name=e010-last12
#SBATCH --partition=nvidia_gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=35G
#SBATCH --output=slurm-e010-%j.out
#SBATCH --error=slurm-e010-%j.err

# Submit with sbatch from the repository root. All setup runs on the compute node.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit this script with sbatch; setup requires a compute allocation" >&2; exit 2; }
[[ "${SLURM_NTASKS:-0}" == 4 ]] || { echo "Exactly four Slurm tasks are required" >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ -f scripts/e010_hpc_setup.py ]] || { echo "Submit from the repository root" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Commit source changes before training" >&2; exit 2; }
export GEOCHEM_WORKSPACE="${GEOCHEM_WORKSPACE:-$PWD}"
PREP_RUN="$GEOCHEM_WORKSPACE/runs/E010/20261017T013747Z-s42-0a349d"
[[ -f "$GEOCHEM_WORKSPACE/data/whole_rock/whole-rock-v1/geochem.db" && -f "$PREP_RUN/artifact_checksums.json" && -f "$PREP_RUN/prepared_data/manifest.json" ]] || {
  echo "Upload geochem.db and the complete frozen preparation run before submitting (see experiments/E010/HPC.md)" >&2; exit 2;
}

unset PYTHONPATH GEOCHEM_RUN_DIR
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load apps/anaconda3/2021.05
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$GEOCHEM_WORKSPACE/.hpc"
SETUP_LOCK="$GEOCHEM_WORKSPACE/.hpc/e010-setup.lock"
mkdir "$SETUP_LOCK" || { echo "Another setup may be active; inspect $SETUP_LOCK before retrying" >&2; exit 2; }
trap 'rmdir "$SETUP_LOCK"' EXIT
ENV_PREFIX="${E010_ENV_PREFIX:-$GEOCHEM_WORKSPACE/.hpc/e010-cu118-v1}"
RECIPE="$(sha256sum environments/hpc-cu118/requirements.txt | cut -d ' ' -f 1)-torch2.5.1cu118-py311-v1"
if [[ -f "$ENV_PREFIX/.e010-ready" ]]; then
  [[ "$(cat "$ENV_PREFIX/.e010-ready")" == "$RECIPE" ]] || { echo "Environment recipe changed; choose a new E010_ENV_PREFIX" >&2; exit 2; }
else
  [[ ! -e "$ENV_PREFIX" ]] || { echo "Incomplete or unmanaged environment at $ENV_PREFIX; preserve it and select a new E010_ENV_PREFIX" >&2; exit 2; }
  echo "Creating E010 Python 3.11 / torch 2.5.1 CUDA 11.8 environment on compute node"
  conda create --yes --prefix "$ENV_PREFIX" --override-channels --channel conda-forge python=3.11 pip=24.3.1
  "$ENV_PREFIX/bin/python" -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r environments/hpc-cu118/requirements.txt
  "$ENV_PREFIX/bin/python" -m pip install --only-binary=:all: torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
  "$ENV_PREFIX/bin/python" -m pip check
  "$ENV_PREFIX/bin/python" -m pip freeze > "$ENV_PREFIX/e010-installed.txt"
  printf '%s\n' "$RECIPE" > "$ENV_PREFIX/.e010-ready"
fi
conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m pip check
python scripts/e010_hpc_setup.py
# Release the setup lock before the long training allocation continues.
rmdir "$SETUP_LOCK"
trap - EXIT
export E010_CONDA_ENV="$ENV_PREFIX"
exec bash scripts/slurm/e010_a6000.sh "$@"
