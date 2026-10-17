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

# GPU stage: reuse the CPU-installed environment, never install packages here.
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
# Legacy conda.sh reads PS1 even in non-interactive shells under set -u.
export PS1="${PS1-}"
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${E010_ENV_PREFIX:-$GEOCHEM_WORKSPACE/.hpc/e010-cu118-v1}"
RECIPE="$(sha256sum environments/hpc-cu118/requirements.txt | cut -d ' ' -f 1)-torch2.5.1cu118-py311-v1"
[[ -f "$ENV_PREFIX/.e010-ready" ]] || { echo "Environment not ready; use bash scripts/submit_e010.sh (CPU setup first)" >&2; exit 2; }
[[ "$(cat "$ENV_PREFIX/.e010-ready")" == "$RECIPE" ]] || { echo "Environment recipe changed; run CPU setup with a new E010_ENV_PREFIX" >&2; exit 2; }
conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m pip check
python scripts/e010_hpc_setup.py
export E010_CONDA_ENV="$ENV_PREFIX"
exec bash scripts/slurm/e010_a6000.sh "$@"
