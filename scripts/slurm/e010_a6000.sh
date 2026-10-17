#!/bin/bash
#SBATCH --job-name=e010-last12
#SBATCH --partition=nvidia_gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=35G

# Submit from the repository root. New coordinator registration happens only
# inside this allocation, never on the file-management node (see HPC.md).
set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run only inside a Slurm allocation" >&2; exit 2; }
REQUEST="${1:-new}"
RUN_DIR=''
if [[ "$REQUEST" == new ]]; then
  ACTION=run
elif [[ "$REQUEST" == resume || "$REQUEST" == run ]]; then
  ACTION="$REQUEST"
  RUN_DIR="${2:?Supply the original coordinator run path}"
else
  # Retain compatibility with the previous RUN_DIR [run|resume] interface.
  RUN_DIR="$REQUEST"
  ACTION="${2:-run}"
fi
[[ "$ACTION" == run || "$ACTION" == resume ]] || { echo "Expected run or resume" >&2; exit 2; }
: "${GEOCHEM_WORKSPACE:?Set GEOCHEM_WORKSPACE to your configured workspace}"

module load apps/anaconda3/2021.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${E010_CONDA_ENV:-element2vec-geochem}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ -f scripts/e010_slurm.py ]] || { echo "Submit from the repository root" >&2; exit 2; }
[[ "${SLURM_NTASKS:-0}" == 4 ]] || { echo "Exactly four Slurm tasks are required" >&2; exit 2; }
nvidia-smi
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR=$(python scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/hpc/scan_last12.json)
fi
echo "E010 coordinator run: $RUN_DIR"
python -u scripts/experiment.py exec --run-dir "$RUN_DIR" -- scripts/e010_slurm.py "$ACTION"
