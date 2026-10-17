#!/bin/bash
#SBATCH --job-name=e010-last12
#SBATCH --partition=nvidia_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=140G

# Submit from the repository root, with an explicit registered coordinator run.
# Supply --output/--error under that run's logs/ when calling sbatch (see HPC.md).
set -euo pipefail

RUN_DIR="${1:?Usage: sbatch [log options] scripts/slurm/e010_a800.sh COORDINATOR_RUN [run|resume]}"
ACTION="${2:-run}"
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
python -u scripts/experiment.py exec --run-dir "$RUN_DIR" -- scripts/e010_slurm.py "$ACTION"
