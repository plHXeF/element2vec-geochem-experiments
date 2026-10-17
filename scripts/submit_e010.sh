#!/bin/bash
# Submission only: no Python, installation, hashing or dataset inspection here.
set -euo pipefail
[[ -f scripts/slurm/e010_env_cpu.sh ]] || { echo "Run from the repository root" >&2; exit 2; }
ACTION="${1:-new}"
[[ "$ACTION" == new || "$ACTION" == resume ]] || { echo "Usage: bash scripts/submit_e010.sh new | resume RUN" >&2; exit 2; }
if [[ "$ACTION" == resume && -z "${2:-}" ]]; then echo "resume requires the original coordinator run path" >&2; exit 2; fi
SETUP_REPLY=$(sbatch --parsable scripts/slurm/e010_env_cpu.sh)
SETUP_JOB="${SETUP_REPLY%%;*}"
[[ "$SETUP_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $SETUP_REPLY" >&2; exit 2; }
echo "CPU environment job: $SETUP_JOB (no GPUs)"
if ! TRAIN_REPLY=$(sbatch --parsable --dependency="afterok:$SETUP_JOB" --kill-on-invalid-dep=yes scripts/slurm/e010_a6000_start.sh "$@"); then
  echo "GPU submission failed; CPU setup $SETUP_JOB remains submitted. After it succeeds, submit e010_a6000_start.sh directly." >&2
  exit 1
fi
echo "GPU training job: ${TRAIN_REPLY%%;*} (waits for successful CPU setup)"
