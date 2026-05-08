#!/usr/bin/env bash
#SBATCH --job-name=dlf-valcmp
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/val-cmp-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/val-cmp-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Val comparison driver (see pyfiles/compare_val_prompts.py).
#
# A - Base: (1) baseline 224 (2) metadata 224 (3) baseline 384 (4) metadata 384
# B - LoRA: (5)(6) best --prompt from A at 224 and 384 (ADAPTER_DIR = LoRA checkpoint)
# C - DoRA: (7) best --prompt and --img-size (ADAPTER_DIR = DoRA checkpoint)
#
# Usage:
#   cd /scratch/fn2174/DL-Final
#   SUITE=A VAL_N=100 sbatch pyfiles/sbatch_compare_val_small.sbatch
#
#   SUITE=B ADAPTER_DIR=data/my_lora PROMPT=metadata VAL_N=100 sbatch ...
#
#   SUITE=C ADAPTER_DIR=data/my_dora PROMPT=metadata IMG_SIZE=384 VAL_N=100 sbatch ...
#
# Optional: SHUFFLE_SEED=42 GPU=0
#
# Run manually with bash (not sh):  bash pyfiles/sbatch_compare_val_small.sbatch

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"
mkdir -p data/logs

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/fn2174/envs/ijepa

SUITE="${SUITE:-A}"
VAL_N="${VAL_N:-100}"
SHUFFLE_SEED="${SHUFFLE_SEED:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
PROMPT="${PROMPT:-}"
IMG_SIZE="${IMG_SIZE:-384}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${ROOT}/data/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HOME"

python -c "import torch; print('cuda:', torch.cuda.is_available(), 'n=', torch.cuda.device_count()); print('gpu0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# Avoid bash arrays so plain `sh` / minimal shells cannot mis-parse the script.
run_compare_val() {
  if [[ -n "${SHUFFLE_SEED}" ]]; then
    python -m pyfiles.compare_val_prompts \
      --root "${ROOT}" \
      --val-n "${VAL_N}" \
      --shuffle-seed "${SHUFFLE_SEED}" \
      "$@"
  else
    python -m pyfiles.compare_val_prompts \
      --root "${ROOT}" \
      --val-n "${VAL_N}" \
      "$@"
  fi
}

case "${SUITE}" in
  A)
    if [[ -n "${ADAPTER_DIR}" ]]; then
      echo "SUITE=A ignores ADAPTER_DIR (base model only)." >&2
    fi
    run_compare_val --suite A
    ;;
  B)
    if [[ -z "${ADAPTER_DIR}" ]]; then
      echo "SUITE=B requires ADAPTER_DIR (LoRA checkpoint)." >&2
      exit 1
    fi
    if [[ -z "${PROMPT}" ]]; then
      echo "SUITE=B requires PROMPT=baseline or PROMPT=metadata (winner from suite A)." >&2
      exit 1
    fi
    run_compare_val --suite B --adapter-dir "${ADAPTER_DIR}" --prompt "${PROMPT}"
    ;;
  C)
    if [[ -z "${ADAPTER_DIR}" ]]; then
      echo "SUITE=C requires ADAPTER_DIR (DoRA checkpoint)." >&2
      exit 1
    fi
    if [[ -z "${PROMPT}" ]]; then
      echo "SUITE=C requires PROMPT=baseline or PROMPT=metadata." >&2
      exit 1
    fi
    run_compare_val --suite C --adapter-dir "${ADAPTER_DIR}" --prompt "${PROMPT}" --img-size "${IMG_SIZE}"
    ;;
  *)
    echo "SUITE must be A, B, or C (got: ${SUITE})" >&2
    exit 1
    ;;
esac

echo "Done. Summary: ${ROOT}/data/val_compare_suite_${SUITE}_summary.csv"
