#!/usr/bin/env bash
#SBATCH --job-name=dlf-eval-B
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/eval-suite-B-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/eval-suite-B-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Run suite B twice: PROMPT=baseline and PROMPT=metadata (224 + 384 each run).
# Typical chaining:
#   sbatch --dependency=afterok:<train_jobid> pyfiles/sbatch_eval_suite_B_dual.sbatch
#
# Env:
#   ADAPTER_DIR=data/adapters/<run>   (required)
#   VAL_N=500 SHUFFLE_SEED=42 GPU=0
#
# Optional partition:
#   sbatch --partition=h200_tand ...

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"
mkdir -p data/logs

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/fn2174/envs/ijepa

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${ROOT}/data/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HOME"

ADAPTER_DIR="${ADAPTER_DIR:?Set ADAPTER_DIR (PEFT adapter path)}"
VAL_N="${VAL_N:-500}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

SHUF_ARGS=()
if [[ -n "${SHUFFLE_SEED}" ]]; then
  SHUF_ARGS+=(--shuffle-seed "${SHUFFLE_SEED}")
fi

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

TAG_BASE="B_prompt_baseline_n${VAL_N}_seed${SHUFFLE_SEED}"
TAG_META="B_prompt_metadata_n${VAL_N}_seed${SHUFFLE_SEED}"

python -m pyfiles.compare_val_prompts \
  --root "${ROOT}" \
  --suite B \
  --adapter-dir "${ADAPTER_DIR}" \
  --prompt baseline \
  --val-n "${VAL_N}" \
  "${SHUF_ARGS[@]}" \
  --output-tag "${TAG_BASE}"

python -m pyfiles.compare_val_prompts \
  --root "${ROOT}" \
  --suite B \
  --adapter-dir "${ADAPTER_DIR}" \
  --prompt metadata \
  --val-n "${VAL_N}" \
  "${SHUF_ARGS[@]}" \
  --output-tag "${TAG_META}"

echo "Wrote summaries under ${ROOT}/data/val_compare_suite_B_*_summary.csv"
