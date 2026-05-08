#!/usr/bin/env bash
#SBATCH --job-name=dlf-eval2s
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/eval-2sizes-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/eval-2sizes-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#
# Evaluate one adapter checkpoint at --img-size 224 and 384 (step 3 of MCQ sweep).
#
#   cd /scratch/fn2174/DL-Final
#   ADAPTER=data/adapters/mcq224_letter MCQ_COMPLETION_MODE=letter VAL_N=1048 \\
#     RUNTAG=mcq224_letter sbatch pyfiles/sbatch_eval_adapter_img_sizes.sbatch
#
# Optional: SHUFFLE_SEED=42 CAPTIONS_CSV=data/captions_val.csv GPU=0

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"
mkdir -p data/logs

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/fn2174/envs/ijepa

ADAPTER="${ADAPTER:-}"
if [[ -z "${ADAPTER}" ]]; then
  echo "Set ADAPTER to adapter dir, e.g. data/adapters/mcq224_letter" >&2
  exit 1
fi

MCQ_COMPLETION_MODE="${MCQ_COMPLETION_MODE:-letter}"
RUNTAG="${RUNTAG:-best}"
VAL_N="${VAL_N:-1048}"
SHUFFLE_SEED="${SHUFFLE_SEED:-}"
CAPTIONS_CSV="${CAPTIONS_CSV:-}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${ROOT}/data/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HOME"

CAP_ARGS=()
if [[ -n "${CAPTIONS_CSV}" ]]; then
  CAP_ARGS+=(--captions-csv "${ROOT}/${CAPTIONS_CSV}")
fi

SHUF_ARGS=()
if [[ -n "${SHUFFLE_SEED}" ]]; then
  SHUF_ARGS+=(--shuffle-seed "${SHUFFLE_SEED}")
fi

FULL_ADAPTER="${ROOT}/${ADAPTER}"

for SZ in 224 384; do
  NAME="${RUNTAG}_eval${SZ}"
  python -m pyfiles.eval_adapters \
    --root "${ROOT}" \
    --val-n "${VAL_N}" \
    --img-size "${SZ}" \
    --mcq-completion-mode "${MCQ_COMPLETION_MODE}" \
    "${CAP_ARGS[@]}" \
    "${SHUF_ARGS[@]}" \
    --runs "${NAME}:${FULL_ADAPTER}"
done

echo "Per-resolution CSV summaries under ${ROOT}/data/val_summary_*.csv and val_eval_*_n*.csv"


