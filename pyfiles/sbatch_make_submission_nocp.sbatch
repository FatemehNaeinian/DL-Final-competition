#!/usr/bin/env bash
#
# Generate a Kaggle submission CSV using a (multi)modal adapter on GPU,
# without overwriting the repo-root `submission.csv`.
#
# Usage:
#   cd /scratch/fn2174/DL-Final
#   RUN_NAME=... ADAPTER_DIR=... OUT_CSV=... IMG_SIZE=384 MCQ_COMPLETION_MODE=letter \
#     sbatch pyfiles/sbatch_make_submission_nocp.sbatch
#

#SBATCH --job-name=make-sub-nocp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/make-sub-nocp-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/make-sub-nocp-%j.err
#SBATCH --account=torch_pr_106_tandon_advanced

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

# ---- Configure via env overrides ----
RUN_NAME="${RUN_NAME:-multimodal_fullft_20260507_1149}"
ADAPTER_DIR="${ADAPTER_DIR:-data/adapters/${RUN_NAME}}"
OUT_CSV="${OUT_CSV:-data/submission_${RUN_NAME}.csv}"
IMG_SIZE="${IMG_SIZE:-384}"
MCQ_COMPLETION_MODE="${MCQ_COMPLETION_MODE:-letter}"
MCQ_CHOICE_MAX_CHARS="${MCQ_CHOICE_MAX_CHARS:-160}"

# Optional captions CSV (id,caption)
CAPTIONS_CSV="${CAPTIONS_CSV:-}"

echo "RUN_NAME=$RUN_NAME"
echo "ADAPTER_DIR=$ADAPTER_DIR"
echo "OUT_CSV=$OUT_CSV"
echo "IMG_SIZE=$IMG_SIZE"
echo "MCQ_COMPLETION_MODE=$MCQ_COMPLETION_MODE"
echo "MCQ_CHOICE_MAX_CHARS=$MCQ_CHOICE_MAX_CHARS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

ARGS=(
  --root "${ROOT}"
  --run-name "$RUN_NAME"
  --adapter-dir "$ADAPTER_DIR"
  --out "$OUT_CSV"
  --img-size "$IMG_SIZE"
  --mcq-completion-mode "$MCQ_COMPLETION_MODE"
  --mcq-choice-max-chars "$MCQ_CHOICE_MAX_CHARS"
)

if [[ -n "${CAPTIONS_CSV}" ]]; then
  ARGS+=(--captions-csv "$CAPTIONS_CSV")
fi

python -m pyfiles.make_submission "${ARGS[@]}"

echo "Done. Wrote: $OUT_CSV"

