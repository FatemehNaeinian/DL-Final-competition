#!/usr/bin/env bash
#SBATCH --job-name=dlf-scmodes
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/scmodes-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/scmodes-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#
# Evaluate scoring modes and dump per-option scores for a split slice.
#
# Usage:
#   cd /scratch/fn2174/DL-Final
#   SPLIT=val N=200 SHUFFLE_SEED=42 IMG_SIZE=224 RUN_NAME=fullft_val200_224 \
#     ADAPTER_DIR=data/adapters/multimodal_fullft_20260507_1149 sbatch pyfiles/sbatch_eval_scoring_modes.sbatch
#
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

SPLIT="${SPLIT:-val}"           # train|val
N="${N:-200}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
IMG_SIZE="${IMG_SIZE:-224}"
RUN_NAME="${RUN_NAME:-scmodes_${SPLIT}_${N}_${IMG_SIZE}}"
ADAPTER_DIR="${ADAPTER_DIR:-}"  # optional; omit => base model
MODES="${MODES:-letter,letter_choice,choice_text}"
MCQ_CHOICE_MAX_CHARS="${MCQ_CHOICE_MAX_CHARS:-160}"
# Optional: write CSVs elsewhere (relative to ROOT or absolute)
SCORES_OUT_DIR="${SCORES_OUT_DIR:-}"

ARGS=(
  --root "${ROOT}"
  --split "${SPLIT}"
  --n "${N}"
  --shuffle-seed "${SHUFFLE_SEED}"
  --img-size "${IMG_SIZE}"
  --run-name "${RUN_NAME}"
  --modes "${MODES}"
  --mcq-choice-max-chars "${MCQ_CHOICE_MAX_CHARS}"
)

if [[ -n "${ADAPTER_DIR}" ]]; then
  ARGS+=(--adapter-dir "${ROOT}/${ADAPTER_DIR}")
fi
if [[ -n "${SCORES_OUT_DIR}" ]]; then
  if [[ "${SCORES_OUT_DIR}" = /* ]]; then
    ARGS+=(--scores-out-dir "${SCORES_OUT_DIR}")
  else
    ARGS+=(--scores-out-dir "${ROOT}/${SCORES_OUT_DIR}")
  fi
fi

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -m pyfiles.eval_scoring_modes "${ARGS[@]}"

