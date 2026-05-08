#!/usr/bin/env bash
#SBATCH --job-name=dlf-mm-fullft
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/train-mm-fullft-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/train-mm-fullft-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#
# Full fine-tune SmolVLM (no LoRA / no DoRA): updates all weights in bf16/fp16.
# Needs substantially more GPU VRAM than QLoRA (~500M model but optimizer states add overhead).
#
# Usage:
#   cd /scratch/fn2174/DL-Final
#   sbatch pyfiles/sbatch_train_full_finetune_mm.sbatch
#
# Overrides:
#   RUN_NAME=smolv_full_e1 LEARNING_RATE=2e-5 EPOCHS=1 MAKE_SUBMISSION=1 sbatch pyfiles/sbatch_train_full_finetune_mm.sbatch
#   sbatch --partition=h200_tand pyfiles/sbatch_train_full_finetune_mm.sbatch

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

RUN_NAME="${RUN_NAME:-multimodal_fullft_$(date +%Y%m%d_%H%M)}"
ADAPTER_OUT="${ADAPTER_OUT:-data/adapters/${RUN_NAME}}"
TRAIN_N="${TRAIN_N:-3109}"
VAL_N="${VAL_N:-1048}"
EPOCHS="${EPOCHS:-1}"
FT_IMG_SIZE="${FT_IMG_SIZE:-224}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
RESUME="${RESUME:-none}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

SUBMIT_ARGS=()
if [[ "${MAKE_SUBMISSION}" == "0" ]]; then
  SUBMIT_ARGS+=(--no-make-submission)
fi

EXTRA_PY_ARGS=(--learning-rate "${LEARNING_RATE}")
if [[ -n "${TRAIN_CAPTIONS_CSV:-}" ]]; then
  EXTRA_PY_ARGS+=(--train-captions-csv "${ROOT}/${TRAIN_CAPTIONS_CSV}")
fi
if [[ -n "${INFER_CAPTIONS_CSV:-}" ]]; then
  EXTRA_PY_ARGS+=(--infer-captions-csv "${ROOT}/${INFER_CAPTIONS_CSV}")
fi
if [[ "${IMAGE_AUG:-0}" == "1" ]]; then
  EXTRA_PY_ARGS+=(--image-augment)
fi
if [[ -n "${FT_METADATA_FIELDS:-}" ]]; then
  EXTRA_PY_ARGS+=(--ft-metadata-fields "${FT_METADATA_FIELDS}")
fi
if [[ -n "${ANSWER_PREFIX:-}" ]]; then
  EXTRA_PY_ARGS+=(--answer-prefix "${ANSWER_PREFIX}")
fi
SUBMISSION_IMG_SIZE="${SUBMISSION_IMG_SIZE:-224}"
EXTRA_PY_ARGS+=(--submission-img-size "${SUBMISSION_IMG_SIZE}")

python -m pyfiles.verify_training_env || true

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "Multimodal FULL fine-tune | ft-img-size=${FT_IMG_SIZE} | epochs=${EPOCHS} | lr=${LEARNING_RATE}"

python -m pyfiles.train_adapter \
  --root "${ROOT}" \
  --run-name "${RUN_NAME}" \
  --adapter-out "${ADAPTER_OUT}" \
  --full-finetune \
  --epochs "${EPOCHS}" \
  --train-n "${TRAIN_N}" \
  --val-n "${VAL_N}" \
  --ft-img-size "${FT_IMG_SIZE}" \
  --resume "${RESUME}" \
  "${EXTRA_PY_ARGS[@]}" \
  "${SUBMIT_ARGS[@]}"

echo "Weights dir: ${ROOT}/${ADAPTER_OUT}"
echo "Checkpoints: ${ROOT}/data/runs/${RUN_NAME}/checkpoints/"
echo "Meta: ${ROOT}/${ADAPTER_OUT}/run_meta.json"
