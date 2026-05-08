#!/usr/bin/env bash
#SBATCH --job-name=dlf-mcqct
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/train-mcq-cont-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/train-mcq-cont-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Multimodal QLoRA + --mcq-contrastive (--mcq-completion-mode letter | letter_choice | choice_text).
# Training image resize: FT_IMG_SIZE (default 224). Inference for val/submission: SUBMISSION_IMG_SIZE.
#
# Example (steps 1–2 of your sweep):
#   cd /scratch/fn2174/DL-Final
#   RUN_NAME=mcq224_letter LR=5e-5 MCQ_COMPLETION_MODE=letter sbatch pyfiles/sbatch_mc_contrastive_lora.sbatch
#   RUN_NAME=mcq224_lchoice LR=5e-5 MCQ_COMPLETION_MODE=letter_choice sbatch pyfiles/sbatch_mc_contrastive_lora.sbatch
#
# Optional choice-order shuffle (step 4):
#   MCQ_SHUFFLE_CHOICE_ORDER=1 sbatch ...
#
# LoRA preset: CFG=attn_only_lora (default) | attn_mlp_lora | …

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

CFG="${CFG:-attn_only_lora}"
RUN_NAME="${RUN_NAME:-mcq_contrast_${MCQ_COMPLETION_MODE:-letter}_$(date +%Y%m%d_%H%M)}"
MCQ_COMPLETION_MODE="${MCQ_COMPLETION_MODE:-letter}"
LR="${LR:-5e-5}"
FT_IMG_SIZE="${FT_IMG_SIZE:-224}"
SUBMISSION_IMG_SIZE="${SUBMISSION_IMG_SIZE:-224}"
EPOCHS="${EPOCHS:-1}"
TRAIN_N="${TRAIN_N:-3109}"
VAL_N="${VAL_N:-1048}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
RESUME="${RESUME:-none}"

MCQ_SHUFFLE_CHOICE_ORDER="${MCQ_SHUFFLE_CHOICE_ORDER:-0}"

SUBMIT_ARGS=()
if [[ "${MAKE_SUBMISSION}" == "0" ]]; then
  SUBMIT_ARGS+=(--no-make-submission)
fi

SHUFF_ARGS=()
if [[ "${MCQ_SHUFFLE_CHOICE_ORDER}" == "1" ]]; then
  SHUFF_ARGS+=(--mcq-shuffle-choice-order)
fi

EXTRA_PY_ARGS=()
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

# Avoid periodic eval OOMs (runs val after train anyway).
if [[ "${NO_EVAL:-1}" == "1" ]]; then
  EXTRA_PY_ARGS+=(--no-eval)
fi

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "MCQ contrastive LoRA | mcq-completion=${MCQ_COMPLETION_MODE} | ft=${FT_IMG_SIZE} | LR=${LR} | cfg=${CFG} | shuffle_choice=${MCQ_SHUFFLE_CHOICE_ORDER}"

python -m pyfiles.train_adapter \
  --root "${ROOT}" \
  --run-name "${RUN_NAME}" \
  --adapter-out "data/adapters/${RUN_NAME}" \
  --cfg "${CFG}" \
  --epochs "${EPOCHS}" \
  --train-n "${TRAIN_N}" \
  --val-n "${VAL_N}" \
  --ft-img-size "${FT_IMG_SIZE}" \
  --learning-rate "${LR}" \
  --submission-img-size "${SUBMISSION_IMG_SIZE}" \
  --mcq-contrastive \
  --mcq-completion-mode "${MCQ_COMPLETION_MODE}" \
  --resume "${RESUME}" \
  "${EXTRA_PY_ARGS[@]}" \
  "${SHUFF_ARGS[@]}" \
  "${SUBMIT_ARGS[@]}"

echo "Adapter: ${ROOT}/data/adapters/${RUN_NAME}"
