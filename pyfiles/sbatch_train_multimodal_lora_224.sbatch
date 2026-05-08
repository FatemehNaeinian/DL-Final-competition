#!/usr/bin/env bash
#SBATCH --job-name=dlf-mm224
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/train-mm-lora-224-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/train-mm-lora-224-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Multimodal QLoRA: images + text (LoRAMCQDataset), training resize **224×224**.
# Uses short MCQ prompt in train_adapter.build_prompt_ft (<image> + question + choices + Answer:).
#
# Usage:
#   cd /scratch/fn2174/DL-Final
#   sbatch pyfiles/sbatch_train_multimodal_lora_224.sbatch
#
# LoRA presets (see pyfiles/lora_configs.py):
#   attn_only_lora   — r=16, q_proj+v_proj only, no DoRA (fewer modules than attn_mlp_lora)
#   attn_mlp_lora    — r=4, q/v + gate/up/down MLP
#   attn_mlp_dora    — same modules as attn_mlp_lora but DoRA
#
# Overrides:
#   RUN_NAME=mm224_a CFG=attn_only_lora EPOCHS=2 MAKE_SUBMISSION=1 sbatch ...
#   RUN_NAME=mm224_a CFG=attn_mlp_lora EPOCHS=2 MAKE_SUBMISSION=0 RESUME=none sbatch ...
#   TRAIN_CAPTIONS_CSV=data/captions_train.csv INFER_CAPTIONS_CSV=data/captions_test.csv
#   IMAGE_AUG=1  FT_METADATA_FIELDS=grade,subject,topic  ANSWER_PREFIX='The correct answer is:'
#   SUBMISSION_IMG_SIZE=288
#   sbatch --partition=h200_tand pyfiles/sbatch_train_multimodal_lora_224.sbatch

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

RUN_NAME="${RUN_NAME:-multimodal_lora_224_$(date +%Y%m%d_%H%M)}"
ADAPTER_OUT="${ADAPTER_OUT:-data/adapters/${RUN_NAME}}"
CFG="${CFG:-attn_only_lora}"
TRAIN_N="${TRAIN_N:-3109}"
VAL_N="${VAL_N:-1048}"
EPOCHS="${EPOCHS:-1}"
FT_IMG_SIZE="${FT_IMG_SIZE:-224}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
RESUME="${RESUME:-none}"

SUBMIT_ARGS=()
if [[ "${MAKE_SUBMISSION}" == "0" ]]; then
  SUBMIT_ARGS+=(--no-make-submission)
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
SUBMISSION_IMG_SIZE="${SUBMISSION_IMG_SIZE:-224}"
EXTRA_PY_ARGS+=(--submission-img-size "${SUBMISSION_IMG_SIZE}")

python -m pyfiles.verify_training_env || true

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "Multimodal LoRA | ft-img-size=${FT_IMG_SIZE} | cfg=${CFG} | epochs=${EPOCHS}"

python -m pyfiles.train_adapter \
  --root "${ROOT}" \
  --run-name "${RUN_NAME}" \
  --adapter-out "${ADAPTER_OUT}" \
  --cfg "${CFG}" \
  --epochs "${EPOCHS}" \
  --train-n "${TRAIN_N}" \
  --val-n "${VAL_N}" \
  --ft-img-size "${FT_IMG_SIZE}" \
  --resume "${RESUME}" \
  "${EXTRA_PY_ARGS[@]}" \
  "${SUBMIT_ARGS[@]}"

echo "Adapter: ${ROOT}/${ADAPTER_OUT}"
echo "Checkpoints: ${ROOT}/data/runs/${RUN_NAME}/checkpoints/"
echo "Meta: ${ROOT}/${ADAPTER_OUT}/run_meta.json"
