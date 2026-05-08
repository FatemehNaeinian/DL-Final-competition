#!/usr/bin/env bash
#SBATCH --job-name=dlf-txt-hg
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/train-textonly-hint-grade-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/train-textonly-hint-grade-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Text-only QLoRA with Question + Hint + Grade in the prompt; >3 epochs (default 4).
#
# Override:
#   EPOCHS=5 RUN_NAME=my_txt_ft bash pyfiles/sbatch_train_textonly_hint_grade.sbatch
#
#   sbatch --partition=h100_tand pyfiles/sbatch_train_textonly_hint_grade.sbatch

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

RUN_NAME="${RUN_NAME:-text_only_hint_grade_e${EPOCHS:-4}}"
ADAPTER_OUT="${ADAPTER_OUT:-data/adapters/${RUN_NAME}}"
CFG="${CFG:-attn_mlp_lora}"
EPOCHS="${EPOCHS:-4}"
TRAIN_N="${TRAIN_N:-3109}"
VAL_N="${VAL_N:-1048}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
RESUME="${RESUME:-none}"

SUBMIT_ARGS=()
if [[ "${MAKE_SUBMISSION}" == "0" ]]; then
  SUBMIT_ARGS+=(--no-make-submission)
fi

python -m pyfiles.verify_training_env || true

python -m pyfiles.train_adapter \
  --root "${ROOT}" \
  --run-name "${RUN_NAME}" \
  --adapter-out "${ADAPTER_OUT}" \
  --cfg "${CFG}" \
  --epochs "${EPOCHS}" \
  --train-n "${TRAIN_N}" \
  --val-n "${VAL_N}" \
  --text-only \
  --text-only-fields "hint,grade" \
  --resume "${RESUME}" \
  "${SUBMIT_ARGS[@]}"

echo "Adapter: ${ROOT}/${ADAPTER_OUT}"
