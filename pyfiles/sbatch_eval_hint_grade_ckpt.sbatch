#!/usr/bin/env bash
#SBATCH --job-name=dlf-eval-hg
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/eval-hint-grade-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/eval-hint-grade-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Multimodal val eval on a PEFT checkpoint dir (e.g. interrupted train save).
#
# Defaults target text_hint_grade_e4 latest checkpoint before SSH drop.
# Override:
#   CKPT_DIR=data/runs/other/checkpoints/checkpoint-XXX VAL_N=500 bash ...

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"
mkdir -p data/logs

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/fn2174/envs/ijepa

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-${ROOT}/data/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HOME"

CKPT_DIR="${CKPT_DIR:-data/runs/text_hint_grade_e4/checkpoints/checkpoint-700}"
RUN_TAG="${RUN_TAG:-text_hint_grade_e4_ckpt700}"
VAL_N="${VAL_N:-1048}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
IMG_SIZE="${IMG_SIZE:-224}"

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python -m pyfiles.eval_adapters \
  --root "${ROOT}" \
  --val-n "${VAL_N}" \
  --img-size "${IMG_SIZE}" \
  --shuffle-seed "${SHUFFLE_SEED}" \
  --runs \
    "base:-" \
    "${RUN_TAG}:${CKPT_DIR}"

echo "Summary CSV written under ${ROOT}/data/val_summary_adapter_compare_*.csv"
