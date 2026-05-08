#!/usr/bin/env bash
#SBATCH --job-name=dlf-submit-1149
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/make-submission-fullft-1149-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/make-submission-fullft-1149-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Submission for adapter saved by train-mm-fullft-8156775.out
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

ADAPTER_DIR="/scratch/fn2174/DL-Final/data/adapters/multimodal_fullft_20260507_1149"
IMG_SIZE=224
SUB_RUN_NAME="multimodal_fullft_20260507_1149"

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python -m pyfiles.make_submission \
  --root "${ROOT}" \
  --run-name "${SUB_RUN_NAME}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --img-size "${IMG_SIZE}" \
  --out "${ROOT}/data/submission.csv"

python -m pyfiles.check_submission \
  --root "${ROOT}" \
  --submission "${ROOT}/data/submission.csv"

cp -f "${ROOT}/data/submission.csv" "${ROOT}/submission.csv"
echo "Wrote ${ROOT}/submission.csv"
