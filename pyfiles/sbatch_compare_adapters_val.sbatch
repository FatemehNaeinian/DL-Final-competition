#!/usr/bin/env bash
#SBATCH --job-name=dlf-adcmp
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/adapter-cmp-%j.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/adapter-cmp-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Compare base SmolVLM vs saved adapters on the SAME small val subset
# (multimodal scoring, baseline prompt from scoring.py).
#
# Override partition:
#   sbatch --partition=h100_tand pyfiles/sbatch_compare_adapters_val.sbatch
#
# Env overrides:
#   VAL_N=100 SHUFFLE_SEED=42 GPU=0

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

VAL_N="${VAL_N:-100}"
IMG_SIZE="${IMG_SIZE:-224}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

SHUF_ARGS=()
if [[ -n "${SHUFFLE_SEED}" ]]; then
  SHUF_ARGS+=(--shuffle-seed "${SHUFFLE_SEED}")
fi

python -m pyfiles.verify_training_env || true

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# Extra adapters: append RUNS_EXTRA e.g.
# RUNS_EXTRA='plan_lora:data/adapters/lora_attn_mlp_plan' bash ...

RUNS=(
  "base:-"
  "quick_textonly_dora:data/quick_textonly_dora"
  "text_only_attn_mlp:data/runs/text_only_attn_mlp/adapter"
)

if [[ -n "${RUNS_EXTRA:-}" ]]; then
  RUNS+=("${RUNS_EXTRA}")
fi

python -m pyfiles.eval_adapters \
  --root "${ROOT}" \
  --val-n "${VAL_N}" \
  --img-size "${IMG_SIZE}" \
  "${SHUF_ARGS[@]}" \
  --runs "${RUNS[@]}"

echo "Per-model CSVs: val_eval_<run>_n${VAL_N}_*.csv"
echo "Summary: val_summary_adapter_compare_n${VAL_N}_seed${SHUFFLE_SEED}.csv (if seed set)"
