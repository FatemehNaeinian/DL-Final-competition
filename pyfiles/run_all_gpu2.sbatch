#!/bin/bash
#SBATCH --job-name=dlfinal
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/home/fn2174/DL_final/data/logs/%x-%j.out
#SBATCH --error=/home/fn2174/DL_final/data/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# This script runs the full 6-experiment pipeline on a single GPU.
# It works both as an sbatch job AND as a plain `bash` script if SLURM
# is not installed (this box doesn't have sbatch).
#
# For 3 GPUs in parallel (train + eval shards + finalize), use:
#   bash /home/fn2174/DL_final/pyfiles/run_parallel_3gpu.sh
#
# Usage (no SLURM):
#   bash /home/fn2174/DL_final/pyfiles/run_all_gpu2.sbatch
#
# Override defaults with environment variables, e.g.:
#   GPU=2 TRAIN_N=2000 VAL_N=500 EPOCHS=1 bash pyfiles/run_all_gpu2.sbatch

set -euo pipefail

ROOT="/home/fn2174/DL_final"
cd "$ROOT"
mkdir -p data/logs

GPU="${GPU:-0}"
TRAIN_N="${TRAIN_N:-1500}"
VAL_N="${VAL_N:-500}"
EPOCHS="${EPOCHS:-1}"
FT_IMG_SIZE="${FT_IMG_SIZE:-192}"
IMG_SIZE="${IMG_SIZE:-224}"

export CUDA_VISIBLE_DEVICES="$GPU"

source "$ROOT/.venv/bin/activate"

python -c "import torch; print('cuda:', torch.cuda.is_available(), 'n=', torch.cuda.device_count()); print('gpu0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# -----------------------------------------------------------------------------
# Run the orchestrator: trains 5 LoRA variants, evaluates each on val.csv,
# also evaluates the base model with self-generated image captions, then
# writes submission.csv using the winner.
# -----------------------------------------------------------------------------
python -m pyfiles.orchestrate \
  --train-n "$TRAIN_N" \
  --val-n "$VAL_N" \
  --epochs "$EPOCHS" \
  --ft-img-size "$FT_IMG_SIZE" \
  --img-size "$IMG_SIZE" \
  --skip-train-existing \
  --make-final-submission

echo "All done. Comparison: $ROOT/data/experiments_summary.csv"
echo "Winner info:         $ROOT/data/winner.json"
echo "Final submission:    $ROOT/submission.csv"
