#!/usr/bin/env bash
#SBATCH --job-name=dlf-adfin
#SBATCH --account=torch_pr_106_tandon_advanced
#SBATCH --output=/scratch/fn2174/DL-Final/data/logs/adfin-val-%A_%a.out
#SBATCH --error=/scratch/fn2174/DL-Final/data/logs/adfin-val-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# SBATCH arraysize is auto from list (17 adapters); inclusive 0-16:
#SBATCH --array=0-16
#
# Full val scoring for every adapter in data/adapters/, using frozen pipeline defaults:
#   letter mode, img 384, mcq-choice-max-chars 160, seed 42 (full-val sample), CSVs under
#   data/results/final_val_letter384/ (override with SCORES_OUT_DIR).
#
# Submit:
#   cd /scratch/fn2174/DL-Final && sbatch pyfiles/sbatch_final_val_all_adapters.sbatch
#
# Optional overrides:
#   ADAPTER_LIST=path/to/list.txt IMG_SIZE=224 MODES=letter sbatch ...

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"

LIST="${ADAPTER_LIST:-${ROOT}/data/adapter_eval_lists/final_val_full_all_adapters.txt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Need SLURM_ARRAY_TASK_ID}"
LINE="$(sed -n "$((TASK_ID + 1))p" "${LIST}" | sed 's/[[:space:]]//g')"
if [[ -z "${LINE}" ]]; then
  echo "No adapter on line $((TASK_ID + 1)) of ${LIST}" >&2
  exit 1
fi

ADAPTER_DIR_REL="data/adapters/${LINE}"
if [[ ! -d "${ROOT}/${ADAPTER_DIR_REL}" ]]; then
  echo "Missing adapter dir: ${ROOT}/${ADAPTER_DIR_REL}" >&2
  exit 1
fi

# --- mirror sbatch_eval_scoring_modes defaults + frozen “final decisions” ---
export SPLIT="${SPLIT:-val}"
export N="${N:-10000}"
export SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
export IMG_SIZE="${IMG_SIZE:-384}"
export MODES="${MODES:-letter}"
export MCQ_CHOICE_MAX_CHARS="${MCQ_CHOICE_MAX_CHARS:-160}"
export SCORES_OUT_DIR="${SCORES_OUT_DIR:-data/results/final_val_letter384}"
export ADAPTER_DIR="${ADAPTER_DIR_REL}"
export RUN_NAME="${RUN_NAME_OVERRIDE:-${LINE}_letter384_seed${SHUFFLE_SEED}}"

mkdir -p data/logs "${ROOT}/${SCORES_OUT_DIR}"

exec bash "${ROOT}/pyfiles/sbatch_eval_scoring_modes.sbatch"
