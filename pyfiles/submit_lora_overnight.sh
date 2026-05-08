#!/usr/bin/env bash
# Submit several multimodal LoRA fine-tunes + validation + submission.csv per job.
#
# Maps to common Kaggle-strategy notes:
#   • attn_mlp_* + mlp_only_lora — attention vs MLP / hybrid targets (#1, #4)
#   • attn_mlp_dora — DoRA (#2)
#   • Optional captions: set TRAIN_CAPTIONS_CSV / INFER_CAPTIONS_CSV before running (#3)
#   • FT_METADATA_FIELDS — subject / grade / topic in FT prompt (#6)
#   • ANSWER_PREFIX — wording experiments (#5)
#   • Larger FT_IMG_SIZE — higher-res fine-tune (#7)
#   • IMAGE_AUG=1 — light augmentation (#8)
#   • Gradient checkpointing stays on in train_adapter (#9)
#
# Usage:
#   cd /scratch/fn2174/DL-Final
#   bash pyfiles/submit_lora_overnight.sh
#
# Optional:
#   PARTITION=h200_tand bash pyfiles/submit_lora_overnight.sh
#   ACCOUNT=torch_pr_106_tandon_advanced bash pyfiles/submit_lora_overnight.sh

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"

PART_ARGS=()
if [[ -n "${PARTITION:-}" ]]; then
  PART_ARGS+=(--partition="${PARTITION}")
fi
ACC_ARGS=()
if [[ -n "${ACCOUNT:-}" ]]; then
  ACC_ARGS+=(--account="${ACCOUNT}")
fi

SBATCH=(sbatch "${PART_ARGS[@]}" "${ACC_ARGS[@]}" --parsable pyfiles/sbatch_train_multimodal_lora_224.sbatch)

submit_one() {
  local id="$1"
  shift
  echo "--- ${id} ---" >&2
  (
    export RUN_NAME="$id"
    export ADAPTER_OUT="data/adapters/${id}"
    export RESUME="${RESUME:-none}"
    export TRAIN_N="${TRAIN_N:-3109}"
    export VAL_N="${VAL_N:-1048}"
    export EPOCHS="${EPOCHS:-2}"
    export MAKE_SUBMISSION="${MAKE_SUBMISSION:-1}"
    export CFG="${CFG:-attn_mlp_lora}"
    export FT_IMG_SIZE="${FT_IMG_SIZE:-224}"
    export SUBMISSION_IMG_SIZE="${SUBMISSION_IMG_SIZE:-224}"
    unset TRAIN_CAPTIONS_CSV INFER_CAPTIONS_CSV IMAGE_AUG FT_METADATA_FIELDS ANSWER_PREFIX || true
    while [[ $# -gt 0 ]]; do
      case "$1" in
        CFG=*|RUN_NAME=*|ADAPTER_OUT=*|EPOCHS=*|FT_IMG_SIZE=*|SUBMISSION_IMG_SIZE=*|\
TRAIN_N=*|VAL_N=*|MAKE_SUBMISSION=*|RESUME=*|TRAIN_CAPTIONS_CSV=*|INFER_CAPTIONS_CSV=*|\
IMAGE_AUG=*|FT_METADATA_FIELDS=*)
          export "$1"
          shift
          ;;
        *)
          echo "Unknown kwarg: $1" >&2
          exit 1
          ;;
      esac
    done
    "${SBATCH[@]}"
  )
}

echo "Submitting overnight multimodal LoRA jobs from ${ROOT}"

JOB_IDS=()

JOB_IDS+=("$(submit_one overnight_mm_attn_mlp_e2 CFG=attn_mlp_lora EPOCHS=2 FT_IMG_SIZE=224 MAKE_SUBMISSION=1)") # baseline MM LoRA
JOB_IDS+=("$(submit_one overnight_mm_attn_mlp_dora_e2 CFG=attn_mlp_dora EPOCHS=2 FT_IMG_SIZE=224 MAKE_SUBMISSION=1)") # DoRA
JOB_IDS+=("$(submit_one overnight_mm_mlp_only_e2 CFG=mlp_only_lora EPOCHS=2 FT_IMG_SIZE=224 MAKE_SUBMISSION=1)") # MLP-only targets
JOB_IDS+=("$(submit_one overnight_mm_attn_mlp_r2_meta CFG=attn_mlp_lora_r2 EPOCHS=2 FT_IMG_SIZE=224 FT_METADATA_FIELDS=grade,subject,topic MAKE_SUBMISSION=1)") # lower rank + metadata
JOB_IDS+=("$(submit_one overnight_mm_attn_mlp_e2_aug CFG=attn_mlp_lora EPOCHS=2 FT_IMG_SIZE=224 IMAGE_AUG=1 MAKE_SUBMISSION=1)") # augmentation
JOB_IDS+=("$(submit_one overnight_mm_attn_mlp_hi288 CFG=attn_mlp_lora EPOCHS=2 FT_IMG_SIZE=288 SUBMISSION_IMG_SIZE=288 MAKE_SUBMISSION=1)") # higher resolution

# Prompt wording (#5): value has spaces — set ANSWER_PREFIX in its own subshell (not KEY=value argv parsing).
JOB_IDS+=("$(
  export RUN_NAME="overnight_mm_answer_cue"
  export ADAPTER_OUT="data/adapters/overnight_mm_answer_cue"
  export CFG="attn_mlp_lora"
  export EPOCHS="${EPOCHS:-2}"
  export FT_IMG_SIZE=224
  export SUBMISSION_IMG_SIZE="${SUBMISSION_IMG_SIZE:-224}"
  export TRAIN_N="${TRAIN_N:-3109}"
  export VAL_N="${VAL_N:-1048}"
  export MAKE_SUBMISSION="${MAKE_SUBMISSION:-1}"
  export RESUME="${RESUME:-none}"
  export ANSWER_PREFIX="The correct answer is:"
  sbatch "${PART_ARGS[@]}" "${ACC_ARGS[@]}" --parsable pyfiles/sbatch_train_multimodal_lora_224.sbatch
)")

echo ""
echo "Submitted job IDs (multimodal LoRA + submission each):"
printf '  %s\n' "${JOB_IDS[@]}"
echo ""
echo "Logs: data/logs/train-mm-lora-224-<jobid>.out"
echo "Adapters: data/adapters/<run_name>/"
echo "Submissions: data/submission_<run_name>.csv per job (copy to Kaggle path manually if needed)"
echo ""
echo "Caption-backed runs (#3): precompute CSVs with  python -m pyfiles.captions  then resubmit with TRAIN_CAPTIONS_CSV=data/captions_train.csv INFER_CAPTIONS_CSV=data/captions_test.csv"
