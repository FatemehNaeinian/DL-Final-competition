#!/usr/bin/env bash
# Submit the DL-Final fine-tuning pipeline jobs with sensible dependency chaining:
#   1) LoRA train (full train/val, no submission)
#   2) Suite-B dual eval (baseline vs metadata) after train completes
#   3) Submission generation after eval completes
#
# Optional second training job (DoRA):
#   SUBMIT_OPTIONAL_DORA=1 bash pyfiles/submit_plan_jobs.sh
#
# Optional Slurm partition:
#   PARTITION=h200_tand bash pyfiles/submit_plan_jobs.sh
#
# Slurm account (Torch):
#   ACCOUNT=torch_pr_xxx_yyy bash pyfiles/submit_plan_jobs.sh

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"

PART_ARGS=()
if [[ -n "${PARTITION:-}" ]]; then
  PART_ARGS+=(--partition="${PARTITION}")
fi

ACCOUNT_ARGS=()
if [[ -n "${ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS+=(--account="${ACCOUNT}")
fi

LORA_RUN_NAME="${LORA_RUN_NAME:-lora_attn_mlp_plan}"
LORA_ADAPTER_OUT="${LORA_ADAPTER_OUT:-data/adapters/${LORA_RUN_NAME}}"

TRAIN_JOB="$(
  RUN_NAME="${LORA_RUN_NAME}" \
    ADAPTER_OUT="${LORA_ADAPTER_OUT}" \
    CFG="${CFG:-attn_mlp_lora}" \
    TRAIN_N="${TRAIN_N:-3109}" \
    VAL_N="${VAL_N:-1048}" \
    EPOCHS="${EPOCHS:-1}" \
    MAKE_SUBMISSION=0 \
    RESUME="${RESUME:-none}" \
    sbatch "${PART_ARGS[@]}" "${ACCOUNT_ARGS[@]}" --parsable pyfiles/sbatch_train_lora.sbatch
)"
echo "LoRA train job: ${TRAIN_JOB}"

if [[ "${SUBMIT_OPTIONAL_DORA:-0}" == "1" ]]; then
  DORA_RUN_NAME="${DORA_RUN_NAME:-dora_attn_mlp_plan}"
  DORA_ADAPTER_OUT="${DORA_ADAPTER_OUT:-data/adapters/${DORA_RUN_NAME}}"
  DORA_JOB="$(
    RUN_NAME="${DORA_RUN_NAME}" \
      ADAPTER_OUT="${DORA_ADAPTER_OUT}" \
      CFG="${DORA_CFG:-attn_mlp_dora}" \
      TRAIN_N="${TRAIN_N:-3109}" \
      VAL_N="${VAL_N:-1048}" \
      EPOCHS="${EPOCHS:-1}" \
      MAKE_SUBMISSION=0 \
      RESUME="${RESUME:-none}" \
      sbatch "${PART_ARGS[@]}" "${ACCOUNT_ARGS[@]}" --parsable pyfiles/sbatch_train_lora.sbatch
  )"
  echo "Optional DoRA train job: ${DORA_JOB}"
fi

EVAL_JOB="$(
  ADAPTER_DIR="${LORA_ADAPTER_OUT}" \
    VAL_N="${EVAL_VAL_N:-500}" \
    SHUFFLE_SEED="${EVAL_SHUFFLE_SEED:-42}" \
    sbatch "${PART_ARGS[@]}" "${ACCOUNT_ARGS[@]}" --parsable --dependency=afterok:"${TRAIN_JOB}" pyfiles/sbatch_eval_suite_B_dual.sbatch
)"
echo "Suite-B eval job (after LoRA train): ${EVAL_JOB}"

SUB_JOB="$(
  ADAPTER_DIR="${LORA_ADAPTER_OUT}" \
    IMG_SIZE="${SUB_IMG_SIZE:-224}" \
    SUB_RUN_NAME="${SUB_RUN_NAME:-plan_kaggle}" \
    sbatch "${PART_ARGS[@]}" "${ACCOUNT_ARGS[@]}" --parsable --dependency=afterok:"${EVAL_JOB}" pyfiles/sbatch_make_submission.sbatch
)"
echo "Submission job (after suite-B eval): ${SUB_JOB}"
