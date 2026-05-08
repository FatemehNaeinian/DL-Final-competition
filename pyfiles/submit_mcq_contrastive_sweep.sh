#!/usr/bin/env bash
# Planned sweep (rule-compliant: <= 5M trainable params total):
#   (1–2) MCQ contrastive LoRA @ FT 224, LR 5e-5 with last-16-layer LoRA recipes:
#         - attn_mlp_lora_r1_last16
#         - attn_mlp_lora_r2_last16
#   (3)   pick winner → eval at 224 and 384
#   (4)   if still weak: add choice-order shuffle augmentation (train only)
#
# Usage:
#   cd /scratch/fn2174/DL-Final
#   bash pyfiles/submit_mcq_contrastive_sweep.sh

set -euo pipefail

ROOT="/scratch/fn2174/DL-Final"
cd "$ROOT"

SBATCH_TRAIN="${ROOT}/pyfiles/sbatch_mc_contrastive_lora.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_106_tandon_advanced}"
ACC_ARGS=(--account="${ACCOUNT}")

common_env=(
  FT_IMG_SIZE=224
  LR=5e-5
  EPOCHS=1
  TRAIN_N=3109
  VAL_N=1048
  MAKE_SUBMISSION=0
  CFG="${CFG:-attn_mlp_lora_r1_last16}"
)

echo "Submitting MCQ contrastive jobs (requires sbatch SLURM)…"

JOB_R1=$(
  env "${common_env[@]}" \
    RUN_NAME=mcq224_cont_r1_last16_letter \
    MCQ_COMPLETION_MODE=letter \
    sbatch "${ACC_ARGS[@]}" --parsable "${SBATCH_TRAIN}"
)

JOB_R2=$(
  env "${common_env[@]}" \
    CFG=attn_mlp_lora_r2_last16 \
    RUN_NAME=mcq224_cont_r2_last16_letter \
    MCQ_COMPLETION_MODE=letter \
    sbatch "${ACC_ARGS[@]}" --parsable "${SBATCH_TRAIN}"
)

echo "Submitted: r1_last16 job ${JOB_R1}, r2_last16 job ${JOB_R2}"
echo "After training: compare run_meta.json val_eval_after_train.val_acc."
echo "Then evaluate best checkpoint at both resolutions:"
echo "  ADAPTER=data/adapters/<RUN_NAME> MCQ_COMPLETION_MODE=letter bash -c '\\"
echo "    sbatch --dependency=afterok:${JOB_R1} pyfiles/sbatch_eval_adapter_img_sizes.sbatch'"
echo "(Use the MCQ_COMPLETION_MODE you trained with; RUNTAG arbitrary label.)"
