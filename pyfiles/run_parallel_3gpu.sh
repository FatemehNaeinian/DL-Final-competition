#!/usr/bin/env bash
# Run the 6-experiment orchestration across 3 GPUs (train lane + eval lane + finalize).
#
# Layout (VRAM-friendly defaults):
#   GPU 0: train text_only + vision_only
#   GPU 1: train mm_attn_only + mm_attn_mlp
#   GPU 2: train mm_attn_mlp_dora
#
# Eval shards (caption_base ONLY on GPU 0 to avoid racing captions_val.csv writes):
#   GPU 0: eval text_only, vision_only, caption_base
#   GPU 1: eval mm_attn_only, mm_attn_mlp
#   GPU 2: eval mm_attn_mlp_dora
#
# Usage:
#   bash /home/fn2174/DL_final/pyfiles/run_parallel_3gpu.sh
#
# Override knobs:
#   TRAIN_N=2000 VAL_N=500 EPOCHS=1 bash pyfiles/run_parallel_3gpu.sh
#
set -euo pipefail

ROOT="${ROOT:-/home/fn2174/DL_final}"
cd "$ROOT"
mkdir -p data/logs data/shards
source "${ROOT}/.venv/bin/activate"

TRAIN_N="${TRAIN_N:-1500}"
VAL_N="${VAL_N:-500}"
EPOCHS="${EPOCHS:-1}"
FT_IMG_SIZE="${FT_IMG_SIZE:-192}"
IMG_SIZE="${IMG_SIZE:-224}"
EXTRA_ORCH_ARGS="${EXTRA_ORCH_ARGS:-}"  # appended to each python invocation

COMMON_ARGS=(
  "--train-n" "${TRAIN_N}"
  "--val-n" "${VAL_N}"
  "--epochs" "${EPOCHS}"
  "--ft-img-size" "${FT_IMG_SIZE}"
  "--img-size" "${IMG_SIZE}"
  "--skip-train-existing"
  ${EXTRA_ORCH_ARGS}
)

echo "========================================== PHASE: TRAIN (3 parallel jobs)"
(
  CUDA_VISIBLE_DEVICES=0 python -m pyfiles.orchestrate --phase train \
    "${COMMON_ARGS[@]}" --only \
    text_only_attn_mlp vision_only_lora \
    >data/logs/train_gpu0.out 2>&1 && echo "[OK] train_gpu0" || echo "[FAIL] train_gpu0" >&2
) &
PID0=$!

(
  CUDA_VISIBLE_DEVICES=1 python -m pyfiles.orchestrate --phase train \
    "${COMMON_ARGS[@]}" --only \
    mm_attn_only mm_attn_mlp \
    >data/logs/train_gpu1.out 2>&1 && echo "[OK] train_gpu1" || echo "[FAIL] train_gpu1" >&2
) &
PID1=$!

(
  CUDA_VISIBLE_DEVICES=2 python -m pyfiles.orchestrate --phase train \
    "${COMMON_ARGS[@]}" --only \
    mm_attn_mlp_dora \
    >data/logs/train_gpu2.out 2>&1 && echo "[OK] train_gpu2" || echo "[FAIL] train_gpu2" >&2
) &
PID2=$!

wait "${PID0}" "${PID1}" "${PID2}" || true

echo "========================================== PHASE: EVAL (3 parallel shards)"
(
  CUDA_VISIBLE_DEVICES=0 python -m pyfiles.orchestrate --phase eval --shard-suffix eval_gpu0 \
    "${COMMON_ARGS[@]}" --only \
    text_only_attn_mlp vision_only_lora caption_base \
    >data/logs/eval_gpu0.out 2>&1 && echo "[OK] eval_gpu0" || echo "[FAIL] eval_gpu0" >&2
) &
EP0=$!

(
  CUDA_VISIBLE_DEVICES=1 python -m pyfiles.orchestrate --phase eval --shard-suffix eval_gpu1 \
    "${COMMON_ARGS[@]}" --only \
    mm_attn_only mm_attn_mlp \
    >data/logs/eval_gpu1.out 2>&1 && echo "[OK] eval_gpu1" || echo "[FAIL] eval_gpu1" >&2
) &
EP1=$!

(
  CUDA_VISIBLE_DEVICES=2 python -m pyfiles.orchestrate --phase eval --shard-suffix eval_gpu2 \
    "${COMMON_ARGS[@]}" --only \
    mm_attn_mlp_dora \
    >data/logs/eval_gpu2.out 2>&1 && echo "[OK] eval_gpu2" || echo "[FAIL] eval_gpu2" >&2
) &
EP2=$!

wait "${EP0}" "${EP1}" "${EP2}" || true

echo "========================================== PHASE: FINALIZE (single GPU)"

# IMPORTANT: shard glob must exclude train-only stub CSVs produced by obsolete runs.
CUDA_VISIBLE_DEVICES=0 python -m pyfiles.orchestrate --phase finalize \
  --merge-from "data/shards/experiments_summary_eval_gpu*.csv" \
  --val-n "${VAL_N}" --img-size "${IMG_SIZE}" \
  ${EXTRA_ORCH_ARGS}

echo ""
echo "Done."
echo "  Merged leaderboard: ${ROOT}/data/experiments_summary.csv"
echo "  Winner meta:       ${ROOT}/data/winner.json"
echo "  Submission:        ${ROOT}/submission.csv"
echo "  Per-GPU logs:      ${ROOT}/data/logs/train_gpu*.out  ${ROOT}/data/logs/eval_gpu*.out"
