#!/usr/bin/env bash
# Run test inference on 3 GPUs in parallel, then merge to data/submission.csv
# Usage: from DL_final/:  bash run_scoring_3gpu.sh
# Optional: LORA_ADAPTER=data/lora_adapter bash run_scoring_3gpu.sh
# Quick smoke test (10 examples, GPU 0 only, no merge):  MAX_ROWS=10 bash run_scoring_3gpu.sh

set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1

NUM_SHARDS="${NUM_SHARDS:-3}"
DATA_DIR="${DATA_DIR:-data}"
IMG_SIZE="${IMG_SIZE:-224}"
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolVLM-500M-Instruct}"
LORA_ARG=()
if [[ -n "${LORA_ADAPTER:-}" ]]; then
  LORA_ARG=(--lora-adapter "${LORA_ADAPTER}")
fi

MAX_ROWS="${MAX_ROWS:-}"
if [[ -n "${MAX_ROWS}" ]]; then
  echo "Quick run: first ${MAX_ROWS} rows of shard 0 only (writes submission_preview.csv, skips merge)."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python score_test_shard.py \
    --data-dir "${DATA_DIR}" \
    --model-id "${MODEL_ID}" \
    --img-size "${IMG_SIZE}" \
    --shard-index 0 \
    --num-shards 1 \
    --max-rows "${MAX_ROWS}" \
    -o "${DATA_DIR}/submission_preview.csv" \
    "${LORA_ARG[@]}"
  echo "Done: ${DATA_DIR}/submission_preview.csv (not a full Kaggle submission)"
  exit 0
fi

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python score_test_shard.py \
    --data-dir "${DATA_DIR}" \
    --model-id "${MODEL_ID}" \
    --img-size "${IMG_SIZE}" \
    --shard-index "$i" \
    --num-shards "${NUM_SHARDS}" \
    -o "${DATA_DIR}/submission_part${i}.csv" \
    "${LORA_ARG[@]}" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

PARTS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  PARTS+=("${DATA_DIR}/submission_part${i}.csv")
done

python merge_submission_shards.py --data-dir "${DATA_DIR}" --parts "${PARTS[@]}" -o "${DATA_DIR}/submission.csv"
echo "Done: ${DATA_DIR}/submission.csv"
