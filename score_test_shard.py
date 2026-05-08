#!/usr/bin/env python3
"""
Shard test-set MCQ scoring across multiple GPUs.

Run one process per GPU with a distinct --shard-index (0 .. num-shards-1).
Set CUDA_VISIBLE_DEVICES before launch so each process sees exactly one GPU as cuda:0.

Example (from repo root, after activating .venv):
  CUDA_VISIBLE_DEVICES=0 python score_test_shard.py --shard-index 0 --num-shards 3 -o data/submission_part0.csv
  CUDA_VISIBLE_DEVICES=1 python score_test_shard.py --shard-index 1 --num-shards 3 -o data/submission_part1.csv
  CUDA_VISIBLE_DEVICES=2 python score_test_shard.py --shard-index 2 --num-shards 3 -o data/submission_part2.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

CHOICE_LETTERS = "ABCDEFGHIJ"


def build_prompt(row: pd.Series, include_answer: bool = False) -> str:
    context_parts = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)

    choices = row["choices"]
    choices_str = "\n".join(f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))

    prompt = "<image>\n"
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    prompt += "Answer:"

    if include_answer:
        answer_idx = int(row["answer"])
        prompt += f" {CHOICE_LETTERS[answer_idx]}"

    return prompt


@torch.inference_mode()
def predict_mcq_index(row: pd.Series, image: Image.Image, model, processor) -> int:
    prompt = build_prompt(row, include_answer=False)

    n = int(row["num_choices"]) if "num_choices" in row else len(row["choices"])
    n = min(n, len(CHOICE_LETTERS))

    completions = [f" {CHOICE_LETTERS[i]}" for i in range(n)]
    full_texts = [prompt + c for c in completions]
    images = [image] * n

    # Prefix length MUST come from the multimodal processor (image + text), not text-only tokenization.
    # Otherwise slices into logits don't align with real completion tokens → degenerate scores (often always 0).
    inp_prompt = processor(text=prompt, images=image, return_tensors="pt")
    prompt_len = int(inp_prompt["input_ids"].shape[1])

    inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    if attn_mask is not None:
        token_logps = token_logps * attn_mask[:, 1:].to(token_logps.dtype)

    # Next-token positions predicting completion: [prompt_len-1 : full_len-1)
    start_pos = max(prompt_len - 1, 0)
    scores = []
    for i in range(n):
        inp_full = processor(text=full_texts[i], images=image, return_tensors="pt")
        full_len = int(inp_full["input_ids"].shape[1])
        if full_len <= prompt_len:
            scores.append(float("-inf"))
            continue
        end_pos = full_len - 1  # exclusive upper index into shift space
        slice_logps = token_logps[i, start_pos:end_pos]
        denom = max(slice_logps.numel(), 1)
        scores.append(slice_logps.sum().item() / denom)

    return int(np.argmax(scores))


def load_and_filter_test(data_dir: Path) -> pd.DataFrame:
    test_df = pd.read_csv(data_dir / "test.csv")
    test_df["choices"] = test_df["choices"].apply(json.loads)
    required = ["id", "image_path", "question", "choices", "num_choices"]
    test_df = test_df.dropna(subset=required).reset_index(drop=True)
    return test_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=3)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument(
        "--lora-adapter",
        type=Path,
        default=None,
        help="Optional PEFT adapter dir (e.g. data/lora_adapter) saved from fine-tuning.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fp32",
        action="store_true",
        help="Load model in float32 on GPU (slower, more VRAM). Use if predictions look degenerate (e.g. always class 0).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most this many rows from this shard (after sharding). Handy for smoke tests (e.g. 10).",
    )
    args = p.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must be in [0, num-shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_dir.resolve()
    img_dir = data_dir / "images"

    test_df = load_and_filter_test(data_dir)
    mask = np.arange(len(test_df)) % args.num_shards == args.shard_index
    shard_df = test_df.loc[mask].reset_index(drop=True)
    if args.max_rows is not None:
        shard_df = shard_df.head(args.max_rows).reset_index(drop=True)
        print(
            f"Shard {args.shard_index}/{args.num_shards}: {len(shard_df)} rows "
            f"(capped by --max-rows={args.max_rows}; full test has {len(test_df)} rows)"
        )
    else:
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(shard_df)} / {len(test_df)} rows")

    from transformers import AutoProcessor

    from pyfiles.common import is_peft_adapter_dir, load_multimodal_mcq_model

    processor = AutoProcessor.from_pretrained(args.model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dtype = torch.float32 if args.fp32 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    weights_dir = args.lora_adapter if args.lora_adapter is not None and args.lora_adapter.is_dir() else None
    devmap = "auto" if torch.cuda.is_available() else None
    model = load_multimodal_mcq_model(
        args.model_id,
        weights_dir,
        device_map=devmap,
        torch_dtype=dtype,
    )
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model.eval()
    if weights_dir is not None:
        tag = "PEFT adapter" if is_peft_adapter_dir(weights_dir) else "full fine-tuned checkpoint"
        print(f"Loaded {tag}:", weights_dir.resolve())

    pred_rows = []
    for i in tqdm(range(len(shard_df)), desc=f"gpu shard {args.shard_index}"):
        row = shard_df.iloc[i]
        img = Image.open(img_dir / row["image_path"]).convert("RGB").resize(
            (args.img_size, args.img_size), Image.BICUBIC
        )
        pred = predict_mcq_index(row, img, model, processor)
        pred_rows.append({"id": row["id"], "answer": int(pred)})

    out = pd.DataFrame(pred_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
