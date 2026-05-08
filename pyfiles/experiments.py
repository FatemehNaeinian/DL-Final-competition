from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from .common import CHOICE_LETTERS, MODEL_ID_DEFAULT, build_prompt, get_paths, load_csvs, load_multimodal_mcq_model


def build_prompt_variant(
    row: pd.Series,
    *,
    include_answer: bool = False,
    answer_prefix: str = "Answer:",
    context_before_question: bool = True,
) -> str:
    """Section 13: change final cue and context/question order."""
    context_parts = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)

    choices = row["choices"]
    n = int(row["num_choices"]) if "num_choices" in row else len(choices)
    n = min(n, len(CHOICE_LETTERS))
    choices_str = "\n".join(f"  {CHOICE_LETTERS[i]}. {choices[i]}" for i in range(n))

    question_block = f"Question: {row['question']}\nChoices:\n{choices_str}\n"

    prompt = "<image>\n"
    if context_before_question:
        if context_str:
            prompt += f"Context:\n{context_str}\n\n"
        prompt += question_block
    else:
        prompt += question_block
        if context_str:
            prompt += f"\nContext:\n{context_str}\n"

    prompt += answer_prefix.rstrip()
    if include_answer:
        prompt += f" {CHOICE_LETTERS[int(row['answer'])]}"
    return prompt


def build_prompt_with_metadata(
    row: pd.Series,
    *,
    include_answer: bool = False,
    fields: tuple[str, ...] = ("grade", "subject", "topic"),
) -> str:
    """Section 14: add metadata header."""
    meta_lines = []
    for f in fields:
        if f in row.index and pd.notna(row[f]) and str(row[f]).strip():
            meta_lines.append(f"{f.replace('_', ' ').title()}: {row[f]}")
    meta_str = "\n".join(meta_lines)

    # reuse baseline prompt pieces
    base = build_prompt(row, include_answer=False)
    if not meta_str:
        out = base
    else:
        # Insert metadata after <image>\n
        assert base.startswith("<image>\n")
        rest = base[len("<image>\n") :]
        out = f"<image>\nMetadata:\n{meta_str}\n\n{rest}"

    if include_answer:
        out += f" {CHOICE_LETTERS[int(row['answer'])]}"
    return out


@torch.inference_mode()
def predict_mcq_index_with_prompt_builder(model, processor, row: pd.Series, image: Image.Image, prompt_builder) -> int:
    """Same scorer as notebook Section 13, but in CLI form."""
    prompt = prompt_builder(row, include_answer=False)

    n = int(row["num_choices"]) if "num_choices" in row else len(row["choices"])
    n = min(n, len(CHOICE_LETTERS))

    completions = [f" {CHOICE_LETTERS[i]}" for i in range(n)]
    full_texts = [prompt + c for c in completions]
    images = [image] * n

    inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    inp_prompt = processor(text=prompt, images=image, return_tensors="pt")
    prompt_len = int(inp_prompt["input_ids"].shape[1])

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

    start_pos = max(prompt_len - 1, 0)
    scores = []
    for i in range(n):
        inp_full = processor(text=full_texts[i], images=image, return_tensors="pt")
        full_len = int(inp_full["input_ids"].shape[1])
        if full_len <= prompt_len:
            scores.append(float("-inf"))
            continue
        end_pos = full_len - 1
        slice_logps = token_logps[i, start_pos:end_pos]
        denom = slice_logps.numel() if slice_logps.numel() > 0 else 1
        scores.append((slice_logps.sum().item()) / denom)

    return int(np.argmax(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--adapter-dir", type=Path, default=None, help="Optional PEFT adapter to load.")
    ap.add_argument("--run-name", type=str, required=True)

    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--make-submission", action="store_true", default=False)

    # Section 13 knobs
    ap.add_argument("--answer-prefix", type=str, default="Answer:")
    ap.add_argument("--context-before-question", action="store_true", default=True)
    ap.add_argument("--question-before-context", action="store_false", dest="context_before_question")

    # Section 14 knobs
    ap.add_argument("--use-metadata", action="store_true", default=False)
    ap.add_argument("--metadata-fields", type=str, default="grade,subject,topic")

    args = ap.parse_args()

    paths = get_paths(args.root)
    train_df, val_df, test_df = load_csvs(paths.data_dir)

    processor = AutoProcessor.from_pretrained(args.model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_multimodal_mcq_model(args.model_id, args.adapter_dir, device_map="auto")
    model.eval()

    fields = tuple([f.strip() for f in args.metadata_fields.split(",") if f.strip()])

    def prompt_builder(row: pd.Series, include_answer: bool = False) -> str:
        if args.use_metadata:
            return build_prompt_with_metadata(row, include_answer=include_answer, fields=fields)
        return build_prompt_variant(
            row,
            include_answer=include_answer,
            answer_prefix=args.answer_prefix,
            context_before_question=args.context_before_question,
        )

    subset = val_df.iloc[: min(args.val_n, len(val_df))].reset_index(drop=True)
    rows = []
    correct = 0
    for i in tqdm(range(len(subset)), desc=f"val {args.run_name}"):
        row = subset.iloc[i]
        img = Image.open(paths.img_dir / row["image_path"]).convert("RGB").resize((args.img_size, args.img_size))
        pred = predict_mcq_index_with_prompt_builder(model, processor, row, img, prompt_builder)
        gold = int(row["answer"])
        ok = int(pred == gold)
        correct += ok
        rows.append(
            {
                "id": row["id"],
                "answer_true": gold,
                "answer_pred": int(pred),
                "correct": ok,
                "num_choices": int(row["num_choices"]),
                "image_path": row["image_path"],
            }
        )

    acc = correct / len(subset) if len(subset) else 0.0
    out_val = paths.data_dir / f"val_eval_{args.run_name}_n{len(subset)}.csv"
    pd.DataFrame(rows).to_csv(out_val, index=False)
    print("Val acc:", f"{acc:.4f}", "saved:", out_val.resolve())

    meta = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir is not None else None,
        "val_n": int(len(subset)),
        "val_acc": float(acc),
        "img_size": int(args.img_size),
        "prompt": {
            "answer_prefix": args.answer_prefix,
            "context_before_question": bool(args.context_before_question),
            "use_metadata": bool(args.use_metadata),
            "metadata_fields": list(fields),
        },
    }
    meta_path = paths.data_dir / "runs" / args.run_name / "experiment_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Saved meta to", meta_path.resolve())

    if args.make_submission:
        pred_rows = []
        for i in tqdm(range(len(test_df)), desc=f"submit {args.run_name}"):
            row = test_df.iloc[i]
            img = Image.open(paths.img_dir / row["image_path"]).convert("RGB").resize((args.img_size, args.img_size))
            pred = predict_mcq_index_with_prompt_builder(model, processor, row, img, prompt_builder)
            pred_rows.append({"id": row["id"], "answer": int(pred)})
        sub_df = pd.DataFrame(pred_rows)
        out_sub = paths.data_dir / f"submission_{args.run_name}.csv"
        sub_df.to_csv(out_sub, index=False)
        print("Saved submission to", out_sub.resolve())


if __name__ == "__main__":
    main()

