"""Val subset comparison with **one** model load per invocation.

Suites (matches competition checklist):

**A — Base model** (no adapter): baseline vs metadata × 224 vs 384.

**B — LoRA (best prompt from A)**: same prompt style at 224 and 384.

**C — Optional DoRA**: single run at chosen prompt + resolution (best from A/B).

Outputs ``data/val_compare_suite_<A|B|C>_summary.csv`` plus per-approach detail CSVs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from .common import MODEL_ID_DEFAULT, build_prompt, get_paths, load_csvs, set_seed, load_multimodal_mcq_model
from .experiments import build_prompt_with_metadata, predict_mcq_index_with_prompt_builder


PromptFn = Callable[..., str]
PromptStyle = Literal["baseline", "metadata"]
Suite = Literal["A", "B", "C"]


def _safe_output_tag_fragment(tag: str | None) -> str:
    if tag is None or str(tag).strip() == "":
        return ""
    safe_chars = []
    for ch in str(tag).strip():
        if ch.isalnum() or ch in "-_":
            safe_chars.append(ch)
        elif ch in " ./:+":
            safe_chars.append("_")
    frag = "".join(safe_chars).strip("_")
    return f"_{frag}" if frag else ""


@dataclass(frozen=True)
class Approach:
    name: str
    img_size: int
    prompt_fn: PromptFn


def _metadata_fields(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _prompt_fn(style: PromptStyle, meta_fields: tuple[str, ...]) -> PromptFn:
    if style == "baseline":
        return lambda row, include_answer=False: build_prompt(row, include_answer=include_answer)

    fields = meta_fields
    return lambda row, include_answer=False: build_prompt_with_metadata(
        row, include_answer=include_answer, fields=fields
    )


def approaches_for_suite(
    suite: Suite,
    *,
    prompt_style: PromptStyle | None,
    meta_fields: tuple[str, ...],
    img_size_c: int,
) -> list[Approach]:
    if suite == "A":
        pb = _prompt_fn("baseline", meta_fields)
        pm = _prompt_fn("metadata", meta_fields)
        return [
            Approach("A1_base_baseline_224", 224, pb),
            Approach("A2_base_metadata_224", 224, pm),
            Approach("A3_base_baseline_384", 384, pb),
            Approach("A4_base_metadata_384", 384, pm),
        ]

    if suite in ("B", "C"):
        assert prompt_style is not None
        pf = _prompt_fn(prompt_style, meta_fields)

    if suite == "B":
        tag = "lora"
        return [
            Approach(f"B5_{tag}_{prompt_style}_224", 224, pf),
            Approach(f"B6_{tag}_{prompt_style}_384", 384, pf),
        ]

    if suite == "C":
        tag = "dora"
        sz = int(img_size_c)
        return [Approach(f"C7_{tag}_{prompt_style}_{sz}", sz, pf)]

    raise ValueError(f"Unknown suite: {suite}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare val approaches (one model load per run).")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--suite", type=str, choices=("A", "B", "C"), required=True)
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="PEFT adapter dir. Required for suite B/C; must be omitted for suite A (base only).",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        choices=("baseline", "metadata"),
        default=None,
        help="Prompt style from suite A. Required for suite B and C.",
    )
    ap.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Square side length for suite C only (e.g. 224 or 384).",
    )
    ap.add_argument(
        "--metadata-fields",
        type=str,
        default="grade,subject,topic",
        help="Comma-separated columns for metadata prompt (suite A metadata arms and B/C when prompt=metadata).",
    )
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="If set, shuffle val with this seed then take first val_n rows (otherwise head(val_n)).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional suffix for summary/meta/detail filenames (avoid overwriting between runs).",
    )
    args = ap.parse_args()

    suite: Suite = args.suite  # type: ignore[assignment]
    meta_fields = _metadata_fields(args.metadata_fields)
    tag_frag = _safe_output_tag_fragment(args.output_tag)

    if suite == "A" and args.adapter_dir is not None:
        raise SystemExit("Suite A is base-model only: omit --adapter-dir.")

    if suite in ("B", "C"):
        if args.adapter_dir is None:
            raise SystemExit(f"Suite {suite} requires --adapter-dir.")
        if not Path(args.adapter_dir).is_dir():
            raise SystemExit(f"Adapter dir not found: {args.adapter_dir}")
        if args.prompt is None:
            raise SystemExit(f"Suite {suite} requires --prompt baseline|metadata (pick best from suite A).")

    if suite == "C":
        if args.img_size is None:
            raise SystemExit("Suite C requires --img-size (e.g. 224 or 384).")

    prompt_style: PromptStyle | None = args.prompt  # type: ignore[assignment]
    if suite == "A":
        prompt_style = None

    set_seed(args.seed)
    paths = get_paths(args.root)
    _, val_df, _ = load_csvs(paths.data_dir)

    n = min(args.val_n, len(val_df))
    if args.shuffle_seed is not None:
        val_part = val_df.sample(n=n, random_state=args.shuffle_seed).reset_index(drop=True)
    else:
        val_part = val_df.iloc[:n].reset_index(drop=True)

    approaches = approaches_for_suite(
        suite,
        prompt_style=prompt_style,
        meta_fields=meta_fields,
        img_size_c=args.img_size or 0,
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_multimodal_mcq_model(args.model_id, args.adapter_dir, device_map="auto")
    model.eval()

    print("suite:", suite, "| cuda:", torch.cuda.is_available(), "| approaches:", len(approaches))
    if torch.cuda.is_available():
        print("gpu0:", torch.cuda.get_device_name(0))

    summary_rows: list[dict] = []

    for approach in approaches:
        correct = 0
        safe_name = approach.name.replace("/", "_")
        out_detail = paths.data_dir / f"val_compare_suite{suite}{tag_frag}_{safe_name}_n{len(val_part)}.csv"
        detail_rows: list[dict] = []

        for i in tqdm(range(len(val_part)), desc=approach.name):
            row = val_part.iloc[i]
            img = (
                Image.open(paths.img_dir / row["image_path"])
                .convert("RGB")
                .resize((approach.img_size, approach.img_size))
            )
            pred = predict_mcq_index_with_prompt_builder(model, processor, row, img, approach.prompt_fn)
            gold = int(row["answer"])
            ok = int(pred == gold)
            correct += ok
            detail_rows.append(
                {
                    "id": row["id"],
                    "answer_true": gold,
                    "answer_pred": pred,
                    "correct": ok,
                    "num_choices": int(row["num_choices"]),
                }
            )

        acc = correct / len(val_part) if len(val_part) else 0.0
        pd.DataFrame(detail_rows).to_csv(out_detail, index=False)
        summary_rows.append(
            {
                "suite": suite,
                "approach": approach.name,
                "img_size": approach.img_size,
                "val_n": len(val_part),
                "correct": correct,
                "val_acc": round(acc, 6),
                "detail_csv": str(out_detail.resolve()),
            }
        )
        print(f"{approach.name:42s} acc={acc:.4f} ({correct}/{len(val_part)})")

    summary_df = pd.DataFrame(summary_rows).sort_values("val_acc", ascending=False).reset_index(drop=True)
    out_summary = paths.data_dir / f"val_compare_suite_{suite}{tag_frag}_summary.csv"
    summary_df.to_csv(out_summary, index=False)

    meta_path = paths.data_dir / f"val_compare_suite_{suite}{tag_frag}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "suite": suite,
                "output_tag": args.output_tag,
                "model_id": args.model_id,
                "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
                "prompt": args.prompt,
                "metadata_fields": list(meta_fields),
                "img_size_c": args.img_size,
                "val_n": len(val_part),
                "shuffle_seed": args.shuffle_seed,
                "summary_csv": str(out_summary.resolve()),
            },
            indent=2,
        )
    )

    print("\nRanking (by val_acc):\n", summary_df.to_string(index=False))
    print("\nWrote", out_summary.resolve())


if __name__ == "__main__":
    main()
