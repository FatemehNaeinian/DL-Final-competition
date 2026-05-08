from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from .common import MODEL_ID_DEFAULT, get_paths, load_csvs, load_multimodal_mcq_model
from .scoring import predict_mcq_index


def make_submission_from_adapter(
    run_name: str,
    adapter_dir: Path | None,
    test_df: pd.DataFrame,
    paths: "Paths",
    model_id: str = MODEL_ID_DEFAULT,
    img_size: int = 224,
    out: Path | None = None,
    captions_csv: Path | None = None,
    mcq_completion_mode: str = "letter",
    mcq_choice_max_chars: int = 160,
) -> Path:
    """Generate a submission CSV from a trained adapter (or the base model if `adapter_dir` is None).

    If `captions_csv` is given, each row's caption is injected into the prompt.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_multimodal_mcq_model(model_id, adapter_dir, device_map="auto")
    model.eval()

    captions: dict[str, str] = {}
    if captions_csv is not None and Path(captions_csv).exists():
        cap_df = pd.read_csv(captions_csv)
        captions = {str(r["id"]): str(r["caption"]) for _, r in cap_df.iterrows()}
    use_caps = bool(captions)

    final_out_path = out if out is not None else paths.data_dir / f"submission_{run_name}.csv"

    pred_rows = []
    for i in tqdm(range(len(test_df)), desc=f"submit: {run_name}"):
        row = test_df.iloc[i]
        img = Image.open(paths.img_dir / row["image_path"]).convert("RGB").resize((img_size, img_size))
        cap = captions.get(str(row["id"])) if use_caps else None
        pred_idx = predict_mcq_index(
            model,
            processor,
            row,
            img,
            caption=cap,
            completion_mode=mcq_completion_mode,
            choice_max_chars=mcq_choice_max_chars,
        )
        pred_rows.append({"id": row["id"], "answer": int(pred_idx)})

    sub_df = pd.DataFrame(pred_rows)
    sub_df.to_csv(final_out_path, index=False)
    return final_out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--adapter-dir", type=Path, default=None,
                    help="Adapter dir; omit to use the bare base model.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--captions-csv", type=Path, default=None,
                    help="Optional captions CSV (id,caption) to augment prompts.")
    ap.add_argument(
        "--mcq-completion-mode",
        type=str,
        default="letter",
        help="Scoring completion: letter | letter_choice | choice_text (must match your eval choice).",
    )
    ap.add_argument(
        "--mcq-choice-max-chars",
        type=int,
        default=160,
        help="Truncation for letter_choice / choice_text scoring.",
    )
    args = ap.parse_args()

    paths = get_paths(args.root)
    _, _, test_df = load_csvs(paths.data_dir)

    submission_path = make_submission_from_adapter(
        run_name=args.run_name,
        adapter_dir=args.adapter_dir,
        test_df=test_df,
        paths=paths,
        model_id=args.model_id,
        img_size=args.img_size,
        out=args.out,
        captions_csv=args.captions_csv,
        mcq_completion_mode=str(args.mcq_completion_mode),
        mcq_choice_max_chars=int(args.mcq_choice_max_chars),
    )
    print("Wrote", len(test_df), "rows to", submission_path.resolve())


if __name__ == "__main__":
    main()
