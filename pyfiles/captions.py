"""Generate one-line image captions using the SmolVLM base model and cache to CSV.

Usage:
    python -m pyfiles.captions --split val --out data/captions_val.csv
    python -m pyfiles.captions --split test --out data/captions_test.csv

The output CSV has columns: id, caption.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from .common import MODEL_ID_DEFAULT, get_paths, load_csvs


CAPTION_PROMPT = (
    "<image>\nDescribe this image in one short sentence focused on objects, "
    "labels, and any visible text. Be concise.\nDescription:"
)


@torch.inference_mode()
def caption_one(model, processor, img: Image.Image, *, max_new_tokens: int = 32) -> str:
    inputs = processor(text=CAPTION_PROMPT, images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    if "Description:" in text:
        text = text.split("Description:")[-1]
    text = " ".join(text.strip().split())
    if len(text) > 200:
        text = text[:199] + "…"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    ap.add_argument("--n", type=int, default=-1, help="Caption only first N rows (-1 = all).")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip rows already present in --out (default).")
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    args = ap.parse_args()

    paths = get_paths(args.root)
    train_df, val_df, test_df = load_csvs(paths.data_dir)
    df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    if args.n > 0:
        df = df.iloc[: args.n].reset_index(drop=True)

    done_ids: set[str] = set()
    if args.resume and args.out.exists():
        prev = pd.read_csv(args.out)
        done_ids = set(prev["id"].astype(str))
        print(f"Resuming: {len(done_ids)} captions already present in {args.out}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, device_map="auto")
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    with args.out.open("a") as f:
        if write_header:
            f.write("id,caption\n")
        for i in tqdm(range(len(df)), desc=f"captions {args.split}"):
            row = df.iloc[i]
            rid = str(row["id"])
            if rid in done_ids:
                continue
            try:
                img = Image.open(paths.img_dir / row["image_path"]).convert("RGB").resize(
                    (args.img_size, args.img_size)
                )
                cap = caption_one(model, processor, img)
            except Exception as e:
                cap = f"[error: {type(e).__name__}]"
            cap_safe = cap.replace('"', "'").replace("\n", " ")
            f.write(f'{rid},"{cap_safe}"\n')
            f.flush()

    print("Saved captions to", args.out.resolve())


if __name__ == "__main__":
    main()
