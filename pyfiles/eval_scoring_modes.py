from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from .common import MODEL_ID_DEFAULT, Paths, get_paths, load_csvs, load_multimodal_mcq_model
from .scoring import score_mcq_options


def _load_captions(captions_csv: Path | None) -> dict[str, str]:
    if captions_csv is None or not captions_csv.exists():
        return {}
    cap_df = pd.read_csv(captions_csv)
    return {str(r["id"]): str(r["caption"]) for _, r in cap_df.iterrows()}


def evaluate_scoring_modes(
    *,
    adapter_dir: Path | None,
    run_name: str,
    df: pd.DataFrame,
    paths: Paths,
    model_id: str = MODEL_ID_DEFAULT,
    n_eval: int = 200,
    img_size: int = 224,
    captions_csv: Path | None = None,
    shuffle_seed: int | None = None,
    modes: tuple[str, ...] = ("letter", "letter_choice", "choice_text"),
    choice_max_chars: int = 160,
    split_name: str = "val",
    scores_out_dir: Path | None = None,
) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_multimodal_mcq_model(model_id, adapter_dir, device_map="auto")
    model.eval()

    captions = _load_captions(captions_csv)
    use_caps = bool(captions)

    n = min(int(n_eval), len(df))
    if shuffle_seed is not None:
        subset = df.sample(n=n, random_state=int(shuffle_seed)).reset_index(drop=True)
        seed_frag = f"seed{int(shuffle_seed)}"
    else:
        subset = df.iloc[:n].reset_index(drop=True)
        seed_frag = "head"

    rows_long: list[dict[str, Any]] = []
    correct_by_mode = {m: 0 for m in modes}

    for i in tqdm(range(len(subset)), desc=f"{split_name} modes {run_name} {img_size}px"):
        row = subset.iloc[i]
        img = Image.open(paths.img_dir / row["image_path"]).convert("RGB").resize((img_size, img_size))
        cap = captions.get(str(row["id"])) if use_caps else None

        # Compute per-mode per-option scores
        scores_by_mode: dict[str, np.ndarray] = {}
        for m in modes:
            st = score_mcq_options(
                model,
                processor,
                row,
                img,
                caption=cap,
                completion_mode=m,
                choice_max_chars=int(choice_max_chars),
            )
            scores_by_mode[m] = st.detach().float().cpu().numpy()

        gold = int(row["answer"])
        for m in modes:
            pred = int(scores_by_mode[m].argmax())
            correct_by_mode[m] += int(pred == gold)

        num_choices = int(row["num_choices"])
        choices = row["choices"]
        if not isinstance(choices, (list, tuple)):
            choices = list(choices)

        for o in range(min(num_choices, len(choices))):
            ctext = str(choices[o])
            feat: dict[str, Any] = {
                "id": row["id"],
                "image_path": row["image_path"],
                "img_size": int(img_size),
                "option_idx": int(o),
                "num_choices": int(num_choices),
                "choice_len": int(len(ctext)),
                "gold_idx": int(gold),
                "is_gold": int(o == gold),
            }
            for m in modes:
                feat[f"score_{m}"] = float(scores_by_mode[m][o])
            rows_long.append(feat)

    out_base = scores_out_dir if scores_out_dir is not None else paths.data_dir
    out_base.mkdir(parents=True, exist_ok=True)
    out_long = out_base / f"{split_name}_scores_{run_name}_{img_size}px_{seed_frag}_n{len(subset)}.csv"
    pd.DataFrame(rows_long).to_csv(out_long, index=False)

    summary = {
        "run_name": run_name,
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "split": str(split_name),
        "n": int(len(subset)),
        "img_size": int(img_size),
        "scores_csv": str(out_long),
    }
    for m in modes:
        summary[f"acc_{m}"] = float(correct_by_mode[m] / len(subset)) if len(subset) else 0.0
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--adapter-dir", type=Path, default=None, help="Weights dir (PEFT adapter or full FT dir).")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--captions-csv", type=Path, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument(
        "--modes",
        type=str,
        default="letter,letter_choice,choice_text",
        help="Comma-separated: letter,letter_choice,choice_text",
    )
    ap.add_argument("--mcq-choice-max-chars", type=int, default=160)
    ap.add_argument(
        "--scores-out-dir",
        type=Path,
        default=None,
        help="Directory for long-form scores CSV (default: <root>/data).",
    )
    args = ap.parse_args()

    paths = get_paths(args.root)
    train_df, val_df, _ = load_csvs(paths.data_dir)
    df = train_df if str(args.split) == "train" else val_df

    modes = tuple(m.strip() for m in str(args.modes).split(",") if m.strip())
    meta = evaluate_scoring_modes(
        adapter_dir=args.adapter_dir,
        run_name=str(args.run_name),
        df=df,
        paths=paths,
        model_id=str(args.model_id),
        n_eval=int(args.n),
        img_size=int(args.img_size),
        captions_csv=args.captions_csv,
        shuffle_seed=args.shuffle_seed,
        modes=modes,  # type: ignore[arg-type]
        choice_max_chars=int(args.mcq_choice_max_chars),
        split_name=str(args.split),
        scores_out_dir=args.scores_out_dir,
    )
    print(pd.DataFrame([meta]).to_string(index=False))


if __name__ == "__main__":
    main()

