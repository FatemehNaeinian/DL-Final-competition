from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from .common import MODEL_ID_DEFAULT, MCQ_COMPLETION_MODES, Paths, get_paths, load_csvs, load_multimodal_mcq_model
from .scoring import predict_mcq_index


def _load_captions(captions_csv: Path | None) -> dict[str, str]:
    if captions_csv is None or not captions_csv.exists():
        return {}
    cap_df = pd.read_csv(captions_csv)
    return {str(r["id"]): str(r["caption"]) for _, r in cap_df.iterrows()}


def evaluate_adapter_dir(
    adapter_dir: Path | None,
    *,
    run_name: str,
    val_df: pd.DataFrame,
    paths: Paths,
    model_id: str = MODEL_ID_DEFAULT,
    val_n: int = 200,
    img_size: int = 224,
    captions_csv: Path | None = None,
    shuffle_seed: int | None = None,
    mcq_completion_mode: str = "letter",
    mcq_choice_max_chars: int = 160,
) -> dict[str, Any]:
    """Load `adapter_dir` (or just the base model if None) and evaluate on `val_df` slice.

    Always uses multimodal scoring (image + text). If `captions_csv` is given,
    each row's caption is injected into the prompt.
    """
    if adapter_dir is not None and not adapter_dir.exists():
        return {"run_name": run_name, "status": "missing", "adapter_dir": str(adapter_dir)}

    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_multimodal_mcq_model(model_id, adapter_dir, device_map="auto")
    model.eval()

    captions = _load_captions(captions_csv)
    use_caps = bool(captions)

    n = min(val_n, len(val_df))
    if shuffle_seed is not None:
        subset = val_df.sample(n=n, random_state=shuffle_seed).reset_index(drop=True)
    else:
        subset = val_df.iloc[:n].reset_index(drop=True)
    rows = []
    correct = 0
    for i in tqdm(range(len(subset)), desc=f"val {run_name}"):
        row = subset.iloc[i]
        img = Image.open(paths.img_dir / row["image_path"]).convert("RGB").resize((img_size, img_size))
        cap = captions.get(str(row["id"])) if use_caps else None
        pred = predict_mcq_index(
            model,
            processor,
            row,
            img,
            caption=cap,
            completion_mode=mcq_completion_mode,
            choice_max_chars=mcq_choice_max_chars,
        )
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
    seed_frag = f"_seed{shuffle_seed}" if shuffle_seed is not None else "_head"
    out_csv = paths.data_dir / f"val_eval_{run_name}_n{len(subset)}{seed_frag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    return {
        "run_name": run_name,
        "status": "ok",
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "val_n": int(len(subset)),
        "val_acc": float(acc),
        "val_csv": str(out_csv),
        "captions_csv": str(captions_csv) if captions_csv is not None else None,
    }


def parse_run_specs(items: list[str]) -> list[tuple[str, Path | None]]:
    out: list[tuple[str, Path | None]] = []
    for it in items:
        if ":" not in it:
            raise ValueError(f"Bad run spec '{it}', expected name:path  (use name:- for base model)")
        name, path = it.split(":", 1)
        out.append((name, None if path == "-" else Path(path)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run specs: <name>:<adapter_dir> (use 'name:-' for the bare base model).",
    )
    ap.add_argument(
        "--captions-csv",
        type=Path,
        default=None,
        help="Optional captions CSV (id,caption) to augment prompts.",
    )
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="If set, sample val with this seed (same subset for every run in this invocation).",
    )
    ap.add_argument(
        "--mcq-completion-mode",
        type=str,
        default="letter",
        choices=sorted(MCQ_COMPLETION_MODES),
        help="Must match contrastive training (letter vs letter_choice).",
    )
    ap.add_argument(
        "--mcq-choice-max-chars",
        type=int,
        default=160,
        help="Truncation for letter_choice / choice_text (match training).",
    )
    args = ap.parse_args()

    paths = get_paths(args.root)
    _, val_df, _ = load_csvs(paths.data_dir)

    summary_rows = []
    for run_name, adapter_dir in parse_run_specs(args.runs):
        result = evaluate_adapter_dir(
            adapter_dir=adapter_dir,
            run_name=run_name,
            val_df=val_df,
            paths=paths,
            model_id=args.model_id,
            val_n=args.val_n,
            img_size=args.img_size,
            captions_csv=args.captions_csv,
            shuffle_seed=args.shuffle_seed,
            mcq_completion_mode=str(args.mcq_completion_mode),
            mcq_choice_max_chars=int(args.mcq_choice_max_chars),
        )
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    suff = f"_n{args.val_n}_seed{args.shuffle_seed}" if args.shuffle_seed is not None else f"_n{args.val_n}_head"
    out_summary = paths.data_dir / f"val_summary_adapter_compare{suff}.csv"
    summary_df.to_csv(out_summary, index=False)
    print("Saved summary to", out_summary.resolve())
    if "val_acc" in summary_df.columns:
        print(summary_df.sort_values(["status", "val_acc"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
