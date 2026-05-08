"""End-to-end orchestrator for the 6-experiment comparison.

Trains 5 LoRA variants on `train.csv`, evaluates each on `val.csv`,
also evaluates the base model with self-generated image captions,
then writes `submission.csv` using the variant with the highest val accuracy.

Single-GPU (default):
    python -m pyfiles.orchestrate --train-n 1500 --val-n 500 --epochs 1 \\
        --skip-train-existing --make-final-submission

3-GPU parallel (wrapper script):
    bash pyfiles/run_parallel_3gpu.sh

Manual multi-GPU shards:
    CUDA_VISIBLE_DEVICES=0 python -m pyfiles.orchestrate --phase train --only text_only_attn_mlp ...
    CUDA_VISIBLE_DEVICES=0 python -m pyfiles.orchestrate --phase eval --shard-suffix eval_gpu0 --only ...
    python -m pyfiles.orchestrate --phase finalize --merge-from \"data/shards/experiments_summary_eval_gpu*.csv\"

Select experiments:
    python -m pyfiles.orchestrate --only mm_attn_mlp_dora caption_base
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from .common import MODEL_ID_DEFAULT, get_paths, load_csvs, save_json, set_seed


@dataclass
class Experiment:
    name: str
    description: str
    cfg: Optional[str] = None        # LoRA recipe key; None => no training
    text_only: bool = False
    vision_only: bool = False
    use_captions: bool = False       # inject captions at inference
    train: bool = True               # whether to fine-tune


DEFAULT_EXPERIMENTS: list[Experiment] = [
    Experiment(
        name="text_only_attn_mlp",
        description="(1) LoRA fine-tune with TEXT only (q/v + MLP, last 6 text layers).",
        cfg="attn_mlp_lora",
        text_only=True,
    ),
    Experiment(
        name="vision_only_lora",
        description="(2) LoRA fine-tune with IMAGES only (vision tower projections).",
        cfg="attn_mlp_lora",  # cfg arg required but vision_only overrides target modules.
        vision_only=True,
    ),
    Experiment(
        name="mm_attn_only",
        description="(3) Multimodal LoRA on attention only (q/v, last 6 text layers).",
        cfg="attn_only_lora",
    ),
    Experiment(
        name="mm_attn_mlp",
        description="(4) Multimodal LoRA expanded to MLP (q/v + gate/up/down).",
        cfg="attn_mlp_lora",
    ),
    Experiment(
        name="mm_attn_mlp_dora",
        description="(5) Multimodal DoRA on attention + MLP.",
        cfg="attn_mlp_dora",
    ),
    Experiment(
        name="caption_base",
        description="(6) Base model + self-generated image captions in the prompt.",
        cfg=None,
        use_captions=True,
        train=False,
    ),
]


def _adapter_dir(paths, name: str) -> Path:
    return paths.data_dir / "runs" / name / "adapter"


def _captions_path(paths, split: str) -> Path:
    return paths.data_dir / f"captions_{split}.csv"


def _ensure_captions(paths, split: str, model_id: str, n: int = -1, img_size: int = 224) -> Path:
    """Generate captions for the given split if not already present."""
    out_csv = _captions_path(paths, split)
    if out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            train_df, val_df, test_df = load_csvs(paths.data_dir)
            df = {"train": train_df, "val": val_df, "test": test_df}[split]
            target = len(df) if n < 0 else min(n, len(df))
            if len(existing) >= target:
                print(f"[captions] {split}: {len(existing)} cached (>= target {target}); skipping.")
                return out_csv
            print(f"[captions] {split}: {len(existing)} cached, need {target}; resuming.")
        except Exception:
            pass

    print(f"[captions] generating for split={split} ...")
    cmd = [
        sys.executable, "-m", "pyfiles.captions",
        "--root", str(paths.root),
        "--model-id", model_id,
        "--split", split,
        "--out", str(out_csv),
        "--img-size", str(img_size),
    ]
    if n > 0:
        cmd += ["--n", str(n)]
    import subprocess
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"captions subprocess failed (rc={rc})")
    return out_csv


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_one(exp: Experiment, *, paths, train_df, val_df, test_df, args) -> Path | None:
    """Run training for a single experiment. Returns the adapter dir (or None if skipped)."""
    if not exp.train:
        return None

    adapter_out = _adapter_dir(paths, exp.name)
    if args.skip_train_existing and (adapter_out / "adapter_config.json").exists():
        print(f"[train] {exp.name}: adapter already exists, skipping.")
        return adapter_out

    from .train_adapter import train_lora_adapter

    print(f"\n=========================================================")
    print(f"[train] {exp.name}: {exp.description}")
    print(f"=========================================================")

    train_lora_adapter(
        lora_cfg_name=exp.cfg or "attn_mlp_lora",
        adapter_out_dir=adapter_out,
        run_name=exp.name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        paths=paths,
        seed=args.seed,
        model_id=args.model_id,
        train_n=args.train_n,
        val_n=args.val_n,
        num_train_epochs=args.epochs,
        ft_img_size=args.ft_img_size,
        text_only=exp.text_only,
        vision_only=exp.vision_only,
        make_submission=False,                   # never per-experiment; we pick a winner.
        run_validation_after_train=False,        # we evaluate centrally below.
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        no_eval=True,                            # save VRAM during training
        num_workers=args.num_workers,
        pin_memory=True,
        resume="auto",
    )
    _free_gpu()
    return adapter_out


def _eval_one(exp: Experiment, *, paths, val_df, args) -> dict:
    """Evaluate one experiment on val.csv. Returns the summary dict."""
    from .eval_adapters import evaluate_adapter_dir

    adapter_dir = _adapter_dir(paths, exp.name) if exp.train else None

    captions_csv: Path | None = None
    if exp.use_captions:
        captions_csv = _ensure_captions(paths, "val", args.model_id, n=args.val_n)

    print(f"\n[eval] {exp.name} ...")
    result = evaluate_adapter_dir(
        adapter_dir=adapter_dir,
        run_name=exp.name,
        val_df=val_df,
        paths=paths,
        model_id=args.model_id,
        val_n=args.val_n,
        img_size=args.img_size,
        captions_csv=captions_csv,
    )
    result["description"] = exp.description
    _free_gpu()
    return result


def _filter_experiments(only: list[str] | None) -> list[Experiment]:
    experiments = DEFAULT_EXPERIMENTS
    if not only:
        return experiments
    names = set(only)
    experiments = [e for e in experiments if e.name in names]
    if not experiments:
        raise SystemExit(
            f"--only filter matched no experiments. Available: {[e.name for e in DEFAULT_EXPERIMENTS]}"
        )
    return experiments


def _run_train_phase(experiments: list[Experiment], *, paths, train_df, val_df, test_df, args) -> list[str]:
    failures: list[str] = []
    for exp in experiments:
        if not exp.train:
            continue
        try:
            _train_one(exp, paths=paths, train_df=train_df, val_df=val_df, test_df=test_df, args=args)
        except Exception as e:
            print(f"[train] {exp.name}: FAILED -> {repr(e)}")
            traceback.print_exc()
            failures.append(exp.name)
    return failures


def _run_eval_phase(
    experiments: list[Experiment],
    *,
    paths,
    val_df,
    args,
    train_failures: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for exp in experiments:
        if exp.train and exp.name in train_failures:
            rows.append(
                {
                    "run_name": exp.name,
                    "status": "train_failed",
                    "description": exp.description,
                    "val_acc": float("nan"),
                }
            )
            continue
        try:
            r = _eval_one(exp, paths=paths, val_df=val_df, args=args)
            rows.append(r)
        except Exception as e:
            print(f"[eval] {exp.name}: FAILED -> {repr(e)}")
            traceback.print_exc()
            rows.append(
                {
                    "run_name": exp.name,
                    "status": "eval_failed",
                    "description": exp.description,
                    "val_acc": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _resolve_glob_patterns(root: Path, patterns: list[str]) -> list[str]:
    """Expand glob patterns; relative paths resolve under `root` (fallback: cwd patterns)."""
    seen: set[str] = set()
    out_paths: list[str] = []
    for pat in patterns:
        cand_list: list[str] = []
        p = Path(pat)
        if p.is_absolute():
            cand_list.extend(glob(pat))
        else:
            cand_list.extend(glob(str(root / pat)))
            cand_list.extend(glob(pat))
        for c in cand_list:
            rp = str(Path(c).resolve())
            if rp not in seen:
                seen.add(rp)
                out_paths.append(rp)
    return out_paths


def _merge_summary_shards(root: Path, patterns: list[str]) -> pd.DataFrame:
    """Load one or more glob patterns (relative to --root ok), concat, dedupe run_name."""
    paths_found = _resolve_glob_patterns(root, patterns)
    if not paths_found:
        raise SystemExit(f"No summary files matched: {patterns} (cwd={root})")
    frames = [pd.read_csv(p) for p in paths_found]
    out = pd.concat(frames, ignore_index=True)
    if "run_name" not in out.columns:
        raise SystemExit(
            "Merged summaries must contain run_name column. "
            f"Matched files columns: {list(out.columns) if len(out.columns) else 'empty'}."
        )
    out = out.dropna(subset=["run_name"])
    out = out.drop_duplicates(subset=["run_name"], keep="last")
    return out


def _finalize_from_summary(
    summary: pd.DataFrame,
    *,
    paths,
    test_df,
    args,
    all_experiments: list[Experiment],
) -> None:
    summary_path = paths.data_dir / "experiments_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n=========================================================")
    print("Experiment comparison (val.csv)")
    print("=========================================================")
    if "val_acc" in summary.columns:
        ranked = summary.sort_values("val_acc", ascending=False, na_position="last")
    else:
        ranked = summary
    cols = [c for c in ["run_name", "val_acc", "val_n", "status", "description"] if c in ranked.columns]
    print(ranked[cols].to_string(index=False))
    print(f"\nSaved comparison to {summary_path.resolve()}")

    if not args.make_final_submission:
        return

    ok = summary[summary["val_acc"].notna()].copy() if "val_acc" in summary.columns else pd.DataFrame()
    if ok.empty:
        print("No successful experiments; cannot pick a winner.")
        return

    winner_row = ok.sort_values("val_acc", ascending=False).iloc[0]
    winner_name = str(winner_row["run_name"])
    winner_acc = float(winner_row["val_acc"])
    print(f"\n>>> Winner: {winner_name}  (val_acc={winner_acc:.4f})")

    winner_exp = next((e for e in all_experiments if e.name == winner_name), None)
    if winner_exp is None:
        print(f"Could not locate experiment definition for {winner_name}; aborting submission.")
        return

    captions_csv: Path | None = None
    if winner_exp.use_captions:
        captions_csv = _ensure_captions(paths, "test", args.model_id)

    from .make_submission import make_submission_from_adapter

    adapter_dir = _adapter_dir(paths, winner_exp.name) if winner_exp.train else None

    sub_path = paths.root / "submission.csv"
    print(f"[submit] generating final submission with {winner_name} ...")
    make_submission_from_adapter(
        run_name=winner_name,
        adapter_dir=adapter_dir,
        test_df=test_df,
        paths=paths,
        model_id=args.model_id,
        img_size=args.img_size,
        out=sub_path,
        captions_csv=captions_csv,
    )

    per_exp_copy = paths.data_dir / f"submission_{winner_name}.csv"
    pd.read_csv(sub_path).to_csv(per_exp_copy, index=False)

    save_json(
        paths.data_dir / "winner.json",
        {
            "winner": winner_name,
            "val_acc": winner_acc,
            "val_n": int(winner_row.get("val_n", args.val_n) or args.val_n),
            "submission_csv": str(sub_path),
            "per_experiment_copy": str(per_exp_copy),
        },
    )
    print(f"[submit] wrote {sub_path.resolve()} (and copy {per_exp_copy.resolve()})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--phase",
        type=str,
        choices=["all", "train", "eval", "finalize"],
        default="all",
        help=(
            "all = train+eval (+ submission if enabled). "
            "train / eval = for parallel GPU jobs (--only assigns experiments per GPU). "
            "finalize = merge shard CSVs (--merge-from) then pick winner and write submission."
        ),
    )
    ap.add_argument(
        "--eval-summary-out",
        type=Path,
        default=None,
        help=(
            "Where to write eval results for this job (parallel mode). "
            "Default: data/experiments_summary.csv for phase=all, "
            "or data/shards/experiments_summary_<suffix>.csv when --shard-suffix is set."
        ),
    )
    ap.add_argument(
        "--shard-suffix",
        type=str,
        default=None,
        help=(
            "With --phase eval only: writes results to "
            "data/shards/experiments_summary_<suffix>.csv unless --eval-summary-out is given."
        ),
    )
    ap.add_argument(
        "--merge-from",
        nargs="+",
        default=None,
        help="Glob patterns for phase finalize, e.g. data/shards/summary_gpu*.csv",
    )

    ap.add_argument("--train-n", type=int, default=1500, help="Train subset size (per experiment).")
    ap.add_argument("--val-n", type=int, default=500, help="Val subset size used for ranking.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--ft-img-size", type=int, default=192)
    ap.add_argument("--img-size", type=int, default=224, help="Inference image size for val/test.")
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="If given, only run experiments with these names.",
    )
    ap.add_argument(
        "--skip-train-existing",
        action="store_true",
        default=True,
        help="If an adapter already exists at the expected path, skip its training (default).",
    )
    ap.add_argument("--no-skip-train-existing", action="store_false", dest="skip_train_existing")

    ap.add_argument(
        "--make-final-submission",
        action="store_true",
        default=True,
        help="After ranking, generate submission.csv with the winning experiment.",
    )
    ap.add_argument("--no-make-final-submission", action="store_false", dest="make_final_submission")

    args = ap.parse_args()

    paths = get_paths(args.root)
    set_seed(args.seed)
    train_df, val_df, test_df = load_csvs(paths.data_dir)

    all_experiments = DEFAULT_EXPERIMENTS

    if args.phase == "finalize":
        if not args.merge_from:
            raise SystemExit("--phase finalize requires --merge-from glob(s), e.g. data/shards/summary_gpu*.csv")
        merged = _merge_summary_shards(paths.root, args.merge_from)
        merged_args = argparse.Namespace(
            make_final_submission=args.make_final_submission,
            val_n=args.val_n,
            img_size=args.img_size,
            model_id=args.model_id,
        )
        _finalize_from_summary(
            merged, paths=paths, test_df=test_df, args=merged_args, all_experiments=all_experiments
        )
        return

    experiments = _filter_experiments(args.only)

    # Where eval shards go (parallel multi-GPU). `--phase train` does not write summary CSVs.
    shards_dir = paths.data_dir / "shards"
    if args.eval_summary_out is None:
        if args.phase == "eval":
            shards_dir.mkdir(parents=True, exist_ok=True)
            if args.shard_suffix:
                args.eval_summary_out = shards_dir / f"experiments_summary_{args.shard_suffix}.csv"
            else:
                args.eval_summary_out = shards_dir / "experiments_summary_single.csv"

    failures: list[str] = []

    if args.phase in ("all", "train"):
        failures = _run_train_phase(
            experiments, paths=paths, train_df=train_df, val_df=val_df, test_df=test_df, args=args
        )

    if args.phase in ("all", "eval"):
        summary_df = _run_eval_phase(experiments, paths=paths, val_df=val_df, args=args, train_failures=failures)
        if args.eval_summary_out:
            args.eval_summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(args.eval_summary_out, index=False)
            print(f"\n[wrote eval summary] {args.eval_summary_out.resolve()}")

        if args.phase == "eval":
            if args.shard_suffix is not None:
                print("[eval] shard complete; merge with --phase finalize when all GPUs finished.")
                return
            if args.only:
                cols = [
                    c
                    for c in ["run_name", "val_acc", "val_n", "status", "description"]
                    if c in summary_df.columns
                ]
                print(summary_df.sort_values("val_acc", ascending=False, na_position="last")[cols].to_string(index=False))
                return
            _finalize_from_summary(
                summary_df, paths=paths, test_df=test_df, args=args, all_experiments=all_experiments
            )
            return

    if args.phase == "all":
        _finalize_from_summary(
            summary_df, paths=paths, test_df=test_df, args=args, all_experiments=experiments
        )
        return

    if args.phase == "train":
        print("[train] done. Next: run --phase eval on each GPU shard, then --phase finalize.")


if __name__ == "__main__":
    main()
