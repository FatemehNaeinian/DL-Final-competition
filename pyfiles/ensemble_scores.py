from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_one(path: Path, score_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "option_idx", "gold_idx", "num_choices", score_col}
    missing = sorted(need - set(df.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    return df[["id", "option_idx", "gold_idx", "num_choices", score_col]].rename(columns={score_col: str(path)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble per-option scores by averaging.")
    ap.add_argument("--score-col", type=str, required=True, help="Column name, e.g. score_letter_choice")
    ap.add_argument("--scores-csv", nargs="+", type=Path, required=True, help="One or more long-form score CSVs.")
    ap.add_argument("--out", type=Path, default=None, help="Optional output CSV of ensembled per-option scores.")
    args = ap.parse_args()

    score_col = str(args.score_col)
    paths = list(args.scores_csv)

    merged: pd.DataFrame | None = None
    for p in paths:
        one = _read_one(p, score_col)
        if merged is None:
            merged = one
        else:
            merged = merged.merge(one, on=["id", "option_idx", "gold_idx", "num_choices"], how="inner")

    assert merged is not None
    score_cols = [c for c in merged.columns if c not in {"id", "option_idx", "gold_idx", "num_choices"}]
    if not score_cols:
        raise SystemExit("No score columns after merge; do the CSVs overlap on id/option_idx?")

    merged["score_ensemble"] = merged[score_cols].mean(axis=1)

    by_q = merged.sort_values(["id", "option_idx"]).groupby("id", sort=False)
    gold = by_q["gold_idx"].first()
    pred_opt = by_q["score_ensemble"].idxmax().map(merged["option_idx"])
    acc = float((pred_opt.values == gold.values).mean()) if len(gold) else 0.0
    print(f"Ensemble acc ({score_col}) over {len(paths)} CSVs on {len(gold)} questions: {acc:.4f}")

    if args.out is not None:
        merged.to_csv(args.out, index=False)
        print("Wrote", args.out.resolve())


if __name__ == "__main__":
    main()

