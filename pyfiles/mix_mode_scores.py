from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix scoring modes from one long-form scores CSV.")
    ap.add_argument("--scores-csv", type=Path, required=True)
    ap.add_argument("--w-letter", type=float, default=0.7, help="Weight for score_letter.")
    ap.add_argument("--w-letter-choice", type=float, default=0.3, help="Weight for score_letter_choice.")
    ap.add_argument("--out", type=Path, default=None, help="Optional output CSV with score_mixed column.")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    need = {"id", "option_idx", "gold_idx", "num_choices", "score_letter", "score_letter_choice"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    w1 = float(args.w_letter)
    w2 = float(args.w_letter_choice)
    if (w1 + w2) == 0:
        raise SystemExit("w-letter + w-letter-choice must be non-zero.")

    df["score_mixed"] = w1 * df["score_letter"] + w2 * df["score_letter_choice"]

    by_q = df.sort_values(["id", "option_idx"]).groupby("id", sort=False)
    gold = by_q["gold_idx"].first()
    pred_opt = by_q["score_mixed"].idxmax().map(df["option_idx"])
    acc = float((pred_opt.values == gold.values).mean()) if len(gold) else 0.0

    print(f"Mixed acc on {len(gold)} questions: {acc:.4f}  (w_letter={w1}, w_letter_choice={w2})")

    if args.out is not None:
        df.to_csv(args.out, index=False)
        print("Wrote", args.out.resolve())


if __name__ == "__main__":
    main()

