from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _per_mode_preds(df: pd.DataFrame) -> dict[str, pd.Series]:
    modes = []
    for c in df.columns:
        if c.startswith("score_"):
            modes.append(c[len("score_") :])

    preds: dict[str, pd.Series] = {}
    for m in modes:
        c = f"score_{m}"
        by_q = df.sort_values(["id", "option_idx"]).groupby("id", sort=False)
        preds[m] = by_q[c].idxmax().map(df["option_idx"])
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze long-form *_scores_*.csv produced by eval_scoring_modes.py")
    ap.add_argument("--scores-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)

    # Basic sanity
    need = {"id", "option_idx", "num_choices", "gold_idx", "choice_len"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    modes = sorted({c[len('score_'):] for c in df.columns if c.startswith("score_")})
    if not modes:
        raise SystemExit("No score_* columns found.")

    by_q = df.sort_values(["id", "option_idx"]).groupby("id", sort=False)
    gold = by_q["gold_idx"].first()
    n_questions = int(gold.shape[0])

    rows = []
    for m in modes:
        c = f"score_{m}"
        pred_opt = by_q[c].idxmax().map(df["option_idx"])
        acc = float((pred_opt.values == gold.values).mean()) if n_questions else 0.0

        # Position bias: accuracy by gold position and predicted position distribution.
        pred_counts = pred_opt.value_counts(normalize=True).sort_index()
        gold_counts = gold.value_counts(normalize=True).sort_index()

        # Length bias: correlation between choice_len and score, plus between choice_len and being predicted.
        corr_len_score = float(df["choice_len"].corr(df[c])) if df["choice_len"].std() > 0 else float("nan")
        # mean choice len for predicted option
        pred_choice_len = (
            df.merge(pred_opt.rename("pred_opt"), left_on="id", right_index=True)
            .query("option_idx == pred_opt")["choice_len"]
            .mean()
        )
        gold_choice_len = (
            df.merge(gold.rename("gold_opt"), left_on="id", right_index=True)
            .query("option_idx == gold_opt")["choice_len"]
            .mean()
        )

        rows.append(
            {
                "mode": m,
                "n_questions": n_questions,
                "acc": acc,
                "corr(choice_len,score)": corr_len_score,
                "mean_choice_len_pred": float(pred_choice_len) if not np.isnan(pred_choice_len) else float("nan"),
                "mean_choice_len_gold": float(gold_choice_len) if not np.isnan(gold_choice_len) else float("nan"),
                "pred_pos_dist": pred_counts.to_dict(),
                "gold_pos_dist": gold_counts.to_dict(),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    if args.out is None:
        out = Path(str(args.scores_csv).replace(".csv", "_analysis.csv"))
    else:
        out = args.out
    out_df.to_csv(out, index=False)
    print(out_df[["mode", "n_questions", "acc", "corr(choice_len,score)", "mean_choice_len_pred", "mean_choice_len_gold"]].to_string(index=False))
    print("Wrote", out.resolve())


if __name__ == "__main__":
    main()

