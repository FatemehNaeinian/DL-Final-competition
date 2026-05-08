from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


REQ_COLS = ("id", "answer_true", "answer_pred", "correct")


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing} (expected at least {list(REQ_COLS)})")
    out = df[list(REQ_COLS)].copy()
    out["id"] = out["id"].astype(str)
    out["answer_true"] = out["answer_true"].astype(int)
    out["answer_pred"] = out["answer_pred"].astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", nargs="+", type=Path, required=True, help="val_eval_*.csv files")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path for ensembled preds")
    ap.add_argument("--tie-break", type=int, default=0, help="Index of model to use on ties (0-based)")
    args = ap.parse_args()

    if not args.vals:
        raise SystemExit("Need at least one --vals path")
    if not (0 <= args.tie_break < len(args.vals)):
        raise SystemExit(f"--tie-break must be in [0,{len(args.vals)-1}]")

    dfs = [_load(p) for p in args.vals]
    base = dfs[args.tie_break].copy()
    base_ids = base["id"].tolist()
    id_set = set(base_ids)

    for p, df in zip(args.vals, dfs):
        if set(df["id"].tolist()) != id_set:
            raise SystemExit(f"{p}: id set mismatch vs tie-break file")

    gold = dict(zip(base["id"], base["answer_true"]))
    maps = [dict(zip(df["id"], df["answer_pred"])) for df in dfs]

    rows = []
    correct = 0
    for rid in base_ids:
        votes = [m[rid] for m in maps]
        c = Counter(votes)
        top = c.most_common()
        best_cnt = top[0][1]
        tied = sorted([a for a, cnt in top if cnt == best_cnt])
        if len(tied) == 1:
            pred = int(tied[0])
        else:
            pred = int(votes[args.tie_break])
        gt = int(gold[rid])
        ok = int(pred == gt)
        correct += ok
        rows.append({"id": rid, "answer_true": gt, "answer_pred_ens": pred, "correct_ens": ok})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    acc = correct / len(rows) if rows else 0.0
    print(f"Ensemble acc: {acc:.6f} ({correct}/{len(rows)}) -> {args.out.resolve()}")


if __name__ == "__main__":
    main()

