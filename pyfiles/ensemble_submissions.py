from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "answer"]:
        raise SystemExit(f"{path}: expected columns ['id','answer'], got {list(df.columns)}")
    df["id"] = df["id"].astype(str)
    df["answer"] = df["answer"].astype(int)
    if df["id"].duplicated().any():
        raise SystemExit(f"{path}: duplicate ids")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", type=Path, required=True, help="Submission CSVs (id,answer).")
    ap.add_argument(
        "--tie-break",
        type=int,
        default=0,
        help="Index of submission to use on ties (0-based, default 0).",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    args = ap.parse_args()

    if not args.subs:
        raise SystemExit("Need at least one --subs path")
    if not (0 <= args.tie_break < len(args.subs)):
        raise SystemExit(f"--tie-break must be in [0,{len(args.subs)-1}]")

    dfs = [_load(p) for p in args.subs]
    base = dfs[args.tie_break].copy()

    # Ensure all have identical id sets
    id0 = set(base["id"].tolist())
    for p, df in zip(args.subs, dfs):
        if set(df["id"].tolist()) != id0:
            raise SystemExit(f"{p}: id set mismatch vs tie-break submission")

    # Map id -> answer for each sub
    maps = [dict(zip(df["id"], df["answer"])) for df in dfs]

    out_rows = []
    for _idx, r in base.iterrows():
        rid = str(r["id"])
        votes = [m[rid] for m in maps]
        c = Counter(votes)
        top_n = c.most_common()
        if not top_n:
            pred = int(r["answer"])
        else:
            best_count = top_n[0][1]
            tied = sorted([a for a, cnt in top_n if cnt == best_count])
            if len(tied) == 1:
                pred = int(tied[0])
            else:
                # tie: pick the tie-break submission's answer
                pred = int(votes[args.tie_break])
        out_rows.append({"id": rid, "answer": pred})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out, index=False)
    print("Wrote", len(out_df), "rows ->", args.out.resolve())


if __name__ == "__main__":
    main()

