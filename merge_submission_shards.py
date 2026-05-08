#!/usr/bin/env python3
"""Merge shard CSVs into a single Kaggle-style submission.csv.

Rows are reordered to match ``test.csv`` / ``sample_submission.csv`` (competition order),
not lexicographic id sort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument(
        "--parts",
        type=Path,
        nargs="+",
        required=True,
        help="Shard CSV paths (e.g. data/submission_part0.csv ...)",
    )
    ap.add_argument("-o", "--output", type=Path, default=Path("data/submission.csv"))
    ap.add_argument(
        "--write-root",
        action="store_true",
        help="Also write submission.csv next to the data dir (e.g. repo root when data-dir is ./data).",
    )
    ap.add_argument(
        "--no-write-root",
        action="store_true",
        help="Disable copying to repo root (overrides default).",
    )
    args = ap.parse_args()

    test_df = pd.read_csv(args.data_dir / "test.csv")
    order = test_df["id"].astype(str)

    parts = [pd.read_csv(p) for p in args.parts]
    df = pd.concat(parts, ignore_index=True)
    if list(df.columns) != ["id", "answer"]:
        raise SystemExit(f"Expected columns id,answer; got {list(df.columns)}")
    df["id"] = df["id"].astype(str)
    df["answer"] = df["answer"].astype(int)

    dup = df["id"].duplicated()
    if dup.any():
        raise SystemExit(f"Duplicate ids in shards: {df.loc[dup, 'id'].tolist()[:20]}")

    test_ids = set(order)
    if len(df) != len(test_ids):
        raise SystemExit(f"Row count {len(df)} != test.csv count {len(test_ids)}")
    if set(df["id"]) != test_ids:
        missing = test_ids - set(df["id"])
        extra = set(df["id"]) - test_ids
        raise SystemExit(f"id mismatch. missing={len(missing)} extra={len(extra)}")

    # Preserve competition row order (same as sample_submission.csv)
    df = (
        pd.DataFrame({"id": order})
        .merge(df, on="id", how="left")
        .reset_index(drop=True)
    )

    if df["answer"].isna().any():
        raise SystemExit("Merge produced NaN answers — missing predictions for some ids.")

    nu = df["answer"].nunique()
    if nu == 1:
        print(
            f"WARNING: every prediction is the same value ({df['answer'].iloc[0]}). "
            "That often means a scoring bug, degenerate model outputs, or you accidentally "
            "uploaded sample_submission.csv instead of real predictions."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Merged {len(df)} rows (test.csv order) -> {args.output.resolve()}")

    write_root = args.write_root or not args.no_write_root
    if write_root:
        root_sub = args.data_dir.resolve().parent / "submission.csv"
        root_sub.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(root_sub, index=False)
        print(f"Also wrote -> {root_sub.resolve()}")


if __name__ == "__main__":
    main()
