#!/usr/bin/env python3
"""Validate submission CSV schema vs test.csv (Kaggle-style checks)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--submission", type=Path, default=None, help="Defaults to <root>/data/submission.csv")
    args = ap.parse_args()

    data_dir = args.root / "data"
    sub_path = args.submission if args.submission is not None else data_dir / "submission.csv"
    test_path = data_dir / "test.csv"

    if not sub_path.is_file():
        print("MISSING_SUBMISSION:", sub_path.resolve())
        return 2
    if not test_path.is_file():
        print("MISSING_TEST:", test_path.resolve())
        return 2

    sub = pd.read_csv(sub_path)
    test = pd.read_csv(test_path)

    errs: list[str] = []

    if list(sub.columns) != ["id", "answer"]:
        errs.append(f"bad_columns:{list(sub.columns)}")

    exp_ids = test["id"].astype(str).tolist()
    got_ids = sub["id"].astype(str).tolist()

    if len(sub) != len(test):
        errs.append(f"row_count:{len(sub)}_expected_{len(test)}")

    if sorted(got_ids) != sorted(exp_ids):
        missing = set(exp_ids) - set(got_ids)
        extra = set(got_ids) - set(exp_ids)
        errs.append(f"id_set_mismatch_missing_{len(missing)}_extra_{len(extra)}")

    if sub["id"].duplicated().any():
        errs.append("duplicate_ids")

    if errs:
        print("FAIL", json.dumps(errs))
        return 1

    # Bounds vs num_choices
    nc = test.set_index(test["id"].astype(str))["num_choices"].to_dict()
    bad_bounds = 0
    for _, r in sub.iterrows():
        aid = str(r["id"])
        ans = int(r["answer"])
        n = int(nc[aid])
        if ans < 0 or ans >= n:
            bad_bounds += 1

    if bad_bounds:
        print("FAIL", json.dumps([f"answer_out_of_range:{bad_bounds}_rows"]))
        return 1

    print("OK", len(sub), "rows ->", sub_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
