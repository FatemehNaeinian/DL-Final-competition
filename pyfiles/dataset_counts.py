#!/usr/bin/env python3
"""Print competitive CSV row counts after the same dropna rules as training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import get_paths, load_csvs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    paths = get_paths(args.root)
    train_df, val_df, test_df = load_csvs(paths.data_dir)
    obj = {"train_n": len(train_df), "val_n": len(val_df), "test_n": len(test_df)}
    print(json.dumps(obj))


if __name__ == "__main__":
    main()
