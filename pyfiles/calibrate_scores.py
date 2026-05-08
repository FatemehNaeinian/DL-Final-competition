from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _softmax_ce_loss(scores: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    # scores: (N, K), gold: (N,)
    return torch.nn.functional.cross_entropy(scores, gold)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiny score calibrator (learns a few weights).")
    ap.add_argument("--scores-csv", type=Path, required=True, help="Output of eval_scoring_modes.py (long form).")
    ap.add_argument(
        "--features",
        type=str,
        default="score_letter,score_letter_choice,score_choice_text,choice_len,option_idx,num_choices",
        help="Comma-separated feature columns to use.",
    )
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="Where to write learned weights JSON/CSV.")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    feats = [c.strip() for c in str(args.features).split(",") if c.strip()]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")

    # Group into (question_id, K options)
    # Assumes each id has options 0..num_choices-1.
    grp = df.groupby("id", sort=False)
    ids = list(grp.groups.keys())

    X_list = []
    y_list = []
    K_list = []
    for qid in ids:
        sub = grp.get_group(qid).sort_values("option_idx")
        K = int(sub["num_choices"].iloc[0])
        sub = sub.iloc[:K]
        X_list.append(sub[feats].to_numpy(dtype=np.float32))
        y_list.append(int(sub["gold_idx"].iloc[0]))
        K_list.append(K)

    maxK = int(max(K_list))
    F = int(len(feats))
    N = len(ids)
    X = np.zeros((N, maxK, F), dtype=np.float32)
    mask = np.zeros((N, maxK), dtype=np.float32)
    for i, (xk, k) in enumerate(zip(X_list, K_list)):
        X[i, :k, :] = xk
        mask[i, :k] = 1.0
    y = np.asarray(y_list, dtype=np.int64)

    torch.manual_seed(int(args.seed))
    X_t = torch.from_numpy(X)
    mask_t = torch.from_numpy(mask)
    y_t = torch.from_numpy(y)

    w = torch.zeros((F,), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(args.lr))

    for _ in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)
        # Linear scoring per option
        s = (X_t @ w) + b  # (N, maxK)
        s = s.masked_fill(mask_t == 0, -1e9)
        loss = _softmax_ce_loss(s, y_t) + float(args.l2) * (w.square().mean())
        loss.backward()
        opt.step()

    with torch.no_grad():
        s = (X_t @ w) + b
        s = s.masked_fill(mask_t == 0, -1e9)
        pred = s.argmax(dim=1).cpu().numpy()
        acc = float((pred == y).mean())

    out = args.out
    if out is None:
        out = Path(str(args.scores_csv).replace(".csv", "_calibration.csv"))
    out_df = pd.DataFrame({"feature": feats + ["__bias__"], "weight": w.detach().cpu().numpy().tolist() + [float(b)]})
    out_df.to_csv(out, index=False)

    print(f"Trained calibrator on {N} questions. Acc={acc:.4f}. Wrote {out.resolve()}")


if __name__ == "__main__":
    main()

