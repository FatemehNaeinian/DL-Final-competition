# DL-Final — Session summary

This document summarizes work done on the **Pixels to Predictions** competition repo under `/scratch/fn2174/DL-Final`: constraints, tooling added, Slurm jobs, and empirical takeaways.

## Competition constraints (recap)

- **Model**: `HuggingFaceTB/SmolVLM-500M-Instruct` only.
- **Metric**: accuracy on hidden test; submission **`submission.csv`** with columns **`id`**, **`answer`** (0-indexed integers).
- **Rules**: competition data only; offline-safe inference; trainable parameters **≤ 5M** (LoRA/adapters count).
- **Data**: `data/train.csv`, `val.csv`, `test.csv`; images under `data/images/`; after required-column dropna, **3109** train rows and **1048** val rows (see `python -m pyfiles.dataset_counts`).

## Environment notes

- A Python **venv** was created at `DL-Final/.venv` with CPU PyTorch for lightweight checks; primary GPU work uses conda **`/scratch/fn2174/envs/ijepa`** (torch 2.5.1+cu121, transformers 4.57.x, peft, etc.).
- **`HF_HOME`** / caches are directed under `data/.hf_home` in sbatch wrappers to reduce login-node temp disk pressure.
- Torch Slurm submissions require **`--account`** (e.g. `torch_pr_106_tandon_advanced`).

## Scripts and code changes

### Validation / comparison

- **`pyfiles/compare_val_prompts.py`**: Suites **A** (base: baseline vs metadata × 224 vs 384), **B** (LoRA + chosen prompt at 224 and 384), **C** (DoRA + best prompt/resolution). Flag **`--output-tag`** avoids overwriting summary/detail CSVs between runs.
- **`pyfiles/sbatch_compare_val_small.sbatch`**: Runs suite A/B/C with conda activation; bash-array-free for portability.
- **`pyfiles/eval_adapters.py`**: **`--shuffle-seed`** for the same random val slice across runs; summary filenames **`val_summary_adapter_compare_n*_seed*.csv`**; per-run detail CSVs include `_seed*` or `_head` suffix.
- **`pyfiles/sbatch_compare_adapters_val.sbatch`**: Compares **base** vs **`data/quick_textonly_dora`** vs **`data/runs/text_only_attn_mlp/adapter`** on a small val subset (multimodal scoring).

### Training

- **`pyfiles/train_adapter.py`** (extensions):
  - **`--text-only-fields`**: comma-separated **`hint`**, **`grade`**, **`lecture`** for text-only prompts (`build_prompt_text_only_mcq`).
  - Post-train **`evaluate_adapter_dir`** uses **`shuffle_seed=training seed`** for consistent subset reporting.
- **`pyfiles/sbatch_train_lora.sbatch`**: General QLoRA job; defaults include full train/val counts, **`MAKE_SUBMISSION=0`**, account line.
- **`pyfiles/sbatch_train_multimodal_lora_224.sbatch`**: **Multimodal** LoRA with **`--ft-img-size 224`** (no `--text-only`).
- **`pyfiles/sbatch_train_textonly_hint_grade.sbatch`**: Text-only run with **hint + grade** in prompt, **`EPOCHS` ≥ 4** default.

### Pipeline helpers

- **`pyfiles/verify_training_env.py`**: Imports torch/transformers/peft/accelerate; **bitsandbytes** only when CUDA is available.
- **`pyfiles/dataset_counts.py`**: Prints JSON train/val/test counts after **`load_csvs`** rules.
- **`pyfiles/check_submission.py`**: Validates **`submission.csv`** vs **`test.csv`** (schema, ids, answer bounds).
- **`pyfiles/submit_plan_jobs.sh`**: Chains train → suite-B dual eval → submission (optional **`PARTITION`** / **`ACCOUNT`**).
- **`pyfiles/sbatch_eval_suite_B_dual.sbatch`**: Suite **B** for baseline and metadata with tagged outputs.
- **`pyfiles/sbatch_make_submission.sbatch`**: **`make_submission`** + **`check_submission`** + copy to repo-root **`submission.csv`**.

## Slurm jobs (examples submitted during setup)

Job IDs depended on the cluster at submit time; representative chain:

- LoRA train → suite-B eval → submission (dependency chain).
- Optional parallel **DoRA** train + dependent suite-B eval.
- Adapter comparison job for legacy **text-only** checkpoints.

Check **`data/logs/`** for `*.out` / `*.err` per job.

## Existing adapters inspected

| Path | Role |
|------|------|
| `data/quick_textonly_dora` | DoRA (`section10_dora`), ~374k trainable params; text-oriented quick run |
| `data/runs/text_only_attn_mlp/adapter` | **`attn_mlp_lora`**, text-only, ~330k trainable params |

## Results tables (for report writing)

Copy the markdown tables below into write-ups; CSV backups live under `data/` where paths are listed.

### Table 1 — Early prompt exploration (legacy seven-variant runner)

**Setup:** `VAL_N=100`, **first 100 rows** of validation (head slice; not the later suite-A script). **GPU:** NVIDIA H200 (`gh129`). Uses older naming (`baseline_ctx_first_Answer_224`, etc.) from the multi-arm `compare_val_prompts` / experiments-style list.

| Rank | Approach | Image size | Val N | Correct | Accuracy |
|:----:|----------|:----------:|:-------:|:-------:|:--------:|
| 1 (tie) | `baseline_ctx_first_Answer_224` | 224 | 100 | 43 | **0.430** |
| 1 (tie) | `variant_ctx_first_Answer_224` | 224 | 100 | 43 | **0.430** |
| 1 (tie) | `metadata_grade_subject_topic_Answer_224` | 224 | 100 | 43 | **0.430** |
| 1 (tie) | `metadata_grade_subject_topic_Answer_384` | 384 | 100 | 43 | **0.430** |
| 5 | `baseline_ctx_first_Answer_384` | 384 | 100 | 42 | 0.420 |
| 6 | `ctx_first_The_correct_answer_is_224` | 224 | 100 | 36 | 0.360 |
| 7 | `question_before_context_Answer_224` | 224 | 100 | 22 | 0.220 |

**Takeaways for prose:** Question-before-context and alternate answer cue **hurt**; baseline/metadata tied at 100 examples; 384 not decisive at **n = 100**.

---

### Table 2 — Suite **A** (canonical four-cell grid, base model only)

**Setup:** `SUITE=A`, `VAL_N=500`, **`SHUFFLE_SEED=42`** (same random 500-example subset for all four cells). **Command:** `bash pyfiles/sbatch_compare_val_small.sbatch` with env vars set (not Slurm). **GPU:** NVIDIA H200. **Output:** `data/val_compare_suite_A_summary.csv` plus per-cell CSVs `val_compare_suiteA_A*_n500.csv`.

| Rank | Cell ID | Configuration | Image size | Val N | Correct | Accuracy |
|:----:|---------|---------------|:----------:|:-------:|:-------:|:--------:|
| 1 | A4 | Base + **metadata** prompt | 384 | 500 | 290 | **0.580** |
| 2 | A3 | Base + **baseline** prompt | 384 | 500 | 289 | **0.578** |
| 3 | A1 | Base + baseline prompt | 224 | 500 | 280 | 0.560 |
| 4 | A2 | Base + metadata prompt | 224 | 500 | 278 | 0.556 |

**Takeaways for prose:** At fixed seed and **n = 500**, **384** beats **224** by ~18 hits (~3.6 pp); metadata edges baseline **only at 384** (+1 hit); at 224, baseline slightly beats metadata (+2 hits).

---

### Shell tip (avoid Slurm path typo)

If you see `Unable to open file pyfiles/sbatch_compare_val_small.sbatchSUITE=A`, there must be a **space** before `SUITE=A`:

```bash
SUITE=A VAL_N=500 SHUFFLE_SEED=42 sbatch pyfiles/sbatch_compare_val_small.sbatch
```

---

## Empirical results (short narrative)

1. **Suite A (base), `VAL_N=500`, seed 42**: See **Table 2**. User chose **224 for training cost** while noting **384 + metadata** was best on this slice.
2. **Adapter comparison (`VAL_N=100`, seed 42)**: **`quick_textonly_dora`** 0.55 vs **base** 0.54 vs **`text_only_attn_mlp`** 0.54 — small gaps; text-only adapters evaluated with **multimodal** scoring.

## Recommended next steps

- Run **`sbatch_train_multimodal_lora_224.sbatch`** for image+text LoRA aligned with “224 for now.”
- After training, run **suite B** with **`ADAPTER_DIR`** pointing at the new adapter; optionally mirror suite-A seeds for fair comparison.
- For final Kaggle upload, regenerate **`submission.csv`** with **`make_submission`** and **`pyfiles.check_submission`**.

## Files worth reading

- `PLAN.md`, `rules.txt`, `project.txt` — competition framing.
- `pyfiles/README.md` — CLI examples for `train_adapter`, `eval_adapters`, `experiments`, `make_submission`.

---

*Generated as a working log of this session; update as new runs complete.*
