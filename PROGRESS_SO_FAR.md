# Progress so far (updated 2026-05-07 evening)

Summary of **legal adapters**, **diagnostic full-FT**, **scoring**, and **what to do next**.

## Constraint

- Competition cap: **≤ 5M trainable parameters** (`rules.txt`).
- **Full fine-tune** (~507M trainable) is **diagnostic only**, not for submission.

---

## Legal adapters (MCQ contrastive, last 16 layers, `letter`)

| Run | Path | Trainable | Val (1048, letter, post-train) |
|-----|------|-----------|----------------------------------|
| r1 | `data/adapters/mcq224_cont_r1_last16_letter` | 220,160 | **0.5878** |
| r2 | `data/adapters/mcq224_cont_r2_last16_letter` | 440,320 | **0.5887** |

**Pick r2** as the better legal checkpoint for the *last-16* family (small margin): r1 ~0.5878 vs r2 ~0.5887 at val1048 (224/letter, after training).

## Final full-val sweep (locked protocol: val1048, 384px, `letter`)

We ran a unified evaluation for all adapters with: **split=val**, **n=1048**, **img_size=384**, **modes=letter**, **seed=42**.

Top results:
- (illegal diagnostic ceiling) `multimodal_fullft_20260507_1149`: **0.638359**
- (best legal) `mcq224_cont_letter`: **0.603053**
- `mcq224_cont_r2_last16_letter`: **0.599237**
- `mcq224_cont_r1_last16_letter`: **0.596374**
- `mcq224_cont_lchoice`: **0.592557**

**Decision (current best legal pipeline)**:
- **Adapter**: `data/adapters/mcq224_cont_letter`
- **Scoring mode**: `letter`
- **Inference image size**: 384
- **Submission CSV**: `data/submission_mcq224_cont_letter_img384_letter.csv` (generated via sbatch)

## Important correction (r2 “2-epoch” job)

Job `8187892` did **not** run the intended `attn_mlp_lora_r2_last16` **2-epoch** recipe. It produced a different **1-epoch** adapter (`cfg_name=attn_only_lora`, `target_modules=[q_proj,v_proj]`), saved as:
- `data/adapters/mcq_contrast_letter_20260507_1822`

## Now running (correct) r2 2 epochs

**In flight:** retrain `mcq224_cont_r2_last16_letter_e2` (LoRA r2 + last-16, `attn_mlp_lora_r2_last16`, **EPOCHS=2**, MCQ mode `letter`), job **`8204236`**.

## Submissions (384, letter) for top legal adapters

We submitted three parallel submissions (384/letter):
- `mcq224_cont_letter` → job **`8204182`**
- `mcq224_cont_r2_last16_letter` → job **`8204183`**
- `mcq224_cont_r1_last16_letter` → job **`8204184`**

---

## Scoring policy **applied** (for legal pipeline)

| Decision | Rationale |
|----------|-----------|
| **Primary mode: `letter`** | On full-FT **val200** (seed 42): `letter` beats `letter_choice` and `choice_text` at both 224 and 384. Train/val slices can disagree at n=100 — use **n≥200** before changing policy. |
| **Inference resolution: 384** | Slight gain vs 224 on diagnostic val200 for `letter` (e.g. 0.575 → 0.590). |
| **`letter` + `letter_choice` linear mix (e.g. 0.7/0.3)** | **Cancelled** for now: hurts vs pure `letter` on full-FT train200/val200 @384; tiny gain on r1@384 only. |
| **`letter_choice` truncation** | On full-FT val200@224, **`mcq-choice-max-chars=80`** is best among {80,120,160,240} for `letter_choice` (0.545 vs 0.535 at 160) and helps `choice_text` — but **still below `letter` (0.575)**. If you ever evaluate `letter_choice`, use **80** unless new evidence says otherwise. |
| **Position bias** | Pred mass on option 0 vs gold distribution → favor **`--mcq-shuffle-choice-order`** in next training runs. |

**Slurm reminder:** `sbatch_eval_scoring_modes.sbatch` reads **environment variables**. Use:

```bash
SPLIT=val N=200 SHUFFLE_SEED=42 IMG_SIZE=224 RUN_NAME=... ADAPTER_DIR=... \
  MCQ_CHOICE_MAX_CHARS=80 MODES=letter_choice,choice_text \
  sbatch pyfiles/sbatch_eval_scoring_modes.sbatch
```

Do **not** pass `KEY=val` as positional args after the script name.

---

## Ensemble (no retrain)

On **diagnostic full-FT**, **val200**, **averaging `score_letter` across 224 and 384** CSVs:

- `python -m pyfiles.ensemble_scores --score-col score_letter --scores-csv ...224... ...384...`
- **Accuracy: 0.595** (vs **0.590** at 384 alone, **0.575** at 224 alone).

**Apply for submission / final eval:** generate two score dumps (same seed, `letter` only) at 224 and 384, then ensemble. (Script: `pyfiles/ensemble_scores.py`.)

**Next:** when `val_scores_r2_val200_384_*.csv` exists, run the same ensemble for **legal r2**.

---

## Diagnostic full-FT reference

- Weights: `data/adapters/multimodal_fullft_20260507_1149`
- Post-train val1048: **0.6288** (illegal baseline)

---

## Code / tooling

- Contrastive crash fix: `TrainingArguments(remove_unused_columns=False)` in `train_adapter.py`.
- `eval_scoring_modes`: `--split`, `--n` (not `--val-n`).
- `mix_mode_scores.py`: weighted mix of `score_letter` / `score_letter_choice` from one long-form CSV.
- `sbatch_make_submission.sbatch`: configurable via `RUN_NAME`, `ADAPTER_DIR`, `IMG_SIZE`, `MCQ_COMPLETION_MODE`, `MCQ_CHOICE_MAX_CHARS`; by default it copies the produced file to repo-root `submission.csv`.
- For parallel runs without overwriting `submission.csv`, use `sbatch_make_submission_nocp.sbatch` (writes only to `OUT_CSV`).

---

## Next planning steps (ordered)

1. Wait for **8204236** (correct r2 last-16, **EPOCHS=2**, `letter`); compare its val1048 against:
   - current best legal at 384/letter: `mcq224_cont_letter` (~0.603053)
   - 1-epoch last-16 r2: `mcq224_cont_r2_last16_letter` (~0.599237)
2. Once the correct r2-2epoch result lands, keep the strongest adapter and re-generate its 384/letter submission (or replace the earlier one).
3. If r2-2epoch does not beat the best legal, consider the next cheap gain: **train r2 with `--mcq-shuffle-choice-order`** (1–2 epochs) and re-evaluate under the same `letter`, 384 protocol.
4. Run **letter-only** `eval_scoring_modes` on any new candidate if needed; otherwise finalize based on the already-produced `final_val_letter384` sweep.
5. Optional: calibration/ensembling only after we lock the best adapter+scoring mode.

---

## Roadmap (do this next, legal-first)

- **Stay legal first**: full fine-tune is diagnostic only; final method must stay under **5M trainable parameters**. Current legal adapters (~220k–440k) leave room for larger-but-still-legal adapters.
- **Base recipe**: use **r2 last-16** as the default starting point: `CFG=attn_mlp_lora_r2_last16`, MCQ contrastive, `letter`, train @224.
- **Hard gate**: **do not launch five more jobs** until **`8204236` finishes**. Compare it cleanly against the current best legal.
- **Scoring choice (locked for now)**:
  - primary: **`letter`**
  - avoid: `choice_text` alone
  - keep `letter_choice` as backup only (if tried, use `--mcq-choice-max-chars 80`)
- **Inference choice (locked for now)**: default to **384**.
- **Try a 224+384 ensemble (no retrain)**: average per-option `score_letter` from 224 and 384 runs; validate on the legal r2 adapter before using on test.
- **Next training experiment after 8204236**: position bias → train with **choice-order shuffling**:
  - r2 last-16, MCQ contrastive, `letter`, `--mcq-shuffle-choice-order`, 2 epochs
- **Scale up within cap**:
  - try **r4 last-16 (attn+MLP)**; compute trainable params and assert **<5M**
  - try **more layers**, not just higher rank (e.g., last-24; or all layers at low rank if still <5M)
- **Low-cost extra trainables** (after the above is stable):
  - unfreeze LayerNorm/RMSNorm weights (tiny param cost)
  - if still <5M, consider unfreezing the multimodal projector (image-language alignment)
- **Best legal model idea**:
  - `LoRA r2/r4 last16 or last24 + train norms + (optional) train projector + MCQ contrastive + choice shuffle`
- **When to use `letter_choice` training**: only if `letter` plateaus; one clean run:
  - r2 last-16, MCQ contrastive, `letter_choice`, `--mcq-choice-max-chars 80`, 384 inference
- **Final stage**: once the best recipe is selected, retrain on **train+val** and generate the final submission.
- **Evaluation discipline**: trust only **full val (1048)** for go/no-go; use 100-val only for debugging.
