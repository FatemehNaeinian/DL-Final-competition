You have **2 days**, so do not try 20 ideas. Your winning strategy is: **build a clean baseline, make 3–5 controlled improvements, submit the best validation-backed version.**

Your competition is a **multimodal multiple-choice science QA task** using image + question/context + choices, and the metric is **accuracy** on hidden labels. The submission must be `submission.csv` with exactly `id, answer`, where `answer` is a **0-indexed integer**.  Also, the rules are strict: only provided data, offline-compatible inference, only `HuggingFaceTB/SmolVLM-500M-Instruct`, free-tier compute, and **≤5M trainable parameters**. 

## Your final competition plan

### Phase 0 — Fix setup first

Your notebook currently failed because Kaggle authentication was missing, then `data/train.csv` was not found. So first priority:

1. Run on Kaggle if possible, because the dataset is already mounted.
2. If using Colab, upload `kaggle.json` and download the competition data properly.
3. Confirm these files exist:

```python
data/train.csv
data/val.csv
data/test.csv
data/images/...
```

Do **not** start tuning until this works.

---

## Phase 1 — Build a strong baseline today

Use the current notebook baseline:

1. Load train/val/test.
2. Parse `choices`.
3. Drop only rows missing required fields.
4. Keep `hint` and `lecture` even if missing.
5. Use **multiple-choice log-likelihood scoring**, not free generation.

This is important: `generate()` can output messy text, but log-likelihood scoring directly compares `" A"`, `" B"`, `" C"` etc. That is the right method for this competition.

Your first serious milestone:

```text
Baseline SmolVLM + log-likelihood scoring
IMG_SIZE = 224
Prompt = context + question + choices + Answer:
Evaluate full validation
Create first submission.csv
Submit once
```

You need this even if it is weak. A valid baseline submission protects you from disaster.

---

## Phase 2 — Run only high-value experiments

Your strategy PDF suggests several improvement directions: MLP LoRA targets, DoRA, self-generated captions, prompt variants, metadata, larger image size, augmentation, and gradient checkpointing.  But not all are equal under time pressure.

Here is the priority order.

| Priority | Experiment              | Why                                                   |
| -------- | ----------------------- | ----------------------------------------------------- |
| 1        | Prompt variants         | Cheap, fast, can improve without training             |
| 2        | Add metadata            | Cheap; dataset has subject/grade/topic/category/skill |
| 3        | Higher image size       | Diagrams may need readable text/axes                  |
| 4        | LoRA attention + MLP    | Strongest training-side improvement                   |
| 5        | DoRA                    | Worth trying only after LoRA works                    |
| 6        | Image augmentation      | Only if training is stable                            |
| 7        | Self-generated captions | Risky/time-consuming; test carefully                  |

---

## Phase 3 — Prompt experiments first

Run these on full validation or at least a fixed validation subset of 500–1000 examples:

### Prompt A — current baseline

```text
<image>
Context:
...
Question: ...
Choices:
 A. ...
 B. ...
Answer:
```

### Prompt B — metadata added

Use:

```text
Metadata:
Grade: ...
Subject: ...
Topic: ...
Category: ...
Skill: ...

Context:
...
Question:
...
Choices:
...
Answer:
```

The project file confirms the dataset has metadata fields including `task, grade, subject, topic, category, skill`.  So use them.

### Prompt C — question before context

Sometimes the model behaves better when it sees the actual question first:

```text
<image>
Question: ...
Choices:
...
Context:
...
Answer:
```

### Prompt D — different answer cue

Try:

```text
Answer:
```

versus

```text
The correct answer is:
```

Pick the best prompt based on validation accuracy. Do **not** judge from 5 examples.

---

## Phase 4 — Image size experiment

Your current image size is 224. That may be too small for charts, maps, labels, and diagrams. Your strategy notes explicitly say if you cannot read axis labels or map text at 224, the model probably cannot either. 

Test:

```text
IMG_SIZE = 224
IMG_SIZE = 336
IMG_SIZE = 384
```

Use the same prompt and same validation subset. If 384 gives better accuracy and does not crash, use it for final inference.

Do not train at high resolution first. Try **inference-only high-res scoring** first.

---

## Phase 5 — LoRA training

Your baseline notebook uses QLoRA and keeps the trainable parameter count under the 5M rule. That is good. But the current basic LoRA targets only `q_proj` and `v_proj`. Your strategy says to expand LoRA targets to MLP layers like `gate_proj`, `up_proj`, and `down_proj`, because attention-only LoRA may be too limited. 

Run these three configs:

### Config 1 — Safe baseline LoRA

```python
r = 4
target_modules = ["q_proj", "v_proj"]
last_k = 8
```

### Config 2 — Better LoRA

```python
r = 4
target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
last_k = 6
```

### Config 3 — Lower-rank wide LoRA

```python
r = 2
target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
last_k = 8 or 10
```

Always print:

```python
ft_model.print_trainable_parameters()
assert trainable <= 5_000_000
```

No exception. If it exceeds 5M, reduce `r` or `last_k`.

---

## Phase 6 — Training settings

Use something like this first:

```text
epochs: 1
train samples: all train if time allows, otherwise 3000–5000
batch size: 2
grad accumulation: 8
lr: 2e-4
fp16: True
eval every 200 steps
save best/checkpoint manually
```

If training is stable and validation improves, run a second version:

```text
epochs: 2
lr: 1e-4 or 2e-4
same prompt as best prompt experiment
```

Do **not** waste time training 10 bad configs. Your goal is one clean adapter that beats baseline.

---

## Phase 7 — DoRA only if LoRA works

Your strategy suggests DoRA as a config-level upgrade with no extra inference cost.  Try it only after Config 2 works:

```python
use_dora=True
r=4
target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
last_k=6
```

If it crashes, skip. Do not burn the competition on DoRA debugging.

---

## Phase 8 — Validation tracking

Create a tiny results table manually. Something like:

| Run | Model | Prompt   | Image size | LoRA targets | Val acc | Public LB |
| --- | ----- | -------- | ---------- | ------------ | ------- | --------- |
| 0   | base  | baseline | 224        | none         | x       | x         |
| 1   | base  | metadata | 224        | none         | x       | x         |
| 2   | base  | metadata | 384        | none         | x       | x         |
| 3   | LoRA  | metadata | 224        | q/v          | x       | x         |
| 4   | LoRA  | metadata | 384        | q/v/mlp      | x       | x         |
| 5   | DoRA  | metadata | 384        | q/v/mlp      | x       | x         |

Use **full validation** for the final comparison. Public leaderboard is noisy because final ranking uses private split.  So do not chase public LB too hard.

---

## Final submission checklist

Before submitting:

```python
assert list(sub_df.columns) == ["id", "answer"]
assert len(sub_df) == len(test_df)
assert sub_df["id"].is_unique
assert sub_df["answer"].dtype.kind in "iu"
assert sub_df["answer"].min() >= 0
assert all(sub_df["answer"] < test_df["num_choices"])
```

Then save:

```python
sub_df.to_csv("submission.csv", index=False)
```

The rules require the final file to be exactly `submission.csv` with two columns: `id` and `answer`. 

---

## What I would do, brutally honestly

Do this order:

1. **Today immediately:** fix data path/authentication.
2. **Then:** run baseline full validation + first submission.
3. **Then:** prompt + metadata experiments.
4. **Then:** image size 336/384 inference experiment.
5. **Tonight:** train LoRA with `q_proj, v_proj, gate_proj, up_proj, down_proj`.
6. **Tomorrow:** evaluate LoRA, try DoRA only if easy.
7. **Final:** submit the model with best validation accuracy, not the most complicated model.

Do **not** over-focus on self-generated captions unless everything else is done. It is clever, but expensive and may introduce noisy text. Your highest-probability gains are: **better prompt, metadata, readable image resolution, and MLP LoRA under 5M.**
