# DL-Final-competition (Pixels to Predictions)

This repo contains code to train and evaluate **parameter-efficient adapters** for the ScienceQA-style **visual multiple-choice** challenge, and to generate a Kaggle-ready `submission.csv`.

## Constraints we follow

- **Base model**: `HuggingFaceTB/SmolVLM-500M-Instruct`
- **No external data**: only competition CSVs + images (optional captions are self-generated with the base model)
- **Submission-eligible training**: **≤ 5M trainable parameters** (LoRA / DoRA adapters)

## Repository layout

- **`final_competition.ipynb`**: the “one notebook” path to generate `submission.csv`
- **`pyfiles/`**: CLI equivalents of the notebook logic (train / eval / submission)
- **`requirements.txt`**: Python deps for notebook + CLI scripts
- **`data/`** (local, not committed): competition CSVs/images, adapters, logs, results
- **`PROGRESS_SO_FAR.md`**: experiment diary + current best settings
- **`rules.txt` / `project.txt`**: competition spec and rules snapshot

## Setup

Create and activate a Python environment, then install deps:

```bash
cd DL-Final
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: install a CUDA-enabled PyTorch build that matches your system/GPU.

## Data placement

Put the Kaggle files under `data/`:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- images referenced by the CSVs, e.g. `data/images/...`

The CSVs should contain `image_path` like `images/train/train_00000.png` (relative to `data/`).

## Generate `submission.csv` (recommended)

Open and run `final_competition.ipynb`.

In Section 1, the key knobs are:

- `IMG_SIZE = 384` (final inference resolution)
- `USE_ADAPTER = True`
- `ADAPTER_DIR = data/adapters/<adapter_name>`

Then run Sections **1 → 5**.  
Section **5a** writes:

- `submission.csv` (repo root; upload this to Kaggle)
- `data/submission.csv` (copy)

## Generate `submission.csv` (CLI)

```bash
python -m pyfiles.make_submission \
  --root . \
  --run-name my_run \
  --adapter-dir data/adapters/mcq224_cont_letter \
  --out data/submission_my_run.csv \
  --img-size 384 \
  --mcq-completion-mode letter
```

## Train an adapter (LoRA / DoRA)

```bash
python -m pyfiles.train_adapter \
  --run-name mcq_lora_run \
  --adapter-out data/adapters/mcq_lora_run \
  --cfg attn_mlp_lora_r2_last16 \
  --epochs 1
```

## What is *not* committed

To keep the repo small, `.gitignore` excludes regenerable artifacts such as:

- `data/adapters/` (adapter weights)
- `data/runs/` (checkpoints)
- `data/logs/`, `data/results/`, HF cache under `data/.hf_home/`
- generated score/eval/submission CSVs

If you need to reproduce a specific run, the intended pattern is: **commit code + configs**, rerun training/evaluation, and regenerate artifacts locally.

