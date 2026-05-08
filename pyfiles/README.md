## `pyfiles/` — run notebook logic as scripts

These scripts are a “clean room” version of `final_competition.ipynb` that:
- run fully from the CLI (better for long runs / “run and sleep”)
- save **distinct** outputs per run (adapters, checkpoints, val CSVs, submissions)
- keep the competition constraints (SmolVLM only, no external data)

### Setup

Use the project venv:

```bash
source /home/fn2174/DL_final/.venv/bin/activate
```

### 1) Train an adapter (LoRA / DoRA) and save everything

Example (DoRA, section-10 style):

```bash
python -m pyfiles.train_adapter \
  --run-name section10_dora \
  --adapter-out data/dora_adapter \
  --cfg section10_dora \
  --epochs 1 \
  --train-n 3000 \
  --val-n 500
```

What gets saved:
- checkpoints: `data/runs/<run_name>/checkpoints/`
- adapter: `data/<...>/` (your `--adapter-out`)
- metadata: `<adapter_out>/run_meta.json`
- submission: `data/submission_<run_name>.csv`

### 2) Evaluate saved adapters on validation

```bash
python -m pyfiles.eval_adapters \
  --val-n 200 \
  --runs section10_dora:data/dora_adapter section9_lora:data/lora_s9_adapter
```

Writes:
- per-run: `data/val_eval_<run_name>_n<N>.csv`
- summary: `data/val_summary.csv`

### 3) Generate a submission from any adapter

```bash
python -m pyfiles.make_submission \
  --run-name section10_dora \
  --adapter-dir data/dora_adapter \
  --out data/submission_section10_dora.csv
```

### 4) Prompt experiments (Sections 13–15) from CLI

Baseline-ish validation:

```bash
python -m pyfiles.experiments --run-name exp_base_224 --val-n 200 --img-size 224
```

Metadata prompt:

```bash
python -m pyfiles.experiments --run-name exp_meta_224 --val-n 200 --img-size 224 --use-metadata
```

Higher-res scoring on val:

```bash
python -m pyfiles.experiments --run-name exp_base_384 --val-n 200 --img-size 384
```

### SLURM (sbatch) “run and sleep”

```bash
sbatch --account=torch_pr_106_tandon_advanced pyfiles/run_all_gpu2.sbatch
```





### 3) Generate a submission from any adapter

```bash
python -m pyfiles.make_submission \
  --run-name fullft \
  --adapter-dir /data/adapters/multimodal_fullft_20260507_1149 \
  --out data/submission_fullft.csv
```