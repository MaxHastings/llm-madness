# llm-madness

A lightweight, end-to-end text LLM pipeline with a web UI for configs, datasets, and run inspection.

<img width="1728" height="993" alt="Screenshot 2026-01-01 at 9 07 46 PM" src="https://github.com/user-attachments/assets/8518f0e6-a1a1-42ad-a3a9-2031e1103327" />

<img width="1728" height="992" alt="Screenshot 2026-01-01 at 9 07 16 PM" src="https://github.com/user-attachments/assets/eea08ced-4e53-409e-bfab-5e17b33d2922" />

## What it does

- Tokenizer: configure BPE settings and build vocabularies from dataset snapshots.
- Training: configure model + optimizer settings and launch runs.
- Datasets: combine multiple `data/**.txt` inputs into a manifest and materialized snapshot.
- Artifacts: every run writes a `run.json` plus stage-specific outputs under `runs/`.
- Inspect: view loss curves, samples, tokenizer stats, and per-token top-k to debug behavior.

## Core concepts

- Tokenizer configs live in `configs/tokenizer/` (defaults tracked; custom UI-generated configs ignored).
- Training configs live in `configs/training/` (defaults tracked; custom UI-generated configs ignored).
- Dataset manifests live under `runs/datasets/<id>/dataset_manifest.json` with a `snapshot.txt`.
- Tokenizer runs output to `runs/tokenizer/<timestamp>/`.
- Training runs output to `runs/train/<timestamp>/`.

## Typical workflow

1) Add text files under `data/`
2) Create a dataset snapshot in the Web UI (Datasets)
3) Generate a tokenizer vocab from the snapshot (Tokenizer Vocabularies)

4) Launch a training run (Training Configs / Training Runs)
5) Inspect outputs and debug behavior (Inspect)

## Run it

```bash
pip install -r requirements.txt
python -m scripts.web_ui
```

## Notes

- `data/` and `runs/` are gitignored by default.
- Use `--set key=value` with CLI scripts to override config fields.
