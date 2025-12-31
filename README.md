# llm-madness

A lightweight, end-to-end text LLM pipeline with clear stages and timestamped runs.

## Structure

- `data/` raw/curated text (gitignored)
- `configs/` editable JSON configs (tracked)
  - `configs/tokenizer/` versioned tokenizer configs
- `runs/` timestamped tokenizer/train runs (gitignored)
- `scripts/` pipeline entry points
- `user_scripts/` your own utilities (gitignored; run manually)
- `llm_madness/` core model + helpers
- `examples/` tiny sample inputs

## Pipeline

1) Ingest text into `data/`
2) Train tokenizer
3) Train model
4) Inspect with the web UI

## Dataset manifests (v1 direction)

- Use the Web UI "Datasets" page to create a `DatasetManifest` from multiple `data/**.txt` files/folders.
- The manifest is saved under `runs/datasets/<id>/dataset_manifest.json` along with a materialized `snapshot.txt` used for training/tokenizer runs.

## Quickstart

```bash
# 1) train tokenizer
python -m scripts.train_tokenizer --config configs/tokenizer/default__v001.json --input data/raw/your_data.txt

# 2) train model
python -m scripts.train_model --config configs/training.json --data data/raw/your_data.txt --tokenizer runs/tokenizer/latest/tokenizer.json

# 3) launch inspector
python -m scripts.web_ui
```

Use `--set key=value` to override any config field (dot paths supported).

## Notes

- Runs write a uniform `run.json` manifest plus stage-specific artifacts inside their run directories.
- Training runs save `training_config.json`, `tokenizer.json`, `logs.jsonl`, `samples.jsonl`, and checkpoints inside `runs/train/<timestamp>/`.
- Tokenizer runs save `tokenizer.json` and `report.json` inside `runs/tokenizer/<timestamp>/`.
- All data and run artifacts are gitignored by default. Keep only configs and scripts in git.
- The web UI includes tokenizer stats, loss curves, samples, and layer-wise top-k inspection.
