# llm-madness

A lightweight, end-to-end text LLM pipeline with clear stages and timestamped runs.

## Structure

- `data/` raw/generated/combined/curated text (gitignored)
- `configs/` editable JSON configs (tracked)
- `runs/` timestamped tokenizer/train runs (gitignored)
- `scripts/` pipeline entry points
- `llm_madness/` core model + helpers
- `examples/` tiny sample inputs

## Pipeline

1) Generate or ingest text
2) Combine/curate datasets
3) Train tokenizer
4) Train model
5) Inspect with the web UI

## Quickstart

```bash
# 1) generate arithmetic data
python -m scripts.data_generate --output-dir data/generated --count 10000 --min-value 0 --max-value 99

# 2) combine into a single dataset
python -m scripts.data_combine --inputs data/generated --output-dir data/combined --shuffle

# 3) train tokenizer
python -m scripts.train_tokenizer --config configs/tokenizer.json

# 4) train model
python -m scripts.train_model --config configs/training.json

# 5) launch inspector
python -m scripts.web_ui
```

## One-command pipeline

```bash
python -m scripts.pipeline
```

Use `--skip-*` to skip stages, or pass per-stage args like `--generate-count`, `--combine-shuffle`, `--tokenizer-config`, and `--training-config`.

## Notes

- Training runs save `run.json`, `training_config.json`, `tokenizer.json`, `logs.jsonl`, `samples.jsonl`, and checkpoints inside `runs/train/<timestamp>/`.
- Tokenizer runs save `tokenizer.json` and `run.json` inside `runs/tokenizer/<timestamp>/`.
- All data and run artifacts are gitignored by default. Keep only configs and scripts in git.
- The web UI includes tokenizer stats, loss curves, samples, and basic attention/MLP introspection.
