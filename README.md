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
# 1) generate some placeholder data
python scripts/data_generate.py --output-dir data/generated

# 2) combine into a single dataset
python scripts/data_combine.py --inputs data/generated --output-dir data/combined --shuffle

# 3) train tokenizer
python scripts/train_tokenizer.py --config configs/tokenizer.json

# 4) train model
python scripts/train_model.py --config configs/training.json

# 5) launch inspector
python scripts/web_ui.py
```

## Notes

- Training runs save `run.json`, `tokenizer.json`, and checkpoints inside `runs/train/<timestamp>/`.
- Tokenizer runs save `tokenizer.json` and `run.json` inside `runs/tokenizer/<timestamp>/`.
- All data and run artifacts are gitignored by default. Keep only configs and scripts in git.
