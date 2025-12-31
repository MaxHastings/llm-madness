#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.stages import run_train
from llm_madness.utils import find_latest_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT model.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.set)

    data_path = args.data
    if data_path is None:
        latest = find_latest_run(Path("data/combined"), filename="combined.txt")
        if latest is None:
            latest = find_latest_run(Path("data/curated"), filename="curated.txt")
        if latest is None:
            raise SystemExit("no dataset found; pass --data")
        data_path = latest

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        latest_tok = find_latest_run(Path("runs/tokenizer"), filename="tokenizer.json")
        if latest_tok is None:
            raise SystemExit("no tokenizer found; pass --tokenizer")
        tokenizer_path = latest_tok

    output_dir = args.output_dir or Path(config.get("output_dir", "runs/train"))
    repo_root = Path(__file__).resolve().parents[1]
    result = run_train(config, data_path, tokenizer_path, output_dir, repo_root)
    print(f"training complete. run saved to {result['run_dir']}")


if __name__ == "__main__":
    main()
