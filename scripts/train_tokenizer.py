#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.stages import run_tokenizer
from llm_madness.utils import find_latest_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.set)
    input_path = args.input
    if input_path is None:
        latest = find_latest_run(Path("data/combined"), filename="combined.txt")
        if latest is None:
            raise SystemExit("no combined dataset found; pass --input")
        input_path = latest
    output_dir = args.output_dir or Path(config.get("output_dir", "runs/tokenizer"))
    repo_root = Path(__file__).resolve().parents[1]
    result = run_tokenizer(config, input_path, output_dir, repo_root)
    print(f"saved tokenizer to {result['output_path']}")


if __name__ == "__main__":
    main()
