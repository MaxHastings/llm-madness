#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.stages import run_combine


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine multiple text files into one dataset.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--inputs", nargs="+", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--shuffle", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    args = parser.parse_args()

    defaults = {"shuffle": False, "seed": 1337, "min_chars": 1, "output_dir": "data/combined"}
    config = load_config(args.config, defaults=defaults, overrides=args.set)
    inputs = args.inputs or [Path(p) for p in config.get("inputs", [])]
    if not inputs:
        raise SystemExit("no inputs provided; pass --inputs or config.inputs")

    if args.shuffle is not None:
        config["shuffle"] = args.shuffle
    if args.seed is not None:
        config["seed"] = args.seed
    if args.min_chars is not None:
        config["min_chars"] = args.min_chars

    output_dir = args.output_dir or Path(config.get("output_dir", "data/combined"))
    repo_root = Path(__file__).resolve().parents[1]
    result = run_combine(config, inputs, output_dir, repo_root)
    print(f"combined into {result['output_path']}")


if __name__ == "__main__":
    main()
