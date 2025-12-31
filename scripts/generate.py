#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.stages import run_generate


DEFAULT_GENERATE_CONFIG = {
    "output_dir": "data/generated",
    "seed": 1234,
    "count": 10000,
    "min_value": 0,
    "max_value": 99,
    "addition": True,
    "subtraction": True,
    "allow_negative": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate arithmetic data via generator registry.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--generator", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-value", type=int, default=None)
    parser.add_argument("--max-value", type=int, default=None)
    parser.add_argument("--addition", action="store_true", default=None, help="include addition problems")
    parser.add_argument("--subtraction", action="store_true", default=None, help="include subtraction problems")
    parser.add_argument("--allow-negative", action="store_true", default=None, help="allow negative subtraction results")
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    args = parser.parse_args()

    config = load_config(args.config, defaults=DEFAULT_GENERATE_CONFIG, overrides=args.set)
    generator_name = args.generator or config.get("generator", "arithmetic")
    generator_config = config.get("generator_config", config)
    output_dir = args.output_dir or Path(config.get("output_dir", "data/generated"))

    overrides = {
        "count": args.count,
        "seed": args.seed,
        "min_value": args.min_value,
        "max_value": args.max_value,
        "addition": args.addition,
        "subtraction": args.subtraction,
        "allow_negative": args.allow_negative,
    }
    for key, value in overrides.items():
        if value is not None:
            generator_config[key] = value

    repo_root = Path(__file__).resolve().parents[1]
    result = run_generate(
        generator_config,
        config,
        output_dir,
        generator_name,
        repo_root,
    )
    print(f"wrote {result['output_path']}")


if __name__ == "__main__":
    main()
