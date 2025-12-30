#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from llm_madness.utils import find_latest_run


def run_step(command: list[str]) -> None:
    print(f"running: {' '.join(command)}")
    subprocess.run(command, check=True)


def resolve_latest_dir(base_dir: Path) -> Path | None:
    latest = find_latest_run(base_dir)
    if latest is None:
        return None
    return latest if latest.is_dir() else latest.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end data → tokenizer → train pipeline.")
    parser.add_argument("--skip-generate", action="store_true", help="skip data generation")
    parser.add_argument("--skip-combine", action="store_true", help="skip data combining")
    parser.add_argument("--skip-tokenizer", action="store_true", help="skip tokenizer training")
    parser.add_argument("--skip-train", action="store_true", help="skip model training")

    parser.add_argument("--generate-output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--generate-count", type=int, default=200000)
    parser.add_argument("--generate-seed", type=int, default=1234)
    parser.add_argument("--generate-min-value", type=int, default=0)
    parser.add_argument("--generate-max-value", type=int, default=99)
    parser.add_argument("--generate-addition", action="store_true")
    parser.add_argument("--generate-subtraction", action="store_true")
    parser.add_argument("--generate-allow-negative", action="store_true")

    parser.add_argument("--combine-inputs", nargs="+", type=Path, default=None)
    parser.add_argument("--combine-output-dir", type=Path, default=Path("data/combined"))
    parser.add_argument("--combine-shuffle", action="store_true")
    parser.add_argument("--combine-seed", type=int, default=1337)
    parser.add_argument("--combine-min-chars", type=int, default=1)

    parser.add_argument("--tokenizer-config", type=Path, default=Path("configs/tokenizer.json"))
    parser.add_argument("--tokenizer-input", type=Path, default=None)
    parser.add_argument("--tokenizer-output-dir", type=Path, default=Path("runs/tokenizer"))

    parser.add_argument("--training-config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--training-data", type=Path, default=None)
    parser.add_argument("--training-tokenizer", type=Path, default=None)
    parser.add_argument("--training-output-dir", type=Path, default=Path("runs/train"))

    args = parser.parse_args()

    generated_dir: Path | None = None
    combined_path: Path | None = None
    tokenizer_path: Path | None = None

    if not args.skip_generate:
        gen_cmd = [
            sys.executable,
            "-m",
            "scripts.data_generate",
            "--output-dir",
            str(args.generate_output_dir),
            "--count",
            str(args.generate_count),
            "--seed",
            str(args.generate_seed),
            "--min-value",
            str(args.generate_min_value),
            "--max-value",
            str(args.generate_max_value),
        ]
        if args.generate_addition:
            gen_cmd.append("--addition")
        if args.generate_subtraction:
            gen_cmd.append("--subtraction")
        if args.generate_allow_negative:
            gen_cmd.append("--allow-negative")
        run_step(gen_cmd)
        generated_dir = resolve_latest_dir(args.generate_output_dir)

    if not args.skip_combine:
        combine_inputs = args.combine_inputs
        if combine_inputs is None:
            if generated_dir is not None:
                combine_inputs = [generated_dir]
            else:
                latest_generated = resolve_latest_dir(Path("data/generated"))
                combine_inputs = [latest_generated] if latest_generated is not None else [Path("data/generated")]
        combine_cmd = [
            sys.executable,
            "-m",
            "scripts.data_combine",
            "--output-dir",
            str(args.combine_output_dir),
            "--seed",
            str(args.combine_seed),
            "--min-chars",
            str(args.combine_min_chars),
        ]
        if args.combine_shuffle:
            combine_cmd.append("--shuffle")
        combine_cmd.extend(["--inputs", *[str(path) for path in combine_inputs]])
        run_step(combine_cmd)
        combined_path = find_latest_run(args.combine_output_dir, filename="combined.txt")

    if not args.skip_tokenizer:
        tokenizer_input = args.tokenizer_input or combined_path
        tok_cmd = [
            sys.executable,
            "-m",
            "scripts.train_tokenizer",
            "--config",
            str(args.tokenizer_config),
            "--output-dir",
            str(args.tokenizer_output_dir),
        ]
        if tokenizer_input is not None:
            tok_cmd.extend(["--input", str(tokenizer_input)])
        run_step(tok_cmd)
        tokenizer_path = find_latest_run(args.tokenizer_output_dir, filename="tokenizer.json")

    if not args.skip_train:
        training_data = args.training_data or combined_path
        training_tokenizer = args.training_tokenizer or tokenizer_path
        train_cmd = [
            sys.executable,
            "-m",
            "scripts.train_model",
            "--config",
            str(args.training_config),
            "--output-dir",
            str(args.training_output_dir),
        ]
        if training_data is not None:
            train_cmd.extend(["--data", str(training_data)])
        if training_tokenizer is not None:
            train_cmd.extend(["--tokenizer", str(training_tokenizer)])
        run_step(train_cmd)


if __name__ == "__main__":
    main()
