#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_madness.tokenizer import train_bpe_tokenizer
from llm_madness.utils import ensure_dir, find_latest_run, git_sha, timestamp, write_json


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tokenizer"))
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input
    if input_path is None:
        latest = find_latest_run(Path("data/combined"), filename="combined.txt")
        if latest is None:
            raise SystemExit("no combined dataset found; pass --input")
        input_path = latest

    run_dir = ensure_dir(args.output_dir / timestamp())
    output_path = run_dir / "tokenizer.json"

    report = train_bpe_tokenizer(
        input_path=input_path,
        output_path=output_path,
        vocab_size=int(config.get("vocab_size", 4096)),
        min_frequency=int(config.get("min_frequency", 2)),
        special_tokens=config.get("special_tokens", ["<|unk|>"]),
        discover_regex=config.get("discover_special_token_regex"),
        add_prefix_space=bool(config.get("add_prefix_space", False)),
        byte_level=bool(config.get("byte_level", True)),
        split_digits=bool(config.get("split_digits", False)),
    )

    git_commit = git_sha(Path(__file__).resolve().parents[1])
    write_json(
        run_dir / "run.json",
        {
            "config": config,
            "report": report,
            "git_sha": git_commit,
        },
    )
    write_json(run_dir / "tokenizer_config.json", config)
    print(f"saved tokenizer to {output_path}")


if __name__ == "__main__":
    main()
