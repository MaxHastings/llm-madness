#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.datasets.manifest import load_dataset_manifest
from llm_madness.stages.tokenizer import run_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.set)
    input_path = args.input
    if args.dataset_manifest is not None:
        manifest = load_dataset_manifest(args.dataset_manifest)
        snapshot_path = Path(manifest.get("snapshot_path", ""))
        if not snapshot_path.exists():
            raise SystemExit(f"dataset snapshot missing: {snapshot_path}")
        input_path = snapshot_path
    if input_path is None:
        raise SystemExit("no dataset provided; pass --input or --dataset-manifest")
    output_dir = args.output_dir or Path(config.get("output_dir", "runs/tokenizer"))
    repo_root = Path(__file__).resolve().parents[1]
    result = run_tokenizer(config, input_path, output_dir, repo_root)
    print(f"saved tokenizer to {result['output_path']}")


if __name__ == "__main__":
    main()
