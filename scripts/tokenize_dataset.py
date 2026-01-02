#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.datasets.manifest import load_dataset_manifest
from llm_madness.stages.tokenize_dataset import run_tokenize_dataset
from llm_madness.utils import find_latest_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize a dataset snapshot into on-disk token binaries.")
    parser.add_argument("--config", type=Path, default=Path("configs/tokenize_dataset/default__v001.json"))
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tokens"))
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.set)

    snapshot_path = args.snapshot
    dataset_manifest = args.dataset_manifest
    if dataset_manifest is not None:
        manifest = load_dataset_manifest(dataset_manifest)
        snapshot_path = Path(str(manifest.get("snapshot_path", "")))
        if not snapshot_path.exists():
            raise SystemExit(f"dataset snapshot missing: {snapshot_path}")
    if snapshot_path is None:
        raise SystemExit("no dataset provided; pass --snapshot or --dataset-manifest")

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        latest_tok = find_latest_run(Path("runs/tokenizer"), filename="tokenizer.json")
        if latest_tok is None:
            raise SystemExit("no tokenizer found; pass --tokenizer")
        tokenizer_path = latest_tok

    repo_root = Path(__file__).resolve().parents[1]
    result = run_tokenize_dataset(
        config,
        snapshot_path,
        tokenizer_path,
        args.output_dir,
        repo_root,
        dataset_manifest=dataset_manifest,
    )
    print(f"saved token dataset to {result['run_dir']}")


if __name__ == "__main__":
    main()

