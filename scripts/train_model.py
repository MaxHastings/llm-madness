#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from llm_madness.config import load_config
from llm_madness.datasets.manifest import load_dataset_manifest
from llm_madness.stages.train import run_train
from llm_madness.utils import find_latest_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT model.")
    parser.add_argument("--config", type=Path, default=Path("configs/training/default__v001.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["fresh", "fork", "resume"])
    parser.add_argument("--device", type=str, default=None, help="override training.device (auto/cpu/cuda/mps)")
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    args = parser.parse_args()

    overrides = list(args.set or [])
    if args.device:
        overrides.append(f"training.device={args.device}")
    config = load_config(args.config, overrides=overrides)

    data_path = args.data
    if args.dataset_manifest is not None:
        manifest = load_dataset_manifest(args.dataset_manifest)
        snapshot_path = Path(manifest.get("snapshot_path", ""))
        if not snapshot_path.exists():
            raise SystemExit(f"dataset snapshot missing: {snapshot_path}")
        data_path = snapshot_path
    if data_path is None:
        raise SystemExit("no dataset provided; pass --data or --dataset-manifest")

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        latest_tok = find_latest_run(Path("runs/tokenizer"), filename="tokenizer.json")
        if latest_tok is None:
            raise SystemExit("no tokenizer found; pass --tokenizer")
        tokenizer_path = latest_tok

    output_dir = args.output_dir or Path(config.get("output_dir", "runs/train"))
    repo_root = Path(__file__).resolve().parents[1]
    result = run_train(
        config,
        data_path,
        tokenizer_path,
        output_dir,
        repo_root,
        init_checkpoint=args.init_checkpoint,
        init_mode=args.mode,
    )
    print(f"training complete. run saved to {result['run_dir']}")


if __name__ == "__main__":
    main()
