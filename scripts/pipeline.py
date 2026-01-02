#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.stages.pipeline import run_pipeline


DEFAULT_PIPELINE_CONFIG = {
    "run": {"name": None, "tags": []},
    "paths": {"data_root": "data", "runs_root": "runs"},
    "tokenizer": {"enabled": True},
    "train": {"enabled": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end data → tokenizer → train pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.json"))
    parser.add_argument("--set", action="append", default=None, help="override config key=value (dot paths ok)")
    parser.add_argument("--train-set", action="append", default=None, help="override training config key=value (dot paths ok)")
    parser.add_argument("--device", type=str, default=None, help="convenience alias for --train-set training.device=...")
    args = parser.parse_args()

    config = load_config(args.config, defaults=DEFAULT_PIPELINE_CONFIG, overrides=args.set)
    if args.device:
        args.train_set = (args.train_set or []) + [f"training.device={args.device}"]
    if args.train_set:
        config.setdefault("train", {})
        train_cfg = config["train"]
        if not isinstance(train_cfg, dict):
            raise SystemExit("pipeline config train must be an object")
        overrides = train_cfg.get("config_overrides")
        if overrides is None:
            overrides = []
        if not isinstance(overrides, list):
            raise SystemExit("pipeline config train.config_overrides must be a list")
        train_cfg["config_overrides"] = overrides + list(args.train_set)
    repo_root = Path(__file__).resolve().parents[1]
    result = run_pipeline(config, repo_root)
    print(f"pipeline run saved to {result['run_dir']}")


if __name__ == "__main__":
    main()
