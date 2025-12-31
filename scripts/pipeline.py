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
    args = parser.parse_args()

    config = load_config(args.config, defaults=DEFAULT_PIPELINE_CONFIG, overrides=args.set)
    repo_root = Path(__file__).resolve().parents[1]
    result = run_pipeline(config, repo_root)
    print(f"pipeline run saved to {result['run_dir']}")


if __name__ == "__main__":
    main()
