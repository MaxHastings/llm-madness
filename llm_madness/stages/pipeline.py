from __future__ import annotations

import json
from pathlib import Path

from llm_madness.config import load_config
from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.stages.tokenizer import run_tokenizer
from llm_madness.stages.train import run_train
from llm_madness.utils import ensure_dir, find_latest_run, timestamp


def _resolve_latest(path: Path) -> Path:
    if "latest" not in path.parts:
        return path
    parts = list(path.parts)
    idx = parts.index("latest")
    base = Path(*parts[:idx])
    suffix = Path(*parts[idx + 1 :])
    latest = find_latest_run(base)
    if latest is None:
        return path
    return latest / suffix


def _log_event(log_path: Path, payload: dict) -> None:
    with log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def run_pipeline(config: dict, repo_root: Path) -> dict:
    runs_root = Path(config.get("paths", {}).get("runs_root", "runs"))
    run_id = config.get("run", {}).get("id")
    run_dir = ensure_dir(runs_root / "pipeline" / (run_id or timestamp()))
    log_path = run_dir / "logs.jsonl"

    start_manifest("pipeline", run_dir, config, inputs={}, outputs={}, repo_root=repo_root)
    outputs: dict[str, str] = {}

    try:
        tokenizer_cfg = config.get("tokenizer", {})
        train_cfg = config.get("train", {})

        tokenizer_path = None
        tokenizer_input = None

        if tokenizer_cfg.get("enabled", True):
            tokenizer_config_path = tokenizer_cfg.get("config", "configs/tokenizer/default__v001.json")
            tokenizer_config = load_config(Path(tokenizer_config_path))
            input_path = tokenizer_cfg.get("input")
            if input_path is None:
                raise SystemExit("pipeline tokenizer input missing; set tokenizer.input in configs/pipeline.json")
            output_dir = Path(tokenizer_cfg.get("output_dir", "runs/tokenizer"))
            tokenizer_input = _resolve_latest(Path(input_path))
            _log_event(log_path, {"stage": "tokenizer", "status": "running"})
            result = run_tokenizer(
                tokenizer_config,
                tokenizer_input,
                output_dir,
                repo_root,
                dataset_manifest=None,
            )
            tokenizer_path = result["output_path"]
            outputs["tokenizer"] = str(result["run_dir"])
            _log_event(log_path, {"stage": "tokenizer", "status": "complete", "run_dir": str(result["run_dir"])})

        if train_cfg.get("enabled", True):
            training_config_path = train_cfg.get("config", "configs/training/default__v001.json")
            training_config = load_config(Path(training_config_path))
            data_path = train_cfg.get("data")
            tokenizer_path_cfg = train_cfg.get("tokenizer")
            if tokenizer_path_cfg is None and tokenizer_path is not None:
                tokenizer_path_cfg = str(tokenizer_path)
            if tokenizer_path_cfg is None:
                raise SystemExit("pipeline tokenizer missing; set train.tokenizer or run tokenizer stage")
            if data_path is None and tokenizer_input is not None:
                data_path = str(tokenizer_input)
            if data_path is None:
                raise SystemExit("pipeline training data missing; set train.data in configs/pipeline.json")
            output_dir = Path(train_cfg.get("output_dir", "runs/train"))
            _log_event(log_path, {"stage": "train", "status": "running"})
            result = run_train(
                training_config,
                _resolve_latest(Path(data_path)),
                _resolve_latest(Path(tokenizer_path_cfg)),
                output_dir,
                repo_root,
            )
            outputs["train"] = str(result["run_dir"])
            _log_event(log_path, {"stage": "train", "status": "complete", "run_dir": str(result["run_dir"])})

        finish_manifest(run_dir, "complete", outputs=outputs)
    except Exception as exc:
        _log_event(log_path, {"stage": "pipeline", "status": "failed", "error": str(exc)})
        finish_manifest(run_dir, "failed", error=str(exc))
        raise
    return {"run_dir": run_dir, "outputs": outputs}
