from __future__ import annotations

import json
from pathlib import Path

from llm_madness.config import deep_merge, load_config
from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.stages.combine import run_combine
from llm_madness.stages.generate import run_generate
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
        generate_cfg = config.get("generate", {})
        combine_cfg = config.get("combine", {})
        tokenizer_cfg = config.get("tokenizer", {})
        train_cfg = config.get("train", {})

        generated = None
        combined = None
        tokenizer_path = None

        if generate_cfg.get("enabled", True):
            generator_name = generate_cfg.get("generator", "arithmetic")
            generator_config_path = generate_cfg.get("config")
            generator_config = {}
            if generator_config_path:
                generator_config = load_config(Path(generator_config_path))
            generator_config = deep_merge(generator_config, generate_cfg.get("params", {}))
            output_dir = Path(generate_cfg.get("output_dir", "data/generated"))
            manifest_cfg = deep_merge(generate_cfg, {"config": generator_config})
            _log_event(log_path, {"stage": "generate", "status": "running"})
            result = run_generate(
                generator_config,
                manifest_cfg,
                output_dir,
                generator_name,
                repo_root,
            )
            generated = result["output_path"]
            outputs["generate"] = str(result["run_dir"])
            _log_event(log_path, {"stage": "generate", "status": "complete", "run_dir": str(result["run_dir"])})

        if combine_cfg.get("enabled", True):
            inputs = combine_cfg.get("inputs", [])
            if not inputs:
                if generated is not None:
                    inputs = [str(Path(generated).parent)]
            input_paths = [_resolve_latest(Path(p)) for p in inputs]
            output_dir = Path(combine_cfg.get("output_dir", "data/combined"))
            combine_params = {
                "shuffle": bool(combine_cfg.get("shuffle", False)),
                "seed": int(combine_cfg.get("seed", 1337)),
                "min_chars": int(combine_cfg.get("min_chars", 1)),
            }
            _log_event(log_path, {"stage": "combine", "status": "running"})
            result = run_combine(combine_params, input_paths, output_dir, repo_root)
            combined = result["output_path"]
            outputs["combine"] = str(result["run_dir"])
            _log_event(log_path, {"stage": "combine", "status": "complete", "run_dir": str(result["run_dir"])})

        if tokenizer_cfg.get("enabled", True):
            tokenizer_config_path = tokenizer_cfg.get("config", "configs/tokenizer.json")
            tokenizer_config = load_config(Path(tokenizer_config_path))
            input_path = tokenizer_cfg.get("input")
            if input_path is None and combined is not None:
                input_path = str(combined)
            if input_path is None:
                input_path = "data/combined/latest/combined.txt"
            output_dir = Path(tokenizer_cfg.get("output_dir", "runs/tokenizer"))
            _log_event(log_path, {"stage": "tokenizer", "status": "running"})
            result = run_tokenizer(
                tokenizer_config,
                _resolve_latest(Path(input_path)),
                output_dir,
                repo_root,
            )
            tokenizer_path = result["output_path"]
            outputs["tokenizer"] = str(result["run_dir"])
            _log_event(log_path, {"stage": "tokenizer", "status": "complete", "run_dir": str(result["run_dir"])})

        if train_cfg.get("enabled", True):
            training_config_path = train_cfg.get("config", "configs/training.json")
            training_config = load_config(Path(training_config_path))
            data_path = train_cfg.get("data")
            tokenizer_path_cfg = train_cfg.get("tokenizer")
            if data_path is None and combined is not None:
                data_path = str(combined)
            if tokenizer_path_cfg is None and tokenizer_path is not None:
                tokenizer_path_cfg = str(tokenizer_path)
            if data_path is None:
                data_path = "data/combined/latest/combined.txt"
            if tokenizer_path_cfg is None:
                tokenizer_path_cfg = "runs/tokenizer/latest/tokenizer.json"
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
