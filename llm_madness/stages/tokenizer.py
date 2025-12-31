from __future__ import annotations

from pathlib import Path

from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.tokenizer import train_bpe_tokenizer
from llm_madness.utils import ensure_dir, timestamp, write_json


def run_tokenizer(
    config: dict,
    input_path: Path,
    output_dir: Path,
    repo_root: Path,
    dataset_manifest: Path | None = None,
) -> dict:
    run_id = config.get("run", {}).get("id") if isinstance(config.get("run"), dict) else None
    run_dir = ensure_dir(output_dir / (run_id or timestamp()))
    output_path = run_dir / "tokenizer.json"
    start_manifest(
        "tokenizer",
        run_dir,
        config,
        inputs={
            "input": str(input_path),
            "dataset_manifest": str(dataset_manifest) if dataset_manifest is not None else None,
        },
        outputs={"tokenizer": str(output_path)},
        repo_root=repo_root,
    )
    config_path = run_dir / "tokenizer_config.json"
    try:
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
        write_json(config_path, config)
        write_json(run_dir / "report.json", report)
        finish_manifest(run_dir, "complete", outputs={"tokenizer": str(output_path)})
    except Exception as exc:
        finish_manifest(run_dir, "failed", error=str(exc))
        raise
    return {"run_dir": run_dir, "output_path": output_path, "report": report}
