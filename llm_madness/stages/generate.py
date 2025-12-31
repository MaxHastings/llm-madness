from __future__ import annotations

from pathlib import Path

from llm_madness.generators import get as get_generator
from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.utils import ensure_dir, timestamp, write_json, write_text


def run_generate(
    generator_config: dict,
    manifest_config: dict,
    output_dir: Path,
    generator_name: str,
    repo_root: Path,
) -> dict:
    run_id = manifest_config.get("run", {}).get("id") if isinstance(manifest_config.get("run"), dict) else None
    run_dir = ensure_dir(output_dir / (run_id or timestamp()))
    manifest = start_manifest(
        "generate",
        run_dir,
        manifest_config,
        inputs={},
        outputs={},
        repo_root=repo_root,
    )
    output_path = run_dir / "generated.txt"
    config_path = run_dir / "config.json"
    try:
        generator = get_generator(generator_name)
        lines = generator.generate(generator_config)
        write_text(output_path, "\n".join(lines) + "\n")
        write_json(config_path, manifest_config)
        meta = generator.metadata(generator_config)
        write_json(run_dir / "metadata.json", meta)
        outputs = {"output": str(output_path)}
        finish_manifest(run_dir, "complete", outputs=outputs)
        manifest["outputs"] = outputs
    except Exception as exc:
        finish_manifest(run_dir, "failed", error=str(exc))
        raise
    return {"run_dir": run_dir, "output_path": output_path}
