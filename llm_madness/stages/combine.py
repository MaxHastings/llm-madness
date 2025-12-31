from __future__ import annotations

import random
from pathlib import Path

from llm_madness.runs import finish_manifest, start_manifest
from llm_madness.utils import ensure_dir, list_text_files, timestamp, write_json, write_text


def run_combine(
    config: dict,
    inputs: list[Path],
    output_dir: Path,
    repo_root: Path,
) -> dict:
    run_id = config.get("run", {}).get("id") if isinstance(config.get("run"), dict) else None
    run_dir = ensure_dir(output_dir / (run_id or timestamp()))
    manifest = start_manifest(
        "combine",
        run_dir,
        config,
        inputs={"inputs": [str(p) for p in inputs]},
        outputs={},
        repo_root=repo_root,
    )
    output_path = run_dir / "combined.txt"
    config_path = run_dir / "config.json"
    try:
        files = list_text_files(inputs)
        if not files:
            raise ValueError("no .txt files found in inputs")

        seed = int(config.get("seed", 1337))
        shuffle = bool(config.get("shuffle", False))
        min_chars = int(config.get("min_chars", 1))
        rng = random.Random(seed)
        lines: list[str] = []
        sources: list[dict] = []

        for path in files:
            text = path.read_text()
            file_lines = [line for line in text.splitlines() if len(line.strip()) >= min_chars]
            lines.extend(file_lines)
            sources.append({"path": str(path), "lines": len(file_lines), "chars": len(text)})

        if shuffle:
            rng.shuffle(lines)

        write_text(output_path, "\n".join(lines) + "\n")
        write_json(
            run_dir / "manifest.json",
            {
                "files": sources,
                "total_lines": len(lines),
                "output": str(output_path),
                "shuffle": shuffle,
                "seed": seed,
                "min_chars": min_chars,
            },
        )
        write_json(config_path, config)
        outputs = {"output": str(output_path)}
        finish_manifest(run_dir, "complete", outputs=outputs)
        manifest["outputs"] = outputs
    except Exception as exc:
        finish_manifest(run_dir, "failed", error=str(exc))
        raise
    return {"run_dir": run_dir, "output_path": output_path}
