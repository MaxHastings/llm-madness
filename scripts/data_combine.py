#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from llm_madness.utils import ensure_dir, list_text_files, timestamp, write_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine multiple text files into one dataset.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/combined"))
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-chars", type=int, default=1)
    args = parser.parse_args()

    files = list_text_files(args.inputs)
    if not files:
        raise SystemExit("no .txt files found in inputs")

    rng = random.Random(args.seed)
    lines: list[str] = []
    sources: list[dict] = []

    for path in files:
        text = path.read_text()
        file_lines = [line for line in text.splitlines() if len(line.strip()) >= args.min_chars]
        lines.extend(file_lines)
        sources.append({"path": str(path), "lines": len(file_lines), "chars": len(text)})

    if args.shuffle:
        rng.shuffle(lines)

    run_dir = ensure_dir(args.output_dir / timestamp())
    output_path = run_dir / "combined.txt"
    write_text(output_path, "\n".join(lines) + "\n")

    write_json(
        run_dir / "manifest.json",
        {
            "files": sources,
            "total_lines": len(lines),
            "output": str(output_path),
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
    )
    print(f"combined {len(files)} files into {output_path}")


if __name__ == "__main__":
    main()
