#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from llm_madness.utils import ensure_dir, timestamp, write_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a text dataset by simple rules.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/curated"))
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--include-regex", type=str, default=None)
    parser.add_argument("--exclude-regex", type=str, default=None)
    args = parser.parse_args()

    text = args.input.read_text()
    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < args.min_chars:
            continue
        if include_re and not include_re.search(stripped):
            continue
        if exclude_re and exclude_re.search(stripped):
            continue
        kept.append(stripped)

    run_dir = ensure_dir(args.output_dir / timestamp())
    output_path = run_dir / "curated.txt"
    write_text(output_path, "\n".join(kept) + "\n")

    write_json(
        run_dir / "manifest.json",
        {
            "input": str(args.input),
            "output": str(output_path),
            "kept_lines": len(kept),
            "min_chars": args.min_chars,
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
        },
    )
    print(f"kept {len(kept)} lines in {output_path}")


if __name__ == "__main__":
    main()
