#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from llm_madness.utils import ensure_dir, timestamp, write_json, write_text

TEMPLATES = [
    "A quiet morning in {place} where {character} decides to {action}.",
    "{character} finds a mysterious {object} and chooses to {action}.",
    "In {place}, {character} learns why {object} matters.",
    "A short note from {character} about {object} in {place}.",
]
PLACES = ["a small town", "a riverside cafe", "an empty train", "a hillside market"]
CHARACTERS = ["Ari", "Noah", "Kim", "Sam"]
OBJECTS = ["map", "letter", "key", "photograph"]
ACTIONS = ["walk away", "investigate", "make a promise", "write it down"]


def generate_lines(count: int, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(count):
        template = rng.choice(TEMPLATES)
        line = template.format(
            place=rng.choice(PLACES),
            character=rng.choice(CHARACTERS),
            object=rng.choice(OBJECTS),
            action=rng.choice(ACTIONS),
        )
        lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple synthetic text data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    run_dir = ensure_dir(args.output_dir / timestamp())
    lines = generate_lines(args.count, args.seed)
    output_path = run_dir / "generated.txt"
    write_text(output_path, "\n".join(lines) + "\n")

    write_json(
        run_dir / "metadata.json",
        {
            "count": args.count,
            "seed": args.seed,
            "output": str(output_path),
        },
    )
    print(f"wrote {len(lines)} lines to {output_path}")


if __name__ == "__main__":
    main()
