#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from llm_madness.utils import ensure_dir, timestamp, write_json, write_text


def generate_lines(
    count: int,
    seed: int | None = None,
    min_value: int = 0,
    max_value: int = 20,
    include_addition: bool = True,
    include_subtraction: bool = True,
    allow_negative: bool = False,
) -> list[str]:
    rng = random.Random(seed)
    ops: list[str] = []
    if include_addition:
        ops.append("+")
    if include_subtraction:
        ops.append("-")
    if not ops:
        raise ValueError("at least one of addition or subtraction must be enabled")

    lines: list[str] = []
    for _ in range(count):
        a = rng.randint(min_value, max_value)
        b = rng.randint(min_value, max_value)
        op = rng.choice(ops)
        if op == "-" and not allow_negative and b > a:
            a, b = b, a
        if op == "+":
            result = a + b
        else:
            result = a - b
        lines.append(f"{a} {op} {b} = {result}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate arithmetic addition/subtraction data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=20)
    parser.add_argument("--addition", action="store_true", help="include addition problems")
    parser.add_argument("--subtraction", action="store_true", help="include subtraction problems")
    parser.add_argument("--allow-negative", action="store_true", help="allow negative subtraction results")
    args = parser.parse_args()

    run_dir = ensure_dir(args.output_dir / timestamp())
    include_add = args.addition or (not args.addition and not args.subtraction)
    include_sub = args.subtraction or (not args.addition and not args.subtraction)
    lines = generate_lines(
        args.count,
        seed=args.seed,
        min_value=args.min_value,
        max_value=args.max_value,
        include_addition=include_add,
        include_subtraction=include_sub,
        allow_negative=args.allow_negative,
    )
    output_path = run_dir / "generated.txt"
    write_text(output_path, "\n".join(lines) + "\n")

    write_json(
        run_dir / "metadata.json",
        {
            "count": args.count,
            "seed": args.seed,
            "min_value": args.min_value,
            "max_value": args.max_value,
            "addition": include_add,
            "subtraction": include_sub,
            "allow_negative": args.allow_negative,
            "output": str(output_path),
        },
    )
    print(f"wrote {len(lines)} lines to {output_path}")


if __name__ == "__main__":
    main()
