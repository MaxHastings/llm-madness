from __future__ import annotations

import random

from .registry import Generator, register


class ArithmeticGenerator(Generator):
    def generate(self, config: dict) -> list[str]:
        seed = config.get("seed")
        rng = random.Random(seed)
        count = int(config.get("count", 10000))
        min_value = int(config.get("min_value", 0))
        max_value = int(config.get("max_value", 99))
        include_addition = bool(config.get("addition", True))
        include_subtraction = bool(config.get("subtraction", True))
        allow_negative = bool(config.get("allow_negative", False))

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
            result = a + b if op == "+" else a - b
            lines.append(f"{a} {op} {b} = {result}")
        return lines

    def metadata(self, config: dict) -> dict:
        return {
            "seed": config.get("seed"),
            "count": config.get("count"),
            "min_value": config.get("min_value"),
            "max_value": config.get("max_value"),
            "addition": bool(config.get("addition", True)),
            "subtraction": bool(config.get("subtraction", True)),
            "allow_negative": bool(config.get("allow_negative", False)),
        }


register("arithmetic", ArithmeticGenerator)
