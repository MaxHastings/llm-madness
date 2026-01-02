from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


_TEMPLATE_RE = re.compile(r"\$\{([^}]+)\}")


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_by_path(target: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = target
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def get_by_path(target: dict, path: str) -> Any:
    cur: Any = target
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"invalid override '{raw}', expected key=value")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"invalid override '{raw}', missing key")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def resolve_templates(config: dict, max_passes: int = 6) -> dict:
    resolved = deepcopy(config)

    def resolve_string(value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            env_val = os.getenv(key)
            if env_val is not None:
                return env_val
            cfg_val = get_by_path(resolved, key)
            if cfg_val is None:
                return match.group(0)
            return str(cfg_val)

        return _TEMPLATE_RE.sub(replace, value)

    def resolve_node(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: resolve_node(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve_node(v) for v in node]
        if isinstance(node, str):
            return resolve_string(node)
        return node

    for _ in range(max_passes):
        updated = resolve_node(resolved)
        if updated == resolved:
            break
        resolved = updated
    return resolved


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    if not overrides:
        return config
    updated = deepcopy(config)
    for raw in overrides:
        key, value = parse_override(raw)
        set_by_path(updated, key, value)
    return updated


def load_config(
    path: Path | str | None,
    defaults: dict | None = None,
    overrides: list[str] | None = None,
) -> dict:
    base = deepcopy(defaults) if defaults is not None else {}
    if path is not None:
        base = deep_merge(base, load_json(path))
    if overrides:
        base = apply_overrides(base, overrides)
    return resolve_templates(base)
