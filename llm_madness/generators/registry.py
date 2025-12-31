from __future__ import annotations

from typing import Callable


class Generator:
    def generate(self, config: dict) -> list[str]:
        raise NotImplementedError

    def metadata(self, config: dict) -> dict:
        return {}


_REGISTRY: dict[str, Callable[[], Generator]] = {}


def register(name: str, generator_class: Callable[[], Generator]) -> None:
    _REGISTRY[name] = generator_class


def get(name: str) -> Generator:
    if name not in _REGISTRY:
        raise KeyError(f"unknown generator '{name}'")
    return _REGISTRY[name]()
