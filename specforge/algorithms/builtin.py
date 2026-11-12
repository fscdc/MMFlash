"""Explicit immutable catalog of the built-in algorithms: MMFlash and the DFlash baseline it extends."""

from __future__ import annotations

from specforge.algorithms.dflash.providers import create_registration as dflash
from specforge.algorithms.mmflash.providers import create_registration as mmflash
from specforge.algorithms.registry import AlgorithmRegistry


def builtin_algorithm_registry() -> AlgorithmRegistry:
    """Return a fresh immutable catalog without module-level mutation."""

    return AlgorithmRegistry((dflash(), mmflash()))


__all__ = ["builtin_algorithm_registry"]
