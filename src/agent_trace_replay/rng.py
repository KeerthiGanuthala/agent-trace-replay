"""Keyed deterministic pseudo-randomness over SHA-256.

Every random draw in the simulation, framework, and policy is a pure function
of `(seed, *key)`. There is no mutable global PRNG state, so import order,
skipped steps, and extra diagnostic calls never perturb unrelated streams.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence, TypeVar

T = TypeVar("T")

_TWO_POW_53 = float(1 << 53)


def _canonical_part(part: Any) -> str:
    if isinstance(part, bool):
        return "bool:true" if part else "bool:false"
    if isinstance(part, int):
        return f"int:{part}"
    if isinstance(part, float):
        return f"float:{part:.12g}"
    return f"str:{part}"


def draw(seed: int | str, *key: Any) -> float:
    """Return a deterministic float in [0, 1) hashed from (seed, *key)."""
    joined = "|".join([_canonical_part(seed), *(_canonical_part(k) for k in key)])
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    top_53_bits = int.from_bytes(digest[:8], "big") >> 11
    return top_53_bits / _TWO_POW_53


def draw_bool(seed: int | str, probability: float, *key: Any) -> bool:
    """Return True with probability `probability` in [0, 1]."""
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return draw(seed, *key) < probability


def draw_int(seed: int | str, low: int, high: int, *key: Any) -> int:
    """Return a deterministic integer in [low, high] inclusive."""
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    span = high - low + 1
    u = draw(seed, *key)
    idx = int(u * span)
    if idx >= span:
        idx = span - 1
    return low + idx


def draw_choice(seed: int | str, options: Sequence[T], *key: Any) -> T:
    """Pick one element deterministically from a non-empty sequence."""
    if not options:
        raise ValueError("options must not be empty")
    idx = draw_int(seed, 0, len(options) - 1, *key)
    return options[idx]
