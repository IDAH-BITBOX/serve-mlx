"""Configuration types shared by the command line and future runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass

_BYTE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*$", re.IGNORECASE)
_BYTE_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "k": 1_000,
    "kb": 1_000,
    "m": 1_000_000,
    "mb": 1_000_000,
    "g": 1_000_000_000,
    "gb": 1_000_000_000,
    "t": 1_000_000_000_000,
    "tb": 1_000_000_000_000,
    "ki": 1 << 10,
    "kib": 1 << 10,
    "mi": 1 << 20,
    "mib": 1 << 20,
    "gi": 1 << 30,
    "gib": 1 << 30,
    "ti": 1 << 40,
    "tib": 1 << 40,
}


def parse_bytes(value: str | int) -> int:
    """Parse a positive decimal (GB) or binary (GiB) byte-size value."""

    if isinstance(value, int):
        if value <= 0:
            raise ValueError("byte budget must be greater than zero")
        return value
    match = _BYTE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid byte size {value!r}; use values such as '24GB' or '24GiB'")
    amount, unit = match.groups()
    multiplier = _BYTE_MULTIPLIERS.get((unit or "").lower())
    if multiplier is None:
        raise ValueError(f"unsupported byte unit in {value!r}")
    result = int(float(amount) * multiplier)
    if result <= 0:
        raise ValueError("byte budget must be greater than zero")
    return result


def parse_resident_budget(value: str | None) -> tuple[int | None, bool]:
    """Parse a cache budget or the M7 ``auto`` sentinel.

    Returns ``(explicit_bytes, auto_enabled)`` so callers cannot mistake the
    no-cache default for a requested automatic cache.
    """

    if value is None:
        return None, False
    normalized = value.lower()
    if normalized == "auto":
        return None, True
    if normalized in {"off", "none", "disabled"}:
        return None, False
    return parse_bytes(value), False


@dataclass(frozen=True)
class RuntimeConfig:
    """Stable subset of the future runtime configuration.

    M4 uses ``resident_budget_bytes`` for the explicit expert-only LRU cache.
    M7 can instead select an automatic cache capacity after measuring the live
    non-expert shell and subtracting explicit memory reservations.
    """

    resident_budget_bytes: int | None = None
    auto_resident_budget: bool = False
    request_id: str | None = None
    prefetch_enabled: bool = False
    io_workers: int = 0
    prefetch_depth: int = 1
    async_gpu: bool = False
