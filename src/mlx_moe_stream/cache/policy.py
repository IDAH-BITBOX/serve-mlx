"""Byte-budgeted global LRU baseline used by M1 trace analysis."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, order=True)
class ExpertKey:
    """Stable cache identity for a routed MoE expert."""

    layer: int
    expert: int


@dataclass(frozen=True)
class CacheSimulationStats:
    capacity_bytes: int
    resident_bytes: int
    calls: int
    hits: int
    misses: int
    byte_hits: int
    byte_misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls if self.calls else 0.0

    @property
    def byte_hit_rate(self) -> float:
        total = self.byte_hits + self.byte_misses
        return self.byte_hits / total if total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.resident_bytes,
            "calls": self.calls,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "byte_hits": self.byte_hits,
            "byte_misses": self.byte_misses,
            "byte_hit_rate": self.byte_hit_rate,
            "evictions": self.evictions,
        }


class LruCacheSimulator:
    """Deterministic byte-budget global LRU with no implicit oversubscription."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("LRU capacity must be greater than zero")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[ExpertKey, int] = OrderedDict()
        self._resident_bytes = 0
        self._calls = self._hits = self._misses = 0
        self._byte_hits = self._byte_misses = self._evictions = 0

    def access(self, key: ExpertKey, nbytes: int = 1) -> bool:
        """Access a bundle and return whether it was resident before the access."""

        if nbytes <= 0:
            raise ValueError("bundle size must be greater than zero")
        self._calls += 1
        existing = self._entries.get(key)
        if existing is not None:
            if existing != nbytes:
                raise ValueError(f"bundle size changed for {key}: {existing} -> {nbytes}")
            self._entries.move_to_end(key)
            self._hits += 1
            self._byte_hits += nbytes
            return True

        self._misses += 1
        self._byte_misses += nbytes
        if nbytes > self.capacity_bytes:
            return False
        while self._resident_bytes + nbytes > self.capacity_bytes:
            _, evicted_bytes = self._entries.popitem(last=False)
            self._resident_bytes -= evicted_bytes
            self._evictions += 1
        self._entries[key] = nbytes
        self._resident_bytes += nbytes
        return False

    def stats(self) -> CacheSimulationStats:
        return CacheSimulationStats(
            capacity_bytes=self.capacity_bytes,
            resident_bytes=self._resident_bytes,
            calls=self._calls,
            hits=self._hits,
            misses=self._misses,
            byte_hits=self._byte_hits,
            byte_misses=self._byte_misses,
            evictions=self._evictions,
        )


def simulate_lru_curve(
    accesses: Iterable[ExpertKey],
    capacities: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00),
) -> list[dict[str, int | float]]:
    """Simulate uniform-sized expert bundles for each requested working-set fraction."""

    access_list = list(accesses)
    working_set = len(set(access_list))
    if working_set == 0:
        return []
    rows: list[dict[str, int | float]] = []
    for fraction in capacities:
        if not 0 < fraction <= 1:
            raise ValueError("cache fractions must be in (0, 1]")
        simulator = LruCacheSimulator(max(1, ceil(working_set * fraction)))
        for key in access_list:
            simulator.access(key)
        row = simulator.stats().to_dict()
        row["capacity_fraction"] = fraction
        row["capacity_experts"] = simulator.capacity_bytes
        rows.append(row)
    return rows
