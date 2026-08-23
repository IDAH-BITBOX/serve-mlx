"""Byte-budgeted, pin-safe resident expert LRU cache for M4."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .policy import ExpertKey


class MemoryBudgetError(RuntimeError):
    """The explicit expert-cache budget cannot admit a required exact expert."""


@dataclass
class ResidentExpert:
    """An MLX expert bundle retained in the Unified Memory working set."""

    key: ExpertKey
    arrays: dict[str, Any]
    nbytes: int
    last_used_step: int
    pin_count: int = 0


@dataclass(frozen=True)
class CacheStats:
    capacity_bytes: int
    resident_bytes: int
    lookup_count: int
    hit_count: int
    miss_count: int
    byte_hit_count: int
    byte_miss_count: int
    eviction_count: int
    admission_bytes: int
    reload_bytes: int
    average_evicted_residence_steps: float
    per_layer_hits: dict[int, int]
    per_layer_misses: dict[int, int]

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.lookup_count if self.lookup_count else 0.0

    @property
    def byte_hit_rate(self) -> float:
        total = self.byte_hit_count + self.byte_miss_count
        return self.byte_hit_count / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.resident_bytes,
            "lookup_count": self.lookup_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "byte_hit_count": self.byte_hit_count,
            "byte_miss_count": self.byte_miss_count,
            "byte_hit_rate": self.byte_hit_rate,
            "eviction_count": self.eviction_count,
            "admission_bytes": self.admission_bytes,
            "reload_bytes": self.reload_bytes,
            "average_evicted_residence_steps": self.average_evicted_residence_steps,
            "per_layer_hits": self.per_layer_hits,
            "per_layer_misses": self.per_layer_misses,
        }


class ResidentCache:
    """Global LRU keyed by ``(layer, expert)`` with an exact byte budget."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("resident cache capacity must be greater than zero")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[ExpertKey, ResidentExpert] = OrderedDict()
        self._resident_bytes = 0
        self._step = 0
        self._lookup_count = self._hit_count = self._miss_count = 0
        self._byte_hits = self._byte_misses = 0
        self._eviction_count = self._admission_bytes = self._reload_bytes = 0
        self._evicted_residence_steps = 0
        self._ever_admitted: set[ExpertKey] = set()
        self._admitted_at: dict[ExpertKey, int] = {}
        self._per_layer_hits: defaultdict[int, int] = defaultdict(int)
        self._per_layer_misses: defaultdict[int, int] = defaultdict(int)

    def get(self, key: ExpertKey, *, nbytes: int = 0) -> ResidentExpert | None:
        """Look up an expert and update LRU state without pinning it."""

        if nbytes < 0:
            raise ValueError("requested byte count cannot be negative")
        self._step += 1
        self._lookup_count += 1
        expert = self._entries.get(key)
        if expert is None:
            self._miss_count += 1
            self._byte_misses += nbytes
            self._per_layer_misses[key.layer] += 1
            return None
        if nbytes and expert.nbytes != nbytes:
            raise ValueError(f"cache byte size mismatch for {key}: {expert.nbytes} != {nbytes}")
        expert.last_used_step = self._step
        self._entries.move_to_end(key)
        self._hit_count += 1
        self._byte_hits += expert.nbytes
        self._per_layer_hits[key.layer] += 1
        return expert

    def contains(self, key: ExpertKey) -> bool:
        """Return residency without changing recency or cache statistics."""

        return key in self._entries

    def reserve(self, nbytes: int) -> None:
        """Evict unpinned LRU entries until a bundle of ``nbytes`` can be admitted."""

        if nbytes <= 0:
            raise ValueError("reserved byte count must be greater than zero")
        if nbytes > self.capacity_bytes:
            raise MemoryBudgetError(
                "expert bundle "
                f"({nbytes} bytes) exceeds cache capacity ({self.capacity_bytes} bytes)"
            )
        self.evict_until(nbytes)

    def admit(self, expert: ResidentExpert) -> ResidentExpert:
        """Admit a newly materialized expert after making the needed capacity available."""

        if expert.nbytes <= 0:
            raise ValueError("resident expert byte count must be greater than zero")
        existing = self._entries.get(expert.key)
        if existing is not None:
            if existing.nbytes != expert.nbytes:
                raise ValueError(f"cache byte size changed for {expert.key}")
            return existing
        self.reserve(expert.nbytes)
        self._step += 1
        expert.last_used_step = self._step
        self._entries[expert.key] = expert
        self._admitted_at[expert.key] = self._step
        self._resident_bytes += expert.nbytes
        self._admission_bytes += expert.nbytes
        if expert.key in self._ever_admitted:
            self._reload_bytes += expert.nbytes
        self._ever_admitted.add(expert.key)
        self._assert_invariant()
        return expert

    def pin(self, key: ExpertKey) -> ResidentExpert:
        """Prevent a resident expert from eviction while its GPU work is in flight."""

        expert = self._entries.get(key)
        if expert is None:
            raise KeyError(f"cannot pin non-resident expert {key}")
        self._step += 1
        expert.last_used_step = self._step
        expert.pin_count += 1
        self._entries.move_to_end(key)
        return expert

    def unpin(self, key: ExpertKey) -> None:
        expert = self._entries.get(key)
        if expert is None:
            raise KeyError(f"cannot unpin non-resident expert {key}")
        if expert.pin_count <= 0:
            raise RuntimeError(f"expert {key} was unpinned without a matching pin")
        expert.pin_count -= 1

    def evict_until(self, required_bytes: int) -> None:
        """Ensure ``resident + required <= capacity`` without evicting pinned entries."""

        if required_bytes < 0:
            raise ValueError("required byte count cannot be negative")
        while self._resident_bytes + required_bytes > self.capacity_bytes:
            victim_key = next(
                (key for key, expert in self._entries.items() if expert.pin_count == 0), None
            )
            if victim_key is None:
                raise MemoryBudgetError(
                    "cache capacity is exhausted by in-flight pinned experts; cannot evict safely"
                )
            victim = self._entries.pop(victim_key)
            admitted_at = self._admitted_at.pop(victim_key)
            self._resident_bytes -= victim.nbytes
            self._eviction_count += 1
            self._evicted_residence_steps += self._step - admitted_at
        self._assert_invariant()

    def evict_to(self, target_bytes: int) -> None:
        """Evict unpinned LRU entries until resident bytes are at most ``target``."""

        if target_bytes < 0:
            raise ValueError("resident target cannot be negative")
        if target_bytes >= self._resident_bytes:
            return
        self._evict_while(lambda: self._resident_bytes > target_bytes)

    def resize(self, capacity_bytes: int) -> None:
        """Shrink or grow capacity, evicting only unpinned entries when needed."""

        if capacity_bytes <= 0:
            raise ValueError("resident cache capacity must be greater than zero")
        previous = self.capacity_bytes
        self.capacity_bytes = capacity_bytes
        try:
            self._evict_while(lambda: self._resident_bytes > self.capacity_bytes)
        except BaseException:
            self.capacity_bytes = previous
            raise
        self._assert_invariant()

    def minimum_entry_bytes(self) -> int | None:
        """Return the smallest currently resident bundle size, if any."""

        return min((entry.nbytes for entry in self._entries.values()), default=None)

    def stats(self) -> CacheStats:
        return CacheStats(
            capacity_bytes=self.capacity_bytes,
            resident_bytes=self._resident_bytes,
            lookup_count=self._lookup_count,
            hit_count=self._hit_count,
            miss_count=self._miss_count,
            byte_hit_count=self._byte_hits,
            byte_miss_count=self._byte_misses,
            eviction_count=self._eviction_count,
            admission_bytes=self._admission_bytes,
            reload_bytes=self._reload_bytes,
            average_evicted_residence_steps=(
                self._evicted_residence_steps / self._eviction_count
                if self._eviction_count
                else 0.0
            ),
            per_layer_hits=dict(self._per_layer_hits),
            per_layer_misses=dict(self._per_layer_misses),
        )

    def _assert_invariant(self) -> None:
        if self._resident_bytes != sum(expert.nbytes for expert in self._entries.values()):
            raise RuntimeError("resident cache byte accounting invariant failed")
        if self._admitted_at.keys() != self._entries.keys():
            raise RuntimeError("resident cache admission accounting invariant failed")
        if self._resident_bytes > self.capacity_bytes:
            raise RuntimeError("resident cache exceeded its explicit byte budget")

    def _evict_while(self, predicate: Callable[[], bool]) -> None:
        while predicate():
            victim_key = next(
                (key for key, expert in self._entries.items() if expert.pin_count == 0), None
            )
            if victim_key is None:
                raise MemoryBudgetError(
                    "cache capacity is exhausted by in-flight pinned experts; cannot evict safely"
                )
            victim = self._entries.pop(victim_key)
            admitted_at = self._admitted_at.pop(victim_key)
            self._resident_bytes -= victim.nbytes
            self._eviction_count += 1
            self._evicted_residence_steps += self._step - admitted_at
        self._assert_invariant()
