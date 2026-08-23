"""M7 memory snapshots, automatic expert budgets, and safe-point pressure handling."""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .cache import MemoryBudgetError, ResidentCache
from .errors import MemoryPressureError

BudgetSource = Literal["auto", "explicit", "disabled"]
PressureAction = Literal["none", "disable_prefetch", "evict", "shrink", "reject"]


@dataclass(frozen=True)
class MemorySnapshot:
    """Observed memory state without changing macOS or MLX global limits."""

    timestamp: float
    physical_memory_bytes: int
    recommended_working_set_bytes: int
    mlx_active_memory_bytes: int
    mlx_cache_memory_bytes: int
    mlx_peak_memory_bytes: int
    process_rss_bytes: int
    swap_total_bytes: int | None
    swap_used_bytes: int | None
    swap_free_bytes: int | None
    device_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryBudgetConfig:
    """Explicit M7 reservations; all values are per-process byte limits."""

    safety_margin_bytes: int = 2_000_000_000
    kv_reserve_bytes: int = 1_000_000_000
    scratch_reserve_bytes: int = 1_000_000_000
    wired_limit_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.safety_margin_bytes < 0:
            raise ValueError("M7 safety margin cannot be negative")
        if self.kv_reserve_bytes < 0:
            raise ValueError("M7 KV reserve cannot be negative")
        if self.scratch_reserve_bytes < 0:
            raise ValueError("M7 scratch reserve cannot be negative")
        if self.wired_limit_bytes is not None and self.wired_limit_bytes <= 0:
            raise ValueError("M7 wired limit must be greater than zero")


@dataclass(frozen=True)
class MemoryBudgetDecision:
    source: BudgetSource
    expert_budget_bytes: int | None
    safe_working_set_bytes: int
    shell_bytes: int
    available_expert_bytes: int
    snapshot: MemorySnapshot

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot"] = self.snapshot.to_dict()
        return payload


@dataclass(frozen=True)
class MemoryPressureEvent:
    action: PressureAction
    snapshot: MemorySnapshot
    resident_before_bytes: int
    resident_after_bytes: int
    cache_capacity_bytes: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot"] = self.snapshot.to_dict()
        return payload


class MemoryBudgetManager:
    """Plan a safe cache budget and react only at expert-layer safe points.

    It never treats swap as capacity. Pressure handling only disables future
    prefetches or evicts unpinned resident experts after their layer has
    synchronized; it cannot change routing or replace required computation.
    """

    def __init__(
        self,
        config: MemoryBudgetConfig | None = None,
        *,
        snapshot_provider: Callable[[], MemorySnapshot] | None = None,
    ) -> None:
        self.config = config or MemoryBudgetConfig()
        self._snapshot_provider = snapshot_provider or collect_memory_snapshot
        self._pressure_snapshot_provider = snapshot_provider or (
            lambda: collect_memory_snapshot(include_os_metrics=False)
        )
        self._decision: MemoryBudgetDecision | None = None
        self._pressure_level = 0
        self._events: list[MemoryPressureEvent] = []

    @property
    def decision(self) -> MemoryBudgetDecision | None:
        return self._decision

    def snapshot(self) -> MemorySnapshot:
        """Return a fresh read-only memory observation for logs and benchmarks."""

        return self._snapshot_provider()

    def plan(
        self,
        *,
        shell_bytes: int,
        requested_expert_budget_bytes: int | None,
        auto_enabled: bool,
        minimum_expert_bytes: int,
    ) -> MemoryBudgetDecision:
        if shell_bytes < 0:
            raise ValueError("M7 shell memory cannot be negative")
        if minimum_expert_bytes <= 0:
            raise ValueError("M7 minimum expert bundle must be greater than zero")
        if requested_expert_budget_bytes is not None and requested_expert_budget_bytes <= 0:
            raise ValueError("M7 resident budget must be greater than zero")
        if requested_expert_budget_bytes is not None and auto_enabled:
            raise ValueError("M7 resident budget cannot be both explicit and automatic")
        snapshot = self._snapshot_provider()
        working_set = snapshot.recommended_working_set_bytes or snapshot.physical_memory_bytes
        safe_working_set = working_set - self.config.safety_margin_bytes
        available = safe_working_set - shell_bytes - self.config.kv_reserve_bytes
        available -= self.config.scratch_reserve_bytes
        if (
            requested_expert_budget_bytes is not None or auto_enabled
        ) and available < minimum_expert_bytes:
            raise MemoryPressureError(
                "M7 has insufficient safe working-set memory for one routed expert: "
                f"available={max(available, 0)} required={minimum_expert_bytes}"
            )
        if requested_expert_budget_bytes is not None:
            if requested_expert_budget_bytes > available:
                raise MemoryPressureError(
                    "requested resident budget exceeds the M7 safe expert budget: "
                    f"requested={requested_expert_budget_bytes} available={available}"
                )
            source: BudgetSource = "explicit"
            budget: int | None = requested_expert_budget_bytes
        elif auto_enabled:
            source = "auto"
            budget = available
        else:
            source = "disabled"
            budget = None
        self._apply_wired_limit_if_requested()
        self._decision = MemoryBudgetDecision(
            source=source,
            expert_budget_bytes=budget,
            safe_working_set_bytes=safe_working_set,
            shell_bytes=shell_bytes,
            available_expert_bytes=available,
            snapshot=snapshot,
        )
        return self._decision

    def enforce(self, cache: ResidentCache) -> MemoryPressureEvent:
        """Run one staged pressure action after an MoE layer has completed."""

        if self._decision is None:
            raise RuntimeError("M7 plan must run before pressure enforcement")
        snapshot = self._pressure_snapshot_provider()
        resident_before = cache.stats().resident_bytes
        if snapshot.mlx_active_memory_bytes <= self._decision.safe_working_set_bytes:
            self._pressure_level = 0
            return self._event("none", snapshot, resident_before, cache)

        if self._pressure_level == 0:
            self._pressure_level = 1
            return self._record("disable_prefetch", snapshot, resident_before, cache)

        excess = snapshot.mlx_active_memory_bytes - self._decision.safe_working_set_bytes
        recovery_bytes = excess + self.config.safety_margin_bytes // 4
        target = max(0, resident_before - recovery_bytes)
        if self._pressure_level == 1:
            self._pressure_level = 2
            try:
                cache.evict_to(target)
            except MemoryBudgetError as error:
                raise MemoryPressureError("M7 cannot evict enough resident experts") from error
            return self._record("evict", snapshot, resident_before, cache)

        if self._pressure_level == 2:
            self._pressure_level = 3
            new_capacity = max(self._minimum_capacity(cache), target)
            try:
                cache.resize(new_capacity)
            except MemoryBudgetError as error:
                raise MemoryPressureError("M7 cannot shrink the resident cache safely") from error
            return self._record("shrink", snapshot, resident_before, cache)

        self._pressure_level = 4
        self._record("reject", snapshot, resident_before, cache)
        raise MemoryPressureError(
            "M7 memory pressure persisted after prefetch disable, eviction, and cache shrink"
        )

    def events(self) -> tuple[MemoryPressureEvent, ...]:
        return tuple(self._events)

    def _minimum_capacity(self, cache: ResidentCache) -> int:
        return cache.minimum_entry_bytes() or 1

    def _record(
        self,
        action: PressureAction,
        snapshot: MemorySnapshot,
        resident_before: int,
        cache: ResidentCache,
    ) -> MemoryPressureEvent:
        event = self._event(action, snapshot, resident_before, cache)
        self._events.append(event)
        return event

    @staticmethod
    def _event(
        action: PressureAction,
        snapshot: MemorySnapshot,
        resident_before: int,
        cache: ResidentCache,
    ) -> MemoryPressureEvent:
        return MemoryPressureEvent(
            action=action,
            snapshot=snapshot,
            resident_before_bytes=resident_before,
            resident_after_bytes=cache.stats().resident_bytes,
            cache_capacity_bytes=cache.capacity_bytes,
        )

    def _apply_wired_limit_if_requested(self) -> None:
        if self.config.wired_limit_bytes is None:
            return
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("M7 wired limit requires MLX") from error
        mx.set_wired_limit(self.config.wired_limit_bytes)


def collect_memory_snapshot(*, include_os_metrics: bool = True) -> MemorySnapshot:
    """Collect M7 observability data on macOS without mutating system state."""

    try:
        import mlx.core as mx
    except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
        raise RuntimeError("M7 memory collection requires MLX") from error
    info = mx.device_info()
    physical = _as_nonnegative_int(info.get("memory_size"))
    recommended = _as_nonnegative_int(info.get("max_recommended_working_set_size"))
    if include_os_metrics:
        process_rss = _process_rss_bytes()
        swap_total, swap_used, swap_free = _swap_usage()
    else:
        process_rss = 0
        swap_total = swap_used = swap_free = None
    return MemorySnapshot(
        timestamp=time.time(),
        physical_memory_bytes=physical,
        recommended_working_set_bytes=recommended,
        mlx_active_memory_bytes=int(mx.get_active_memory()),
        mlx_cache_memory_bytes=int(mx.get_cache_memory()),
        mlx_peak_memory_bytes=int(mx.get_peak_memory()),
        process_rss_bytes=process_rss,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_free_bytes=swap_free,
        device_name=str(info.get("device_name")) if info.get("device_name") else None,
    )


def _process_rss_bytes() -> int:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip()) * 1024
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0


def _swap_usage() -> tuple[int | None, int | None, int | None]:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    match = re.search(
        r"total\s+=\s+([\d.]+)([KMG])\s+used\s+=\s+([\d.]+)([KMG])\s+free\s+=\s+([\d.]+)([KMG])",
        result.stdout,
    )
    if match is None:
        return None, None, None
    total, total_unit, used, used_unit, free, free_unit = match.groups()
    return (
        _unit_bytes(total, total_unit),
        _unit_bytes(used, used_unit),
        _unit_bytes(free, free_unit),
    )


def _unit_bytes(value: str, unit: str) -> int:
    return int(float(value) * {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[unit])


def _as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
