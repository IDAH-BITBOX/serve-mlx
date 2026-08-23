from __future__ import annotations

from collections.abc import Iterator

import pytest

from mlx_moe_stream.cache import ExpertKey, ResidentCache, ResidentExpert
from mlx_moe_stream.errors import MemoryPressureError
from mlx_moe_stream.memory import MemoryBudgetConfig, MemoryBudgetManager, MemorySnapshot


def _snapshot(*, active: int = 0) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=1.0,
        physical_memory_bytes=1_000,
        recommended_working_set_bytes=1_000,
        mlx_active_memory_bytes=active,
        mlx_cache_memory_bytes=0,
        mlx_peak_memory_bytes=active,
        process_rss_bytes=0,
        swap_total_bytes=0,
        swap_used_bytes=0,
        swap_free_bytes=0,
        device_name="test",
    )


def _provider(snapshots: list[MemorySnapshot]) -> Iterator[MemorySnapshot]:
    yield from snapshots


def _expert(key: ExpertKey, nbytes: int) -> ResidentExpert:
    return ResidentExpert(key=key, arrays={}, nbytes=nbytes, last_used_step=0)


def test_auto_budget_measures_shell_and_reservations_before_cache_allocation():
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(
            safety_margin_bytes=100,
            kv_reserve_bytes=100,
            scratch_reserve_bytes=100,
        ),
        snapshot_provider=lambda: _snapshot(),
    )

    decision = manager.plan(
        shell_bytes=100,
        requested_expert_budget_bytes=None,
        auto_enabled=True,
        minimum_expert_bytes=10,
    )

    assert decision.source == "auto"
    assert decision.safe_working_set_bytes == 900
    assert decision.available_expert_bytes == 600
    assert decision.expert_budget_bytes == 600


def test_explicit_budget_is_limited_by_the_same_safe_budget_and_no_cache_is_allowed():
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(
            safety_margin_bytes=100,
            kv_reserve_bytes=100,
            scratch_reserve_bytes=100,
        ),
        snapshot_provider=lambda: _snapshot(),
    )
    with pytest.raises(MemoryPressureError, match="exceeds"):
        manager.plan(
            shell_bytes=100,
            requested_expert_budget_bytes=601,
            auto_enabled=False,
            minimum_expert_bytes=10,
        )

    disabled = manager.plan(
        shell_bytes=10_000,
        requested_expert_budget_bytes=None,
        auto_enabled=False,
        minimum_expert_bytes=10,
    )
    assert disabled.source == "disabled"
    assert disabled.expert_budget_bytes is None


def test_budget_plan_rejects_ambiguous_or_nonpositive_explicit_requests():
    manager = MemoryBudgetManager(snapshot_provider=lambda: _snapshot())
    with pytest.raises(ValueError, match="greater than zero"):
        manager.plan(
            shell_bytes=100,
            requested_expert_budget_bytes=0,
            auto_enabled=False,
            minimum_expert_bytes=10,
        )
    with pytest.raises(ValueError, match="both explicit and automatic"):
        manager.plan(
            shell_bytes=100,
            requested_expert_budget_bytes=10,
            auto_enabled=True,
            minimum_expert_bytes=10,
        )


def test_pressure_actions_are_staged_at_safe_points_then_reject_persisting_pressure():
    snapshots = _provider(
        [
            _snapshot(),
            _snapshot(active=1_000),
            _snapshot(active=1_000),
            _snapshot(active=1_000),
            _snapshot(active=1_000),
        ]
    )
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(safety_margin_bytes=100, kv_reserve_bytes=0, scratch_reserve_bytes=0),
        snapshot_provider=lambda: next(snapshots),
    )
    manager.plan(
        shell_bytes=100,
        requested_expert_budget_bytes=None,
        auto_enabled=True,
        minimum_expert_bytes=10,
    )
    cache = ResidentCache(capacity_bytes=100)
    cache.admit(_expert(ExpertKey(0, 0), 50))
    cache.admit(_expert(ExpertKey(0, 1), 50))

    assert manager.enforce(cache).action == "disable_prefetch"
    evicted = manager.enforce(cache)
    assert evicted.action == "evict"
    assert evicted.resident_after_bytes == 0
    shrunk = manager.enforce(cache)
    assert shrunk.action == "shrink"
    assert cache.capacity_bytes == 1
    with pytest.raises(MemoryPressureError, match="persisted"):
        manager.enforce(cache)
    assert [event.action for event in manager.events()] == [
        "disable_prefetch",
        "evict",
        "shrink",
        "reject",
    ]
