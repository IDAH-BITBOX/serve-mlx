from __future__ import annotations

import logging

import pytest
from mlx_moe_stream.memory import (
    MemoryBudgetConfig,
    MemoryBudgetManager,
    MemorySnapshot,
    adaptive_safety_margin_bytes,
)

_GIB = 1024**3


def _snapshot(*, physical: int = 1_000, recommended: int = 1_000) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=1.0,
        physical_memory_bytes=physical,
        recommended_working_set_bytes=recommended,
        mlx_active_memory_bytes=0,
        mlx_cache_memory_bytes=0,
        mlx_peak_memory_bytes=0,
        process_rss_bytes=0,
        swap_total_bytes=0,
        swap_used_bytes=0,
        swap_free_bytes=0,
        device_name="test",
    )


# --- explain() -----------------------------------------------------------


def test_explain_contains_every_budget_term_and_the_final_available_bytes():
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(safety_margin_bytes=100, kv_reserve_bytes=50, scratch_reserve_bytes=25),
        snapshot_provider=lambda: _snapshot(physical=1_000, recommended=1_000),
    )
    decision = manager.plan(
        shell_bytes=100,
        requested_expert_budget_bytes=None,
        auto_enabled=True,
        minimum_expert_bytes=10,
    )

    text = decision.explain()
    for term in (
        "recommended_working_set",
        "shell",
        "safety_margin",
        "kv_reserve",
        "scratch_reserve",
        "available_expert",
        "working_set",
    ):
        assert term in text

    # The five terms must sum (as raw bytes, independent of the rendered
    # string) to exactly available_expert_bytes when no explicit override
    # is applied -- working_set falls back to recommended_working_set here.
    computed = (
        decision.recommended_working_set_bytes
        - decision.safety_margin_bytes
        - decision.shell_bytes
        - decision.kv_reserve_bytes
        - decision.scratch_reserve_bytes
    )
    assert computed == decision.available_expert_bytes
    assert decision.working_set_source == "recommended"


# --- adaptive_safety_margin_bytes ----------------------------------------


def test_adaptive_margin_matches_probed_mac_mini_m4_16gb():
    margin = adaptive_safety_margin_bytes(16 * _GIB, 12_713_115_648)
    assert margin == 1 * _GIB  # OS already withheld more than a quarter; hits the 1GiB floor


def test_adaptive_margin_hits_the_8gib_cap_on_a_256gib_mac_studio():
    # NOTE: the requesting spec's completion table says "256/222.7GiB -> 1GiB"
    # for this case. Applying the literal formula
    # clamp(physical//4 - (physical - recommended), 1GiB, 8GiB) as specified
    # gives 8GiB here, not 1GiB: physical//4 = 64GiB, and the OS withheld only
    # ~33.3GiB (256 - 222.7GiB) which is far *less* than a quarter of 256GiB,
    # so the pre-clamp candidate is ~30.7GiB and the formula saturates at its
    # own 8GiB upper bound -- the same 8GiB the existing (non-adaptive)
    # automatic_safety_margin_bytes already returns for this machine, since a
    # quarter of 256GiB is also capped to 8GiB there. This is documented as a
    # deliberate deviation from the literal completion-table value; see the
    # final task report.
    margin = adaptive_safety_margin_bytes(256 * _GIB, 239_143_780_352)
    assert margin == 8 * _GIB


def test_adaptive_margin_reserves_a_quarter_when_there_is_no_os_withholding_gap():
    margin = adaptive_safety_margin_bytes(16 * _GIB, 16 * _GIB)
    assert margin == 4 * _GIB


def test_adaptive_margin_requires_known_physical_memory():
    with pytest.raises(ValueError, match="physical memory"):
        adaptive_safety_margin_bytes(0, 0)


# --- explicit --max-unified-memory override in plan() --------------------


def test_explicit_working_set_can_exceed_the_os_recommendation():
    # A Mac mini's recommended_working_set (11.84GiB) already double-deducts
    # against physical (16GiB); --max-unified-memory lets an operator raise
    # the working set explicitly, up to physical memory, to explicitly
    # resolve that double deduction.
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(
            safety_margin_bytes=1 * _GIB,
            kv_reserve_bytes=0,
            scratch_reserve_bytes=0,
            explicit_working_set_bytes=14 * _GIB,
        ),
        snapshot_provider=lambda: _snapshot(physical=16 * _GIB, recommended=12_713_115_648),
    )
    decision = manager.plan(
        shell_bytes=0,
        requested_expert_budget_bytes=None,
        auto_enabled=True,
        minimum_expert_bytes=10,
    )
    assert decision.working_set_bytes == 14 * _GIB
    assert decision.working_set_source == "explicit"
    assert decision.available_expert_bytes == 13 * _GIB
    # Bigger than the ~10.8GiB the unadjusted OS recommendation would allow
    # with this same 1GiB safety margin (12_713_115_648 - 1GiB = ~10.84GiB).
    assert decision.available_expert_bytes > decision.recommended_working_set_bytes - (1 * _GIB)


def test_explicit_working_set_rejects_values_above_physical_memory():
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(explicit_working_set_bytes=17 * _GIB),
        snapshot_provider=lambda: _snapshot(physical=16 * _GIB, recommended=12_713_115_648),
    )
    with pytest.raises(ValueError, match="exceeds physical memory"):
        manager.plan(
            shell_bytes=0,
            requested_expert_budget_bytes=None,
            auto_enabled=True,
            minimum_expert_bytes=10,
        )


def test_explicit_working_set_above_recommended_warns_but_still_applies(caplog):
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(explicit_working_set_bytes=14 * _GIB),
        snapshot_provider=lambda: _snapshot(physical=16 * _GIB, recommended=12_713_115_648),
    )
    with caplog.at_level(logging.WARNING, logger="mlx_moe_stream.memory"):
        decision = manager.plan(
            shell_bytes=0,
            requested_expert_budget_bytes=None,
            auto_enabled=True,
            minimum_expert_bytes=10,
        )
    assert decision.working_set_bytes == 14 * _GIB
    assert any(
        "exceeds the OS-recommended working set" in record.message for record in caplog.records
    )


def test_default_behavior_is_unchanged_when_max_unified_memory_is_not_set():
    manager = MemoryBudgetManager(
        MemoryBudgetConfig(
            safety_margin_bytes=100, kv_reserve_bytes=100, scratch_reserve_bytes=100
        ),
        snapshot_provider=lambda: _snapshot(physical=1_000, recommended=1_000),
    )
    decision = manager.plan(
        shell_bytes=100,
        requested_expert_budget_bytes=None,
        auto_enabled=True,
        minimum_expert_bytes=10,
    )
    assert decision.working_set_source == "recommended"
    assert decision.working_set_bytes == 1_000
    assert decision.available_expert_bytes == 600
