from __future__ import annotations

import logging

from mlx_moe_stream import memory as memory_module
from mlx_moe_stream.kv_cache import (
    KvCacheConfig,
    estimate_bf16_kv_bytes,
    make_memory_manager,
    memory_config_with_kv_reserve,
    resolve_kv_cache,
)
from mlx_moe_stream.memory import MemoryBudgetConfig, MemoryBudgetManager, MemorySnapshot


def _snapshot(*, gib: int = 10) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=0.0,
        physical_memory_bytes=gib * 1024**3,
        recommended_working_set_bytes=gib * 1024**3,
        mlx_active_memory_bytes=0,
        mlx_cache_memory_bytes=0,
        mlx_peak_memory_bytes=0,
        process_rss_bytes=0,
        swap_total_bytes=None,
        swap_used_bytes=None,
        swap_free_bytes=None,
        device_name="test",
    )


def _attention_config(*, layers: int = 30) -> dict[str, object]:
    return {
        "text_config": {
            "num_hidden_layers": layers,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 256,
        }
    }


def _budget() -> MemoryBudgetConfig:
    return MemoryBudgetConfig(
        safety_margin_bytes=1024**3,
        kv_reserve_bytes=64 * 1024**2,
        scratch_reserve_bytes=1024**3,
    )


def test_kv_estimate_excludes_qwen_linear_attention_layers():
    config = {
        "text_config": {
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2,
        }
    }

    assert estimate_bf16_kv_bytes(config, 100) == 6_400


def test_kv_auto_uses_bf16_when_the_estimate_fits():
    decision = resolve_kv_cache(
        KvCacheConfig(mode="auto", max_context_tokens=4_096),
        model_config=_attention_config(),
        shell_bytes=1024**3,
        memory_config=_budget(),
        snapshot=_snapshot(),
    )

    assert decision.effective_mode == "bf16"
    assert decision.kv_bits is None
    assert decision.reserve_bytes == decision.estimated_bf16_bytes


def test_kv_auto_uses_8bit_when_bf16_is_under_pressure():
    decision = resolve_kv_cache(
        KvCacheConfig(mode="auto", max_context_tokens=12_000),
        model_config=_attention_config(),
        shell_bytes=1024**3,
        memory_config=_budget(),
        snapshot=_snapshot(),
    )

    assert decision.effective_mode == "8bit"
    assert decision.kv_bits == 8
    assert decision.estimated_bytes < decision.estimated_bf16_bytes


def test_kv_auto_uses_4bit_for_oversized_contexts():
    decision = resolve_kv_cache(
        KvCacheConfig(mode="auto", max_context_tokens=100_000),
        model_config=_attention_config(),
        shell_bytes=1024**3,
        memory_config=_budget(),
        snapshot=_snapshot(),
    )

    assert decision.effective_mode == "4bit"
    assert decision.kv_bits == 4


def test_explicit_kv_mode_overrides_the_auto_policy():
    decision = resolve_kv_cache(
        KvCacheConfig(mode="bf16", max_context_tokens=100_000),
        model_config=_attention_config(),
        shell_bytes=1024**3,
        memory_config=_budget(),
        snapshot=_snapshot(),
    )

    assert decision.effective_mode == "bf16"
    assert decision.kv_bits is None
    assert decision.reason == "explicit CLI selection"


def _explicit_working_set_snapshot() -> MemorySnapshot:
    """A Mac mini-shaped snapshot where the OS recommendation undershoots physical."""

    return MemorySnapshot(
        timestamp=0.0,
        physical_memory_bytes=16 * 1024**3,
        recommended_working_set_bytes=12 * 1024**3,
        mlx_active_memory_bytes=0,
        mlx_cache_memory_bytes=0,
        mlx_peak_memory_bytes=0,
        process_rss_bytes=0,
        swap_total_bytes=None,
        swap_used_bytes=None,
        swap_free_bytes=None,
        device_name="test",
    )


# --- memory_config_with_kv_reserve() must preserve explicit_working_set_bytes ---
# Regression coverage for a defect where memory_config_with_kv_reserve() rebuilt
# MemoryBudgetConfig from scratch (dropping every field it did not explicitly
# copy) instead of dataclasses.replace()-ing the caller's config, silently
# discarding --max-unified-memory before the M7 budget was ever planned.


def test_memory_config_with_kv_reserve_preserves_explicit_working_set_bytes():
    config = MemoryBudgetConfig(
        safety_margin_bytes=2 * 1024**3,
        kv_reserve_bytes=1 * 1024**3,
        scratch_reserve_bytes=1 * 1024**3,
        wired_limit_bytes=8 * 1024**3,
        explicit_working_set_bytes=14 * 1024**3,
    )
    decision = resolve_kv_cache(
        KvCacheConfig(mode="bf16", max_context_tokens=100_000),
        model_config=_attention_config(),
        shell_bytes=1024**3,
        memory_config=config,
        snapshot=_explicit_working_set_snapshot(),
    )

    wrapped = memory_config_with_kv_reserve(config, decision)

    assert wrapped.explicit_working_set_bytes == 14 * 1024**3
    assert wrapped.safety_margin_bytes == config.safety_margin_bytes
    assert wrapped.scratch_reserve_bytes == config.scratch_reserve_bytes
    assert wrapped.wired_limit_bytes == config.wired_limit_bytes
    assert wrapped.kv_reserve_bytes == decision.reserve_bytes


def test_explicit_working_set_survives_kv_reserve_wrap_and_plan(caplog):
    """The exact bug path: resolve_kv_cache() -> memory_config_with_kv_reserve()
    -> MemoryBudgetManager.plan() must still see source="explicit" and the
    --max-unified-memory byte value, and must still warn because 14GiB exceeds
    the 12GiB OS recommendation in _explicit_working_set_snapshot().
    """

    memory_config = MemoryBudgetConfig(
        safety_margin_bytes=1 * 1024**3,
        kv_reserve_bytes=64 * 1024**2,
        scratch_reserve_bytes=1 * 1024**3,
        explicit_working_set_bytes=14 * 1024**3,
    )
    snapshot = _explicit_working_set_snapshot()
    decision = resolve_kv_cache(
        KvCacheConfig(mode="bf16", max_context_tokens=4_096),
        model_config=_attention_config(),
        shell_bytes=1 * 1024**3,
        memory_config=memory_config,
        snapshot=snapshot,
    )
    wrapped_config = memory_config_with_kv_reserve(memory_config, decision)
    manager = MemoryBudgetManager(wrapped_config, snapshot_provider=lambda: snapshot)

    with caplog.at_level(logging.WARNING, logger="mlx_moe_stream.memory"):
        budget_decision = manager.plan(
            shell_bytes=1 * 1024**3,
            requested_expert_budget_bytes=None,
            auto_enabled=True,
            minimum_expert_bytes=10,
        )

    assert budget_decision.working_set_source == "explicit"
    assert budget_decision.working_set_bytes == 14 * 1024**3
    assert any(
        "exceeds the OS-recommended working set" in record.message for record in caplog.records
    )


def test_make_memory_manager_preserves_explicit_working_set(monkeypatch, caplog):
    """Same regression, through the actual make_memory_manager() production path."""

    snapshot = _explicit_working_set_snapshot()
    monkeypatch.setattr(memory_module, "collect_memory_snapshot", lambda **_: snapshot)
    memory_config = MemoryBudgetConfig(
        safety_margin_bytes=1 * 1024**3,
        kv_reserve_bytes=64 * 1024**2,
        scratch_reserve_bytes=1 * 1024**3,
        explicit_working_set_bytes=14 * 1024**3,
    )

    manager, decision = make_memory_manager(
        memory_config,
        KvCacheConfig(mode="bf16", max_context_tokens=4_096),
        model_config=_attention_config(),
        shell_bytes=1 * 1024**3,
    )
    assert manager.config.explicit_working_set_bytes == 14 * 1024**3

    with caplog.at_level(logging.WARNING, logger="mlx_moe_stream.memory"):
        budget_decision = manager.plan(
            shell_bytes=1 * 1024**3,
            requested_expert_budget_bytes=None,
            auto_enabled=True,
            minimum_expert_bytes=10,
        )

    assert budget_decision.working_set_source == "explicit"
    assert budget_decision.working_set_bytes == 14 * 1024**3
    assert decision.reserve_bytes == manager.config.kv_reserve_bytes
