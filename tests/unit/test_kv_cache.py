from __future__ import annotations

from mlx_moe_stream.kv_cache import KvCacheConfig, estimate_bf16_kv_bytes, resolve_kv_cache
from mlx_moe_stream.memory import MemoryBudgetConfig, MemorySnapshot


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
