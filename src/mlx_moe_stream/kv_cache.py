"""Bounded KV-cache precision policy for local streamed-MoE serving."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .memory import MemoryBudgetConfig, MemoryBudgetManager, MemorySnapshot

KvCacheMode = Literal["auto", "bf16", "8bit", "4bit"]


@dataclass(frozen=True)
class KvCacheConfig:
    """Requested KV precision and the largest context to reserve for."""

    mode: KvCacheMode = "auto"
    max_context_tokens: int = 4_096
    group_size: int = 64

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "bf16", "8bit", "4bit"}:
            raise ValueError(f"unsupported KV cache mode {self.mode!r}")
        if self.max_context_tokens <= 0:
            raise ValueError("KV cache max context tokens must be greater than zero")
        if self.group_size <= 0:
            raise ValueError("KV cache group size must be greater than zero")


@dataclass(frozen=True)
class KvCacheDecision:
    """Resolved generation kwargs and a conservative Unified Memory reservation."""

    requested_mode: KvCacheMode
    effective_mode: Literal["bf16", "8bit", "4bit"]
    max_context_tokens: int
    kv_bits: int | None
    group_size: int
    estimated_bf16_bytes: int
    estimated_bytes: int
    reserve_bytes: int
    allowance_bytes: int
    reason: str

    def generation_kwargs(self) -> dict[str, int]:
        """Return only non-default MLX generation controls."""

        if self.kv_bits is None:
            return {}
        return {"kv_bits": self.kv_bits, "kv_group_size": self.group_size}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_kv_cache(
    config: KvCacheConfig | None,
    *,
    model_config: dict[str, Any],
    shell_bytes: int,
    memory_config: MemoryBudgetConfig | None,
    snapshot: MemorySnapshot,
) -> KvCacheDecision:
    """Resolve KV precision after measuring the actual non-expert shell.

    The estimate is intentionally conservative. It models keys and values for
    the attention layers that can grow with context, caps sliding-attention
    layers at their declared window, and leaves 75% of post-shell headroom to
    MLX scratch space and streamed experts. The actual MLX cache grows lazily.
    """

    requested = config or KvCacheConfig()
    budget = memory_config or MemoryBudgetConfig()
    bf16_bytes = estimate_bf16_kv_bytes(model_config, requested.max_context_tokens)
    safe_working_set = (
        snapshot.recommended_working_set_bytes or snapshot.physical_memory_bytes
    ) - budget.safety_margin_bytes
    headroom = max(0, safe_working_set - shell_bytes - budget.scratch_reserve_bytes)
    allowance = max(0, headroom // 4)

    estimates = {
        "bf16": bf16_bytes,
        "8bit": _quantized_estimate(bf16_bytes, bits=8, group_size=requested.group_size),
        "4bit": _quantized_estimate(bf16_bytes, bits=4, group_size=requested.group_size),
    }
    if requested.mode == "auto":
        if estimates["bf16"] <= allowance:
            effective = "bf16"
            reason = "BF16 estimate fits the automatic KV allowance"
        elif estimates["8bit"] <= allowance:
            effective = "8bit"
            reason = "BF16 exceeds the automatic KV allowance; selected 8-bit"
        else:
            effective = "4bit"
            reason = "8-bit exceeds the automatic KV allowance; selected 4-bit"
    else:
        effective = requested.mode
        reason = "explicit CLI selection"

    kv_bits = {"bf16": None, "8bit": 8, "4bit": 4}[effective]
    estimated = estimates[effective]
    reserve = max(budget.kv_reserve_bytes, estimated)
    return KvCacheDecision(
        requested_mode=requested.mode,
        effective_mode=effective,
        max_context_tokens=requested.max_context_tokens,
        kv_bits=kv_bits,
        group_size=requested.group_size,
        estimated_bf16_bytes=bf16_bytes,
        estimated_bytes=estimated,
        reserve_bytes=reserve,
        allowance_bytes=allowance,
        reason=reason,
    )


def memory_config_with_kv_reserve(
    config: MemoryBudgetConfig | None, decision: KvCacheDecision
) -> MemoryBudgetConfig:
    """Return M7 settings with enough KV reservation for the selected policy."""

    base = config or MemoryBudgetConfig()
    return dataclasses.replace(base, kv_reserve_bytes=decision.reserve_bytes)


def make_memory_manager(
    memory_config: MemoryBudgetConfig | None,
    kv_cache_config: KvCacheConfig | None,
    *,
    model_config: dict[str, Any],
    shell_bytes: int,
) -> tuple[MemoryBudgetManager, KvCacheDecision]:
    """Create the M7 manager with the resolved KV reservation included."""

    base_manager = MemoryBudgetManager(memory_config)
    decision = resolve_kv_cache(
        kv_cache_config,
        model_config=model_config,
        shell_bytes=shell_bytes,
        memory_config=memory_config,
        snapshot=base_manager.snapshot(),
    )
    return MemoryBudgetManager(memory_config_with_kv_reserve(memory_config, decision)), decision


def estimate_bf16_kv_bytes(model_config: dict[str, Any], max_context_tokens: int) -> int:
    """Conservatively estimate attention KV bytes for one request's max context."""

    text_config = model_config.get("text_config", model_config)
    if not isinstance(text_config, dict):
        raise ValueError("model config must contain a text_config object")
    layers = _positive_int(text_config, "num_hidden_layers")
    heads = _positive_int(
        text_config,
        "num_key_value_heads",
        fallback_key="num_attention_heads",
    )
    head_dim = text_config.get("head_dim")
    if not isinstance(head_dim, int) or head_dim <= 0:
        hidden = _positive_int(text_config, "hidden_size")
        attention_heads = _positive_int(text_config, "num_attention_heads")
        if hidden % attention_heads:
            raise ValueError("hidden_size must divide evenly by num_attention_heads")
        head_dim = hidden // attention_heads

    layer_types = text_config.get("layer_types")
    sliding_window = text_config.get("sliding_window")
    total_tokens = 0
    for layer in range(layers):
        layer_type = ""
        if isinstance(layer_types, list) and layer < len(layer_types):
            layer_type = layer_types[layer]
        if layer_type == "linear_attention":
            # Recurrent state is not a token-by-token attention KV cache.
            continue
        token_count = max_context_tokens
        if layer_type == "sliding_attention" and isinstance(sliding_window, int):
            token_count = min(token_count, max(sliding_window, 0))
        total_tokens += token_count

    # keys + values, [heads, head_dim], BF16 (2 bytes).
    return total_tokens * heads * head_dim * 2 * 2


def _quantized_estimate(bf16_bytes: int, *, bits: int, group_size: int) -> int:
    elements = bf16_bytes // 2
    values = (elements * bits + 7) // 8
    # MLX affine KV caches retain one FP16 scale per quantization group.
    scales = ((elements + group_size - 1) // group_size) * 2
    return values + scales


def _positive_int(config: dict[str, Any], key: str, *, fallback_key: str | None = None) -> int:
    value = config.get(key)
    if value is None and fallback_key is not None:
        value = config.get(fallback_key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"model config requires positive integer {key!r}")
    return value
