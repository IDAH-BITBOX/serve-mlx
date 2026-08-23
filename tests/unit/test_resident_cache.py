from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_moe_stream.cache import ExpertKey, MemoryBudgetError, ResidentCache, ResidentExpert
from mlx_moe_stream.runtime import CachedExpertRuntime
from mlx_moe_stream.storage import build_qwen3_moe_manifest


def _expert(key: ExpertKey, nbytes: int = 2) -> ResidentExpert:
    return ResidentExpert(key=key, arrays={}, nbytes=nbytes, last_used_step=0)


def test_global_byte_lru_evicts_least_recent_unpinned_expert():
    cache = ResidentCache(capacity_bytes=4)
    first = ExpertKey(0, 0)
    second = ExpertKey(0, 1)
    third = ExpertKey(1, 0)
    cache.admit(_expert(first))
    cache.admit(_expert(second))
    assert cache.get(first, nbytes=2) is not None

    cache.admit(_expert(third))

    assert cache.get(second, nbytes=2) is None
    stats = cache.stats()
    assert stats.resident_bytes == 4
    assert stats.eviction_count == 1
    assert stats.per_layer_hits == {0: 1}
    assert stats.per_layer_misses == {0: 1}


def test_pinned_entries_cannot_be_evicted_and_oversubscription_is_explicit():
    cache = ResidentCache(capacity_bytes=4)
    first = ExpertKey(0, 0)
    second = ExpertKey(0, 1)
    cache.admit(_expert(first))
    cache.admit(_expert(second))
    cache.pin(first)
    cache.pin(second)

    with pytest.raises(MemoryBudgetError, match="pinned"):
        cache.reserve(1)
    with pytest.raises(MemoryBudgetError, match="exceeds"):
        cache.reserve(5)

    cache.unpin(second)
    cache.reserve(2)
    assert cache.get(second, nbytes=2) is None
    assert cache.get(first, nbytes=2) is not None


def test_resize_and_aggressive_eviction_preserve_pin_safety():
    cache = ResidentCache(capacity_bytes=6)
    first = ExpertKey(0, 0)
    second = ExpertKey(0, 1)
    third = ExpertKey(0, 2)
    cache.admit(_expert(first))
    cache.admit(_expert(second))
    cache.admit(_expert(third))
    cache.pin(first)

    cache.evict_to(2)
    assert cache.stats().resident_bytes == 2
    assert cache.contains(first)
    cache.unpin(first)
    cache.resize(1)

    assert cache.capacity_bytes == 1
    assert cache.stats().resident_bytes == 0


def _write_tiny_quantized_model(path: Path) -> None:
    path.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, np.ndarray] = {}
    for projection, shape in (
        ("gate_proj", (2, 3)),
        ("up_proj", (2, 3)),
        ("down_proj", (3, 2)),
    ):
        for field, dtype in (
            ("weight", np.uint32),
            ("scales", np.float16),
            ("biases", np.float16),
        ):
            name = f"model.layers.0.mlp.switch_mlp.{projection}.{field}"
            tensors[name] = np.arange(np.prod((1, *shape)), dtype=dtype).reshape((1, *shape))
    save_file(tensors, path / "model.safetensors")


def test_cached_runtime_reads_a_bundle_once_then_reuses_its_mlx_arrays(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_tiny_quantized_model(model_path)
    manifest = build_qwen3_moe_manifest(model_path)
    key = ExpertKey(0, 0)
    bundle = manifest.expert_bundles[key]
    runtime = CachedExpertRuntime(manifest, capacity_bytes=bundle.total_bytes)
    try:
        first = runtime.resolve(0, 0)
        second = runtime.resolve(0, 0)
        stats = runtime.stats()
    finally:
        runtime.close()

    assert first.arrays is second.arrays
    assert stats.expert_resolutions == 2
    assert stats.bytes_read == bundle.total_bytes
    assert stats.read_count == len(bundle.tensors)
    assert stats.cache is not None
    assert stats.cache.hit_count == 1
    assert stats.cache.miss_count == 1
    assert stats.cache.resident_bytes == bundle.total_bytes


def test_cached_runtime_rejects_an_expert_larger_than_its_budget_before_io(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_tiny_quantized_model(model_path)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle = manifest.expert_bundles[ExpertKey(0, 0)]
    runtime = CachedExpertRuntime(manifest, capacity_bytes=bundle.total_bytes - 1)
    try:
        with pytest.raises(MemoryBudgetError, match="exceeds"):
            runtime.resolve(0, 0)
        stats = runtime.stats()
    finally:
        runtime.close()

    assert stats.bytes_read == 0
    assert stats.cache is not None
    assert stats.cache.miss_count == 1
