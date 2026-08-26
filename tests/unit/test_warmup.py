"""M13 Node 6: CachedExpertRuntime.warmup() must preload the resident cache
directly (no cache.get(), no _expert_resolutions) in ascending (file, offset)
order, stopping before it would ever have to evict, honoring a deadline and
a live active-memory ceiling, and never letting a reader failure escape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.runtime import CachedExpertRuntime, NoCacheExpertRuntime, WarmupStats
from mlx_moe_stream.storage import build_qwen3_moe_manifest
from safetensors.numpy import save_file


def _write_quantized_model(path: Path, *, num_experts: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "num_experts": num_experts,
        "num_experts_per_tok": 1,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, np.ndarray] = {}
    for expert in range(num_experts):
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
                # safetensors keys must be unique per file; MoE experts are
                # stacked along a leading axis in the real Qwen3 layout, so
                # give each expert its own slice of a stacked tensor.
                tensors.setdefault(
                    name, np.zeros((num_experts, *shape), dtype=dtype)
                )[expert] = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + expert
    save_file(tensors, path / "model.safetensors")


def _manifest(tmp_path: Path, *, num_experts: int = 4):
    model_path = tmp_path / "model"
    _write_quantized_model(model_path, num_experts=num_experts)
    return build_qwen3_moe_manifest(model_path)


def test_warmup_admits_exactly_floor_capacity_over_bundle_with_no_eviction(tmp_path: Path):
    manifest = _manifest(tmp_path, num_experts=4)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes
    # Capacity for exactly 2 of the 4 uniform bundles.
    capacity = bundle_bytes * 2 + 1
    runtime = CachedExpertRuntime(manifest, capacity_bytes=capacity)
    try:
        keys = list(manifest.expert_bundles.keys())
        stats = runtime.warmup(keys)
        cache_stats = runtime.cache.stats()
        runtime_stats = runtime.stats()
    finally:
        runtime.close()

    assert isinstance(stats, WarmupStats)
    assert stats.admitted == 2
    assert stats.stop_reason == "capacity"
    assert stats.requested == 4
    assert stats.reader_errors == 0
    assert cache_stats.eviction_count == 0
    assert cache_stats.resident_bytes == bundle_bytes * 2

    # Hard rule: hit/miss counters and expert_resolutions must be untouched.
    assert cache_stats.hit_count == 0
    assert cache_stats.miss_count == 0
    assert runtime_stats.expert_resolutions == 0
    assert runtime_stats.warmup is stats


def test_warmup_respects_a_deadline_and_returns_partial_progress(tmp_path: Path):
    manifest = _manifest(tmp_path, num_experts=4)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes
    runtime = CachedExpertRuntime(manifest, capacity_bytes=bundle_bytes * 4)
    try:
        keys = list(manifest.expert_bundles.keys())
        # A deadline already in the past: the loop must stop before doing
        # any admission at all and return normally (no exception).
        stats = runtime.warmup(keys, deadline=0.0)
    finally:
        runtime.close()

    assert stats.stop_reason == "deadline"
    assert stats.admitted == 0
    assert stats.requested == 4


def test_warmup_reader_failure_is_a_warning_not_an_exception(tmp_path: Path, caplog):
    manifest = _manifest(tmp_path, num_experts=4)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes
    runtime = CachedExpertRuntime(manifest, capacity_bytes=bundle_bytes * 4)
    keys = list(manifest.expert_bundles.keys())
    failing_key = keys[0]

    original_read_bundle = runtime._read_bundle

    def flaky_read_bundle(key, bundle):
        if key == failing_key:
            raise OSError("synthetic read failure")
        return original_read_bundle(key, bundle)

    runtime._read_bundle = flaky_read_bundle
    try:
        import logging

        with caplog.at_level(logging.WARNING, logger="mlx_moe_stream.runtime"):
            stats = runtime.warmup(keys)
    finally:
        runtime.close()

    assert stats.reader_errors == 1
    assert stats.admitted == 3
    assert stats.stop_reason == "completed"
    assert any("could not load expert" in record.message for record in caplog.records)


def test_warmup_never_calls_cache_get_or_bumps_resolutions_even_on_reentry(tmp_path: Path):
    """Warming an already-resident key must be a silent skip, not a hit."""

    manifest = _manifest(tmp_path, num_experts=2)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes
    runtime = CachedExpertRuntime(manifest, capacity_bytes=bundle_bytes * 2)
    try:
        keys = list(manifest.expert_bundles.keys())
        runtime.warmup(keys)
        stats_second = runtime.warmup(keys)  # everything already resident
        cache_stats = runtime.cache.stats()
    finally:
        runtime.close()

    assert stats_second.admitted == 0
    assert cache_stats.hit_count == 0
    assert cache_stats.miss_count == 0


def test_no_cache_runtime_warmup_is_a_logged_noop(tmp_path: Path, caplog):
    manifest = _manifest(tmp_path, num_experts=2)
    runtime = NoCacheExpertRuntime(manifest)
    try:
        import logging

        with caplog.at_level(logging.INFO, logger="mlx_moe_stream.runtime"):
            result = runtime.warmup(list(manifest.expert_bundles.keys()))
    finally:
        runtime.close()

    assert result is None
    assert any("warmup skipped" in record.message for record in caplog.records)
    assert runtime.stats().warmup is None


@pytest.mark.parametrize("deadline", [None])
def test_warmup_orders_keys_by_ascending_file_offset(tmp_path: Path, deadline):
    manifest = _manifest(tmp_path, num_experts=4)
    runtime = CachedExpertRuntime(manifest, capacity_bytes=10**9)
    try:
        keys = list(reversed(manifest.expert_bundles.keys()))
        ordered = runtime._warmup_order(keys)

        def offset_of(key: ExpertKey) -> int:
            bundle = manifest.expert_bundles[key]
            return min(t.offset for t in bundle.tensors)

        offsets = [offset_of(key) for key in ordered]
        assert offsets == sorted(offsets)
    finally:
        runtime.close()

# --- N7: the production-shaped code paths test_warmup.py was missing -------


def test_warmup_stops_at_an_active_memory_ceiling(tmp_path: Path):
    """active_memory_ceiling (real MLX get_active_memory(), not a mock) must
    stop admission the moment the ceiling is crossed, before the loop would
    otherwise stop for capacity or completion reasons. Previously untested:
    only "capacity" and "deadline" stop_reason paths had coverage."""

    manifest = _manifest(tmp_path, num_experts=4)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes
    runtime = CachedExpertRuntime(manifest, capacity_bytes=bundle_bytes * 4)
    try:
        keys = list(manifest.expert_bundles.keys())
        # An impossibly tight ceiling: materializing and mx.eval()-ing even
        # the very first bundle already exceeds it, so the loop must stop
        # immediately without admitting anything.
        stats = runtime.warmup(keys, active_memory_ceiling=0)
        cache_stats = runtime.cache.stats()
    finally:
        runtime.close()

    assert stats.stop_reason == "memory_ceiling"
    assert stats.admitted == 0
    assert stats.requested == 4
    assert cache_stats.resident_bytes == 0


def test_warmup_completes_through_the_io_worker_pipeline(tmp_path: Path):
    """The production serve/generate command runs with --io-workers 8
    --prefetch-depth 4 --async-gpu, which routes warmup() through the
    self._loader.prefetch()/demand() pipeline (self._loader is not None)
    instead of the direct single-threaded _read_bundle() path every other
    warmup test here exercises with io_workers=0 (no loader at all)."""

    manifest = _manifest(tmp_path, num_experts=4)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes
    runtime = CachedExpertRuntime(
        manifest,
        capacity_bytes=bundle_bytes * 4,
        io_workers=2,
        prefetch_depth=2,
    )
    try:
        keys = list(manifest.expert_bundles.keys())
        stats = runtime.warmup(keys)
        cache_stats = runtime.cache.stats()
    finally:
        runtime.close()

    assert stats.stop_reason == "completed"
    assert stats.admitted == 4
    assert stats.requested == 4
    assert stats.reader_errors == 0
    assert cache_stats.resident_bytes == bundle_bytes * 4
    # Hard rule still holds through the pipelined path too: warmup never
    # touches hit/miss counters or expert_resolutions.
    assert cache_stats.hit_count == 0
    assert cache_stats.miss_count == 0
