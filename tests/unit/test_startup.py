"""StartupResult must carry shell_bytes so a startup report can print the
shell/expert/KV split without reaching into memory_budget."""

from __future__ import annotations

import json
from pathlib import Path

import mlx_moe_stream.startup as startup_module
import numpy as np
import pytest
from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.hardware import HardwareProfile
from mlx_moe_stream.manifest import (
    ExpertBundleSpec,
    ExpertWorkingSet,
    ModelManifest,
    QuantizationSpec,
    TensorSpan,
)
from mlx_moe_stream.memory import MemoryBudgetDecision, MemorySnapshot
from mlx_moe_stream.runtime import CachedExpertRuntime, NoCacheExpertRuntime
from mlx_moe_stream.startup import (
    StartupAdvisorInput,
    StartupResult,
    decide_startup,
    prepare_streaming_runtime,
)
from mlx_moe_stream.storage import build_qwen3_moe_manifest
from safetensors.numpy import save_file


def _bundle(layer: int, expert: int, nbytes: int) -> ExpertBundleSpec:
    span = TensorSpan(
        file=Path("/nonexistent/model.safetensors"),
        tensor_name=f"layer.{layer}.expert.{expert}.weight",
        offset=0,
        nbytes=nbytes,
        shape=(nbytes,),
        dtype="uint8",
        role="weight",
    )
    key = ExpertKey(layer=layer, expert=expert)
    return ExpertBundleSpec(
        key=key, tensors=(span,), total_bytes=nbytes, quantization=QuantizationSpec()
    )


def _manifest(num_layers: int = 1, num_experts: int = 2, bundle_bytes: int = 100) -> ModelManifest:
    bundles = {}
    for layer in range(num_layers):
        for expert in range(num_experts):
            bundles[ExpertKey(layer, expert)] = _bundle(layer, expert, bundle_bytes)
    return ModelManifest(
        format_version=1,
        model_type="qwen3_moe",
        source_model="synthetic",
        source_model_path=Path("/nonexistent"),
        num_layers=num_layers,
        num_experts=num_experts,
        experts_per_token=1,
        quantization=QuantizationSpec(),
        non_expert_weight_files=(),
        expert_bundles=bundles,
    )


def _model_config(num_layers: int = 1) -> dict[str, object]:
    return {
        "text_config": {
            "num_hidden_layers": num_layers,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 256,
        }
    }


def test_startup_result_carries_shell_bytes_matching_the_measured_input():
    manifest = _manifest()
    shell_bytes = 123_456_789

    startup = prepare_streaming_runtime(
        manifest,
        shell_bytes=shell_bytes,
        model_config=_model_config(),
        resident_budget_bytes=None,
        auto_resident_budget=False,
        memory_config=None,
        kv_cache_config=None,
        runtime_options={},
    )
    try:
        assert isinstance(startup, StartupResult)
        assert startup.shell_bytes == shell_bytes
        # StartupResult.shell_bytes must never drift from the value the M7
        # budget itself was planned against -- a caller printing a startup
        # report can use either field and get the same number.
        assert startup.shell_bytes == startup.memory_budget.shell_bytes
        assert isinstance(startup.runtime, NoCacheExpertRuntime)
    finally:
        startup.runtime.close()


# --- M13 Node 7: decide_startup's three-mode branch, pinned by an exact byte
# inequality only (never an estimate). ------------------------------------


def _hardware(
    *, physical: int = 17_179_869_184, recommended: int = 12_713_115_648, wired_limit_mb: int = 0
) -> HardwareProfile:
    return HardwareProfile(
        device_name="Apple M4",
        physical_memory_bytes=physical,
        recommended_working_set_bytes=recommended,
        gpu_core_count=10,
        cpu_performance_cores=4,
        cpu_efficiency_cores=6,
        wired_limit_mb=wired_limit_mb,
        disk_free_bytes=None,
        disk_total_bytes=None,
        vm_page_size_bytes=None,
        compressor_pages_stored=None,
        compressor_pages_occupied=None,
        compressor_compressions=None,
        compressor_decompressions=None,
    )


def _working_set(*, total_bytes: int, bundle_count: int = 4) -> ExpertWorkingSet:
    mean = total_bytes / bundle_count
    return ExpertWorkingSet(
        total_bytes=total_bytes,
        bundle_count=bundle_count,
        mean_bundle_bytes=mean,
        min_bundle_bytes=int(mean),
        max_bundle_bytes=int(mean),
        per_token_full_miss_bytes=mean * 2,
    )


def _budget(
    *, expert_budget_bytes: int | None, snapshot: MemorySnapshot | None = None
) -> MemoryBudgetDecision:
    snapshot = snapshot or MemorySnapshot(
        timestamp=1.0,
        physical_memory_bytes=17_179_869_184,
        recommended_working_set_bytes=12_713_115_648,
        mlx_active_memory_bytes=0,
        mlx_cache_memory_bytes=0,
        mlx_peak_memory_bytes=0,
        process_rss_bytes=0,
        swap_total_bytes=None,
        swap_used_bytes=None,
        swap_free_bytes=None,
        device_name="Apple M4",
    )
    return MemoryBudgetDecision(
        source="auto" if expert_budget_bytes is not None else "disabled",
        expert_budget_bytes=expert_budget_bytes,
        safe_working_set_bytes=8_713_115_648,
        shell_bytes=3_496_275_176,
        available_expert_bytes=expert_budget_bytes or 0,
        snapshot=snapshot,
        recommended_working_set_bytes=12_713_115_648,
        safety_margin_bytes=4_294_967_296,
        kv_reserve_bytes=1_392_640_000,
        scratch_reserve_bytes=2_147_483_648,
        working_set_bytes=12_713_115_648,
        working_set_source="recommended",
    )


def _advisor_input() -> StartupAdvisorInput:
    return StartupAdvisorInput(
        bandwidth_bytes_per_sec=2.0e9, bandwidth_source="default", current_quantization_bits=8
    )


def test_decide_startup_picks_no_cache_when_budget_is_disabled():
    decision = decide_startup(
        _hardware(),
        _working_set(total_bytes=8_000_000_000),
        _budget(expert_budget_bytes=None),
        _advisor_input(),
    )
    assert decision.mode == "no_cache"
    assert decision.warmup_keys == ()
    assert decision.report["mode"] == "no_cache"


def test_decide_startup_picks_streaming_when_budget_is_smaller_than_working_set():
    decision = decide_startup(
        _hardware(),
        _working_set(total_bytes=8_000_000_000),
        _budget(expert_budget_bytes=1_381_749_528),
        _advisor_input(),
        expert_keys=(ExpertKey(0, 0), ExpertKey(0, 1)),
    )
    assert decision.mode == "streaming"
    # Streaming never warms: a partial LRU cache would just be evicted.
    assert decision.warmup_keys == ()
    assert decision.report["resident_fraction"] == pytest.approx(1_381_749_528 / 8_000_000_000)
    assert decision.report["hit_rate_method"] == "power_law"


def test_decide_startup_picks_full_residency_at_the_exact_boundary():
    """The branch is `>=`, not `>` -- an exact-fit budget must still warm."""

    working_set = _working_set(total_bytes=8_000_000_000)
    keys = (ExpertKey(0, 0), ExpertKey(0, 1), ExpertKey(1, 0), ExpertKey(1, 1))
    decision = decide_startup(
        _hardware(),
        working_set,
        _budget(expert_budget_bytes=working_set.total_bytes),
        _advisor_input(),
        expert_keys=keys,
    )
    assert decision.mode == "full_residency"
    assert set(decision.warmup_keys) == set(keys)
    assert decision.report["hit_rate"] == 1.0
    assert decision.report["hit_rate_method"] == "exact_full_residency"


def test_decide_startup_full_residency_when_budget_exceeds_working_set():
    """The mac studio case: 208GiB available vs. 31.87GiB expert working set."""

    working_set = _working_set(total_bytes=34_225_520_640)  # 31.87GiB
    decision = decide_startup(
        _hardware(physical=256 * 1024**3, recommended=239_143_780_352),
        working_set,
        _budget(expert_budget_bytes=208 * 1024**3),
        _advisor_input(),
        expert_keys=tuple(ExpertKey(0, e) for e in range(4)),
    )
    assert decision.mode == "full_residency"
    assert len(decision.warmup_keys) == 4


def test_decide_startup_report_is_json_serializable():
    decision = decide_startup(
        _hardware(),
        _working_set(total_bytes=8_000_000_000),
        _budget(expert_budget_bytes=1_381_749_528),
        _advisor_input(),
    )
    json.dumps(decision.report)  # must not raise


def test_decide_startup_full_residency_report_is_strictly_json_serializable():
    """The streaming case above never exercises hit_rate == 1.0, so it never
    caught decode_ceiling_tps() returning inf for a fully-resident cache --
    that inf used to reach metrics_snapshot()["startup"] and --startup-report
    verbatim, and inf is not valid JSON (allow_nan=True, json.dumps's
    default, silently emits the non-standard Infinity token instead of
    raising). Pin both: the report must serialize even with allow_nan=False,
    and decode_ceiling_tps itself must already be the clamped None, not inf.
    """

    working_set = _working_set(total_bytes=8_000_000_000)
    decision = decide_startup(
        _hardware(),
        working_set,
        _budget(expert_budget_bytes=working_set.total_bytes),  # exact fit -> full_residency
        _advisor_input(),
        expert_keys=(ExpertKey(0, 0), ExpertKey(0, 1)),
    )
    assert decision.mode == "full_residency"
    assert decision.report["decode_ceiling_tps"] is None
    serialized = json.dumps(decision.report, allow_nan=False)  # must not raise
    assert "Infinity" not in serialized


# --- prepare_streaming_runtime end-to-end mode wiring (real MLX, tiny model) -----


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
    tensors: dict[str, object] = {}
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
                tensors.setdefault(name, np.zeros((num_experts, *shape), dtype=dtype))[
                    expert
                ] = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + expert
    save_file(tensors, path / "model.safetensors")


def test_prepare_streaming_runtime_warms_on_full_residency(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_quantized_model(model_path, num_experts=4)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes

    startup = prepare_streaming_runtime(
        manifest,
        shell_bytes=0,
        model_config=_model_config(1),
        resident_budget_bytes=bundle_bytes * 4,  # exactly the whole working set
        auto_resident_budget=False,
        memory_config=None,
        kv_cache_config=None,
        runtime_options={},
        startup_io_probe="off",
        warmup="auto",
    )
    try:
        assert startup.decision.mode == "full_residency"
        stats = startup.runtime.stats()
        assert stats.warmup is not None
        assert stats.warmup.admitted == 4
        assert stats.cache.hit_count == 0 and stats.cache.miss_count == 0
    finally:
        startup.runtime.close()


def test_prepare_streaming_runtime_skips_warmup_when_streaming(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_quantized_model(model_path, num_experts=4)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes

    startup = prepare_streaming_runtime(
        manifest,
        shell_bytes=0,
        model_config=_model_config(1),
        resident_budget_bytes=bundle_bytes * 2,  # smaller than the 4-bundle working set
        auto_resident_budget=False,
        memory_config=None,
        kv_cache_config=None,
        runtime_options={},
        startup_io_probe="off",
        warmup="auto",
    )
    try:
        assert startup.decision.mode == "streaming"
        stats = startup.runtime.stats()
        assert stats.warmup is None
        assert stats.cache.resident_bytes == 0
    finally:
        startup.runtime.close()


def test_prepare_streaming_runtime_warmup_off_never_warms_even_at_full_residency(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_quantized_model(model_path, num_experts=4)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes

    startup = prepare_streaming_runtime(
        manifest,
        shell_bytes=0,
        model_config=_model_config(1),
        resident_budget_bytes=bundle_bytes * 4,
        auto_resident_budget=False,
        memory_config=None,
        kv_cache_config=None,
        runtime_options={},
        startup_io_probe="off",
        warmup="off",
    )
    try:
        assert startup.decision.mode == "full_residency"
        assert startup.runtime.stats().warmup is None
        assert startup.runtime.stats().cache.resident_bytes == 0
    finally:
        startup.runtime.close()

# --- N7: the M11 advisor must never block engine startup -------------------


def test_prepare_streaming_runtime_degrades_when_the_advisor_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """manifest.expert_working_set(), probe_hardware(), and decide_startup()
    were previously called with no guard at all in _decide_and_act: any
    ordinary exception from any of them (a hardware probe failure, a bad
    manifest, an out-of-range fraction) used to fail engine startup even
    though the M11 advisor is purely advisory/log-only. It must instead
    degrade to a minimal decision and let the engine load."""

    model_path = tmp_path / "model"
    _write_quantized_model(model_path, num_experts=4)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated hardware probe failure")

    monkeypatch.setattr(startup_module, "probe_hardware", _raise)

    startup = prepare_streaming_runtime(
        manifest,
        shell_bytes=0,
        model_config=_model_config(1),
        resident_budget_bytes=bundle_bytes * 2,  # smaller than the 4-bundle working set
        auto_resident_budget=False,
        memory_config=None,
        kv_cache_config=None,
        runtime_options={},
        startup_io_probe="off",
        warmup="auto",
    )
    try:
        # The M7 budget itself (already planned before the advisor ever
        # runs) still says this is a partial cache -> streaming, never
        # full_residency (we no longer trust the inputs enough to warm the
        # whole working set) and never a crash.
        assert startup.decision.mode == "streaming"
        assert startup.decision.warmup_keys == ()
        assert "advisor_error" in startup.decision.report
        # The engine still loaded: the runtime is real and usable, not a
        # casualty of the advisor failure.
        assert startup.runtime.stats().cache.resident_bytes == 0
    finally:
        startup.runtime.close()


def test_prepare_streaming_runtime_closes_the_runtime_when_planning_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """_decide_and_act only catches plain Exception. Anything it does not
    catch must still close the SafetensorsExpertStore mmap and I/O worker
    threads prepare_streaming_runtime already created for its runtime before
    propagating -- otherwise a bug (or a deliberately unusual failure) in
    the M13 planning step leaks them."""

    model_path = tmp_path / "model"
    _write_quantized_model(model_path, num_experts=4)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle_bytes = next(iter(manifest.expert_bundles.values())).total_bytes

    class _Boom(BaseException):
        """Deliberately not an Exception subclass, so it is guaranteed to
        skip _decide_and_act's except-Exception degrade path and reach
        prepare_streaming_runtime's own close-then-reraise guard."""

    def _raise(*args, **kwargs):
        raise _Boom("simulated non-Exception planning failure")

    monkeypatch.setattr(startup_module, "probe_hardware", _raise)

    close_calls: list[CachedExpertRuntime] = []
    original_close = CachedExpertRuntime.close

    def _tracking_close(self: CachedExpertRuntime) -> None:
        close_calls.append(self)
        original_close(self)

    monkeypatch.setattr(CachedExpertRuntime, "close", _tracking_close)

    with pytest.raises(_Boom):
        prepare_streaming_runtime(
            manifest,
            shell_bytes=0,
            model_config=_model_config(1),
            resident_budget_bytes=bundle_bytes * 4,  # full residency -> CachedExpertRuntime
            auto_resident_budget=False,
            memory_config=None,
            kv_cache_config=None,
            runtime_options={},
            startup_io_probe="off",
            warmup="auto",
        )

    assert len(close_calls) == 1
