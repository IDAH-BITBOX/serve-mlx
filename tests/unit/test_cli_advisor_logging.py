"""The `generate` command must build and log an M11 performance advisory
after M7 planning, using the real advisor.py wiring end to end (with the
expensive model-loading and hardware probes replaced by fakes)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlx_moe_stream import cli
from mlx_moe_stream.advisor import resolve_startup_bandwidth_bytes_per_sec
from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.hardware import HardwareProfile
from mlx_moe_stream.manifest import (
    ExpertBundleSpec,
    ModelManifest,
    QuantizationSpec,
    TensorSpan,
)
from mlx_moe_stream.memory import MemoryBudgetDecision, MemorySnapshot
from mlx_moe_stream.runtime import RuntimeStats
from mlx_moe_stream.startup import StartupAdvisorInput, decide_startup


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


def _manifest() -> ModelManifest:
    # Bundle sizes are picked so total_bytes comfortably exceeds the fake
    # M7 budget below (~1.38GB), landing this scenario in the power-law
    # estimate branch rather than the exact-full-residency branch.
    bundles = {
        ExpertKey(layer, expert): _bundle(layer, expert, 1_000_000_000)
        for layer in range(2)
        for expert in range(4)
    }
    return ModelManifest(
        format_version=1,
        model_type="qwen3_moe",
        source_model="synthetic",
        source_model_path=Path("/nonexistent"),
        num_layers=2,
        num_experts=4,
        experts_per_token=1,
        quantization=QuantizationSpec(bits=8),
        non_expert_weight_files=(),
        expert_bundles=bundles,
    )


def _memory_budget() -> MemoryBudgetDecision:
    snapshot = MemorySnapshot(
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
        source="auto",
        expert_budget_bytes=1_381_749_528,
        safe_working_set_bytes=8_713_115_648,
        shell_bytes=3_496_275_176,
        available_expert_bytes=1_381_749_528,
        snapshot=snapshot,
        recommended_working_set_bytes=12_713_115_648,
        safety_margin_bytes=4_294_967_296,
        kv_reserve_bytes=1_392_640_000,
        scratch_reserve_bytes=2_147_483_648,
        working_set_bytes=12_713_115_648,
        working_set_source="recommended",
    )


class _FakeRuntime:
    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest

    def stats(self) -> RuntimeStats:
        return RuntimeStats(expert_resolutions=0, bytes_read=0, read_count=0)

    def timeline(self) -> tuple:
        return ()


class _FakeEngine:
    def __init__(self, manifest: ModelManifest, memory_budget: MemoryBudgetDecision) -> None:
        self.runtime = _FakeRuntime(manifest)
        self.memory_budget = memory_budget
        self.kv_cache = None
        self.memory_manager = SimpleNamespace(snapshot=lambda: memory_budget.snapshot)
        self.closed = False
        # Mirrors StreamingEngine.startup_decision: prepare_streaming_runtime
        # sets this on a real engine during load. generate() now reuses it
        # (via cli._report_startup_decision) instead of recomputing the M11
        # advisory from scratch after load, so tests must set it explicitly.
        self.startup_decision = None

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        del prompt, max_tokens
        return "generated text"

    def close(self) -> None:
        self.closed = True


class _CollectingHandler(logging.Handler):
    """caplog cannot see this package's logger: configure_logging() sets
    ``propagate = False`` on it (by design, so a long-lived server logs to
    its own rotating file instead of bubbling to the root logger's stdio
    handler). Attach a plain handler directly to it instead."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_generate_logs_an_m11_advisory_using_the_real_planned_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest = _manifest()
    budget = _memory_budget()
    fake_engine = _FakeEngine(manifest, budget)

    # generate no longer probes hardware/bandwidth or calls decide_startup()
    # itself after load (that used to duplicate what prepare_streaming_runtime
    # already did during load_streaming_model, and could disagree with it) --
    # it now only re-logs/dumps engine.startup_decision, exactly what a real
    # engine's prepare_streaming_runtime call would have already set. Build
    # that same StartupDecision here and hang it on the fake engine so this
    # test still exercises the real advisor.py/startup.py wiring end to end.
    hardware_profile = HardwareProfile(
        device_name="Apple M4",
        physical_memory_bytes=budget.snapshot.physical_memory_bytes,
        recommended_working_set_bytes=budget.snapshot.recommended_working_set_bytes,
        gpu_core_count=10,
        cpu_performance_cores=4,
        cpu_efficiency_cores=6,
        wired_limit_mb=0,
        disk_free_bytes=None,
        disk_total_bytes=None,
        vm_page_size_bytes=None,
        compressor_pages_stored=None,
        compressor_pages_occupied=None,
        compressor_compressions=None,
        compressor_decompressions=None,
    )
    bandwidth_bytes_per_sec, bandwidth_source = resolve_startup_bandwidth_bytes_per_sec(
        "off", manifest
    )
    advisor_input = StartupAdvisorInput(
        bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
        bandwidth_source=bandwidth_source,
        current_quantization_bits=manifest.quantization.bits,
    )
    fake_engine.startup_decision = decide_startup(
        hardware_profile,
        manifest.expert_working_set(),
        budget,
        advisor_input,
        expert_keys=tuple(manifest.expert_bundles.keys()),
    )

    monkeypatch.setattr(cli, "load_streaming_model", lambda *args, **kwargs: fake_engine)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")  # never read: load_streaming_model is faked above

    handler = _CollectingHandler()
    package_logger = logging.getLogger("mlx_moe_stream")
    package_logger.addHandler(handler)
    try:
        exit_code = cli.main(
            [
                "generate",
                "--manifest",
                str(manifest_path),
                "--prompt",
                "hello",
                "--startup-io-probe",
                "off",
            ]
        )
    finally:
        package_logger.removeHandler(handler)

    assert exit_code == 0
    assert fake_engine.closed is True
    advisory_messages = [m for m in handler.messages if "M11 performance advisor" in m]
    assert len(advisory_messages) == 1
    message = advisory_messages[0]
    assert "estimate(model=power_law, alpha=0.30, anchored on M4/16GB)" in message
    assert "bandwidth source=default" in message

    suggestion_messages = [m for m in handler.messages if "M11 suggestion" in m]
    # Mac mini-shaped budget: both the wired-limit and double-deduction
    # suggestions should fire for this snapshot.
    assert len(suggestion_messages) == 2


def test_generate_startup_io_probe_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest = _manifest()
    fake_engine = _FakeEngine(manifest, _memory_budget())
    monkeypatch.setattr(cli, "load_streaming_model", lambda *args, **kwargs: fake_engine)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    exit_code = cli.main(
        [
            "generate",
            "--manifest",
            str(manifest_path),
            "--prompt",
            "hi",
            "--startup-io-probe",
            "not-a-number",
        ]
    )

    # main() catches the ValueError from parse_startup_io_probe and reports
    # it as a normal CLI error (exit code 2), not an uncaught exception.
    assert exit_code == 2
