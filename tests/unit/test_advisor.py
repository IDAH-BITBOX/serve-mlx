"""Tests for the M11 performance advisor (src/mlx_moe_stream/advisor.py).

Covers: the power-law hit-rate estimate reproducing the single Mac mini
M4/16GB anchor point, the exact-trace path matching LruCacheSimulator
bit-for-bit, the "estimates never gate a branch" rule, and the SSD
bandwidth probe's time cap and runtime-metrics isolation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from mlx_moe_stream.advisor import (
    DEFAULT_BANDWIDTH_BYTES_PER_SEC,
    POWER_LAW_ALPHA,
    build_advisory,
    decode_ceiling_tps,
    double_deduction_suggestion,
    estimate_hit_rate,
    four_bit_repack_working_set_bytes,
    parse_startup_io_probe,
    probe_read_bandwidth,
    resident_fraction,
    resolve_startup_bandwidth_bytes_per_sec,
    wired_limit_suggestion,
)
from mlx_moe_stream.cache import ExpertKey, LruCacheSimulator
from mlx_moe_stream.manifest import ExpertWorkingSet
from mlx_moe_stream.runtime import NoCacheExpertRuntime
from mlx_moe_stream.storage import build_qwen3_moe_manifest
from safetensors.numpy import save_file

_GIB = 1024**3

# --- Mac mini M4/16GB anchor numbers (see advisor.py module docstring) ---
_MM_TOTAL_BYTES = 34_225_520_640
_MM_MEAN_BUNDLE_BYTES = 3_342_336
_MM_PER_TOKEN_FULL_MISS_BYTES = 1_069_547_520.0
_MM_AVAILABLE_EXPERT_BYTES = 1_381_749_528  # already shell-corrected M7 budget


def _mac_mini_working_set() -> ExpertWorkingSet:
    return ExpertWorkingSet(
        total_bytes=_MM_TOTAL_BYTES,
        bundle_count=40 * 256,
        mean_bundle_bytes=_MM_MEAN_BUNDLE_BYTES,
        min_bundle_bytes=_MM_MEAN_BUNDLE_BYTES,
        max_bundle_bytes=_MM_MEAN_BUNDLE_BYTES,
        per_token_full_miss_bytes=_MM_PER_TOKEN_FULL_MISS_BYTES,
    )


def _write_tiny_model(path: Path, *, num_experts: int = 32) -> None:
    """A KB-scale real safetensors model, matching test_safetensors_store.py's helper."""

    path.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (path / "config.json").write_text(json.dumps(config))
    tensors: dict[str, np.ndarray] = {}
    for projection, shape in (("gate_proj", (2, 3)), ("up_proj", (2, 3)), ("down_proj", (3, 2))):
        for field, dtype in (("weight", np.uint32), ("scales", np.float16), ("biases", np.float16)):
            values = np.arange(num_experts * int(np.prod(shape)), dtype=dtype).reshape(
                (num_experts, *shape)
            )
            name = f"model.layers.0.mlp.switch_mlp.{projection}.{field}"
            tensors[name] = values
    save_file(tensors, path / "model.safetensors")


class _FakeSlowStore:
    """A throwaway store double whose ``read_bundle`` sleeps, for the probe's deadline test."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds
        self.reads = 0

    def read_bundle(self, bundle: object) -> dict[str, bytes]:
        time.sleep(self._delay_seconds)
        self.reads += 1
        return {}

    def close(self) -> None:
        pass


# --- ① resident_fraction ---------------------------------------------------


def test_resident_fraction_matches_the_macmini_anchor():
    fraction = resident_fraction(_MM_AVAILABLE_EXPERT_BYTES, _mac_mini_working_set())
    assert fraction == pytest.approx(0.0404, abs=0.001)


def test_resident_fraction_is_zero_with_no_cache():
    assert resident_fraction(None, _mac_mini_working_set()) == 0.0


# --- ② estimate_hit_rate: power law reproduces the single anchor -----------


def test_power_law_hit_rate_reproduces_the_macmini_anchor_within_tolerance():
    fraction = resident_fraction(_MM_AVAILABLE_EXPERT_BYTES, _mac_mini_working_set())
    estimate = estimate_hit_rate(fraction)
    assert estimate.method == "power_law"
    # Completion criterion: hit estimate 0.38 +/- 0.03.
    assert estimate.hit_rate == pytest.approx(0.38, abs=0.03)


def test_power_law_alpha_constant_matches_its_documented_calibration():
    # alpha solves 0.3827 = 0.0404 ** alpha exactly (the anchor measurement).
    import math

    solved_alpha = math.log(0.3827) / math.log(0.0404)
    assert POWER_LAW_ALPHA == pytest.approx(solved_alpha, abs=0.001)


def test_power_law_hit_rate_is_clamped_to_never_fall_below_the_fraction():
    # For any f in (0, 1) and ALPHA < 1, f**ALPHA > f already, but the
    # explicit max() must still hold as a safety net.
    for fraction in (0.01, 0.25, 0.5, 0.99):
        estimate = estimate_hit_rate(fraction)
        assert estimate.hit_rate >= fraction


def test_estimate_hit_rate_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match="resident_fraction"):
        estimate_hit_rate(1.5)


# --- ② estimate_hit_rate: trace path matches LruCacheSimulator exactly -----


def test_trace_path_matches_a_direct_lru_simulator_replay_bit_for_bit():
    trace = [
        ExpertKey(0, 0),
        ExpertKey(0, 1),
        ExpertKey(0, 2),
        ExpertKey(0, 0),  # hit
        ExpertKey(0, 3),  # evicts something at capacity=2 bundles
        ExpertKey(0, 1),
    ]
    capacity_bytes = 2_000  # 2 bundles of 1000 bytes each

    estimate = estimate_hit_rate(
        0.5, trace=trace, capacity_bytes=capacity_bytes, bundle_bytes=1_000
    )
    assert estimate.method == "trace"

    reference = LruCacheSimulator(capacity_bytes)
    for key in trace:
        reference.access(key, 1_000)

    assert estimate.hit_rate == reference.stats().hit_rate


def test_trace_path_requires_capacity_bytes():
    with pytest.raises(ValueError, match="capacity_bytes"):
        estimate_hit_rate(0.5, trace=[ExpertKey(0, 0)])


# --- ③ decode_ceiling_tps ---------------------------------------------------


def test_decode_ceiling_matches_the_macmini_verification_arithmetic():
    # 2.98 tok/s * 1.069GB * (1 - 0.383) = 1.96GB/s (background verification).
    ceiling = decode_ceiling_tps(
        bandwidth_bytes_per_sec=1.96e9,
        per_token_full_miss_bytes=_MM_PER_TOKEN_FULL_MISS_BYTES,
        hit_rate=0.383,
    )
    # Completion criterion: decode ceiling 2.98 +/- 0.5 tok/s.
    assert ceiling == pytest.approx(2.98, abs=0.5)


def test_decode_ceiling_is_infinite_at_hit_rate_one():
    ceiling = decode_ceiling_tps(
        bandwidth_bytes_per_sec=1.0e9, per_token_full_miss_bytes=1.0e9, hit_rate=1.0
    )
    assert ceiling == float("inf")


def test_decode_ceiling_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        decode_ceiling_tps(bandwidth_bytes_per_sec=0, per_token_full_miss_bytes=1, hit_rate=0.5)
    with pytest.raises(ValueError):
        decode_ceiling_tps(bandwidth_bytes_per_sec=1, per_token_full_miss_bytes=1, hit_rate=1.5)


# --- ④ recommendations ------------------------------------------------------


def test_four_bit_repack_working_set_halves_from_eight_bit():
    working_set = _mac_mini_working_set()
    repacked = four_bit_repack_working_set_bytes(working_set, 8)
    assert repacked == working_set.total_bytes // 2


def test_wired_limit_suggestion_fires_only_when_unset_and_headroom_exists():
    assert (
        wired_limit_suggestion(
            wired_limit_mb=0,
            recommended_working_set_bytes=12_713_115_648,
            physical_memory_bytes=17_179_869_184,
        )
        is not None
    )
    # Already set -> no suggestion.
    assert (
        wired_limit_suggestion(
            wired_limit_mb=15360,
            recommended_working_set_bytes=12_713_115_648,
            physical_memory_bytes=17_179_869_184,
        )
        is None
    )
    # No real headroom (recommended within 1GiB of physical) -> no suggestion.
    assert (
        wired_limit_suggestion(
            wired_limit_mb=0,
            recommended_working_set_bytes=16_500_000_000,
            physical_memory_bytes=17_179_869_184,
        )
        is None
    )


def test_double_deduction_suggestion_fires_on_the_macmini_case_only():
    # Mac mini: OS withholds 17.18 - 12.71 = 4.47GB >= a quarter of 16GiB (4GiB).
    assert (
        double_deduction_suggestion(
            physical_memory_bytes=17_179_869_184, recommended_working_set_bytes=12_713_115_648
        )
        is not None
    )
    # Mac Studio: OS withholds only ~33GiB of 256GiB, well under a quarter.
    assert (
        double_deduction_suggestion(
            physical_memory_bytes=256 * _GIB, recommended_working_set_bytes=239_143_780_352
        )
        is None
    )


# --- Node 4: compressor-aware double-deduction suggestion -------------------
#
# Measured finding this guards: on the Mac mini M4/16GB, shrinking
# --memory-safety-margin from 4.000GiB (auto) to 1.000GiB (adaptive) raised
# hit rate 0.38 -> 0.55 but *lowered* decode 2.65 -> 2.43 tok/s (compressor
# thrash). double_deduction_suggestion must therefore never recommend
# shrinking the margin unconditionally on a small-memory device.

_MM_PHYSICAL_BYTES = 17_179_869_184  # 16GiB Mac mini
_MM_RECOMMENDED_BYTES = 12_713_115_648


def test_double_deduction_suggestion_cautions_on_low_memory_with_no_compressor_telemetry():
    """No compressor probe available (None) -> old behavior, but with the caution appended."""

    suggestion = double_deduction_suggestion(
        physical_memory_bytes=_MM_PHYSICAL_BYTES,
        recommended_working_set_bytes=_MM_RECOMMENDED_BYTES,
    )
    assert suggestion is not None
    assert "--memory-safety-margin adaptive" in suggestion
    assert "CAUTION" in suggestion
    assert "2.65" in suggestion and "2.43" in suggestion  # measured decode figures cited
    assert "0.38" in suggestion and "0.55" in suggestion  # measured hit-rate figures cited


def test_double_deduction_suggestion_still_recommends_when_compressor_is_idle():
    """Compressor telemetry present but idle (0 pages occupied) -> still recommend, with caution."""

    suggestion = double_deduction_suggestion(
        physical_memory_bytes=_MM_PHYSICAL_BYTES,
        recommended_working_set_bytes=_MM_RECOMMENDED_BYTES,
        compressor_pages_occupied=0,
        vm_page_size_bytes=16384,
    )
    assert suggestion is not None
    assert "--memory-safety-margin adaptive" in suggestion
    assert "SUGGESTION SUPPRESSED" not in suggestion
    assert "CAUTION" in suggestion


def test_double_deduction_suggestion_is_suppressed_when_compressor_already_active():
    """Compressor already holds pages on a <=32GiB device -> suppress the actionable flags."""

    suggestion = double_deduction_suggestion(
        physical_memory_bytes=_MM_PHYSICAL_BYTES,
        recommended_working_set_bytes=_MM_RECOMMENDED_BYTES,
        compressor_pages_occupied=24_992,
        vm_page_size_bytes=16384,
    )
    assert suggestion is not None
    assert "SUGGESTION SUPPRESSED" in suggestion
    assert "--memory-safety-margin adaptive" not in suggestion
    assert "--max-unified-memory" not in suggestion
    # the measured caution's own figures still ride along with the suppression message
    assert "2.43" in suggestion


def test_double_deduction_suggestion_does_not_suppress_on_large_memory_devices():
    """The compressor-active suppression only applies at/under the 32GiB low-memory cutoff."""

    physical = 64 * _GIB
    recommended = int(physical * 0.7)  # OS withholds 30% >= a quarter -> double deduction fires
    suggestion = double_deduction_suggestion(
        physical_memory_bytes=physical,
        recommended_working_set_bytes=recommended,
        compressor_pages_occupied=999_999,
        vm_page_size_bytes=16384,
    )
    assert suggestion is not None
    assert "SUGGESTION SUPPRESSED" not in suggestion
    assert "--memory-safety-margin adaptive" in suggestion
    assert "CAUTION" not in suggestion


def test_build_advisory_wires_compressor_telemetry_into_double_deduction_suggestion():
    """build_advisory must forward compressor_pages_occupied/vm_page_size_bytes end to end."""

    common_kwargs = dict(
        expert_budget_bytes=_MM_AVAILABLE_EXPERT_BYTES,
        working_set=_mac_mini_working_set(),
        bandwidth_bytes_per_sec=1.96e9,
        wired_limit_mb=0,
        physical_memory_bytes=_MM_PHYSICAL_BYTES,
        recommended_working_set_bytes=_MM_RECOMMENDED_BYTES,
    )

    idle = build_advisory(**common_kwargs, compressor_pages_occupied=0, vm_page_size_bytes=16384)
    assert idle.double_deduction_suggestion is not None
    assert "SUGGESTION SUPPRESSED" not in idle.double_deduction_suggestion

    active = build_advisory(
        **common_kwargs, compressor_pages_occupied=24_992, vm_page_size_bytes=16384
    )
    assert active.double_deduction_suggestion is not None
    assert "SUGGESTION SUPPRESSED" in active.double_deduction_suggestion

    # Omitting the new kwargs entirely (existing callers) must not raise and
    # must preserve the pre-existing (un-suppressed) recommendation.
    unaware = build_advisory(**common_kwargs)
    assert unaware.double_deduction_suggestion is not None
    assert "SUGGESTION SUPPRESSED" not in unaware.double_deduction_suggestion


# --- the exact-branch rule ("추정치는 어떤 분기에도 사용하지 않는다") ------------


def test_covers_full_working_set_is_exact_not_estimated():
    working_set = ExpertWorkingSet(
        total_bytes=1_000,
        bundle_count=1,
        mean_bundle_bytes=1_000,
        min_bundle_bytes=1_000,
        max_bundle_bytes=1_000,
        per_token_full_miss_bytes=1_000.0,
    )
    # Exactly equal to total_bytes -> covers_full_working_set True, hit=1.0
    # via the EXACT path (method must not be "power_law", even though
    # 1.0 ** ALPHA also happens to equal 1.0 -- the method label proves
    # which code path actually ran).
    exact = build_advisory(
        expert_budget_bytes=1_000, working_set=working_set, bandwidth_bytes_per_sec=1.0e9
    )
    assert exact.covers_full_working_set is True
    assert exact.hit_rate_estimate.method == "exact_full_residency"
    assert exact.hit_rate_estimate.hit_rate == 1.0
    # inf is not valid JSON; build_advisory clamps it to None so /metrics
    # and --startup-report never have to emit a non-finite float.
    assert exact.decode_ceiling_tps is None

    # One byte under -> covers_full_working_set False, estimate path used.
    just_under = build_advisory(
        expert_budget_bytes=999, working_set=working_set, bandwidth_bytes_per_sec=1.0e9
    )
    assert just_under.covers_full_working_set is False
    assert just_under.hit_rate_estimate.method == "power_law"
    assert just_under.hit_rate_estimate.hit_rate < 1.0


def test_build_advisory_never_calls_the_estimator_when_the_exact_branch_fires(monkeypatch):
    """If the exact bytes inequality holds, the estimator must not even run."""

    import mlx_moe_stream.advisor as advisor_module

    def _boom(*args, **kwargs):
        raise AssertionError("estimate_hit_rate must not be called on the exact branch")

    monkeypatch.setattr(advisor_module, "estimate_hit_rate", _boom)
    working_set = ExpertWorkingSet(
        total_bytes=500,
        bundle_count=1,
        mean_bundle_bytes=500,
        min_bundle_bytes=500,
        max_bundle_bytes=500,
        per_token_full_miss_bytes=500.0,
    )
    # expert_budget_bytes > total_bytes also counts as "covers everything".
    advisory = advisor_module.build_advisory(
        expert_budget_bytes=600, working_set=working_set, bandwidth_bytes_per_sec=1.0e9
    )
    assert advisory.covers_full_working_set is True
    assert advisory.hit_rate_estimate.hit_rate == 1.0


def test_build_advisory_log_line_carries_the_power_law_label_and_measured_inputs():
    working_set = _mac_mini_working_set()
    advisory = build_advisory(
        expert_budget_bytes=_MM_AVAILABLE_EXPERT_BYTES,
        working_set=working_set,
        bandwidth_bytes_per_sec=1.96e9,
    )
    assert "estimate(model=power_law, alpha=0.30, anchored on M4/16GB)" in advisory.log_line
    for term in (
        "budget=",
        "total=",
        "fraction=",
        "bandwidth=",
        "per_token_bytes=",
        "hit=",
        "decode_ceiling_tps=",
    ):
        assert term in advisory.log_line


def test_build_advisory_reproduces_the_macmini_completion_criteria():
    advisory = build_advisory(
        expert_budget_bytes=_MM_AVAILABLE_EXPERT_BYTES,
        working_set=_mac_mini_working_set(),
        bandwidth_bytes_per_sec=1.96e9,
        current_quantization_bits=8,
        wired_limit_mb=0,
        physical_memory_bytes=17_179_869_184,
        recommended_working_set_bytes=12_713_115_648,
    )
    assert advisory.hit_rate_estimate.hit_rate == pytest.approx(0.38, abs=0.03)
    assert advisory.decode_ceiling_tps == pytest.approx(2.98, abs=0.5)
    assert advisory.wired_limit_suggestion is not None
    assert advisory.double_deduction_suggestion is not None
    assert advisory.four_bit_repack_working_set_bytes == _MM_TOTAL_BYTES // 2


# --- decode_ceiling_tps confidence labeling ---------------------------------
#
# Real /metrics measurement found the power-law hit-rate estimate accurate
# (0.549 predicted / 0.551 measured) but decode_ceiling_tps -- a pure
# I/O-bandwidth upper bound -- badly overestimated actual decode (4.455
# tok/s predicted ceiling vs 2.43 tok/s measured), because it has no term
# for CPU overhead or macOS memory-compressor thrashing. The log line (and
# the PerformanceAdvisory/decode_ceiling_tps docstrings) must make that
# confidence gap explicit rather than presenting both numbers as equally
# trustworthy.


def test_log_line_marks_decode_ceiling_as_a_low_confidence_upper_bound():
    advisory = build_advisory(
        expert_budget_bytes=_MM_AVAILABLE_EXPERT_BYTES,
        working_set=_mac_mini_working_set(),
        bandwidth_bytes_per_sec=1.96e9,
    )
    assert "upper bound" in advisory.log_line
    assert "low confidence" in advisory.log_line
    # the measured mismatch figures ride along so the caution is concrete,
    # not just a vague hedge
    assert "2.43" in advisory.log_line and "4.455" in advisory.log_line


def test_log_line_marks_hit_rate_as_high_confidence_and_validated():
    advisory = build_advisory(
        expert_budget_bytes=_MM_AVAILABLE_EXPERT_BYTES,
        working_set=_mac_mini_working_set(),
        bandwidth_bytes_per_sec=1.96e9,
    )
    assert "high confidence" in advisory.log_line
    assert "validated" in advisory.log_line
    # the measured validation figures for the power-law hit-rate estimate
    assert "0.549" in advisory.log_line and "0.551" in advisory.log_line


def test_log_line_has_no_low_confidence_ceiling_tag_at_full_residency():
    """At full residency there is no I/O-bound ceiling at all (inf, clamped

    to None on the dataclass) -- the log line must not claim a "low
    confidence" upper bound number that isn't really an upper bound on
    anything (nothing is disk-bound).
    """

    working_set = ExpertWorkingSet(
        total_bytes=1_000,
        bundle_count=1,
        mean_bundle_bytes=1_000,
        min_bundle_bytes=1_000,
        max_bundle_bytes=1_000,
        per_token_full_miss_bytes=1_000.0,
    )
    advisory = build_advisory(
        expert_budget_bytes=1_000, working_set=working_set, bandwidth_bytes_per_sec=1.0e9
    )
    assert advisory.decode_ceiling_tps is None
    assert "no I/O-bound ceiling" in advisory.log_line
    assert "low confidence" not in advisory.log_line
    # the exact-residency hit rate is still labeled high confidence
    assert "high confidence" in advisory.log_line


def test_decode_ceiling_tps_docstring_documents_the_upper_bound_caution():
    """The function's own docstring must state the ceiling/low-confidence caveat,

    not just the log line -- callers reading the source (rather than a log)
    need the same warning.
    """

    doc = decode_ceiling_tps.__doc__ or ""
    assert "UPPER BOUND" in doc
    assert "LOW CONFIDENCE" in doc
    assert "2.43" in doc and "4.455" in doc


# --- Node 5: probe_read_bandwidth -------------------------------------------


def test_probe_read_bandwidth_respects_the_deadline_with_a_slow_reader():
    model_path_bundles = _fake_bundles(count=64)
    fake_store = _FakeSlowStore(delay_seconds=0.05)

    started = time.monotonic()
    result = probe_read_bandwidth(
        _FakeManifest(model_path_bundles),
        max_seconds=0.2,
        sample_count=64,
        store_factory=lambda: fake_store,
    )
    elapsed = time.monotonic() - started

    # Bounded overshoot: at most one read's own latency past the deadline.
    assert elapsed < 0.2 + 0.05 + 0.1
    assert fake_store.reads >= 1
    # Some reads completed before the deadline tripped -> a real number back.
    assert result is None or result > 0


def test_probe_read_bandwidth_returns_none_with_no_bundles():
    assert probe_read_bandwidth(_FakeManifest({}), store_factory=lambda: _FakeSlowStore(0)) is None


def test_probe_read_bandwidth_on_a_real_tiny_model_never_touches_runtime_metrics(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_tiny_model(model_path, num_experts=32)
    manifest = build_qwen3_moe_manifest(model_path)

    bandwidth = probe_read_bandwidth(manifest, sample_count=16, max_seconds=2.0)
    assert bandwidth is not None
    assert bandwidth > 0

    runtime = NoCacheExpertRuntime(manifest)
    try:
        # Completion criterion: the probe used its own throwaway store, so a
        # freshly constructed runtime's metrics must still read zero.
        assert runtime.stats().bytes_read == 0
    finally:
        runtime.close()


def test_probe_read_bandwidth_finishes_quickly_on_a_real_tiny_model(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_tiny_model(model_path, num_experts=32)
    manifest = build_qwen3_moe_manifest(model_path)

    started = time.monotonic()
    probe_read_bandwidth(manifest, max_seconds=2.0)
    elapsed = time.monotonic() - started
    assert elapsed <= 2.0 + 0.5  # generous CI-noise margin around the 2.0s cap


# --- --startup-io-probe parsing ---------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [(None, "auto"), ("auto", "auto"), ("AUTO", "auto"), ("off", "off"), ("OFF", "off")],
)
def test_parse_startup_io_probe_literals(raw, expected):
    assert parse_startup_io_probe(raw) == expected


def test_parse_startup_io_probe_accepts_an_explicit_bytes_per_sec_number():
    assert parse_startup_io_probe("2500000000") == pytest.approx(2_500_000_000.0)


def test_parse_startup_io_probe_rejects_garbage():
    with pytest.raises(ValueError, match="startup-io-probe"):
        parse_startup_io_probe("not-a-number")
    with pytest.raises(ValueError):
        parse_startup_io_probe("-5")


def test_resolve_startup_bandwidth_off_uses_the_default_without_probing(monkeypatch):
    import mlx_moe_stream.advisor as advisor_module

    def _boom(*args, **kwargs):
        raise AssertionError("probe_read_bandwidth must not run when probing is off")

    monkeypatch.setattr(advisor_module, "probe_read_bandwidth", _boom)
    bandwidth, source = resolve_startup_bandwidth_bytes_per_sec("off", manifest=object())
    assert bandwidth == DEFAULT_BANDWIDTH_BYTES_PER_SEC
    assert source == "default"


def test_resolve_startup_bandwidth_explicit_value_passes_through():
    bandwidth, source = resolve_startup_bandwidth_bytes_per_sec(3.5e9, manifest=object())
    assert bandwidth == 3.5e9
    assert source == "explicit"


def test_resolve_startup_bandwidth_auto_falls_back_to_default_when_probe_fails(monkeypatch):
    import mlx_moe_stream.advisor as advisor_module

    monkeypatch.setattr(advisor_module, "probe_read_bandwidth", lambda manifest: None)
    bandwidth, source = resolve_startup_bandwidth_bytes_per_sec("auto", manifest=object())
    assert bandwidth == DEFAULT_BANDWIDTH_BYTES_PER_SEC
    assert source == "default"


def test_resolve_startup_bandwidth_auto_uses_the_probed_value(monkeypatch):
    import mlx_moe_stream.advisor as advisor_module

    monkeypatch.setattr(advisor_module, "probe_read_bandwidth", lambda manifest: 1.5e9)
    bandwidth, source = resolve_startup_bandwidth_bytes_per_sec("auto", manifest=object())
    assert bandwidth == 1.5e9
    assert source == "probed"


# --- fakes for the deadline test (no real files needed) --------------------


class _FakeBundle:
    def __init__(self, total_bytes: int, offset: int) -> None:
        self.total_bytes = total_bytes
        self.tensors = [_FakeTensor(offset)]


class _FakeTensor:
    def __init__(self, offset: int) -> None:
        self.offset = offset


class _FakeManifest:
    def __init__(self, expert_bundles: dict) -> None:
        self.expert_bundles = expert_bundles


def _fake_bundles(*, count: int) -> dict:
    return {i: _FakeBundle(total_bytes=1_000, offset=i * 1_000) for i in range(count)}
