"""M11 performance advisor: resident-fraction -> hit-rate -> decode-ceiling.

This module turns the M7 memory budget's ``expert_budget_bytes`` and the
manifest's :class:`~mlx_moe_stream.manifest.ExpertWorkingSet` into a decode
throughput ceiling and a short list of actionable recommendations.

Hard rule enforced throughout this module: **estimated hit rates never drive
a branch.** The only conditional that changes what advice is produced is the
exact byte inequality ``expert_budget_bytes >= working_set.total_bytes``
(the resident cache can, or cannot, hold every expert bundle at once).
Every hit-rate number -- whether from the power-law model or a trace
simulation -- is computed *after* that branch and only ever feeds the
decode-ceiling number and the log line; it is never compared against a
threshold to pick a code path. See ``build_advisory`` below.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil

from .cache import ExpertKey, LruCacheSimulator
from .manifest import ExpertWorkingSet, ModelManifest
from .storage.base import StorageReadError
from .storage.safetensors_store import SafetensorsExpertStore

_logger = logging.getLogger(__name__)

_GIB = 1024**3

#: Power-law hit-rate exponent: hit_rate ~= resident_fraction ** ALPHA.
#:
#: MEASUREMENT: anchored on a *single* production /metrics data point taken
#: on the Mac mini M4/16GB: resident_fraction f=0.0404 (a 1.382GB resident
#: expert cache over a 31.87GiB / 34,225,520,640-byte working set) produced
#: an observed cache hit rate of 0.3827. Solving hit = f ** alpha for alpha
#: gives alpha = ln(0.3827) / ln(0.0404) ~= 0.2985.
#:
#: CAUTION -- this is a ONE-POINT calibration, not a fitted curve across
#: multiple (f, hit) observations. It is reasonable to trust near f~0.04;
#: extrapolating to very different fractions (e.g. f > 0.5, or a
#: differently-shaped routing distribution) is unverified and should be
#: treated as an order-of-magnitude guess only, never as a precise number.
POWER_LAW_ALPHA = 0.2985

#: Default read bandwidth assumed when no I/O probe is run or the probe
#: fails, measured on the Mac mini M4's internal SSD.
DEFAULT_BANDWIDTH_BYTES_PER_SEC = 2.0 * 1_000_000_000  # 2.0 GB/s, M4 default

_DEFAULT_PROBE_SAMPLE_COUNT = 16
_DEFAULT_PROBE_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_PROBE_MAX_SECONDS = 2.0


@dataclass(frozen=True)
class HitRateEstimate:
    """One hit-rate number plus how it was produced."""

    resident_fraction: float
    hit_rate: float
    method: str  # "exact_full_residency", "trace", or "power_law"


def resident_fraction(expert_budget_bytes: int | None, working_set: ExpertWorkingSet) -> float:
    """① The fraction of the total expert working set that fits resident.

    Returns 0.0 when there is no resident cache at all (``expert_budget_bytes``
    is ``None``, i.e. the M4/M7 "disabled" cache source).
    """

    if working_set.total_bytes <= 0:
        raise ValueError("working set total_bytes must be greater than zero")
    if expert_budget_bytes is None:
        return 0.0
    if expert_budget_bytes < 0:
        raise ValueError("expert_budget_bytes cannot be negative")
    return expert_budget_bytes / working_set.total_bytes


def estimate_hit_rate(
    fraction: float,
    *,
    trace: Sequence[ExpertKey] | None = None,
    capacity_bytes: int | None = None,
    bundle_bytes: int = 1,
) -> HitRateEstimate:
    """② Estimate the steady-state cache hit rate for resident fraction ``f``.

    Two independent paths, chosen only by whether a ``trace`` was supplied:

    * **Exact path** -- if ``trace`` (a sequence of per-access
      :class:`~mlx_moe_stream.cache.ExpertKey`) is given, replay it through
      the existing :class:`~mlx_moe_stream.cache.LruCacheSimulator` at the
      planned ``capacity_bytes`` and report its real observed hit rate.
      This is not an estimate; it is measured from the trace.
    * **Estimation path** -- otherwise, use the power law
      ``hit ~= f ** ALPHA`` (see ``POWER_LAW_ALPHA`` for its single-anchor
      calibration and caveats), clamped to never estimate a hit rate below
      the resident fraction itself (``max(f, f ** ALPHA)``).
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("resident_fraction must be in [0, 1]")
    if trace is not None:
        if capacity_bytes is None:
            raise ValueError("capacity_bytes is required to simulate a trace")
        simulator = LruCacheSimulator(capacity_bytes)
        for key in trace:
            simulator.access(key, bundle_bytes)
        return HitRateEstimate(
            resident_fraction=fraction, hit_rate=simulator.stats().hit_rate, method="trace"
        )

    hit = fraction**POWER_LAW_ALPHA if fraction > 0 else 0.0
    hit = max(fraction, hit)
    return HitRateEstimate(resident_fraction=fraction, hit_rate=hit, method="power_law")


def decode_ceiling_tps(
    *, bandwidth_bytes_per_sec: float, per_token_full_miss_bytes: float, hit_rate: float
) -> float:
    """③ The decode tokens/sec ceiling if every miss byte is I/O bound.

    ``tok/s = bandwidth / (per_token_full_miss_bytes * (1 - hit_rate))`` --
    the exact inverse of the verification arithmetic used to derive the
    Mac mini's measured bandwidth (``tok/s * per_token_bytes * (1 - hit) =
    bandwidth``). A hit rate of 1.0 has no I/O-bound ceiling (returns
    ``inf``): every expert is already resident.

    CAUTION -- this is a theoretical I/O-bandwidth UPPER BOUND (ceiling), not
    a measured or predicted actual decode throughput, and it is LOW CONFIDENCE
    compared to ``estimate_hit_rate``'s ``hit_rate`` number. Production
    validation on the Mac mini M4/16GB (margin=1.000GiB/adaptive
    point, see ``_MEASURED_MARGIN_CAUTION``) found the hit-rate estimate
    accurate (0.549 predicted vs 0.551 measured) while this ceiling was not:
    4.455 tok/s predicted vs only 2.43 tok/s measured decode. The gap exists
    because this formula only accounts for I/O time (bytes / bandwidth); it
    has no term for CPU overhead or macOS's memory-compressor thrashing
    under reduced headroom, both of which measurably slow decode
    independent of disk reads. Callers must treat the return value as "no
    faster than this," never as "approximately this."
    """

    if bandwidth_bytes_per_sec <= 0:
        raise ValueError("bandwidth_bytes_per_sec must be greater than zero")
    if per_token_full_miss_bytes <= 0:
        raise ValueError("per_token_full_miss_bytes must be greater than zero")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError("hit_rate must be in [0, 1]")
    miss_fraction = 1.0 - hit_rate
    if miss_fraction <= 0.0:
        return float("inf")
    return bandwidth_bytes_per_sec / (per_token_full_miss_bytes * miss_fraction)


def budget_bytes_for_target_hit_rate(target_hit_rate: float, working_set: ExpertWorkingSet) -> int:
    """④a Expert-cache bytes the power-law model needs to reach ``target_hit_rate``.

    Inverts ``hit = f ** ALPHA`` for ``f``, i.e. ``f = hit ** (1 / ALPHA)``,
    then scales by the working set's total bytes. This is advisory output
    only (see the module docstring): it must never gate a branch.
    """

    if not 0.0 < target_hit_rate <= 1.0:
        raise ValueError("target_hit_rate must be in (0, 1]")
    fraction = target_hit_rate ** (1.0 / POWER_LAW_ALPHA)
    return min(working_set.total_bytes, ceil(fraction * working_set.total_bytes))


def four_bit_repack_working_set_bytes(working_set: ExpertWorkingSet, current_bits: int) -> int:
    """④b Working-set bytes if every expert bundle were repacked to 4-bit.

    ``total_bytes * 4 / current_bits`` -- pure linear rescaling assuming the
    bundle byte count scales directly with quantization bit width (true for
    the uniform group-quantized weight layout this project reads).
    """

    if current_bits <= 0:
        raise ValueError("current_bits must be greater than zero")
    return ceil(working_set.total_bytes * 4 / current_bits)


def wired_limit_suggestion(
    *, wired_limit_mb: int | None, recommended_working_set_bytes: int, physical_memory_bytes: int
) -> str | None:
    """④c Suggest raising ``iogpu.wired_limit_mb`` when macOS is under-granting MLX.

    Only fires when the wired limit is unset (``0`` -- the ``sysctl`` default
    meaning "no explicit limit") *and* the OS's recommended working set is
    more than 1 GiB below physical memory, i.e. there is real headroom macOS
    is not handing to MLX by default.
    """

    if wired_limit_mb != 0:
        return None
    if recommended_working_set_bytes >= physical_memory_bytes - _GIB:
        return None
    target_mb = (physical_memory_bytes - _GIB) // (1024 * 1024)
    return (
        f"wired_limit_mb is unset and the OS recommendation "
        f"({recommended_working_set_bytes / _GIB:.2f}GiB) leaves "
        f"{(physical_memory_bytes - recommended_working_set_bytes) / _GIB:.2f}GiB of physical "
        f"memory unused by MLX; consider `sudo sysctl iogpu.wired_limit_mb={target_mb}` "
        "to raise the ceiling before relying on --max-unified-memory"
    )


#: Physical memory at or below which the M4/16GB compressor-thrash finding
#: (see ``_MEASURED_MARGIN_CAUTION`` below) applies. Chosen to cover the
#: 16GiB Mac mini the finding was measured on plus one size class up (a
#: 32GiB Mac mini/Studio); it is NOT independently verified past 16GiB --
#: treat 32GiB as a conservative "still small enough to be cautious" cutoff,
#: not a calibrated threshold.
_LOW_MEMORY_PHYSICAL_BYTES_THRESHOLD = 32 * _GIB

#: MEASUREMENT: production run on the Mac mini M4/16GB (expert working set
#: 31.87GiB) sweeping ``--memory-safety-margin``: shrinking the margin raised
#: the observed cache hit rate but *lowered* decode throughput, because the
#: extra resident bytes came at the cost of macOS's memory compressor
#: thrashing under the reduced headroom:
#:
#:   margin=4.000GiB (auto, default): hit=0.381, decode=2.65-2.98 tok/s (best)
#:   margin=2.328GiB:                 hit=0.474, decode=2.48 tok/s
#:   margin=1.000GiB (adaptive):      hit=0.551, decode=2.43 tok/s (worst)
#:
#: i.e. this specific device got *slower* the more its safety margin was
#: cut, despite a strictly improving hit rate, because a smaller margin
#: leaves the OS less room before it starts compressing pages. This is the
#: opposite of what the "double deduction = wasted memory" framing below
#: implies in isolation, and is why this suggestion must never be applied
#: blind on a small-memory device.
_MEASURED_MARGIN_CAUTION = (
    "CAUTION (measured on Mac mini M4/16GB, expert working set 31.87GiB): shrinking "
    "the safety margin from 4.000GiB to 1.000GiB raised the cache hit rate from 0.38 "
    "to 0.55, but *lowered* decode throughput from 2.65-2.98 tok/s to 2.43 tok/s, "
    "because macOS's memory compressor started thrashing under the reduced headroom. "
    "On physical memory this small, the withheld quarter also functions as compressor "
    "thrash headroom -- do not shrink it without a fresh on-device measurement"
)


def double_deduction_suggestion(
    *,
    physical_memory_bytes: int,
    recommended_working_set_bytes: int,
    compressor_pages_occupied: int | None = None,
    vm_page_size_bytes: int | None = None,
) -> str | None:
    """④d Detect the Mac-mini-class double safety-margin deduction.

    ``automatic_safety_margin_bytes`` reserves a further quarter of physical
    memory on top of whatever macOS already withheld from
    ``recommended_working_set_bytes``. Arithmetically, when the OS has
    *already* withheld a quarter (or more) of physical memory, applying the
    automatic margin on top double-counts that reservation. Detected here via
    the same arithmetic ``adaptive_safety_margin_bytes`` uses internally: the
    OS's already-withheld share is ``physical - recommended``; a double
    deduction is in play once that share alone reaches a quarter of physical
    memory.

    That arithmetic framing is *not* the whole story, though: measurement on
    a 16GiB Mac mini (see ``_MEASURED_MARGIN_CAUTION``) found that shrinking
    the margin to reclaim this "double-counted" memory made decode *slower*,
    not faster, because the reclaimed headroom is also what keeps macOS's
    memory compressor from thrashing. So this function no longer recommends
    shrinking the margin unconditionally:

    * On physical memory > 32GiB (``_LOW_MEMORY_PHYSICAL_BYTES_THRESHOLD``),
      or when no compressor telemetry is available, it still returns the
      actionable suggestion, but always with the measured caution appended
      -- never apply it without a fresh on-device measurement.
    * On physical memory <= 32GiB *and* live ``compressor_pages_occupied``
      telemetry (from :func:`mlx_moe_stream.hardware.probe_hardware`) shows
      the compressor already holds pages (> 0), the actionable
      ``--memory-safety-margin``/``--max-unified-memory`` recommendation is
      suppressed outright: applying it while the compressor is already
      active is the exact condition the measurement above reproduced.
    """

    if physical_memory_bytes <= 0:
        raise ValueError("physical_memory_bytes must be greater than zero")
    already_withheld = physical_memory_bytes - recommended_working_set_bytes
    if already_withheld < physical_memory_bytes // 4:
        return None

    suggested_gb = max(1, (physical_memory_bytes - _GIB) // 1_000_000_000)
    low_memory = physical_memory_bytes <= _LOW_MEMORY_PHYSICAL_BYTES_THRESHOLD
    compressor_active = compressor_pages_occupied is not None and compressor_pages_occupied > 0

    base = (
        f"the OS already withholds {already_withheld / _GIB:.2f}GiB "
        f"(>= a quarter of physical memory) from the recommended working set, which "
        "arithmetically double-counts against --memory-safety-margin auto (the default)"
    )

    if low_memory and compressor_active:
        occupied_desc = (
            f"{compressor_pages_occupied} pages"
            if vm_page_size_bytes is None
            else f"{compressor_pages_occupied * vm_page_size_bytes / _GIB:.3f}GiB"
        )
        return (
            f"{base}; SUGGESTION SUPPRESSED -- the memory compressor is already active "
            f"(compressor_pages_occupied={occupied_desc}) on this <=32GiB-class device, "
            "which is exactly the condition under which shrinking the safety margin measured "
            "*slower* decode despite a higher hit rate. Not suggesting a margin change here; only "
            f"shrink the margin after a fresh on-device measurement. {_MEASURED_MARGIN_CAUTION}"
        )

    suggestion = (
        f"{base}. Use `--memory-safety-margin adaptive` and/or `--max-unified-memory "
        f"{suggested_gb}` to stop deducting it twice"
    )
    if low_memory:
        suggestion = f"{suggestion}. {_MEASURED_MARGIN_CAUTION}"
    return suggestion


@dataclass(frozen=True)
class PerformanceAdvisory:
    """The full advisor output for one planned M7 budget.

    decode_ceiling_tps is None whenever decode_ceiling_tps() (see
    that function's own docstring) found no I/O-bound ceiling at all -- every
    expert already resident, so there is nothing disk-bound to report. It is
    deliberately clamped from inf to None here (never surfaced past
    this dataclass as a non-finite float): inf is not valid JSON, and
    this advisory's report dict is serialized verbatim for both
    /metrics and --startup-report.

    RELIABILITY NOTE -- ``decode_ceiling_tps`` and ``hit_rate_estimate`` do
    NOT carry the same confidence, even though they are computed back to
    back from the same inputs. ``hit_rate_estimate`` was validated against a
    real Mac mini M4/16GB measurement (0.549 predicted / 0.551 measured --
    accurate). ``decode_ceiling_tps`` is only an I/O-bandwidth theoretical
    UPPER BOUND (see decode_ceiling_tps()'s docstring) and was NOT accurate
    on that same hardware (4.455 tok/s predicted ceiling vs 2.43 tok/s
    measured decode: CPU overhead and memory-compressor thrashing are not
    modeled). Never report decode_ceiling_tps to a user/log as if it were an
    expected tok/s figure with the same confidence as hit_rate -- ``log_line``
    below tags each number with its own confidence for exactly this reason.
    """

    resident_fraction: float
    hit_rate_estimate: HitRateEstimate
    decode_ceiling_tps: float | None
    covers_full_working_set: bool
    budget_bytes_for_target_hit: int
    target_hit_rate: float
    four_bit_repack_working_set_bytes: int | None
    wired_limit_suggestion: str | None
    double_deduction_suggestion: str | None
    log_line: str


def build_advisory(
    *,
    expert_budget_bytes: int | None,
    working_set: ExpertWorkingSet,
    bandwidth_bytes_per_sec: float,
    trace: Sequence[ExpertKey] | None = None,
    target_hit_rate: float = 0.9,
    current_quantization_bits: int | None = None,
    wired_limit_mb: int | None = None,
    physical_memory_bytes: int | None = None,
    recommended_working_set_bytes: int | None = None,
    compressor_pages_occupied: int | None = None,
    vm_page_size_bytes: int | None = None,
) -> PerformanceAdvisory:
    """Combine ①-④ into one advisory, honoring the exact-branch rule.

    ``covers_full_working_set`` is the ONLY branch in this function that
    changes behavior, and it uses the exact inequality
    ``expert_budget_bytes >= working_set.total_bytes`` -- never an estimate.
    When it is true the hit rate is exactly 1.0 (every expert bundle already
    fits resident), computed with no power-law or trace estimation at all.

    ``compressor_pages_occupied``/``vm_page_size_bytes`` are the optional
    ``HardwareProfile`` compressor probe fields (:mod:`mlx_moe_stream.hardware`);
    they only ever feed ``double_deduction_suggestion``'s low-memory
    compressor-activity check and default to ``None`` (probe unavailable),
    which preserves this function's pre-existing behavior for every
    existing caller.
    """

    covers_full_working_set = (
        expert_budget_bytes is not None and expert_budget_bytes >= working_set.total_bytes
    )
    fraction = resident_fraction(expert_budget_bytes, working_set)
    if covers_full_working_set:
        hit_estimate = HitRateEstimate(
            resident_fraction=fraction, hit_rate=1.0, method="exact_full_residency"
        )
    else:
        hit_estimate = estimate_hit_rate(
            fraction,
            trace=trace,
            capacity_bytes=expert_budget_bytes,
            bundle_bytes=max(1, round(working_set.mean_bundle_bytes)),
        )

    ceiling = decode_ceiling_tps(
        bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
        per_token_full_miss_bytes=working_set.per_token_full_miss_bytes,
        hit_rate=hit_estimate.hit_rate,
    )
    # JSON has no representation for inf; a fully-resident cache has no
    # I/O-bound ceiling to report at all, so clamp it to None rather than
    # let a non-finite float reach the /metrics or --startup-report wire
    # format (json.dumps(..., allow_nan=False) would otherwise raise here).
    json_safe_ceiling = None if ceiling == float("inf") else ceiling

    needed_budget = budget_bytes_for_target_hit_rate(target_hit_rate, working_set)
    four_bit_bytes = (
        four_bit_repack_working_set_bytes(working_set, current_quantization_bits)
        if current_quantization_bits is not None and current_quantization_bits != 4
        else None
    )
    wired_suggestion = (
        wired_limit_suggestion(
            wired_limit_mb=wired_limit_mb,
            recommended_working_set_bytes=recommended_working_set_bytes,
            physical_memory_bytes=physical_memory_bytes,
        )
        if wired_limit_mb is not None
        and recommended_working_set_bytes is not None
        and physical_memory_bytes is not None
        else None
    )
    double_suggestion = (
        double_deduction_suggestion(
            physical_memory_bytes=physical_memory_bytes,
            recommended_working_set_bytes=recommended_working_set_bytes,
            compressor_pages_occupied=compressor_pages_occupied,
            vm_page_size_bytes=vm_page_size_bytes,
        )
        if physical_memory_bytes is not None and recommended_working_set_bytes is not None
        else None
    )

    if hit_estimate.method == "power_law":
        label = "estimate(model=power_law, alpha=0.30, anchored on M4/16GB)"
        # Validated against a real production measurement (see
        # decode_ceiling_tps()'s docstring): 0.549 predicted / 0.551
        # measured on the Mac mini M4/16GB -- high confidence.
        hit_confidence = "high confidence, validated 0.549 predicted/0.551 measured on M4/16GB"
    elif hit_estimate.method == "trace":
        label = "estimate(model=lru_trace_simulation, exact replay)"
        hit_confidence = "high confidence, exact trace replay"
    else:
        label = "exact(full_working_set_resident)"
        hit_confidence = "high confidence, exact (full residency)"

    # decode_ceiling_tps is only an I/O-bandwidth theoretical upper bound,
    # NOT a throughput prediction -- see decode_ceiling_tps()'s docstring
    # and PerformanceAdvisory's RELIABILITY NOTE. Explicitly low confidence
    # relative to hit_confidence above: on the same Mac mini M4/16GB
    # measurement, this ceiling predicted 4.455 tok/s vs only 2.43 tok/s
    # actually measured (CPU overhead / memory-compressor thrashing, not I/O,
    # dominated), while the hit-rate estimate above was accurate.
    if ceiling == float("inf"):
        ceiling_confidence = "no I/O-bound ceiling (full residency)"
    else:
        ceiling_confidence = (
            "upper bound, low confidence -- actual may be much lower "
            "(measured 2.43 vs predicted 4.455 tok/s on M4/16GB)"
        )

    log_line = (
        f"{label} budget={expert_budget_bytes} total={working_set.total_bytes} "
        f"fraction={fraction:.4f} bandwidth={bandwidth_bytes_per_sec:.3e} "
        f"per_token_bytes={working_set.per_token_full_miss_bytes:.0f} "
        f"hit={hit_estimate.hit_rate:.4f} ({hit_confidence}) "
        f"decode_ceiling_tps={ceiling:.3f} ({ceiling_confidence})"
    )

    return PerformanceAdvisory(
        resident_fraction=fraction,
        hit_rate_estimate=hit_estimate,
        decode_ceiling_tps=json_safe_ceiling,
        covers_full_working_set=covers_full_working_set,
        budget_bytes_for_target_hit=needed_budget,
        target_hit_rate=target_hit_rate,
        four_bit_repack_working_set_bytes=four_bit_bytes,
        wired_limit_suggestion=wired_suggestion,
        double_deduction_suggestion=double_suggestion,
        log_line=log_line,
    )


def probe_read_bandwidth(
    manifest: ModelManifest,
    *,
    max_bytes: int = _DEFAULT_PROBE_MAX_BYTES,
    max_seconds: float = _DEFAULT_PROBE_MAX_SECONDS,
    sample_count: int = _DEFAULT_PROBE_SAMPLE_COUNT,
    store_factory: Callable[[], SafetensorsExpertStore] = SafetensorsExpertStore,
) -> float | None:
    """Measure real SSD read bandwidth from ``sample_count`` bundles spread
    evenly across the manifest's disk offsets (~53MB for the default 16
    bundles at the Mac mini's ~3.34MB mean bundle size).

    Always opens its OWN throwaway ``store_factory()`` instance (default
    :class:`SafetensorsExpertStore`) and closes it before returning. This
    probe must never share, or read through, the long-lived runtime's
    store: doing so would pollute the runtime's ``StorageReadMetrics``
    (``storage/safetensors_store.py``) -- the very ``bytes_read`` /
    ``cache_hit_rate`` counters this whole investigation depends on being
    accurate.

    Enforced with a check-before-each-read wall-clock deadline: once
    ``max_seconds`` has elapsed, no further reads are issued. This bounds
    runaway probes to at most one bundle's own read latency past the cap;
    it is not a hard preemptive timeout on an in-flight read (a real
    preemptive cutoff would require abandoning a blocked OS thread, which
    risks hanging interpreter shutdown far worse than a slow probe would).

    Returns ``None`` (falling back to a caller-chosen default) if the
    manifest has no bundles, every read is cut off before completing a
    single bundle, or any read fails.
    """

    bundles = sorted(
        manifest.expert_bundles.values(),
        key=lambda bundle: min(tensor.offset for tensor in bundle.tensors),
    )
    if not bundles:
        return None
    if sample_count >= len(bundles):
        indices: list[int] = list(range(len(bundles)))
    else:
        step = len(bundles) / sample_count
        indices = sorted({int(i * step) for i in range(sample_count)})

    store = store_factory()
    total_bytes = 0
    start = time.monotonic()
    try:
        for index in indices:
            if time.monotonic() - start >= max_seconds:
                break
            bundle = bundles[index]
            if total_bytes + bundle.total_bytes > max_bytes:
                break
            try:
                store.read_bundle(bundle)
            except (StorageReadError, OSError) as error:
                _logger.debug("I/O bandwidth probe read failed, aborting probe: %s", error)
                break
            total_bytes += bundle.total_bytes
    finally:
        store.close()
    elapsed = time.monotonic() - start
    if total_bytes <= 0 or elapsed <= 0:
        return None
    return total_bytes / elapsed


def parse_startup_io_probe(value: str | None) -> str | float:
    """Parse ``--startup-io-probe {auto,off,<bytes/s>}``.

    Returns the literal string ``"auto"`` or ``"off"``, or a parsed
    ``float`` bytes/sec for an explicit numeric override.
    """

    if value is None or value.strip().lower() == "auto":
        return "auto"
    if value.strip().lower() == "off":
        return "off"
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"--startup-io-probe must be 'auto', 'off', or a bytes/sec number; got {value!r}"
        ) from error
    if parsed <= 0:
        raise ValueError("--startup-io-probe bytes/sec must be greater than zero")
    return parsed


def resolve_startup_bandwidth_bytes_per_sec(
    probe_setting: str | float, manifest: ModelManifest
) -> tuple[float, str]:
    """Resolve the bandwidth advisor input from a parsed ``--startup-io-probe`` value.

    Returns ``(bandwidth_bytes_per_sec, source)`` where ``source`` is
    ``"probed"``, ``"default"``, or ``"explicit"`` for logging.
    """

    if probe_setting == "off":
        return DEFAULT_BANDWIDTH_BYTES_PER_SEC, "default"
    if probe_setting == "auto":
        probed = probe_read_bandwidth(manifest)
        if probed is None:
            return DEFAULT_BANDWIDTH_BYTES_PER_SEC, "default"
        return probed, "probed"
    return float(probe_setting), "explicit"
