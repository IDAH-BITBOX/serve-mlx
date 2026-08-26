"""Shared adapter startup helpers.

Every family adapter's ``load_shell`` (or ``load_vlm_streaming``) follows the
exact same sequence right after its non-expert shell weights are loaded and
quantized: measure the live shell footprint, plan the M7 memory budget, then
build whichever of the two expert runtimes the plan selected.  This module
extracts that sequence so the four call sites do not hand-duplicate it, and
so later hardware-adaptive startup work (warmup / budget reporting) has a
single insertion point instead of four.

M13 adds two things at that same single insertion point: ``decide_startup``
picks one of three startup modes from the exact M7 budget (never an
estimate), and ``prepare_streaming_runtime`` runs the M13 Node 6 expert-cache
warmup when that mode is ``"full_residency"``. Both are exercised by every
family adapter automatically because they only need the manifest, the
already-built runtime, and the already-planned budget -- never the model --
so they fit before ``replace_moe_blocks`` runs, without needing the model
object this function is never handed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .advisor import build_advisory, resolve_startup_bandwidth_bytes_per_sec
from .cache import ExpertKey
from .hardware import HardwareProfile, probe_hardware
from .kv_cache import KvCacheConfig, KvCacheDecision, make_memory_manager
from .manifest import ExpertWorkingSet, ModelManifest
from .memory import MemoryBudgetConfig, MemoryBudgetDecision, MemoryBudgetManager
from .runtime import CachedExpertRuntime, NoCacheExpertRuntime

_logger = logging.getLogger(__name__)

StartupMode = Literal["full_residency", "streaming", "no_cache"]


@dataclass(frozen=True)
class StartupAdvisorInput:
    """The bandwidth/quantization facts ``decide_startup`` needs from a caller.

    Everything else ``build_advisory`` needs (``expert_budget_bytes``,
    ``working_set``, ``wired_limit_mb``, physical/recommended memory) already
    lives on ``budget`` and ``hardware``, which ``decide_startup`` receives
    directly -- this only carries the two inputs that come from a probe
    (``--startup-io-probe``) or the manifest's own quantization metadata.
    """

    bandwidth_bytes_per_sec: float
    bandwidth_source: str
    current_quantization_bits: int | None = None
    target_hit_rate: float = 0.9


@dataclass(frozen=True)
class StartupDecision:
    """The M13 exact-inequality branch plus everything a caller needs to act on it.

    ``mode`` is decided by ONE exact byte comparison -- never an estimate:
    ``expert_budget_bytes >= working_set.total_bytes`` means the resident
    cache can hold every expert bundle at once (``"full_residency"``); a
    smaller but nonzero budget is ``"streaming"``; no cache at all
    (``expert_budget_bytes is None``) is ``"no_cache"``.

    ``warmup_keys`` is only populated for ``"full_residency"``: warming a
    cache too small to hold the whole working set is pure LRU churn (M13
    deliberately skips it for ``"streaming"``).

    ``report`` is a JSON-safe dict (used by ``--startup-report`` and
    ``/metrics``' ``"startup"`` key) with every number a caller would want to
    log or persist: the M7 budget breakdown, the M11 advisory numbers, and
    the suggestions ``build_advisory`` produced.
    """

    mode: StartupMode
    warmup_keys: tuple[ExpertKey, ...]
    report: dict[str, Any]


def decide_startup(
    hardware: HardwareProfile,
    working_set: ExpertWorkingSet,
    budget: MemoryBudgetDecision,
    advisor_input: StartupAdvisorInput,
    *,
    expert_keys: Sequence[ExpertKey] = (),
) -> StartupDecision:
    """Pick the M13 startup mode and build its advisory report.

    Pure and side-effect free (no logging): callers decide what to do with
    ``mode``/``warmup_keys`` (trigger a real warmup, skip one) and what to
    log or persist from ``report``. Kept pure so the three-mode branch stays
    directly unit-testable without capturing logs.

    ``expert_keys`` is accepted as a keyword-only extra rather than folded
    into ``working_set`` (an aggregate with no per-bundle keys) so a caller
    that already has the manifest's bundle keys can hand them over for
    ``warmup_keys`` without this function needing the whole manifest. This
    is a deliberate small addition to the ``decide_startup(hardware,
    working_set, budget, advisor_input)`` signature from the task spec --
    the four positional/keyword inputs are unchanged, ``expert_keys``
    defaults to ``()`` so every existing call shape still works, and there
    is no other way to hand back concrete keys to warm.
    """

    expert_budget = budget.expert_budget_bytes
    if expert_budget is None:
        mode: StartupMode = "no_cache"
    elif expert_budget >= working_set.total_bytes:
        mode = "full_residency"
    else:
        mode = "streaming"

    advisory = build_advisory(
        expert_budget_bytes=expert_budget,
        working_set=working_set,
        bandwidth_bytes_per_sec=advisor_input.bandwidth_bytes_per_sec,
        target_hit_rate=advisor_input.target_hit_rate,
        current_quantization_bits=advisor_input.current_quantization_bits,
        wired_limit_mb=hardware.wired_limit_mb,
        physical_memory_bytes=hardware.physical_memory_bytes,
        recommended_working_set_bytes=hardware.recommended_working_set_bytes,
        compressor_pages_occupied=hardware.compressor_pages_occupied,
        vm_page_size_bytes=hardware.vm_page_size_bytes,
    )

    warmup_keys = tuple(expert_keys) if mode == "full_residency" else ()

    report: dict[str, Any] = {
        "mode": mode,
        "explain": budget.explain(),
        "budget": budget.to_dict(),
        "working_set": working_set.to_dict(),
        "bandwidth_bytes_per_sec": advisor_input.bandwidth_bytes_per_sec,
        "bandwidth_source": advisor_input.bandwidth_source,
        "resident_fraction": advisory.resident_fraction,
        "hit_rate": advisory.hit_rate_estimate.hit_rate,
        "hit_rate_method": advisory.hit_rate_estimate.method,
        "decode_ceiling_tps": advisory.decode_ceiling_tps,
        "log_line": advisory.log_line,
        "target_hit_rate": advisory.target_hit_rate,
        "budget_bytes_for_target_hit": advisory.budget_bytes_for_target_hit,
        "four_bit_repack_working_set_bytes": advisory.four_bit_repack_working_set_bytes,
        "wired_limit_suggestion": advisory.wired_limit_suggestion,
        "double_deduction_suggestion": advisory.double_deduction_suggestion,
        "warmup_key_count": len(warmup_keys),
    }
    return StartupDecision(mode=mode, warmup_keys=warmup_keys, report=report)


@dataclass
class StartupResult:
    """Everything a ``load_shell`` needs after planning the M7 memory budget.

    ``shell_bytes`` is carried alongside ``memory_budget`` (which already
    stores the same value) so a startup report can print the shell/expert/KV
    split without reaching into the budget decision. ``decision`` is the
    M13 ``decide_startup`` result this same call already acted on -- it
    either ran a warmup for ``"full_residency"`` or skipped one for
    ``"streaming"``/``"no_cache"`` -- so a caller can log or persist
    ``decision.report`` again without recomputing anything.
    """

    memory_manager: MemoryBudgetManager
    kv_cache: KvCacheDecision | None
    memory_budget: MemoryBudgetDecision
    runtime: NoCacheExpertRuntime | CachedExpertRuntime
    shell_bytes: int
    decision: StartupDecision


def prepare_streaming_runtime(
    manifest: ModelManifest,
    *,
    shell_bytes: int,
    model_config: dict[str, Any],
    resident_budget_bytes: int | None,
    auto_resident_budget: bool,
    memory_config: MemoryBudgetConfig | None,
    kv_cache_config: KvCacheConfig | None,
    runtime_options: dict[str, Any],
    startup_io_probe: str | float = "auto",
    warmup: Literal["auto", "off"] = "auto",
    warmup_timeout_seconds: float = 300.0,
) -> StartupResult:
    """Plan the M7 memory budget and build the matching expert runtime.

    ``shell_bytes`` must already reflect the actual quantized non-expert
    shell (measured via ``mx.get_active_memory()``) -- this is deliberately
    not a model config estimate, since MLX layout and quantization determine
    the live Unified Memory footprint.

    ``runtime_options`` carries every per-adapter runtime kwarg (``io_workers``,
    ``prefetch_depth``, ``async_gpu``, ``predictor``, ``predictive_config``,
    and optionally ``expert_activation``) except ``memory_manager``, which
    this function supplies itself once planning is complete.

    M13 additions, all optional with defaults that preserve the M7 behavior:
    ``startup_io_probe`` feeds the M11 bandwidth advisor exactly like
    ``--startup-io-probe`` (``"auto"`` probes the manifest, ``"off"`` uses
    the M4 default, or an explicit bytes/sec float); ``warmup`` is
    ``"auto"`` (run the Node 6 warmup when ``decide_startup`` picks
    ``"full_residency"``) or ``"off"`` (never warm, regardless of mode);
    ``warmup_timeout_seconds`` bounds the warmup wall clock (default 300s,
    matching ``--warmup-timeout``).

    This is deliberately the ONLY place any of the M13 mode decision or
    warmup runs: it sits after every family adapter measures its quantized
    shell and before any of them calls ``replace_moe_blocks``, so one call
    here covers all four adapters (``qwen3_moe``, ``qwen3_5_moe``,
    ``gemma4``, ``vlm``) without duplicating the decision or the warmup loop
    at each call site.
    """

    memory_manager, kv_cache = make_memory_manager(
        memory_config,
        kv_cache_config,
        model_config=model_config,
        shell_bytes=shell_bytes,
    )
    memory_budget = memory_manager.plan(
        shell_bytes=shell_bytes,
        requested_expert_budget_bytes=resident_budget_bytes,
        auto_enabled=auto_resident_budget,
        minimum_expert_bytes=max(bundle.total_bytes for bundle in manifest.expert_bundles.values()),
    )
    options = dict(runtime_options)
    options["memory_manager"] = memory_manager
    runtime: NoCacheExpertRuntime | CachedExpertRuntime
    if memory_budget.expert_budget_bytes is None:
        runtime = NoCacheExpertRuntime(manifest, **options)
    else:
        runtime = CachedExpertRuntime(
            manifest,
            capacity_bytes=memory_budget.expert_budget_bytes,
            **options,
        )

    _logger.info("M7 memory budget: %s", memory_budget.explain())

    # runtime above already holds a real SafetensorsExpertStore mmap plus
    # (when io_workers > 0) live I/O worker threads. _decide_and_act's own
    # exception handling degrades a failed M13 advisor to a minimal decision
    # (see below) so it should never raise for that reason alone, but this
    # is a second, independent guarantee: whatever does escape it here still
    # must not leak the runtime this function already built.
    try:
        decision = _decide_and_act(
            manifest=manifest,
            runtime=runtime,
            memory_budget=memory_budget,
            startup_io_probe=startup_io_probe,
            warmup=warmup,
            warmup_timeout_seconds=warmup_timeout_seconds,
        )
    except BaseException:
        runtime.close()
        raise

    return StartupResult(
        memory_manager=memory_manager,
        kv_cache=kv_cache,
        memory_budget=memory_budget,
        runtime=runtime,
        shell_bytes=shell_bytes,
        decision=decision,
    )


def _decide_and_act(
    *,
    manifest: ModelManifest,
    runtime: NoCacheExpertRuntime | CachedExpertRuntime,
    memory_budget: MemoryBudgetDecision,
    startup_io_probe: str | float,
    warmup: Literal["auto", "off"],
    warmup_timeout_seconds: float,
) -> StartupDecision:
    """Build the M13 ``StartupDecision`` and run (or skip) its warmup.

    Split out of ``prepare_streaming_runtime`` only for readability; nothing
    here is reused elsewhere, so it stays private.
    """

    try:
        working_set = manifest.expert_working_set()
        hardware = probe_hardware(memory_budget.snapshot)
        bandwidth_bytes_per_sec, bandwidth_source = resolve_startup_bandwidth_bytes_per_sec(
            startup_io_probe, manifest
        )
        advisor_input = StartupAdvisorInput(
            bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
            bandwidth_source=bandwidth_source,
            current_quantization_bits=manifest.quantization.bits,
        )
        decision = decide_startup(
            hardware,
            working_set,
            memory_budget,
            advisor_input,
            expert_keys=tuple(manifest.expert_bundles.keys()),
        )
    except Exception:  # noqa: BLE001 - the M11 advisor is log-only; it must never block engine startup
        # Unlike the warmup calls below, nothing here was wrapped before:
        # manifest.expert_working_set(), probe_hardware(),
        # resolve_startup_bandwidth_bytes_per_sec(), and decide_startup()/
        # build_advisory() can all raise (bad manifest metadata, a hardware
        # probe failure, an out-of-range fraction, current_bits <= 0, ...).
        # Degrade to the smallest decision that still lets the engine load:
        # no warmup (we no longer trust the inputs enough to pick
        # "full_residency" and warm the whole working set), but preserve
        # whatever mode the M7 budget alone (already planned, already
        # trusted) implies so downstream mode-dispatch below still works.
        fallback_mode: StartupMode = (
            "no_cache" if memory_budget.expert_budget_bytes is None else "streaming"
        )
        _logger.warning(
            "M13 startup advisor failed while planning; degrading to mode=%s with no "
            "warmup so the engine still loads",
            fallback_mode,
            exc_info=True,
        )
        decision = StartupDecision(
            mode=fallback_mode,
            warmup_keys=(),
            report={
                "mode": fallback_mode,
                "explain": memory_budget.explain(),
                "budget": memory_budget.to_dict(),
                "advisor_error": "M13 startup advisor failed; see logs for details",
            },
        )

    if warmup == "off":
        _logger.info("M13 startup warmup disabled via --warmup off")
        return decision

    if decision.mode == "full_residency":
        try:
            stats = runtime.warmup(
                decision.warmup_keys,
                deadline=time.monotonic() + warmup_timeout_seconds,
                active_memory_ceiling=memory_budget.safe_working_set_bytes,
            )
        except Exception:  # noqa: BLE001 - a warmup failure must never fail engine startup
            _logger.warning(
                "M13 startup warmup raised; continuing without a full warmup", exc_info=True
            )
        else:
            if stats is not None:
                _logger.info(
                    "M13 startup warmup: requested=%s admitted=%s bytes=%s elapsed=%.2fs "
                    "stop_reason=%s reader_errors=%s",
                    stats.requested,
                    stats.admitted,
                    stats.bytes_admitted,
                    stats.elapsed_seconds,
                    stats.stop_reason,
                    stats.reader_errors,
                )
    elif decision.mode == "no_cache":
        try:
            runtime.warmup(())
        except Exception:  # noqa: BLE001 - see above
            _logger.warning("M13 startup warmup call failed", exc_info=True)
    else:
        # report is the full build_advisory() report in the normal case,
        # but may be the minimal degraded dict from the except block above
        # (no advisory keys at all) -- every access here is defensive so a
        # degraded streaming decision still logs cleanly instead of raising
        # a KeyError out of what is already an exception-recovery path.
        report = decision.report
        four_bit_bytes = report.get("four_bit_repack_working_set_bytes")
        target_hit_rate = report.get("target_hit_rate")
        needed_budget_bytes = report.get("budget_bytes_for_target_hit")
        _logger.warning(
            "M13 streaming mode: the resident expert cache cannot hold the full expert "
            "working set; warmup skipped (warming a cache this small would just be "
            "repeatedly evicted) | %s | %s | suggestions: target_hit_%s_needs_budget=%s "
            "four_bit_repack_working_set=%s wired_limit=%s safety_margin=%s",
            memory_budget.explain(),
            report.get("log_line", "n/a"),
            (f"{target_hit_rate:.2f}" if target_hit_rate is not None else "n/a"),
            (
                f"{needed_budget_bytes / (1024**3):.3f}GiB"
                if needed_budget_bytes is not None
                else "n/a"
            ),
            (f"{four_bit_bytes / (1024**3):.3f}GiB" if four_bit_bytes is not None else "n/a"),
            report.get("wired_limit_suggestion") or "n/a",
            report.get("double_deduction_suggestion") or "n/a",
        )

    return decision
