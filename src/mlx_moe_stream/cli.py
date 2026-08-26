"""Command line interface for the implemented M0–M8.5 milestones."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .advisor import parse_startup_io_probe
from .cache import MemoryBudgetError
from .config import parse_bytes, parse_resident_budget
from .errors import MlxMoeStreamError
from .kv_cache import KvCacheConfig
from .logging import (
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_MAX_BYTES,
    configure_logging,
)
from .memory import (
    MemoryBudgetConfig,
    adaptive_safety_margin_bytes,
    automatic_safety_margin_bytes,
    collect_memory_snapshot,
)
from .models import load_streaming_model
from .prefetch import (
    PredictivePrefetchConfig,
    load_transition_predictor,
    train_transition_predictor,
)
from .routing import load_trace, summarize_trace, trace_qwen3_generation, write_summary
from .server import (
    DEFAULT_CONNECTION_TIMEOUT,
    LocalApiServer,
    LocalGenerationService,
    ModelRegistration,
    ModelRegistry,
    ServerConfig,
    is_loopback_host,
    run_local_server,
)
from .startup import StartupDecision
from .storage import build_streaming_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-moe-stream", description="MLX out-of-core MoE tools")
    parser.add_argument("--verbose", action="store_true", help="emit diagnostic logs")
    commands = parser.add_subparsers(dest="command", required=True)

    trace = commands.add_parser(
        "trace", help="trace Qwen3-MoE routes using normal mlx-lm execution"
    )
    trace.add_argument(
        "--model", required=True, help="local MLX model path or Hugging Face repository"
    )
    trace.add_argument("--prompt", required=True, help="prompt to execute")
    trace.add_argument("--max-tokens", type=int, default=64, help="number of decode calls to trace")
    trace.add_argument("--prefill-step-size", type=int, default=2048)
    trace.add_argument("--request-id", help="stable identifier written into the JSONL trace")
    trace.add_argument("--output", type=Path, required=True, help="destination JSONL trace file")
    trace.add_argument("--summary", type=Path, help="optional destination summary JSON file")

    simulate = commands.add_parser(
        "simulate", help="simulate global LRU locality from a JSONL trace"
    )
    simulate.add_argument(
        "--trace", type=Path, required=True, help="route JSONL emitted by the trace command"
    )
    simulate.add_argument("--output", type=Path, help="optional summary JSON destination")

    train_predictor = commands.add_parser(
        "train-predictor", help="train an M10 next-layer expert predictor from route traces"
    )
    train_predictor.add_argument(
        "--trace", type=Path, action="append", required=True, help="M1 route JSONL; repeatable"
    )
    train_predictor.add_argument(
        "--model-type",
        choices=("qwen3_moe", "qwen3_5_moe", "gemma4"),
        required=True,
        help="must match the prepared manifest that will use this predictor",
    )
    train_predictor.add_argument("--output", type=Path, required=True)
    train_predictor.add_argument("--overwrite", action="store_true")

    prepare = commands.add_parser(
        "prepare", help="inspect a supported MoE checkpoint and write an exact-read manifest"
    )
    prepare.add_argument(
        "--model", required=True, help="local model path or Hugging Face repository"
    )
    prepare.add_argument("--output", type=Path, required=True, help="directory for manifest.json")
    prepare.add_argument(
        "--overwrite", action="store_true", help="replace an existing manifest.json"
    )

    generate = commands.add_parser(
        "generate", help="run exact streamed-MoE generation from a prepared manifest"
    )
    generate.add_argument("--manifest", type=Path, required=True, help="M2 manifest.json path")
    generate.add_argument("--prompt", required=True, help="prompt to execute")
    generate.add_argument("--max-tokens", type=int, default=64, help="maximum tokens to generate")
    generate.add_argument(
        "--resident-budget",
        help=(
            "resident expert-cache capacity: auto, off, or a byte size (for example 2GB or 2048MiB)"
        ),
    )
    generate.add_argument(
        "--memory-safety-margin",
        default="auto",
        help=(
            "M7 Unified Memory kept free from expert cache; auto reserves 25%% of physical "
            "memory, adaptive reserves 25%% minus what the OS already withheld from "
            "the recommended working set (default: auto)"
        ),
    )
    generate.add_argument(
        "--max-unified-memory",
        help=(
            "explicit Unified Memory working-set ceiling for the M7 budget, replacing "
            "the OS-recommended working set; a bare number is GB (for example 14), or "
            "use a sized value such as 14GiB (default: unset, uses the OS recommendation)"
        ),
    )
    generate.add_argument(
        "--kv-reserve",
        default="1GB",
        help="minimum M7 reservation for KV cache growth (default: 1GB)",
    )
    _add_kv_cache_options(generate, include_max_context=True)
    generate.add_argument(
        "--scratch-reserve",
        default="1GB",
        help="M7 reserve for transient MLX working memory (default: 1GB)",
    )
    generate.add_argument(
        "--wired-limit",
        help="optional M7 MLX wired-memory limit; unset by default",
    )
    generate.add_argument(
        "--memory-summary",
        type=Path,
        help="optional JSON destination for M7 startup, final, and pressure memory metrics",
    )
    generate.add_argument(
        "--startup-io-probe",
        default="auto",
        help=(
            "M11 SSD read-bandwidth input for the performance advisor: auto probes "
            "~53MB from the manifest at startup, off skips probing and uses the M4 "
            "default (2.0GB/s), or supply an explicit bytes/sec number (default: auto)"
        ),
    )
    _add_startup_options(generate, include_io_probe=False)
    generate.add_argument(
        "--prefill-strategy",
        choices=("expert_major", "token_major"),
        default="expert_major",
        help="M5 prefill scheduler (default: expert_major)",
    )
    generate.add_argument(
        "--prefill-order",
        choices=("resident_first", "expert_id", "disk_offset"),
        default="resident_first",
        help="expert-major group order (default: resident_first)",
    )
    generate.add_argument(
        "--io-workers",
        type=int,
        default=0,
        help="enable M6 bounded exact-read workers (0 disables overlap)",
    )
    generate.add_argument(
        "--prefetch-depth",
        type=int,
        default=1,
        help="number of known routed experts to read ahead per M6 scheduler step",
    )
    generate.add_argument(
        "--async-gpu",
        action="store_true",
        help="enqueue M6 expert kernels with MLX async_eval (requires --io-workers)",
    )
    generate.add_argument(
        "--timeline",
        type=Path,
        help="optional JSON destination for M6 load/materialize/GPU events",
    )
    _add_predictive_prefetch_options(generate)

    serve = commands.add_parser("serve", help="run the M8 localhost OpenAI-compatible API")
    serve_inputs = serve.add_mutually_exclusive_group(required=True)
    serve_inputs.add_argument(
        "--manifest",
        type=Path,
        help="single-model compatibility mode: prepared manifest.json path",
    )
    serve_inputs.add_argument(
        "--model",
        action="append",
        metavar="MODEL_ID=MANIFEST",
        help="M9 registration; repeat to expose multiple lazy-loaded models",
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="loopback bind address (default: 127.0.0.1)"
    )
    serve.add_argument("--port", type=int, default=8000, help="listening port (default: 8000)")
    serve.add_argument(
        "--model-id",
        help="single-model ID, or default model ID when using repeated --model",
    )
    serve.add_argument(
        "--resident-budget",
        default=None,
        help=(
            "M4/M7 expert-cache capacity: auto, off, or a byte size "
            "(default: auto for text; off with --vision)"
        ),
    )
    serve.add_argument(
        "--memory-safety-margin",
        default="auto",
        help=(
            "M7 Unified Memory kept free from expert cache; auto reserves 25%% of physical "
            "memory, adaptive reserves 25%% minus what the OS already withheld from "
            "the recommended working set (default: auto)"
        ),
    )
    serve.add_argument(
        "--max-unified-memory",
        help=(
            "explicit Unified Memory working-set ceiling for the M7 budget, replacing "
            "the OS-recommended working set; a bare number is GB (for example 14), or "
            "use a sized value such as 14GiB (default: unset, uses the OS recommendation)"
        ),
    )
    serve.add_argument(
        "--kv-reserve",
        default="1GB",
        help="minimum M7 reservation for KV cache growth (default: 1GB)",
    )
    _add_kv_cache_options(serve, include_max_context=False)
    serve.add_argument(
        "--scratch-reserve",
        default="1GB",
        help="M7 transient-workspace reservation (VLM uses at least 2GiB)",
    )
    serve.add_argument("--wired-limit")
    _add_startup_options(serve, include_io_probe=True)
    preload_group = serve.add_mutually_exclusive_group()
    preload_group.add_argument(
        "--preload",
        dest="preload",
        action="store_true",
        default=None,
        help=(
            "M13 activate the default model before serving; default is on for a single "
            "registered model, off for repeated --model registrations"
        ),
    )
    preload_group.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="M13 never activate a model before the first request",
    )
    serve.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4_096,
        help="reject requests with more prompt tokens (default: 4096)",
    )
    serve.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="maximum completion tokens per request (default: 256)",
    )
    serve.add_argument(
        "--max-request-bytes",
        default="1MB",
        help="maximum JSON request body size (default: 1MB)",
    )
    serve.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help=(
            "tokens per prefill chunk; lower values reduce long-context MoE peak memory "
            "at the cost of speed (default: 2048 text; 256 with --vision)"
        ),
    )
    serve.add_argument(
        "--vision",
        action="store_true",
        help="M12: load a supported Qwen3.5/Gemma4 vision tower for image chat",
    )
    serve.add_argument(
        "--prefill-strategy",
        choices=("expert_major", "token_major"),
        default="expert_major",
    )
    serve.add_argument(
        "--prefill-order",
        choices=("resident_first", "expert_id", "disk_offset"),
        default="resident_first",
    )
    serve.add_argument("--io-workers", type=int, default=0)
    serve.add_argument("--prefetch-depth", type=int, default=1)
    serve.add_argument("--async-gpu", action="store_true")
    serve.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="also write rotating logs here instead of relying on shell redirection",
    )
    serve.add_argument(
        "--log-max-bytes",
        type=int,
        default=DEFAULT_LOG_MAX_BYTES,
        help=f"rotate --log-file above this size (default: {DEFAULT_LOG_MAX_BYTES})",
    )
    serve.add_argument(
        "--log-backups",
        type=int,
        default=DEFAULT_LOG_BACKUP_COUNT,
        help=f"rotated --log-file copies to keep (default: {DEFAULT_LOG_BACKUP_COUNT})",
    )
    serve.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="write the server pid here and remove it on a clean shutdown",
    )
    serve.add_argument(
        "--connection-timeout",
        type=float,
        default=DEFAULT_CONNECTION_TIMEOUT,
        help=(
            "seconds a single socket operation may block; 0 disables the timeout "
            f"(default: {DEFAULT_CONNECTION_TIMEOUT:g})"
        ),
    )
    _add_predictive_prefetch_options(serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(
        args.verbose,
        log_file=getattr(args, "log_file", None),
        max_bytes=getattr(args, "log_max_bytes", DEFAULT_LOG_MAX_BYTES),
        backup_count=getattr(args, "log_backups", DEFAULT_LOG_BACKUP_COUNT),
    )
    try:
        if args.command == "trace":
            tracer = trace_qwen3_generation(
                args.model,
                args.prompt,
                max_tokens=args.max_tokens,
                output_path=args.output,
                request_id=args.request_id,
                prefill_step_size=args.prefill_step_size,
            )
            summary = summarize_trace(tracer.events)
            if args.summary is not None:
                write_summary(args.summary, summary)
            logger.info("wrote %s route events to %s", len(tracer.events), args.output)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "simulate":
            summary = summarize_trace(load_trace(args.trace))
            if args.output is not None:
                write_summary(args.output, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "train-predictor":
            events = [event for trace_path in args.trace for event in load_trace(trace_path)]
            predictor = train_transition_predictor(events, model_type=args.model_type)
            predictor.write(args.output, overwrite=args.overwrite)
            payload = {
                "predictor": str(args.output),
                "model_type": predictor.model_type,
                "num_layers": predictor.num_layers,
                "num_experts": predictor.num_experts,
                "source_experts": sum(len(sources) for sources in predictor.transitions.values()),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "prepare":
            manifest = build_streaming_manifest(args.model)
            manifest_path = args.output / "manifest.json"
            manifest.write(manifest_path, overwrite=args.overwrite)
            payload = {
                "manifest": str(manifest_path),
                "model_type": manifest.model_type,
                "bundles": len(manifest.expert_bundles),
                "bytes_per_expert": {
                    "min": min(bundle.total_bytes for bundle in manifest.expert_bundles.values()),
                    "max": max(bundle.total_bytes for bundle in manifest.expert_bundles.values()),
                },
            }
            logger.info(
                "wrote %s expert bundles to %s", len(manifest.expert_bundles), manifest_path
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "generate":
            if args.max_tokens < 0:
                raise ValueError("--max-tokens must be zero or greater")
            resident_budget, auto_resident_budget = parse_resident_budget(args.resident_budget)
            memory_config = _memory_config(args)
            predictor, predictive_config = _predictive_options(args)
            kv_cache_config = KvCacheConfig(
                mode=args.kv_cache,
                max_context_tokens=args.kv_max_context,
            )
            # Parsed before the (potentially expensive) model load so a bad
            # --startup-io-probe value fails fast instead of after loading.
            probe_setting = parse_startup_io_probe(args.startup_io_probe)
            engine = load_streaming_model(
                args.manifest,
                resident_budget_bytes=resident_budget,
                auto_resident_budget=auto_resident_budget,
                memory_config=memory_config,
                kv_cache_config=kv_cache_config,
                prefill_strategy=args.prefill_strategy,
                prefill_order=args.prefill_order,
                io_workers=args.io_workers,
                prefetch_depth=args.prefetch_depth,
                async_gpu=args.async_gpu,
                predictor=predictor,
                predictive_config=predictive_config,
                startup_io_probe=probe_setting,
                warmup=args.warmup,
                warmup_timeout_seconds=args.warmup_timeout,
            )
            timeline = ()
            final_snapshot = None
            try:
                output = engine.generate(args.prompt, max_tokens=args.max_tokens)
                final_snapshot = engine.memory_manager.snapshot()
            finally:
                engine.close()
            stats = engine.runtime.stats()
            timeline = engine.runtime.timeline()
            if args.timeline is not None:
                args.timeline.write_text(
                    json.dumps(
                        [
                            {
                                "name": event.name,
                                "timestamp": event.timestamp,
                                "layer": event.key.layer if event.key is not None else None,
                                "expert": event.key.expert if event.key is not None else None,
                            }
                            for event in timeline
                        ],
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if args.memory_summary is not None:
                args.memory_summary.write_text(
                    json.dumps(
                        {
                            "budget": engine.memory_budget.to_dict(),
                            "kv_cache": (
                                engine.kv_cache.to_dict() if engine.kv_cache is not None else None
                            ),
                            "final_snapshot": (
                                final_snapshot.to_dict() if final_snapshot is not None else None
                            ),
                            "pressure_events": [event.to_dict() for event in stats.memory_events],
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            print(output)
            budget = engine.memory_budget
            logger.info(
                "M7 memory budget: source=%s shell=%s safe_working_set=%s "
                "available_expert=%s cache_budget=%s",
                budget.source,
                budget.shell_bytes,
                budget.safe_working_set_bytes,
                budget.available_expert_bytes,
                budget.expert_budget_bytes,
            )
            # Reuse the StartupDecision prepare_streaming_runtime already
            # computed during load_streaming_model() above (engine.startup_decision)
            # instead of re-probing bandwidth/hardware and recomputing decide_startup
            # here. Re-running the M11 advisor after load previously repeated the
            # ~53MB --startup-io-probe auto read and re-ran probe_hardware, which
            # could produce a *different* bandwidth reading than the one load already
            # planned around, making this log and --startup-report disagree with
            # engine.startup_decision.report.
            _report_startup_decision(logger, engine, args.startup_report)
            if engine.kv_cache is not None:
                logger.info(
                    "KV cache: requested=%s effective=%s context=%s estimate=%s reserve=%s",
                    engine.kv_cache.requested_mode,
                    engine.kv_cache.effective_mode,
                    engine.kv_cache.max_context_tokens,
                    engine.kv_cache.estimated_bytes,
                    engine.kv_cache.reserve_bytes,
                )
            if stats.memory_events:
                logger.warning(
                    "M7 pressure actions: %s",
                    ", ".join(event.action for event in stats.memory_events),
                )
            if stats.cache is None:
                logger.info(
                    "M3/M5 generation read %s bytes in %s reads for %s expert resolutions",
                    stats.bytes_read,
                    stats.read_count,
                    stats.expert_resolutions,
                )
            else:
                logger.info(
                    "M4/M5 resident cache: reads=%s bytes=%s resolutions=%s hits=%s misses=%s "
                    "hit_rate=%.2f%% resident=%s/%s evictions=%s reload_bytes=%s",
                    stats.read_count,
                    stats.bytes_read,
                    stats.expert_resolutions,
                    stats.cache.hit_count,
                    stats.cache.miss_count,
                    stats.cache.hit_rate * 100,
                    stats.cache.resident_bytes,
                    stats.cache.capacity_bytes,
                    stats.cache.eviction_count,
                    stats.cache.reload_bytes,
                )
            if stats.io_overlap is not None:
                io = stats.io_overlap
                logger.info(
                    "M6 overlap: workers=%s depth=%s async_gpu=%s demand=%s prefetch="
                    "%s submitted=%s hits=%s coalesced=%s skipped=%s timeline_events=%s",
                    io.workers,
                    io.prefetch_depth,
                    io.async_gpu,
                    io.loader.demand_requests,
                    io.loader.prefetch_requests,
                    io.loader.prefetch_submitted,
                    io.loader.prefetch_hits,
                    io.loader.coalesced_requests,
                    io.loader.skipped_prefetches,
                    len(timeline),
                )
            if stats.predictive_prefetch is not None:
                predictive = stats.predictive_prefetch
                logger.info(
                    "M10 predictive prefetch: calls=%s candidates=%s submitted=%s "
                    "used=%s unused=%s skipped(confidence=%s limit=%s bytes=%s runtime=%s)",
                    predictive.prediction_calls,
                    predictive.candidates_considered,
                    predictive.submitted,
                    stats.io_overlap.loader.predictive_hits if stats.io_overlap else 0,
                    stats.io_overlap.loader.predictive_unused if stats.io_overlap else 0,
                    predictive.skipped_confidence,
                    predictive.skipped_candidate_limit,
                    predictive.skipped_byte_budget,
                    predictive.skipped_runtime,
                )
            return 0
        if args.command == "serve":
            if not is_loopback_host(args.host):
                raise ValueError("M8 has no authentication and only permits a loopback --host")
            if not 1 <= args.port <= 65_535:
                raise ValueError("--port must be in 1..65535")
            resident_budget, auto_resident_budget = _serve_resident_budget(
                args.resident_budget, vision=args.vision
            )
            memory_config = _memory_config(
                args,
                minimum_scratch_reserve_bytes=(2 * 1024**3 if args.vision else 0),
            )
            predictor, predictive_config = _predictive_options(args)
            prefill_step_size = _serve_prefill_step_size(args.prefill_step_size, vision=args.vision)
            kv_cache_config = KvCacheConfig(
                mode=args.kv_cache,
                max_context_tokens=args.max_prompt_tokens + args.max_tokens,
            )
            # Parsed before any model can load so a bad --startup-io-probe value
            # fails fast, matching the M11 `generate` behavior.
            probe_setting = parse_startup_io_probe(args.startup_io_probe)
            registrations = _serve_registrations(
                args.manifest,
                args.model,
                single_model_id=args.model_id or "mlx-moe-stream",
            )
            default_model_id = args.model_id or registrations[0].model_id
            if default_model_id not in {registration.model_id for registration in registrations}:
                raise ValueError(f"--model-id {default_model_id!r} is not a registered model")
            server_config = ServerConfig(
                model_id=default_model_id,
                max_prompt_tokens=args.max_prompt_tokens,
                max_completion_tokens=args.max_tokens,
                max_request_bytes=parse_bytes(args.max_request_bytes),
                prefill_step_size=prefill_step_size,
            )
            registry = ModelRegistry(
                registrations,
                load_engine=lambda manifest_path: load_streaming_model(
                    manifest_path,
                    resident_budget_bytes=resident_budget,
                    auto_resident_budget=auto_resident_budget,
                    memory_config=memory_config,
                    kv_cache_config=kv_cache_config,
                    prefill_strategy=args.prefill_strategy,
                    prefill_order=args.prefill_order,
                    io_workers=args.io_workers,
                    prefetch_depth=args.prefetch_depth,
                    async_gpu=args.async_gpu,
                    vision=args.vision,
                    predictor=predictor,
                    predictive_config=predictive_config,
                    startup_io_probe=probe_setting,
                    warmup=args.warmup,
                    warmup_timeout_seconds=args.warmup_timeout,
                ),
            )
            service = LocalGenerationService(config=server_config, registry=registry)
            preload = _resolve_preload(args.preload, registration_count=len(registrations))
            try:
                if preload:
                    engine = registry.activate(default_model_id)
                    _report_startup_decision(logger, engine, args.startup_report)
                server = LocalApiServer(
                    args.host,
                    args.port,
                    service,
                    connection_timeout=args.connection_timeout or None,
                )
                bound_host, bound_port = server.server_address[:2]
                logger.info(
                    "M9 serving default=%s registered=%s at http://%s:%s "
                    "(one active generation, one active engine)",
                    server_config.model_id,
                    ",".join(registration.model_id for registration in registrations),
                    bound_host,
                    bound_port,
                )
                run_local_server(server, pid_file=args.pid_file)
            finally:
                service.close()
            return 0
        parser.error(
            f"'{args.command}' is not implemented until a later milestone; "
            "use 'trace' or 'simulate'"
        )
    except (MemoryBudgetError, MlxMoeStreamError, ValueError, OSError) as error:
        logger.error("%s", error)
        return 2
    return 0


def _serve_registrations(
    manifest: Path | None,
    model_specs: list[str] | None,
    *,
    single_model_id: str,
) -> tuple[ModelRegistration, ...]:
    if manifest is not None and model_specs:
        raise ValueError("use either --manifest or repeated --model, not both")
    if manifest is not None:
        return (ModelRegistration(single_model_id, manifest),)
    if not model_specs:
        raise ValueError("serve requires --manifest or at least one --model MODEL_ID=MANIFEST")
    return tuple(ModelRegistration.parse(value) for value in model_specs)


def _memory_config(
    args: argparse.Namespace, *, minimum_scratch_reserve_bytes: int = 0
) -> MemoryBudgetConfig:
    """Resolve one CLI memory policy before the model shell is loaded."""

    margin = args.memory_safety_margin
    normalized_margin = margin.lower() if isinstance(margin, str) else None
    if normalized_margin in ("auto", "adaptive"):
        snapshot = collect_memory_snapshot(include_os_metrics=False)
        if normalized_margin == "auto":
            safety_margin_bytes = automatic_safety_margin_bytes(snapshot.physical_memory_bytes)
        else:
            safety_margin_bytes = adaptive_safety_margin_bytes(
                snapshot.physical_memory_bytes, snapshot.recommended_working_set_bytes
            )
    else:
        safety_margin_bytes = parse_bytes(margin)
    scratch_reserve_bytes = max(parse_bytes(args.scratch_reserve), minimum_scratch_reserve_bytes)
    return MemoryBudgetConfig(
        safety_margin_bytes=safety_margin_bytes,
        kv_reserve_bytes=parse_bytes(args.kv_reserve),
        scratch_reserve_bytes=scratch_reserve_bytes,
        wired_limit_bytes=(parse_bytes(args.wired_limit) if args.wired_limit else None),
        explicit_working_set_bytes=_parse_max_unified_memory_bytes(
            getattr(args, "max_unified_memory", None)
        ),
    )


def _parse_max_unified_memory_bytes(value: str | None) -> int | None:
    """Parse --max-unified-memory: a bare number means gigabytes.

    Reuses the same ``parse_bytes`` size grammar as --kv-reserve/--scratch-reserve
    for anything that is not a bare number (for example ``14GiB``), so both
    ``--max-unified-memory 12`` (= 12GB) and ``--max-unified-memory 12GiB``
    are accepted.
    """

    if value is None:
        return None
    stripped = value.strip()
    if re.fullmatch(r"\d+(\.\d+)?", stripped):
        return parse_bytes(f"{stripped}GB")
    return parse_bytes(stripped)


def _serve_resident_budget(value: str | None, *, vision: bool) -> tuple[int | None, bool]:
    """Resolve the safe cache default before an optional VLM shell is loaded."""

    if value is None:
        # A vision tower and projector are permanent Unified Memory residents.
        # On a 16 GiB Mac, letting auto-cache consume all remaining budget makes
        # image-prefill activations the most likely source of MLX OOM errors.
        value = "off" if vision else "auto"
    return parse_resident_budget(value)


def _serve_prefill_step_size(value: int | None, *, vision: bool) -> int:
    """Keep image-prefill activation peaks bounded unless explicitly overridden."""

    return value if value is not None else (256 if vision else 2_048)


def _add_kv_cache_options(parser: argparse.ArgumentParser, *, include_max_context: bool) -> None:
    parser.add_argument(
        "--kv-cache",
        choices=("auto", "bf16", "8bit", "4bit"),
        default="auto",
        help="KV cache precision: auto, bf16, 8bit, or 4bit (default: auto)",
    )
    if include_max_context:
        parser.add_argument(
            "--kv-max-context",
            type=int,
            default=4_096,
            help="largest context to reserve KV cache for (default: 4096)",
        )


def _add_startup_options(parser: argparse.ArgumentParser, *, include_io_probe: bool) -> None:
    """M13 warmup/report flags shared by ``generate`` and ``serve``.

    ``generate`` already declares its own ``--startup-io-probe`` (with M11
    wording specific to that command), so ``include_io_probe`` lets ``serve``
    opt into the same flag here instead of duplicating its help text.
    """

    if include_io_probe:
        parser.add_argument(
            "--startup-io-probe",
            default="auto",
            help=(
                "M11 SSD read-bandwidth input for the performance advisor: auto probes "
                "~53MB from the manifest at startup, off skips probing and uses the M4 "
                "default (2.0GB/s), or supply an explicit bytes/sec number (default: auto)"
            ),
        )
    parser.add_argument(
        "--warmup",
        choices=("auto", "off"),
        default="auto",
        help=(
            "M13 startup expert-cache warmup: auto preloads every expert bundle when the "
            "resident cache can hold the full working set (full_residency mode); off never "
            "warms, regardless of mode (default: auto)"
        ),
    )
    parser.add_argument(
        "--warmup-timeout",
        type=float,
        default=300.0,
        help="maximum seconds the M13 startup warmup may run before returning partial progress "
        "(default: 300); this is a soft deadline checked once per bundle between reads, not a "
        "preemptive cutoff, so the actual run can overshoot it by up to one bundle's own "
        "read+materialize+eval time",
    )
    parser.add_argument(
        "--startup-report",
        type=Path,
        help="optional JSON destination for the M13 startup mode/budget/advisory report",
    )


def _resolve_preload(explicit: bool | None, *, registration_count: int) -> bool:
    """M13 ``--preload``/``--no-preload`` default: on for one model, off for several.

    An explicit ``--preload``/``--no-preload`` always wins; ``explicit`` is
    ``None`` only when neither flag was passed.
    """

    if explicit is not None:
        return explicit
    return registration_count == 1


def _report_startup_decision(
    logger: logging.Logger, engine: Any, startup_report_path: Path | None
) -> None:
    """Log the M11 advisory for an already-loaded engine and dump ``--startup-report``.

    Reuses the ``StartupDecision`` ``prepare_streaming_runtime`` already
    computed and stored on ``engine.startup_decision`` -- this never reruns
    the M11 advisor or the M7 hardware probe. A bare ``StreamingEngine``-like
    object without that attribute (for example an older or hand-built fake in
    a test) simply skips this reporting.
    """

    decision: StartupDecision | None = getattr(engine, "startup_decision", None)
    if decision is None:
        return
    report = decision.report
    logger.info(
        "M11 performance advisor (bandwidth source=%s): %s",
        report.get("bandwidth_source"),
        report.get("log_line"),
    )
    if report.get("wired_limit_suggestion"):
        logger.info("M11 suggestion: %s", report["wired_limit_suggestion"])
    if report.get("double_deduction_suggestion"):
        logger.info("M11 suggestion: %s", report["double_deduction_suggestion"])
    if startup_report_path is not None:
        # allow_nan=False: inf is not valid JSON. advisor.py already clamps a
        # full_residency decode_ceiling_tps to None before it reaches this
        # report, but this is the belt to that suspender -- any future
        # non-finite value fails loudly here instead of writing invalid JSON
        # to disk.
        startup_report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str, allow_nan=False),
            encoding="utf-8",
        )


def _add_predictive_prefetch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--predictor",
        type=Path,
        help="M10 predictor JSON created by train-predictor; requires --io-workers >= 1",
    )
    parser.add_argument(
        "--predictive-prefetch-candidates",
        type=int,
        default=4,
        help="maximum predicted next-layer experts per router call (default: 4)",
    )
    parser.add_argument(
        "--predictive-min-confidence",
        type=float,
        default=0.25,
        help="minimum learned next-expert probability in [0,1] (default: 0.25)",
    )
    parser.add_argument(
        "--predictive-prefetch-budget",
        default="32MB",
        help="maximum speculative expert bytes per router call (default: 32MB)",
    )


def _predictive_options(
    args: argparse.Namespace,
) -> tuple[object | None, PredictivePrefetchConfig | None]:
    if args.predictor is None:
        return None, None
    return (
        load_transition_predictor(args.predictor),
        PredictivePrefetchConfig(
            max_candidates=args.predictive_prefetch_candidates,
            min_confidence=args.predictive_min_confidence,
            max_bytes=parse_bytes(args.predictive_prefetch_budget),
        ),
    )
