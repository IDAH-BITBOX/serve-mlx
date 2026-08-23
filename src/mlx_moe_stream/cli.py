"""Command line interface for the implemented M0–M8.5 milestones."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .cache import MemoryBudgetError
from .config import parse_bytes, parse_resident_budget
from .errors import MlxMoeStreamError
from .logging import configure_logging
from .memory import MemoryBudgetConfig
from .models import load_streaming_model
from .prefetch import (
    PredictivePrefetchConfig,
    load_transition_predictor,
    train_transition_predictor,
)
from .routing import load_trace, summarize_trace, trace_qwen3_generation, write_summary
from .server import (
    LocalApiServer,
    LocalGenerationService,
    ModelRegistration,
    ModelRegistry,
    ServerConfig,
    is_loopback_host,
    run_local_server,
)
from .storage import build_streaming_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-moe-stream", description="MLX out-of-core MoE tools"
    )
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
            "enable the M4 byte-budgeted resident expert cache; use 'auto' for "
            "the M7 safe working-set budget (for example 2GB, 2048MiB, or auto)"
        ),
    )
    generate.add_argument(
        "--memory-safety-margin",
        default="2GB",
        help="M7 Unified Memory margin excluded from auto/explicit cache budgets (default: 2GB)",
    )
    generate.add_argument(
        "--kv-reserve",
        default="1GB",
        help="M7 reserve for mlx-lm KV cache growth (default: 1GB)",
    )
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
        default="auto",
        help="M4/M7 expert-cache capacity (default: auto)",
    )
    serve.add_argument("--memory-safety-margin", default="2GB")
    serve.add_argument("--kv-reserve", default="1GB")
    serve.add_argument("--scratch-reserve", default="1GB")
    serve.add_argument("--wired-limit")
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
    _add_predictive_prefetch_options(serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(args.verbose)
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
            memory_config = MemoryBudgetConfig(
                safety_margin_bytes=parse_bytes(args.memory_safety_margin),
                kv_reserve_bytes=parse_bytes(args.kv_reserve),
                scratch_reserve_bytes=parse_bytes(args.scratch_reserve),
                wired_limit_bytes=(parse_bytes(args.wired_limit) if args.wired_limit else None),
            )
            predictor, predictive_config = _predictive_options(args)
            engine = load_streaming_model(
                args.manifest,
                resident_budget_bytes=resident_budget,
                auto_resident_budget=auto_resident_budget,
                memory_config=memory_config,
                prefill_strategy=args.prefill_strategy,
                prefill_order=args.prefill_order,
                io_workers=args.io_workers,
                prefetch_depth=args.prefetch_depth,
                async_gpu=args.async_gpu,
                predictor=predictor,
                predictive_config=predictive_config,
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
                            "final_snapshot": (
                                final_snapshot.to_dict() if final_snapshot is not None else None
                            ),
                            "pressure_events": [
                                event.to_dict() for event in stats.memory_events
                            ],
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
            resident_budget, auto_resident_budget = parse_resident_budget(args.resident_budget)
            memory_config = MemoryBudgetConfig(
                safety_margin_bytes=parse_bytes(args.memory_safety_margin),
                kv_reserve_bytes=parse_bytes(args.kv_reserve),
                scratch_reserve_bytes=parse_bytes(args.scratch_reserve),
                wired_limit_bytes=(parse_bytes(args.wired_limit) if args.wired_limit else None),
            )
            predictor, predictive_config = _predictive_options(args)
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
            )
            registry = ModelRegistry(
                registrations,
                load_engine=lambda manifest_path: load_streaming_model(
                    manifest_path,
                    resident_budget_bytes=resident_budget,
                    auto_resident_budget=auto_resident_budget,
                    memory_config=memory_config,
                    prefill_strategy=args.prefill_strategy,
                    prefill_order=args.prefill_order,
                    io_workers=args.io_workers,
                    prefetch_depth=args.prefetch_depth,
                    async_gpu=args.async_gpu,
                    predictor=predictor,
                    predictive_config=predictive_config,
                ),
            )
            service = LocalGenerationService(config=server_config, registry=registry)
            try:
                server = LocalApiServer(
                    args.host,
                    args.port,
                    service,
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
                run_local_server(server)
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
