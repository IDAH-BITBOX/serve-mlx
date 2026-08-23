#!/usr/bin/env python3
"""Compare M6 sequential, threaded-I/O, and async-GPU expert-major prefill."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_moe_stream.config import parse_resident_budget
from mlx_moe_stream.models import load_qwen3_moe_streaming


def _run_case(
    manifest: Path,
    prompt: str,
    resident_budget: int | None,
    auto_resident_budget: bool,
    workers: int,
    prefetch_depth: int,
    async_gpu: bool,
) -> dict[str, Any]:
    engine = load_qwen3_moe_streaming(
        manifest,
        resident_budget_bytes=resident_budget,
        auto_resident_budget=auto_resident_budget,
        prefill_strategy="expert_major",
        prefill_order="resident_first",
        io_workers=workers,
        prefetch_depth=prefetch_depth,
        async_gpu=async_gpu,
    )
    timeline = ()
    try:
        token_ids = engine.tokenizer.encode(prompt, add_special_tokens=True)
        tokens = mx.array(token_ids, dtype=mx.uint32)
        started = time.perf_counter()
        logits = engine.model(tokens[None])
        mx.eval(logits)
        elapsed = time.perf_counter() - started
        stats = engine.runtime.stats()
        timeline = engine.runtime.timeline()
    finally:
        engine.close()
        del engine
        gc.collect()
        mx.clear_cache()

    events = Counter(event.name for event in timeline)
    payload: dict[str, Any] = {
        "prompt_tokens": len(token_ids),
        "elapsed_seconds": elapsed,
        "prefill_tokens_per_second": len(token_ids) / elapsed if elapsed else 0.0,
        "disk_bytes": stats.bytes_read,
        "read_count": stats.read_count,
        "timeline_events": dict(events),
    }
    if stats.io_overlap is not None:
        payload["io_overlap"] = {
            "workers": stats.io_overlap.workers,
            "prefetch_depth": stats.io_overlap.prefetch_depth,
            "async_gpu": stats.io_overlap.async_gpu,
            **asdict(stats.io_overlap.loader),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--resident-budget", help="optional M4/M7 cache capacity, for example 2GB or auto"
    )
    parser.add_argument("--io-workers", type=int, default=1)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be greater than zero")
    if args.io_workers <= 0:
        parser.error("--io-workers must be greater than zero")
    if args.prefetch_depth <= 0:
        parser.error("--prefetch-depth must be greater than zero")

    prompt = " ".join([args.prompt] * args.repeat)
    budget, auto_resident_budget = parse_resident_budget(args.resident_budget)
    cases = {
        "sequential": (0, 0, False),
        "threaded_io": (args.io_workers, args.prefetch_depth, False),
        "async_gpu": (args.io_workers, args.prefetch_depth, True),
    }
    result = {
        name: _run_case(
            args.manifest,
            prompt,
            budget,
            auto_resident_budget,
            workers,
            depth,
            async_gpu,
        )
        for name, (workers, depth, async_gpu) in cases.items()
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
