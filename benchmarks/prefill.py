#!/usr/bin/env python3
"""Measure token-major versus M5 expert-major sparse prefill exactly."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_moe_stream.config import parse_resident_budget
from mlx_moe_stream.models import load_qwen3_moe_streaming


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the prompt text to construct a longer prefill workload",
    )
    parser.add_argument(
        "--resident-budget", help="optional M4/M7 cache capacity, for example 2GB or auto"
    )
    parser.add_argument(
        "--prefill-strategy",
        choices=("expert_major", "token_major"),
        default="expert_major",
    )
    parser.add_argument(
        "--prefill-order",
        choices=("resident_first", "expert_id", "disk_offset"),
        default="resident_first",
    )
    parser.add_argument("--io-workers", type=int, default=0)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--async-gpu", action="store_true")
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be greater than zero")

    budget, auto_resident_budget = parse_resident_budget(args.resident_budget)
    engine = load_qwen3_moe_streaming(
        args.manifest,
        resident_budget_bytes=budget,
        auto_resident_budget=auto_resident_budget,
        prefill_strategy=args.prefill_strategy,
        prefill_order=args.prefill_order,
        io_workers=args.io_workers,
        prefetch_depth=args.prefetch_depth,
        async_gpu=args.async_gpu,
    )
    timeline = ()
    try:
        prompt = " ".join([args.prompt] * args.repeat)
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

    layer_stats = [
        {
            "layer": item.layer,
            "tokens": item.token_count,
            "routes": item.route_count,
            "unique_experts": item.unique_experts,
            "order": item.order,
        }
        for item in stats.prefill_layers
    ]
    payload: dict[str, Any] = {
        "prefill_strategy": args.prefill_strategy,
        "prefill_order": args.prefill_order,
        "resident_budget_bytes": budget,
        "auto_resident_budget": auto_resident_budget,
        "prompt_tokens": len(token_ids),
        "ttft_seconds": elapsed,
        "prefill_tokens_per_second": len(token_ids) / elapsed if elapsed else 0.0,
        "disk_bytes": stats.bytes_read,
        "read_count": stats.read_count,
        "average_read_size": stats.bytes_read / stats.read_count if stats.read_count else 0.0,
        "expert_resolutions": stats.expert_resolutions,
        "unique_expert_union_per_layer": layer_stats,
    }
    if stats.cache is not None:
        cache = stats.cache.to_dict()
        cache.pop("per_layer_hits")
        cache.pop("per_layer_misses")
        payload["cache"] = cache
    if stats.io_overlap is not None:
        payload["io_overlap"] = {
            "workers": stats.io_overlap.workers,
            "prefetch_depth": stats.io_overlap.prefetch_depth,
            "async_gpu": stats.io_overlap.async_gpu,
            **stats.io_overlap.loader.__dict__,
            "timeline_events": len(timeline),
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
