#!/usr/bin/env python3
"""Measure M3 no-cache or M4 resident-cache decode under explicit budgets."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import stream_generate

from mlx_moe_stream.config import parse_resident_budget
from mlx_moe_stream.manifest import load_manifest
from mlx_moe_stream.models import load_qwen3_moe_streaming


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def _run_once(
    manifest_path: Path,
    prompt: str,
    max_tokens: int,
    budget: int | None,
    auto_resident_budget: bool,
) -> dict[str, Any]:
    engine = load_qwen3_moe_streaming(
        manifest_path,
        resident_budget_bytes=budget,
        auto_resident_budget=auto_resident_budget,
    )
    try:
        responses = []
        step_seconds = []
        generator = stream_generate(engine.model, engine.tokenizer, prompt, max_tokens=max_tokens)
        while True:
            started = time.perf_counter()
            try:
                response = next(generator)
            except StopIteration:
                break
            step_seconds.append(time.perf_counter() - started)
            responses.append(response)
        stats = engine.runtime.stats()
    finally:
        engine.close()
        del engine
        gc.collect()
        mx.clear_cache()

    generated_tokens = responses[-1].generation_tokens if responses else 0
    decode_steps = step_seconds[1:] if len(step_seconds) > 1 else step_seconds
    payload: dict[str, Any] = {
        "mode": "resident-cache" if stats.cache is not None else "no-cache",
        "resident_budget_bytes": budget,
        "auto_resident_budget": auto_resident_budget,
        "generated_text": "".join(response.text for response in responses),
        "generated_tokens": generated_tokens,
        "prompt_tokens": responses[-1].prompt_tokens if responses else 0,
        "prompt_seconds": step_seconds[0] if step_seconds else 0.0,
        "decode_tokens_per_second": (
            generated_tokens / sum(decode_steps) if decode_steps and generated_tokens else 0.0
        ),
        "decode_p50_seconds": _percentile(decode_steps, 0.50),
        "decode_p95_seconds": _percentile(decode_steps, 0.95),
        "disk_bytes": stats.bytes_read,
        "disk_bytes_per_token": stats.bytes_read / generated_tokens if generated_tokens else 0.0,
        "read_count": stats.read_count,
        "expert_resolutions": stats.expert_resolutions,
    }
    if stats.cache is not None:
        cache = stats.cache
        cache_payload = cache.to_dict()
        cache_payload.pop("per_layer_hits")
        cache_payload.pop("per_layer_misses")
        payload["cache"] = {
            **cache_payload,
            "evictions_per_token": (
                cache.eviction_count / generated_tokens if generated_tokens else 0.0
            ),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resident-budget", help="cache capacity, for example 2GB, 2048MiB, or auto"
    )
    group.add_argument(
        "--budget-fraction",
        action="append",
        type=float,
        help="repeat to run forced oversubscription at a fraction of all expert bytes",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.budget_fraction and any(not 0 < value <= 1 for value in args.budget_fraction):
        parser.error("--budget-fraction must be in (0, 1]")

    if args.budget_fraction:
        manifest = load_manifest(args.manifest)
        total_expert_bytes = sum(
            bundle.total_bytes for bundle in manifest.expert_bundles.values()
        )
        budgets: list[tuple[int | None, bool]] = [
            (int(total_expert_bytes * value), False) for value in args.budget_fraction
        ]
    elif args.resident_budget:
        budgets = [parse_resident_budget(args.resident_budget)]
    else:
        budgets = [(None, False)]

    for budget, auto_resident_budget in budgets:
        result = _run_once(
            args.manifest,
            args.prompt,
            args.max_tokens,
            budget,
            auto_resident_budget,
        )
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
