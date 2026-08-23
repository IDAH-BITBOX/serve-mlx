#!/usr/bin/env python3
"""Compare exact M3–M7 Qwen3 logits and greedy tokens with mlx-lm reference."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from mlx_moe_stream.config import parse_resident_budget
from mlx_moe_stream.manifest import load_manifest
from mlx_moe_stream.models import load_qwen3_moe_streaming
from mlx_moe_stream.routing import Qwen3MoeTraceSession, RouteTracer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True, help="repeat for each prompt")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--resident-budget", help="exercise the M4/M7 resident cache, for example 2GB or auto"
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

    manifest = load_manifest(args.manifest)
    reference_model, tokenizer = load(str(manifest.source_model_path))
    references = []
    for prompt_index, prompt in enumerate(args.prompt):
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        tokens = mx.array(token_ids, dtype=mx.uint32)
        tracer = RouteTracer(request_id=f"reference-{prompt_index}")
        trace_session = Qwen3MoeTraceSession(reference_model, tracer)
        with trace_session, tracer.model_call("prefill", len(token_ids)):
            logits = reference_model(tokens[None]).astype(mx.float32)
        mx.eval(logits)
        references.append(
            {
                "logits": np.array(logits),
                "routes": [(event.layer_id, event.expert_ids) for event in tracer.events],
                "greedy": int(mx.argmax(logits[:, -1, :], axis=-1).item()),
            }
        )
    # The trace session keeps a direct model reference after its context has
    # restored the wrapped blocks. Release it before M7 measures the streaming
    # shell, otherwise the reference model would incorrectly consume the whole
    # recommended working set.
    del trace_session, tracer, logits, tokens, reference_model
    gc.collect()
    mx.clear_cache()

    resident_budget, auto_resident_budget = parse_resident_budget(args.resident_budget)
    engine = load_qwen3_moe_streaming(
        args.manifest,
        resident_budget_bytes=resident_budget,
        auto_resident_budget=auto_resident_budget,
        prefill_strategy=args.prefill_strategy,
        prefill_order=args.prefill_order,
        io_workers=args.io_workers,
        prefetch_depth=args.prefetch_depth,
        async_gpu=args.async_gpu,
    )
    try:
        prompt_pairs = zip(args.prompt, references, strict=True)
        for prompt_index, (prompt, reference) in enumerate(prompt_pairs):
            token_ids = engine.tokenizer.encode(prompt, add_special_tokens=True)
            tokens = mx.array(token_ids, dtype=mx.uint32)
            route_start = len(engine.runtime.route_history())
            logits = engine.model(tokens[None]).astype(mx.float32)
            mx.eval(logits)
            actual = np.array(logits)
            routes = list(engine.runtime.route_history()[route_start:])
            router_ok = reference["routes"] == routes
            logits_ok = np.allclose(reference["logits"], actual, atol=args.atol, rtol=args.rtol)
            greedy = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            print(
                f"prompt={prompt_index} tokens={len(token_ids)} router_exact={router_ok} "
                f"logits_allclose={logits_ok} greedy_exact={reference['greedy'] == greedy} "
                f"max_abs={np.max(np.abs(reference['logits'] - actual)):.6g}"
            )
            if not router_ok or not logits_ok or reference["greedy"] != greedy:
                return 1
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
