#!/usr/bin/env python3
"""Measure M2 exact expert-bundle reads from a prepared manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.manifest import load_manifest
from mlx_moe_stream.storage import SafetensorsExpertStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")

    manifest = load_manifest(args.manifest)
    key = ExpertKey(args.layer, args.expert)
    try:
        bundle = manifest.expert_bundles[key]
    except KeyError:
        parser.error(f"manifest has no expert bundle {key}")
    started = time.perf_counter()
    with SafetensorsExpertStore() as store:
        for _ in range(args.iterations):
            store.read_bundle(bundle)
        metrics = store.metrics()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "bundle": {"layer": key.layer, "expert": key.expert},
                "bundle_bytes": bundle.total_bytes,
                "bytes_read": metrics.bytes_read,
                "read_count": metrics.read_count,
                "elapsed_seconds": elapsed,
                "effective_mib_per_second": metrics.bytes_read / elapsed / (1 << 20),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
