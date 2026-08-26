#!/usr/bin/env python3
"""Rewrite a streaming manifest into a bundle-major (contiguous) expert layout.

Source checkpoints store MoE experts role-major: every expert's ``gate_weight``
slice lives inside one big ``gate_proj.weight`` tensor, its ``up_scales`` inside
another, and so on. Reading a single expert therefore costs one pread per span,
at offsets hundreds of megabytes apart, so the device sees small random reads
instead of one large sequential one.

This tool copies every expert's spans back to back into a single blob and emits a
manifest pointing at the new offsets. The bytes are unchanged, so generation is
bit-identical; only the physical layout differs. ``SafetensorsExpertStore``
coalesces adjacent spans, so a repacked expert costs exactly one read.

The source checkpoint is never modified. To roll back, point the server at the
original manifest again.

Example:
    python tools/repack_bundle_major.py \
        --manifest prepared-model/manifest.json \
        --out-blob prepared-model/experts-bundlemajor.bin \
        --out-manifest prepared-model/manifest-bundlemajor.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

_GIB = 1024**3


def _bundle_sort_key(key: str) -> tuple[int, int]:
    layer, expert = key.split(":")
    return int(layer), int(expert)


class _FdCache:
    """Reusable read-only descriptors, one per source shard."""

    def __init__(self) -> None:
        self._fds: dict[str, int] = {}

    def get(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def _read_bundle(fds: _FdCache, bundle: dict) -> list[bytes]:
    chunks = []
    for span in bundle["tensors"]:
        data = os.pread(fds.get(span["file"]), span["nbytes"], span["offset"])
        if len(data) != span["nbytes"]:
            raise SystemExit(f"short read: {span['tensor_name']} @{span['offset']}")
        chunks.append(data)
    return chunks


def _verify(fds: _FdCache, src: dict, dst: dict, keys: list[str], blob: str, samples: int) -> None:
    print(f"verifying {samples} random bundles against the source...", flush=True)
    rnd = random.Random(99)
    fd = os.open(blob, os.O_RDONLY)
    bad = 0
    try:
        for key in rnd.sample(keys, min(samples, len(keys))):
            for old, new in zip(src[key]["tensors"], dst[key]["tensors"]):
                a = os.pread(fds.get(old["file"]), old["nbytes"], old["offset"])
                b = os.pread(fd, new["nbytes"], new["offset"])
                if hashlib.sha256(a).digest() != hashlib.sha256(b).digest():
                    print(f"  MISMATCH {key} {old['role']}")
                    bad += 1
    finally:
        os.close(fd)
    if bad:
        raise SystemExit(f"VERIFICATION FAILED: {bad} mismatched spans")
    print("verification OK: all sampled spans are byte-identical", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="source manifest to repack")
    parser.add_argument("--out-blob", required=True, help="bundle-major blob to write")
    parser.add_argument("--out-manifest", required=True, help="manifest describing the new blob")
    parser.add_argument("--lookahead", type=int, default=16, help="bundles read ahead of the writer")
    parser.add_argument("--verify-samples", type=int, default=200, help="bundles to re-read and hash")
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    bundles = manifest["expert_bundles"]
    # Layer-major, then expert id, so neighbouring expert ids also land adjacent.
    keys = sorted(bundles, key=_bundle_sort_key)
    total = sum(bundles[k]["total_bytes"] for k in keys)
    print(f"bundles={len(keys)} total={total / _GIB:.2f} GiB -> {args.out_blob}", flush=True)

    stat = os.statvfs(os.path.dirname(os.path.abspath(args.out_blob)) or ".")
    avail = stat.f_bavail * stat.f_frsize
    if avail < total * 1.02:
        raise SystemExit(
            f"not enough space: need {total / _GIB:.1f} GiB, have {avail / _GIB:.1f} GiB"
        )

    repacked = json.loads(json.dumps(manifest))
    fds = _FdCache()
    pos = 0
    started = time.time()

    try:
        with ThreadPoolExecutor(max_workers=8) as pool, open(args.out_blob, "wb", buffering=1 << 20) as out:
            pending: deque = deque()
            remaining = iter(keys)

            def submit_next() -> None:
                try:
                    key = next(remaining)
                except StopIteration:
                    return
                pending.append((key, pool.submit(_read_bundle, fds, bundles[key])))

            for _ in range(args.lookahead):
                submit_next()

            done = 0
            while pending:
                key, future = pending.popleft()
                submit_next()
                for span, data in zip(repacked["expert_bundles"][key]["tensors"], future.result()):
                    span["file"] = os.path.abspath(args.out_blob)
                    span["offset"] = pos
                    pos += len(data)
                    out.write(data)
                done += 1
                if done % 512 == 0:
                    elapsed = time.time() - started
                    print(
                        f"  {done}/{len(keys)}  {pos / _GIB:.1f} GiB  {pos / elapsed / 1e6:.0f} MB/s"
                        f"  eta {elapsed * (len(keys) / done - 1) / 60:.1f} min",
                        flush=True,
                    )

        print(f"written {pos / _GIB:.2f} GiB in {(time.time() - started) / 60:.1f} min", flush=True)
        with open(args.out_manifest, "w") as handle:
            json.dump(repacked, handle)
        print(f"manifest -> {args.out_manifest}", flush=True)

        _verify(fds, bundles, repacked["expert_bundles"], keys, args.out_blob, args.verify_samples)

        gaps = sum(
            1
            for key in keys
            for a, b in zip(
                repacked["expert_bundles"][key]["tensors"],
                repacked["expert_bundles"][key]["tensors"][1:],
            )
            if b["offset"] != a["offset"] + a["nbytes"]
        )
        print(f"contiguity check: {gaps} non-adjacent pairs across all bundles (expect 0)", flush=True)
    finally:
        fds.close()


if __name__ == "__main__":
    main()
