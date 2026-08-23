#!/usr/bin/env python3
"""Run the M1 Qwen3-MoE routing-trace benchmark."""

from mlx_moe_stream.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["trace", *__import__("sys").argv[1:]]))

