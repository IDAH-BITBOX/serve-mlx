"""Model-family adapters and manifest-based streaming dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..manifest import load_manifest
from .gemma4 import Gemma4Adapter, load_gemma4_streaming
from .qwen3_5_moe import Qwen35MoeAdapter, load_qwen3_5_moe_streaming
from .qwen3_moe import Qwen3MoeAdapter, StreamingEngine, load_qwen3_moe_streaming


def load_streaming_model(manifest_path: str | Path, **kwargs: Any) -> StreamingEngine:
    """Load the adapter selected by the prepared manifest's exact layout type."""

    manifest = load_manifest(Path(manifest_path))
    adapters = {
        "qwen3_moe": Qwen3MoeAdapter,
        "qwen3_5_moe": Qwen35MoeAdapter,
        "gemma4": Gemma4Adapter,
    }
    try:
        adapter = adapters[manifest.model_type]()
    except KeyError as error:  # Manifest validation normally catches this first.
        raise ValueError(f"no streaming adapter for {manifest.model_type!r}") from error
    if kwargs.get("vision") and manifest.model_type == "qwen3_moe":
        raise ValueError("--vision supports Qwen3.5-MoE and Gemma 4 manifests, not Qwen3-MoE")
    return adapter.load_shell(manifest, **kwargs)


__all__ = [
    "Gemma4Adapter",
    "Qwen3MoeAdapter",
    "Qwen35MoeAdapter",
    "StreamingEngine",
    "load_gemma4_streaming",
    "load_qwen3_5_moe_streaming",
    "load_qwen3_moe_streaming",
    "load_streaming_model",
]
