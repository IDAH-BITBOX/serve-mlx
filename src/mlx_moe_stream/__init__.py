"""Exact out-of-core Mixture-of-Experts tooling for MLX."""

from .config import RuntimeConfig, parse_bytes, parse_resident_budget
from .manifest import ExpertBundleSpec, ModelManifest, TensorSpan, load_manifest
from .memory import MemoryBudgetConfig, MemoryBudgetManager, MemorySnapshot
from .routing import RouteEvent, RouteTracer, summarize_trace

__all__ = [
    "ExpertBundleSpec",
    "MemoryBudgetConfig",
    "MemoryBudgetManager",
    "MemorySnapshot",
    "ModelManifest",
    "RouteEvent",
    "RouteTracer",
    "RuntimeConfig",
    "TensorSpan",
    "load_manifest",
    "parse_bytes",
    "parse_resident_budget",
    "summarize_trace",
]

__version__ = "0.1.0a7"
