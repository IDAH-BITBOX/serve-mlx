"""Explicit errors used instead of hidden runtime fallbacks."""


class MlxMoeStreamError(Exception):
    """Base class for package errors."""


class OptionalRuntimeDependencyError(MlxMoeStreamError):
    """MLX or mlx-lm is required for the requested operation."""


class UnsupportedModelError(MlxMoeStreamError):
    """The supplied model does not expose the supported Qwen3-MoE contract."""


class TraceProtocolError(MlxMoeStreamError):
    """A model call was not registered with the route tracer correctly."""


class MemoryPressureError(MlxMoeStreamError):
    """The runtime cannot continue within its explicit safe memory budget."""
