"""Correctness-first per-expert execution for streamed MLX MoE families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..manifest import QuantizationSpec


@dataclass(frozen=True)
class MaterializedExpert:
    """One temporary MLX representation of an exact expert bundle."""

    arrays: dict[str, Any]
    nbytes: int


class ReferenceExpertBackend:
    """Execute one expert at a time using MLX's existing quantized matmul."""

    def __init__(
        self,
        quantization: QuantizationSpec,
        *,
        activation: Literal["swiglu", "geglu"] = "swiglu",
    ) -> None:
        self.quantization = quantization
        if activation not in {"swiglu", "geglu"}:
            raise ValueError(f"unsupported routed-expert activation {activation!r}")
        self.activation = activation

    def execute(self, x: Any, expert: MaterializedExpert) -> Any:
        """Evaluate the family-selected gated expert without approximation."""

        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.models.activations import swiglu
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("the reference expert backend requires mlx and mlx-lm") from error

        up = self._linear(mx, x, expert.arrays, "up")
        gate = self._linear(mx, x, expert.arrays, "gate")
        hidden = swiglu(gate, up) if self.activation == "swiglu" else nn.gelu_approx(gate) * up
        return self._linear(mx, hidden, expert.arrays, "down")

    def _linear(self, mx: Any, x: Any, arrays: dict[str, Any], prefix: str) -> Any:
        weight = arrays[f"{prefix}_weight"]
        if weight.dtype == mx.uint32:
            if self.quantization.bits is None or self.quantization.group_size is None:
                raise ValueError(
                    "quantized expert weights require bits and group_size in the manifest"
                )
            output = mx.quantized_matmul(
                x,
                weight,
                arrays[f"{prefix}_scales"],
                arrays.get(f"{prefix}_biases"),
                transpose=True,
                group_size=self.quantization.group_size,
                bits=self.quantization.bits,
                mode=self.quantization.mode or "affine",
            )
        else:
            output = mx.matmul(x, weight.swapaxes(-1, -2))
        direct_bias = arrays.get(f"{prefix}_bias")
        return output + direct_bias if direct_bias is not None else output
