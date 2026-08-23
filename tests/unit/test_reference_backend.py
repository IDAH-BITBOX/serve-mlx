from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_moe_stream.execution import MaterializedExpert, ReferenceExpertBackend
from mlx_moe_stream.manifest import QuantizationSpec


def test_reference_backend_matches_direct_quantized_mlx_ops():
    quantization = QuantizationSpec(bits=4, group_size=32)
    x = mx.arange(32, dtype=mx.float32) / 16 - 1
    weights = {
        "up": mx.reshape(mx.arange(2048, dtype=mx.float32) / 200, (64, 32)),
        "gate": mx.reshape(mx.arange(2048, dtype=mx.float32) / 170, (64, 32)),
        "down": mx.reshape(mx.arange(2048, dtype=mx.float32) / 110, (32, 64)),
    }
    arrays = {}
    for prefix, weight in weights.items():
        packed, scales, biases = mx.quantize(weight, group_size=32, bits=4)
        arrays[f"{prefix}_weight"] = packed
        arrays[f"{prefix}_scales"] = scales
        arrays[f"{prefix}_biases"] = biases

    backend = ReferenceExpertBackend(quantization)
    actual = backend.execute(x, MaterializedExpert(arrays=arrays, nbytes=0))
    up = mx.quantized_matmul(
        x,
        arrays["up_weight"],
        arrays["up_scales"],
        arrays["up_biases"],
        group_size=32,
        bits=4,
    )
    gate = mx.quantized_matmul(
        x,
        arrays["gate_weight"],
        arrays["gate_scales"],
        arrays["gate_biases"],
        group_size=32,
        bits=4,
    )
    expected = mx.quantized_matmul(
        mx.sigmoid(gate) * gate * up,
        arrays["down_weight"],
        arrays["down_scales"],
        arrays["down_biases"],
        group_size=32,
        bits=4,
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()


def test_reference_backend_supports_gemma_geglu_exactly():
    quantization = QuantizationSpec(bits=4, group_size=32)
    x = mx.arange(32, dtype=mx.float32) / 16 - 1
    weights = {
        "up": mx.reshape(mx.arange(2048, dtype=mx.float32) / 200, (64, 32)),
        "gate": mx.reshape(mx.arange(2048, dtype=mx.float32) / 170, (64, 32)),
        "down": mx.reshape(mx.arange(2048, dtype=mx.float32) / 110, (32, 64)),
    }
    arrays = {}
    for prefix, weight in weights.items():
        packed, scales, biases = mx.quantize(weight, group_size=32, bits=4)
        arrays[f"{prefix}_weight"] = packed
        arrays[f"{prefix}_scales"] = scales
        arrays[f"{prefix}_biases"] = biases

    actual = ReferenceExpertBackend(quantization, activation="geglu").execute(
        x, MaterializedExpert(arrays=arrays, nbytes=0)
    )
    up = mx.quantized_matmul(
        x, arrays["up_weight"], arrays["up_scales"], arrays["up_biases"], group_size=32, bits=4
    )
    gate = mx.quantized_matmul(
        x,
        arrays["gate_weight"],
        arrays["gate_scales"],
        arrays["gate_biases"],
        group_size=32,
        bits=4,
    )
    expected = mx.quantized_matmul(
        nn.gelu_approx(gate) * up,
        arrays["down_weight"],
        arrays["down_scales"],
        arrays["down_biases"],
        group_size=32,
        bits=4,
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
