from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_moe_stream.models.qwen3_moe import StreamingSwitchGLU


class _FakeRuntime:
    def __init__(self) -> None:
        self.prefetch_depth = 0
        self.calls: list[tuple[int, tuple[int, ...]]] = []
        self.routes: list[tuple[int, list[list[int]]]] = []
        self.prefill: list[dict[str, Any]] = []
        self.orders: list[str] = []
        self.aborted = False

    def execute(self, layer: int, expert: int, x: Any) -> Any:
        self.calls.append((expert, tuple(x.shape)))
        return x + expert * 10

    def execute_group(self, layer: int, expert: int, x: Any) -> Any:
        self.calls.append((expert, tuple(x.shape)))
        return x + expert * 10

    def prefetch(self, layer: int, expert: int) -> bool:
        return False

    def record_routes(self, layer: int, expert_rows: list[list[int]]) -> None:
        self.routes.append((layer, expert_rows))

    def order_experts(self, layer: int, experts: list[int], order: str) -> list[int]:
        assert layer == 7
        self.orders.append(order)
        return sorted(experts, reverse=True)

    def record_prefill_layer(self, layer: int, **kwargs: Any) -> None:
        self.prefill.append({"layer": layer, **kwargs})

    def synchronize_batch(self, output: Any) -> Any:
        mx.eval(output)
        return output

    def abort_batch(self) -> None:
        self.aborted = True


@pytest.mark.parametrize("order", ["resident_first", "expert_id", "disk_offset"])
def test_expert_major_prefill_groups_routes_and_scatters_to_original_topk_slots(order: str):
    runtime = _FakeRuntime()
    block = StreamingSwitchGLU(runtime, 7, prefill_strategy="expert_major", prefill_order=order)
    x = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    indices = mx.array([[[2, 1], [1, 2], [2, 2]]], dtype=mx.uint32)

    actual = block(x, indices)
    mx.eval(actual)

    expected = np.array(
        [
            [
                [[21.0, 22.0], [11.0, 12.0]],
                [[13.0, 14.0], [23.0, 24.0]],
                [[25.0, 26.0], [25.0, 26.0]],
            ]
        ],
        dtype=np.float32,
    )
    assert np.array_equal(np.array(actual), expected)
    assert runtime.calls == [(2, (4, 2)), (1, (2, 2))]
    assert runtime.routes == [(7, [[2, 1], [1, 2], [2, 2]])]
    assert runtime.prefill == [
        {
            "layer": 7,
            "token_count": 3,
            "route_count": 6,
            "unique_experts": 2,
            "order": order,
        }
    ]
    assert runtime.orders == [order]
    assert not runtime.aborted


def test_token_major_decode_keeps_one_execute_per_route():
    runtime = _FakeRuntime()
    block = StreamingSwitchGLU(
        runtime, 7, prefill_strategy="token_major", prefill_order="resident_first"
    )
    x = mx.ones((1, 2, 2))
    indices = mx.array([[[1, 2], [2, 1]]], dtype=mx.uint32)

    actual = block(x, indices)
    mx.eval(actual)

    assert tuple(actual.shape) == (1, 2, 2, 2)
    assert runtime.calls == [(1, (2,)), (2, (2,)), (2, (2,)), (1, (2,))]
    assert runtime.prefill == []
