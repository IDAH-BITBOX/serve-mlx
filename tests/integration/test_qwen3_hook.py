from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

from mlx_moe_stream.routing import Qwen3MoeTraceSession, RouteTracer


class _SingleMoeLayerModel:
    def __init__(self, block):
        self.layers = [SimpleNamespace(mlp=block)]

    def __call__(self, x):
        return self.layers[0].mlp(x)


def test_qwen3_hook_preserves_output_and_records_valid_routes():
    args = SimpleNamespace(
        hidden_size=4,
        moe_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    model = _SingleMoeLayerModel(Qwen3MoeSparseMoeBlock(args))
    hidden = mx.random.normal((1, 3, 4))
    baseline = model(hidden)
    mx.eval(baseline)

    tracer = RouteTracer(request_id="hook")
    with Qwen3MoeTraceSession(model, tracer), tracer.model_call("prefill", 3):
        observed = model(hidden)
        mx.eval(observed)

    assert mx.allclose(baseline, observed, atol=1e-5, rtol=1e-5).item()
    assert model.layers[0].mlp.__class__ is Qwen3MoeSparseMoeBlock
    assert [event.token_index for event in tracer.events] == [0, 1, 2]
    assert all(event.num_experts == 3 for event in tracer.events)
    assert all(0 <= expert < 3 for event in tracer.events for expert in event.expert_ids)
