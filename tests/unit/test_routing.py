from pathlib import Path

import pytest

from mlx_moe_stream.errors import TraceProtocolError
from mlx_moe_stream.routing import RouteTracer, load_trace, summarize_trace


def _record_sample_trace(tracer: RouteTracer) -> None:
    with tracer.model_call("prefill", 3):
        tracer.record_routes(
            0,
            [[[0, 1], [0, 1], [1, 2]]],
            [[[0.7, 0.3], [0.8, 0.2], [0.6, 0.4]]],
            num_experts=4,
        )
        tracer.record_routes(
            1,
            [[[2, 3], [2, 3], [3, 2]]],
            [[[0.6, 0.4], [0.6, 0.4], [0.7, 0.3]]],
            num_experts=4,
        )
    with tracer.model_call("decode", 1):
        tracer.record_routes(0, [[[1, 2]]], [[[0.9, 0.1]]], num_experts=4)


def test_model_call_assigns_the_same_absolute_indices_to_each_layer():
    tracer = RouteTracer(request_id="test")
    _record_sample_trace(tracer)

    layer_zero = [event.token_index for event in tracer.events if event.layer_id == 0]
    layer_one = [event.token_index for event in tracer.events if event.layer_id == 1]
    assert layer_zero == [0, 1, 2, 3]
    assert layer_one == [0, 1, 2]
    assert tracer.events[-1].phase == "decode"


def test_trace_jsonl_round_trip_and_summary(tmp_path: Path):
    trace_path = tmp_path / "routes.jsonl"
    with RouteTracer(request_id="test", output_path=trace_path) as tracer:
        _record_sample_trace(tracer)

    events = load_trace(trace_path)
    summary = summarize_trace(events, capacities=(0.5, 1.0))
    assert len(events) == 7
    assert summary["layers"]["0"]["num_experts"] == 4
    assert summary["layers"]["0"]["unique_experts"] == 3
    assert summary["layers"]["0"]["mean_consecutive_jaccard"] == pytest.approx(2 / 3)
    assert summary["cache_simulation"][1]["hit_rate"] > 0


def test_record_requires_an_explicit_model_call_context():
    tracer = RouteTracer()
    with pytest.raises(TraceProtocolError, match="model_call"):
        tracer.record_routes(0, [[[0, 1]]], [[[0.5, 0.5]]])


def test_invalid_expert_index_fails_fast():
    tracer = RouteTracer()
    with tracer.model_call("prefill", 1), pytest.raises(TraceProtocolError, match="outside"):
        tracer.record_routes(0, [[[4, 1]]], [[[0.5, 0.5]]], num_experts=4)


def test_load_trace_rejects_corrupt_event_shape(tmp_path: Path):
    trace_path = tmp_path / "corrupt.jsonl"
    trace_path.write_text(
        '{"request_id":"bad","phase":"prefill","token_index":0,"layer_id":0,'
        '"expert_ids":[0,1],"router_scores":[1.0],"timestamp":"now"}\n'
    )
    with pytest.raises(ValueError, match="invalid route trace"):
        load_trace(trace_path)
