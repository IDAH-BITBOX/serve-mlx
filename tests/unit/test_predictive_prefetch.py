from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.prefetch import (
    PredictionCandidate,
    PredictivePrefetchConfig,
    PredictivePrefetchScheduler,
    TransitionPredictor,
    load_transition_predictor,
    train_transition_predictor,
)
from mlx_moe_stream.routing import RouteEvent


def _event(token: int, layer: int, experts: tuple[int, ...]) -> RouteEvent:
    return RouteEvent(
        request_id="request",
        phase="decode",
        token_index=token,
        layer_id=layer,
        expert_ids=experts,
        router_scores=tuple(1.0 / len(experts) for _ in experts),
        timestamp="2026-01-01T00:00:00+00:00",
        num_experts=5,
        top_k=len(experts),
    )


def _manifest(*, num_experts: int = 5) -> SimpleNamespace:
    bundles = {
        ExpertKey(layer, expert): SimpleNamespace(total_bytes=10)
        for layer in range(2)
        for expert in range(num_experts)
    }
    return SimpleNamespace(
        model_type="qwen3_moe",
        num_layers=2,
        num_experts=num_experts,
        expert_bundles=bundles,
    )


def test_training_ranks_conditional_next_layer_experts_and_round_trips(tmp_path: Path):
    predictor = train_transition_predictor(
        [
            _event(0, 0, (1,)),
            _event(0, 1, (3,)),
            _event(1, 0, (1,)),
            _event(1, 1, (4,)),
            _event(2, 0, (1,)),
            _event(2, 1, (3,)),
        ],
        model_type="qwen3_moe",
    )

    assert predictor.predict(0, [[1]]) == (
        PredictionCandidate(expert=3, confidence=pytest.approx(2 / 3)),
        PredictionCandidate(expert=4, confidence=pytest.approx(1 / 3)),
    )
    output = tmp_path / "predictor.json"
    predictor.write(output)
    assert load_transition_predictor(output) == predictor


def test_scheduler_enforces_confidence_candidate_and_byte_limits():
    predictor = TransitionPredictor(
        format_version=1,
        model_type="qwen3_moe",
        num_layers=2,
        num_experts=5,
        transitions={
            0: {
                1: (
                    PredictionCandidate(expert=3, confidence=0.8),
                    PredictionCandidate(expert=4, confidence=0.2),
                )
            }
        },
    )
    submitted: list[tuple[int, int]] = []
    scheduler = PredictivePrefetchScheduler(
        _manifest(),
        predictor,
        PredictivePrefetchConfig(max_candidates=1, min_confidence=0.25, max_bytes=10),
    )

    scheduler.schedule(0, [[1]], lambda layer, expert: submitted.append((layer, expert)) is None)

    assert submitted == [(1, 3)]
    stats = scheduler.stats()
    assert stats.prediction_calls == 1
    assert stats.submitted == 1
    assert stats.skipped_confidence == 1


def test_scheduler_rejects_a_predictor_for_the_wrong_manifest_shape():
    predictor = TransitionPredictor(
        format_version=1,
        model_type="qwen3_moe",
        num_layers=2,
        num_experts=5,
        transitions={},
    )
    with pytest.raises(ValueError, match="dimensions"):
        PredictivePrefetchScheduler(
            _manifest(num_experts=6), predictor, PredictivePrefetchConfig()
        )
