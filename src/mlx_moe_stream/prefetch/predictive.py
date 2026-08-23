"""Trace-trained, bounded next-layer expert prefetching for M10."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..cache import ExpertKey
from ..manifest import ModelManifest
from ..routing import RouteEvent


@dataclass(frozen=True)
class PredictionCandidate:
    """One predicted expert in the layer after an observed route."""

    expert: int
    confidence: float


@dataclass(frozen=True)
class PredictivePrefetchConfig:
    """Explicit limits for speculative M10 I/O; all values are per layer call."""

    max_candidates: int = 4
    min_confidence: float = 0.25
    max_bytes: int = 32_000_000

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("M10 maximum predicted experts must be greater than zero")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("M10 minimum prediction confidence must be in [0, 1]")
        if self.max_bytes <= 0:
            raise ValueError("M10 predictive prefetch byte budget must be greater than zero")


@dataclass(frozen=True)
class PredictivePrefetchStats:
    """Observable M10 prediction decisions, excluding known-route M6 reads."""

    prediction_calls: int
    candidates_considered: int
    submitted: int
    skipped_confidence: int
    skipped_candidate_limit: int
    skipped_byte_budget: int
    skipped_runtime: int


@dataclass(frozen=True)
class TransitionPredictor:
    """A deterministic conditional distribution over next-layer experts.

    The predictor is trained from already observed route traces. It never
    changes a router output or substitutes an expert result: it only proposes
    an exact byte read after the current layer's router has completed.
    """

    format_version: int
    model_type: str
    num_layers: int
    num_experts: int
    transitions: dict[int, dict[int, tuple[PredictionCandidate, ...]]]

    def validate(self) -> None:
        if self.format_version != 1:
            raise ValueError(f"unsupported M10 predictor format {self.format_version}")
        if not self.model_type:
            raise ValueError("M10 predictor requires a model_type")
        if self.num_layers <= 1 or self.num_experts <= 0:
            raise ValueError("M10 predictor requires at least two layers and one expert")
        for layer, sources in self.transitions.items():
            if layer < 0 or layer >= self.num_layers - 1:
                raise ValueError(f"M10 predictor has invalid source layer {layer}")
            for source, candidates in sources.items():
                if source < 0 or source >= self.num_experts or not candidates:
                    raise ValueError(f"M10 predictor has invalid source expert {source}")
                seen: set[int] = set()
                probability_sum = 0.0
                for candidate in candidates:
                    if candidate.expert in seen or not 0 <= candidate.expert < self.num_experts:
                        raise ValueError("M10 predictor has duplicate or invalid target expert")
                    if not 0.0 < candidate.confidence <= 1.0:
                        raise ValueError("M10 predictor has invalid target probability")
                    seen.add(candidate.expert)
                    probability_sum += candidate.confidence
                if probability_sum > 1.000001:
                    raise ValueError("M10 predictor target probabilities exceed one")

    def validate_manifest(self, manifest: ModelManifest) -> None:
        self.validate()
        if self.model_type != manifest.model_type:
            raise ValueError(
                "M10 predictor model type does not match manifest: "
                f"{self.model_type!r} != {manifest.model_type!r}"
            )
        if self.num_layers != manifest.num_layers or self.num_experts != manifest.num_experts:
            raise ValueError("M10 predictor layer/expert dimensions do not match manifest")

    def predict(
        self, layer: int, expert_rows: Iterable[Iterable[int]]
    ) -> tuple[PredictionCandidate, ...]:
        """Rank candidate experts for ``layer + 1`` from the observed rows."""

        if layer < 0 or layer >= self.num_layers - 1:
            return ()
        observed = [int(expert) for row in expert_rows for expert in row]
        if not observed:
            return ()
        source_transitions = self.transitions.get(layer, {})
        scores: dict[int, float] = defaultdict(float)
        for source in observed:
            for candidate in source_transitions.get(source, ()):
                scores[candidate.expert] += candidate.confidence
        denominator = len(observed)
        return tuple(
            PredictionCandidate(expert=expert, confidence=score / denominator)
            for expert, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": self.format_version,
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "transitions": {
                str(layer): {
                    str(source): [asdict(candidate) for candidate in candidates]
                    for source, candidates in sorted(sources.items())
                }
                for layer, sources in sorted(self.transitions.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TransitionPredictor:
        try:
            raw_transitions = value["transitions"]
            if not isinstance(raw_transitions, dict):
                raise ValueError("transitions must be an object")
            transitions = {
                int(layer): {
                    int(source): tuple(
                        PredictionCandidate(
                            expert=int(candidate["expert"]),
                            confidence=float(candidate["confidence"]),
                        )
                        for candidate in candidates
                    )
                    for source, candidates in sources.items()
                }
                for layer, sources in raw_transitions.items()
            }
            predictor = cls(
                format_version=int(value["format_version"]),
                model_type=str(value["model_type"]),
                num_layers=int(value["num_layers"]),
                num_experts=int(value["num_experts"]),
                transitions=transitions,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid M10 predictor") from error
        predictor.validate()
        return predictor

    def write(self, path: Path, *, overwrite: bool = False) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite M10 predictor: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class PredictivePrefetchScheduler:
    """Apply predictor candidates to exact runtime prefetch with hard limits."""

    def __init__(
        self,
        manifest: ModelManifest,
        predictor: TransitionPredictor,
        config: PredictivePrefetchConfig,
    ) -> None:
        predictor.validate_manifest(manifest)
        self._manifest = manifest
        self._predictor = predictor
        self._config = config
        self._prediction_calls = 0
        self._candidates_considered = 0
        self._submitted = 0
        self._skipped_confidence = 0
        self._skipped_candidate_limit = 0
        self._skipped_byte_budget = 0
        self._skipped_runtime = 0

    def schedule(
        self,
        layer: int,
        expert_rows: list[list[int]],
        submit: Callable[[int, int], bool],
    ) -> None:
        candidates = self._predictor.predict(layer, expert_rows)
        if not candidates:
            return
        self._prediction_calls += 1
        self._candidates_considered += len(candidates)
        selected = 0
        selected_bytes = 0
        target_layer = layer + 1
        for candidate in candidates:
            if candidate.confidence < self._config.min_confidence:
                self._skipped_confidence += 1
                continue
            if selected >= self._config.max_candidates:
                self._skipped_candidate_limit += 1
                continue
            bundle = self._manifest.expert_bundles[ExpertKey(target_layer, candidate.expert)]
            if selected_bytes + bundle.total_bytes > self._config.max_bytes:
                self._skipped_byte_budget += 1
                continue
            selected += 1
            selected_bytes += bundle.total_bytes
            if submit(target_layer, candidate.expert):
                self._submitted += 1
            else:
                self._skipped_runtime += 1

    def stats(self) -> PredictivePrefetchStats:
        return PredictivePrefetchStats(
            prediction_calls=self._prediction_calls,
            candidates_considered=self._candidates_considered,
            submitted=self._submitted,
            skipped_confidence=self._skipped_confidence,
            skipped_candidate_limit=self._skipped_candidate_limit,
            skipped_byte_budget=self._skipped_byte_budget,
            skipped_runtime=self._skipped_runtime,
        )


def train_transition_predictor(
    events: Iterable[RouteEvent], *, model_type: str
) -> TransitionPredictor:
    """Train next-layer conditional expert distributions from a route trace."""

    events = tuple(events)
    if not events:
        raise ValueError("M10 predictor training requires at least one route event")
    declared_expert_counts = {
        event.num_experts for event in events if event.num_experts is not None
    }
    if len(declared_expert_counts) != 1:
        raise ValueError("M10 route events require one declared num_experts value")
    num_experts = declared_expert_counts.pop()
    assert num_experts is not None
    grouped: dict[tuple[str, str, int], dict[int, RouteEvent]] = defaultdict(dict)
    for event in events:
        key = (event.request_id, event.phase, event.token_index)
        previous = grouped[key].setdefault(event.layer_id, event)
        if previous is not event and previous != event:
            raise ValueError("M10 route trace has conflicting events for one token/layer")
    num_layers = max(event.layer_id for event in events) + 1
    if num_layers <= 1:
        raise ValueError("M10 predictor training requires routes from at least two layers")
    counts: dict[int, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    for by_layer in grouped.values():
        for layer, source_event in by_layer.items():
            target_event = by_layer.get(layer + 1)
            if target_event is None:
                continue
            for source in source_event.expert_ids:
                for target in target_event.expert_ids:
                    counts[layer][source][target] += 1
    transitions: dict[int, dict[int, tuple[PredictionCandidate, ...]]] = {}
    for layer, source_counts in counts.items():
        transitions[layer] = {}
        for source, target_counts in source_counts.items():
            total = sum(target_counts.values())
            transitions[layer][source] = tuple(
                PredictionCandidate(expert=target, confidence=count / total)
                for target, count in sorted(
                    target_counts.items(), key=lambda item: (-item[1], item[0])
                )
            )
    predictor = TransitionPredictor(
        format_version=1,
        model_type=model_type,
        num_layers=num_layers,
        num_experts=num_experts,
        transitions=transitions,
    )
    predictor.validate()
    return predictor


def load_transition_predictor(path: Path) -> TransitionPredictor:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read M10 predictor {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"M10 predictor must be a JSON object: {path}")
    return TransitionPredictor.from_dict(value)
