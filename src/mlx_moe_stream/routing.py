"""Qwen3-MoE routing observation and offline locality analysis.

This module intentionally has no import-time MLX dependency.  The trace data
and cache analysis can run in ordinary Python environments; only attaching a
hook or executing a trace requires MLX and mlx-lm.
"""

from __future__ import annotations

import inspect
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import log2
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .cache.policy import ExpertKey, simulate_lru_curve
from .errors import OptionalRuntimeDependencyError, TraceProtocolError, UnsupportedModelError

TracePhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class RouteEvent:
    """One router top-k result for a token at one MoE layer."""

    request_id: str
    phase: TracePhase
    token_index: int
    layer_id: int
    expert_ids: tuple[int, ...]
    router_scores: tuple[float, ...]
    timestamp: str
    num_experts: int | None = None
    top_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expert_ids"] = list(self.expert_ids)
        data["router_scores"] = list(self.router_scores)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouteEvent:
        try:
            event = cls(
                request_id=str(data["request_id"]),
                phase=_validate_phase(data["phase"]),
                token_index=int(data["token_index"]),
                layer_id=int(data["layer_id"]),
                expert_ids=tuple(int(v) for v in data["expert_ids"]),
                router_scores=tuple(float(v) for v in data["router_scores"]),
                timestamp=str(data["timestamp"]),
                num_experts=_optional_int(data.get("num_experts")),
                top_k=_optional_int(data.get("top_k")),
            )
            if not event.expert_ids or len(event.expert_ids) != len(event.router_scores):
                raise ValueError("expert_ids and router_scores must be non-empty and equally sized")
            if event.num_experts is not None and any(
                expert < 0 or expert >= event.num_experts for expert in event.expert_ids
            ):
                raise ValueError("expert index is outside the declared expert range")
            return event
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid route event: {data!r}") from error


@dataclass(frozen=True)
class _ModelCall:
    phase: TracePhase
    token_indices: tuple[int, ...]


class RouteTracer:
    """Collect route events without changing router decisions.

    Callers that invoke a model directly must bracket every forward call with
    :meth:`model_call`; this gives all MoE layers the same absolute token
    indices. :func:`trace_qwen3_generation` does this automatically.
    """

    def __init__(self, request_id: str | None = None, output_path: Path | None = None) -> None:
        self.request_id = request_id or str(uuid4())
        self.output_path = output_path
        self.events: list[RouteEvent] = []
        self._next_token_index = 0
        self._active_call: _ModelCall | None = None
        self._output_file = None

    def __enter__(self) -> RouteTracer:
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_file = self.output_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None

    @contextmanager
    def model_call(self, phase: TracePhase, token_count: int) -> Iterator[None]:
        """Register the phase and absolute token IDs for one batch-size-one forward call."""

        if self._active_call is not None:
            raise TraceProtocolError("route tracer model_call contexts cannot be nested")
        if token_count <= 0:
            raise TraceProtocolError("model_call token_count must be greater than zero")
        phase = _validate_phase(phase)
        indices = tuple(range(self._next_token_index, self._next_token_index + token_count))
        self._next_token_index += token_count
        self._active_call = _ModelCall(phase=phase, token_indices=indices)
        try:
            yield
        finally:
            self._active_call = None

    def record_routes(
        self,
        layer_id: int,
        expert_ids: Any,
        router_scores: Any,
        *,
        num_experts: int | None = None,
        top_k: int | None = None,
    ) -> None:
        """Record a router result matrix shaped ``[batch, tokens, top_k]``.

        M1 deliberately supports batch size one. A mismatch is an explicit error
        rather than silently assigning incorrect token indices to a batch.
        """

        expert_rows = _rows_from_array_like(expert_ids)
        score_rows = _rows_from_array_like(router_scores)
        if len(expert_rows) != len(score_rows):
            raise TraceProtocolError("expert ID rows and score rows have different lengths")
        if any(
            len(ids) != len(scores) for ids, scores in zip(expert_rows, score_rows, strict=True)
        ):
            raise TraceProtocolError("expert IDs and scores have different top-k widths")
        if not expert_rows:
            return
        if self._active_call is None:
            raise TraceProtocolError(
                "route capture requires 'with tracer.model_call(phase, token_count)' around model()"
            )
        active = self._active_call
        if len(expert_rows) != len(active.token_indices):
            raise TraceProtocolError(
                "M1 route tracing supports batch size one; router rows do not match model_call "
                f"token_count ({len(expert_rows)} != {len(active.token_indices)})"
            )
        actual_top_k = top_k if top_k is not None else len(expert_rows[0])
        for token_index, ids, scores in zip(
            active.token_indices, expert_rows, score_rows, strict=True
        ):
            ids_tuple = tuple(int(value) for value in ids)
            if num_experts is not None and any(
                value < 0 or value >= num_experts for value in ids_tuple
            ):
                raise TraceProtocolError(
                    f"layer {layer_id} emitted an expert index outside [0, {num_experts})"
                )
            event = RouteEvent(
                request_id=self.request_id,
                phase=active.phase,
                token_index=token_index,
                layer_id=int(layer_id),
                expert_ids=ids_tuple,
                router_scores=tuple(float(value) for value in scores),
                timestamp=datetime.now(timezone.utc).isoformat(),
                num_experts=num_experts,
                top_k=actual_top_k,
            )
            self.events.append(event)
            if self._output_file is not None:
                self._output_file.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")
                self._output_file.flush()


class Qwen3MoeTraceSession(AbstractContextManager["Qwen3MoeTraceSession"]):
    """Temporarily replace supported Qwen3 MoE blocks with trace-only wrappers."""

    def __init__(self, model: Any, tracer: RouteTracer) -> None:
        self.model = model
        self.tracer = tracer
        self._replacements: list[tuple[Any, Any]] = []

    def __enter__(self) -> Qwen3MoeTraceSession:
        layers = _model_layers(self.model)
        for layer_id, layer in enumerate(layers):
            block = getattr(layer, "mlp", None)
            if not _is_supported_qwen3_moe_block(block):
                continue
            wrapper = _TracingQwen3MoeBlock(block, self.tracer, layer_id)
            layer.mlp = wrapper
            self._replacements.append((layer, block))
        if not self._replacements:
            raise UnsupportedModelError(
                "no Qwen3-MoE sparse blocks found; expected blocks with gate, switch_mlp, "
                "num_experts, and top_k"
            )
        return self

    def __exit__(self, *_: object) -> None:
        for layer, original in reversed(self._replacements):
            layer.mlp = original
        self._replacements.clear()


class _TracingQwen3MoeBlock:
    """Exact copy of mlx-lm's current Qwen3 sparse block router/execution path."""

    def __init__(self, block: Any, tracer: RouteTracer, layer_id: int) -> None:
        self._block = block
        self._tracer = tracer
        self._layer_id = layer_id

    def __call__(self, x: Any) -> Any:
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - misconfigured runtime
            message = "Qwen3 route tracing requires the 'mlx' package"
            raise OptionalRuntimeDependencyError(message) from error

        gates = self._block.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self._block.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self._block.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)

        # Evaluation is deliberately tracing-only. It observes exactly the same
        # arrays later passed to switch_mlp and does not alter their values.
        mx.eval(inds, scores)
        self._tracer.record_routes(
            self._layer_id,
            inds,
            scores,
            num_experts=int(self._block.num_experts),
            top_k=int(k),
        )
        y = self._block.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


def trace_qwen3_generation(
    model_path: str,
    prompt: str,
    *,
    max_tokens: int,
    output_path: Path,
    request_id: str | None = None,
    prefill_step_size: int = 2048,
) -> RouteTracer:
    """Run normal greedy mlx-lm decoding while emitting an exact Qwen3 route trace."""

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models import cache
    except ModuleNotFoundError as error:
        raise OptionalRuntimeDependencyError(
            "the trace command requires mlx, mlx-lm, and their runtime dependencies"
        ) from error

    if max_tokens < 0:
        raise ValueError("max_tokens must be zero or greater")
    if prefill_step_size <= 0:
        raise ValueError("prefill_step_size must be greater than zero")

    model, tokenizer = _load_model_without_remote_code(load, model_path)
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not token_ids:
        raise ValueError("prompt tokenization produced no tokens")

    prompt_tokens = mx.array(token_ids)
    prompt_cache = cache.make_prompt_cache(model)
    with RouteTracer(request_id=request_id, output_path=output_path) as tracer:
        with Qwen3MoeTraceSession(model, tracer):
            remaining = prompt_tokens
            # Match mlx_lm.generate_step: reserve the final prompt token for the
            # first greedy sample and put all previous tokens through prefill.
            while remaining.size > 1:
                count = min(prefill_step_size, int(remaining.size - 1))
                with tracer.model_call("prefill", count):
                    model(remaining[:count][None], cache=prompt_cache)
                mx.eval([entry.state for entry in prompt_cache])
                remaining = remaining[count:]
                mx.clear_cache()

            with tracer.model_call("prefill", 1):
                logits = model(remaining[None], cache=prompt_cache)
            mx.eval(logits)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

            for _ in range(max_tokens):
                with tracer.model_call("decode", 1):
                    logits = model(next_token[:, None], cache=prompt_cache)
                mx.eval(logits)
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
    return tracer


def load_trace(path: Path) -> list[RouteEvent]:
    """Load and validate a JSONL route trace."""

    events: list[RouteEvent] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                event = RouteEvent.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid route trace at {path}:{line_number}") from error
            events.append(event)
    return events


def summarize_trace(
    events: Sequence[RouteEvent],
    capacities: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00),
) -> dict[str, Any]:
    """Produce the M1 routing-locality and uniform-byte LRU report."""

    ordered_events = sorted(events, key=lambda event: (event.token_index, event.layer_id))
    layer_events: dict[int, list[RouteEvent]] = defaultdict(list)
    for event in ordered_events:
        layer_events[event.layer_id].append(event)

    layers: dict[str, Any] = {}
    for layer_id, values in sorted(layer_events.items()):
        frequency = Counter(expert for event in values for expert in event.expert_ids)
        previous_by_phase: dict[TracePhase, RouteEvent] = {}
        jaccards: list[float] = []
        overlaps: list[int] = []
        working_set: set[int] = set()
        working_set_curve: list[dict[str, int]] = []
        for call_index, event in enumerate(values, start=1):
            current = set(event.expert_ids)
            previous = previous_by_phase.get(event.phase)
            if previous is not None:
                prior = set(previous.expert_ids)
                union = current | prior
                jaccards.append(len(current & prior) / len(union) if union else 1.0)
                overlaps.append(len(current & prior))
            previous_by_phase[event.phase] = event
            working_set.update(current)
            working_set_curve.append({"calls": call_index, "unique_experts": len(working_set)})

        total_calls = sum(frequency.values())
        entropy = (
            -sum((count / total_calls) * log2(count / total_calls) for count in frequency.values())
            if total_calls
            else 0.0
        )
        declared_num_experts = {
            event.num_experts for event in values if event.num_experts is not None
        }
        declared_top_k = {event.top_k for event in values if event.top_k is not None}
        layers[str(layer_id)] = {
            "events": len(values),
            "num_experts": next(iter(declared_num_experts), max(working_set, default=-1) + 1),
            "top_k": next(iter(declared_top_k), len(values[0].expert_ids) if values else 0),
            "expert_frequency": {str(expert): frequency[expert] for expert in sorted(frequency)},
            "route_entropy_bits": entropy,
            "mean_consecutive_jaccard": _mean(jaccards),
            "mean_consecutive_exact_overlap": _mean(overlaps),
            "unique_experts": len(working_set),
            "cumulative_working_set": working_set_curve,
        }

    accesses = [
        ExpertKey(event.layer_id, expert) for event in ordered_events for expert in event.expert_ids
    ]
    cache_curve = simulate_lru_curve(accesses, capacities)
    return {
        "schema_version": 1,
        "events": len(ordered_events),
        "requests": sorted({event.request_id for event in ordered_events}),
        "layers": layers,
        "cache_simulation": cache_curve,
        "assumptions": {
            "cache_policy": "global_lru",
            "bundle_size": "uniform; M2 manifest byte sizes are not available yet",
            "ordering": "token_index, then layer_id",
        },
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write a stable, human-readable trace summary JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_layers(model: Any) -> Sequence[Any]:
    direct_layers = getattr(model, "layers", None)
    if direct_layers is not None:
        return direct_layers
    nested_model = getattr(model, "model", None)
    nested_layers = getattr(nested_model, "layers", None)
    if nested_layers is not None:
        return nested_layers
    raise UnsupportedModelError("model has no discoverable decoder layers")


def _load_model_without_remote_code(load: Any, model_path: str) -> tuple[Any, Any]:
    """Load across mlx-lm versions without ever opting into remote model code."""

    # mlx-lm 0.31.x did not expose trust_remote_code on ``load``. Its loader
    # therefore cannot enable remote Python model code through this API. Newer
    # versions do expose the option, and must keep it explicitly disabled.
    load_kwargs: dict[str, Any] = {}
    if "trust_remote_code" in inspect.signature(load).parameters:
        load_kwargs["trust_remote_code"] = False
    return load(model_path, **load_kwargs)


def _is_supported_qwen3_moe_block(block: Any) -> bool:
    return block is not None and all(
        hasattr(block, attribute)
        for attribute in ("gate", "switch_mlp", "num_experts", "top_k", "norm_topk_prob")
    )


def _rows_from_array_like(value: Any) -> list[list[Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise TraceProtocolError("router result must be an array or nested sequence")

    rows: list[list[Any]] = []

    def visit(item: Any) -> None:
        if not isinstance(item, list | tuple):
            raise TraceProtocolError("router result does not have a top-k axis")
        if item and all(not isinstance(value, list | tuple) for value in item):
            rows.append(list(item))
            return
        for child in item:
            visit(child)

    visit(value)
    return rows


def _validate_phase(value: Any) -> TracePhase:
    if value not in ("prefill", "decode"):
        raise ValueError(f"invalid trace phase {value!r}")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(values) / len(values) if values else None
