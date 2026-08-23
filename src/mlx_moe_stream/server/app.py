"""Bounded localhost OpenAI-compatible API with streaming M11 protocol support."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from ..errors import MemoryPressureError
from ..models import StreamingEngine
from .protocol import (
    ApiRequestError,
    Endpoint,
    GenerationOptions,
    StopSequenceBuffer,
    ThinkingStreamParser,
    ToolCallStreamParser,
    add_response_format_instruction,
    enforce_tool_choice,
    image_sources,
    normalize_messages,
    parse_generation_options,
    parse_tool_calls,
    render_chat_prompt,
    split_reasoning,
    validate_structured_output,
)
from .registry import ModelRegistration, ModelRegistry

LOGGER = logging.getLogger(__name__)

_REMOTE_IMAGE_USER_AGENT = "mlx-moe-stream/0.1 (+https://github.com/IDAH-BITBOX/serve-mlx)"
_REMOTE_IMAGE_TIMEOUT_SECONDS = 15
_MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class ServerConfig:
    """Deliberate limits for a serialized local streaming server."""

    model_id: str = "mlx-moe-stream"
    max_prompt_tokens: int = 4_096
    max_completion_tokens: int = 256
    max_request_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("default model ID cannot be empty")
        if self.max_prompt_tokens <= 0:
            raise ValueError("maximum prompt tokens must be greater than zero")
        if self.max_completion_tokens < 0:
            raise ValueError("maximum completion tokens cannot be negative")
        if self.max_request_bytes <= 0:
            raise ValueError("maximum request bytes must be greater than zero")


@dataclass(frozen=True)
class GenerationMetric:
    request_id: str
    model_id: str
    endpoint: Endpoint
    elapsed_seconds: float
    ttft_seconds: float | None
    prompt_tokens: int
    completion_tokens: int
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    disk_bytes: int
    cache_hit_rate: float | None
    resident_bytes: int | None
    mlx_peak_memory_bytes: int
    predictive_prefetch_submitted: int | None
    predictive_prefetch_hits: int | None
    predictive_prefetch_unused: int | None


class ServerMetrics:
    """Thread-safe M8–M11 aggregate and last-request observability state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._busy_rejections_total = 0
        self._active_generations = 0
        self._last_generation: GenerationMetric | None = None

    def begin_generation(self) -> None:
        with self._lock:
            self._requests_total += 1
            self._active_generations += 1

    def complete_generation(self, metric: GenerationMetric) -> None:
        with self._lock:
            self._active_generations -= 1
            self._completed_total += 1
            self._last_generation = metric

    def fail_generation(self) -> None:
        with self._lock:
            self._active_generations -= 1
            self._failed_total += 1

    def reject_busy(self) -> None:
        with self._lock:
            self._busy_rejections_total += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self._requests_total,
                "completed_total": self._completed_total,
                "failed_total": self._failed_total,
                "busy_rejections_total": self._busy_rejections_total,
                "active_generations": self._active_generations,
                "last_generation": (
                    asdict(self._last_generation) if self._last_generation is not None else None
                ),
            }


@dataclass(frozen=True)
class _Prompt:
    text: str
    initial_reasoning: bool = False
    vlm_inputs: dict[str, Any] | None = None
    token_count: int | None = None


@dataclass
class _ActiveGeneration:
    request_id: str
    model_id: str
    endpoint: Endpoint
    engine: StreamingEngine
    prompt: _Prompt
    prompt_tokens: int
    options: GenerationOptions
    started: float
    first_token_at: float | None = None
    completion_tokens: int = 0
    stopped: bool = False
    settled: bool = False


class LocalGenerationService:
    """Serialize generation and lazily keep one selected engine in memory."""

    def __init__(
        self,
        engine: StreamingEngine | None = None,
        config: ServerConfig | None = None,
        *,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.config = config or ServerConfig()
        if registry is not None and engine is not None:
            raise ValueError("pass an engine or a model registry, not both")
        self._legacy_engine = engine
        if registry is None:
            if engine is None:
                raise ValueError("M9 service requires an engine or a model registry")
            registry = ModelRegistry(
                [ModelRegistration(self.config.model_id, Path("<in-memory>"))],
                load_engine=self._load_legacy_engine,
            )
        if not registry.contains(self.config.model_id):
            raise ValueError(f"default model {self.config.model_id!r} is not registered")
        self.registry = registry
        self.metrics = ServerMetrics()
        self._generation_slot = threading.BoundedSemaphore(value=1)

    @property
    def engine(self) -> StreamingEngine | None:
        """The current engine, retained for embedding compatibility."""

        return self.registry.active_engine()

    def close(self) -> None:
        self.registry.close()
        self._legacy_engine = None

    def health(self) -> dict[str, Any]:
        registry = self.registry.snapshot()
        return {
            "status": "ok",
            "default_model_id": self.config.model_id,
            "active_model_id": registry["active_model_id"],
            "registered_model_ids": registry["registered_model_ids"],
            "active_generations": self.metrics.snapshot()["active_generations"],
        }

    def models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": registration.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
                for registration in self.registry.registrations()
            ],
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        snapshot = self.metrics.snapshot()
        snapshot["registry"] = self.registry.snapshot()
        engine = self.registry.active_engine()
        kv_cache = getattr(engine, "kv_cache", None) if engine is not None else None
        snapshot["kv_cache"] = kv_cache.to_dict() if kv_cache is not None else None
        memory_budget = getattr(engine, "memory_budget", None) if engine is not None else None
        snapshot["memory_budget"] = memory_budget.to_dict() if memory_budget is not None else None
        return snapshot

    def completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ApiRequestError("'prompt' must be a string", parameter="prompt")
        options = self._options(payload, endpoint="completion")
        if options.stream:
            raise ApiRequestError(
                "use the streaming completion transport for stream=true", parameter="stream"
            )
        active = self._begin_generation(
            options=options,
            endpoint="completion",
            prompt_builder=lambda _: _Prompt(prompt),
        )
        try:
            result = self._collect_text(active)
        except BaseException:
            self._fail_generation(active)
            raise
        metric = self._complete_generation(active)
        return {
            "id": metric.request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": options.model_id,
            "choices": [
                {
                    "text": result,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": _finish_reason(active),
                }
            ],
            "usage": _usage(metric),
        }

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        options = self._options(payload, endpoint="chat_completion")
        messages = add_response_format_instruction(
            normalize_messages(payload.get("messages")), options.response_format
        )
        if options.stream:
            raise ApiRequestError(
                "use the streaming chat transport for stream=true", parameter="stream"
            )
        active = self._begin_generation(
            options=options,
            endpoint="chat_completion",
            prompt_builder=lambda engine: self._chat_prompt(messages, engine, options),
        )
        try:
            raw_result = self._collect_text(active)
            reasoning, visible = split_reasoning(
                raw_result, initial_reasoning=active.prompt.initial_reasoning
            )
            visible, tool_calls = parse_tool_calls(
                visible, {tool["function"]["name"] for tool in options.template_tools}
            )
            enforce_tool_choice(options, tool_calls)
            validate_structured_output(visible, options.response_format)
        except BaseException:
            self._fail_generation(active)
            raise
        metric = self._complete_generation(active)
        message: dict[str, Any] = {"role": "assistant", "content": visible or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": metric.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": options.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else _finish_reason(active),
                }
            ],
            "usage": _usage(metric),
        }

    def completion_events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ApiRequestError("'prompt' must be a string", parameter="prompt")
        options = self._options(payload, endpoint="completion")
        if not options.stream:
            raise ApiRequestError(
                "stream=true is required for the SSE transport", parameter="stream"
            )
        active = self._begin_generation(
            options=options,
            endpoint="completion",
            prompt_builder=lambda _: _Prompt(prompt),
        )
        return self._completion_events(active)

    def chat_completion_events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        options = self._options(payload, endpoint="chat_completion")
        if not options.stream:
            raise ApiRequestError(
                "stream=true is required for the SSE transport", parameter="stream"
            )
        messages = add_response_format_instruction(
            normalize_messages(payload.get("messages")), options.response_format
        )
        active = self._begin_generation(
            options=options,
            endpoint="chat_completion",
            prompt_builder=lambda engine: self._chat_prompt(messages, engine, options),
        )
        return self._chat_events(active)

    def _options(self, payload: dict[str, Any], *, endpoint: Endpoint) -> GenerationOptions:
        return parse_generation_options(
            payload,
            endpoint=endpoint,
            default_model_id=self.config.model_id,
            model_exists=self.registry.contains,
            max_completion_tokens=self.config.max_completion_tokens,
        )

    @staticmethod
    def _chat_prompt(
        messages: list[dict[str, Any]], engine: StreamingEngine, options: GenerationOptions
    ) -> _Prompt:
        processor = getattr(engine, "processor", None)
        images = image_sources(messages)
        if images and processor is None:
            raise ApiRequestError(
                "this model was loaded without vision; restart serve with --vision",
                parameter="messages",
            )
        rendered, initial_reasoning = render_chat_prompt(
            messages,
            processor or engine.tokenizer,
            tools=options.template_tools,
            reasoning_effort=options.reasoning_effort,
        )
        if processor is None:
            return _Prompt(rendered, initial_reasoning)
        try:
            loaded_images = [_load_vlm_image(source) for source in images]
            inputs = processor(
                images=loaded_images or None,
                text=rendered,
                return_tensors="mlx",
                padding=False,
            )
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("processor did not return input_ids")
        except (ImportError, ValueError, OSError) as error:
            raise ApiRequestError(
                f"could not process image input: {error}", parameter="messages"
            ) from error
        return _Prompt(
            rendered,
            initial_reasoning,
            vlm_inputs=dict(inputs),
            token_count=int(input_ids.size),
        )

    def _begin_generation(
        self,
        *,
        options: GenerationOptions,
        endpoint: Endpoint,
        prompt_builder: Callable[[StreamingEngine], _Prompt],
    ) -> _ActiveGeneration:
        if not self._generation_slot.acquire(blocking=False):
            self.metrics.reject_busy()
            raise ApiRequestError(
                "one generation is already active", status=HTTPStatus.TOO_MANY_REQUESTS, code="busy"
            )
        self.metrics.begin_generation()
        started = time.perf_counter()
        try:
            engine = self.registry.activate(options.model_id)
            prompt = prompt_builder(engine)
            prompt_tokens = prompt.token_count
            if prompt_tokens is None:
                prompt_tokens = _count_tokens(
                    engine.tokenizer, prompt.text, add_special_tokens=True
                )
            if prompt_tokens > self.config.max_prompt_tokens:
                raise ApiRequestError(
                    f"prompt has {prompt_tokens} tokens; limit is {self.config.max_prompt_tokens}",
                    parameter="prompt",
                )
            return _ActiveGeneration(
                request_id=_request_id(endpoint),
                model_id=options.model_id,
                endpoint=endpoint,
                engine=engine,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                options=options,
                started=started,
            )
        except BaseException:
            self.metrics.fail_generation()
            self._generation_slot.release()
            raise

    def _collect_text(self, active: _ActiveGeneration) -> str:
        return "".join(self._text_fragments(active))

    def _text_fragments(self, active: _ActiveGeneration) -> Iterator[str]:
        stop_buffer = StopSequenceBuffer(active.options.stop)
        responses: Any = None
        try:
            if active.options.seed is not None:
                # The server serializes generation, so the process-global MLX RNG
                # cannot race another request.
                mx.random.seed(active.options.seed)
            processor = getattr(active.engine, "processor", None)
            if processor is None:
                responses = stream_generate(
                    active.engine.model,
                    active.engine.tokenizer,
                    active.prompt.text,
                    max_tokens=active.options.max_tokens,
                    **self._mlx_generation_kwargs(active.options, active.engine),
                )
            else:
                if active.prompt.vlm_inputs is None:
                    # /v1/completions has no chat content parts. Let MLX-VLM
                    # prepare a text-only input using its normal processor path.
                    responses = _stream_vlm_generate(
                        active.engine.model,
                        processor,
                        active.prompt.text,
                        max_tokens=active.options.max_tokens,
                        **self._mlx_generation_kwargs(active.options, active.engine),
                    )
                else:
                    inputs = dict(active.prompt.vlm_inputs)
                    input_ids = inputs.pop("input_ids", None)
                    if input_ids is None:
                        raise RuntimeError("VLM prompt is missing input_ids")
                    pixel_values = inputs.pop("pixel_values", None)
                    mask = inputs.pop("attention_mask", None)
                    responses = _stream_vlm_generate(
                        active.engine.model,
                        processor,
                        active.prompt.text,
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        mask=mask,
                        max_tokens=active.options.max_tokens,
                        **inputs,
                        **self._mlx_generation_kwargs(active.options, active.engine),
                    )
            for response in responses:
                if active.first_token_at is None:
                    active.first_token_at = time.perf_counter()
                active.completion_tokens = int(
                    getattr(response, "generation_tokens", active.completion_tokens)
                )
                active.prompt_tokens = int(getattr(response, "prompt_tokens", active.prompt_tokens))
                emitted = stop_buffer.feed(str(getattr(response, "text", "")))
                if emitted:
                    yield emitted
                if stop_buffer.stopped:
                    active.stopped = True
                    break
            tail = stop_buffer.flush()
            if tail:
                yield tail
        finally:
            close = getattr(responses, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _mlx_generation_kwargs(
        options: GenerationOptions, engine: StreamingEngine | None = None
    ) -> dict[str, Any]:
        processors = make_logits_processors(
            logit_bias=options.logit_bias,
            presence_penalty=options.presence_penalty,
            frequency_penalty=options.frequency_penalty,
        )
        kwargs: dict[str, Any] = {
            "sampler": make_sampler(temp=options.temperature, top_p=options.top_p)
        }
        if processors:
            kwargs["logits_processors"] = processors
        kv_cache = getattr(engine, "kv_cache", None) if engine is not None else None
        if kv_cache is not None:
            kwargs.update(kv_cache.generation_kwargs())
        return kwargs

    def _completion_events(self, active: _ActiveGeneration) -> Iterator[dict[str, Any]]:
        try:
            for fragment in self._text_fragments(active):
                yield _completion_chunk(active, text=fragment)
        except BaseException:
            self._fail_generation(active)
            raise
        metric = self._complete_generation(active)
        yield _completion_chunk(active, text="", finish_reason=_finish_reason(active))
        if active.options.include_usage:
            yield {
                "id": active.request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": active.model_id,
                "choices": [],
                "usage": _usage(metric),
            }

    def _chat_events(self, active: _ActiveGeneration) -> Iterator[dict[str, Any]]:
        thinking = ThinkingStreamParser(initial_reasoning=active.prompt.initial_reasoning)
        tool_parser = ToolCallStreamParser(
            {tool["function"]["name"] for tool in active.options.template_tools}
        )
        tool_calls: list[dict[str, Any]] = []
        try:
            yield _chat_chunk(active, {"role": "assistant"})
            for raw_fragment in self._text_fragments(active):
                for kind, fragment in thinking.feed(raw_fragment):
                    yield from self._chat_protocol_events(
                        active, kind, fragment, tool_parser, tool_calls
                    )
            for kind, fragment in thinking.flush():
                yield from self._chat_protocol_events(
                    active, kind, fragment, tool_parser, tool_calls
                )
            for kind, fragment in tool_parser.flush():
                yield from self._chat_tool_events(active, kind, fragment, tool_calls)
            enforce_tool_choice(active.options, tool_calls)
        except BaseException:
            self._fail_generation(active)
            raise
        metric = self._complete_generation(active)
        yield _chat_chunk(
            active,
            {},
            finish_reason="tool_calls" if tool_calls else _finish_reason(active),
        )
        if active.options.include_usage:
            yield {
                "id": active.request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": active.model_id,
                "choices": [],
                "usage": _usage(metric),
            }

    def _chat_protocol_events(
        self,
        active: _ActiveGeneration,
        kind: str,
        fragment: str,
        tool_parser: ToolCallStreamParser,
        tool_calls: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        if kind == "reasoning_content":
            if fragment:
                yield _chat_chunk(active, {"reasoning_content": fragment})
            return
        for tool_kind, tool_fragment in tool_parser.feed(fragment):
            yield from self._chat_tool_events(active, tool_kind, tool_fragment, tool_calls)

    @staticmethod
    def _chat_tool_events(
        active: _ActiveGeneration,
        kind: str,
        fragment: Any,
        tool_calls: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        if kind == "content":
            if fragment:
                yield _chat_chunk(active, {"content": fragment})
            return
        if kind != "tool_calls":  # pragma: no cover - defensive parser contract
            raise RuntimeError(f"unexpected tool stream fragment {kind!r}")
        for call in fragment:
            index = len(tool_calls)
            tool_calls.append(call)
            yield _chat_chunk(
                active,
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": call["function"],
                        }
                    ]
                },
            )

    def _complete_generation(self, active: _ActiveGeneration) -> GenerationMetric:
        if active.settled:
            raise RuntimeError("generation was already settled")
        active.settled = True
        elapsed = time.perf_counter() - active.started
        ttft = active.first_token_at - active.started if active.first_token_at is not None else None
        stats = active.engine.runtime.stats()
        snapshot = active.engine.memory_manager.snapshot()
        cache = getattr(stats, "cache", None)
        predictive = getattr(stats, "predictive_prefetch", None)
        io_overlap = getattr(stats, "io_overlap", None)
        loader = io_overlap.loader if io_overlap is not None else None
        metric = GenerationMetric(
            request_id=active.request_id,
            model_id=active.model_id,
            endpoint=active.endpoint,
            elapsed_seconds=elapsed,
            ttft_seconds=ttft,
            prompt_tokens=active.prompt_tokens,
            completion_tokens=active.completion_tokens,
            prefill_tokens_per_second=(active.prompt_tokens / ttft if ttft else 0.0),
            decode_tokens_per_second=(
                max(active.completion_tokens - 1, 0) / (elapsed - ttft)
                if ttft is not None and elapsed > ttft
                else 0.0
            ),
            disk_bytes=stats.bytes_read,
            cache_hit_rate=cache.hit_rate if cache is not None else None,
            resident_bytes=cache.resident_bytes if cache is not None else None,
            mlx_peak_memory_bytes=snapshot.mlx_peak_memory_bytes,
            predictive_prefetch_submitted=(predictive.submitted if predictive else None),
            predictive_prefetch_hits=(loader.predictive_hits if predictive and loader else None),
            predictive_prefetch_unused=(
                loader.predictive_unused if predictive and loader else None
            ),
        )
        self.metrics.complete_generation(metric)
        self._generation_slot.release()
        LOGGER.info(
            "M11 request=%s endpoint=%s ttft=%.3fs prefill_tok_s=%.2f "
            "decode_tok_s=%.2f cache_hit=%s disk_bytes=%s resident_bytes=%s",
            metric.request_id,
            metric.endpoint,
            metric.ttft_seconds or 0.0,
            metric.prefill_tokens_per_second,
            metric.decode_tokens_per_second,
            metric.cache_hit_rate,
            metric.disk_bytes,
            metric.resident_bytes,
        )
        return metric

    def _fail_generation(self, active: _ActiveGeneration) -> None:
        if active.settled:
            return
        active.settled = True
        self.metrics.fail_generation()
        self._generation_slot.release()

    def _load_legacy_engine(self, _: Path) -> StreamingEngine:
        if self._legacy_engine is None:
            raise RuntimeError("the legacy engine has been closed")
        return self._legacy_engine


def _stream_vlm_generate(*args: Any, **kwargs: Any) -> Any:
    """Late import keeps image serving an explicit optional dependency."""

    try:
        from mlx_vlm import stream_generate as generate
    except ModuleNotFoundError as error:  # pragma: no cover - loader already guards this
        raise RuntimeError("VLM serving requires `pip install mlx-moe-stream[vlm]`") from error
    return generate(*args, **kwargs)


def _load_vlm_image(source: str) -> Any:
    """Load one local, URL, or data-URI image with the optional VLM package."""

    try:
        from mlx_vlm.utils import load_image
    except ModuleNotFoundError as error:  # pragma: no cover - loader already guards this
        raise RuntimeError("VLM serving requires `pip install mlx-moe-stream[vlm]`") from error
    if urlparse(source).scheme in {"http", "https"}:
        return _load_remote_vlm_image(source, load_image)
    return load_image(source)


def _load_remote_vlm_image(source: str, load_image: Callable[[Any], Any]) -> Any:
    """Fetch a direct image URL with a descriptive User-Agent before decoding.

    Some image CDNs, including Wikimedia, reject Python clients with the
    default urllib/requests User-Agent.  Passing a decoded byte stream to the
    MLX-VLM loader keeps local paths and data URIs on its normal path while
    making direct remote image URLs reliable.
    """

    request = urllib.request.Request(source, headers={"User-Agent": _REMOTE_IMAGE_USER_AGENT})
    with urllib.request.urlopen(request, timeout=_REMOTE_IMAGE_TIMEOUT_SECONDS) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise ValueError(
                f"remote image URL returned {content_type!r}, not image data; "
                "use a direct image URL rather than a share or viewer page"
            )
        payload = response.read(_MAX_REMOTE_IMAGE_BYTES + 1)
    if len(payload) > _MAX_REMOTE_IMAGE_BYTES:
        raise ValueError(f"remote image exceeds the {_MAX_REMOTE_IMAGE_BYTES // 1024**2} MiB limit")
    return load_image(BytesIO(payload))


class LocalApiServer(ThreadingHTTPServer):
    """Threaded HTTP transport; generation itself stays serialized in the service."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, service: LocalGenerationService) -> None:
        self.service = service
        super().__init__((host, port), _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    server: LocalApiServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(HTTPStatus.OK, self.server.service.health())
        elif path == "/v1/models":
            self._send_json(HTTPStatus.OK, self.server.service.models())
        elif path == "/metrics":
            self._send_json(HTTPStatus.OK, self.server.service.metrics_snapshot())
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "endpoint not found", code="not_found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        path = urlparse(self.path).path
        if path not in {"/v1/completions", "/v1/chat/completions"}:
            self._send_error(HTTPStatus.NOT_FOUND, "endpoint not found", code="not_found")
            return
        try:
            payload = self._read_json_body()
            stream = payload.get("stream") is True
            if path == "/v1/completions":
                response = (
                    self.server.service.completion_events(payload)
                    if stream
                    else self.server.service.completions(payload)
                )
            else:
                response = (
                    self.server.service.chat_completion_events(payload)
                    if stream
                    else self.server.service.chat_completions(payload)
                )
        except ApiRequestError as error:
            self._send_error(error.status, str(error), parameter=error.parameter, code=error.code)
        except MemoryPressureError as error:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(error), code="memory_pressure")
        except Exception:  # pragma: no cover - defensive boundary for a live server
            LOGGER.exception("M11 request setup failed")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")
        else:
            if stream:
                self._send_sse(response)
            else:
                self._send_json(HTTPStatus.OK, response)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ApiRequestError("Content-Length is required")
        try:
            body_size = int(content_length)
        except ValueError as error:
            raise ApiRequestError("invalid Content-Length") from error
        if body_size < 0 or body_size > self.server.service.config.max_request_bytes:
            raise ApiRequestError(
                f"request body exceeds {self.server.service.config.max_request_bytes} bytes",
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            payload = json.loads(self.rfile.read(body_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiRequestError("request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ApiRequestError("request JSON must be an object")
        return payload

    def _send_sse(self, events: Iterator[dict[str, Any]]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for event in events:
                self._send_sse_data(event)
        except ApiRequestError as error:
            self._send_sse_data(
                _error_payload(
                    error.status, str(error), parameter=error.parameter, code=error.code
                ),
                event="error",
            )
        except MemoryPressureError as error:
            self._send_sse_data(
                _error_payload(HTTPStatus.SERVICE_UNAVAILABLE, str(error), code="memory_pressure"),
                event="error",
            )
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("M11 SSE client disconnected")
            return
        except Exception:  # pragma: no cover - defensive live transport boundary
            LOGGER.exception("M11 SSE request failed")
            self._send_sse_data(
                _error_payload(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error"),
                event="error",
            )
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("M11 SSE client disconnected before [DONE]")

    def _send_sse_data(self, payload: dict[str, Any], *, event: str | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if event is not None:
            self.wfile.write(f"event: {event}\n".encode("ascii"))
        self.wfile.write(b"data: " + body + b"\n\n")
        self.wfile.flush()

    def _send_error(
        self,
        status: HTTPStatus,
        message: str,
        *,
        parameter: str | None = None,
        code: str | None = None,
    ) -> None:
        self._send_json(status, _error_payload(status, message, parameter=parameter, code=code))

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug("M11 HTTP %s", format % args)


def is_loopback_host(host: str) -> bool:
    """Whether the no-auth server may bind the supplied explicit host."""

    return host in {"127.0.0.1", "localhost"}


def run_local_server(server: LocalApiServer) -> None:
    """Serve until Ctrl-C, always closing the listening socket afterwards."""

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("M11 server interrupted")
    finally:
        server.server_close()


def _completion_chunk(
    active: _ActiveGeneration, *, text: str, finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "id": active.request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": active.model_id,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
    }


def _chat_chunk(
    active: _ActiveGeneration, delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "id": active.request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": active.model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _count_tokens(tokenizer: Any, text: str, *, add_special_tokens: bool) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _finish_reason(active: _ActiveGeneration) -> str:
    if active.stopped:
        return "stop"
    return (
        "length"
        if active.options.max_tokens and active.completion_tokens >= active.options.max_tokens
        else "stop"
    )


def _request_id(endpoint: Endpoint) -> str:
    prefix = "chatcmpl" if endpoint == "chat_completion" else "cmpl"
    return f"{prefix}-{uuid.uuid4().hex}"


def _usage(metric: GenerationMetric) -> dict[str, int]:
    return {
        "prompt_tokens": metric.prompt_tokens,
        "completion_tokens": metric.completion_tokens,
        "total_tokens": metric.prompt_tokens + metric.completion_tokens,
    }


def _error_payload(
    status: HTTPStatus,
    message: str,
    *,
    parameter: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error"
            if status < HTTPStatus.INTERNAL_SERVER_ERROR
            else "server_error",
            "param": parameter,
            "code": code,
        }
    }
