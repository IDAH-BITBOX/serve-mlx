"""Bounded localhost OpenAI-compatible API with M9 lazy model switching."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from mlx_lm import stream_generate

from ..errors import MemoryPressureError
from ..models import StreamingEngine
from .registry import ModelRegistration, ModelRegistry

LOGGER = logging.getLogger(__name__)
Endpoint = Literal["completion", "chat_completion"]


class ApiRequestError(ValueError):
    """An OpenAI-shaped request that this deliberately small server rejects."""

    def __init__(
        self,
        message: str,
        *,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
        parameter: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.parameter = parameter
        self.code = code


@dataclass(frozen=True)
class ServerConfig:
    """Intentional M8/M9 limits for serialized local generation."""

    model_id: str = "mlx-moe-stream"
    max_prompt_tokens: int = 4_096
    max_completion_tokens: int = 256
    max_request_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("default model ID cannot be empty")
        if self.max_prompt_tokens <= 0:
            raise ValueError("M8 maximum prompt tokens must be greater than zero")
        if self.max_completion_tokens < 0:
            raise ValueError("M8 maximum completion tokens cannot be negative")
        if self.max_request_bytes <= 0:
            raise ValueError("M8 maximum request bytes must be greater than zero")


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
    """Thread-safe M8 aggregate and last-request observability state."""

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
        """The current engine, retained for M8 embedding compatibility."""

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
        return snapshot

    def completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ApiRequestError("'prompt' must be a string", parameter="prompt")
        model_id, max_tokens = self._validate_generation_options(payload)
        result, metric = self._generate(
            model_id=model_id,
            prompt_builder=lambda _: prompt,
            max_tokens=max_tokens,
            endpoint="completion",
        )
        return {
            "id": metric.request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "text": result,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": _finish_reason(metric.completion_tokens, max_tokens),
                }
            ],
            "usage": _usage(metric),
        }

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = self._normalize_messages(payload.get("messages"))
        model_id, max_tokens = self._validate_generation_options(payload)
        result, metric = self._generate(
            model_id=model_id,
            prompt_builder=lambda engine: self._render_chat_prompt(messages, engine.tokenizer),
            max_tokens=max_tokens,
            endpoint="chat_completion",
        )
        return {
            "id": metric.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": _finish_reason(metric.completion_tokens, max_tokens),
                }
            ],
            "usage": _usage(metric),
        }

    def _validate_generation_options(self, payload: dict[str, Any]) -> tuple[str, int]:
        model = payload.get("model", self.config.model_id)
        if not isinstance(model, str) or not self.registry.contains(model):
            raise ApiRequestError(
                f"unknown local model {model!r}",
                status=HTTPStatus.NOT_FOUND,
                parameter="model",
                code="model_not_found",
            )
        if payload.get("stream", False):
            raise ApiRequestError("stream=true is not implemented in M8", parameter="stream")
        if payload.get("n", 1) != 1:
            raise ApiRequestError("only n=1 is supported", parameter="n")
        for option in (
            "temperature",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
            "tools",
            "tool_choice",
            "response_format",
        ):
            if option in payload:
                raise ApiRequestError(
                    f"{option!r} is not implemented; M8 uses exact greedy generation",
                    parameter=option,
                )
        max_tokens = payload.get("max_tokens", self.config.max_completion_tokens)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ApiRequestError(
                "'max_tokens' must be a non-negative integer", parameter="max_tokens"
            )
        if max_tokens > self.config.max_completion_tokens:
            raise ApiRequestError(
                f"'max_tokens' exceeds the server limit ({self.config.max_completion_tokens})",
                parameter="max_tokens",
            )
        return model, max_tokens

    def _normalize_messages(self, messages: Any) -> list[dict[str, str]]:
        if not isinstance(messages, list) or not messages:
            raise ApiRequestError("'messages' must be a non-empty array", parameter="messages")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ApiRequestError(
                    f"messages[{index}] must be an object", parameter="messages"
                )
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ApiRequestError(
                    f"messages[{index}].role is unsupported", parameter="messages"
                )
            if not isinstance(content, str):
                raise ApiRequestError(
                    f"messages[{index}].content must be a string", parameter="messages"
                )
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _render_chat_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str):
                return rendered
            raise ApiRequestError("tokenizer chat template did not return text")
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def _generate(
        self,
        *,
        model_id: str,
        prompt_builder: Callable[[StreamingEngine], str],
        max_tokens: int,
        endpoint: Endpoint,
    ) -> tuple[str, GenerationMetric]:
        if not self._generation_slot.acquire(blocking=False):
            self.metrics.reject_busy()
            raise ApiRequestError(
                "one generation is already active", status=HTTPStatus.TOO_MANY_REQUESTS, code="busy"
            )
        self.metrics.begin_generation()
        started = time.perf_counter()
        first_token_at: float | None = None
        output_parts: list[str] = []
        completion_tokens = 0
        try:
            engine = self.registry.activate(model_id)
            prompt = prompt_builder(engine)
            prompt_tokens = _count_tokens(engine.tokenizer, prompt, add_special_tokens=True)
            if prompt_tokens > self.config.max_prompt_tokens:
                raise ApiRequestError(
                    f"prompt has {prompt_tokens} tokens; limit is {self.config.max_prompt_tokens}",
                    parameter="prompt",
                )
            for response in stream_generate(
                engine.model,
                engine.tokenizer,
                prompt,
                max_tokens=max_tokens,
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_parts.append(str(response.text))
                completion_tokens = int(getattr(response, "generation_tokens", completion_tokens))
                prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens))
            elapsed = time.perf_counter() - started
            ttft = first_token_at - started if first_token_at is not None else None
            stats = engine.runtime.stats()
            snapshot = engine.memory_manager.snapshot()
            cache = stats.cache
            predictive = getattr(stats, "predictive_prefetch", None)
            io_overlap = getattr(stats, "io_overlap", None)
            loader = io_overlap.loader if io_overlap is not None else None
            metric = GenerationMetric(
                request_id=_request_id(endpoint),
                model_id=model_id,
                endpoint=endpoint,
                elapsed_seconds=elapsed,
                ttft_seconds=ttft,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prefill_tokens_per_second=(prompt_tokens / ttft if ttft else 0.0),
                decode_tokens_per_second=(
                    max(completion_tokens - 1, 0) / (elapsed - ttft)
                    if ttft is not None and elapsed > ttft
                    else 0.0
                ),
                disk_bytes=stats.bytes_read,
                cache_hit_rate=cache.hit_rate if cache is not None else None,
                resident_bytes=cache.resident_bytes if cache is not None else None,
                mlx_peak_memory_bytes=snapshot.mlx_peak_memory_bytes,
                predictive_prefetch_submitted=(predictive.submitted if predictive else None),
                predictive_prefetch_hits=(loader.predictive_hits if predictive else None),
                predictive_prefetch_unused=(loader.predictive_unused if predictive else None),
            )
        except BaseException:
            self.metrics.fail_generation()
            raise
        else:
            self.metrics.complete_generation(metric)
            LOGGER.info(
                "M8 request=%s endpoint=%s ttft=%.3fs prefill_tok_s=%.2f "
                "decode_tok_s=%.2f cache_hit=%s disk_bytes=%s resident_bytes=%s "
                "predictive_submitted=%s predictive_hits=%s",
                metric.request_id,
                metric.endpoint,
                metric.ttft_seconds or 0.0,
                metric.prefill_tokens_per_second,
                metric.decode_tokens_per_second,
                metric.cache_hit_rate,
                metric.disk_bytes,
                metric.resident_bytes,
                metric.predictive_prefetch_submitted,
                metric.predictive_prefetch_hits,
            )
            return "".join(output_parts), metric
        finally:
            self._generation_slot.release()

    def _load_legacy_engine(self, _: Path) -> StreamingEngine:
        if self._legacy_engine is None:
            raise RuntimeError("the legacy M8 engine has been closed")
        return self._legacy_engine


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
            if path == "/v1/completions":
                response = self.server.service.completions(payload)
            else:
                response = self.server.service.chat_completions(payload)
        except ApiRequestError as error:
            self._send_error(error.status, str(error), parameter=error.parameter, code=error.code)
        except MemoryPressureError as error:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(error), code="memory_pressure")
        except Exception:  # pragma: no cover - defensive boundary for a live server
            LOGGER.exception("M8 request failed")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")
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

    def _send_error(
        self,
        status: HTTPStatus,
        message: str,
        *,
        parameter: str | None = None,
        code: str | None = None,
    ) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": (
                        "invalid_request_error"
                        if status < HTTPStatus.INTERNAL_SERVER_ERROR
                        else "server_error"
                    ),
                    "param": parameter,
                    "code": code,
                }
            },
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug("M8 HTTP %s", format % args)


def is_loopback_host(host: str) -> bool:
    """Whether the no-auth M8 server may bind the supplied explicit host."""

    return host in {"127.0.0.1", "localhost"}


def run_local_server(server: LocalApiServer) -> None:
    """Serve until Ctrl-C, always closing the listening socket afterwards."""

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("M8 server interrupted")
    finally:
        server.server_close()


def _count_tokens(tokenizer: Any, text: str, *, add_special_tokens: bool) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _finish_reason(completion_tokens: int, max_tokens: int) -> str:
    return "length" if max_tokens and completion_tokens >= max_tokens else "stop"


def _request_id(endpoint: Endpoint) -> str:
    prefix = "chatcmpl" if endpoint == "chat_completion" else "cmpl"
    return f"{prefix}-{uuid.uuid4().hex}"


def _usage(metric: GenerationMetric) -> dict[str, int]:
    return {
        "prompt_tokens": metric.prompt_tokens,
        "completion_tokens": metric.completion_tokens,
        "total_tokens": metric.prompt_tokens + metric.completion_tokens,
    }
