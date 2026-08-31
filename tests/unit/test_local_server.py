from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.message import Message
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_moe_stream.errors import MemoryPressureError
from mlx_moe_stream.server import (
    LocalApiServer,
    LocalGenerationService,
    ModelRegistration,
    ModelRegistry,
    ServerConfig,
    app,
)
from mlx_moe_stream.server.app import ApiRequestError


@dataclass
class _Response:
    text: str
    prompt_tokens: int
    generation_tokens: int


class _RemoteImageResponse:
    def __init__(self, payload: bytes, content_type: str) -> None:
        self.payload = payload
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> _RemoteImageResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self, _: int) -> bytes:
        return self.payload


def test_remote_image_loader_uses_a_user_agent_and_passes_bytes_to_vlm(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, *, timeout: int):
        captured["request"] = request
        captured["timeout"] = timeout
        return _RemoteImageResponse(b"image-bytes", "image/jpeg")

    def fake_load_image(source: BytesIO) -> str:
        assert isinstance(source, BytesIO)
        return source.read().decode()

    monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)

    assert app._load_remote_vlm_image(
        "https://upload.wikimedia.org/image.jpg", fake_load_image
    ) == ("image-bytes")
    request = captured["request"]
    assert request.get_header("User-agent") == app._REMOTE_IMAGE_USER_AGENT
    assert captured["timeout"] == app._REMOTE_IMAGE_TIMEOUT_SECONDS


def test_remote_image_loader_rejects_share_or_viewer_html(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        app.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _RemoteImageResponse(b"<html></html>", "text/html"),
    )

    with pytest.raises(ValueError, match="direct image URL"):
        app._load_remote_vlm_image("https://share.google/example", lambda _: object())


class _Tokenizer:
    def __init__(self) -> None:
        self.chat_messages: list[dict[str, str]] | None = None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        del add_special_tokens
        return text.split()

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.chat_messages = messages
        return "CHAT " + " ".join(message["content"] for message in messages)


class _QwenTokenizer(_Tokenizer):
    chat_template = "{% if tools %}tools{% endif %} {{ enable_thinking }}"

    def __init__(self) -> None:
        super().__init__()
        self.template_kwargs: dict[str, Any] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.chat_messages = messages
        self.template_kwargs = kwargs
        return "CHAT " + " ".join(str(message["content"]) for message in messages)


class _Engine:
    def __init__(self) -> None:
        self.closed = False
        self.model = object()
        self.tokenizer = _Tokenizer()
        self.runtime = SimpleNamespace(
            stats=lambda: SimpleNamespace(
                bytes_read=123,
                cache=SimpleNamespace(hit_rate=0.5, resident_bytes=456),
            )
        )
        self.memory_manager = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(mlx_peak_memory_bytes=789)
        )

    def close(self) -> None:
        self.closed = True


class _VisionProcessor:
    chat_template = "{% if tools %}tools{% endif %} {{ enable_thinking }}"

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.chat_messages: list[dict[str, Any]] | None = None
        self.images: list[Any] | None = None
        self.text: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: Any,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.chat_messages = messages
        return "VISION CHAT"

    def __call__(self, *, images: list[Any] | None, text: str, **_: Any) -> dict[str, Any]:
        self.images = images
        self.text = text
        return {
            "input_ids": SimpleNamespace(size=3),
            "pixel_values": "pixels",
            "image_grid_thw": "grid",
        }


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> LocalGenerationService:
    def fake_stream_generate(*_: Any, **__: Any):
        yield _Response("hello", 2, 1)
        yield _Response(" world", 2, 2)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", fake_stream_generate)
    return LocalGenerationService(
        _Engine(), ServerConfig(model_id="test-moe", max_prompt_tokens=8, max_completion_tokens=4)
    )


def test_openai_completion_and_chat_shapes_include_usage_and_metrics(
    service: LocalGenerationService,
):
    completion = service.completions({"model": "test-moe", "prompt": "hello there"})
    assert completion["object"] == "text_completion"
    assert completion["choices"][0]["text"] == "hello world"
    assert completion["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}

    chat = service.chat_completions(
        {"model": "test-moe", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"] == {"role": "assistant", "content": "hello world"}
    assert service.engine.tokenizer.chat_messages == [{"role": "user", "content": "hello"}]
    metrics = service.metrics.snapshot()
    assert metrics["completed_total"] == 2
    assert metrics["last_generation"]["disk_bytes"] == 123
    assert metrics["last_generation"]["resident_bytes"] == 456


def test_metrics_failure_after_generation_does_not_block_the_next_request(
    service: LocalGenerationService, caplog: pytest.LogCaptureFixture
):
    assert service._legacy_engine is not None

    def fail_snapshot() -> None:
        raise RuntimeError("metrics backend unavailable")

    service._legacy_engine.memory_manager.snapshot = fail_snapshot
    with caplog.at_level("ERROR"):
        first = service.completions({"model": "test-moe", "prompt": "first"})
    second = service.completions({"model": "test-moe", "prompt": "second"})

    assert first["choices"][0]["text"] == "hello world"
    assert second["choices"][0]["text"] == "hello world"
    metrics = service.metrics.snapshot()
    assert metrics["completed_total"] == 2
    assert metrics["failed_total"] == 0
    assert metrics["active_generations"] == 0
    assert metrics["last_generation"]["mlx_peak_memory_bytes"] == 0
    assert metrics["last_generation"]["observability_available"] is False
    assert "generation metrics collection failed" in caplog.text


def test_generation_limits_and_single_active_generation_are_explicit(
    service: LocalGenerationService,
):
    with pytest.raises(ApiRequestError, match="server limit"):
        service.completions({"model": "test-moe", "prompt": "hello", "max_tokens": 5})
    with pytest.raises(ApiRequestError, match="streaming completion transport"):
        service.completions({"model": "test-moe", "prompt": "hello", "stream": True})
    with pytest.raises(ApiRequestError, match="limit is 8"):
        service.completions(
            {"model": "test-moe", "prompt": "one two three four five six seven eight nine"}
        )

    assert service._generation_slot.acquire(blocking=False)
    try:
        with pytest.raises(ApiRequestError, match="already active") as error:
            service.completions({"model": "test-moe", "prompt": "hello"})
        assert error.value.status == 429
    finally:
        service._generation_slot.release()


def test_mlx_out_of_memory_becomes_actionable_memory_pressure(
    service: LocalGenerationService, monkeypatch: pytest.MonkeyPatch
):
    def oom_stream_generate(*_: Any, **__: Any):
        raise RuntimeError(
            "[METAL] Command buffer execution failed: "
            "Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        )
        yield  # pragma: no cover - establishes generator type

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", oom_stream_generate)
    with pytest.raises(MemoryPressureError, match="ran out of Unified Memory"):
        service.completions({"model": "test-moe", "prompt": "hello"})
    assert service.metrics.snapshot()["active_generations"] == 0

    def normal_stream_generate(*_: Any, **__: Any):
        yield _Response("recovered", 1, 1)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", normal_stream_generate)
    assert service.completions({"model": "test-moe", "prompt": "retry"})["choices"][0][
        "text"
    ] == "recovered"


def test_m11_streaming_stop_sequence_and_usage(
    service: LocalGenerationService, monkeypatch: pytest.MonkeyPatch
):
    def fake_stream_generate(*_: Any, **__: Any):
        yield _Response("Hello E", 1, 1)
        yield _Response("ND ignored", 1, 2)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", fake_stream_generate)
    events = list(
        service.completion_events(
            {
                "model": "test-moe",
                "prompt": "hello",
                "stream": True,
                "stop": "END",
                "stream_options": {"include_usage": True},
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 0.1,
                "frequency_penalty": -0.1,
                "logit_bias": {"7": 1.0},
            }
        )
    )

    text_events = [event["choices"][0]["text"] for event in events if event["choices"]]
    assert text_events == ["Hello ", ""]
    assert events[1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_m12_image_chat_uses_vision_processor_and_vlm_stream(
    service: LocalGenerationService, monkeypatch: pytest.MonkeyPatch
):
    assert service._legacy_engine is not None
    processor = _VisionProcessor()
    service._legacy_engine.processor = processor
    captured: dict[str, Any] = {}
    loaded_sources: list[str] = []

    def fake_vlm_stream_generate(*args: Any, **kwargs: Any):
        captured["args"] = args
        captured["kwargs"] = kwargs
        yield _Response("a tiny image", 3, 1)

    monkeypatch.setattr("mlx_moe_stream.server.app._stream_vlm_generate", fake_vlm_stream_generate)
    monkeypatch.setattr(
        "mlx_moe_stream.server.app._load_vlm_image",
        lambda source: loaded_sources.append(str(source)) or object(),
    )
    response = service.chat_completions(
        {
            "model": "test-moe",
            "reasoning_effort": "none",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/cat.png"},
                        },
                        {"type": "text", "text": "Describe it."},
                    ],
                }
            ],
        }
    )

    assert response["choices"][0]["message"]["content"] == "a tiny image"
    assert processor.chat_messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": processor.chat_messages[0]["content"][0]["image"]},
                {"type": "text", "text": "Describe it."},
            ],
        }
    ]
    assert processor.images is not None and len(processor.images) == 1
    assert loaded_sources == ["https://example.test/cat.png"]
    assert processor.text == "VISION CHAT"
    assert captured["args"][1] is processor
    assert captured["kwargs"]["input_ids"].size == 3
    assert captured["kwargs"]["pixel_values"] == "pixels"
    assert captured["kwargs"]["image_grid_thw"] == "grid"


def test_m12_image_chat_requires_a_vision_loaded_engine(service: LocalGenerationService):
    with pytest.raises(ApiRequestError, match="without vision"):
        service.chat_completions(
            {
                "model": "test-moe",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.test/image.png",
                            }
                        ],
                    }
                ],
            }
        )


def test_m12_text_completion_uses_vlm_preparation(service: LocalGenerationService, monkeypatch):
    assert service._legacy_engine is not None
    processor = _VisionProcessor()
    service._legacy_engine.processor = processor
    captured: dict[str, Any] = {}

    def fake_vlm_stream_generate(*args: Any, **kwargs: Any):
        captured["args"] = args
        captured["kwargs"] = kwargs
        yield _Response("completed", 2, 1)

    monkeypatch.setattr("mlx_moe_stream.server.app._stream_vlm_generate", fake_vlm_stream_generate)
    response = service.completions({"model": "test-moe", "prompt": "Complete this"})

    assert response["choices"][0]["text"] == "completed"
    assert captured["args"][1] is processor
    assert "input_ids" not in captured["kwargs"]


def test_kv_cache_generation_kwargs_are_forwarded_to_mlx(service, monkeypatch):
    service.config = ServerConfig(
        model_id="test-moe",
        max_prompt_tokens=8,
        max_completion_tokens=4,
        prefill_step_size=256,
    )
    assert service._legacy_engine is not None
    service._legacy_engine.kv_cache = SimpleNamespace(
        generation_kwargs=lambda: {"kv_bits": 4, "kv_group_size": 64},
        to_dict=lambda: {"effective_mode": "4bit"},
    )
    captured: dict[str, Any] = {}

    def fake_stream_generate(*_: Any, **kwargs: Any):
        captured.update(kwargs)
        yield _Response("ok", 1, 1)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", fake_stream_generate)
    response = service.completions({"model": "test-moe", "prompt": "hello"})

    assert response["choices"][0]["text"] == "ok"
    assert captured["kv_bits"] == 4
    assert captured["kv_group_size"] == 64
    assert captured["prefill_step_size"] == 256
    assert service.metrics_snapshot()["kv_cache"] == {"effective_mode": "4bit"}


def test_server_config_rejects_a_nonpositive_prefill_step_size():
    with pytest.raises(ValueError, match="prefill step size"):
        ServerConfig(prefill_step_size=0)


def test_metrics_include_the_active_memory_budget(service: LocalGenerationService):
    assert service._legacy_engine is not None
    service._legacy_engine.memory_budget = SimpleNamespace(
        to_dict=lambda: {"expert_budget_bytes": 2}
    )
    service.completions({"model": "test-moe", "prompt": "hello"})

    assert service.metrics_snapshot()["memory_budget"] == {"expert_budget_bytes": 2}


def test_metrics_include_the_m13_startup_decision_report(service: LocalGenerationService):
    assert service._legacy_engine is not None
    service._legacy_engine.startup_decision = SimpleNamespace(
        report={"mode": "full_residency", "resident_fraction": 1.0}
    )
    service.completions({"model": "test-moe", "prompt": "hello"})

    assert service.metrics_snapshot()["startup"] == {
        "mode": "full_residency",
        "resident_fraction": 1.0,
    }


def test_metrics_startup_is_none_without_a_startup_decision(service: LocalGenerationService):
    assert service._legacy_engine is not None
    service.completions({"model": "test-moe", "prompt": "hello"})

    assert service.metrics_snapshot()["startup"] is None


def test_m11_thinking_and_qwen_tool_calls(
    service: LocalGenerationService, monkeypatch: pytest.MonkeyPatch
):
    assert service._legacy_engine is not None
    service._legacy_engine.tokenizer = _QwenTokenizer()

    def fake_stream_generate(*_: Any, **__: Any):
        yield _Response("route first</think>\n<tool_call>\n<function=get_weather>\n", 2, 1)
        yield _Response('<parameter=city>\n"Seoul"\n</parameter>\n</function>\n</tool_call>', 2, 2)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", fake_stream_generate)
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    response = service.chat_completions(
        {
            "model": "test-moe",
            "messages": [{"role": "user", "content": "Seoul weather?"}],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )
    message = response["choices"][0]["message"]
    assert message["content"] is None
    assert message["reasoning_content"] == "route first"
    assert message["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city":"Seoul"}',
    }
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert service.engine is not None
    assert service.engine.tokenizer.template_kwargs["tools"] == [tool]


def test_m11_streaming_chat_separates_thinking_and_tool_delta(
    service: LocalGenerationService, monkeypatch: pytest.MonkeyPatch
):
    assert service._legacy_engine is not None
    service._legacy_engine.tokenizer = _QwenTokenizer()

    def fake_stream_generate(*_: Any, **__: Any):
        yield _Response("plan", 2, 1)
        yield _Response("</think>\n<tool_call><function=get_time>", 2, 2)
        yield _Response("</function></tool_call>", 2, 3)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", fake_stream_generate)
    events = list(
        service.chat_completion_events(
            {
                "model": "test-moe",
                "messages": [{"role": "user", "content": "time?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_time", "parameters": {"type": "object"}},
                    }
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
    )
    deltas = [event["choices"][0]["delta"] for event in events if event["choices"]]
    assert deltas[0] == {"role": "assistant"}
    assert {"reasoning_content": "plan"} in deltas
    tool_delta = next(delta for delta in deltas if "tool_calls" in delta)
    assert tool_delta["tool_calls"][0]["function"] == {"name": "get_time", "arguments": "{}"}
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["usage"]["completion_tokens"] == 3


def test_m11_json_schema_response_is_checked(
    service: LocalGenerationService, monkeypatch: pytest.MonkeyPatch
):
    service.config = ServerConfig(
        model_id="test-moe", max_prompt_tokens=128, max_completion_tokens=4
    )
    assert service._legacy_engine is not None
    service._legacy_engine.tokenizer = _QwenTokenizer()

    def fake_stream_generate(*_: Any, **__: Any):
        yield _Response('{"answer":"ok"}', 2, 1)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", fake_stream_generate)
    response = service.chat_completions(
        {
            "model": "test-moe",
            "messages": [{"role": "user", "content": "answer"}],
            "reasoning_effort": "none",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        }
    )
    assert response["choices"][0]["message"]["content"] == '{"answer":"ok"}'

    with pytest.raises(ApiRequestError, match="streaming structured output"):
        service.chat_completion_events(
            {
                "model": "test-moe",
                "messages": [{"role": "user", "content": "answer"}],
                "stream": True,
                "response_format": {"type": "json_object"},
            }
        )


def test_http_endpoints_return_openai_json(
    service: LocalGenerationService, monkeypatch: pytest.MonkeyPatch
):
    mlx_call_threads: list[int | None] = []

    def tracking_stream_generate(*_: Any, **__: Any):
        # MLX work must remain on the one persistent serve_forever thread, not
        # a new ThreadingHTTPServer handler thread for every request.
        mlx_call_threads.append(threading.current_thread().ident)
        yield _Response("hello", 2, 1)
        yield _Response(" world", 2, 2)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", tracking_stream_generate)
    server = LocalApiServer("127.0.0.1", 0, service)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        health = _get_json(f"{base_url}/health")
        assert health["status"] == "ok"
        assert _get_json(f"{base_url}/v1/models")["data"][0]["id"] == "test-moe"
        assert _get_json(f"{base_url}/v1/models/test-moe") == {
            "id": "test-moe",
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }
        ollama_model = _post_json(f"{base_url}/api/show", {"name": "test-moe"})
        assert ollama_model["details"] == {
            "parent_model": "",
            "format": "mlx",
            "family": "local",
            "families": ["local"],
            "parameter_size": "unknown",
            "quantization_level": "unknown",
        }
        assert ollama_model["capabilities"] == ["completion", "tools"]
        response = _post_json(
            f"{base_url}/v1/chat/completions",
            {"model": "test-moe", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response["choices"][0]["message"]["content"] == "hello world"
        # The second request proves that handling a completed request does not
        # end the HTTP serving loop or leave the serialized slot occupied.
        second = _post_json(
            f"{base_url}/v1/completions", {"model": "test-moe", "prompt": "again"}
        )
        assert second["choices"][0]["text"] == "hello world"
        assert _get_json(f"{base_url}/metrics")["completed_total"] == 2
        assert _get_json(f"{base_url}/health")["active_generations"] == 0
        assert mlx_call_threads == [worker.ident, worker.ident]

        def oom_stream_generate(*_: Any, **__: Any):
            raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
            yield  # pragma: no cover - establishes generator type

        monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", oom_stream_generate)
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(f"{base_url}/v1/completions", {"model": "test-moe", "prompt": "oom"})
        assert error.value.code == 503
        assert json.loads(error.value.read())["error"]["code"] == "memory_pressure"
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(f"{base_url}/v1/completions", {"model": "wrong", "prompt": "hello"})
        assert error.value.code == 404
        assert json.loads(error.value.read())["error"]["code"] == "model_not_found"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_http_sse_streams_openai_chunks_and_done(service: LocalGenerationService):
    server = LocalApiServer("127.0.0.1", 0, service)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address[:2]
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "test-moe",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310 - localhost test
            assert response.headers.get_content_type() == "text/event-stream"
            body = response.read().decode("utf-8")
        assert '"role":"assistant"' in body
        assert '"content":"hello"' in body
        assert '"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}' in body
        assert body.endswith("data: [DONE]\n\n")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_ollama_show_reports_qwen_thinking_and_enabled_vision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_model": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "model_type": "qwen3_5_moe",
                "quantization": {"bits": 4},
            }
        )
    )
    monkeypatch.setattr(
        "mlx_moe_stream.server.app.stream_generate",
        lambda *_args, **_kwargs: iter(()),
    )
    registry = ModelRegistry(
        [ModelRegistration("qwen", manifest)],
        load_engine=lambda _path: _Engine(),
    )
    service = LocalGenerationService(
        config=ServerConfig(model_id="qwen", vision_enabled=True), registry=registry
    )

    show = service.ollama_show({"name": "qwen"})

    assert show["details"]["parameter_size"] == "35B"
    assert show["details"]["quantization_level"] == "Q4"
    assert show["capabilities"] == ["completion", "tools", "thinking", "vision"]
    assert registry.snapshot()["active_model_id"] is None


def test_registry_lazily_switches_models_without_two_active_engines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def fake_stream_generate(*_: Any, **__: Any):
        yield _Response("ok", 1, 1)

    monkeypatch.setattr("mlx_moe_stream.server.app.stream_generate", fake_stream_generate)
    loaded: list[tuple[Path, _Engine]] = []

    def load_engine(manifest: Path) -> _Engine:
        engine = _Engine()
        loaded.append((manifest, engine))
        return engine

    registry = ModelRegistry(
        [
            ModelRegistration("qwen", tmp_path / "qwen.json"),
            ModelRegistration("gemma", tmp_path / "gemma.json"),
        ],
        load_engine=load_engine,
    )
    service = LocalGenerationService(
        config=ServerConfig(model_id="qwen", max_prompt_tokens=8, max_completion_tokens=4),
        registry=registry,
    )

    assert service.models()["data"] == [
        {"id": "qwen", "object": "model", "created": 0, "owned_by": "local"},
        {"id": "gemma", "object": "model", "created": 0, "owned_by": "local"},
    ]
    assert registry.snapshot()["active_model_id"] is None
    assert not loaded

    gemma_reply = service.completions({"model": "gemma", "prompt": "hello"})
    assert gemma_reply["model"] == "gemma"
    assert [path.name for path, _ in loaded] == ["gemma.json"]
    assert registry.snapshot()["active_model_id"] == "gemma"

    qwen_reply = service.completions({"model": "qwen", "prompt": "hello"})
    assert qwen_reply["model"] == "qwen"
    assert [path.name for path, _ in loaded] == ["gemma.json", "qwen.json"]
    assert loaded[0][1].closed
    assert registry.snapshot()["switches_total"] == 1
    assert registry.snapshot()["active_model_id"] == "qwen"

    service.close()
    assert loaded[1][1].closed
    assert registry.snapshot()["active_model_id"] is None


@pytest.mark.parametrize("value", ["", "missing-separator", "=manifest.json", "model="])
def test_model_registration_requires_an_explicit_id_and_manifest(value: str):
    with pytest.raises(ValueError, match="MODEL_ID=MANIFEST"):
        ModelRegistration.parse(value)


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - localhost test server
        return json.loads(response.read())


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310 - localhost test server
        return json.loads(response.read())
