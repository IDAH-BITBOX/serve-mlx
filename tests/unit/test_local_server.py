from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_moe_stream.server import (
    LocalApiServer,
    LocalGenerationService,
    ModelRegistration,
    ModelRegistry,
    ServerConfig,
)
from mlx_moe_stream.server.app import ApiRequestError


@dataclass
class _Response:
    text: str
    prompt_tokens: int
    generation_tokens: int


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


def test_generation_limits_and_single_active_generation_are_explicit(
    service: LocalGenerationService,
):
    with pytest.raises(ApiRequestError, match="server limit"):
        service.completions({"model": "test-moe", "prompt": "hello", "max_tokens": 5})
    with pytest.raises(ApiRequestError, match="not implemented"):
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


def test_http_endpoints_return_openai_json(service: LocalGenerationService):
    server = LocalApiServer("127.0.0.1", 0, service)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        health = _get_json(f"{base_url}/health")
        assert health["status"] == "ok"
        assert _get_json(f"{base_url}/v1/models")["data"][0]["id"] == "test-moe"
        response = _post_json(
            f"{base_url}/v1/chat/completions",
            {"model": "test-moe", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response["choices"][0]["message"]["content"] == "hello world"
        assert _get_json(f"{base_url}/metrics")["completed_total"] == 1
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(f"{base_url}/v1/completions", {"model": "wrong", "prompt": "hello"})
        assert error.value.code == 404
        assert json.loads(error.value.read())["error"]["code"] == "model_not_found"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


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
