"""M13 Node 8: `serve --preload`/`--no-preload` gating.

Uses the same fake-engine pattern as ``tests/unit/test_local_server.py``'s
``ModelRegistry`` tests: a fake ``load_streaming_model`` records every load,
so ``registry.snapshot()["loads_total"]`` (asserted indirectly through the
``loaded`` list, which is 1:1 with a real load) shows whether ``serve``
activated the default model before entering ``run_local_server``.

``LocalApiServer`` and ``run_local_server`` are both faked so this test never
binds a real socket or blocks in ``serve_forever()``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlx_moe_stream import cli


class _FakeEngine:
    def __init__(self) -> None:
        self.closed = False
        self.startup_decision = None
        self.kv_cache = None
        self.memory_budget = SimpleNamespace(to_dict=lambda: {})

    def close(self) -> None:
        self.closed = True


class _FakeServer:
    def __init__(
        self, host: str, port: int, service: Any, *, connection_timeout: float | None = None
    ) -> None:
        self.server_address = (host, port)
        self.service = service
        self.connection_timeout = connection_timeout

    def server_close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("explicit", "registration_count", "expected"),
    [
        (None, 1, True),
        (None, 2, False),
        (False, 1, False),
        (True, 2, True),
    ],
)
def test_resolve_preload_default_matches_single_vs_multi_model_registration(
    explicit: bool | None, registration_count: int, expected: bool
) -> None:
    assert cli._resolve_preload(explicit, registration_count=registration_count) is expected


def _serve_with_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, extra_args: list[str]
) -> list[Path]:
    loaded: list[Path] = []

    def fake_load_streaming_model(manifest_path: Any, **kwargs: Any) -> _FakeEngine:
        loaded.append(Path(manifest_path))
        return _FakeEngine()

    monkeypatch.setattr(cli, "load_streaming_model", fake_load_streaming_model)
    monkeypatch.setattr(cli, "LocalApiServer", _FakeServer)
    monkeypatch.setattr(cli, "run_local_server", lambda *args, **kwargs: None)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    exit_code = cli.main(["serve", "--manifest", str(manifest_path), *extra_args])
    assert exit_code == 0
    return loaded


def test_serve_preloads_the_default_model_by_default_for_a_single_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _serve_with_fakes(monkeypatch, tmp_path, extra_args=[])
    assert len(loaded) == 1


def test_serve_no_preload_skips_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _serve_with_fakes(monkeypatch, tmp_path, extra_args=["--no-preload"])
    assert len(loaded) == 0


def test_serve_explicit_preload_flag_still_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _serve_with_fakes(monkeypatch, tmp_path, extra_args=["--preload"])
    assert len(loaded) == 1
