"""Lazy one-active-engine registry for local multi-model serving."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import StreamingEngine


@dataclass(frozen=True)
class ModelRegistration:
    """One externally visible OpenAI model ID and its prepared manifest."""

    model_id: str
    manifest_path: Path

    def __post_init__(self) -> None:
        if not self.model_id or self.model_id.strip() != self.model_id:
            raise ValueError("model ID must be non-empty and may not have surrounding whitespace")
        if not str(self.manifest_path):
            raise ValueError("manifest path must not be empty")

    @classmethod
    def parse(cls, value: str) -> ModelRegistration:
        """Parse the CLI's ``MODEL_ID=MANIFEST`` representation."""

        model_id, separator, manifest = value.partition("=")
        if not separator or not model_id or not manifest:
            raise ValueError("--model must use MODEL_ID=MANIFEST")
        return cls(model_id=model_id, manifest_path=Path(manifest))


class ModelRegistry:
    """Own registrations while keeping exactly zero or one engine live.

    Activation is intentionally serialized. An old engine is closed and its
    last Python reference discarded before the next shell can allocate, which
    prevents a Qwen/Gemma switch from temporarily holding both large shells in
    Unified Memory.
    """

    def __init__(
        self,
        registrations: Iterable[ModelRegistration],
        *,
        load_engine: Callable[[Path], StreamingEngine],
    ) -> None:
        models = tuple(registrations)
        if not models:
            raise ValueError("M9 model registry requires at least one model")
        self._registrations = {registration.model_id: registration for registration in models}
        if len(self._registrations) != len(models):
            raise ValueError("M9 model registry has duplicate model IDs")
        self._load_engine = load_engine
        self._lock = threading.RLock()
        self._active_model_id: str | None = None
        self._active_engine: StreamingEngine | None = None
        self._loads_total = 0
        self._unloads_total = 0
        self._switches_total = 0
        self._load_failures_total = 0

    @property
    def default_model_id(self) -> str:
        return next(iter(self._registrations))

    def contains(self, model_id: str) -> bool:
        return model_id in self._registrations

    def registrations(self) -> tuple[ModelRegistration, ...]:
        return tuple(self._registrations.values())

    def active_engine(self) -> StreamingEngine | None:
        with self._lock:
            return self._active_engine

    def activate(self, model_id: str) -> StreamingEngine:
        """Return the requested engine, unloading any different active model first."""

        with self._lock:
            try:
                registration = self._registrations[model_id]
            except KeyError as error:
                raise ValueError(f"unknown registered model {model_id!r}") from error
            if self._active_model_id == model_id and self._active_engine is not None:
                return self._active_engine

            had_active_model = self._active_engine is not None
            self._unload_active_locked()
            try:
                engine = self._load_engine(registration.manifest_path)
            except BaseException:
                self._load_failures_total += 1
                raise
            self._active_model_id = model_id
            self._active_engine = engine
            self._loads_total += 1
            if had_active_model:
                self._switches_total += 1
            return engine

    def close(self) -> None:
        """Close the active engine, if any, without touching downloaded models."""

        with self._lock:
            self._unload_active_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "registered_model_ids": [
                    registration.model_id for registration in self.registrations()
                ],
                "active_model_id": self._active_model_id,
                "loads_total": self._loads_total,
                "unloads_total": self._unloads_total,
                "switches_total": self._switches_total,
                "load_failures_total": self._load_failures_total,
            }

    def _unload_active_locked(self) -> None:
        engine = self._active_engine
        if engine is None:
            return
        self._active_engine = None
        self._active_model_id = None
        try:
            engine.close()
        finally:
            del engine
            gc.collect()
            self._unloads_total += 1
