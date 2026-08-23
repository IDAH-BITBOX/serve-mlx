"""Storage interfaces that are independent from MLX model objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..manifest import ExpertBundleSpec


class StorageReadError(IOError):
    """An exact source-byte read could not be completed safely."""


@dataclass(frozen=True)
class StorageReadMetrics:
    bytes_read: int = 0
    read_count: int = 0


class ExpertStore(Protocol):
    """Disk-backed source of exact expert tensor bytes."""

    def read_bundle(self, bundle: ExpertBundleSpec) -> Mapping[str, bytes]: ...

    def metrics(self) -> StorageReadMetrics: ...
