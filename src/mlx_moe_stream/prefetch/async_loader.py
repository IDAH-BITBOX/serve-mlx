"""Bounded, priority-aware exact expert-byte loading for M6."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from itertools import count
from typing import Literal, Protocol

from ..cache import ExpertKey
from ..manifest import ExpertBundleSpec


class BundleReader(Protocol):
    def read_bundle(self, bundle: ExpertBundleSpec) -> Mapping[str, bytes]: ...


@dataclass(frozen=True)
class TimelineEvent:
    name: str
    timestamp: float
    key: ExpertKey | None = None


@dataclass(frozen=True)
class LoaderStats:
    demand_requests: int
    prefetch_requests: int
    prefetch_submitted: int
    prefetch_hits: int
    coalesced_requests: int
    skipped_prefetches: int
    predictive_requests: int = 0
    predictive_submitted: int = 0
    predictive_hits: int = 0
    predictive_unused: int = 0


@dataclass
class _LoadRequest:
    future: Future[dict[str, bytes]]
    origin: str


class AsyncExpertLoader:
    """Coalesce exact bundle reads through a small demand-priority worker pool.

    The loader owns only raw CPU byte buffers.  MLX materialization remains on
    the model thread, so thread-safety and cache admission stay explicit.
    """

    _STOP = object()

    def __init__(self, reader: BundleReader, *, workers: int, max_inflight: int) -> None:
        if workers <= 0:
            raise ValueError("M6 I/O workers must be greater than zero")
        if max_inflight < workers:
            raise ValueError("M6 max in-flight loads must be at least the worker count")
        self._reader = reader
        self._max_inflight = max_inflight
        self._queue: queue.PriorityQueue[tuple[int, int, object]] = queue.PriorityQueue()
        self._requests: dict[ExpertKey, _LoadRequest] = {}
        self._sequence = count()
        self._lock = threading.Lock()
        self._closed = False
        self._events: list[TimelineEvent] = []
        self._demand_requests = 0
        self._prefetch_requests = 0
        self._prefetch_submitted = 0
        self._prefetch_hits = 0
        self._coalesced_requests = 0
        self._skipped_prefetches = 0
        self._predictive_requests = 0
        self._predictive_submitted = 0
        self._predictive_hits = 0
        self._predictive_unused = 0
        self._workers = [
            threading.Thread(target=self._run, name=f"mlx-moe-io-{index}", daemon=True)
            for index in range(workers)
        ]
        for worker in self._workers:
            worker.start()

    def prefetch(
        self,
        key: ExpertKey,
        bundle: ExpertBundleSpec,
        *,
        origin: Literal["prefetch", "predictive"] = "prefetch",
    ) -> bool:
        """Submit a speculative exact read if a bounded slot is available."""

        with self._lock:
            self._prefetch_requests += 1
            if origin == "predictive":
                self._predictive_requests += 1
            if key in self._requests:
                self._coalesced_requests += 1
                return True
            if self._closed or len(self._requests) >= self._max_inflight:
                self._skipped_prefetches += 1
                return False
            self._submit_locked(key, bundle, priority=1, origin=origin)
            self._prefetch_submitted += 1
            if origin == "predictive":
                self._predictive_submitted += 1
            return True

    def demand(self, key: ExpertKey, bundle: ExpertBundleSpec) -> dict[str, bytes]:
        """Return exact bytes, coalescing with a prefetch when possible."""

        direct_fallback = False
        with self._lock:
            self._demand_requests += 1
            request = self._requests.get(key)
            if request is None:
                if self._closed:
                    raise RuntimeError("M6 I/O loader is closed")
                if len(self._requests) >= self._max_inflight:
                    direct_fallback = True
                else:
                    request = self._submit_locked(key, bundle, priority=0, origin="demand")
            else:
                self._coalesced_requests += 1
            if direct_fallback:
                future = None
                request_origin = None
            else:
                assert request is not None
                future = request.future
                request_origin = request.origin
        if direct_fallback:
            self.record("load_start", key)
            try:
                return dict(self._reader.read_bundle(bundle))
            finally:
                self.record("load_end", key)
        try:
            assert future is not None
            result = future.result()
        finally:
            with self._lock:
                if self._requests.get(key) is request:
                    self._requests.pop(key)
                if request_origin == "prefetch":
                    self._prefetch_hits += 1
                elif request_origin == "predictive":
                    self._predictive_hits += 1
        return result

    def record(self, name: str, key: ExpertKey | None = None) -> None:
        with self._lock:
            self._events.append(TimelineEvent(name=name, timestamp=time.perf_counter(), key=key))

    def stats(self) -> LoaderStats:
        with self._lock:
            return LoaderStats(
                demand_requests=self._demand_requests,
                prefetch_requests=self._prefetch_requests,
                prefetch_submitted=self._prefetch_submitted,
                prefetch_hits=self._prefetch_hits,
                coalesced_requests=self._coalesced_requests,
                skipped_prefetches=self._skipped_prefetches,
                predictive_requests=self._predictive_requests,
                predictive_submitted=self._predictive_submitted,
                predictive_hits=self._predictive_hits,
                predictive_unused=self._predictive_unused,
            )

    def timeline(self) -> tuple[TimelineEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._predictive_unused += sum(
                request.origin == "predictive" for request in self._requests.values()
            )
            for _ in self._workers:
                self._queue.put((2, next(self._sequence), self._STOP))
        for worker in self._workers:
            worker.join()

    def _submit_locked(
        self, key: ExpertKey, bundle: ExpertBundleSpec, *, priority: int, origin: str
    ) -> _LoadRequest:
        request = _LoadRequest(future=Future(), origin=origin)
        self._requests[key] = request
        self._queue.put((priority, next(self._sequence), (key, bundle, request)))
        return request

    def _run(self) -> None:
        while True:
            _, _, item = self._queue.get()
            if item is self._STOP:
                return
            key, bundle, request = item
            self.record("load_start", key)
            try:
                result = dict(self._reader.read_bundle(bundle))
            except BaseException as error:
                request.future.set_exception(error)
            else:
                request.future.set_result(result)
            finally:
                self.record("load_end", key)
