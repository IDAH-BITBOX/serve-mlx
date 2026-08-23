"""Exact routed-expert runtimes for M3–M7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .cache import CacheStats, ExpertKey, ResidentCache, ResidentExpert
from .execution import MaterializedExpert, ReferenceExpertBackend
from .manifest import ModelManifest
from .memory import MemoryBudgetManager, MemoryPressureEvent
from .prefetch import (
    AsyncExpertLoader,
    LoaderStats,
    PredictivePrefetchConfig,
    PredictivePrefetchScheduler,
    PredictivePrefetchStats,
    TimelineEvent,
    TransitionPredictor,
)
from .storage import SafetensorsExpertStore, materialize_mlx_array

PrefillOrder = Literal["expert_id", "resident_first", "disk_offset"]
PREFILL_ORDERS: frozenset[str] = frozenset(
    {"expert_id", "resident_first", "disk_offset"}
)


@dataclass(frozen=True)
class PrefillLayerStats:
    """One M5 expert-major scheduling decision for a sparse layer."""

    layer: int
    token_count: int
    route_count: int
    unique_experts: int
    order: PrefillOrder


@dataclass(frozen=True)
class IoOverlapStats:
    """M6 demand/prefetch state for one runtime instance."""

    workers: int
    prefetch_depth: int
    async_gpu: bool
    loader: LoaderStats


@dataclass(frozen=True)
class RuntimeStats:
    expert_resolutions: int
    bytes_read: int
    read_count: int
    cache: CacheStats | None = None
    prefill_layers: tuple[PrefillLayerStats, ...] = ()
    io_overlap: IoOverlapStats | None = None
    memory_events: tuple[MemoryPressureEvent, ...] = ()
    predictive_prefetch: PredictivePrefetchStats | None = None


class NoCacheExpertRuntime:
    """Exact no-cache baseline: every M3 route group reads from SSD afresh."""

    def __init__(
        self,
        manifest: ModelManifest,
        *,
        expert_activation: Literal["swiglu", "geglu"] = "swiglu",
        predictor: TransitionPredictor | None = None,
        predictive_config: PredictivePrefetchConfig | None = None,
        io_workers: int = 0,
        prefetch_depth: int = 1,
        async_gpu: bool = False,
        memory_manager: MemoryBudgetManager | None = None,
    ) -> None:
        _validate_io_settings(io_workers, prefetch_depth, async_gpu)
        _validate_predictive_settings(predictor, predictive_config, io_workers)
        self.manifest = manifest
        self.store = SafetensorsExpertStore()
        self._loader = _make_loader(self.store, io_workers, prefetch_depth)
        self._io_workers = io_workers
        self._prefetch_depth = prefetch_depth if self._loader is not None else 0
        self._async_gpu = async_gpu
        self._memory_manager = memory_manager
        self._predictive = _make_predictive_scheduler(manifest, predictor, predictive_config)
        self.backend = ReferenceExpertBackend(manifest.quantization, activation=expert_activation)
        self._expert_resolutions = 0
        self._route_history: list[tuple[int, tuple[int, ...]]] = []
        self._prefill_layers: list[PrefillLayerStats] = []

    def close(self) -> None:
        if self._loader is not None:
            self._loader.close()
        self.store.close()

    def resolve(self, layer: int, expert: int) -> MaterializedExpert:
        key = ExpertKey(layer, expert)
        try:
            bundle = self.manifest.expert_bundles[key]
        except KeyError as error:
            raise ValueError(f"route selected expert outside manifest: {key}") from error
        raw_tensors = self._read_bundle(key, bundle)
        arrays = self._materialize(key, bundle, raw_tensors)
        self._expert_resolutions += 1
        return MaterializedExpert(arrays=arrays, nbytes=bundle.total_bytes)

    def execute(self, layer: int, expert: int, x: Any) -> Any:
        key = ExpertKey(layer, expert)
        output = self.backend.execute(x, self.resolve(layer, expert))
        self._enqueue_compute(key, output)
        return output

    def execute_group(self, layer: int, expert: int, x: Any) -> Any:
        """Load one expert once, preserving vector-kernel numerics per token.

        MLX's batched quantized matmul is not bitwise-equivalent to repeated
        vector calls for the supported Qwen quantization.  M5 therefore groups
        materialization and I/O while retaining the M3 vector execution
        primitive as the exact reference semantic.
        """

        if x.ndim != 2:
            raise ValueError("expert-major execution expects x=[tokens, hidden]")
        materialized = self.resolve(layer, expert)
        output = self._mx().stack(
            [
                self.backend.execute(x[token_index], materialized)
                for token_index in range(x.shape[0])
            ]
        )
        self._enqueue_compute(ExpertKey(layer, expert), output)
        return output

    def synchronize_batch(self, output: Any) -> Any:
        """Match the cache-runtime interface without changing the M3 baseline."""

        if self._async_gpu:
            self._mx().eval(output)
            self._record_event("gpu_done")
        return output

    def abort_batch(self) -> None:
        """Match the cache-runtime interface without retaining any M3 state."""

    def record_routes(self, layer: int, expert_rows: list[list[int]]) -> None:
        """Record exact router IDs for M3 correctness comparison only."""

        self._route_history.extend(
            (layer, tuple(int(expert) for expert in row)) for row in expert_rows
        )
        self._schedule_predictions(layer, expert_rows)

    def record_prefill_layer(
        self,
        layer: int,
        *,
        token_count: int,
        route_count: int,
        unique_experts: int,
        order: PrefillOrder,
    ) -> None:
        self._prefill_layers.append(
            PrefillLayerStats(layer, token_count, route_count, unique_experts, order)
        )

    def order_experts(
        self, layer: int, experts: list[int], order: PrefillOrder
    ) -> list[int]:
        return _order_experts(self.manifest, layer, experts, order)

    @property
    def prefetch_depth(self) -> int:
        return self._prefetch_depth

    def prefetch(
        self, layer: int, expert: int, *, origin: Literal["prefetch", "predictive"] = "prefetch"
    ) -> bool:
        if self._loader is None:
            return False
        key = ExpertKey(layer, expert)
        try:
            bundle = self.manifest.expert_bundles[key]
        except KeyError as error:
            raise ValueError(f"route selected expert outside manifest: {key}") from error
        return self._loader.prefetch(key, bundle, origin=origin)

    def timeline(self) -> tuple[TimelineEvent, ...]:
        return self._loader.timeline() if self._loader is not None else ()

    def route_history(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return tuple(self._route_history)

    def stats(self) -> RuntimeStats:
        metrics = self.store.metrics()
        return RuntimeStats(
            expert_resolutions=self._expert_resolutions,
            bytes_read=metrics.bytes_read,
            read_count=metrics.read_count,
            prefill_layers=tuple(self._prefill_layers),
            io_overlap=self._io_stats(),
            memory_events=(self._memory_manager.events() if self._memory_manager else ()),
            predictive_prefetch=(self._predictive.stats() if self._predictive else None),
        )

    def _read_bundle(self, key: ExpertKey, bundle: Any) -> dict[str, bytes]:
        if self._loader is None:
            return self.store.read_bundle(bundle)
        return self._loader.demand(key, bundle)

    def _materialize(
        self, key: ExpertKey, bundle: Any, raw_tensors: dict[str, bytes]
    ) -> dict[str, Any]:
        self._record_event("materialize_start", key)
        try:
            mx = self._mx()
            return {
                tensor.role: materialize_mlx_array(
                    raw_tensors[tensor.role], tensor.dtype, tensor.shape, mx
                )
                for tensor in bundle.tensors
            }
        finally:
            self._record_event("materialize_end", key)

    def _enqueue_compute(self, key: ExpertKey, output: Any) -> None:
        if self._async_gpu:
            self._record_event("gpu_enqueue", key)
            self._mx().async_eval(output)

    def _record_event(self, name: str, key: ExpertKey | None = None) -> None:
        if self._loader is not None:
            self._loader.record(name, key)

    def _io_stats(self) -> IoOverlapStats | None:
        if self._loader is None:
            return None
        return IoOverlapStats(
            workers=self._io_workers,
            prefetch_depth=self._prefetch_depth,
            async_gpu=self._async_gpu,
            loader=self._loader.stats(),
        )

    def _schedule_predictions(self, layer: int, expert_rows: list[list[int]]) -> None:
        if self._predictive is not None:
            self._predictive.schedule(
                layer,
                expert_rows,
                lambda target_layer, expert: self.prefetch(
                    target_layer, expert, origin="predictive"
                ),
            )

    @staticmethod
    def _mx() -> Any:
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("NoCacheExpertRuntime requires MLX") from error
        return mx


class CachedExpertRuntime:
    """M4 exact runtime with a global byte-budgeted resident expert cache.

    A miss reads and materializes exactly one manifest bundle.  A hit reuses
    the same MLX arrays.  Experts remain pinned until the full sparse MoE layer
    output has been evaluated, making eviction safe with MLX's lazy execution
    without introducing a synchronization point per routed expert.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        *,
        capacity_bytes: int,
        expert_activation: Literal["swiglu", "geglu"] = "swiglu",
        predictor: TransitionPredictor | None = None,
        predictive_config: PredictivePrefetchConfig | None = None,
        io_workers: int = 0,
        prefetch_depth: int = 1,
        async_gpu: bool = False,
        memory_manager: MemoryBudgetManager | None = None,
    ) -> None:
        _validate_io_settings(io_workers, prefetch_depth, async_gpu)
        _validate_predictive_settings(predictor, predictive_config, io_workers)
        self.manifest = manifest
        self.store = SafetensorsExpertStore()
        self._loader = _make_loader(self.store, io_workers, prefetch_depth)
        self._io_workers = io_workers
        self._prefetch_depth = prefetch_depth if self._loader is not None else 0
        self._async_gpu = async_gpu
        self._memory_manager = memory_manager
        self._predictive = _make_predictive_scheduler(manifest, predictor, predictive_config)
        self.backend = ReferenceExpertBackend(manifest.quantization, activation=expert_activation)
        self.cache = ResidentCache(capacity_bytes)
        self._expert_resolutions = 0
        self._route_history: list[tuple[int, tuple[int, ...]]] = []
        self._prefill_layers: list[PrefillLayerStats] = []
        self._batch_pins: list[ExpertKey] = []

    def close(self) -> None:
        self.abort_batch()
        if self._loader is not None:
            self._loader.close()
        self.store.close()

    def resolve(self, layer: int, expert: int) -> MaterializedExpert:
        """Return a resident expert, loading it only on an LRU miss."""

        resident = self._resolve_resident(layer, expert)
        return MaterializedExpert(arrays=resident.arrays, nbytes=resident.nbytes)

    def execute(self, layer: int, expert: int, x: Any) -> Any:
        resident = self._resolve_resident(layer, expert)
        self.cache.pin(resident.key)
        self._batch_pins.append(resident.key)
        try:
            output = self.backend.execute(
                x, MaterializedExpert(arrays=resident.arrays, nbytes=resident.nbytes)
            )
            self._enqueue_compute(resident.key, output)
            return output
        except BaseException:
            self._batch_pins.pop()
            self.cache.unpin(resident.key)
            raise

    def execute_group(self, layer: int, expert: int, x: Any) -> Any:
        """Reuse one resident expert while retaining exact vector-kernel math."""

        if x.ndim != 2:
            raise ValueError("expert-major execution expects x=[tokens, hidden]")
        resident = self._resolve_resident(layer, expert)
        self.cache.pin(resident.key)
        self._batch_pins.append(resident.key)
        materialized = MaterializedExpert(arrays=resident.arrays, nbytes=resident.nbytes)
        try:
            output = self._mx().stack(
                [
                    self.backend.execute(x[token_index], materialized)
                    for token_index in range(x.shape[0])
                ]
            )
            self._enqueue_compute(resident.key, output)
            return output
        except BaseException:
            self._batch_pins.pop()
            self.cache.unpin(resident.key)
            raise

    def synchronize_batch(self, output: Any) -> Any:
        """Finish one sparse layer before making its pinned experts evictable."""

        try:
            self._mx().eval(output)
        finally:
            self._release_batch_pins()
        if self._memory_manager is not None:
            event = self._memory_manager.enforce(self.cache)
            if event.action == "disable_prefetch":
                self._prefetch_depth = 0
        if self._async_gpu:
            self._record_event("gpu_done")
        return output

    def abort_batch(self) -> None:
        """Release pins after a failed layer execution without evaluating it."""

        self._release_batch_pins()

    def record_routes(self, layer: int, expert_rows: list[list[int]]) -> None:
        self._route_history.extend(
            (layer, tuple(int(expert) for expert in row)) for row in expert_rows
        )
        self._schedule_predictions(layer, expert_rows)

    def record_prefill_layer(
        self,
        layer: int,
        *,
        token_count: int,
        route_count: int,
        unique_experts: int,
        order: PrefillOrder,
    ) -> None:
        self._prefill_layers.append(
            PrefillLayerStats(layer, token_count, route_count, unique_experts, order)
        )

    def order_experts(
        self, layer: int, experts: list[int], order: PrefillOrder
    ) -> list[int]:
        return _order_experts(self.manifest, layer, experts, order, self.cache)

    @property
    def prefetch_depth(self) -> int:
        return self._prefetch_depth

    def prefetch(
        self, layer: int, expert: int, *, origin: Literal["prefetch", "predictive"] = "prefetch"
    ) -> bool:
        if self._loader is None:
            return False
        key = ExpertKey(layer, expert)
        if self.cache.contains(key):
            return False
        try:
            bundle = self.manifest.expert_bundles[key]
        except KeyError as error:
            raise ValueError(f"route selected expert outside manifest: {key}") from error
        return self._loader.prefetch(key, bundle, origin=origin)

    def timeline(self) -> tuple[TimelineEvent, ...]:
        return self._loader.timeline() if self._loader is not None else ()

    def route_history(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return tuple(self._route_history)

    def stats(self) -> RuntimeStats:
        metrics = self.store.metrics()
        return RuntimeStats(
            expert_resolutions=self._expert_resolutions,
            bytes_read=metrics.bytes_read,
            read_count=metrics.read_count,
            cache=self.cache.stats(),
            prefill_layers=tuple(self._prefill_layers),
            io_overlap=self._io_stats(),
            memory_events=(self._memory_manager.events() if self._memory_manager else ()),
            predictive_prefetch=(self._predictive.stats() if self._predictive else None),
        )

    def _resolve_resident(self, layer: int, expert: int) -> ResidentExpert:
        key = ExpertKey(layer, expert)
        try:
            bundle = self.manifest.expert_bundles[key]
        except KeyError as error:
            raise ValueError(f"route selected expert outside manifest: {key}") from error

        resident = self.cache.get(key, nbytes=bundle.total_bytes)
        if resident is None:
            # Reserve before issuing I/O so an impossible budget fails without
            # reading an SSD bundle that cannot be retained safely.
            self.cache.reserve(bundle.total_bytes)
            raw_tensors = self._read_bundle(key, bundle)
            arrays = self._materialize(key, bundle, raw_tensors)
            resident = self.cache.admit(
                ResidentExpert(
                    key=key,
                    arrays=arrays,
                    nbytes=bundle.total_bytes,
                    last_used_step=0,
                )
            )
        self._expert_resolutions += 1
        return resident

    def _release_batch_pins(self) -> None:
        while self._batch_pins:
            self.cache.unpin(self._batch_pins.pop())

    def _read_bundle(self, key: ExpertKey, bundle: Any) -> dict[str, bytes]:
        if self._loader is None:
            return self.store.read_bundle(bundle)
        return self._loader.demand(key, bundle)

    def _materialize(
        self, key: ExpertKey, bundle: Any, raw_tensors: dict[str, bytes]
    ) -> dict[str, Any]:
        self._record_event("materialize_start", key)
        try:
            mx = self._mx()
            return {
                tensor.role: materialize_mlx_array(
                    raw_tensors[tensor.role], tensor.dtype, tensor.shape, mx
                )
                for tensor in bundle.tensors
            }
        finally:
            self._record_event("materialize_end", key)

    def _enqueue_compute(self, key: ExpertKey, output: Any) -> None:
        if self._async_gpu:
            self._record_event("gpu_enqueue", key)
            self._mx().async_eval(output)

    def _record_event(self, name: str, key: ExpertKey | None = None) -> None:
        if self._loader is not None:
            self._loader.record(name, key)

    def _io_stats(self) -> IoOverlapStats | None:
        if self._loader is None:
            return None
        return IoOverlapStats(
            workers=self._io_workers,
            prefetch_depth=self._prefetch_depth,
            async_gpu=self._async_gpu,
            loader=self._loader.stats(),
        )

    def _schedule_predictions(self, layer: int, expert_rows: list[list[int]]) -> None:
        if self._predictive is not None:
            self._predictive.schedule(
                layer,
                expert_rows,
                lambda target_layer, expert: self.prefetch(
                    target_layer, expert, origin="predictive"
                ),
            )

    @staticmethod
    def _mx() -> Any:
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("CachedExpertRuntime requires MLX") from error
        return mx


def _order_experts(
    manifest: ModelManifest,
    layer: int,
    experts: list[int],
    order: PrefillOrder,
    cache: ResidentCache | None = None,
) -> list[int]:
    """Return a deterministic M5 group order without affecting cache metrics."""

    if order not in PREFILL_ORDERS:
        raise ValueError(f"unsupported prefill order {order!r}")
    if order == "expert_id":
        return sorted(experts)
    if order == "resident_first":
        return sorted(
            experts,
            key=lambda expert: (
                not (cache is not None and cache.contains(ExpertKey(layer, expert))),
                expert,
            ),
        )

    def disk_offset(expert: int) -> tuple[str, int, int]:
        bundle = manifest.expert_bundles[ExpertKey(layer, expert)]
        first = min(bundle.tensors, key=lambda tensor: (str(tensor.file), tensor.offset))
        return str(first.file), first.offset, expert

    return sorted(experts, key=disk_offset)


def _validate_io_settings(io_workers: int, prefetch_depth: int, async_gpu: bool) -> None:
    if io_workers < 0:
        raise ValueError("M6 I/O worker count cannot be negative")
    if prefetch_depth < 0:
        raise ValueError("M6 prefetch depth cannot be negative")
    if async_gpu and io_workers == 0:
        raise ValueError("M6 async GPU evaluation requires at least one I/O worker")


def _validate_predictive_settings(
    predictor: TransitionPredictor | None,
    config: PredictivePrefetchConfig | None,
    io_workers: int,
) -> None:
    if (predictor is None) != (config is None):
        raise ValueError("M10 predictor and predictive prefetch config must be supplied together")
    if predictor is not None and io_workers == 0:
        raise ValueError("M10 predictive prefetch requires at least one M6 I/O worker")


def _make_predictive_scheduler(
    manifest: ModelManifest,
    predictor: TransitionPredictor | None,
    config: PredictivePrefetchConfig | None,
) -> PredictivePrefetchScheduler | None:
    if predictor is None:
        return None
    assert config is not None
    return PredictivePrefetchScheduler(manifest, predictor, config)


def _make_loader(
    store: SafetensorsExpertStore, io_workers: int, prefetch_depth: int
) -> AsyncExpertLoader | None:
    if io_workers == 0:
        return None
    return AsyncExpertLoader(
        store,
        workers=io_workers,
        max_inflight=max(io_workers, io_workers + prefetch_depth),
    )
