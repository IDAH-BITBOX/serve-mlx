"""Qwen3-MoE adapter with exact streamed experts and M7 memory budgets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.nn as nn

from ..kv_cache import KvCacheConfig, KvCacheDecision
from ..manifest import ModelManifest, load_manifest
from ..memory import MemoryBudgetConfig, MemoryBudgetDecision, MemoryBudgetManager
from ..prefetch import PredictivePrefetchConfig, TransitionPredictor
from ..runtime import PREFILL_ORDERS, CachedExpertRuntime, NoCacheExpertRuntime, PrefillOrder
from ..startup import StartupDecision, prepare_streaming_runtime
from ..storage import load_nonexpert_weights


@dataclass
class StreamingEngine:
    """A single-user exact engine with one of the routed-expert runtimes."""

    model: Any
    tokenizer: Any
    runtime: NoCacheExpertRuntime | CachedExpertRuntime
    memory_manager: MemoryBudgetManager
    memory_budget: MemoryBudgetDecision
    processor: Any | None = None
    kv_cache: KvCacheDecision | None = None
    startup_decision: StartupDecision | None = None

    def close(self) -> None:
        self.runtime.close()

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        kv_kwargs = self.kv_cache.generation_kwargs() if self.kv_cache is not None else {}
        if self.processor is not None:
            try:
                from mlx_vlm import generate
            except ModuleNotFoundError as error:  # pragma: no cover - optional VLM dependency
                raise RuntimeError("VLM generation requires mlx-vlm") from error
            result = generate(
                self.model,
                self.processor,
                prompt,
                max_tokens=max_tokens,
                verbose=False,
                **kv_kwargs,
            )
            return result.text
        try:
            from mlx_lm import generate
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("streaming generation requires mlx-lm") from error
        return generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            verbose=False,
            **kv_kwargs,
        )


class Qwen3MoeAdapter:
    """Build a Qwen3 shell without ever creating a full expert tensor bank."""

    def probe(self, config: dict[str, Any]) -> bool:
        return config.get("model_type") == "qwen3_moe"

    def load_shell(
        self,
        manifest: ModelManifest,
        *,
        resident_budget_bytes: int | None = None,
        auto_resident_budget: bool = False,
        memory_config: MemoryBudgetConfig | None = None,
        kv_cache_config: KvCacheConfig | None = None,
        predictor: TransitionPredictor | None = None,
        predictive_config: PredictivePrefetchConfig | None = None,
        prefill_strategy: str = "expert_major",
        prefill_order: PrefillOrder = "resident_first",
        io_workers: int = 0,
        prefetch_depth: int = 1,
        async_gpu: bool = False,
        startup_io_probe: str | float = "auto",
        warmup: str = "auto",
        warmup_timeout_seconds: float = 300.0,
    ) -> StreamingEngine:
        """Load the shell, plan M7 memory, then install the streamed MoE path."""

        if prefill_strategy not in {"token_major", "expert_major"}:
            raise ValueError(f"unsupported prefill strategy {prefill_strategy!r}")
        if prefill_order not in PREFILL_ORDERS:
            raise ValueError(f"unsupported prefill order {prefill_order!r}")

        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.models import qwen3_moe
            from mlx_lm.utils import load_tokenizer
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("the Qwen3 streaming adapter requires mlx and mlx-lm") from error

        config = _load_config(manifest.source_model_path)
        if not self.probe(config):
            raise ValueError(f"unsupported model type {config.get('model_type')!r}")
        model = _build_model_without_expert_bank(qwen3_moe, nn, config)
        runtime: NoCacheExpertRuntime | CachedExpertRuntime | None = None
        try:
            weights = load_nonexpert_weights(manifest)
            _quantize_shell(model, nn, config, weights)
            model.load_weights(list(weights.items()), strict=True)
            del weights
            model.eval()
            mx.eval(model.parameters())

            # Measure the actual quantized non-expert shell before allocating
            # any resident expert arrays.  This is deliberately not a model
            # config estimate: MLX layout and quantization determine the live
            # Unified Memory footprint.
            shell_bytes = int(mx.get_active_memory())
            startup = prepare_streaming_runtime(
                manifest,
                shell_bytes=shell_bytes,
                model_config=config,
                resident_budget_bytes=resident_budget_bytes,
                auto_resident_budget=auto_resident_budget,
                memory_config=memory_config,
                kv_cache_config=kv_cache_config,
                runtime_options={
                    "io_workers": io_workers,
                    "prefetch_depth": prefetch_depth,
                    "async_gpu": async_gpu,
                    "predictor": predictor,
                    "predictive_config": predictive_config,
                },
                startup_io_probe=startup_io_probe,
                warmup=warmup,
                warmup_timeout_seconds=warmup_timeout_seconds,
            )
            memory_manager = startup.memory_manager
            kv_cache = startup.kv_cache
            memory_budget = startup.memory_budget
            runtime = startup.runtime
            replaced = self.replace_moe_blocks(
                model,
                runtime,
                prefill_strategy=prefill_strategy,
                prefill_order=prefill_order,
            )
            if replaced != manifest.num_layers:
                raise ValueError(f"replaced {replaced} MoE blocks; expected {manifest.num_layers}")
            model.eval()
            tokenizer = load_tokenizer(manifest.source_model_path)
        except BaseException:
            if runtime is not None:
                runtime.close()
            raise
        return StreamingEngine(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            memory_manager=memory_manager,
            memory_budget=memory_budget,
            kv_cache=kv_cache,
            startup_decision=startup.decision,
        )

    def replace_moe_blocks(
        self,
        model: Any,
        runtime: NoCacheExpertRuntime | CachedExpertRuntime,
        *,
        prefill_strategy: str = "expert_major",
        prefill_order: PrefillOrder = "resident_first",
    ) -> int:
        """Reuse each original router gate and replace only its expert execution path."""

        replaced = 0
        for layer_id, layer in enumerate(model.layers):
            block = getattr(layer, "mlp", None)
            if not _is_qwen3_sparse_block(block):
                continue
            layer.mlp = StreamingQwen3MoeBlock(
                block.gate,
                block,
                runtime,
                layer_id,
                prefill_strategy=prefill_strategy,
                prefill_order=prefill_order,
            )
            replaced += 1
        return replaced


def load_qwen3_moe_streaming(
    manifest_path: str | Path,
    *,
    resident_budget_bytes: int | None = None,
    auto_resident_budget: bool = False,
    memory_config: MemoryBudgetConfig | None = None,
    kv_cache_config: KvCacheConfig | None = None,
    predictor: TransitionPredictor | None = None,
    predictive_config: PredictivePrefetchConfig | None = None,
    prefill_strategy: str = "expert_major",
    prefill_order: PrefillOrder = "resident_first",
    io_workers: int = 0,
    prefetch_depth: int = 1,
    async_gpu: bool = False,
    startup_io_probe: str | float = "auto",
    warmup: str = "auto",
    warmup_timeout_seconds: float = 300.0,
) -> StreamingEngine:
    """Load an M2 manifest into the exact M3–M7 streaming engine."""

    return Qwen3MoeAdapter().load_shell(
        load_manifest(Path(manifest_path)),
        resident_budget_bytes=resident_budget_bytes,
        auto_resident_budget=auto_resident_budget,
        memory_config=memory_config,
        kv_cache_config=kv_cache_config,
        predictor=predictor,
        predictive_config=predictive_config,
        prefill_strategy=prefill_strategy,
        prefill_order=prefill_order,
        io_workers=io_workers,
        prefetch_depth=prefetch_depth,
        async_gpu=async_gpu,
        startup_io_probe=startup_io_probe,
        warmup=warmup,
        warmup_timeout_seconds=warmup_timeout_seconds,
    )


def _build_model_without_expert_bank(qwen3_moe: Any, nn: Any, config: dict[str, Any]) -> Any:
    """Instantiate upstream Qwen3 classes while preventing SwitchGLU allocation."""

    original_switch_glu = qwen3_moe.SwitchGLU

    class NoWeightSwitchGLU(nn.Module):
        def __init__(self, *_: Any, **__: Any) -> None:
            super().__init__()

        def __call__(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("an un-replaced no-weight SwitchGLU was executed")

    qwen3_moe.SwitchGLU = NoWeightSwitchGLU
    try:
        model_args = qwen3_moe.ModelArgs.from_dict(config)
        return qwen3_moe.Model(model_args)
    finally:
        qwen3_moe.SwitchGLU = original_switch_glu


class StreamingQwen3MoeBlock(nn.Module):
    """Router-identical Qwen3 sparse block delegating each expert to the runtime."""

    def __init__(
        self,
        gate: Any,
        original_block: Any,
        runtime: NoCacheExpertRuntime | CachedExpertRuntime,
        layer_id: int,
        *,
        prefill_strategy: str,
        prefill_order: PrefillOrder,
    ):
        super().__init__()
        self.gate = gate
        self.num_experts = int(original_block.num_experts)
        self.top_k = int(original_block.top_k)
        self.norm_topk_prob = bool(original_block.norm_topk_prob)
        self.switch_mlp = StreamingSwitchGLU(
            runtime,
            layer_id,
            prefill_strategy=prefill_strategy,
            prefill_order=prefill_order,
        )

    def __call__(self, x: Any) -> Any:
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("streaming Qwen3 execution requires MLX") from error

        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class StreamingSwitchGLU(nn.Module):
    """Exact SwitchGLU with token-major decode and M5 expert-major prefill."""

    def __init__(
        self,
        runtime: NoCacheExpertRuntime | CachedExpertRuntime,
        layer_id: int,
        *,
        prefill_strategy: str = "expert_major",
        prefill_order: PrefillOrder = "resident_first",
    ) -> None:
        super().__init__()
        # The runtime owns no model parameters; do not make it part of MLX's
        # module tree or serializer traversal.
        object.__setattr__(self, "_runtime", runtime)
        self._layer_id = layer_id
        self._prefill_strategy = prefill_strategy
        self._prefill_order = prefill_order

    def __call__(self, x: Any, indices: Any) -> Any:
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("streaming SwitchGLU requires MLX") from error

        if x.ndim != 3 or indices.ndim != 3 or x.shape[:2] != indices.shape[:2]:
            raise ValueError(
                "M3 StreamingSwitchGLU expects x=[batch,tokens,hidden] and matching indices"
            )
        flat_x = x.reshape((-1, x.shape[-1]))
        flat_indices = indices.reshape((-1, indices.shape[-1]))
        mx.eval(flat_indices)
        expert_rows = flat_indices.tolist()
        self._runtime.record_routes(self._layer_id, expert_rows)
        try:
            if self._prefill_strategy == "expert_major" and len(expert_rows) > 1:
                output = self._expert_major_prefill(flat_x, expert_rows, x.shape[-1], mx)
            else:
                output = self._token_major(flat_x, expert_rows, x.shape[-1], mx)
            output = output.reshape((*indices.shape, x.shape[-1]))
            return self._runtime.synchronize_batch(output)
        except BaseException:
            self._runtime.abort_batch()
            raise

    def _token_major(
        self, flat_x: Any, expert_rows: list[list[int]], hidden_size: int, mx: Any
    ) -> Any:
        token_outputs = []
        for token_index, expert_ids in enumerate(expert_rows):
            self._schedule_prefetch(expert_ids[: self._runtime.prefetch_depth])
            route_outputs = [
                self._execute_token_route(token_index, route_index, int(expert), expert_ids, flat_x)
                for route_index, expert in enumerate(expert_ids)
            ]
            token_outputs.append(mx.stack(route_outputs, axis=0))
        return mx.stack(token_outputs, axis=0).reshape(
            (len(expert_rows), len(expert_rows[0]), hidden_size)
        )

    def _expert_major_prefill(
        self, flat_x: Any, expert_rows: list[list[int]], hidden_size: int, mx: Any
    ) -> Any:
        """Execute each routed expert once for its full selected-token group."""

        top_k = len(expert_rows[0])
        groups: dict[int, list[tuple[int, int]]] = {}
        for token_index, expert_ids in enumerate(expert_rows):
            if len(expert_ids) != top_k:
                raise ValueError("router rows must have a consistent top-k width")
            for route_index, expert in enumerate(expert_ids):
                groups.setdefault(int(expert), []).append((token_index, route_index))
        expert_order = self._runtime.order_experts(
            self._layer_id, list(groups), self._prefill_order
        )
        self._runtime.record_prefill_layer(
            self._layer_id,
            token_count=len(expert_rows),
            route_count=len(expert_rows) * top_k,
            unique_experts=len(groups),
            order=self._prefill_order,
        )
        route_outputs: list[list[Any | None]] = [[None] * top_k for _ in range(len(expert_rows))]
        self._schedule_prefetch(expert_order[: self._runtime.prefetch_depth])
        for expert_index, expert in enumerate(expert_order):
            next_index = expert_index + self._runtime.prefetch_depth
            if next_index < len(expert_order):
                self._schedule_prefetch([expert_order[next_index]])
            routes = groups[expert]
            token_indices = mx.array([token for token, _ in routes], dtype=mx.uint32)
            values = self._runtime.execute_group(self._layer_id, expert, flat_x[token_indices])
            for group_index, (token_index, route_index) in enumerate(routes):
                route_outputs[token_index][route_index] = values[group_index]
        if any(value is None for row in route_outputs for value in row):
            raise RuntimeError("expert-major scheduler left a routed output unfilled")
        return mx.stack(
            [mx.stack([value for value in row if value is not None]) for row in route_outputs]
        ).reshape((len(expert_rows), top_k, hidden_size))

    def _execute_token_route(
        self,
        token_index: int,
        route_index: int,
        expert: int,
        expert_ids: list[int],
        flat_x: Any,
    ) -> Any:
        next_index = route_index + self._runtime.prefetch_depth
        if next_index < len(expert_ids):
            self._schedule_prefetch([expert_ids[next_index]])
        return self._runtime.execute(self._layer_id, expert, flat_x[token_index])

    def _schedule_prefetch(self, experts: list[int]) -> None:
        if self._runtime.prefetch_depth == 0:
            return
        for expert in experts:
            self._runtime.prefetch(self._layer_id, int(expert))


def _is_qwen3_sparse_block(block: Any) -> bool:
    return block is not None and all(
        hasattr(block, attribute)
        for attribute in ("gate", "switch_mlp", "num_experts", "top_k", "norm_topk_prob")
    )


def _quantize_shell(model: Any, nn: Any, config: dict[str, Any], weights: dict[str, Any]) -> None:
    quantization = config.get("quantization")
    if quantization is None:
        return
    mode = quantization.get("mode", "affine")
    if mode != "affine":
        raise ValueError(f"M3 supports only affine MLX quantization, got {mode!r}")
    group_size = int(quantization["group_size"])
    bits = int(quantization["bits"])

    def class_predicate(path: str, module: Any) -> bool | dict[str, int]:
        per_layer = quantization.get(path)
        if per_layer is None and path.startswith("language_model."):
            per_layer = quantization.get(path.removeprefix("language_model."))
        if per_layer is not None:
            return per_layer
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=class_predicate,
    )


def _load_config(model_path: Path) -> dict[str, Any]:
    try:
        value = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model config from {model_path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"model config must be an object: {model_path}")
    return value
