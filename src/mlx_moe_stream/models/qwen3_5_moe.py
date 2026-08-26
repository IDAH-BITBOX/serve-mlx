"""Text-only Qwen3.5-MoE adapter with streamed routed experts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.nn as nn

from ..kv_cache import KvCacheConfig
from ..manifest import ModelManifest, load_manifest
from ..memory import MemoryBudgetConfig
from ..prefetch import PredictivePrefetchConfig, TransitionPredictor
from ..runtime import PREFILL_ORDERS, CachedExpertRuntime, NoCacheExpertRuntime, PrefillOrder
from ..startup import prepare_streaming_runtime
from ..storage import load_nonexpert_weights
from .qwen3_moe import (
    StreamingEngine,
    StreamingSwitchGLU,
    _load_config,
    _quantize_shell,
)


class Qwen35MoeAdapter:
    """Load Qwen3.5's text model without allocating its 256-expert banks.

    Qwen3.5 checkpoints are multimodal, but this adapter deliberately accepts
    only the ``language_model`` subtree.  Vision inputs are out of M8.5 scope.
    """

    def probe(self, config: dict[str, Any]) -> bool:
        return config.get("model_type") == "qwen3_5_moe"

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
        vision: bool = False,
        startup_io_probe: str | float = "auto",
        warmup: str = "auto",
        warmup_timeout_seconds: float = 300.0,
    ) -> StreamingEngine:
        if prefill_strategy not in {"token_major", "expert_major"}:
            raise ValueError(f"unsupported prefill strategy {prefill_strategy!r}")
        if prefill_order not in PREFILL_ORDERS:
            raise ValueError(f"unsupported prefill order {prefill_order!r}")
        if manifest.model_type != "qwen3_5_moe":
            raise ValueError(f"Qwen3.5 adapter cannot load {manifest.model_type!r}")
        if vision:
            from .vlm import load_vlm_streaming

            return load_vlm_streaming(
                manifest,
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

        try:
            import mlx.core as mx
            import mlx.nn as mlx_nn
            from mlx_lm.models import qwen3_5_moe, qwen3_next
            from mlx_lm.utils import load_tokenizer
        except ModuleNotFoundError as error:  # pragma: no cover - normal dependency path
            raise RuntimeError("the Qwen3.5 streaming adapter requires mlx and mlx-lm") from error

        config = _load_config(manifest.source_model_path)
        if not self.probe(config):
            raise ValueError(f"unsupported model type {config.get('model_type')!r}")
        model = _build_model_without_expert_bank(qwen3_5_moe, qwen3_next, mlx_nn, config)
        runtime: NoCacheExpertRuntime | CachedExpertRuntime | None = None
        try:
            weights = load_nonexpert_weights(
                manifest, include=lambda name: name.startswith("language_model.")
            )
            _quantize_shell(model, mlx_nn, config, weights)
            model.load_weights(list(weights.items()), strict=True)
            del weights
            model.eval()
            mx.eval(model.parameters())

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
                    "expert_activation": "swiglu",
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
        prefill_strategy: str,
        prefill_order: PrefillOrder,
    ) -> int:
        replaced = 0
        for layer_id, layer in enumerate(model.layers):
            block = getattr(layer, "mlp", None)
            if not _is_qwen35_sparse_block(block):
                continue
            layer.mlp = StreamingQwen35MoeBlock(
                block,
                runtime,
                layer_id,
                prefill_strategy=prefill_strategy,
                prefill_order=prefill_order,
            )
            replaced += 1
        return replaced


def load_qwen3_5_moe_streaming(
    manifest_path: str | Path,
    **kwargs: Any,
) -> StreamingEngine:
    """Load a prepared Qwen3.5-MoE text manifest into the streaming engine."""

    return Qwen35MoeAdapter().load_shell(load_manifest(Path(manifest_path)), **kwargs)


def _build_model_without_expert_bank(
    qwen3_5_moe: Any, qwen3_next: Any, mlx_nn: Any, config: dict[str, Any]
) -> Any:
    """Patch the upstream expert constructor only while the shell is built."""

    original_switch_glu = qwen3_next.SwitchGLU

    class NoWeightSwitchGLU(mlx_nn.Module):
        def __init__(self, *_: Any, **__: Any) -> None:
            super().__init__()

        def __call__(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("an un-replaced no-weight Qwen3.5 SwitchGLU was executed")

    qwen3_next.SwitchGLU = NoWeightSwitchGLU
    try:
        return qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(config))
    finally:
        qwen3_next.SwitchGLU = original_switch_glu


class StreamingQwen35MoeBlock(nn.Module):
    """Qwen3.5 router plus its resident shared expert and streamed experts."""

    def __init__(
        self,
        original_block: Any,
        runtime: NoCacheExpertRuntime | CachedExpertRuntime,
        layer_id: int,
        *,
        prefill_strategy: str,
        prefill_order: PrefillOrder,
    ) -> None:
        super().__init__()
        self.gate = original_block.gate
        self.num_experts = int(original_block.num_experts)
        self.top_k = int(original_block.top_k)
        self.norm_topk_prob = bool(original_block.norm_topk_prob)
        self.shared_expert = original_block.shared_expert
        self.shared_expert_gate = original_block.shared_expert_gate
        self.switch_mlp = StreamingSwitchGLU(
            runtime,
            layer_id,
            prefill_strategy=prefill_strategy,
            prefill_order=prefill_order,
        )

    def __call__(self, x: Any) -> Any:
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - normal dependency path
            raise RuntimeError("streaming Qwen3.5 execution requires MLX") from error

        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return routed + shared


class StreamingQwen35VlmMoeBlock(nn.Module):
    """Qwen3.5 VLM's sparse block backed by the common SSD runtime.

    MLX-VLM's block has the same routing math as the text adapter but its
    optional speculative ``target_verify`` hook must not bypass streamed
    experts.  M12 therefore fails explicitly for that unsupported path.
    """

    def __init__(
        self,
        original_block: Any,
        runtime: NoCacheExpertRuntime | CachedExpertRuntime,
        layer_id: int,
        *,
        prefill_strategy: str,
        prefill_order: PrefillOrder,
    ) -> None:
        super().__init__()
        self.gate = original_block.gate
        self.num_experts = int(original_block.num_experts)
        self.top_k = int(original_block.top_k)
        self.shared_expert = original_block.shared_expert
        self.shared_expert_gate = original_block.shared_expert_gate
        self.switch_mlp = StreamingSwitchGLU(
            runtime,
            layer_id,
            prefill_strategy=prefill_strategy,
            prefill_order=prefill_order,
        )

    def __call__(self, x: Any, target_verify: bool = False) -> Any:
        if target_verify:
            raise RuntimeError("M12 does not support Qwen target-verify with streamed experts")
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
            raise RuntimeError("streaming Qwen3.5 VLM execution requires MLX") from error

        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        scores /= mx.sum(scores, axis=-1, keepdims=True)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return routed + shared


def _is_qwen35_sparse_block(block: Any) -> bool:
    return block is not None and all(
        hasattr(block, attribute)
        for attribute in (
            "gate",
            "switch_mlp",
            "num_experts",
            "top_k",
            "norm_topk_prob",
            "shared_expert",
            "shared_expert_gate",
        )
    )
