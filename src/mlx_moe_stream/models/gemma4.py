"""Text-only Gemma 4 MoE adapter with SSD-streamed GeGLU experts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.nn as nn

from ..manifest import ModelManifest, load_manifest
from ..memory import MemoryBudgetConfig, MemoryBudgetManager
from ..prefetch import PredictivePrefetchConfig, TransitionPredictor
from ..runtime import PREFILL_ORDERS, CachedExpertRuntime, NoCacheExpertRuntime, PrefillOrder
from ..storage import load_nonexpert_weights
from .qwen3_moe import (
    StreamingEngine,
    StreamingSwitchGLU,
    _load_config,
    _quantize_shell,
)


class Gemma4Adapter:
    """Build Gemma 4's text shell while retaining only routed experts on demand."""

    def probe(self, config: dict[str, Any]) -> bool:
        return config.get("model_type") == "gemma4"

    def load_shell(
        self,
        manifest: ModelManifest,
        *,
        resident_budget_bytes: int | None = None,
        auto_resident_budget: bool = False,
        memory_config: MemoryBudgetConfig | None = None,
        predictor: TransitionPredictor | None = None,
        predictive_config: PredictivePrefetchConfig | None = None,
        prefill_strategy: str = "expert_major",
        prefill_order: PrefillOrder = "resident_first",
        io_workers: int = 0,
        prefetch_depth: int = 1,
        async_gpu: bool = False,
        vision: bool = False,
    ) -> StreamingEngine:
        if prefill_strategy not in {"token_major", "expert_major"}:
            raise ValueError(f"unsupported prefill strategy {prefill_strategy!r}")
        if prefill_order not in PREFILL_ORDERS:
            raise ValueError(f"unsupported prefill order {prefill_order!r}")
        if manifest.model_type != "gemma4":
            raise ValueError(f"Gemma 4 adapter cannot load {manifest.model_type!r}")
        if vision:
            from .vlm import load_vlm_streaming

            return load_vlm_streaming(
                manifest,
                resident_budget_bytes=resident_budget_bytes,
                auto_resident_budget=auto_resident_budget,
                memory_config=memory_config,
                predictor=predictor,
                predictive_config=predictive_config,
                prefill_strategy=prefill_strategy,
                prefill_order=prefill_order,
                io_workers=io_workers,
                prefetch_depth=prefetch_depth,
                async_gpu=async_gpu,
            )

        try:
            import mlx.core as mx
            import mlx.nn as mlx_nn
            from mlx_lm.models import gemma4_text
            from mlx_lm.utils import load_tokenizer
        except ModuleNotFoundError as error:  # pragma: no cover - normal dependency path
            raise RuntimeError("the Gemma 4 streaming adapter requires mlx and mlx-lm") from error

        source_config = _load_config(manifest.source_model_path)
        if not self.probe(source_config):
            raise ValueError(f"unsupported model type {source_config.get('model_type')!r}")
        text_config = source_config.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("Gemma 4 config requires a text_config object")
        model = _build_model_without_expert_bank(gemma4_text, mlx_nn, text_config)
        runtime: NoCacheExpertRuntime | CachedExpertRuntime | None = None
        try:
            weights = load_nonexpert_weights(
                manifest, include=lambda name: name.startswith("language_model.")
            )
            weights = _strip_language_model_prefix(weights)
            _quantize_shell(model, mlx_nn, _text_quantization_config(source_config), weights)
            model.load_weights(list(weights.items()), strict=True)
            del weights
            model.eval()
            mx.eval(model.parameters())

            memory_manager = MemoryBudgetManager(memory_config)
            memory_budget = memory_manager.plan(
                shell_bytes=int(mx.get_active_memory()),
                requested_expert_budget_bytes=resident_budget_bytes,
                auto_enabled=auto_resident_budget,
                minimum_expert_bytes=max(
                    bundle.total_bytes for bundle in manifest.expert_bundles.values()
                ),
            )
            runtime_options = {
                "expert_activation": "geglu",
                "io_workers": io_workers,
                "prefetch_depth": prefetch_depth,
                "async_gpu": async_gpu,
                "memory_manager": memory_manager,
                "predictor": predictor,
                "predictive_config": predictive_config,
            }
            if memory_budget.expert_budget_bytes is None:
                runtime = NoCacheExpertRuntime(manifest, **runtime_options)
            else:
                runtime = CachedExpertRuntime(
                    manifest,
                    capacity_bytes=memory_budget.expert_budget_bytes,
                    **runtime_options,
                )
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
            if not _is_gemma4_moe_layer(layer):
                continue
            layer.experts = StreamingGemmaExperts(
                runtime,
                layer_id,
                prefill_strategy=prefill_strategy,
                prefill_order=prefill_order,
            )
            replaced += 1
        return replaced


def load_gemma4_streaming(manifest_path: str | Path, **kwargs: Any) -> StreamingEngine:
    """Load a prepared Gemma 4 text manifest into the streaming engine."""

    return Gemma4Adapter().load_shell(load_manifest(Path(manifest_path)), **kwargs)


def _build_model_without_expert_bank(
    gemma4_text: Any, mlx_nn: Any, text_config: dict[str, Any]
) -> Any:
    original_experts = gemma4_text.Experts

    class NoWeightExperts(mlx_nn.Module):
        def __init__(self, *_: Any, **__: Any) -> None:
            super().__init__()

        def __call__(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("an un-replaced no-weight Gemma 4 Experts module was executed")

    gemma4_text.Experts = NoWeightExperts
    try:
        return gemma4_text.Model(gemma4_text.ModelArgs.from_dict(text_config))
    finally:
        gemma4_text.Experts = original_experts


class StreamingGemmaExperts(nn.Module):
    """Gemma's Router-compatible expert module backed by the common runtime."""

    def __init__(
        self,
        runtime: NoCacheExpertRuntime | CachedExpertRuntime,
        layer_id: int,
        *,
        prefill_strategy: str,
        prefill_order: PrefillOrder,
    ) -> None:
        super().__init__()
        self.switch_glu = StreamingSwitchGLU(
            runtime,
            layer_id,
            prefill_strategy=prefill_strategy,
            prefill_order=prefill_order,
        )

    def __call__(self, x: Any, top_k_indices: Any, top_k_weights: Any) -> Any:
        y = self.switch_glu(x, top_k_indices)
        return (top_k_weights[..., None] * y).sum(axis=-2)


def _is_gemma4_moe_layer(layer: Any) -> bool:
    return bool(getattr(layer, "enable_moe", False)) and all(
        hasattr(layer, attribute) for attribute in ("router", "experts")
    )


def _strip_language_model_prefix(weights: dict[str, Any]) -> dict[str, Any]:
    prefix = "language_model."
    return {name.removeprefix(prefix): value for name, value in weights.items()}


def _text_quantization_config(source_config: dict[str, Any]) -> dict[str, Any]:
    """Map multimodal checkpoint quantization paths onto ``gemma4_text.Model``."""

    quantization = source_config.get("quantization")
    if not isinstance(quantization, dict):
        return {}
    prefix = "language_model."
    adjusted = {
        (name.removeprefix(prefix) if name.startswith(prefix) else name): value
        for name, value in quantization.items()
    }
    return {"quantization": adjusted}
