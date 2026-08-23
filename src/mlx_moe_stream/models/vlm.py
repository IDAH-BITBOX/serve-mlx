"""M12 MLX-VLM shells with the exact SSD-streamed MoE runtime retained."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.nn as nn

from ..manifest import ModelManifest
from ..memory import MemoryBudgetConfig, MemoryBudgetManager
from ..prefetch import PredictivePrefetchConfig, TransitionPredictor
from ..runtime import PREFILL_ORDERS, CachedExpertRuntime, NoCacheExpertRuntime, PrefillOrder
from ..storage import load_nonexpert_weights
from .gemma4 import StreamingGemmaExperts
from .qwen3_5_moe import StreamingQwen35VlmMoeBlock
from .qwen3_moe import StreamingEngine, _load_config, _quantize_shell


def load_vlm_streaming(
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
) -> StreamingEngine:
    """Load a VLM shell while leaving every routed expert bundle on SSD.

    MLX-VLM normally creates and fills the complete expert banks.  M12 patches
    only their construction during model creation, then installs the same
    M3–M10 streaming expert blocks used by text-only serving.  Vision tower,
    projector, router, shared expert, and all dense text weights remain normal
    MLX modules in Unified Memory.
    """

    if manifest.model_type not in {"qwen3_5_moe", "gemma4"}:
        raise ValueError(f"M12 does not support VLM loading for {manifest.model_type!r}")
    if prefill_strategy not in {"token_major", "expert_major"}:
        raise ValueError(f"unsupported prefill strategy {prefill_strategy!r}")
    if prefill_order not in PREFILL_ORDERS:
        raise ValueError(f"unsupported prefill order {prefill_order!r}")
    try:
        import mlx.core as mx
        import mlx.nn as mlx_nn
        from mlx_vlm import utils as vlm_utils
    except ModuleNotFoundError as error:  # pragma: no cover - optional VLM dependency
        raise RuntimeError(
            "M12 vision serving requires `pip install mlx-moe-stream[vlm]`"
        ) from error

    source_config = _load_config(manifest.source_model_path)
    model_class, _ = vlm_utils.get_model_and_args(source_config)
    model_config = model_class.ModelConfig.from_dict(source_config)
    model_config = vlm_utils.update_module_configs(
        model_config,
        model_class,
        source_config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    apply_defaults = getattr(vlm_utils, "apply_generation_config_defaults", None)
    if callable(apply_defaults):
        model_config = apply_defaults(model_config, source_config)
    model = _build_vlm_model_without_expert_bank(manifest.model_type, model_class, model_config)
    runtime: NoCacheExpertRuntime | CachedExpertRuntime | None = None
    try:
        weights = load_nonexpert_weights(manifest)
        weights = _sanitize_vlm_weights(model, model_class, model_config, weights, vlm_utils)
        _quantize_shell(model, mlx_nn, source_config, weights)
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
            "io_workers": io_workers,
            "prefetch_depth": prefetch_depth,
            "async_gpu": async_gpu,
            "memory_manager": memory_manager,
            "predictor": predictor,
            "predictive_config": predictive_config,
        }
        if manifest.model_type == "qwen3_5_moe":
            runtime_options["expert_activation"] = "swiglu"
        else:
            runtime_options["expert_activation"] = "geglu"
        if memory_budget.expert_budget_bytes is None:
            runtime = NoCacheExpertRuntime(manifest, **runtime_options)
        else:
            runtime = CachedExpertRuntime(
                manifest, capacity_bytes=memory_budget.expert_budget_bytes, **runtime_options
            )
        replaced = _replace_vlm_moe_blocks(
            model,
            manifest,
            runtime,
            prefill_strategy=prefill_strategy,
            prefill_order=prefill_order,
        )
        if replaced != manifest.num_layers:
            raise ValueError(f"replaced {replaced} MoE blocks; expected {manifest.num_layers}")
        processor = _load_vlm_processor(manifest.model_type, manifest.source_model_path)
    except BaseException:
        if runtime is not None:
            runtime.close()
        raise
    return StreamingEngine(
        model=model,
        tokenizer=processor.tokenizer,
        runtime=runtime,
        memory_manager=memory_manager,
        memory_budget=memory_budget,
        processor=processor,
    )


def _build_vlm_model_without_expert_bank(
    model_type: str, model_class: Any, model_config: Any
) -> Any:
    """Patch the VLM family expert constructor only during shell construction."""

    if model_type == "qwen3_5_moe":
        from mlx_vlm.models.qwen3_5_moe import language

        original = language.SwitchGLU

        class NoWeightSwitchGLU(nn.Module):
            def __init__(self, *_: Any, **__: Any) -> None:
                super().__init__()

            def __call__(self, *_: Any, **__: Any) -> Any:
                raise RuntimeError("an unreplaced M12 Qwen routed expert was executed")

        language.SwitchGLU = NoWeightSwitchGLU
        try:
            return model_class.Model(model_config)
        finally:
            language.SwitchGLU = original

    from mlx_vlm.models.gemma4 import language

    original = language.Experts

    class NoWeightExperts(nn.Module):
        def __init__(self, *_: Any, **__: Any) -> None:
            super().__init__()

        def __call__(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("an unreplaced M12 Gemma routed expert was executed")

    language.Experts = NoWeightExperts
    try:
        return model_class.Model(model_config)
    finally:
        language.Experts = original


def _sanitize_vlm_weights(
    model: Any, model_class: Any, model_config: Any, weights: dict[str, Any], vlm_utils: Any
) -> dict[str, Any]:
    """Match MLX-VLM's normal sanitize sequence after selective reads."""

    weights = vlm_utils.sanitize_weights(model, weights)
    if hasattr(model_class, "VisionModel") and hasattr(model_config, "vision_config"):
        weights = vlm_utils.sanitize_weights(
            model_class.VisionModel, weights, model_config.vision_config
        )
    if hasattr(model_class, "LanguageModel") and hasattr(model_config, "text_config"):
        weights = vlm_utils.sanitize_weights(
            model_class.LanguageModel, weights, model_config.text_config
        )
    return weights


def _replace_vlm_moe_blocks(
    model: Any,
    manifest: ModelManifest,
    runtime: NoCacheExpertRuntime | CachedExpertRuntime,
    *,
    prefill_strategy: str,
    prefill_order: PrefillOrder,
) -> int:
    layers = model.language_model.model.layers
    replaced = 0
    for layer_id, layer in enumerate(layers):
        if manifest.model_type == "qwen3_5_moe":
            block = getattr(layer, "mlp", None)
            if not _is_qwen35_vlm_sparse_block(block):
                continue
            layer.mlp = StreamingQwen35VlmMoeBlock(
                block,
                runtime,
                layer_id,
                prefill_strategy=prefill_strategy,
                prefill_order=prefill_order,
            )
        elif bool(getattr(layer, "enable_moe", False)) and hasattr(layer, "experts"):
            layer.experts = StreamingGemmaExperts(
                runtime,
                layer_id,
                prefill_strategy=prefill_strategy,
                prefill_order=prefill_order,
            )
        else:
            continue
        replaced += 1
    return replaced


def _is_qwen35_vlm_sparse_block(block: Any) -> bool:
    return block is not None and all(
        hasattr(block, attribute)
        for attribute in (
            "gate",
            "switch_mlp",
            "num_experts",
            "top_k",
            "shared_expert",
            "shared_expert_gate",
        )
    )


def _load_vlm_processor(model_type: str, model_path: Path) -> Any:
    if model_type == "qwen3_5_moe":
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        processor = Qwen3VLProcessor.from_pretrained(model_path)
    else:
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor

        processor = Gemma4Processor.from_pretrained(model_path)
    # Direct family processors avoid Transformers' torch-backed AutoProcessor.
    # Reproduce the streamer setup normally performed by mlx_vlm.load_processor.
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    tokenizer = processor.tokenizer
    detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
    processor.detokenizer = detokenizer_class(tokenizer)
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    tokenizer.stopping_criteria = StoppingCriteria(eos_token_ids, tokenizer)
    return processor
