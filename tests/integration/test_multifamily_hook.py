"""Small end-to-end shell tests for the M8.5 family adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

from mlx_moe_stream.models import load_streaming_model
from mlx_moe_stream.storage import build_streaming_manifest


def _small_qwen_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "vocab_size": 64,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "full_attention_interval": 1,
        },
    }


def _small_gemma_config() -> dict[str, Any]:
    return {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "num_kv_shared_layers": 0,
            "num_experts": 2,
            "top_k_experts": 1,
            "moe_intermediate_size": 16,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "enable_moe_block": True,
            "tie_word_embeddings": True,
            "layer_types": ["full_attention"],
            "hidden_size_per_layer_input": 0,
        },
    }


def _write_checkpoint(path: Path, config: dict[str, Any], model: Any, *, prefix: str) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        f"{prefix}{name}": np.asarray(value)
        for name, value in tree_flatten(model.parameters())
    }
    save_file(tensors, path / "model.safetensors")


@pytest.mark.parametrize("family", ["qwen3_5_moe", "gemma4"])
def test_text_family_adapter_matches_tiny_upstream_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
):
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda _: object())
    if family == "qwen3_5_moe":
        from mlx_lm.models import qwen3_5_moe

        config = _small_qwen_config()
        reference = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(config))
        prefix = ""
    else:
        from mlx_lm.models import gemma4_text

        config = _small_gemma_config()
        reference = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(config["text_config"]))
        prefix = "language_model."
    reference.eval()
    model_path = tmp_path / family
    _write_checkpoint(model_path, config, reference, prefix=prefix)

    manifest_path = tmp_path / "manifest.json"
    build_streaming_manifest(model_path).write(manifest_path)
    engine = load_streaming_model(manifest_path)
    try:
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        expected = reference(tokens)
        actual = engine.model(tokens)
        mx.eval(expected, actual)
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
    finally:
        engine.close()
