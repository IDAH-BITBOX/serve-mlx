from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.manifest import ExpertBundleSpec, load_manifest
from mlx_moe_stream.storage import (
    SafetensorsExpertStore,
    StorageReadError,
    build_qwen3_moe_manifest,
    build_streaming_manifest,
    load_nonexpert_weights,
)


def _write_model(
    path: Path, *, split: bool = False, num_experts: int = 4
) -> dict[str, np.ndarray]:
    path.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (path / "config.json").write_text(json.dumps(config))
    tensors: dict[str, np.ndarray] = {}
    for projection, shape in (("gate_proj", (2, 3)), ("up_proj", (2, 3)), ("down_proj", (3, 2))):
        for field, dtype in (("weight", np.uint32), ("scales", np.float16), ("biases", np.float16)):
            values = np.arange(num_experts * np.prod(shape), dtype=dtype).reshape(
                (num_experts, *shape)
            )
            if split:
                for expert in range(num_experts):
                    name = f"model.layers.0.mlp.experts.{expert}.{projection}.{field}"
                    tensors[name] = values[expert]
            else:
                name = f"model.layers.0.mlp.switch_mlp.{projection}.{field}"
                tensors[name] = values
    save_file(tensors, path / "model.safetensors")
    return tensors


def _write_multimodal_family_model(path: Path, model_type: str) -> dict[str, np.ndarray]:
    """Create one text subtree plus ignored multimodal tensors for M8.5 tests."""

    path.mkdir()
    if model_type == "qwen3_5_moe":
        text_config = {
            "num_hidden_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
        expert_prefix = "language_model.model.layers.0.mlp.switch_mlp"
    elif model_type == "gemma4":
        text_config = {
            "num_hidden_layers": 1,
            "num_experts": 4,
            "top_k_experts": 2,
        }
        expert_prefix = "language_model.model.layers.0.experts.switch_glu"
    else:  # pragma: no cover - test helper guard
        raise AssertionError(model_type)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "text_config": text_config,
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            }
        )
    )
    tensors: dict[str, np.ndarray] = {
        "language_model.model.embed_tokens.weight": np.arange(12, dtype=np.float16).reshape(3, 4),
        "vision_tower.ignored.weight": np.ones((2, 2), dtype=np.float16),
    }
    for projection, shape in (
        ("gate_proj", (2, 3)),
        ("up_proj", (2, 3)),
        ("down_proj", (3, 2)),
    ):
        for field, dtype in (("weight", np.uint32), ("scales", np.float16), ("biases", np.float16)):
            tensors[f"{expert_prefix}.{projection}.{field}"] = np.arange(
                4 * np.prod(shape), dtype=dtype
            ).reshape((4, *shape))
    save_file(tensors, path / "model.safetensors")
    return tensors


@pytest.mark.parametrize("split", [False, True])
def test_manifest_and_pread_match_safetensors_slice_bitwise(tmp_path: Path, split: bool):
    model_path = tmp_path / ("split" if split else "leading")
    _write_model(model_path, split=split)
    manifest = build_qwen3_moe_manifest(model_path)
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    loaded = load_manifest(manifest_path)
    bundle = loaded.expert_bundles[ExpertKey(0, 2)]

    with SafetensorsExpertStore() as store:
        actual = store.read_bundle(bundle)
        metrics = store.metrics()

    assert metrics.bytes_read == bundle.total_bytes
    assert metrics.read_count == 9
    with safe_open(model_path / "model.safetensors", framework="np") as source:
        for tensor in bundle.tensors:
            if split:
                expected = source.get_tensor(tensor.tensor_name)
            else:
                expected = source.get_slice(tensor.tensor_name)[2]
            assert actual[tensor.role] == expected.tobytes()


def test_exact_read_is_smaller_than_full_leading_axis_tensor(tmp_path: Path):
    model_path = tmp_path / "model"
    tensors = _write_model(model_path)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle = manifest.expert_bundles[ExpertKey(0, 3)]
    with SafetensorsExpertStore() as store:
        store.read_bundle(bundle)
        metrics = store.metrics()

    full_expert_tensor_bytes = sum(
        array.nbytes for name, array in tensors.items() if "switch_mlp" in name
    )
    assert metrics.bytes_read == bundle.total_bytes
    assert metrics.bytes_read * 4 == full_expert_tensor_bytes


def test_random_one_hundred_experts_match_source_slices(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_model(model_path, num_experts=128)
    manifest = build_qwen3_moe_manifest(model_path)
    selected = random.Random(0).sample(range(128), 100)
    with safe_open(model_path / "model.safetensors", framework="np") as source:
        with SafetensorsExpertStore() as store:
            for expert in selected:
                bundle = manifest.expert_bundles[ExpertKey(0, expert)]
                actual = store.read_bundle(bundle)
                for tensor in bundle.tensors:
                    expected = source.get_slice(tensor.tensor_name)[expert]
                    assert actual[tensor.role] == expected.tobytes()
            metrics = store.metrics()
    assert metrics.bytes_read == sum(
        manifest.expert_bundles[ExpertKey(0, expert)].total_bytes for expert in selected
    )


def test_corrupt_span_and_missing_required_weight_fail_fast(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_model(model_path)
    manifest = build_qwen3_moe_manifest(model_path)
    bundle = manifest.expert_bundles[ExpertKey(0, 0)]
    corrupted_span = replace(bundle.tensors[0], offset=bundle.tensors[0].file.stat().st_size + 1)
    corrupted_bundle = ExpertBundleSpec(
        key=bundle.key,
        tensors=(corrupted_span, *bundle.tensors[1:]),
        total_bytes=bundle.total_bytes,
        quantization=bundle.quantization,
    )
    with SafetensorsExpertStore() as store, pytest.raises(StorageReadError, match="exceeds"):
        store.read_bundle(corrupted_bundle)

    missing = tmp_path / "missing"
    _write_model(missing)
    with safe_open(missing / "model.safetensors", framework="np") as source:
        subset = {
            name: source.get_tensor(name)
            for name in source.keys()
            if not name.endswith("up_proj.weight")
        }
    save_file(subset, missing / "replaced.safetensors")
    (missing / "model.safetensors").unlink()
    (missing / "replaced.safetensors").rename(missing / "model.safetensors")
    with pytest.raises(ValueError, match="missing routed expert weights"):
        build_qwen3_moe_manifest(missing)


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "gemma4"])
def test_multimodal_moe_families_keep_only_text_experts_on_disk(
    tmp_path: Path, model_type: str
):
    model_path = tmp_path / model_type
    tensors = _write_multimodal_family_model(model_path, model_type)
    manifest = build_streaming_manifest(model_path)

    assert manifest.model_type == model_type
    assert manifest.num_layers == 1
    assert manifest.num_experts == 4
    assert manifest.experts_per_token == 2
    bundle = manifest.expert_bundles[ExpertKey(0, 3)]
    with SafetensorsExpertStore() as store:
        actual = store.read_bundle(bundle)
    for tensor in bundle.tensors:
        expected = tensors[tensor.tensor_name][3]
        assert actual[tensor.role] == expected.tobytes()

    shell = load_nonexpert_weights(
        manifest, include=lambda name: name.startswith("language_model.")
    )
    assert set(shell) == {"language_model.model.embed_tokens.weight"}
