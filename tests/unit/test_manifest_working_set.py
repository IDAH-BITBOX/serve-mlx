from __future__ import annotations

from pathlib import Path

import pytest
from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.manifest import (
    ExpertBundleSpec,
    ExpertWorkingSet,
    ModelManifest,
    QuantizationSpec,
    TensorSpan,
)


def _bundle(layer: int, expert: int, nbytes: int) -> ExpertBundleSpec:
    """One synthetic single-tensor bundle; the source file need not exist

    because expert_working_set() only does arithmetic over already-loaded
    manifest metadata -- zero additional file I/O.
    """

    span = TensorSpan(
        file=Path("/nonexistent/model.safetensors"),
        tensor_name=f"layer.{layer}.expert.{expert}.weight",
        offset=0,
        nbytes=nbytes,
        shape=(nbytes,),
        dtype="uint8",
        role="weight",
    )
    key = ExpertKey(layer=layer, expert=expert)
    return ExpertBundleSpec(
        key=key, tensors=(span,), total_bytes=nbytes, quantization=QuantizationSpec()
    )


def _manifest(
    num_layers: int, num_experts: int, experts_per_token: int, bundle_bytes
) -> ModelManifest:
    bundles = {}
    for layer in range(num_layers):
        for expert in range(num_experts):
            nbytes = bundle_bytes(layer, expert)
            bundles[ExpertKey(layer, expert)] = _bundle(layer, expert, nbytes)
    return ModelManifest(
        format_version=1,
        model_type="qwen3_moe",
        source_model="synthetic",
        source_model_path=Path("/nonexistent"),
        num_layers=num_layers,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        quantization=QuantizationSpec(),
        non_expert_weight_files=(),
        expert_bundles=bundles,
    )


def test_expert_working_set_matches_hand_computed_totals_for_uniform_bundles():
    # Mirrors the real Qwen3-MoE shape used on the Mac mini: 40 layers x 256
    # experts, uniform 3,342,336-byte bundles, experts_per_token=8.
    manifest = _manifest(40, 256, 8, lambda layer, expert: 3_342_336)

    working_set = manifest.expert_working_set()

    assert isinstance(working_set, ExpertWorkingSet)
    assert working_set.bundle_count == 40 * 256
    assert working_set.total_bytes == 40 * 256 * 3_342_336
    assert working_set.mean_bundle_bytes == 3_342_336
    assert working_set.min_bundle_bytes == 3_342_336
    assert working_set.max_bundle_bytes == 3_342_336
    # per-token full-miss bytes = num_layers * experts_per_token * mean_bundle_bytes
    assert working_set.per_token_full_miss_bytes == 40 * 8 * 3_342_336
    assert working_set.per_token_full_miss_bytes == pytest.approx(1_069_547_520)


def test_expert_working_set_reports_true_min_and_max_for_nonuniform_bundles():
    sizes = {(0, 0): 100, (0, 1): 300, (1, 0): 200, (1, 1): 400}
    manifest = _manifest(2, 2, 1, lambda layer, expert: sizes[(layer, expert)])

    working_set = manifest.expert_working_set()

    assert working_set.bundle_count == 4
    assert working_set.total_bytes == 1_000
    assert working_set.min_bundle_bytes == 100
    assert working_set.max_bundle_bytes == 400
    assert working_set.mean_bundle_bytes == 250
    assert working_set.per_token_full_miss_bytes == 2 * 1 * 250


def test_expert_working_set_requires_no_additional_file_access(monkeypatch):
    """A manifest pointing at nonexistent files must still summarize cleanly."""

    manifest = _manifest(1, 4, 2, lambda layer, expert: 1_000 + expert)

    def _forbidden_stat(self):
        raise AssertionError("expert_working_set() must not touch the filesystem")

    monkeypatch.setattr(Path, "stat", _forbidden_stat)

    working_set = manifest.expert_working_set()
    assert working_set.bundle_count == 4


def test_expert_working_set_rejects_an_empty_manifest():
    manifest = _manifest(0, 0, 1, lambda layer, expert: 0)
    # num_layers/num_experts are 0 here purely to produce zero bundles; this
    # bypasses validate() on purpose since we only exercise expert_working_set().
    with pytest.raises(ValueError, match="no expert bundles"):
        manifest.expert_working_set()
