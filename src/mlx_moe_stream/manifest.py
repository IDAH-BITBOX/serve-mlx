"""Validated manifest for selective streamed-MoE expert reads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cache.policy import ExpertKey

SUPPORTED_MODEL_TYPES = frozenset({"qwen3_moe", "qwen3_5_moe", "gemma4"})


@dataclass(frozen=True)
class QuantizationSpec:
    """Quantization metadata required by a later expert execution backend."""

    bits: int | None = None
    group_size: int | None = None
    mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> QuantizationSpec:
        value = value or {}
        return cls(
            bits=_optional_int(value.get("bits")),
            group_size=_optional_int(value.get("group_size")),
            mode=value.get("mode"),
        )


@dataclass(frozen=True)
class TensorSpan:
    """The exact byte range for one tensor belonging to one expert."""

    file: Path
    tensor_name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.file),
            "tensor_name": self.tensor_name,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TensorSpan:
        return cls(
            file=Path(value["file"]),
            tensor_name=str(value["tensor_name"]),
            offset=int(value["offset"]),
            nbytes=int(value["nbytes"]),
            shape=tuple(int(dimension) for dimension in value["shape"]),
            dtype=str(value["dtype"]),
            role=str(value["role"]),
        )


@dataclass(frozen=True)
class ExpertBundleSpec:
    """All source tensor spans needed to execute one routed expert exactly."""

    key: ExpertKey
    tensors: tuple[TensorSpan, ...]
    total_bytes: int
    quantization: QuantizationSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": {"layer": self.key.layer, "expert": self.key.expert},
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "total_bytes": self.total_bytes,
            "quantization": self.quantization.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExpertBundleSpec:
        key_value = value["key"]
        return cls(
            key=ExpertKey(layer=int(key_value["layer"]), expert=int(key_value["expert"])),
            tensors=tuple(TensorSpan.from_dict(tensor) for tensor in value["tensors"]),
            total_bytes=int(value["total_bytes"]),
            quantization=QuantizationSpec.from_dict(value.get("quantization")),
        )


@dataclass(frozen=True)
class ExpertWorkingSet:
    """Aggregate expert-bundle byte totals used to size the M7 memory budget."""

    total_bytes: int
    bundle_count: int
    mean_bundle_bytes: float
    min_bundle_bytes: int
    max_bundle_bytes: int
    per_token_full_miss_bytes: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelManifest:
    """Portable JSON description of a prepared source model's expert bytes."""

    format_version: int
    model_type: str
    source_model: str
    source_model_path: Path
    num_layers: int
    num_experts: int
    experts_per_token: int
    quantization: QuantizationSpec
    non_expert_weight_files: tuple[Path, ...]
    expert_bundles: dict[ExpertKey, ExpertBundleSpec]

    def validate(self, *, check_files: bool = True) -> None:
        if self.format_version != 1:
            raise ValueError(f"unsupported manifest format version {self.format_version}")
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"unsupported manifest model type {self.model_type!r}")
        if self.num_layers <= 0 or self.num_experts <= 0 or self.experts_per_token <= 0:
            raise ValueError("manifest layer and expert counts must be greater than zero")
        expected_bundles = self.num_layers * self.num_experts
        if len(self.expert_bundles) != expected_bundles:
            raise ValueError(
                f"manifest has {len(self.expert_bundles)} bundles; expected {expected_bundles}"
            )
        for layer in range(self.num_layers):
            for expert in range(self.num_experts):
                key = ExpertKey(layer, expert)
                bundle = self.expert_bundles.get(key)
                if bundle is None:
                    raise ValueError(f"manifest is missing expert bundle {key}")
                if bundle.key != key:
                    raise ValueError(f"expert bundle key mismatch for {key}")
                if bundle.total_bytes != sum(tensor.nbytes for tensor in bundle.tensors):
                    raise ValueError(f"incorrect total_bytes for {key}")
                roles = {tensor.role for tensor in bundle.tensors}
                if len(roles) != len(bundle.tensors) or not bundle.tensors:
                    raise ValueError(f"expert bundle {key} has duplicate or missing tensor roles")
                for tensor in bundle.tensors:
                    if tensor.offset < 0 or tensor.nbytes <= 0 or not tensor.shape:
                        raise ValueError(f"invalid tensor span {tensor.tensor_name!r} in {key}")
                    if check_files:
                        try:
                            file_size = tensor.file.stat().st_size
                        except OSError as error:
                            message = f"manifest source file is unavailable: {tensor.file}"
                            raise ValueError(message) from error
                        if tensor.offset + tensor.nbytes > file_size:
                            range_description = f"{tensor.file}:{tensor.offset}+{tensor.nbytes}"
                            raise ValueError(
                                f"tensor span exceeds source file: {range_description}"
                            )

    def expert_working_set(self) -> ExpertWorkingSet:
        """Summarize routed-expert byte totals already present in this manifest.

        Pure arithmetic over ``self.expert_bundles``: no file I/O, since the
        manifest is already loaded and validated by the time a caller wants
        this for M7 memory-budget sizing.
        """

        if not self.expert_bundles:
            raise ValueError("manifest has no expert bundles to summarize")
        bundle_bytes = [bundle.total_bytes for bundle in self.expert_bundles.values()]
        bundle_count = len(bundle_bytes)
        total_bytes = sum(bundle_bytes)
        mean_bundle_bytes = total_bytes / bundle_count
        return ExpertWorkingSet(
            total_bytes=total_bytes,
            bundle_count=bundle_count,
            mean_bundle_bytes=mean_bundle_bytes,
            min_bundle_bytes=min(bundle_bytes),
            max_bundle_bytes=max(bundle_bytes),
            per_token_full_miss_bytes=self.num_layers * self.experts_per_token * mean_bundle_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_type": self.model_type,
            "source_model": self.source_model,
            "source_model_path": str(self.source_model_path),
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "quantization": self.quantization.to_dict(),
            "non_expert_weight_files": [str(path) for path in self.non_expert_weight_files],
            "expert_bundles": {
                f"{key.layer}:{key.expert}": bundle.to_dict()
                for key, bundle in sorted(self.expert_bundles.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelManifest:
        bundles: dict[ExpertKey, ExpertBundleSpec] = {}
        for bundle_value in value["expert_bundles"].values():
            bundle = ExpertBundleSpec.from_dict(bundle_value)
            if bundle.key in bundles:
                raise ValueError(f"duplicate expert bundle {bundle.key}")
            bundles[bundle.key] = bundle
        manifest = cls(
            format_version=int(value["format_version"]),
            model_type=str(value["model_type"]),
            source_model=str(value["source_model"]),
            source_model_path=Path(value["source_model_path"]),
            num_layers=int(value["num_layers"]),
            num_experts=int(value["num_experts"]),
            experts_per_token=int(value["experts_per_token"]),
            quantization=QuantizationSpec.from_dict(value.get("quantization")),
            non_expert_weight_files=tuple(Path(path) for path in value["non_expert_weight_files"]),
            expert_bundles=bundles,
        )
        manifest.validate()
        return manifest

    def write(self, path: Path, *, overwrite: bool = False) -> None:
        self.validate()
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")


def load_manifest(path: Path) -> ModelManifest:
    """Load and validate a prepared M2 manifest."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {path}") from error
    return ModelManifest.from_dict(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
