"""Safetensors headers, streamed-MoE manifests, and exact ``os.pread`` reads."""

from __future__ import annotations

import json
import os
import re
import struct
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..cache.policy import ExpertKey
from ..manifest import ExpertBundleSpec, ModelManifest, QuantizationSpec, TensorSpan
from .base import StorageReadError, StorageReadMetrics

_MAX_HEADER_SIZE = 100 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_QWEN3_LEADING_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.(?P<field>weight|scales|biases|bias)$"
)
_QWEN3_SPLIT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.(?P<field>weight|scales|biases|bias)$"
)
_QWEN3_5_LEADING_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.(?P<field>weight|scales|biases|bias)$"
)
_GEMMA4_LEADING_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.experts\.switch_glu\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.(?P<field>weight|scales|biases|bias)$"
)
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_NUMPY_DTYPES = {
    "BOOL": np.bool_,
    "I8": np.int8,
    "U8": np.uint8,
    "I16": np.int16,
    "U16": np.uint16,
    "F16": np.float16,
    "I32": np.int32,
    "U32": np.uint32,
    "F32": np.float32,
    "I64": np.int64,
    "U64": np.uint64,
    "F64": np.float64,
}


@dataclass(frozen=True)
class SafeTensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    nbytes: int


@dataclass(frozen=True)
class SafeTensorHeader:
    file: Path
    data_start: int
    tensors: dict[str, SafeTensorInfo]

    @classmethod
    def read(cls, path: Path) -> SafeTensorHeader:
        path = path.resolve()
        try:
            file_size = path.stat().st_size
            with path.open("rb") as source:
                raw_header_size = source.read(8)
                if len(raw_header_size) != 8:
                    raise StorageReadError(f"truncated safetensors prefix: {path}")
                header_size = struct.unpack("<Q", raw_header_size)[0]
                invalid_size = header_size <= 0 or header_size > _MAX_HEADER_SIZE
                if invalid_size or 8 + header_size > file_size:
                    raise StorageReadError(f"invalid safetensors header size in {path}")
                raw_header = source.read(header_size)
        except OSError as error:
            raise StorageReadError(f"cannot read safetensors header: {path}") from error
        try:
            header_value = json.loads(raw_header)
        except json.JSONDecodeError as error:
            raise StorageReadError(f"invalid JSON safetensors header: {path}") from error
        if not isinstance(header_value, dict):
            raise StorageReadError(f"safetensors header must be an object: {path}")

        data_start = 8 + header_size
        tensors: dict[str, SafeTensorInfo] = {}
        for name, value in header_value.items():
            if name == "__metadata__":
                continue
            if not isinstance(value, dict):
                raise StorageReadError(f"invalid tensor metadata for {name!r} in {path}")
            try:
                dtype = str(value["dtype"])
                shape = tuple(int(dimension) for dimension in value["shape"])
                data_offsets = value["data_offsets"]
                start, end = int(data_offsets[0]), int(data_offsets[1])
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise StorageReadError(f"invalid tensor offsets for {name!r} in {path}") from error
            if dtype not in _DTYPE_BYTES:
                raise StorageReadError(f"unsupported safetensors dtype {dtype!r} for {name!r}")
            if any(dimension <= 0 for dimension in shape) or start < 0 or end < start:
                raise StorageReadError(f"invalid tensor shape or range for {name!r} in {path}")
            expected_nbytes = _product(shape) * _DTYPE_BYTES[dtype]
            if end - start != expected_nbytes:
                message = (
                    f"tensor size mismatch for {name!r} in {path}: "
                    f"expected {expected_nbytes}, got {end - start}"
                )
                raise StorageReadError(
                    message
                )
            if data_start + end > file_size:
                raise StorageReadError(f"tensor range exceeds file for {name!r} in {path}")
            tensors[name] = SafeTensorInfo(
                dtype=dtype, shape=shape, data_offset=data_start + start, nbytes=end - start
            )
        if not tensors:
            raise StorageReadError(f"safetensors file has no tensors: {path}")
        return cls(file=path, data_start=data_start, tensors=tensors)


@dataclass(frozen=True)
class _TensorLocation:
    file: Path
    info: SafeTensorInfo


@dataclass(frozen=True)
class _ExpertTensorLayout:
    """The source-name convention for one streamed MoE family."""

    leading_pattern: re.Pattern[str]
    split_pattern: re.Pattern[str] | None = None


_LAYOUTS: dict[str, _ExpertTensorLayout] = {
    "qwen3_moe": _ExpertTensorLayout(_QWEN3_LEADING_PATTERN, _QWEN3_SPLIT_PATTERN),
    "qwen3_5_moe": _ExpertTensorLayout(_QWEN3_5_LEADING_PATTERN),
    "gemma4": _ExpertTensorLayout(_GEMMA4_LEADING_PATTERN),
}


class SafetensorsExpertStore:
    """Thread-safe exact ``os.pread`` of selected expert ranges without a shard load."""

    def __init__(self) -> None:
        self._descriptors: dict[Path, int] = {}
        self._bytes_read = 0
        self._read_count = 0
        self._lock = threading.RLock()

    def __enter__(self) -> SafetensorsExpertStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            for descriptor in self._descriptors.values():
                os.close(descriptor)
            self._descriptors.clear()

    def metrics(self) -> StorageReadMetrics:
        with self._lock:
            return StorageReadMetrics(bytes_read=self._bytes_read, read_count=self._read_count)

    def read_tensor(self, tensor: TensorSpan) -> bytes:
        if tensor.offset < 0 or tensor.nbytes <= 0:
            raise StorageReadError(f"invalid requested span {tensor.tensor_name!r}")
        path = tensor.file.resolve()
        try:
            with self._lock:
                descriptor = self._descriptors.get(path)
                if descriptor is None:
                    descriptor = os.open(path, os.O_RDONLY)
                    self._descriptors[path] = descriptor
                file_size = os.fstat(descriptor).st_size
            if tensor.offset + tensor.nbytes > file_size:
                raise StorageReadError(
                    f"requested range exceeds source file: {path}:{tensor.offset}+{tensor.nbytes}"
                )
            data = os.pread(descriptor, tensor.nbytes, tensor.offset)
        except StorageReadError:
            raise
        except OSError as error:
            raise StorageReadError(f"failed to pread {path}") from error
        if len(data) != tensor.nbytes:
            raise StorageReadError(
                f"short read for {tensor.tensor_name!r}: expected {tensor.nbytes}, got {len(data)}"
            )
        with self._lock:
            self._bytes_read += len(data)
            self._read_count += 1
        return data

    def read_bundle(self, bundle: ExpertBundleSpec) -> dict[str, bytes]:
        tensors: dict[str, bytes] = {}
        for tensor in bundle.tensors:
            if tensor.role in tensors:
                message = f"duplicate tensor role in bundle {bundle.key}: {tensor.role}"
                raise StorageReadError(message)
            tensors[tensor.role] = self.read_tensor(tensor)
        if sum(len(value) for value in tensors.values()) != bundle.total_bytes:
            raise StorageReadError(f"incorrect read length for expert bundle {bundle.key}")
        return tensors


def resolve_model_path(model: str | Path) -> Path:
    """Resolve a local directory or download a standard safetensors HF snapshot."""

    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists():
        raise ValueError(f"model path is not a directory: {path}")
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
        raise ValueError("Hugging Face model resolution requires huggingface_hub") from error
    downloaded = snapshot_download(
        str(model),
        allow_patterns=["*.json", "model*.safetensors", "tokenizer.*", "*.jinja", "*.txt"],
    )
    return Path(downloaded).resolve()


def build_qwen3_moe_manifest(model: str | Path) -> ModelManifest:
    """Inspect Qwen3-MoE headers and build a fail-fast selective-read manifest."""

    manifest = build_streaming_manifest(model)
    if manifest.model_type != "qwen3_moe":
        raise ValueError(
            "build_qwen3_moe_manifest only accepts model_type='qwen3_moe'; "
            "use build_streaming_manifest for other supported families"
        )
    return manifest


def build_streaming_manifest(model: str | Path) -> ModelManifest:
    """Build an exact-read manifest for a supported MLX MoE checkpoint.

    This inspection reads safetensors headers only.  It deliberately rejects
    unrecognised layouts so a new family cannot silently classify dense or
    routed tensors incorrectly.
    """

    model_path = resolve_model_path(model)
    source_config = _load_config(model_path)
    model_type = str(source_config.get("model_type", ""))
    try:
        layout = _LAYOUTS[model_type]
    except KeyError as error:
        raise ValueError(
            "supported streamed-MoE model types are "
            f"{', '.join(sorted(_LAYOUTS))}; got {model_type!r}"
        ) from error
    text_config = _text_config(source_config, model_type)
    num_layers = _required_positive_int(text_config, "num_hidden_layers")
    num_experts = _required_positive_int(text_config, "num_experts")
    experts_per_token = _experts_per_token(text_config, model_type)
    quantization = QuantizationSpec.from_dict(source_config.get("quantization"))
    locations, shard_files = _tensor_locations(model_path)
    bundles: dict[ExpertKey, ExpertBundleSpec] = {}
    for layer in range(num_layers):
        bundles.update(
            _build_layer_bundles(layer, num_experts, locations, quantization, layout)
        )
    manifest = ModelManifest(
        format_version=1,
        model_type=model_type,
        source_model=str(model),
        source_model_path=model_path,
        num_layers=num_layers,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        quantization=quantization,
        non_expert_weight_files=tuple(shard_files),
        expert_bundles=bundles,
    )
    manifest.validate()
    return manifest


def load_nonexpert_weights(
    manifest: ModelManifest, *, include: Callable[[str], bool] | None = None
) -> dict[str, Any]:
    """Materialize only non-routed source tensors as MLX arrays.

    This is the M3 shell-loader path. It reads all non-expert tensors one at a
    time; routed ``switch_mlp`` tensors remain on disk until an expert route
    explicitly resolves them.
    """

    try:
        import mlx.core as mx
    except ModuleNotFoundError as error:  # pragma: no cover - package dependency is normal
        raise RuntimeError("loading a streaming shell requires MLX") from error

    locations, _ = _tensor_locations(manifest.source_model_path)
    weights: dict[str, Any] = {}
    with SafetensorsExpertStore() as store:
        for name, location in sorted(locations.items()):
            if is_routed_expert_tensor(name, model_type=manifest.model_type):
                continue
            if include is not None and not include(name):
                continue
            info = location.info
            span = TensorSpan(
                file=location.file,
                tensor_name=name,
                offset=info.data_offset,
                nbytes=info.nbytes,
                shape=info.shape,
                dtype=info.dtype,
                role=name,
            )
            weights[name] = materialize_mlx_array(
                store.read_tensor(span), info.dtype, info.shape, mx
            )
    return weights


def materialize_mlx_array(data: bytes, dtype: str, shape: tuple[int, ...], mx: Any) -> Any:
    """Create one MLX array from a precisely read safetensors byte range."""

    if dtype == "BF16":
        storage = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        return mx.array(storage).view(mx.bfloat16)
    numpy_dtype = _NUMPY_DTYPES.get(dtype)
    if numpy_dtype is None:
        raise ValueError(f"M3 does not support materializing safetensors dtype {dtype!r}")
    return mx.array(np.frombuffer(data, dtype=numpy_dtype).reshape(shape))


def is_routed_expert_tensor(name: str, *, model_type: str = "qwen3_moe") -> bool:
    """Whether a tensor stays on disk until an expert route uses it."""

    try:
        layout = _LAYOUTS[model_type]
    except KeyError as error:
        raise ValueError(f"unsupported streamed-MoE model type {model_type!r}") from error
    return layout.leading_pattern.fullmatch(name) is not None or (
        layout.split_pattern is not None and layout.split_pattern.fullmatch(name) is not None
    )


def _text_config(source_config: dict[str, Any], model_type: str) -> dict[str, Any]:
    if model_type == "qwen3_moe":
        return source_config
    value = source_config.get("text_config")
    if not isinstance(value, dict):
        raise ValueError(f"{model_type} config requires an object 'text_config'")
    return value


def _experts_per_token(config: dict[str, Any], model_type: str) -> int:
    name = "top_k_experts" if model_type == "gemma4" else "num_experts_per_tok"
    return _required_positive_int(config, name)


def _load_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model config {config_path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"model config must be an object: {config_path}")
    return value


def _required_positive_int(config: dict[str, Any], name: str) -> int:
    try:
        value = int(config[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"model config requires positive integer {name!r}") from error
    if value <= 0:
        raise ValueError(f"model config requires positive integer {name!r}")
    return value


def _tensor_locations(model_path: Path) -> tuple[dict[str, _TensorLocation], list[Path]]:
    shard_files = _discover_shards(model_path)
    locations: dict[str, _TensorLocation] = {}
    for shard in shard_files:
        header = SafeTensorHeader.read(shard)
        for name, info in header.tensors.items():
            if name in locations:
                raise ValueError(f"tensor {name!r} appears in multiple safetensors shards")
            locations[name] = _TensorLocation(file=header.file, info=info)
    return locations, shard_files


def _discover_shards(model_path: Path) -> list[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            names = sorted(set(index["weight_map"].values()))
        except (KeyError, OSError, json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid safetensors index {index_path}") from error
        shards = []
        for name in names:
            relative_name = Path(str(name))
            if (
                relative_name.is_absolute()
                or relative_name.parent != Path(".")
                or relative_name.name != str(name)
            ):
                raise ValueError(f"unsafe shard name in safetensors index: {name!r}")
            # A Hugging Face snapshot normally exposes model shards as symlinks
            # into the content-addressed blob cache. The lexical index name is
            # constrained here; SafeTensorHeader resolves the target later.
            candidate = model_path / relative_name
            shards.append(candidate)
    else:
        shards = sorted(path.resolve() for path in model_path.glob("model*.safetensors"))
    if not shards:
        raise ValueError(f"no model*.safetensors shards found in {model_path}")
    for shard in shards:
        if not shard.is_file():
            raise ValueError(f"safetensors shard is missing: {shard}")
    return shards


def _build_layer_bundles(
    layer: int,
    num_experts: int,
    locations: dict[str, _TensorLocation],
    quantization: QuantizationSpec,
    layout: _ExpertTensorLayout,
) -> dict[ExpertKey, ExpertBundleSpec]:
    leading = _leading_fields(layer, locations, layout.leading_pattern)
    split = _split_fields(layer, locations, layout.split_pattern)
    if leading and split:
        raise ValueError(f"layer {layer} mixes leading-axis and split-expert tensors")
    if leading:
        return _build_leading_axis_bundles(layer, num_experts, leading, quantization)
    if split:
        return _build_split_bundles(layer, num_experts, split, quantization)
    raise ValueError(f"layer {layer} has no supported routed expert tensor layout")


def _leading_fields(
    layer: int, locations: dict[str, _TensorLocation], pattern: re.Pattern[str]
) -> dict[tuple[str, str], tuple[str, _TensorLocation]]:
    fields: dict[tuple[str, str], tuple[str, _TensorLocation]] = {}
    for name, location in locations.items():
        match = pattern.fullmatch(name)
        if match is not None and int(match["layer"]) == layer:
            fields[(match["projection"], match["field"])] = (name, location)
    return fields


def _split_fields(
    layer: int,
    locations: dict[str, _TensorLocation],
    pattern: re.Pattern[str] | None,
) -> dict[tuple[int, str, str], tuple[str, _TensorLocation]]:
    fields: dict[tuple[int, str, str], tuple[str, _TensorLocation]] = {}
    if pattern is None:
        return fields
    for name, location in locations.items():
        match = pattern.fullmatch(name)
        if match is not None and int(match["layer"]) == layer:
            fields[(int(match["expert"]), match["projection"], match["field"])] = (name, location)
    return fields


def _build_leading_axis_bundles(
    layer: int,
    num_experts: int,
    fields: dict[tuple[str, str], tuple[str, _TensorLocation]],
    quantization: QuantizationSpec,
) -> dict[ExpertKey, ExpertBundleSpec]:
    _validate_required_weights(layer, fields)
    for (projection, field), (name, location) in fields.items():
        info = location.info
        if len(info.shape) < 2 or info.shape[0] != num_experts or info.nbytes % num_experts:
            raise ValueError(
                f"leading expert axis is not a contiguous [{num_experts}, ...] tensor: {name}"
            )
        _validate_projection_field(projection, field, name)
    bundles: dict[ExpertKey, ExpertBundleSpec] = {}
    for expert in range(num_experts):
        spans = []
        for (projection, field), (name, location) in sorted(fields.items()):
            info = location.info
            expert_bytes = info.nbytes // num_experts
            spans.append(
                TensorSpan(
                    file=location.file,
                    tensor_name=name,
                    offset=info.data_offset + expert * expert_bytes,
                    nbytes=expert_bytes,
                    shape=info.shape[1:],
                    dtype=info.dtype,
                    role=_role(projection, field),
                )
            )
        key = ExpertKey(layer, expert)
        bundles[key] = ExpertBundleSpec(
            key=key,
            tensors=tuple(spans),
            total_bytes=sum(span.nbytes for span in spans),
            quantization=quantization,
        )
    return bundles


def _build_split_bundles(
    layer: int,
    num_experts: int,
    fields: dict[tuple[int, str, str], tuple[str, _TensorLocation]],
    quantization: QuantizationSpec,
) -> dict[ExpertKey, ExpertBundleSpec]:
    bundles: dict[ExpertKey, ExpertBundleSpec] = {}
    for expert in range(num_experts):
        expert_fields = {
            (projection, field): value
            for (candidate, projection, field), value in fields.items()
            if candidate == expert
        }
        _validate_required_weights(layer, expert_fields)
        spans = []
        for (projection, field), (name, location) in sorted(expert_fields.items()):
            _validate_projection_field(projection, field, name)
            info = location.info
            spans.append(
                TensorSpan(
                    file=location.file,
                    tensor_name=name,
                    offset=info.data_offset,
                    nbytes=info.nbytes,
                    shape=info.shape,
                    dtype=info.dtype,
                    role=_role(projection, field),
                )
            )
        key = ExpertKey(layer, expert)
        bundles[key] = ExpertBundleSpec(
            key=key,
            tensors=tuple(spans),
            total_bytes=sum(span.nbytes for span in spans),
            quantization=quantization,
        )
    unknown_experts = {expert for expert, _, _ in fields if expert >= num_experts}
    if unknown_experts:
        raise ValueError(
            f"layer {layer} has split experts outside config range: {sorted(unknown_experts)}"
        )
    return bundles


def _validate_required_weights(
    layer: int, fields: Iterable[tuple[str, str]] | dict[Any, Any]
) -> None:
    available = set(fields)
    missing = [
        (projection, "weight")
        for projection in _PROJECTIONS
        if (projection, "weight") not in available
    ]
    if missing:
        raise ValueError(f"layer {layer} is missing routed expert weights: {missing}")


def _validate_projection_field(projection: str, field: str, name: str) -> None:
    if projection not in _PROJECTIONS or field not in {"weight", "scales", "biases", "bias"}:
        raise ValueError(f"unsupported expert tensor {name!r}")


def _role(projection: str, field: str) -> str:
    return f"{projection.removesuffix('_proj')}_{field}"


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
