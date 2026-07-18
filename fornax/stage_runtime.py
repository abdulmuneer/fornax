from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import re
import struct
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SUPPORTED_DTYPES = {"bf16", "fp16", "fp32"}
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
TERMINAL_STATUSES = {"ok", "cancelled", "deadline", "rejected", "failed"}
RequestReplayKey = tuple[str, int, str, int, int, int, str]


class StageRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, *, kv_mutated: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.kv_mutated = kv_mutated


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_hash(value: str, field_name: str) -> None:
    if not HASH_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be sha256:<64 lowercase hex>")


def _require_uuid(value: str, field_name: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be a lower-case canonical UUID")


def _float_to_bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def _bf16_to_float(data: bytes) -> float:
    high = struct.unpack("<H", data)[0]
    return struct.unpack("<f", struct.pack("<I", high << 16))[0]


def encode_values(values: tuple[float, ...], dtype: str) -> bytes:
    if dtype == "bf16":
        return b"".join(_float_to_bf16(value) for value in values)
    if dtype == "fp16":
        return b"".join(struct.pack("<e", float(value)) for value in values)
    if dtype == "fp32":
        return b"".join(struct.pack("<f", float(value)) for value in values)
    raise ValueError(f"unsupported tensor dtype: {dtype}")


def decode_values(payload: bytes, dtype: str) -> tuple[float, ...]:
    scalar_bytes = DTYPE_BYTES.get(dtype)
    if scalar_bytes is None:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    if len(payload) % scalar_bytes:
        raise ValueError("payload length is not divisible by scalar size")
    if dtype == "bf16":
        return tuple(
            _bf16_to_float(payload[offset : offset + 2])
            for offset in range(0, len(payload), 2)
        )
    code = "<e" if dtype == "fp16" else "<f"
    return tuple(
        struct.unpack(code, payload[offset : offset + scalar_bytes])[0]
        for offset in range(0, len(payload), scalar_bytes)
    )


@dataclass(frozen=True)
class TensorDescriptor:
    kind: str
    dtype: str
    shape: tuple[int, ...]
    layout: str = "contiguous_row_major"
    logical_elements: int | None = None
    payload_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"activation", "logits"}:
            raise ValueError("tensor kind must be activation or logits")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported tensor dtype: {self.dtype}")
        if not self.shape or any(
            isinstance(dimension, bool) or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("tensor shape must contain positive dimensions")
        if self.layout != "contiguous_row_major":
            raise ValueError("only contiguous_row_major is supported")
        elements = math.prod(self.shape)
        expected_bytes = elements * DTYPE_BYTES[self.dtype]
        if self.logical_elements is not None and self.logical_elements != elements:
            raise ValueError("logical_elements does not match shape")
        if self.payload_bytes is not None and self.payload_bytes != expected_bytes:
            raise ValueError("payload_bytes does not match shape and dtype")
        object.__setattr__(self, "logical_elements", elements)
        object.__setattr__(self, "payload_bytes", expected_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "logical_elements": self.logical_elements,
            "payload_bytes": self.payload_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TensorDescriptor":
        if not isinstance(value, dict):
            raise ValueError("tensor descriptor must be an object")
        return cls(
            kind=str(value.get("kind", "")),
            dtype=str(value.get("dtype", "")),
            shape=tuple(int(item) for item in value.get("shape", [])),
            layout=str(value.get("layout", "")),
            logical_elements=(
                int(value["logical_elements"])
                if value.get("logical_elements") is not None
                else None
            ),
            payload_bytes=(
                int(value["payload_bytes"])
                if value.get("payload_bytes") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Tensor:
    descriptor: TensorDescriptor
    payload: bytes

    def __post_init__(self) -> None:
        if len(self.payload) != self.descriptor.payload_bytes:
            raise ValueError("tensor payload length does not match descriptor")
        values = decode_values(self.payload, self.descriptor.dtype)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("tensor values must be finite")

    @classmethod
    def from_values(
        cls,
        values: list[float] | tuple[float, ...],
        *,
        kind: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> "Tensor":
        descriptor = TensorDescriptor(kind=kind, dtype=dtype, shape=shape)
        flat = tuple(float(value) for value in values)
        if len(flat) != descriptor.logical_elements:
            raise ValueError("value count does not match tensor shape")
        return cls(descriptor=descriptor, payload=encode_values(flat, dtype))

    def values(self) -> tuple[float, ...]:
        return decode_values(self.payload, self.descriptor.dtype)


@dataclass(frozen=True)
class BufferAdapterHealth:
    """Observable staging-buffer ownership for a tensor buffer adapter."""

    inflight_imports: int
    inflight_bytes: int
    high_water_bytes: int
    max_imported_bytes: int
    import_operations: int
    export_operations: int
    copy_operations: int


@dataclass
class ImportedTensorBuffer:
    """One adapter-owned tensor import.

    ``native_handle`` is deliberately opaque.  A MAX adapter can place a device
    allocation, registered host buffer, or runtime tensor there; the Python
    reference adapter uses a ``bytearray``.  Ownership remains with the adapter
    until :meth:`TensorBufferAdapter.release` is called.
    """

    descriptor: TensorDescriptor
    native_handle: Any
    nbytes: int
    owner: str
    purpose: str
    copy_performed: bool
    descriptor_validated: bool
    payload_validated: bool
    released: bool = False


@runtime_checkable
class TensorBufferAdapter(Protocol):
    """Explicit logical-tensor to backend-buffer staging seam.

    This protocol is optional for a :class:`StageExecutable`; it does not add a
    required method to Stage Backend API v2.  It gives physical adapter authors
    a concrete place to account for import/export copies while the Python
    implementation remains the correctness oracle.
    """

    def import_tensor(
        self,
        tensor: Tensor,
        *,
        expected: TensorDescriptor,
        purpose: str,
    ) -> ImportedTensorBuffer: ...

    def export_tensor(
        self,
        imported: ImportedTensorBuffer,
        *,
        expected: TensorDescriptor,
    ) -> Tensor: ...

    def release(self, imported: ImportedTensorBuffer) -> None: ...

    def health(self) -> BufferAdapterHealth: ...


class PythonTensorBufferAdapter:
    """Bounded, copy-explicit staging used by the CPU reference backend."""

    def __init__(self, *, max_imported_bytes: int = 256 * 1024 * 1024) -> None:
        if (
            isinstance(max_imported_bytes, bool)
            or not isinstance(max_imported_bytes, int)
            or max_imported_bytes <= 0
        ):
            raise ValueError("max_imported_bytes must be a positive integer")
        self._max_imported_bytes = max_imported_bytes
        self._inflight_imports = 0
        self._inflight_bytes = 0
        self._high_water_bytes = 0
        self._import_operations = 0
        self._export_operations = 0
        self._copy_operations = 0
        self._active_handles: set[int] = set()
        self._lock = threading.RLock()

    @staticmethod
    def _validate_descriptor(
        actual: TensorDescriptor, expected: TensorDescriptor
    ) -> None:
        if actual != expected:
            raise StageRuntimeError(
                "TENSOR_CONTRACT",
                "native-buffer import descriptor does not match expected contract",
            )

    def import_tensor(
        self,
        tensor: Tensor,
        *,
        expected: TensorDescriptor,
        purpose: str,
    ) -> ImportedTensorBuffer:
        _require_non_empty(purpose, "purpose")
        self._validate_descriptor(tensor.descriptor, expected)
        nbytes = len(tensor.payload)
        with self._lock:
            if self._inflight_bytes + nbytes > self._max_imported_bytes:
                raise StageRuntimeError(
                    "ADMISSION",
                    "native-buffer staging capacity is exhausted",
                )
            self._inflight_imports += 1
            self._inflight_bytes += nbytes
            self._high_water_bytes = max(
                self._high_water_bytes, self._inflight_bytes
            )
        try:
            # ``bytearray`` guarantees an owned copy even when Tensor.payload is
            # already immutable bytes.  Reconstructing Tensor performs the
            # byte-count and finite-value validation at the import boundary.
            storage = bytearray(tensor.payload)
            Tensor(descriptor=expected, payload=bytes(storage))
        except Exception:
            with self._lock:
                self._inflight_imports -= 1
                self._inflight_bytes -= nbytes
            raise
        imported = ImportedTensorBuffer(
            descriptor=expected,
            native_handle=storage,
            nbytes=nbytes,
            owner="python-reference-staging",
            purpose=purpose,
            copy_performed=True,
            descriptor_validated=True,
            payload_validated=True,
        )
        with self._lock:
            self._active_handles.add(id(imported))
            self._import_operations += 1
            self._copy_operations += 1
        return imported

    def export_tensor(
        self,
        imported: ImportedTensorBuffer,
        *,
        expected: TensorDescriptor,
    ) -> Tensor:
        with self._lock:
            if imported.released or id(imported) not in self._active_handles:
                raise StageRuntimeError(
                    "TENSOR_CONTRACT",
                    "cannot export a released or foreign native buffer",
                )
            self._validate_descriptor(imported.descriptor, expected)
            if imported.nbytes != expected.payload_bytes:
                raise StageRuntimeError(
                    "TENSOR_CONTRACT",
                    "native-buffer byte count does not match expected contract",
                )
            if not isinstance(imported.native_handle, bytearray):
                raise StageRuntimeError(
                    "TENSOR_CONTRACT", "Python staging handle must be a bytearray"
                )
            tensor = Tensor(
                descriptor=expected,
                payload=bytes(imported.native_handle),
            )
            self._export_operations += 1
            self._copy_operations += 1
            return tensor

    def release(self, imported: ImportedTensorBuffer) -> None:
        with self._lock:
            if imported.released:
                return
            if id(imported) not in self._active_handles:
                raise StageRuntimeError(
                    "TENSOR_CONTRACT", "cannot release a foreign native buffer"
                )
            self._active_handles.remove(id(imported))
            imported.released = True
            self._inflight_imports -= 1
            self._inflight_bytes -= imported.nbytes
        if isinstance(imported.native_handle, bytearray):
            imported.native_handle.clear()

    def health(self) -> BufferAdapterHealth:
        with self._lock:
            return BufferAdapterHealth(
                inflight_imports=self._inflight_imports,
                inflight_bytes=self._inflight_bytes,
                high_water_bytes=self._high_water_bytes,
                max_imported_bytes=self._max_imported_bytes,
                import_operations=self._import_operations,
                export_operations=self._export_operations,
                copy_operations=self._copy_operations,
            )


@dataclass(frozen=True)
class StageManifest:
    manifest_version: int
    model_id: str
    model_snapshot: str
    model_config_hash: str
    tokenizer_hash: str
    template_hash: str
    max_build_id: str
    fornax_abi_major: int
    fornax_abi_minor: int
    plan_id: str
    plan_hash: str
    stage_id: str
    stage_index: int
    layer_start: int
    layer_end: int
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    kv_policy: str
    weight_artifacts: tuple[dict[str, Any], ...] = ()
    device_requirement: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.manifest_version != 1:
            raise ValueError("manifest_version must equal 1")
        for name in (
            "model_id",
            "model_snapshot",
            "max_build_id",
            "stage_id",
        ):
            _require_non_empty(str(getattr(self, name)), name)
        for name in (
            "model_config_hash",
            "tokenizer_hash",
            "template_hash",
            "plan_hash",
        ):
            _require_hash(str(getattr(self, name)), name)
        _require_uuid(self.plan_id, "plan_id")
        if self.fornax_abi_major != 1 or self.fornax_abi_minor < 0:
            raise ValueError("manifest requires ABI major 1 and non-negative minor")
        if self.stage_index < 0 or self.layer_start < 0 or self.layer_end < self.layer_start:
            raise ValueError("stage/layer indices are invalid")
        if self.kv_policy != "stage_local":
            raise ValueError("Phase 0.5 requires stage_local KV")
        for name, contract in (
            ("input_contract", self.input_contract),
            ("output_contract", self.output_contract),
        ):
            if not isinstance(contract, dict):
                raise ValueError(f"{name} must be an object")
            if contract.get("dtype") not in {"bf16", "fp16"}:
                raise ValueError(f"{name}.dtype must be bf16 or fp16")
            if contract.get("layout") != "contiguous_row_major":
                raise ValueError(f"{name}.layout must be contiguous_row_major")
            if not isinstance(contract.get("hidden_size"), int) or contract["hidden_size"] <= 0:
                raise ValueError(f"{name}.hidden_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "model_id": self.model_id,
            "model_snapshot": self.model_snapshot,
            "model_config_hash": self.model_config_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "template_hash": self.template_hash,
            "max_build_id": self.max_build_id,
            "fornax_abi_major": self.fornax_abi_major,
            "fornax_abi_minor": self.fornax_abi_minor,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "kv_policy": self.kv_policy,
            "weight_artifacts": list(self.weight_artifacts),
            "device_requirement": self.device_requirement,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageManifest":
        return cls(
            manifest_version=int(value.get("manifest_version", 0)),
            model_id=str(value.get("model_id", "")),
            model_snapshot=str(value.get("model_snapshot", "")),
            model_config_hash=str(value.get("model_config_hash", "")),
            tokenizer_hash=str(value.get("tokenizer_hash", "")),
            template_hash=str(value.get("template_hash", "")),
            max_build_id=str(value.get("max_build_id", "")),
            fornax_abi_major=int(value.get("fornax_abi_major", 0)),
            fornax_abi_minor=int(value.get("fornax_abi_minor", 0)),
            plan_id=str(value.get("plan_id", "")),
            plan_hash=str(value.get("plan_hash", "")),
            stage_id=str(value.get("stage_id", "")),
            stage_index=int(value.get("stage_index", -1)),
            layer_start=int(value.get("layer_start", -1)),
            layer_end=int(value.get("layer_end", -1)),
            input_contract=dict(value.get("input_contract", {})),
            output_contract=dict(value.get("output_contract", {})),
            kv_policy=str(value.get("kv_policy", "")),
            weight_artifacts=tuple(dict(item) for item in value.get("weight_artifacts", [])),
            device_requirement=dict(value.get("device_requirement", {})),
        )

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class StageHandle:
    handle_id: str
    stage_id: str
    manifest_hash: str


@dataclass(frozen=True)
class StageHealth:
    state: str
    stage_id: str
    manifest_hash: str
    inflight: int
    degraded: bool = False
    live_requests: int = 0
    completed_results: int = 0
    transform_cache_entries: int = 0
    max_live_requests: int = 0
    max_completed_results_per_request: int = 0
    max_transform_cache_entries: int = 0
    completed_result_bytes: int = 0
    completed_result_high_water_bytes: int = 0
    transform_cache_bytes: int = 0
    transform_cache_high_water_bytes: int = 0
    max_completed_result_bytes: int = 0
    max_transform_cache_bytes: int = 0
    release_tombstones: int = 0
    max_release_tombstones: int = 0
    request_idle_timeout_ns: int = 0
    execution_lease_timeout_ns: int = 0
    release_tombstone_ttl_ns: int = 0
    expired_requests: int = 0
    expired_execution_leases: int = 0
    native_buffer_imports: int = 0
    native_buffer_bytes: int = 0
    native_buffer_high_water_bytes: int = 0
    max_native_buffer_bytes: int = 0
    native_buffer_copy_operations: int = 0


@dataclass(frozen=True)
class BackendCapabilities:
    """Facts reported by a backend before a stage manifest is loaded.

    Requested manifest values are intentionally not inputs to this object.  The
    worker records them separately and fails startup when a known observed fact
    is incompatible with the request.
    """

    backend: str
    build_id: str
    device_identity: str | None
    supported_dtypes: tuple[str, ...]
    abi_versions: tuple[tuple[int, int], ...] = ((1, 0),)
    memory_bytes: int | None = None
    supported_operations: tuple[str, ...] = ("stage_execute",)
    supported_quantizations: tuple[str, ...] = ()
    max_frame_bytes: int = 256 * 1024 * 1024
    source: str = "backend"

    def __post_init__(self) -> None:
        _require_non_empty(self.backend, "backend")
        _require_non_empty(self.build_id, "build_id")
        if self.device_identity is not None:
            _require_non_empty(self.device_identity, "device_identity")
        if not self.supported_dtypes:
            raise ValueError("supported_dtypes must not be empty")
        unknown_dtypes = sorted(set(self.supported_dtypes) - SUPPORTED_DTYPES)
        if unknown_dtypes:
            raise ValueError(f"unsupported capability dtypes: {unknown_dtypes}")
        if not self.abi_versions or any(
            major <= 0 or minor < 0 for major, minor in self.abi_versions
        ):
            raise ValueError("abi_versions must contain positive major/non-negative minor pairs")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive when known")
        if self.max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        if self.source != "backend":
            raise ValueError("capability source must be backend")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "build_id": self.build_id,
            "device_identity": self.device_identity,
            "supported_dtypes": list(self.supported_dtypes),
            "abi_versions": [
                {"major": major, "minor": minor}
                for major, minor in self.abi_versions
            ],
            "memory_bytes": self.memory_bytes,
            "supported_operations": list(self.supported_operations),
            "supported_quantizations": list(self.supported_quantizations),
            "max_frame_bytes": self.max_frame_bytes,
            "source": self.source,
        }


@dataclass(frozen=True)
class StageBackendSpec:
    """JSON-serializable worker backend construction request.

    Physical adapters use ``kind="max"`` plus an importable ``module:factory``.
    The factory receives a plain options dictionary and must return a
    :class:`StageExecutable`.  This keeps multiprocessing startup explicit and
    prevents a missing physical adapter from falling back to simulation.
    """

    kind: str
    options: dict[str, Any] = field(default_factory=dict)
    factory: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"reference", "simulated-max", "max"}:
            raise ValueError(f"unsupported stage backend kind: {self.kind}")
        if not isinstance(self.options, dict):
            raise ValueError("backend options must be an object")
        canonical_json_bytes(self.options)
        if self.kind == "max":
            if not self.factory or ":" not in self.factory:
                raise ValueError("max backend requires an importable module:factory")
        elif self.factory is not None:
            raise ValueError("factory is only valid for the max backend")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "options": self.options, "factory": self.factory}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageBackendSpec":
        if not isinstance(value, dict):
            raise ValueError("backend spec must be an object")
        return cls(
            kind=str(value.get("kind", "")),
            options=dict(value.get("options", {})),
            factory=(str(value["factory"]) if value.get("factory") else None),
        )

    @classmethod
    def simulated(cls, profile: "SimulationProfile") -> "StageBackendSpec":
        return cls(kind="simulated-max", options=profile.to_dict())


@dataclass(frozen=True)
class StageRequest:
    plan_id: str
    plan_hash: str
    request_id: str
    microbatch_id: str
    sequence_no: int
    phase: str
    token_start: int
    token_count: int
    input_activation: Tensor
    kv_epoch: int
    deadline_ns: int
    trace_context: dict[str, str]

    def __post_init__(self) -> None:
        _require_uuid(self.plan_id, "plan_id")
        _require_hash(self.plan_hash, "plan_hash")
        _require_uuid(self.request_id, "request_id")
        _require_non_empty(self.microbatch_id, "microbatch_id")
        if (
            isinstance(self.sequence_no, bool)
            or not isinstance(self.sequence_no, int)
            or self.sequence_no < 0
        ):
            raise ValueError("sequence_no must be a non-negative integer")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if self.token_start < 0 or self.token_count <= 0:
            raise ValueError("token interval is invalid")
        if self.input_activation.descriptor.kind != "activation":
            raise ValueError("input_activation must be an activation tensor")
        if self.input_activation.descriptor.shape[0] != self.token_count:
            raise ValueError("token_count must match activation rows")
        if self.kv_epoch < 0 or self.deadline_ns <= 0:
            raise ValueError("kv_epoch/deadline are invalid")
        if not self.trace_context.get("trace_id") or not self.trace_context.get("span_id"):
            raise ValueError("trace_context requires trace_id and span_id")


@dataclass(frozen=True)
class StageResult:
    plan_id: str
    plan_hash: str
    request_id: str
    microbatch_id: str
    sequence_no: int
    status: str
    output_kind: str | None
    output_tensor: Tensor | None
    kv_epoch_before: int
    kv_epoch_after: int
    timings_ns: dict[str, int]
    error: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.status not in TERMINAL_STATUSES:
            raise ValueError(f"unsupported result status: {self.status}")
        if self.status == "ok":
            if self.output_kind not in {"activation", "logits"} or self.output_tensor is None:
                raise ValueError("successful result requires output tensor")
        elif self.output_tensor is not None:
            raise ValueError("failed result cannot contain output tensor")
        if self.kv_epoch_after < self.kv_epoch_before:
            raise ValueError("KV epoch cannot move backwards")


@dataclass(frozen=True)
class CancelResult:
    request_id: str
    cancelled: bool
    execution_started: bool
    kv_mutated: bool


@dataclass(frozen=True)
class ReleaseResult:
    """Counts of request-owned runtime state removed by ``release``."""

    request_id: str
    released: bool
    kv_state_released: int
    cancel_state_released: int
    execution_state_released: int
    idempotency_results_released: int
    tombstoned: bool = False
    reason: str | None = None

    @property
    def state_entries_released(self) -> int:
        return (
            self.kv_state_released
            + self.cancel_state_released
            + self.execution_state_released
            + self.idempotency_results_released
        )


@dataclass(frozen=True)
class DrainResult:
    drained: bool
    inflight: int


@dataclass(frozen=True)
class UnloadResult:
    unloaded: bool


@dataclass(frozen=True)
class LifecycleSweepResult:
    """State reclaimed by one opportunistic reference-runtime sweep."""

    expired_requests: int
    expired_execution_leases: int
    pruned_tombstones: int
    live_requests: int
    release_tombstones: int


@runtime_checkable
class StageExecutable(Protocol):
    def capabilities(self) -> BackendCapabilities: ...

    def load(self, manifest: StageManifest) -> StageHandle: ...

    def health(self, handle: StageHandle) -> StageHealth: ...

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult: ...

    def cancel(self, handle: StageHandle, request_id: str, reason: str) -> CancelResult: ...

    def release(self, handle: StageHandle, request_id: str) -> ReleaseResult: ...

    def drain(self, handle: StageHandle, deadline_ns: int) -> DrainResult: ...

    def unload(self, handle: StageHandle) -> UnloadResult: ...


@dataclass
class _RequestState:
    kv_epoch: int = 0
    cancelled: bool = False
    execution_started: bool = False
    inflight: int = 0
    highest_sequence_no: int = -1
    last_activity_ns: int = 0
    execution_lease_id: int | None = None
    execution_lease_expires_ns: int = 0
    completed: OrderedDict[RequestReplayKey, StageResult] = field(
        default_factory=OrderedDict
    )


@dataclass(frozen=True)
class _ReleaseTombstone:
    request_id: str
    released_at_ns: int
    expires_at_ns: int
    reason: str
    highest_sequence_no: int


@dataclass
class _LoadedStage:
    manifest: StageManifest
    max_live_requests: int
    max_completed_results_per_request: int
    max_transform_cache_entries: int
    max_completed_result_bytes: int
    max_transform_cache_bytes: int
    max_release_tombstones: int
    request_idle_timeout_ns: int
    execution_lease_timeout_ns: int
    release_tombstone_ttl_ns: int
    state: str = "READY"
    inflight: int = 0
    next_execution_lease_id: int = 1
    expired_requests: int = 0
    expired_execution_leases: int = 0
    requests: dict[str, _RequestState] = field(default_factory=dict)
    release_tombstones: OrderedDict[str, _ReleaseTombstone] = field(
        default_factory=OrderedDict
    )
    transform_cache: OrderedDict[str, Tensor] = field(default_factory=OrderedDict)
    completed_lru: OrderedDict[
        tuple[str, RequestReplayKey], int
    ] = field(default_factory=OrderedDict)
    completed_result_bytes: int = 0
    completed_result_high_water_bytes: int = 0
    transform_cache_bytes: int = 0
    transform_cache_high_water_bytes: int = 0


def deterministic_stage_values(
    values: tuple[float, ...],
    *,
    rows: int,
    width: int,
    layer_start: int,
    layer_end: int,
) -> tuple[float, ...]:
    matrix = [list(values[row * width : (row + 1) * width]) for row in range(rows)]
    for layer_id in range(layer_start, layer_end + 1):
        output: list[list[float]] = []
        for row in matrix:
            next_row: list[float] = []
            for output_dim in range(width):
                acc = (((layer_id + 5) * (output_dim + 13)) % 23 - 11) / 230.0
                value = row[output_dim]
                neighbor = row[(output_dim + layer_id + 1) % width]
                diagonal = (((layer_id + 3) * (output_dim + 11)) % 43 - 21) / 43.0
                mixing = (((layer_id + 7) * (output_dim + 5)) % 29 - 14) / 58.0
                acc += value * diagonal + neighbor * mixing
                next_row.append(round(max(acc, 0.0), 7))
            output.append(next_row)
        matrix = output
    return tuple(value for row in matrix for value in row)


class ReferenceStageBackend:
    backend_name = "reference"
    _RESULT_OVERHEAD_BYTES = 512
    _TRANSFORM_OVERHEAD_BYTES = 128

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_live_requests: int = 4096,
        max_completed_results_per_request: int = 64,
        max_transform_cache_entries: int = 256,
        max_completed_result_bytes: int = 64 * 1024 * 1024,
        max_transform_cache_bytes: int = 64 * 1024 * 1024,
        max_release_tombstones: int = 8192,
        request_idle_timeout_ns: int = 15 * 60 * 1_000_000_000,
        execution_lease_timeout_ns: int = 5 * 60 * 1_000_000_000,
        release_tombstone_ttl_ns: int = 30 * 60 * 1_000_000_000,
        max_native_buffer_bytes: int = 256 * 1024 * 1024,
        buffer_adapter: TensorBufferAdapter | None = None,
    ) -> None:
        limits = {
            "max_live_requests": max_live_requests,
            "max_completed_results_per_request": max_completed_results_per_request,
            "max_transform_cache_entries": max_transform_cache_entries,
            "max_completed_result_bytes": max_completed_result_bytes,
            "max_transform_cache_bytes": max_transform_cache_bytes,
            "max_release_tombstones": max_release_tombstones,
            "request_idle_timeout_ns": request_idle_timeout_ns,
            "execution_lease_timeout_ns": execution_lease_timeout_ns,
            "release_tombstone_ttl_ns": release_tombstone_ttl_ns,
            "max_native_buffer_bytes": max_native_buffer_bytes,
        }
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._clock_ns = clock_ns
        self._max_live_requests = max_live_requests
        self._max_completed_results_per_request = max_completed_results_per_request
        self._max_transform_cache_entries = max_transform_cache_entries
        self._max_completed_result_bytes = max_completed_result_bytes
        self._max_transform_cache_bytes = max_transform_cache_bytes
        self._max_release_tombstones = max_release_tombstones
        self._request_idle_timeout_ns = request_idle_timeout_ns
        self._execution_lease_timeout_ns = execution_lease_timeout_ns
        self._release_tombstone_ttl_ns = release_tombstone_ttl_ns
        self._buffer_adapter = buffer_adapter or PythonTensorBufferAdapter(
            max_imported_bytes=max_native_buffer_bytes
        )
        if not isinstance(self._buffer_adapter, TensorBufferAdapter):
            raise ValueError("buffer_adapter must implement TensorBufferAdapter")
        self._loaded: dict[str, _LoadedStage] = {}
        self._lock = threading.RLock()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            build_id="fornax-reference-python-v1",
            device_identity="cpu-reference",
            supported_dtypes=tuple(sorted(SUPPORTED_DTYPES)),
            memory_bytes=None,
        )

    def load(self, manifest: StageManifest) -> StageHandle:
        handle = StageHandle(
            handle_id=f"{manifest.stage_id}:{manifest.manifest_hash[7:23]}",
            stage_id=manifest.stage_id,
            manifest_hash=manifest.manifest_hash,
        )
        with self._lock:
            if handle.handle_id in self._loaded:
                raise StageRuntimeError("STALE_PLAN", "manifest is already loaded")
            self._loaded[handle.handle_id] = _LoadedStage(
                manifest=manifest,
                max_live_requests=self._max_live_requests,
                max_completed_results_per_request=(
                    self._max_completed_results_per_request
                ),
                max_transform_cache_entries=self._max_transform_cache_entries,
                max_completed_result_bytes=self._max_completed_result_bytes,
                max_transform_cache_bytes=self._max_transform_cache_bytes,
                max_release_tombstones=self._max_release_tombstones,
                request_idle_timeout_ns=self._request_idle_timeout_ns,
                execution_lease_timeout_ns=self._execution_lease_timeout_ns,
                release_tombstone_ttl_ns=self._release_tombstone_ttl_ns,
            )
        return handle

    def _stage(self, handle: StageHandle) -> _LoadedStage:
        stage = self._loaded.get(handle.handle_id)
        if stage is None or stage.manifest.manifest_hash != handle.manifest_hash:
            raise StageRuntimeError("STALE_PLAN", "unknown or stale stage handle")
        return stage

    @staticmethod
    def _prune_tombstones(stage: _LoadedStage, now_ns: int) -> int:
        pruned = 0
        for request_id, tombstone in tuple(stage.release_tombstones.items()):
            if tombstone.expires_at_ns > now_ns:
                continue
            del stage.release_tombstones[request_id]
            pruned += 1
        return pruned

    @staticmethod
    def _remember_tombstone(
        stage: _LoadedStage,
        *,
        request_id: str,
        now_ns: int,
        reason: str,
        highest_sequence_no: int,
    ) -> None:
        existing = stage.release_tombstones.pop(request_id, None)
        if existing is None and len(stage.release_tombstones) >= stage.max_release_tombstones:
            raise StageRuntimeError(
                "TOMBSTONE_CAPACITY",
                "release tombstone capacity is exhausted; live fences are preserved",
            )
        stage.release_tombstones[request_id] = _ReleaseTombstone(
            request_id=request_id,
            released_at_ns=now_ns,
            expires_at_ns=now_ns + stage.release_tombstone_ttl_ns,
            reason=reason,
            highest_sequence_no=highest_sequence_no,
        )

    def _remove_request_state(
        self,
        stage: _LoadedStage,
        request_id: str,
        *,
        now_ns: int,
        reason: str,
    ) -> _RequestState | None:
        request_state = stage.requests.get(request_id)
        if request_state is None:
            self._remember_tombstone(
                stage,
                request_id=request_id,
                now_ns=now_ns,
                reason=reason,
                highest_sequence_no=-1,
            )
            return None
        # Install the bounded fence before discarding any request-owned state.
        # If no slot is available, fail closed and retain the live state so a
        # stale identity can never be made executable by evicting an older
        # tombstone.
        self._remember_tombstone(
            stage,
            request_id=request_id,
            now_ns=now_ns,
            reason=reason,
            highest_sequence_no=request_state.highest_sequence_no,
        )
        for key in tuple(request_state.completed):
            self._drop_completed(stage, request_id, key)
        if request_state.inflight:
            stage.inflight = max(0, stage.inflight - request_state.inflight)
        del stage.requests[request_id]
        return request_state

    def _sweep_stage(
        self, stage: _LoadedStage, now_ns: int
    ) -> LifecycleSweepResult:
        pruned = self._prune_tombstones(stage, now_ns)
        expired_requests = 0
        expired_execution_leases = 0
        for request_id, request_state in tuple(stage.requests.items()):
            if (
                request_state.inflight
                and request_state.execution_lease_expires_ns <= now_ns
            ):
                self._remove_request_state(
                    stage,
                    request_id,
                    now_ns=now_ns,
                    reason="execution_lease_expired",
                )
                expired_execution_leases += 1
                continue
            if (
                not request_state.inflight
                and request_state.last_activity_ns + stage.request_idle_timeout_ns
                <= now_ns
            ):
                self._remove_request_state(
                    stage,
                    request_id,
                    now_ns=now_ns,
                    reason="request_idle_expired",
                )
                expired_requests += 1
        stage.expired_requests += expired_requests
        stage.expired_execution_leases += expired_execution_leases
        return LifecycleSweepResult(
            expired_requests=expired_requests,
            expired_execution_leases=expired_execution_leases,
            pruned_tombstones=pruned,
            live_requests=len(stage.requests),
            release_tombstones=len(stage.release_tombstones),
        )

    def sweep_expired(self, handle: StageHandle) -> LifecycleSweepResult:
        """Run the T0/T1 opportunistic lifecycle sweeper immediately."""

        with self._lock:
            stage = self._stage(handle)
            return self._sweep_stage(stage, self._clock_ns())

    def health(self, handle: StageHandle) -> StageHealth:
        with self._lock:
            stage = self._stage(handle)
            self._sweep_stage(stage, self._clock_ns())
            buffer_health = self._buffer_adapter.health()
            return StageHealth(
                state=stage.state,
                stage_id=stage.manifest.stage_id,
                manifest_hash=stage.manifest.manifest_hash,
                inflight=stage.inflight,
                degraded=stage.state == "FAILED",
                live_requests=len(stage.requests),
                completed_results=sum(
                    len(request.completed) for request in stage.requests.values()
                ),
                transform_cache_entries=len(stage.transform_cache),
                max_live_requests=stage.max_live_requests,
                max_completed_results_per_request=(
                    stage.max_completed_results_per_request
                ),
                max_transform_cache_entries=stage.max_transform_cache_entries,
                completed_result_bytes=stage.completed_result_bytes,
                completed_result_high_water_bytes=(
                    stage.completed_result_high_water_bytes
                ),
                transform_cache_bytes=stage.transform_cache_bytes,
                transform_cache_high_water_bytes=(
                    stage.transform_cache_high_water_bytes
                ),
                max_completed_result_bytes=stage.max_completed_result_bytes,
                max_transform_cache_bytes=stage.max_transform_cache_bytes,
                release_tombstones=len(stage.release_tombstones),
                max_release_tombstones=stage.max_release_tombstones,
                request_idle_timeout_ns=stage.request_idle_timeout_ns,
                execution_lease_timeout_ns=stage.execution_lease_timeout_ns,
                release_tombstone_ttl_ns=stage.release_tombstone_ttl_ns,
                expired_requests=stage.expired_requests,
                expired_execution_leases=stage.expired_execution_leases,
                native_buffer_imports=buffer_health.inflight_imports,
                native_buffer_bytes=buffer_health.inflight_bytes,
                native_buffer_high_water_bytes=buffer_health.high_water_bytes,
                max_native_buffer_bytes=buffer_health.max_imported_bytes,
                native_buffer_copy_operations=buffer_health.copy_operations,
            )

    @staticmethod
    def _request_epoch(stage: _LoadedStage, request_id: str) -> int:
        state = stage.requests.get(request_id)
        return 0 if state is None else state.kv_epoch

    @staticmethod
    def _tensor_digest(tensor: Tensor) -> str:
        return hashlib.sha256(
            canonical_json_bytes(tensor.descriptor.to_dict()) + tensor.payload
        ).hexdigest()

    @classmethod
    def _request_key(cls, request: StageRequest) -> RequestReplayKey:
        return (
            request.microbatch_id,
            request.sequence_no,
            request.phase,
            request.token_start,
            request.token_count,
            request.kv_epoch,
            cls._tensor_digest(request.input_activation),
        )

    def _admit_request(
        self, stage: _LoadedStage, request_id: str, now_ns: int
    ) -> _RequestState | None:
        state = stage.requests.get(request_id)
        if state is not None:
            state.last_activity_ns = now_ns
            return state
        if len(stage.requests) >= stage.max_live_requests:
            return None
        state = _RequestState(last_activity_ns=now_ns)
        stage.requests[request_id] = state
        return state

    @classmethod
    def _result_retained_bytes(cls, result: StageResult) -> int:
        payload_bytes = (
            len(result.output_tensor.payload)
            if result.output_tensor is not None
            else 0
        )
        return payload_bytes + cls._RESULT_OVERHEAD_BYTES

    @classmethod
    def _transform_retained_bytes(cls, tensor: Tensor) -> int:
        return len(tensor.payload) + cls._TRANSFORM_OVERHEAD_BYTES

    @staticmethod
    def _completed_lru_key(
        request_id: str, key: RequestReplayKey
    ) -> tuple[str, RequestReplayKey]:
        return (request_id, key)

    def _touch_completed(
        self,
        stage: _LoadedStage,
        request_id: str,
        key: RequestReplayKey,
    ) -> None:
        state = stage.requests[request_id]
        state.completed.move_to_end(key)
        global_key = self._completed_lru_key(request_id, key)
        if global_key in stage.completed_lru:
            stage.completed_lru.move_to_end(global_key)

    def _drop_completed(
        self,
        stage: _LoadedStage,
        request_id: str,
        key: RequestReplayKey,
    ) -> None:
        state = stage.requests.get(request_id)
        if state is not None:
            state.completed.pop(key, None)
        retained = stage.completed_lru.pop(
            self._completed_lru_key(request_id, key), 0
        )
        stage.completed_result_bytes -= retained

    def _remember_completed(
        self,
        stage: _LoadedStage,
        request_id: str,
        key: RequestReplayKey,
        result: StageResult,
    ) -> None:
        state = stage.requests[request_id]
        if key in state.completed:
            self._drop_completed(stage, request_id, key)
        retained = self._result_retained_bytes(result)
        state.completed[key] = result
        global_key = self._completed_lru_key(request_id, key)
        stage.completed_lru[global_key] = retained
        stage.completed_result_bytes += retained
        while len(state.completed) > stage.max_completed_results_per_request:
            oldest_key = next(iter(state.completed))
            self._drop_completed(stage, request_id, oldest_key)
        while stage.completed_result_bytes > stage.max_completed_result_bytes:
            oldest_request_id, oldest_key = next(iter(stage.completed_lru))
            self._drop_completed(stage, oldest_request_id, oldest_key)
        stage.completed_result_high_water_bytes = max(
            stage.completed_result_high_water_bytes,
            stage.completed_result_bytes,
        )

    def _remember_transform(
        self,
        stage: _LoadedStage,
        cache_key: str,
        tensor: Tensor,
    ) -> None:
        previous = stage.transform_cache.pop(cache_key, None)
        if previous is not None:
            stage.transform_cache_bytes -= self._transform_retained_bytes(previous)
        stage.transform_cache[cache_key] = tensor
        stage.transform_cache_bytes += self._transform_retained_bytes(tensor)
        while (
            len(stage.transform_cache) > stage.max_transform_cache_entries
            or stage.transform_cache_bytes > stage.max_transform_cache_bytes
        ):
            _, evicted = stage.transform_cache.popitem(last=False)
            stage.transform_cache_bytes -= self._transform_retained_bytes(evicted)
        stage.transform_cache_high_water_bytes = max(
            stage.transform_cache_high_water_bytes,
            stage.transform_cache_bytes,
        )

    def _transform_tensor(
        self,
        manifest: StageManifest,
        request: StageRequest,
    ) -> Tensor:
        descriptor = request.input_activation.descriptor
        expected_input = TensorDescriptor(
            kind="activation",
            dtype=str(manifest.input_contract["dtype"]),
            shape=descriptor.shape,
            layout=str(manifest.input_contract["layout"]),
        )
        imported_input = self._buffer_adapter.import_tensor(
            request.input_activation,
            expected=expected_input,
            purpose="stage-input",
        )
        try:
            oracle_input = self._buffer_adapter.export_tensor(
                imported_input,
                expected=expected_input,
            )
        finally:
            self._buffer_adapter.release(imported_input)
        values = deterministic_stage_values(
            oracle_input.values(),
            rows=descriptor.shape[0],
            width=descriptor.shape[1],
            layer_start=manifest.layer_start,
            layer_end=manifest.layer_end,
        )
        oracle_output = Tensor.from_values(
            values,
            kind=str(manifest.output_contract.get("kind", "activation")),
            dtype=str(manifest.output_contract["dtype"]),
            shape=descriptor.shape,
        )
        imported_output = self._buffer_adapter.import_tensor(
            oracle_output,
            expected=oracle_output.descriptor,
            purpose="stage-output",
        )
        try:
            return self._buffer_adapter.export_tensor(
                imported_output,
                expected=oracle_output.descriptor,
            )
        finally:
            self._buffer_adapter.release(imported_output)

    def _failure_result(
        self,
        request: StageRequest,
        *,
        status: str,
        code: str,
        message: str,
        kv_before: int,
        kv_after: int | None = None,
    ) -> StageResult:
        return StageResult(
            plan_id=request.plan_id,
            plan_hash=request.plan_hash,
            request_id=request.request_id,
            microbatch_id=request.microbatch_id,
            sequence_no=request.sequence_no,
            status=status,
            output_kind=None,
            output_tensor=None,
            kv_epoch_before=kv_before,
            kv_epoch_after=kv_before if kv_after is None else kv_after,
            timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
            error={"code": code, "message": message[:256]},
        )

    def _finalize_execution_result(
        self,
        request: StageRequest,
        result: StageResult,
    ) -> StageResult:
        """Decorate a newly executed result before it becomes replay-visible."""

        return result

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult:
        with self._lock:
            stage = self._stage(handle)
            now_ns = self._clock_ns()
            self._sweep_stage(stage, now_ns)
            manifest = stage.manifest
            current_epoch = self._request_epoch(stage, request.request_id)
            if stage.state != "READY":
                return self._failure_result(
                    request,
                    status="rejected",
                    code="EXECUTION",
                    message=f"stage is {stage.state.lower()}",
                    kv_before=current_epoch,
                )
            if request.plan_id != manifest.plan_id or request.plan_hash != manifest.plan_hash:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="STALE_PLAN",
                    message="request plan does not match loaded manifest",
                    kv_before=current_epoch,
                )
            if request.input_activation.descriptor.dtype != manifest.input_contract["dtype"]:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="TENSOR_CONTRACT",
                    message="input dtype does not match manifest",
                    kv_before=current_epoch,
                )
            if request.input_activation.descriptor.shape[1] != manifest.input_contract["hidden_size"]:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="TENSOR_CONTRACT",
                    message="input hidden size does not match manifest",
                    kv_before=current_epoch,
                )
            tombstone = stage.release_tombstones.get(request.request_id)
            if tombstone is not None:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="REQUEST_TOMBSTONED",
                    message=(
                        "request identity is fenced by a bounded release "
                        f"tombstone ({tombstone.reason})"
                    ),
                    kv_before=0,
                )
            key = self._request_key(request)
            request_state = stage.requests.get(request.request_id)
            if request_state is not None:
                cached = request_state.completed.get(key)
                if cached is not None:
                    request_state.last_activity_ns = now_ns
                    self._touch_completed(
                        stage, request.request_id, key
                    )
                    return cached
                if request.sequence_no <= request_state.highest_sequence_no:
                    raise StageRuntimeError(
                        "SEQUENCE",
                        "request result is outside the bounded replay window",
                    )
            if request.kv_epoch != current_epoch:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="STALE_PLAN",
                    message="KV epoch does not match stage ownership",
                    kv_before=current_epoch,
                )
            if request_state is not None and request_state.cancelled:
                return self._failure_result(
                    request,
                    status="cancelled",
                    code="CANCELLED",
                    message="request was cancelled before execution",
                    kv_before=current_epoch,
                )
            if now_ns >= request.deadline_ns:
                return self._failure_result(
                    request,
                    status="deadline",
                    code="DEADLINE",
                    message="deadline expired before execution",
                    kv_before=current_epoch,
                )
            if request_state is None:
                request_state = self._admit_request(
                    stage, request.request_id, now_ns
                )
                if request_state is None:
                    return self._failure_result(
                        request,
                        status="rejected",
                        code="ADMISSION",
                        message="live request state capacity is exhausted",
                        kv_before=0,
                    )
            if request_state.inflight:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="ADMISSION",
                    message="request already has inflight stage work",
                    kv_before=request_state.kv_epoch,
                )
            stage.inflight += 1
            request_state.inflight += 1
            request_state.execution_started = True
            request_state.last_activity_ns = now_ns
            execution_lease_id = stage.next_execution_lease_id
            stage.next_execution_lease_id += 1
            request_state.execution_lease_id = execution_lease_id
            request_state.execution_lease_expires_ns = min(
                request.deadline_ns,
                now_ns + stage.execution_lease_timeout_ns,
            )

        started = self._clock_ns()
        try:
            output_kind = str(manifest.output_contract.get("kind", "activation"))
            cache_key = self._tensor_digest(request.input_activation)
            with self._lock:
                cached_output = stage.transform_cache.get(cache_key)
                if cached_output is not None:
                    stage.transform_cache.move_to_end(cache_key)
            if cached_output is None:
                output = self._transform_tensor(manifest, request)
                with self._lock:
                    self._remember_transform(stage, cache_key, output)
            else:
                output = cached_output
            finished = self._clock_ns()
            result = StageResult(
                plan_id=request.plan_id,
                plan_hash=request.plan_hash,
                request_id=request.request_id,
                microbatch_id=request.microbatch_id,
                sequence_no=request.sequence_no,
                status="ok",
                output_kind=output_kind,
                output_tensor=output,
                kv_epoch_before=current_epoch,
                kv_epoch_after=current_epoch + 1,
                timings_ns={
                    "queue": 0,
                    "execute": max(0, finished - started),
                    "pack": 0,
                    "unpack": 0,
                },
            )
            result = self._finalize_execution_result(request, result)
        except StageRuntimeError as exc:
            result = self._failure_result(
                request,
                status="failed",
                code=exc.code,
                message=exc.message,
                kv_before=current_epoch,
            )
        except Exception as exc:  # noqa: BLE001 - backend converts to stable error.
            result = self._failure_result(
                request,
                status="failed",
                code="EXECUTION",
                message=str(exc),
                kv_before=current_epoch,
            )

        with self._lock:
            stage = self._loaded.get(handle.handle_id)
            if stage is None or stage.manifest.manifest_hash != handle.manifest_hash:
                return self._failure_result(
                    request,
                    status="failed",
                    code="LEASE_EXPIRED",
                    message="stage unloaded before execution lease committed",
                    kv_before=current_epoch,
                )
            finished_ns = self._clock_ns()
            self._sweep_stage(stage, finished_ns)
            active_state = stage.requests.get(request.request_id)
            if (
                active_state is not request_state
                or active_state.execution_lease_id != execution_lease_id
            ):
                return self._failure_result(
                    request,
                    status="failed",
                    code="LEASE_EXPIRED",
                    message="execution completed after its lease was fenced",
                    kv_before=current_epoch,
                )
            stage.inflight -= 1
            request_state.inflight -= 1
            request_state.execution_lease_id = None
            request_state.execution_lease_expires_ns = 0
            request_state.last_activity_ns = finished_ns
            if result.kv_epoch_after > request_state.kv_epoch:
                request_state.kv_epoch = result.kv_epoch_after
            request_state.highest_sequence_no = max(
                request_state.highest_sequence_no, request.sequence_no
            )
            self._remember_completed(
                stage, request.request_id, key, result
            )
            return result

    def cancel(self, handle: StageHandle, request_id: str, reason: str) -> CancelResult:
        _require_uuid(request_id, "request_id")
        _require_non_empty(reason, "reason")
        with self._lock:
            stage = self._stage(handle)
            now_ns = self._clock_ns()
            self._sweep_stage(stage, now_ns)
            if request_id in stage.release_tombstones:
                raise StageRuntimeError(
                    "REQUEST_TOMBSTONED",
                    "cannot cancel a request fenced by a release tombstone",
                )
            request_state = self._admit_request(stage, request_id, now_ns)
            if request_state is None:
                raise StageRuntimeError(
                    "ADMISSION", "live request state capacity is exhausted"
                )
            started = request_state.execution_started
            kv_mutated = request_state.kv_epoch > 0
            request_state.cancelled = True
            request_state.last_activity_ns = now_ns
            return CancelResult(
                request_id=request_id,
                cancelled=not kv_mutated,
                execution_started=started,
                kv_mutated=kv_mutated,
            )

    def release(self, handle: StageHandle, request_id: str) -> ReleaseResult:
        _require_uuid(request_id, "request_id")
        with self._lock:
            stage = self._stage(handle)
            now_ns = self._clock_ns()
            self._sweep_stage(stage, now_ns)
            request_state = stage.requests.get(request_id)
            if request_state is None:
                existing = stage.release_tombstones.get(request_id)
                reason = (
                    existing.reason
                    if existing is not None
                    else "explicit_release_absent"
                )
                if existing is None:
                    self._remember_tombstone(
                        stage,
                        request_id=request_id,
                        now_ns=now_ns,
                        reason=reason,
                        highest_sequence_no=-1,
                    )
                return ReleaseResult(
                    request_id=request_id,
                    released=False,
                    kv_state_released=0,
                    cancel_state_released=0,
                    execution_state_released=0,
                    idempotency_results_released=0,
                    tombstoned=True,
                    reason=reason,
                )
            if request_state.inflight:
                raise StageRuntimeError(
                    "REQUEST_INFLIGHT",
                    "cannot release request state while stage work is inflight",
                )
            result = ReleaseResult(
                request_id=request_id,
                released=True,
                kv_state_released=int(request_state.kv_epoch > 0),
                cancel_state_released=int(request_state.cancelled),
                execution_state_released=int(request_state.execution_started),
                idempotency_results_released=len(request_state.completed),
                tombstoned=True,
                reason="explicit_release",
            )
            self._remove_request_state(
                stage,
                request_id,
                now_ns=now_ns,
                reason="explicit_release",
            )
            return result

    def drain(self, handle: StageHandle, deadline_ns: int) -> DrainResult:
        with self._lock:
            stage = self._stage(handle)
            self._sweep_stage(stage, self._clock_ns())
            stage.state = "DRAINING"
            drained = stage.inflight == 0
            if not drained and self._clock_ns() >= deadline_ns:
                return DrainResult(drained=False, inflight=stage.inflight)
            return DrainResult(drained=drained, inflight=stage.inflight)

    def unload(self, handle: StageHandle) -> UnloadResult:
        with self._lock:
            stage = self._stage(handle)
            if stage.inflight:
                raise StageRuntimeError("EXECUTION", "cannot unload with inflight work")
            del self._loaded[handle.handle_id]
            return UnloadResult(unloaded=True)


@dataclass(frozen=True)
class SimulationProfile:
    scenario_id: str
    stage_service_ns: int = 1_000_000
    pack_ns: int = 10_000
    unpack_ns: int = 10_000
    jitter_fraction: float = 0.0
    seed: int = 0
    fault: str = "none"
    memory_limit_bytes: int = 1 << 30
    build_id: str = "simulated-max-v1"
    device_identity: str | None = None
    supported_dtypes: tuple[str, ...] = ("bf16", "fp16", "fp32")
    supported_operations: tuple[str, ...] = ("stage_execute",)
    supported_quantizations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.scenario_id, "scenario_id")
        if self.stage_service_ns < 0 or self.pack_ns < 0 or self.unpack_ns < 0:
            raise ValueError("simulated timings must be non-negative")
        if not 0.0 <= self.jitter_fraction <= 1.0:
            raise ValueError("jitter_fraction must be 0..1")
        if self.fault not in {
            "none",
            "slow_stage",
            "disconnect",
            "corruption",
            "timeout",
            "cancel",
        }:
            raise ValueError(f"unsupported simulated fault: {self.fault}")
        if self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        _require_non_empty(self.build_id, "build_id")
        if self.device_identity is not None:
            _require_non_empty(self.device_identity, "device_identity")
        if not self.supported_dtypes:
            raise ValueError("supported_dtypes must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "stage_service_ns": self.stage_service_ns,
            "pack_ns": self.pack_ns,
            "unpack_ns": self.unpack_ns,
            "jitter_fraction": self.jitter_fraction,
            "seed": self.seed,
            "fault": self.fault,
            "memory_limit_bytes": self.memory_limit_bytes,
            "build_id": self.build_id,
            "device_identity": self.device_identity,
            "supported_dtypes": list(self.supported_dtypes),
            "supported_operations": list(self.supported_operations),
            "supported_quantizations": list(self.supported_quantizations),
        }


class SimulatedMaxStageBackend(ReferenceStageBackend):
    backend_name = "simulated-max"

    def __init__(
        self,
        profile: SimulationProfile,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_live_requests: int = 4096,
        max_completed_results_per_request: int = 64,
        max_transform_cache_entries: int = 256,
        max_completed_result_bytes: int = 64 * 1024 * 1024,
        max_transform_cache_bytes: int = 64 * 1024 * 1024,
        max_release_tombstones: int = 8192,
        request_idle_timeout_ns: int = 15 * 60 * 1_000_000_000,
        execution_lease_timeout_ns: int = 5 * 60 * 1_000_000_000,
        release_tombstone_ttl_ns: int = 30 * 60 * 1_000_000_000,
        max_native_buffer_bytes: int = 256 * 1024 * 1024,
        buffer_adapter: TensorBufferAdapter | None = None,
    ) -> None:
        super().__init__(
            clock_ns=clock_ns,
            max_live_requests=max_live_requests,
            max_completed_results_per_request=max_completed_results_per_request,
            max_transform_cache_entries=max_transform_cache_entries,
            max_completed_result_bytes=max_completed_result_bytes,
            max_transform_cache_bytes=max_transform_cache_bytes,
            max_release_tombstones=max_release_tombstones,
            request_idle_timeout_ns=request_idle_timeout_ns,
            execution_lease_timeout_ns=execution_lease_timeout_ns,
            release_tombstone_ttl_ns=release_tombstone_ttl_ns,
            max_native_buffer_bytes=max_native_buffer_bytes,
            buffer_adapter=buffer_adapter,
        )
        self.profile = profile
        self._random = random.Random(profile.seed)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            build_id=self.profile.build_id,
            device_identity=self.profile.device_identity,
            supported_dtypes=tuple(self.profile.supported_dtypes),
            memory_bytes=self.profile.memory_limit_bytes,
            supported_operations=tuple(self.profile.supported_operations),
            supported_quantizations=tuple(self.profile.supported_quantizations),
        )

    def _finalize_execution_result(
        self,
        request: StageRequest,
        result: StageResult,
    ) -> StageResult:
        if result.status != "ok":
            return result
        jitter = 1.0 + self._random.uniform(
            -self.profile.jitter_fraction, self.profile.jitter_fraction
        )
        service_ns = int(self.profile.stage_service_ns * jitter)
        if self.profile.fault == "slow_stage":
            service_ns *= 4
        timings = dict(result.timings_ns)
        timings.update(
            {
                "execute": service_ns,
                "pack": self.profile.pack_ns,
                "unpack": self.profile.unpack_ns,
            }
        )
        output = result.output_tensor
        error = None
        status = result.status
        if self.profile.fault == "corruption":
            output = None
            status = "failed"
            error = {"code": "CHECKSUM", "message": "simulated output corruption"}
        return StageResult(
            plan_id=result.plan_id,
            plan_hash=result.plan_hash,
            request_id=result.request_id,
            microbatch_id=result.microbatch_id,
            sequence_no=result.sequence_no,
            status=status,
            output_kind=result.output_kind if status == "ok" else None,
            output_tensor=output,
            kv_epoch_before=result.kv_epoch_before,
            kv_epoch_after=result.kv_epoch_after,
            timings_ns=timings,
            error=error,
        )

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult:
        with self._lock:
            stage = self._stage(handle)
            now_ns = self._clock_ns()
            self._sweep_stage(stage, now_ns)
            current_epoch = self._request_epoch(stage, request.request_id)
            request_state = stage.requests.get(request.request_id)
            if request_state is not None:
                key = self._request_key(request)
                cached = request_state.completed.get(key)
                if cached is not None:
                    request_state.last_activity_ns = now_ns
                    self._touch_completed(stage, request.request_id, key)
                    return cached
        if request.input_activation.descriptor.payload_bytes > self.profile.memory_limit_bytes:
            return self._failure_result(
                request,
                status="rejected",
                code="TENSOR_CONTRACT",
                message="simulated memory limit exceeded",
                kv_before=current_epoch,
            )
        if self.profile.fault == "disconnect":
            raise StageRuntimeError("EXECUTION", "simulated channel disconnect")
        if self.profile.fault == "cancel":
            self.cancel(handle, request.request_id, "simulated cancellation")
        if self.profile.fault == "timeout":
            return self._failure_result(
                request,
                status="deadline",
                code="DEADLINE",
                message="simulated execution deadline",
                kv_before=current_epoch,
            )

        return super().execute(handle, request)


class MaxStageBackend:
    """Explicit physical-backend availability boundary for Phase 0.5.

    The class exists so orchestration can name the third backend without silently
    falling back to simulation. Physical execution is unavailable until a MAX
    adapter is supplied and passes the common conformance suite.
    """

    backend_name = "max"

    def __init__(self, adapter: StageExecutable | None = None) -> None:
        self._adapter = adapter

    @property
    def available(self) -> bool:
        return self._adapter is not None

    def _require(self) -> StageExecutable:
        if self._adapter is None:
            raise StageRuntimeError(
                "EXECUTION",
                "physical MAX StageExecutable adapter is unavailable",
            )
        return self._adapter

    def capabilities(self) -> BackendCapabilities:
        return self._require().capabilities()

    def load(self, manifest: StageManifest) -> StageHandle:
        return self._require().load(manifest)

    def health(self, handle: StageHandle) -> StageHealth:
        return self._require().health(handle)

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult:
        return self._require().execute(handle, request)

    def cancel(self, handle: StageHandle, request_id: str, reason: str) -> CancelResult:
        return self._require().cancel(handle, request_id, reason)

    def release(self, handle: StageHandle, request_id: str) -> ReleaseResult:
        return self._require().release(handle, request_id)

    def drain(self, handle: StageHandle, deadline_ns: int) -> DrainResult:
        return self._require().drain(handle, deadline_ns)

    def unload(self, handle: StageHandle) -> UnloadResult:
        return self._require().unload(handle)


def create_stage_backend(spec: StageBackendSpec) -> StageExecutable:
    """Construct a backend from a process-safe specification."""

    if spec.kind == "reference":
        if spec.options:
            raise ValueError("reference backend does not accept serialized options")
        return ReferenceStageBackend()
    if spec.kind == "simulated-max":
        return SimulatedMaxStageBackend(SimulationProfile(**spec.options))

    assert spec.kind == "max" and spec.factory is not None
    module_name, factory_name = spec.factory.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise StageRuntimeError(
            "EXECUTION", f"cannot import MAX backend factory {spec.factory}"
        ) from exc
    backend = factory(dict(spec.options))
    if not isinstance(backend, StageExecutable):
        required_methods = (
            "capabilities",
            "load",
            "health",
            "execute",
            "cancel",
            "release",
            "drain",
            "unload",
        )
        missing = [
            name for name in required_methods if not callable(getattr(backend, name, None))
        ]
        detail = (
            f"; missing required methods: {', '.join(missing)}"
            if missing
            else ""
        )
        raise StageRuntimeError(
            "EXECUTION",
            f"MAX backend factory {spec.factory} returned an invalid StageExecutable"
            f" adapter{detail}",
        )
    return backend


def attest_backend_capabilities(
    backend: StageExecutable,
    manifest: StageManifest,
) -> dict[str, Any]:
    """Compare backend-originated facts with manifest requirements before load."""

    observed = backend.capabilities()
    requirement = manifest.device_requirement
    requested_dtypes = {
        str(manifest.input_contract["dtype"]),
        str(manifest.output_contract["dtype"]),
        *(str(value) for value in requirement.get("dtypes", [])),
    }
    requested_operations = tuple(
        sorted(str(value) for value in requirement.get("operations", []))
    )
    requested_quantization = requirement.get("quantization")
    requested = {
        "backend": requirement.get("backend"),
        "build_id": manifest.max_build_id,
        "device_identity": requirement.get("device_identity"),
        "dtypes": sorted(requested_dtypes),
        "abi_major": manifest.fornax_abi_major,
        "abi_minor": manifest.fornax_abi_minor,
        "minimum_memory_bytes": requirement.get("minimum_memory_bytes"),
        "operations": list(requested_operations),
        "quantization": requested_quantization,
    }
    mismatches: list[str] = []
    if requested["backend"] and requested["backend"] != observed.backend:
        mismatches.append(
            f"backend requested={requested['backend']} observed={observed.backend}"
        )
    if requested["build_id"] != observed.build_id:
        mismatches.append(
            f"build_id requested={requested['build_id']} observed={observed.build_id}"
        )
    if requested["device_identity"] != observed.device_identity:
        mismatches.append(
            "device_identity requested="
            f"{requested['device_identity']} observed={observed.device_identity}"
        )
    missing_dtypes = sorted(requested_dtypes - set(observed.supported_dtypes))
    if missing_dtypes:
        mismatches.append(f"unsupported dtypes={missing_dtypes}")
    if (
        manifest.fornax_abi_major,
        manifest.fornax_abi_minor,
    ) not in observed.abi_versions:
        mismatches.append(
            f"ABI {manifest.fornax_abi_major}.{manifest.fornax_abi_minor} unsupported"
        )
    minimum_memory = requested["minimum_memory_bytes"]
    if minimum_memory is not None:
        if isinstance(minimum_memory, bool) or not isinstance(minimum_memory, int):
            mismatches.append("minimum_memory_bytes is invalid")
        elif observed.memory_bytes is None:
            mismatches.append("backend did not attest memory_bytes")
        elif observed.memory_bytes < minimum_memory:
            mismatches.append(
                f"memory requested={minimum_memory} observed={observed.memory_bytes}"
            )
    missing_operations = sorted(
        set(requested_operations) - set(observed.supported_operations)
    )
    if missing_operations:
        mismatches.append(f"unsupported operations={missing_operations}")
    if (
        requested_quantization is not None
        and str(requested_quantization) not in observed.supported_quantizations
    ):
        mismatches.append(f"unsupported quantization={requested_quantization}")

    attestation = {
        "schema_version": 1,
        "checked_before_load": True,
        "manifest_hash": manifest.manifest_hash,
        "compatible": not mismatches,
        "observed": observed.to_dict(),
        "requested": requested,
        "mismatches": mismatches,
    }
    if mismatches:
        raise StageRuntimeError(
            "CAPABILITY_MISMATCH",
            "backend capability attestation failed: " + "; ".join(mismatches),
        )
    return attestation
