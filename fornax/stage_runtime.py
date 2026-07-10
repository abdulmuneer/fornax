from __future__ import annotations

import hashlib
import json
import math
import random
import re
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SUPPORTED_DTYPES = {"bf16", "fp16", "fp32"}
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
TERMINAL_STATUSES = {"ok", "cancelled", "deadline", "rejected", "failed"}


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
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
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
class DrainResult:
    drained: bool
    inflight: int


@dataclass(frozen=True)
class UnloadResult:
    unloaded: bool


@runtime_checkable
class StageExecutable(Protocol):
    def load(self, manifest: StageManifest) -> StageHandle: ...

    def health(self, handle: StageHandle) -> StageHealth: ...

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult: ...

    def cancel(self, handle: StageHandle, request_id: str, reason: str) -> CancelResult: ...

    def drain(self, handle: StageHandle, deadline_ns: int) -> DrainResult: ...

    def unload(self, handle: StageHandle) -> UnloadResult: ...


@dataclass
class _LoadedStage:
    manifest: StageManifest
    state: str = "READY"
    inflight: int = 0
    kv_epochs: dict[str, int] = field(default_factory=dict)
    cancelled: set[str] = field(default_factory=set)
    completed: dict[tuple[str, str, int, str], StageResult] = field(default_factory=dict)
    execution_started: set[str] = field(default_factory=set)
    transform_cache: dict[str, Tensor] = field(default_factory=dict)


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

    def __init__(self, *, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._loaded: dict[str, _LoadedStage] = {}
        self._lock = threading.RLock()

    def load(self, manifest: StageManifest) -> StageHandle:
        handle = StageHandle(
            handle_id=f"{manifest.stage_id}:{manifest.manifest_hash[7:23]}",
            stage_id=manifest.stage_id,
            manifest_hash=manifest.manifest_hash,
        )
        with self._lock:
            if handle.handle_id in self._loaded:
                raise StageRuntimeError("STALE_PLAN", "manifest is already loaded")
            self._loaded[handle.handle_id] = _LoadedStage(manifest=manifest)
        return handle

    def _stage(self, handle: StageHandle) -> _LoadedStage:
        stage = self._loaded.get(handle.handle_id)
        if stage is None or stage.manifest.manifest_hash != handle.manifest_hash:
            raise StageRuntimeError("STALE_PLAN", "unknown or stale stage handle")
        return stage

    def health(self, handle: StageHandle) -> StageHealth:
        with self._lock:
            stage = self._stage(handle)
            return StageHealth(
                state=stage.state,
                stage_id=stage.manifest.stage_id,
                manifest_hash=stage.manifest.manifest_hash,
                inflight=stage.inflight,
                degraded=stage.state == "FAILED",
            )

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

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult:
        with self._lock:
            stage = self._stage(handle)
            manifest = stage.manifest
            if stage.state != "READY":
                return self._failure_result(
                    request,
                    status="rejected",
                    code="EXECUTION",
                    message=f"stage is {stage.state.lower()}",
                    kv_before=stage.kv_epochs.get(request.request_id, 0),
                )
            if request.plan_id != manifest.plan_id or request.plan_hash != manifest.plan_hash:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="STALE_PLAN",
                    message="request plan does not match loaded manifest",
                    kv_before=stage.kv_epochs.get(request.request_id, 0),
                )
            if request.input_activation.descriptor.dtype != manifest.input_contract["dtype"]:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="TENSOR_CONTRACT",
                    message="input dtype does not match manifest",
                    kv_before=stage.kv_epochs.get(request.request_id, 0),
                )
            if request.input_activation.descriptor.shape[1] != manifest.input_contract["hidden_size"]:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="TENSOR_CONTRACT",
                    message="input hidden size does not match manifest",
                    kv_before=stage.kv_epochs.get(request.request_id, 0),
                )
            key = (
                request.request_id,
                request.microbatch_id,
                request.sequence_no,
                request.phase,
            )
            if key in stage.completed:
                return stage.completed[key]
            current_epoch = stage.kv_epochs.get(request.request_id, 0)
            if request.kv_epoch != current_epoch:
                return self._failure_result(
                    request,
                    status="rejected",
                    code="STALE_PLAN",
                    message="KV epoch does not match stage ownership",
                    kv_before=current_epoch,
                )
            if request.request_id in stage.cancelled:
                return self._failure_result(
                    request,
                    status="cancelled",
                    code="CANCELLED",
                    message="request was cancelled before execution",
                    kv_before=current_epoch,
                )
            if self._clock_ns() >= request.deadline_ns:
                return self._failure_result(
                    request,
                    status="deadline",
                    code="DEADLINE",
                    message="deadline expired before execution",
                    kv_before=current_epoch,
                )
            stage.inflight += 1
            stage.execution_started.add(request.request_id)

        started = self._clock_ns()
        try:
            descriptor = request.input_activation.descriptor
            output_kind = str(manifest.output_contract.get("kind", "activation"))
            cache_key = hashlib.sha256(request.input_activation.payload).hexdigest()
            with self._lock:
                cached_output = stage.transform_cache.get(cache_key)
            if cached_output is None:
                values = deterministic_stage_values(
                    request.input_activation.values(),
                    rows=descriptor.shape[0],
                    width=descriptor.shape[1],
                    layer_start=manifest.layer_start,
                    layer_end=manifest.layer_end,
                )
                output = Tensor.from_values(
                    values,
                    kind=output_kind,
                    dtype=str(manifest.output_contract["dtype"]),
                    shape=descriptor.shape,
                )
                with self._lock:
                    stage.transform_cache[cache_key] = output
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
        except Exception as exc:  # noqa: BLE001 - backend converts to stable error.
            result = self._failure_result(
                request,
                status="failed",
                code="EXECUTION",
                message=str(exc),
                kv_before=current_epoch,
            )

        with self._lock:
            stage = self._stage(handle)
            stage.inflight -= 1
            if result.status == "ok":
                stage.kv_epochs[request.request_id] = result.kv_epoch_after
                stage.completed[key] = result
            return result

    def cancel(self, handle: StageHandle, request_id: str, reason: str) -> CancelResult:
        _require_uuid(request_id, "request_id")
        _require_non_empty(reason, "reason")
        with self._lock:
            stage = self._stage(handle)
            started = request_id in stage.execution_started
            kv_mutated = stage.kv_epochs.get(request_id, 0) > 0
            stage.cancelled.add(request_id)
            return CancelResult(
                request_id=request_id,
                cancelled=not kv_mutated,
                execution_started=started,
                kv_mutated=kv_mutated,
            )

    def drain(self, handle: StageHandle, deadline_ns: int) -> DrainResult:
        with self._lock:
            stage = self._stage(handle)
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


class SimulatedMaxStageBackend(ReferenceStageBackend):
    backend_name = "simulated-max"

    def __init__(
        self,
        profile: SimulationProfile,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__(clock_ns=clock_ns)
        self.profile = profile
        self._random = random.Random(profile.seed)

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult:
        stage = self._stage(handle)
        if request.input_activation.descriptor.payload_bytes > self.profile.memory_limit_bytes:
            return self._failure_result(
                request,
                status="rejected",
                code="TENSOR_CONTRACT",
                message="simulated memory limit exceeded",
                kv_before=stage.kv_epochs.get(request.request_id, 0),
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
                kv_before=stage.kv_epochs.get(request.request_id, 0),
            )

        result = super().execute(handle, request)
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

    def load(self, manifest: StageManifest) -> StageHandle:
        return self._require().load(manifest)

    def health(self, handle: StageHandle) -> StageHealth:
        return self._require().health(handle)

    def execute(self, handle: StageHandle, request: StageRequest) -> StageResult:
        return self._require().execute(handle, request)

    def cancel(self, handle: StageHandle, request_id: str, reason: str) -> CancelResult:
        return self._require().cancel(handle, request_id, reason)

    def drain(self, handle: StageHandle, deadline_ns: int) -> DrainResult:
        return self._require().drain(handle, deadline_ns)

    def unload(self, handle: StageHandle) -> UnloadResult:
        return self._require().unload(handle)
