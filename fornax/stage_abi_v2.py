"""FNX2 ragged-batch logical contract and strict wire codec.

This module is deliberately independent from :mod:`fornax.stage_abi`.  FNX1
bytes are frozen; ragged multi-sequence execution is a new exact ABI major.
The types here describe logical tensors and sequence ownership.  A physical
adapter may translate them to backend-native layouts, but may not expose those
layouts on the wire.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import struct
import uuid
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Iterable

from .stage_abi import crc32c
from .stage_runtime import HASH_RE, canonical_json_bytes, canonical_sha256


MAGIC = b"FNX2"
ABI_MAJOR = 2
ABI_MINOR = 0
PRELUDE = struct.Struct("!4sHHHHIQQII")
PRELUDE_BYTES = PRELUDE.size
MAX_METADATA_BYTES = 256 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
DTYPE_BYTES = {"int32": 4, "bf16": 2, "fp16": 2, "fp32": 4}
TERMINAL_STATUSES = {"ok", "cancelled", "deadline", "rejected", "failed"}
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class RaggedFrameError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        sequence_no: int | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.sequence_no = sequence_no
        self.recoverable = recoverable


class RaggedMessageKind(IntEnum):
    BATCH = 1
    RESULT = 2
    CREDIT = 3
    CANCEL = 4
    RELEASE = 5
    ACK = 6
    ERROR = 7
    CONTROL = 8


def _require_uuid(value: str, field_name: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be a lower-case canonical UUID")


def _require_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be sha256:<64 lowercase hex>")


def _require_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_negative(
    value: int, field_name: str, *, maximum: int = UINT64_MAX
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise ValueError(
            f"{field_name} must be a non-negative integer no greater than {maximum}"
        )


def _require_positive(
    value: int, field_name: str, *, maximum: int = UINT64_MAX
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise ValueError(
            f"{field_name} must be a positive integer no greater than {maximum}"
        )


def _require_object_fields(
    value: Any,
    field_name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    optional = optional or set()
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{field_name} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{field_name} has unknown fields: {unknown}")
    return value


def derive_execution_lease_id(
    plan_generation_id: str,
    plan_id: str,
    plan_hash: str,
    request_id: str,
) -> str:
    """Derive a lease that is deterministically scoped to one plan generation.

    The generation UUID is created when the worker route starts and is never
    reused across restarts. Released lease tombstones are retained for that
    generation, so bounded tombstone exhaustion fails closed instead of making
    an old lease admissible again. This is stale-generation fencing, not an
    authentication token or authorization capability: all derivation inputs are
    public.
    """

    _require_uuid(plan_generation_id, "plan_generation_id")
    _require_uuid(plan_id, "plan_id")
    _require_hash(plan_hash, "plan_hash")
    _require_uuid(request_id, "request_id")
    return str(
        uuid.uuid5(
            uuid.UUID(plan_generation_id),
            f"fnx2:{plan_id}:{plan_hash}:{request_id}",
        )
    )


def _float_to_bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def _bf16_to_float(data: bytes) -> float:
    high = struct.unpack("<H", data)[0]
    return struct.unpack("<f", struct.pack("<I", high << 16))[0]


def _encode_values(values: Iterable[int | float], dtype: str) -> bytes:
    if dtype == "int32":
        encoded: list[bytes] = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < INT32_MIN
                or value > INT32_MAX
            ):
                raise ValueError("int32 tensor values must be exact signed 32-bit integers")
            encoded.append(struct.pack("<i", value))
        return b"".join(encoded)
    checked: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("floating tensor values must be JSON numbers")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("tensor values must be finite")
        checked.append(converted)
    if dtype == "bf16":
        try:
            return b"".join(_float_to_bf16(value) for value in checked)
        except (OverflowError, struct.error) as exc:
            raise ValueError("bf16 tensor value is out of range") from exc
    if dtype == "fp16":
        try:
            return b"".join(struct.pack("<e", value) for value in checked)
        except (OverflowError, struct.error) as exc:
            raise ValueError("fp16 tensor value is out of range") from exc
    if dtype == "fp32":
        try:
            return b"".join(struct.pack("<f", value) for value in checked)
        except (OverflowError, struct.error) as exc:
            raise ValueError("fp32 tensor value is out of range") from exc
    raise ValueError(f"unsupported tensor dtype: {dtype}")


def _decode_values(payload: bytes, dtype: str) -> tuple[int | float, ...]:
    scalar_bytes = DTYPE_BYTES.get(dtype)
    if scalar_bytes is None:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    if len(payload) % scalar_bytes:
        raise ValueError("payload length is not divisible by scalar size")
    if dtype == "int32":
        return tuple(
            struct.unpack("<i", payload[offset : offset + 4])[0]
            for offset in range(0, len(payload), 4)
        )
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
class LogicalTensorDescriptor:
    kind: str
    dtype: str
    shape: tuple[int, ...]
    layout: str = "contiguous_row_major"
    logical_elements: int | None = None
    payload_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"token_ids", "activation", "logits"}:
            raise ValueError("tensor kind must be token_ids, activation, or logits")
        expected_dtype = "int32" if self.kind == "token_ids" else None
        if expected_dtype is not None and self.dtype != expected_dtype:
            raise ValueError("token_ids must use int32")
        if self.kind != "token_ids" and self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("activation/logits must use a floating dtype")
        expected_rank = 1 if self.kind == "token_ids" else 2
        if len(self.shape) != expected_rank or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item > UINT32_MAX
            for item in self.shape
        ):
            raise ValueError(
                f"{self.kind} tensor must have rank {expected_rank} and uint32 dimensions"
            )
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

    @property
    def row_count(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int | None:
        return None if self.kind == "token_ids" else self.shape[1]

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
    def from_dict(cls, value: dict[str, Any]) -> "LogicalTensorDescriptor":
        value = _require_object_fields(
            value,
            "tensor descriptor",
            required={
                "kind",
                "dtype",
                "shape",
                "layout",
                "logical_elements",
                "payload_bytes",
            },
        )
        if not isinstance(value["shape"], list):
            raise ValueError("tensor descriptor shape must be an array")
        return cls(
            kind=value["kind"],
            dtype=value["dtype"],
            shape=tuple(value["shape"]),
            layout=value["layout"],
            logical_elements=value["logical_elements"],
            payload_bytes=value["payload_bytes"],
        )


@dataclass(frozen=True)
class LogicalTensor:
    descriptor: LogicalTensorDescriptor
    payload: bytes

    def __post_init__(self) -> None:
        if len(self.payload) != self.descriptor.payload_bytes:
            raise ValueError("tensor payload length does not match descriptor")
        values = _decode_values(self.payload, self.descriptor.dtype)
        if self.descriptor.dtype != "int32" and any(
            not math.isfinite(float(value)) for value in values
        ):
            raise ValueError("tensor values must be finite")

    @classmethod
    def from_values(
        cls,
        values: Iterable[int | float],
        *,
        kind: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> "LogicalTensor":
        descriptor = LogicalTensorDescriptor(kind=kind, dtype=dtype, shape=shape)
        flat = tuple(values)
        if len(flat) != descriptor.logical_elements:
            raise ValueError("value count does not match tensor shape")
        return cls(descriptor, _encode_values(flat, dtype))

    def values(self) -> tuple[int | float, ...]:
        return _decode_values(self.payload, self.descriptor.dtype)

    def rows(self, start: int, count: int) -> "LogicalTensor":
        _require_non_negative(start, "row start")
        _require_positive(count, "row count")
        if start + count > self.descriptor.row_count:
            raise ValueError("row slice exceeds tensor")
        width = self.descriptor.width or 1
        values = self.values()[start * width : (start + count) * width]
        shape = (count,) if self.descriptor.kind == "token_ids" else (count, width)
        return LogicalTensor.from_values(
            values,
            kind=self.descriptor.kind,
            dtype=self.descriptor.dtype,
            shape=shape,
        )

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(
            canonical_json_bytes(self.descriptor.to_dict()) + self.payload
        ).hexdigest()


@dataclass(frozen=True)
class RaggedStageManifest:
    plan_id: str
    plan_hash: str
    stage_id: str
    stage_index: int
    layer_start: int
    layer_end: int
    stage_role: str
    input_kind: str
    output_kind: str
    dtype: str
    hidden_size: int
    vocabulary_size: int | None = None
    build_id: str = "fornax-reference-fnx2"
    abi_major: int = ABI_MAJOR
    abi_minor: int = ABI_MINOR

    def __post_init__(self) -> None:
        _require_uuid(self.plan_id, "plan_id")
        _require_hash(self.plan_hash, "plan_hash")
        _require_string(self.stage_id, "stage_id")
        _require_string(self.build_id, "build_id")
        if self.abi_major != ABI_MAJOR or self.abi_minor != ABI_MINOR:
            raise ValueError("ragged manifest requires exact ABI 2.0")
        if self.stage_role not in {"first", "middle", "final"}:
            raise ValueError("stage_role must be first, middle, or final")
        if self.input_kind not in {"token_ids", "activation"}:
            raise ValueError("input_kind must be token_ids or activation")
        if self.output_kind not in {"activation", "logits"}:
            raise ValueError("output_kind must be activation or logits")
        if self.stage_role != "first" and self.input_kind != "activation":
            raise ValueError("only a first stage may accept token_ids")
        if self.stage_role != "final" and self.output_kind != "activation":
            raise ValueError("only a final stage may emit logits")
        if self.stage_role == "final" and self.output_kind != "logits":
            raise ValueError("final stage must emit logits")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        _require_non_negative(self.stage_index, "stage_index")
        _require_non_negative(self.layer_start, "layer_start")
        if self.layer_end < self.layer_start:
            raise ValueError("layer_end must not precede layer_start")
        _require_positive(self.hidden_size, "hidden_size")
        if self.output_kind == "logits":
            if self.vocabulary_size is None:
                raise ValueError("final logits require vocabulary_size")
            _require_positive(self.vocabulary_size, "vocabulary_size")
        elif self.vocabulary_size is not None:
            raise ValueError("vocabulary_size is only valid for a logits stage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "stage_role": self.stage_role,
            "input_kind": self.input_kind,
            "output_kind": self.output_kind,
            "dtype": self.dtype,
            "hidden_size": self.hidden_size,
            "vocabulary_size": self.vocabulary_size,
            "build_id": self.build_id,
            "abi_major": self.abi_major,
            "abi_minor": self.abi_minor,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RaggedStageManifest":
        value = _require_object_fields(
            value,
            "ragged stage manifest",
            required={
                "plan_id",
                "plan_hash",
                "stage_id",
                "stage_index",
                "layer_start",
                "layer_end",
                "stage_role",
                "input_kind",
                "output_kind",
                "dtype",
                "hidden_size",
                "vocabulary_size",
                "build_id",
                "abi_major",
                "abi_minor",
            },
        )
        return cls(
            plan_id=value["plan_id"],
            plan_hash=value["plan_hash"],
            stage_id=value["stage_id"],
            stage_index=value["stage_index"],
            layer_start=value["layer_start"],
            layer_end=value["layer_end"],
            stage_role=value["stage_role"],
            input_kind=value["input_kind"],
            output_kind=value["output_kind"],
            dtype=value["dtype"],
            hidden_size=value["hidden_size"],
            vocabulary_size=value["vocabulary_size"],
            build_id=value["build_id"],
            abi_major=value["abi_major"],
            abi_minor=value["abi_minor"],
        )

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class SequenceSlice:
    request_id: str
    request_sequence_no: int
    input_row_start: int
    input_row_count: int
    token_position_start: int
    kv_epoch: int
    deadline_budget_ns: int
    execution_lease_id: str
    trace_id: str
    span_id: str

    def __post_init__(self) -> None:
        _require_uuid(self.request_id, "request_id")
        _require_uuid(self.execution_lease_id, "execution_lease_id")
        _require_non_negative(self.request_sequence_no, "request_sequence_no")
        _require_non_negative(
            self.input_row_start, "input_row_start", maximum=UINT32_MAX
        )
        _require_positive(self.input_row_count, "input_row_count", maximum=UINT32_MAX)
        _require_non_negative(self.token_position_start, "token_position_start")
        _require_non_negative(self.kv_epoch, "kv_epoch")
        # A monotonic timestamp is meaningful only inside the process which
        # sampled it.  FNX2 therefore carries a remaining budget.  Zero is a
        # valid, already-expired budget and lets an upstream stage fail closed
        # without inventing a timestamp in the receiver's clock domain.
        _require_non_negative(self.deadline_budget_ns, "deadline_budget_ns")
        _require_string(self.trace_id, "trace_id")
        _require_string(self.span_id, "span_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_sequence_no": self.request_sequence_no,
            "input_row_start": self.input_row_start,
            "input_row_count": self.input_row_count,
            "token_position_start": self.token_position_start,
            "kv_epoch": self.kv_epoch,
            "deadline_budget_ns": self.deadline_budget_ns,
            "execution_lease_id": self.execution_lease_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SequenceSlice":
        value = _require_object_fields(
            value,
            "sequence slice",
            required={
                "request_id",
                "request_sequence_no",
                "input_row_start",
                "input_row_count",
                "token_position_start",
                "kv_epoch",
                "deadline_budget_ns",
                "execution_lease_id",
                "trace_id",
                "span_id",
            },
        )
        return cls(
            request_id=value["request_id"],
            request_sequence_no=value["request_sequence_no"],
            input_row_start=value["input_row_start"],
            input_row_count=value["input_row_count"],
            token_position_start=value["token_position_start"],
            kv_epoch=value["kv_epoch"],
            deadline_budget_ns=value["deadline_budget_ns"],
            execution_lease_id=value["execution_lease_id"],
            trace_id=value["trace_id"],
            span_id=value["span_id"],
        )


@dataclass(frozen=True)
class BatchDescriptor:
    batch_id: str
    batch_sequence_no: int
    phase: str
    input_row_count: int
    sequences: tuple[SequenceSlice, ...]

    def __post_init__(self) -> None:
        _require_uuid(self.batch_id, "batch_id")
        _require_non_negative(self.batch_sequence_no, "batch_sequence_no")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        _require_positive(self.input_row_count, "input_row_count", maximum=UINT32_MAX)
        if not self.sequences:
            raise ValueError("sequences must not be empty")
        ordered = sorted(self.sequences, key=lambda item: item.input_row_start)
        if list(self.sequences) != ordered:
            raise ValueError("sequences must be sorted by input_row_start")
        cursor = 0
        identities: set[tuple[str, str]] = set()
        for item in self.sequences:
            if item.input_row_start != cursor:
                raise ValueError("sequence slices must cover input rows without gaps or overlap")
            cursor += item.input_row_count
            identity = (item.request_id, item.execution_lease_id)
            if identity in identities:
                raise ValueError("batch contains duplicate request/lease identity")
            identities.add(identity)
            if self.phase == "decode" and item.input_row_count != 1:
                raise ValueError("decode requires exactly one row per active sequence")
        if cursor != self.input_row_count:
            raise ValueError("sequence slices must cover every input row exactly once")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_sequence_no": self.batch_sequence_no,
            "phase": self.phase,
            "input_row_count": self.input_row_count,
            "sequences": [item.to_dict() for item in self.sequences],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BatchDescriptor":
        value = _require_object_fields(
            value,
            "batch descriptor",
            required={
                "batch_id",
                "batch_sequence_no",
                "phase",
                "input_row_count",
                "sequences",
            },
        )
        raw_sequences = value["sequences"]
        if not isinstance(raw_sequences, list):
            raise ValueError("batch sequences must be a list")
        return cls(
            batch_id=value["batch_id"],
            batch_sequence_no=value["batch_sequence_no"],
            phase=value["phase"],
            input_row_count=value["input_row_count"],
            sequences=tuple(SequenceSlice.from_dict(item) for item in raw_sequences),
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def with_stage_epochs(
        self, epochs: dict[tuple[str, str], int], *, stage_id: str
    ) -> "BatchDescriptor":
        return replace(
            self,
            sequences=tuple(
                replace(
                    item,
                    kv_epoch=epochs.get(
                        (stage_id, item.request_id, item.execution_lease_id), 0
                    ),
                )
                for item in self.sequences
            ),
        )


@dataclass(frozen=True)
class SequenceResult:
    request_id: str
    execution_lease_id: str
    request_sequence_no: int
    status: str
    input_row_start: int
    input_row_count: int
    output_row_start: int | None
    output_row_count: int
    kv_epoch_before: int
    kv_epoch_after: int
    error: dict[str, str] | None
    timings_ns: dict[str, int]

    def __post_init__(self) -> None:
        _require_uuid(self.request_id, "request_id")
        _require_uuid(self.execution_lease_id, "execution_lease_id")
        _require_non_negative(self.request_sequence_no, "request_sequence_no")
        if self.status not in TERMINAL_STATUSES:
            raise ValueError(f"unsupported sequence status: {self.status}")
        _require_non_negative(
            self.input_row_start, "input_row_start", maximum=UINT32_MAX
        )
        _require_positive(self.input_row_count, "input_row_count", maximum=UINT32_MAX)
        _require_non_negative(
            self.output_row_count, "output_row_count", maximum=UINT32_MAX
        )
        _require_non_negative(self.kv_epoch_before, "kv_epoch_before")
        _require_non_negative(self.kv_epoch_after, "kv_epoch_after")
        if self.kv_epoch_after < self.kv_epoch_before:
            raise ValueError("KV epoch cannot move backwards")
        if self.status == "ok":
            if self.output_row_start is None or self.output_row_count <= 0:
                raise ValueError("successful sequence requires output rows")
            _require_non_negative(
                self.output_row_start, "output_row_start", maximum=UINT32_MAX
            )
            if self.error is not None:
                raise ValueError("successful sequence cannot contain an error")
        else:
            if self.output_row_start is not None or self.output_row_count != 0:
                raise ValueError("failed sequence cannot contain output rows")
            if not isinstance(self.error, dict) or set(self.error) != {"code", "message"}:
                raise ValueError("failed sequence requires a typed error")
            _require_string(self.error.get("code"), "error.code")
            _require_string(self.error.get("message"), "error.message")
            if len(self.error["code"]) > 64 or len(self.error["message"]) > 256:
                raise ValueError("per-sequence error diagnostics exceed bounded size")
            if self.kv_epoch_after != self.kv_epoch_before:
                raise ValueError("failed sequence cannot mutate KV")
        required_timing = {"queue", "execute", "pack", "unpack"}
        if set(self.timings_ns) != required_timing or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.timings_ns.values()
        ):
            raise ValueError("timings_ns must contain non-negative queue/execute/pack/unpack")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "execution_lease_id": self.execution_lease_id,
            "request_sequence_no": self.request_sequence_no,
            "status": self.status,
            "input_row_start": self.input_row_start,
            "input_row_count": self.input_row_count,
            "output_row_start": self.output_row_start,
            "output_row_count": self.output_row_count,
            "kv_epoch_before": self.kv_epoch_before,
            "kv_epoch_after": self.kv_epoch_after,
            "error": self.error,
            "timings_ns": self.timings_ns,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SequenceResult":
        value = _require_object_fields(
            value,
            "sequence result",
            required={
                "request_id",
                "execution_lease_id",
                "request_sequence_no",
                "status",
                "input_row_start",
                "input_row_count",
                "output_row_start",
                "output_row_count",
                "kv_epoch_before",
                "kv_epoch_after",
                "error",
                "timings_ns",
            },
        )
        raw_error = value["error"]
        raw_timings = value["timings_ns"]
        if raw_error is not None and not isinstance(raw_error, dict):
            raise ValueError("sequence result error must be an object or null")
        if not isinstance(raw_timings, dict):
            raise ValueError("sequence result timings_ns must be an object")
        return cls(
            request_id=value["request_id"],
            execution_lease_id=value["execution_lease_id"],
            request_sequence_no=value["request_sequence_no"],
            status=value["status"],
            input_row_start=value["input_row_start"],
            input_row_count=value["input_row_count"],
            output_row_start=value["output_row_start"],
            output_row_count=value["output_row_count"],
            kv_epoch_before=value["kv_epoch_before"],
            kv_epoch_after=value["kv_epoch_after"],
            error=(dict(raw_error) if raw_error is not None else None),
            timings_ns=dict(raw_timings),
        )


@dataclass(frozen=True)
class RaggedBatchRequest:
    plan_id: str
    plan_hash: str
    manifest_hash: str
    descriptor: BatchDescriptor
    input_tensor: LogicalTensor

    def __post_init__(self) -> None:
        _require_uuid(self.plan_id, "plan_id")
        _require_hash(self.plan_hash, "plan_hash")
        _require_hash(self.manifest_hash, "manifest_hash")
        if self.input_tensor.descriptor.row_count != self.descriptor.input_row_count:
            raise ValueError("batch input_row_count must match tensor rows")


@dataclass(frozen=True)
class RaggedBatchResult:
    batch_id: str
    batch_sequence_no: int
    manifest_hash: str
    results: tuple[SequenceResult, ...]
    output_tensor: LogicalTensor | None

    def __post_init__(self) -> None:
        _require_uuid(self.batch_id, "batch_id")
        _require_non_negative(self.batch_sequence_no, "batch_sequence_no")
        _require_hash(self.manifest_hash, "manifest_hash")
        if not self.results:
            raise ValueError("batch result must contain sequence results")
        ordered = sorted(self.results, key=lambda item: (item.request_id, item.execution_lease_id))
        if list(self.results) != ordered:
            raise ValueError("sequence results must be in request identity order")
        cursor = 0
        for item in self.results:
            if item.status == "ok":
                if item.output_row_start != cursor:
                    raise ValueError("successful output ranges must be compact and contiguous")
                cursor += item.output_row_count
        if cursor == 0 and self.output_tensor is not None:
            raise ValueError("all-failed batch cannot contain output tensor")
        if cursor > 0:
            if self.output_tensor is None or self.output_tensor.descriptor.row_count != cursor:
                raise ValueError("output tensor rows must match compacted successful results")

    def validate_for(self, request: RaggedBatchRequest) -> None:
        """Require exact, bijective request/result correlation."""

        if (
            self.batch_id != request.descriptor.batch_id
            or self.batch_sequence_no != request.descriptor.batch_sequence_no
            or self.manifest_hash != request.manifest_hash
        ):
            raise ValueError("result batch correlation does not match request")
        expected = {
            (item.request_id, item.execution_lease_id): (
                item.request_sequence_no,
                item.input_row_start,
                item.input_row_count,
            )
            for item in request.descriptor.sequences
        }
        actual = {
            (item.request_id, item.execution_lease_id): (
                item.request_sequence_no,
                item.input_row_start,
                item.input_row_count,
            )
            for item in self.results
        }
        if actual != expected or len(self.results) != len(request.descriptor.sequences):
            raise ValueError("result identities or input mappings do not match request")
        requested = {
            (item.request_id, item.execution_lease_id): item
            for item in request.descriptor.sequences
        }
        for result in self.results:
            source = requested[(result.request_id, result.execution_lease_id)]
            if result.kv_epoch_before != source.kv_epoch:
                raise ValueError("result KV-before does not match request epoch")
            if result.status == "ok":
                if (
                    result.kv_epoch_after != source.kv_epoch + 1
                    or result.output_row_count != source.input_row_count
                ):
                    raise ValueError("successful result has an invalid KV/output transition")
            elif result.kv_epoch_after != source.kv_epoch:
                raise ValueError("failed result mutated KV relative to the request")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RaggedFrameError("METADATA", f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _decode_metadata(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise RaggedFrameError("METADATA", "metadata is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise RaggedFrameError("METADATA", f"invalid metadata JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise RaggedFrameError("METADATA", str(exc)) from exc
    if not isinstance(value, dict):
        raise RaggedFrameError("METADATA", "metadata must be an object")
    if canonical_json_bytes(value) != payload:
        raise RaggedFrameError("METADATA", "metadata must use canonical JSON encoding")
    return value


def _tensor_from_metadata(metadata: dict[str, Any], payload: bytes) -> LogicalTensor:
    try:
        descriptor = LogicalTensorDescriptor.from_dict(metadata.get("tensor", {}))
        return LogicalTensor(descriptor, payload)
    except (TypeError, ValueError) as exc:
        raise RaggedFrameError("TENSOR_CONTRACT", str(exc)) from exc


def _validate_frame_metadata(
    kind: RaggedMessageKind,
    metadata: dict[str, Any],
    payload: bytes,
    sequence_no: int,
) -> None:
    try:
        _require_non_negative(metadata.get("sequence_no"), "sequence_no")
    except ValueError as exc:
        raise RaggedFrameError("SEQUENCE", str(exc)) from exc
    if metadata["sequence_no"] != sequence_no:
        raise RaggedFrameError("SEQUENCE", "metadata sequence_no does not match prelude")
    if kind == RaggedMessageKind.BATCH:
        required = {
            "sequence_no",
            "plan_id",
            "plan_hash",
            "manifest_hash",
            "source_stage",
            "destination_stage",
            "batch",
            "batch_digest",
            "tensor",
        }
        try:
            _require_object_fields(metadata, "BATCH metadata", required=required)
            _require_uuid(metadata["plan_id"], "plan_id")
            _require_hash(metadata["plan_hash"], "plan_hash")
            _require_hash(metadata["manifest_hash"], "manifest_hash")
            _require_string(metadata["source_stage"], "source_stage")
            _require_string(metadata["destination_stage"], "destination_stage")
            descriptor = BatchDescriptor.from_dict(metadata["batch"])
        except (TypeError, ValueError) as exc:
            raise RaggedFrameError("BATCH_CONTRACT", str(exc)) from exc
        if metadata.get("batch_digest") != descriptor.digest:
            raise RaggedFrameError("BATCH_CONTRACT", "batch descriptor digest mismatch")
        tensor = _tensor_from_metadata(metadata, payload)
        if tensor.descriptor.row_count != descriptor.input_row_count:
            raise RaggedFrameError("TENSOR_CONTRACT", "tensor rows do not match batch")
        return
    if kind == RaggedMessageKind.RESULT:
        required = {
            "sequence_no",
            "batch_id",
            "batch_sequence_no",
            "manifest_hash",
            "results",
        }
        try:
            _require_object_fields(
                metadata,
                "RESULT metadata",
                required=required,
                optional={"tensor"},
            )
            if bool(payload) != ("tensor" in metadata):
                raise ValueError("RESULT tensor metadata and payload must appear together")
            _require_uuid(metadata["batch_id"], "batch_id")
            _require_hash(metadata["manifest_hash"], "manifest_hash")
            _require_non_negative(
                metadata["batch_sequence_no"], "batch_sequence_no"
            )
            if not isinstance(metadata["results"], list):
                raise ValueError("RESULT results must be an array")
            results = tuple(SequenceResult.from_dict(item) for item in metadata["results"])
            tensor = _tensor_from_metadata(metadata, payload) if payload else None
            RaggedBatchResult(
                batch_id=metadata["batch_id"],
                batch_sequence_no=metadata["batch_sequence_no"],
                manifest_hash=metadata["manifest_hash"],
                results=results,
                output_tensor=tensor,
            )
        except (TypeError, ValueError) as exc:
            raise RaggedFrameError("BATCH_RESULT", str(exc)) from exc
        return
    if payload:
        raise RaggedFrameError("FRAME_SIZE", f"{kind.name} cannot contain a payload")
    if kind == RaggedMessageKind.CREDIT:
        required = {
            "frames",
            "payload_bytes",
            "rows",
            "sequences",
            "live_requests",
            "kv_bytes",
            "replay_bytes",
        }
        try:
            _require_object_fields(
                metadata, "CREDIT metadata", required={"sequence_no", *required}
            )
            for name in required:
                _require_non_negative(metadata[name], f"credit {name}")
        except ValueError as exc:
            raise RaggedFrameError("METADATA", str(exc)) from exc
    elif kind == RaggedMessageKind.CANCEL:
        try:
            _require_object_fields(
                metadata,
                "CANCEL metadata",
                required={
                    "sequence_no",
                    "control_id",
                    "plan_id",
                    "plan_hash",
                    "manifest_hash",
                    "stage_id",
                    "plan_generation_id",
                    "request_id",
                    "execution_lease_id",
                },
            )
            for name in {
                "control_id",
                "plan_id",
                "plan_generation_id",
                "request_id",
                "execution_lease_id",
            }:
                _require_uuid(metadata[name], name)
            for name in {"plan_hash", "manifest_hash"}:
                _require_hash(metadata[name], name)
            _require_string(metadata["stage_id"], "stage_id")
        except ValueError as exc:
            raise RaggedFrameError("METADATA", str(exc)) from exc
    elif kind == RaggedMessageKind.RELEASE:
        try:
            _require_object_fields(
                metadata,
                "RELEASE metadata",
                required={
                    "sequence_no",
                    "control_id",
                    "plan_id",
                    "plan_hash",
                    "manifest_hash",
                    "stage_id",
                    "plan_generation_id",
                    "request_id",
                    "execution_lease_id",
                    "expected_final_epoch",
                },
            )
            for name in {
                "control_id",
                "plan_id",
                "plan_generation_id",
                "request_id",
                "execution_lease_id",
            }:
                _require_uuid(metadata[name], name)
            for name in {"plan_hash", "manifest_hash"}:
                _require_hash(metadata[name], name)
            _require_string(metadata["stage_id"], "stage_id")
            if metadata["expected_final_epoch"] is not None:
                _require_non_negative(
                    metadata["expected_final_epoch"], "expected_final_epoch"
                )
        except ValueError as exc:
            raise RaggedFrameError("METADATA", str(exc)) from exc
    elif kind == RaggedMessageKind.CONTROL:
        try:
            operation = metadata.get("control")
            common = {
                "sequence_no",
                "control_id",
                "control",
                "plan_id",
                "plan_hash",
                "manifest_hash",
                "stage_id",
                "plan_generation_id",
            }
            if operation == "hello":
                required = common | {"supported_versions"}
            elif operation in {"shutdown", "credit_refresh"}:
                required = common
            else:
                raise ValueError("unsupported control operation")
            _require_object_fields(metadata, "CONTROL metadata", required=required)
            _require_uuid(metadata["control_id"], "control_id")
            _require_uuid(metadata["plan_id"], "plan_id")
            _require_uuid(metadata["plan_generation_id"], "plan_generation_id")
            _require_hash(metadata["plan_hash"], "plan_hash")
            _require_hash(metadata["manifest_hash"], "manifest_hash")
            _require_string(metadata["stage_id"], "stage_id")
            if operation == "hello" and metadata["supported_versions"] != [
                [ABI_MAJOR, ABI_MINOR]
            ]:
                raise ValueError("hello must advertise exact FNX2 2.0")
        except ValueError as exc:
            raise RaggedFrameError("METADATA", str(exc)) from exc
    elif kind == RaggedMessageKind.ERROR:
        try:
            _require_object_fields(
                metadata,
                "ERROR metadata",
                required={"sequence_no", "code", "message", "control_id"},
            )
            _require_string(metadata["code"], "code")
            _require_string(metadata["message"], "message")
            if metadata["control_id"] is not None:
                _require_uuid(metadata["control_id"], "control_id")
        except ValueError as exc:
            raise RaggedFrameError("METADATA", str(exc)) from exc
    elif kind == RaggedMessageKind.ACK:
        try:
            operation = metadata.get("operation")
            fields_by_operation = {
                "cancel": {
                    "sequence_no",
                    "control_id",
                    "operation",
                    "request_id",
                    "execution_lease_id",
                    "cancelled",
                    "inflight",
                    "kv_mutated",
                },
                "release": {
                    "sequence_no",
                    "control_id",
                    "operation",
                    "request_id",
                    "execution_lease_id",
                    "released",
                    "kv_epoch",
                    "completed_results_released",
                    "tombstone_present",
                },
                "shutdown": {
                    "sequence_no",
                    "control_id",
                    "operation",
                    "stage_id",
                },
                "credit_refresh": {
                    "sequence_no",
                    "control_id",
                    "operation",
                    "stage_id",
                },
                "hello": {
                    "sequence_no",
                    "control_id",
                    "operation",
                    "plan_id",
                    "plan_hash",
                    "manifest_hash",
                    "stage_id",
                    "plan_generation_id",
                    "selected_version",
                },
            }
            if operation not in fields_by_operation:
                raise ValueError("unsupported ACK operation")
            _require_object_fields(
                metadata,
                "ACK metadata",
                required=fields_by_operation[operation],
            )
            _require_uuid(metadata["control_id"], "control_id")
            _require_string(metadata["operation"], "operation")
            if operation in {"cancel", "release"}:
                _require_uuid(metadata["request_id"], "request_id")
                _require_uuid(metadata["execution_lease_id"], "execution_lease_id")
            if operation == "cancel":
                for name in {"cancelled", "inflight", "kv_mutated"}:
                    if not isinstance(metadata[name], bool):
                        raise ValueError(f"ACK {name} must be boolean")
            elif operation == "release":
                for name in {"released", "tombstone_present"}:
                    if not isinstance(metadata[name], bool):
                        raise ValueError(f"ACK {name} must be boolean")
                if metadata["kv_epoch"] is not None:
                    _require_non_negative(metadata["kv_epoch"], "kv_epoch")
                _require_non_negative(
                    metadata["completed_results_released"],
                    "completed_results_released",
                )
            elif operation in {"shutdown", "credit_refresh"}:
                _require_string(metadata["stage_id"], "stage_id")
            else:
                _require_uuid(metadata["plan_id"], "plan_id")
                _require_uuid(metadata["plan_generation_id"], "plan_generation_id")
                _require_hash(metadata["plan_hash"], "plan_hash")
                _require_hash(metadata["manifest_hash"], "manifest_hash")
                _require_string(metadata["stage_id"], "stage_id")
                if metadata["selected_version"] != [ABI_MAJOR, ABI_MINOR]:
                    raise ValueError("hello ACK must select exact FNX2 2.0")
        except ValueError as exc:
            raise RaggedFrameError("METADATA", str(exc)) from exc


@dataclass(frozen=True)
class RaggedFrame:
    kind: RaggedMessageKind
    sequence_no: int
    metadata: dict[str, Any]
    payload: bytes = b""
    flags: int = 0
    abi_major: int = ABI_MAJOR
    abi_minor: int = ABI_MINOR
    crc: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RaggedMessageKind):
            raise ValueError("frame kind must be a RaggedMessageKind")
        _require_non_negative(self.sequence_no, "sequence_no")
        if not isinstance(self.metadata, dict):
            raise ValueError("frame metadata must be an object")
        if not isinstance(self.payload, bytes):
            raise ValueError("frame payload must be bytes")
        if self.flags != 0:
            raise ValueError("reserved frame flags must be zero")
        if self.abi_major != ABI_MAJOR or self.abi_minor != ABI_MINOR:
            raise ValueError("unsupported ragged ABI version")

    @classmethod
    def from_batch(
        cls,
        request: RaggedBatchRequest,
        *,
        sequence_no: int,
        source_stage: str,
        destination_stage: str,
    ) -> "RaggedFrame":
        metadata = {
            "sequence_no": sequence_no,
            "plan_id": request.plan_id,
            "plan_hash": request.plan_hash,
            "manifest_hash": request.manifest_hash,
            "source_stage": source_stage,
            "destination_stage": destination_stage,
            "batch": request.descriptor.to_dict(),
            "batch_digest": request.descriptor.digest,
            "tensor": request.input_tensor.descriptor.to_dict(),
        }
        return cls(RaggedMessageKind.BATCH, sequence_no, metadata, request.input_tensor.payload)

    @classmethod
    def from_result(cls, result: RaggedBatchResult, *, sequence_no: int) -> "RaggedFrame":
        metadata: dict[str, Any] = {
            "sequence_no": sequence_no,
            "batch_id": result.batch_id,
            "batch_sequence_no": result.batch_sequence_no,
            "manifest_hash": result.manifest_hash,
            "results": [item.to_dict() for item in result.results],
        }
        payload = b""
        if result.output_tensor is not None:
            metadata["tensor"] = result.output_tensor.descriptor.to_dict()
            payload = result.output_tensor.payload
        return cls(RaggedMessageKind.RESULT, sequence_no, metadata, payload)

    def batch_request(self) -> RaggedBatchRequest:
        if self.kind != RaggedMessageKind.BATCH:
            raise RaggedFrameError("BATCH_CONTRACT", "frame is not a batch")
        return RaggedBatchRequest(
            plan_id=self.metadata["plan_id"],
            plan_hash=self.metadata["plan_hash"],
            manifest_hash=self.metadata["manifest_hash"],
            descriptor=BatchDescriptor.from_dict(self.metadata["batch"]),
            input_tensor=_tensor_from_metadata(self.metadata, self.payload),
        )

    def batch_result(self) -> RaggedBatchResult:
        if self.kind != RaggedMessageKind.RESULT:
            raise RaggedFrameError("BATCH_RESULT", "frame is not a result")
        return RaggedBatchResult(
            batch_id=self.metadata["batch_id"],
            batch_sequence_no=self.metadata["batch_sequence_no"],
            manifest_hash=self.metadata["manifest_hash"],
            results=tuple(SequenceResult.from_dict(item) for item in self.metadata["results"]),
            output_tensor=(
                _tensor_from_metadata(self.metadata, self.payload) if self.payload else None
            ),
        )


def encode_ragged_frame(
    frame: RaggedFrame, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
) -> bytes:
    metadata_bytes = canonical_json_bytes(frame.metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise RaggedFrameError("FRAME_SIZE", "metadata exceeds configured maximum")
    if len(frame.payload) > max_payload_bytes:
        raise RaggedFrameError("FRAME_SIZE", "payload exceeds configured maximum")
    _validate_frame_metadata(frame.kind, frame.metadata, frame.payload, frame.sequence_no)
    checksum = crc32c(metadata_bytes + frame.payload)
    return PRELUDE.pack(
        MAGIC,
        frame.abi_major,
        frame.abi_minor,
        int(frame.kind),
        frame.flags,
        len(metadata_bytes),
        len(frame.payload),
        frame.sequence_no,
        checksum,
        0,
    ) + metadata_bytes + frame.payload


def decode_ragged_frame(
    data: bytes, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
) -> RaggedFrame:
    if len(data) < PRELUDE_BYTES:
        raise RaggedFrameError("FRAME_SIZE", "short frame prelude")
    (
        magic,
        major,
        minor,
        raw_kind,
        flags,
        metadata_length,
        payload_length,
        sequence_no,
        expected_crc,
        reserved,
    ) = PRELUDE.unpack(data[:PRELUDE_BYTES])
    if magic != MAGIC or major != ABI_MAJOR or minor != ABI_MINOR:
        raise RaggedFrameError("ABI_VERSION", "unsupported frame magic or ABI version")
    if flags or reserved:
        raise RaggedFrameError("ABI_VERSION", "reserved frame fields must be zero")
    if metadata_length > MAX_METADATA_BYTES or payload_length > max_payload_bytes:
        raise RaggedFrameError("FRAME_SIZE", "frame length exceeds configured maximum")
    expected_length = PRELUDE_BYTES + metadata_length + payload_length
    if len(data) != expected_length:
        raise RaggedFrameError("FRAME_SIZE", "truncated or overlong frame")
    try:
        kind = RaggedMessageKind(raw_kind)
    except ValueError as exc:
        raise RaggedFrameError("ABI_VERSION", f"unknown message kind: {raw_kind}") from exc
    metadata_bytes = data[PRELUDE_BYTES : PRELUDE_BYTES + metadata_length]
    payload = data[PRELUDE_BYTES + metadata_length :]
    if crc32c(metadata_bytes + payload) != expected_crc:
        raise RaggedFrameError("CHECKSUM", "CRC32C mismatch")
    metadata = _decode_metadata(metadata_bytes)
    _validate_frame_metadata(kind, metadata, payload, sequence_no)
    return RaggedFrame(
        kind=kind,
        sequence_no=sequence_no,
        metadata=metadata,
        payload=payload,
        flags=flags,
        abi_major=major,
        abi_minor=minor,
        crc=expected_crc,
    )


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        value = channel.recv(remaining)
        if not value:
            raise RaggedFrameError("FRAME_SIZE", "connection closed during frame")
        chunks.append(value)
        remaining -= len(value)
    return b"".join(chunks)


def read_ragged_frame(
    channel: socket.socket, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
) -> RaggedFrame:
    prelude = _recv_exact(channel, PRELUDE_BYTES)
    unpacked = PRELUDE.unpack(prelude)
    metadata_length = unpacked[5]
    payload_length = unpacked[6]
    if metadata_length > MAX_METADATA_BYTES or payload_length > max_payload_bytes:
        raise RaggedFrameError("FRAME_SIZE", "incoming frame exceeds configured maximum")
    body = _recv_exact(channel, metadata_length + payload_length)
    try:
        return decode_ragged_frame(
            prelude + body, max_payload_bytes=max_payload_bytes
        )
    except RaggedFrameError as exc:
        # The body is fully length-delimited and consumed. Semantic failures can
        # therefore produce one bounded ERROR and leave the channel aligned.
        # Integrity/version/size failures remain terminal.
        recoverable_codes = {
            "METADATA",
            "TENSOR_CONTRACT",
            "BATCH_CONTRACT",
            "BATCH_RESULT",
            "SEQUENCE",
        }
        if exc.code in recoverable_codes:
            raise RaggedFrameError(
                exc.code,
                exc.message,
                sequence_no=unpacked[7],
                recoverable=True,
            ) from exc
        raise


def send_ragged_frame(
    channel: socket.socket,
    frame: RaggedFrame,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> int:
    encoded = encode_ragged_frame(frame, max_payload_bytes=max_payload_bytes)
    channel.sendall(encoded)
    return len(encoded)
