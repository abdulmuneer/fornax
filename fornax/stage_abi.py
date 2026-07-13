from __future__ import annotations

import json
import re
import socket
import struct
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .stage_runtime import HASH_RE, Tensor, TensorDescriptor, canonical_json_bytes


MAGIC = b"FNX1"
ABI_MAJOR = 1
ABI_MINOR = 0
PRELUDE = struct.Struct("!4sHHHHIQQII")
PRELUDE_BYTES = PRELUDE.size
MAX_METADATA_BYTES = 64 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024


class MessageKind(IntEnum):
    ACTIVATION = 1
    LOGITS = 2
    KV_PAGE = 3
    EXPERT_BATCH = 4
    CREDIT = 5
    ACK = 6
    CANCEL = 7
    ERROR = 8
    HEARTBEAT = 9


DATA_KINDS = {MessageKind.ACTIVATION, MessageKind.LOGITS}
RESERVED_DATA_KINDS = {MessageKind.KV_PAGE, MessageKind.EXPERT_BATCH}
CONTROL_KINDS = {
    MessageKind.CREDIT,
    MessageKind.ACK,
    MessageKind.CANCEL,
    MessageKind.ERROR,
    MessageKind.HEARTBEAT,
}
UUID_FIELDS = {"plan_id", "request_id"}
HASH_FIELDS = {"plan_hash", "manifest_hash"}
REQUIRED_DATA_METADATA = {
    "plan_id",
    "plan_hash",
    "manifest_hash",
    "request_id",
    "microbatch_id",
    "sequence_no",
    "source_stage",
    "destination_stage",
    "phase",
    "token_start",
    "token_count",
    "kv_epoch",
    "tensor",
    "deadline_ns",
    "trace_id",
    "span_id",
}


class FrameError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _crc32c_table() -> tuple[int, ...]:
    polynomial = 0x82F63B78
    table: list[int] = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ polynomial if crc & 1 else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for value in data:
        crc = CRC32C_TABLE[(crc ^ value) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FrameError("METADATA", f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _decode_metadata(payload: bytes) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_reject_duplicate_pairs)
    except UnicodeDecodeError as exc:
        raise FrameError("METADATA", "metadata is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise FrameError("METADATA", f"invalid metadata JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise FrameError("METADATA", "metadata must be a JSON object")
    return value


def _canonical_uuid(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise FrameError("METADATA", f"{field_name} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise FrameError("METADATA", f"{field_name} must be a UUID") from exc
    if str(parsed) != value:
        raise FrameError("METADATA", f"{field_name} must be canonical lower-case")


def _validate_data_metadata(
    metadata: dict[str, Any],
    *,
    sequence_no: int,
    payload_bytes: int,
    kind: MessageKind,
) -> None:
    missing = sorted(REQUIRED_DATA_METADATA - set(metadata))
    if missing:
        raise FrameError("METADATA", f"missing metadata fields: {missing}")
    for field_name in UUID_FIELDS:
        _canonical_uuid(metadata.get(field_name), field_name)
    for field_name in HASH_FIELDS:
        value = metadata.get(field_name)
        if not isinstance(value, str) or not HASH_RE.fullmatch(value):
            raise FrameError("METADATA", f"{field_name} must be sha256:<64 lowercase hex>")
    if metadata.get("sequence_no") != sequence_no:
        raise FrameError("SEQUENCE", "metadata sequence_no does not match prelude")
    if metadata.get("phase") not in {"prefill", "decode"}:
        raise FrameError("METADATA", "phase must be prefill or decode")
    for field_name in ("token_start", "kv_epoch"):
        value = metadata.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FrameError("METADATA", f"{field_name} must be non-negative")
    token_count = metadata.get("token_count")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise FrameError("METADATA", "token_count must be positive")
    deadline_ns = metadata.get("deadline_ns")
    if isinstance(deadline_ns, bool) or not isinstance(deadline_ns, int) or deadline_ns <= 0:
        raise FrameError("METADATA", "deadline_ns must be positive")
    try:
        descriptor = TensorDescriptor.from_dict(metadata.get("tensor"))
    except (TypeError, ValueError) as exc:
        raise FrameError("TENSOR_CONTRACT", str(exc)) from exc
    expected_kind = "activation" if kind == MessageKind.ACTIVATION else "logits"
    if descriptor.kind != expected_kind:
        raise FrameError("TENSOR_CONTRACT", "message and tensor kind differ")
    if descriptor.payload_bytes != payload_bytes:
        raise FrameError("TENSOR_CONTRACT", "tensor payload_bytes does not match frame")
    if descriptor.shape[0] != token_count:
        raise FrameError("TENSOR_CONTRACT", "token_count does not match tensor rows")


def validate_metadata(
    metadata: dict[str, Any],
    *,
    sequence_no: int,
    payload_bytes: int,
    kind: MessageKind,
) -> None:
    if kind in RESERVED_DATA_KINDS:
        raise FrameError("ABI_VERSION", f"{kind.name} is reserved in Phase 0.5")
    if kind in DATA_KINDS:
        _validate_data_metadata(
            metadata,
            sequence_no=sequence_no,
            payload_bytes=payload_bytes,
            kind=kind,
        )
        return
    if kind not in CONTROL_KINDS:
        raise FrameError("ABI_VERSION", f"unsupported message kind: {int(kind)}")
    if payload_bytes:
        raise FrameError("FRAME_SIZE", f"{kind.name} must not contain tensor payload")
    if metadata.get("sequence_no") != sequence_no:
        raise FrameError("SEQUENCE", "control metadata sequence_no does not match prelude")
    if kind in {MessageKind.ACK, MessageKind.CANCEL, MessageKind.ERROR}:
        for field_name in ("request_id", "microbatch_id"):
            if not metadata.get(field_name):
                raise FrameError("METADATA", f"{kind.name} requires {field_name}")
    if kind == MessageKind.CREDIT:
        messages = metadata.get("messages")
        byte_credit = metadata.get("bytes")
        if (
            isinstance(messages, bool)
            or not isinstance(messages, int)
            or messages < 0
            or isinstance(byte_credit, bool)
            or not isinstance(byte_credit, int)
            or byte_credit < 0
        ):
            raise FrameError("METADATA", "CREDIT requires non-negative messages/bytes")


@dataclass(frozen=True)
class Frame:
    kind: MessageKind
    sequence_no: int
    metadata: dict[str, Any]
    payload: bytes = b""
    flags: int = 0
    abi_major: int = ABI_MAJOR
    abi_minor: int = ABI_MINOR
    crc: int | None = None

    def __post_init__(self) -> None:
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        if self.flags != 0:
            raise ValueError("reserved frame flags must be zero")
        if self.abi_major != ABI_MAJOR or self.abi_minor < 0:
            raise ValueError("unsupported ABI version")

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        *,
        sequence_no: int,
        metadata: dict[str, Any],
    ) -> "Frame":
        kind = (
            MessageKind.ACTIVATION
            if tensor.descriptor.kind == "activation"
            else MessageKind.LOGITS
        )
        merged = dict(metadata)
        merged["sequence_no"] = sequence_no
        merged["tensor"] = tensor.descriptor.to_dict()
        return cls(
            kind=kind,
            sequence_no=sequence_no,
            metadata=merged,
            payload=tensor.payload,
        )

    def tensor(self) -> Tensor:
        if self.kind not in DATA_KINDS:
            raise FrameError("TENSOR_CONTRACT", "control frame has no tensor")
        return Tensor(TensorDescriptor.from_dict(self.metadata["tensor"]), self.payload)


def encode_frame(frame: Frame, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> bytes:
    metadata_bytes = canonical_json_bytes(frame.metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise FrameError("FRAME_SIZE", "metadata exceeds 64 KiB")
    if len(frame.payload) > max_payload_bytes:
        raise FrameError("FRAME_SIZE", "payload exceeds configured maximum")
    validate_metadata(
        frame.metadata,
        sequence_no=frame.sequence_no,
        payload_bytes=len(frame.payload),
        kind=frame.kind,
    )
    checksum = crc32c(metadata_bytes + frame.payload)
    prelude = PRELUDE.pack(
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
    )
    return prelude + metadata_bytes + frame.payload


def decode_frame(data: bytes, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> Frame:
    if len(data) < PRELUDE_BYTES:
        raise FrameError("FRAME_SIZE", "short frame prelude")
    (
        magic,
        abi_major,
        abi_minor,
        raw_kind,
        flags,
        metadata_length,
        payload_length,
        sequence_no,
        expected_crc,
        reserved,
    ) = PRELUDE.unpack(data[:PRELUDE_BYTES])
    if magic != MAGIC or abi_major != ABI_MAJOR:
        raise FrameError("ABI_VERSION", "unsupported frame magic or major version")
    if flags or reserved:
        raise FrameError("ABI_VERSION", "reserved frame fields must be zero")
    if metadata_length > MAX_METADATA_BYTES or payload_length > max_payload_bytes:
        raise FrameError("FRAME_SIZE", "frame length exceeds configured maximum")
    expected_length = PRELUDE_BYTES + metadata_length + payload_length
    if len(data) != expected_length:
        raise FrameError("FRAME_SIZE", "truncated or overlong frame")
    try:
        kind = MessageKind(raw_kind)
    except ValueError as exc:
        raise FrameError("ABI_VERSION", f"unknown message kind: {raw_kind}") from exc
    metadata_bytes = data[PRELUDE_BYTES : PRELUDE_BYTES + metadata_length]
    payload = data[PRELUDE_BYTES + metadata_length :]
    actual_crc = crc32c(metadata_bytes + payload)
    if actual_crc != expected_crc:
        raise FrameError("CHECKSUM", "CRC32C mismatch")
    metadata = _decode_metadata(metadata_bytes)
    validate_metadata(
        metadata,
        sequence_no=sequence_no,
        payload_bytes=payload_length,
        kind=kind,
    )
    return Frame(
        kind=kind,
        sequence_no=sequence_no,
        metadata=metadata,
        payload=payload,
        flags=flags,
        abi_major=abi_major,
        abi_minor=abi_minor,
        crc=expected_crc,
    )


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise FrameError("FRAME_SIZE", "connection closed during frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(
    channel: socket.socket, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
) -> Frame:
    prelude = _recv_exact(channel, PRELUDE_BYTES)
    unpacked = PRELUDE.unpack(prelude)
    metadata_length = unpacked[5]
    payload_length = unpacked[6]
    if metadata_length > MAX_METADATA_BYTES or payload_length > max_payload_bytes:
        raise FrameError("FRAME_SIZE", "incoming frame exceeds configured maximum")
    body = _recv_exact(channel, metadata_length + payload_length)
    return decode_frame(prelude + body, max_payload_bytes=max_payload_bytes)


def send_frame(
    channel: socket.socket,
    frame: Frame,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> int:
    encoded = encode_frame(frame, max_payload_bytes=max_payload_bytes)
    channel.sendall(encoded)
    return len(encoded)


@dataclass
class SequenceTracker:
    next_sequence: int = 0
    acknowledged: dict[int, int] | None = None

    def __post_init__(self) -> None:
        if self.acknowledged is None:
            self.acknowledged = {}

    def accept(self, frame: Frame) -> str:
        checksum = int(frame.crc or crc32c(canonical_json_bytes(frame.metadata) + frame.payload))
        known = self.acknowledged.get(frame.sequence_no)
        if known is not None:
            if known != checksum:
                raise FrameError("SEQUENCE", "conflicting duplicate frame")
            return "duplicate"
        if frame.sequence_no != self.next_sequence:
            raise FrameError(
                "SEQUENCE",
                f"expected sequence {self.next_sequence}, got {frame.sequence_no}",
            )
        self.acknowledged[frame.sequence_no] = checksum
        self.next_sequence += 1
        return "accepted"


def validate_frame_identity(
    frame: Frame,
    *,
    plan_id: str,
    plan_hash: str,
    manifest_hash: str,
    destination_stage: str,
) -> None:
    expected = {
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "manifest_hash": manifest_hash,
        "destination_stage": destination_stage,
    }
    for field_name, value in expected.items():
        if frame.metadata.get(field_name) != value:
            raise FrameError("STALE_PLAN", f"{field_name} does not match installed route")
