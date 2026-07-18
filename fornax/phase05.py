from __future__ import annotations

import hashlib
import math
import platform
import resource
import socket
import struct
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .engine_v0 import (
    AdmissionScheduler,
    EngineV0Error,
    StageChannel,
    WorkerControlClient,
    _serve_connection,
    start_two_worker_engine,
    stop_two_worker_engine,
)
from .io import read_json, write_json
from .stage_abi import (
    ABI_MAJOR,
    ABI_MINOR,
    MAGIC,
    PRELUDE,
    Frame,
    FrameError,
    MessageKind,
    SequenceTracker,
    crc32c,
    decode_frame,
    encode_frame,
    read_frame,
    send_frame,
    validate_frame_identity,
)
from .stage_runtime import (
    MaxStageBackend,
    ReferenceStageBackend,
    SimulationProfile,
    SimulatedMaxStageBackend,
    StageBackendSpec,
    StageManifest,
    StageRequest,
    StageResult,
    StageRuntimeError,
    Tensor,
    attest_backend_capabilities,
    create_stage_backend,
)


RECORD_KIND = "phase05-engine-v0-evidence"
EVIDENCE_CLASS = "T1-simulation"
PLAN_ID = "05050505-0505-4050-8050-050505050505"
PLAN_HASH = "sha256:" + "0" * 64
MODEL_CONFIG_HASH = "sha256:" + "1" * 64
TOKENIZER_HASH = "sha256:" + "2" * 64
TEMPLATE_HASH = "sha256:" + "3" * 64
CONTEXT_POINTS = (16, 128, 512, 4096)
CONCURRENCY_POINTS = (1, 4, 8)
FAULTS = ("none", "slow_stage", "disconnect", "corruption", "timeout", "cancel", "stale_plan")
CURRENT_STAGE_CONFORMANCE_CHECKS = frozenset(
    {
        "optional-request-sequence-omitted",
        "malformed-request-sequence-rejected",
        "minor-version-rejected",
        "sequence-history-bounded",
        "backend-factory-capability-attestation",
        "duplicate-wire-response-no-reexecution",
        "nonfinite-rejection-worker-liveness",
    }
)
SCENARIOS: dict[str, dict[str, float]] = {
    "S-WORST-DESKTOP": {
        "link_gbit_s": 1.0,
        "rtt_ms": 5.0,
        "payload_factor": 0.5,
        "stage_ratio": 0.25,
        "jitter_fraction": 0.20,
    },
    "S-DESKTOP": {
        "link_gbit_s": 10.0,
        "rtt_ms": 1.0,
        "payload_factor": 0.7,
        "stage_ratio": 0.5,
        "jitter_fraction": 0.05,
    },
    "S-PROSUMER-25": {
        "link_gbit_s": 25.0,
        "rtt_ms": 0.5,
        "payload_factor": 0.7,
        "stage_ratio": 0.5,
        "jitter_fraction": 0.05,
    },
    "S-PROSUMER-100": {
        "link_gbit_s": 100.0,
        "rtt_ms": 0.1,
        "payload_factor": 0.9,
        "stage_ratio": 1.0,
        "jitter_fraction": 0.05,
    },
    "S-COMPUTE-SKEW": {
        "link_gbit_s": 100.0,
        "rtt_ms": 0.1,
        "payload_factor": 0.9,
        "stage_ratio": 0.25,
        "jitter_fraction": 0.05,
    },
}


def phase05_manifests(hidden_size: int = 2048) -> tuple[StageManifest, StageManifest]:
    common = {
        "manifest_version": 1,
        "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "model_snapshot": "85864749cd611b4353ce1decdb286193298f64c7",
        "model_config_hash": MODEL_CONFIG_HASH,
        "tokenizer_hash": TOKENIZER_HASH,
        "template_hash": TEMPLATE_HASH,
        "max_build_id": "simulated-max-phase05",
        "fornax_abi_major": 1,
        "fornax_abi_minor": 0,
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "kv_policy": "stage_local",
    }
    contracts = {
        "input_contract": {
            "kind": "activation",
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": hidden_size,
        },
        "output_contract": {
            "kind": "activation",
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": hidden_size,
        },
    }
    stage0 = StageManifest(
        **common,
        **contracts,
        stage_id="stage-0",
        stage_index=0,
        layer_start=0,
        layer_end=13,
        weight_artifacts=(
            {
                "path": "phase05-stage0-fixture.bin",
                "size": 16_000_000_000,
                "sha256": "sha256:" + "4" * 64,
                "layer_start": 0,
                "layer_end": 13,
            },
        ),
        device_requirement={
            "backend": "simulated-max",
            "device_identity": "sim-nvidia-stage-0",
            "minimum_memory_bytes": 17_179_869_184,
            "dtypes": ["bf16"],
            "assumptions": ["SA-002", "SA-003", "SA-004", "SA-006"],
        },
    )
    stage1_contracts = dict(contracts)
    stage1_contracts["output_contract"] = {
        **contracts["output_contract"],
        "kind": "logits",
    }
    stage1 = StageManifest(
        **common,
        **stage1_contracts,
        stage_id="stage-1",
        stage_index=1,
        layer_start=14,
        layer_end=26,
        weight_artifacts=(
            {
                "path": "phase05-stage1-fixture.bin",
                "size": 15_413_626_576,
                "sha256": "sha256:" + "5" * 64,
                "layer_start": 14,
                "layer_end": 26,
            },
        ),
        device_requirement={
            "backend": "simulated-max",
            "device_identity": "sim-apple-stage-1",
            "minimum_memory_bytes": 17_179_869_184,
            "dtypes": ["bf16"],
            "assumptions": ["SA-001", "SA-003", "SA-004", "SA-006"],
        },
    )
    return stage0, stage1


def phase05_tensor(rows: int, hidden_size: int = 2048) -> Tensor:
    values = [
        ((((row + 2) * (column + 5)) % 31) - 15) / 31.0
        for row in range(rows)
        for column in range(hidden_size)
    ]
    return Tensor.from_values(
        values,
        kind="activation",
        dtype="bf16",
        shape=(rows, hidden_size),
    )


def _request(
    manifest: StageManifest,
    tensor: Tensor,
    *,
    request_id: str,
    phase: str = "prefill",
    sequence_no: int = 0,
    kv_epoch: int = 0,
    deadline_ns: int | None = None,
    plan_hash: str | None = None,
) -> StageRequest:
    return StageRequest(
        plan_id=manifest.plan_id,
        plan_hash=plan_hash or manifest.plan_hash,
        request_id=request_id,
        microbatch_id="microbatch-0",
        sequence_no=sequence_no,
        phase=phase,
        token_start=0,
        token_count=tensor.descriptor.shape[0],
        input_activation=tensor,
        kv_epoch=kv_epoch,
        deadline_ns=deadline_ns or time.monotonic_ns() + 60_000_000_000,
        trace_context={"trace_id": f"trace-{request_id}", "span_id": "span-0"},
    )


def _frame_metadata(manifest: StageManifest, *, sequence_no: int = 0) -> dict[str, Any]:
    return {
        "plan_id": manifest.plan_id,
        "plan_hash": manifest.plan_hash,
        "manifest_hash": manifest.manifest_hash,
        "request_id": "10101010-1010-4010-8010-101010101010",
        "microbatch_id": "microbatch-0",
        "sequence_no": sequence_no,
        "source_stage": "orchestrator",
        "destination_stage": manifest.stage_id,
        "phase": "prefill",
        "token_start": 0,
        "token_count": 2,
        "kv_epoch": 0,
        "deadline_ns": time.monotonic_ns() + 60_000_000_000,
        "trace_id": "trace-conformance",
        "span_id": "span-conformance",
    }


def _expect_error(code: str, operation: Callable[[], Any]) -> bool:
    try:
        operation()
    except (FrameError, StageRuntimeError, EngineV0Error, ValueError) as exc:
        return getattr(exc, "code", None) == code or code in str(exc)
    return False


def run_stage_conformance() -> dict[str, Any]:
    manifests = phase05_manifests()
    tensor = phase05_tensor(2)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, evidence: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "evidence": evidence})

    frame = Frame.from_tensor(
        tensor,
        sequence_no=0,
        metadata=_frame_metadata(manifests[0]),
    )
    encoded = encode_frame(frame)
    decoded = decode_frame(encoded)
    check(
        "valid-bf16-activation-frame",
        decoded.tensor().values() == tensor.values(),
        "FNX1 encode/decode round trip",
    )
    check(
        "optional-request-sequence-omitted",
        "request_sequence_no" not in decoded.metadata,
        "FNX1 v1 falls back to the frame sequence",
    )
    malformed_extension = _frame_metadata(manifests[0])
    malformed_extension["request_sequence_no"] = None
    check(
        "malformed-request-sequence-rejected",
        _expect_error(
            "METADATA",
            lambda: encode_frame(
                Frame.from_tensor(
                    tensor, sequence_no=0, metadata=malformed_extension
                )
            ),
        ),
        "optional request sequence must be a non-negative integer",
    )
    logits = Tensor.from_values(
        tensor.values(), kind="logits", dtype="bf16", shape=tensor.descriptor.shape
    )
    logits_metadata = _frame_metadata(manifests[1])
    logits_metadata["source_stage"] = manifests[1].stage_id
    logits_metadata["destination_stage"] = "orchestrator"
    logits_frame = Frame.from_tensor(logits, sequence_no=0, metadata=logits_metadata)
    check(
        "valid-bf16-logits-frame",
        decode_frame(encode_frame(logits_frame)).tensor().descriptor.kind == "logits",
        "LOGITS kind and tensor descriptor agree",
    )

    control_frames = [
        Frame(
            MessageKind.CREDIT,
            0,
            {"sequence_no": 0, "messages": 1, "bytes": 1024},
        ),
        Frame(
            MessageKind.ACK,
            0,
            {"sequence_no": 0, "request_id": "r", "microbatch_id": "m"},
        ),
        Frame(
            MessageKind.CANCEL,
            0,
            {"sequence_no": 0, "request_id": "r", "microbatch_id": "m"},
        ),
        Frame(
            MessageKind.ERROR,
            0,
            {"sequence_no": 0, "request_id": "r", "microbatch_id": "m"},
        ),
        Frame(MessageKind.HEARTBEAT, 0, {"sequence_no": 0, "control": "health"}),
    ]
    check(
        "control-kinds-zero-payload",
        all(decode_frame(encode_frame(item)).payload == b"" for item in control_frames),
        "CREDIT/ACK/CANCEL/ERROR/HEARTBEAT encode with no tensor payload",
    )
    check("short-prelude-rejected", _expect_error("FRAME_SIZE", lambda: decode_frame(b"FNX1")), "short prelude")
    check(
        "oversized-metadata-rejected",
        _expect_error(
            "FRAME_SIZE",
            lambda: encode_frame(
                Frame(
                    MessageKind.HEARTBEAT,
                    0,
                    {"sequence_no": 0, "padding": "x" * (65 * 1024)},
                )
            ),
        ),
        "metadata >64 KiB",
    )
    check(
        "oversized-payload-rejected",
        _expect_error("FRAME_SIZE", lambda: encode_frame(frame, max_payload_bytes=1)),
        "configured payload maximum",
    )
    check(
        "truncated-payload-rejected",
        _expect_error("FRAME_SIZE", lambda: decode_frame(encoded[:-1])),
        "declared length exceeds bytes received",
    )
    corrupted = bytearray(encoded)
    corrupted[-1] ^= 1
    check(
        "checksum-rejected",
        _expect_error("CHECKSUM", lambda: decode_frame(bytes(corrupted))),
        "CRC32C mismatch",
    )
    bad_version = bytearray(encoded)
    bad_version[4:6] = struct.pack("!H", 2)
    check(
        "major-version-rejected",
        _expect_error("ABI_VERSION", lambda: decode_frame(bytes(bad_version))),
        "ABI major 2",
    )
    bad_minor = bytearray(encoded)
    bad_minor[6:8] = struct.pack("!H", ABI_MINOR + 1)
    check(
        "minor-version-rejected",
        _expect_error("ABI_VERSION", lambda: decode_frame(bytes(bad_minor))),
        "unsupported future ABI minor",
    )
    duplicate_metadata = b'{"sequence_no":0,"sequence_no":0}'
    duplicate_encoded = PRELUDE.pack(
        MAGIC,
        ABI_MAJOR,
        ABI_MINOR,
        int(MessageKind.HEARTBEAT),
        0,
        len(duplicate_metadata),
        0,
        0,
        crc32c(duplicate_metadata),
        0,
    ) + duplicate_metadata
    check(
        "duplicate-metadata-rejected",
        _expect_error("METADATA", lambda: decode_frame(duplicate_encoded)),
        "duplicate JSON key",
    )
    check(
        "reserved-payload-kinds-rejected",
        all(
            _expect_error(
                "ABI_VERSION",
                lambda kind=kind: encode_frame(
                    Frame(kind, 0, {"sequence_no": 0})
                ),
            )
            for kind in (MessageKind.KV_PAGE, MessageKind.EXPERT_BATCH)
        ),
        "KV_PAGE/EXPERT_BATCH deferred in Phase 0.5",
    )
    check(
        "stale-plan-and-destination-rejected",
        _expect_error(
            "STALE_PLAN",
            lambda: validate_frame_identity(
                decoded,
                plan_id=manifests[0].plan_id,
                plan_hash="sha256:" + "f" * 64,
                manifest_hash=manifests[0].manifest_hash,
                destination_stage=manifests[0].stage_id,
            ),
        ),
        "installed route identity",
    )
    tracker = SequenceTracker()
    first_state = tracker.accept(decoded)
    duplicate_state = tracker.accept(decoded)
    altered = Frame.from_tensor(
        Tensor.from_values(
            list(tensor.values()[:-1]) + [0.0],
            kind="activation",
            dtype="bf16",
            shape=tensor.descriptor.shape,
        ),
        sequence_no=0,
        metadata=_frame_metadata(manifests[0]),
    )
    altered = decode_frame(encode_frame(altered))
    check(
        "duplicate-same-crc-idempotent",
        first_state == "accepted" and duplicate_state == "duplicate",
        "SequenceTracker remembers acknowledged CRC",
    )
    check(
        "duplicate-different-crc-rejected",
        _expect_error("SEQUENCE", lambda: tracker.accept(altered)),
        "conflicting duplicate",
    )
    out_of_order = SequenceTracker()
    future_frame = Frame(
        MessageKind.HEARTBEAT, 1, {"sequence_no": 1, "control": "health"}
    )
    future_frame = decode_frame(encode_frame(future_frame))
    check(
        "out-of-order-rejected",
        _expect_error("SEQUENCE", lambda: out_of_order.accept(future_frame)),
        "expected sequence zero",
    )
    bounded_tracker = SequenceTracker(max_entries=1)
    bounded_tracker.accept(decoded)
    bounded_tracker.accept(future_frame)
    check(
        "sequence-history-bounded",
        set(bounded_tracker.acknowledged or {}) == {1}
        and _expect_error("SEQUENCE", lambda: bounded_tracker.accept(decoded)),
        "one-entry replay window evicts older digests and fails old retries closed",
    )

    backend_spec = StageBackendSpec.simulated(
        SimulationProfile(
            scenario_id="conformance-attestation",
            build_id=manifests[0].max_build_id,
            device_identity=str(
                manifests[0].device_requirement["device_identity"]
            ),
            memory_limit_bytes=32 * 1024 * 1024 * 1024,
        )
    )
    attestation = attest_backend_capabilities(
        create_stage_backend(StageBackendSpec.from_dict(backend_spec.to_dict())),
        manifests[0],
    )
    check(
        "backend-factory-capability-attestation",
        attestation["compatible"]
        and attestation["checked_before_load"]
        and attestation["observed"]["source"] == "backend",
        "serializable factory reports backend-originated facts before load",
    )

    class CountingBackend(ReferenceStageBackend):
        def __init__(self) -> None:
            super().__init__()
            self.execute_calls = 0

        def execute(self, handle: Any, request: StageRequest) -> Any:
            self.execute_calls += 1
            return super().execute(handle, request)

    counting_backend = CountingBackend()
    counting_handle = counting_backend.load(manifests[0])
    server, client = socket.socketpair()

    def serve_wire_conformance() -> None:
        with server:
            _serve_connection(
                server,
                backend=counting_backend,
                handle=counting_handle,
                manifest=manifests[0],
                max_payload_bytes=256 * 1024 * 1024,
            )

    wire_thread = threading.Thread(target=serve_wire_conformance, daemon=True)
    wire_thread.start()
    duplicate_wire_ok = False
    nonfinite_liveness_ok = False
    wire_evidence = "socketpair conformance did not complete"
    try:
        negotiate = Frame(
            MessageKind.HEARTBEAT,
            0,
            {
                "sequence_no": 0,
                "control": "negotiate",
                "plan_id": manifests[0].plan_id,
                "plan_hash": manifests[0].plan_hash,
                "manifest_hash": manifests[0].manifest_hash,
                "destination_stage": manifests[0].stage_id,
                "abi_major": 1,
                "abi_minor": 0,
            },
        )
        send_frame(client, negotiate)
        ready = read_frame(client)
        initial_credit = read_frame(client)
        data_metadata = _frame_metadata(manifests[0])
        data_metadata["sequence_no"] = 1
        data_frame = Frame.from_tensor(
            tensor, sequence_no=1, metadata=data_metadata
        )
        send_frame(client, data_frame)
        first_responses = tuple(read_frame(client) for _ in range(3))
        send_frame(client, data_frame)
        replayed_responses = tuple(read_frame(client) for _ in range(3))
        duplicate_wire_ok = (
            ready.metadata.get("control") == "ready"
            and initial_credit.kind == MessageKind.CREDIT
            and first_responses == replayed_responses
            and counting_backend.execute_calls == 1
        )

        nonfinite_metadata = _frame_metadata(manifests[0])
        nonfinite_metadata["sequence_no"] = 2
        nonfinite_metadata["request_id"] = "21212121-2121-4121-8121-212121212121"
        nonfinite_metadata["tensor"] = tensor.descriptor.to_dict()
        nonfinite = Frame(
            MessageKind.ACTIVATION,
            2,
            nonfinite_metadata,
            payload=b"\xc0\x7f" + tensor.payload[2:],
        )
        send_frame(client, nonfinite)
        nonfinite_error = read_frame(client)
        restored_credit = read_frame(client)

        valid_metadata = _frame_metadata(manifests[0])
        valid_metadata["sequence_no"] = 3
        valid_metadata["request_id"] = "23232323-2323-4323-8323-232323232323"
        valid = Frame.from_tensor(tensor, sequence_no=3, metadata=valid_metadata)
        send_frame(client, valid)
        valid_responses = tuple(read_frame(client) for _ in range(3))
        nonfinite_liveness_ok = (
            nonfinite_error.kind == MessageKind.ERROR
            and nonfinite_error.metadata.get("code") == "TENSOR_CONTRACT"
            and restored_credit.kind == MessageKind.CREDIT
            and valid_responses[0].kind == MessageKind.ACK
            and valid_responses[1].kind == MessageKind.ACTIVATION
            and valid_responses[2].kind == MessageKind.CREDIT
            and counting_backend.execute_calls == 2
        )
        shutdown = Frame(
            MessageKind.HEARTBEAT,
            4,
            {"sequence_no": 4, "control": "shutdown"},
        )
        send_frame(client, shutdown)
        shutdown_response = read_frame(client)
        wire_evidence = (
            "exact replay reused response; nonfinite input restored credit; "
            f"shutdown={shutdown_response.metadata.get('control')}"
        )
    except (FrameError, OSError, ValueError) as exc:
        wire_evidence = str(exc)
    finally:
        client.close()
        wire_thread.join(timeout=2.0)
    check(
        "duplicate-wire-response-no-reexecution",
        duplicate_wire_ok,
        wire_evidence,
    )
    check(
        "nonfinite-rejection-worker-liveness",
        nonfinite_liveness_ok and not wire_thread.is_alive(),
        wire_evidence,
    )

    backend_outputs: dict[str, tuple[float, ...]] = {}
    for backend in (
        ReferenceStageBackend(),
        SimulatedMaxStageBackend(
            SimulationProfile(scenario_id="conformance", stage_service_ns=100_000)
        ),
    ):
        handle = backend.load(manifests[0])
        request_id = "20202020-2020-4020-8020-202020202020"
        result = backend.execute(
            handle, _request(manifests[0], tensor, request_id=request_id)
        )
        assert result.output_tensor is not None
        backend_outputs[backend.backend_name] = result.output_tensor.values()
        duplicate_result = backend.execute(
            handle, _request(manifests[0], tensor, request_id=request_id)
        )
        check(
            f"{backend.backend_name}-at-most-once",
            duplicate_result == result and result.kv_epoch_after == 1,
            "duplicate execution returns cached StageResult",
        )
        expired = backend.execute(
            handle,
            _request(
                manifests[0],
                tensor,
                request_id=str(uuid.uuid4()),
                deadline_ns=1,
            ),
        )
        check(
            f"{backend.backend_name}-deadline-before-execution",
            expired.status == "deadline" and expired.kv_epoch_after == 0,
            "expired absolute deadline",
        )
        cancel_id = str(uuid.uuid4())
        backend.cancel(handle, cancel_id, "conformance cancel")
        cancelled = backend.execute(
            handle, _request(manifests[0], tensor, request_id=cancel_id)
        )
        check(
            f"{backend.backend_name}-cancel-before-execution",
            cancelled.status == "cancelled" and cancelled.kv_epoch_after == 0,
            "cancelled request does not mutate KV",
        )
    check(
        "reference-simulated-output-parity",
        backend_outputs["reference"] == backend_outputs["simulated-max"],
        "shared StageExecutable transform",
    )

    scheduler = AdmissionScheduler(max_inflight=1, max_queued=1, microbatch_size=1)
    deadline = time.monotonic_ns() + 1_000_000_000
    scheduler.submit("queued", object(), deadline)
    queue_full = not scheduler.submit("rejected", object(), deadline)
    queued_cancel = scheduler.cancel("queued")
    check(
        "queue-bound-and-queued-cancel",
        queue_full and queued_cancel and scheduler.stats["queued"] == 0,
        "bounded admission queue and cleanup",
    )
    check(
        "physical-max-no-fallback",
        _expect_error("EXECUTION", lambda: MaxStageBackend().load(manifests[0])),
        "missing adapter is explicit unavailable status",
    )
    return {
        "check_count": len(checks),
        "passed_count": sum(1 for item in checks if item["ok"]),
        "checks": checks,
    }


def _reference_pipeline(manifests: tuple[StageManifest, StageManifest], tensor: Tensor) -> Tensor:
    current = tensor
    request_id = "30303030-3030-4030-8030-303030303030"
    for stage_manifest in manifests:
        backend = ReferenceStageBackend()
        handle = backend.load(stage_manifest)
        result = backend.execute(
            handle,
            _request(stage_manifest, current, request_id=request_id),
        )
        if result.status != "ok" or result.output_tensor is None:
            raise RuntimeError("reference pipeline failed")
        current = result.output_tensor
    return current


def _max_abs_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def _process_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _top1(tensor: Tensor) -> int:
    width = tensor.descriptor.shape[1]
    first_row = tensor.values()[:width]
    return max(range(width), key=lambda index: first_row[index])


def run_loopback_engine(
    *,
    sustained_wall_seconds: int,
    sustained_min_iterations: int,
) -> dict[str, Any]:
    manifests = phase05_manifests()
    profiles = (
        SimulationProfile(
            scenario_id="S-DESKTOP-stage-0",
            stage_service_ns=1_000_000,
            pack_ns=20_000,
            unpack_ns=20_000,
            jitter_fraction=0.05,
            seed=11,
            memory_limit_bytes=32 * 1024 * 1024 * 1024,
            build_id="simulated-max-phase05",
            device_identity="sim-nvidia-stage-0",
        ),
        SimulationProfile(
            scenario_id="S-DESKTOP-stage-1",
            stage_service_ns=2_000_000,
            pack_ns=20_000,
            unpack_ns=20_000,
            jitter_fraction=0.05,
            seed=12,
            memory_limit_bytes=32 * 1024 * 1024 * 1024,
            build_id="simulated-max-phase05",
            device_identity="sim-apple-stage-1",
        ),
    )
    tensor = phase05_tensor(8)
    reference = _reference_pipeline(manifests, tensor)
    workers, channels, orchestrator = start_two_worker_engine(manifests, profiles)
    replacement: StageChannel | None = None
    max_message_credit = 0
    max_byte_credit = 0
    min_message_credit = 1
    min_byte_credit = min(channel.byte_credit for channel in channels)
    retired_channel_events: list[dict[str, Any]] = []
    result_data: dict[str, Any]
    try:
        control_plane: list[dict[str, Any]] = []
        control_clients: list[WorkerControlClient] = []
        for stage_manifest, worker in zip(manifests, workers):
            endpoint = worker.endpoint
            assert endpoint is not None
            control = WorkerControlClient(
                stage_manifest,
                str(endpoint["host"]),
                int(endpoint["control_port"]),
            )
            control_clients.append(control)
            control_plane.append(
                {
                    "health": control.health(),
                    "capabilities": control.capabilities(),
                    "status": control.status(),
                    "install": control.install(),
                }
            )
        request_id = "40404040-4040-4040-8040-404040404040"
        prefill = orchestrator.execute(
            tensor,
            request_id=request_id,
            phase="prefill",
            request_sequence_no=0,
            deadline_ns=time.monotonic_ns() + 60_000_000_000,
        )
        decode = orchestrator.execute(
            tensor,
            request_id=request_id,
            phase="decode",
            request_sequence_no=1,
            deadline_ns=time.monotonic_ns() + 60_000_000_000,
        )
        if prefill.status != "ok" or decode.status != "ok" or decode.output_tensor is None:
            raise RuntimeError("loopback prefill/decode failed")

        no_credit_observed = False
        saved_credit = channels[0].message_credit
        channels[0].message_credit = 0
        try:
            channels[0].execute(
                _request(
                    manifests[0], tensor, request_id="41414141-4141-4141-8141-414141414141"
                )
            )
        except EngineV0Error as exc:
            no_credit_observed = exc.code == "NO_CREDIT"
        finally:
            channels[0].message_credit = saved_credit

        cancel_id = "42424242-4242-4242-8242-424242424242"
        cancel_ack = channels[0].cancel(cancel_id, "microbatch-0", "phase05 cancellation")
        cancelled = channels[0].execute(
            _request(manifests[0], tensor, request_id=cancel_id)
        )

        endpoint = workers[0].endpoint
        assert endpoint is not None
        retired_channel_events.extend(channels[0].events)
        channels[0].disconnect()
        time.sleep(0.02)
        replacement = StageChannel(
            manifests[0], str(endpoint["host"]), int(endpoint["port"])
        )
        replacement.connect()
        channels[0] = replacement
        orchestrator.stages[0] = (manifests[0], replacement)
        reconnect_result = orchestrator.execute(
            tensor,
            request_id="43434343-4343-4343-8343-434343434343",
            phase="prefill",
            request_sequence_no=0,
            deadline_ns=time.monotonic_ns() + 60_000_000_000,
        )

        sustained_concurrency = 8
        configured_process_rss_limit_bytes = 1_073_741_824
        parent_rss_start = _process_max_rss_bytes()
        worker_rss_samples: list[dict[str, Any]] = []

        def sample_worker_rss(elapsed_ns: int) -> None:
            for control in control_clients:
                status = control.status()
                worker_rss_samples.append(
                    {
                        "elapsed_ns": elapsed_ns,
                        "stage_id": status["stage_id"],
                        "process_max_rss_bytes": int(
                            status["process_max_rss_bytes"]
                        ),
                    }
                )

        active_requests = 0
        max_observed_concurrency = 0
        active_lock = threading.Lock()

        def execute_sustained(
            iteration: int, start_barrier: threading.Barrier
        ) -> StageResult:
            nonlocal active_requests, max_observed_concurrency
            with active_lock:
                active_requests += 1
                max_observed_concurrency = max(
                    max_observed_concurrency, active_requests
                )
            try:
                start_barrier.wait(timeout=10.0)
                return orchestrator.execute(
                    tensor,
                    request_id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL, f"fornax-phase05-{iteration}"
                        )
                    ),
                    phase="prefill",
                    request_sequence_no=0,
                    deadline_ns=time.monotonic_ns() + 60_000_000_000,
                    microbatch_id=f"sustained-{iteration}",
                )
            finally:
                with active_lock:
                    active_requests -= 1

        sustained_started = time.monotonic_ns()
        sustained_deadline = sustained_started + sustained_wall_seconds * 1_000_000_000
        completed_iterations = 0
        wave_count = 0
        sample_worker_rss(0)
        with ThreadPoolExecutor(max_workers=sustained_concurrency) as executor:
            while (
                time.monotonic_ns() < sustained_deadline
                or completed_iterations < sustained_min_iterations
            ):
                wave_started = time.monotonic_ns()
                start_barrier = threading.Barrier(sustained_concurrency)
                futures = [
                    executor.submit(
                        execute_sustained,
                        completed_iterations + offset,
                        start_barrier,
                    )
                    for offset in range(sustained_concurrency)
                ]
                for iteration, future in enumerate(
                    futures, start=completed_iterations
                ):
                    result = future.result()
                    if result.status != "ok":
                        raise RuntimeError(
                            f"sustained iteration {iteration} failed: {result.error}"
                        )
                completed_iterations += sustained_concurrency
                wave_count += 1
                if wave_count % 60 == 0:
                    sample_worker_rss(time.monotonic_ns() - sustained_started)
                max_message_credit = max(
                    max_message_credit,
                    *(channel.message_credit for channel in channels),
                )
                max_byte_credit = max(
                    max_byte_credit, *(channel.byte_credit for channel in channels)
                )
                min_message_credit = min(
                    min_message_credit,
                    *(channel.message_credit for channel in channels),
                )
                min_byte_credit = min(
                    min_byte_credit, *(channel.byte_credit for channel in channels)
                )
                now = time.monotonic_ns()
                if now < sustained_deadline:
                    next_wave = min(
                        sustained_deadline, wave_started + 1_000_000_000
                    )
                    if next_wave > now:
                        time.sleep((next_wave - now) / 1_000_000_000)
        finished = time.monotonic_ns()
        sample_worker_rss(finished - sustained_started)
        parent_rss_end = _process_max_rss_bytes()
        max_worker_rss_bytes = max(
            (
                int(sample["process_max_rss_bytes"])
                for sample in worker_rss_samples
            ),
            default=0,
        )
        max_process_rss_bytes = max(
            parent_rss_start, parent_rss_end, max_worker_rss_bytes
        )
        candidate = decode.output_tensor
        channel_event_sets = [retired_channel_events] + [
            channel.events for channel in channels
        ]
        channel_metrics = []
        for events in channel_event_sets:
            channel_metrics.append(
                {
                    "event_count": len(events),
                    "frames_sent": sum(
                        1 for event in events if event["kind"] == "frame_sent"
                    ),
                    "frames_received": sum(
                        1 for event in events if event["kind"] == "frame_received"
                    ),
                    "payload_bytes_sent": sum(
                        int(event.get("payload_bytes", 0))
                        for event in events
                        if event["kind"] == "frame_sent"
                    ),
                    "wire_bytes_sent": sum(
                        int(event.get("wire_bytes", 0))
                        for event in events
                        if event["kind"] == "frame_sent"
                    ),
                    "credit_events": sum(
                        1 for event in events if event["kind"] == "credit_received"
                    ),
                }
            )
        trace_samples = orchestrator.events[:4] + orchestrator.events[-4:]
        result_data = {
            "worker_pids": [worker.pid for worker in workers],
            "worker_count": len(workers),
            "independent_processes": len({worker.pid for worker in workers}) == 2,
            "control_plane": control_plane,
            "prefill_status": prefill.status,
            "decode_status": decode.status,
            "output_kind": candidate.descriptor.kind,
            "max_abs_error": _max_abs_error(candidate.values(), reference.values()),
            "reference_top1": _top1(reference),
            "candidate_top1": _top1(candidate),
            "no_credit_observed": no_credit_observed,
            "cancel_acknowledged": bool(cancel_ack.get("cancelled")),
            "cancel_status": cancelled.status,
            "reconnect_status": reconnect_result.status,
            "channel_event_counts": [len(channel.events) for channel in channels],
            "orchestrator_event_count": len(orchestrator.events),
            "evidence_ledger": {
                "evidence_class": EVIDENCE_CLASS,
                "measurement_kind": "simulation",
                "plan_id": PLAN_ID,
                "plan_hash": PLAN_HASH,
                "scenario_id": "S-DESKTOP",
                "assumption_ids": [
                    "SA-001",
                    "SA-002",
                    "SA-003",
                    "SA-004",
                    "SA-005",
                    "SA-006",
                ],
                "channel_metrics": channel_metrics,
                "trace_samples": trace_samples,
                "activation_contents_logged": False,
            },
            "sustained": {
                "duration_kind": "wall-clock-with-real-concurrent-loopback",
                "configured_wall_seconds": sustained_wall_seconds,
                "minimum_loopback_iterations": sustained_min_iterations,
                "actual_loopback_iterations": completed_iterations,
                "concurrency": sustained_concurrency,
                "max_observed_concurrency": max_observed_concurrency,
                "context_tokens_modeled": 4096,
                "activation_rows_per_request": tensor.descriptor.shape[0],
                "wall_elapsed_ns": max(0, finished - sustained_started),
                "wave_count": wave_count,
                "max_message_credit": max_message_credit,
                "min_message_credit": min_message_credit,
                "max_byte_credit": max_byte_credit,
                "min_byte_credit": min_byte_credit,
                "configured_message_credit": 1,
                "configured_byte_credit": channels[0].max_payload_bytes,
                "configured_process_rss_limit_bytes": configured_process_rss_limit_bytes,
                "parent_max_rss_bytes_start": parent_rss_start,
                "parent_max_rss_bytes_end": parent_rss_end,
                "max_worker_rss_bytes": max_worker_rss_bytes,
                "max_observed_process_rss_bytes": max_process_rss_bytes,
                "worker_rss_samples": worker_rss_samples,
                "queue_bound_exceeded": max_message_credit > 1
                or min_message_credit < 0,
                "memory_bound_exceeded": max_byte_credit
                > channels[0].max_payload_bytes
                or min_byte_credit < 0
                or max_process_rss_bytes > configured_process_rss_limit_bytes,
            },
        }
    finally:
        stop_two_worker_engine(workers, channels)
    result_data["cleanup_completed"] = all(
        worker.process is None for worker in workers
    ) and all(channel.channel is None for channel in channels)
    return result_data


def run_scenario_matrix() -> list[dict[str, Any]]:
    manifests = phase05_manifests()
    rows: list[dict[str, Any]] = []
    for scenario_index, (scenario_id, scenario) in enumerate(SCENARIOS.items()):
        for context_tokens in CONTEXT_POINTS:
            for concurrency in CONCURRENCY_POINTS:
                tensor = phase05_tensor(concurrency)
                current = tensor
                total_stage_ns = 0
                for stage_index, stage_manifest in enumerate(manifests):
                    service_base = int(1_000_000 * (1 + math.log2(context_tokens)))
                    ratio = 1.0 if stage_index == 0 else float(scenario["stage_ratio"])
                    service_ns = int(service_base / max(ratio, 0.01))
                    backend = SimulatedMaxStageBackend(
                        SimulationProfile(
                            scenario_id=f"{scenario_id}-stage-{stage_index}",
                            stage_service_ns=service_ns,
                            jitter_fraction=float(scenario["jitter_fraction"]),
                            seed=scenario_index * 1000 + context_tokens + concurrency + stage_index,
                        )
                    )
                    handle = backend.load(stage_manifest)
                    result = backend.execute(
                        handle,
                        _request(
                            stage_manifest,
                            current,
                            request_id=str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_URL,
                                    f"{scenario_id}-{context_tokens}-{concurrency}",
                                )
                            ),
                        ),
                    )
                    if result.status != "ok" or result.output_tensor is None:
                        raise RuntimeError(f"scenario {scenario_id} failed")
                    current = result.output_tensor
                    total_stage_ns += result.timings_ns["execute"]
                payload_bytes = tensor.descriptor.payload_bytes
                usable_bytes_s = (
                    scenario["link_gbit_s"] * 1_000_000_000 / 8 * scenario["payload_factor"]
                )
                transfer_ns = int(
                    (payload_bytes / usable_bytes_s + scenario["rtt_ms"] / 2000.0)
                    * 1_000_000_000
                )
                fill_efficiency = concurrency / (concurrency + len(manifests) - 1)
                balance = min(1.0, float(scenario["stage_ratio"]))
                utilization = fill_efficiency * (0.5 + 0.5 * balance)
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "assumption_ids": ["SA-005", "SA-007", "SA-008"],
                        "measurement_kind": "simulation",
                        "context_tokens": context_tokens,
                        "concurrency": concurrency,
                        "tensor_rows": concurrency,
                        "context_execution_mode": "modeled-timing-and-resource-envelope",
                        "payload_bytes": payload_bytes,
                        "stage_service_ns": total_stage_ns,
                        "transfer_ns": transfer_ns,
                        "modeled_end_to_end_ns": total_stage_ns + transfer_ns,
                        "pipeline_utilization": utilization,
                        "output_finite": all(math.isfinite(value) for value in current.values()),
                    }
                )
    return rows


def run_fault_matrix() -> list[dict[str, Any]]:
    manifest = phase05_manifests()[0]
    tensor = phase05_tensor(2)
    rows: list[dict[str, Any]] = []
    for index, fault in enumerate(FAULTS):
        backend = SimulatedMaxStageBackend(
            SimulationProfile(
                scenario_id=f"fault-{fault}",
                stage_service_ns=100_000,
                fault="none" if fault == "stale_plan" else fault,
                seed=index,
            )
        )
        handle = backend.load(manifest)
        request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"fornax-fault-{fault}"))
        request = _request(
            manifest,
            tensor,
            request_id=request_id,
            plan_hash=("sha256:" + "f" * 64) if fault == "stale_plan" else None,
        )
        try:
            result = backend.execute(handle, request)
            rows.append(
                {
                    "fault": fault,
                    "status": result.status,
                    "error_code": result.error["code"] if result.error else None,
                    "kv_mutated": result.kv_epoch_after > result.kv_epoch_before,
                }
            )
        except StageRuntimeError as exc:
            rows.append(
                {
                    "fault": fault,
                    "status": "disconnected",
                    "error_code": exc.code,
                    "kv_mutated": exc.kv_mutated,
                }
            )
    return rows


def run_scheduler_sweep() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concurrency in CONCURRENCY_POINTS:
        scheduler = AdmissionScheduler(
            max_inflight=concurrency,
            max_queued=concurrency * 2,
            microbatch_size=concurrency,
        )
        deadline = time.monotonic_ns() + 10_000_000_000
        submitted = [f"c{concurrency}-r{index}" for index in range(concurrency * 2)]
        accepted = [scheduler.submit(request_id, request_id, deadline) for request_id in submitted]
        overflow_rejected = not scheduler.submit(
            f"c{concurrency}-overflow", None, deadline
        )
        admitted_order: list[str] = []
        while scheduler.stats["queued"]:
            microbatch = scheduler.next_microbatch(time.monotonic_ns())
            admitted_order.extend(item.request_id for item in microbatch)
            for item in microbatch:
                scheduler.complete(item.request_id)
        expired_id = f"c{concurrency}-expired"
        scheduler.submit(expired_id, None, 1)
        expired_batch = scheduler.next_microbatch(time.monotonic_ns())
        cancel_id = f"c{concurrency}-cancel"
        scheduler.submit(cancel_id, None, deadline)
        cancelled = scheduler.cancel(cancel_id)
        rows.append(
            {
                "concurrency": concurrency,
                "submitted": len(submitted),
                "all_admitted": all(accepted),
                "fifo": admitted_order == submitted,
                "overflow_rejected": overflow_rejected,
                "deadline_removed": not expired_batch
                and any(
                    event["kind"] == "request_deadline"
                    and event["request_id"] == expired_id
                    for event in scheduler.events
                ),
                "cancelled": cancelled,
                "final_stats": scheduler.stats,
                "event_count": len(scheduler.events),
            }
        )
    return rows


def physical_backend_disposition() -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[1]
    max_cli = workspace / "external/modular/bazel-bin/max/python/max/_entrypoints/pipelines"
    max_source = workspace / "external/modular"
    # Phase 0.5 is deliberately generated without a physical factory. Physical
    # adapter discovery/conformance is an explicit separate command so a local
    # checkout can never make this T1 artifact appear physical by accident.
    adapter_available = MaxStageBackend().available
    return {
        "status": "available" if adapter_available else "unavailable",
        "physical_adapter_available": adapter_available,
        "source_checkout_present": max_source.exists(),
        "source_built_cli_present": max_cli.exists(),
        "local_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "two_qualifying_physical_nodes_available": False,
        "formal_g2_passed": False,
        "reason": (
            "No physical MAX backend factory is configured for this T1 run and no "
            "qualifying two-node heterogeneous fleet is asserted; S05-12 is explicitly dispositioned "
            "without falling back to simulation or blocking S05-1 through S05-11."
        ),
    }


def _fault_matrix_ok(rows: list[dict[str, Any]]) -> bool:
    expected = {
        "none": ("ok", None),
        "slow_stage": ("ok", None),
        "disconnect": ("disconnected", "EXECUTION"),
        "corruption": ("failed", "CHECKSUM"),
        "timeout": ("deadline", "DEADLINE"),
        "cancel": ("cancelled", "CANCELLED"),
        "stale_plan": ("rejected", "STALE_PLAN"),
    }
    return all(
        (row.get("status"), row.get("error_code")) == expected[row["fault"]]
        for row in rows
    )


def run_phase05_engine_v0(
    out: str | Path | None = None,
    *,
    sustained_wall_seconds: int = 1800,
    sustained_min_iterations: int = 1800,
) -> dict[str, Any]:
    if sustained_wall_seconds < 1 or sustained_min_iterations < 1:
        raise ValueError("sustained duration and iterations must be positive")
    conformance = run_stage_conformance()
    loopback = run_loopback_engine(
        sustained_wall_seconds=sustained_wall_seconds,
        sustained_min_iterations=sustained_min_iterations,
    )
    scenarios = run_scenario_matrix()
    faults = run_fault_matrix()
    scheduler = run_scheduler_sweep()
    physical = physical_backend_disposition()
    expected_scenario_rows = len(SCENARIOS) * len(CONTEXT_POINTS) * len(CONCURRENCY_POINTS)
    checks = [
        {
            "name": "stage-conformance",
            "ok": conformance["passed_count"] == conformance["check_count"],
        },
        {
            "name": "two-independent-workers",
            "ok": loopback["independent_processes"] and loopback["worker_count"] == 2,
        },
        {
            "name": "prefill-decode-versioned-experimental-abi",
            "ok": loopback["prefill_status"] == "ok" and loopback["decode_status"] == "ok",
        },
        {
            "name": "http-control-plane",
            "ok": len(loopback["control_plane"]) == 2
            and all(
                row["health"]["state"] == "READY"
                and row["capabilities"]["backend"] == "simulated-max"
                and row["install"]["unchanged"]
                for row in loopback["control_plane"]
            ),
        },
        {
            "name": "reference-output-parity",
            "ok": loopback["max_abs_error"] <= 0.02
            and loopback["reference_top1"] == loopback["candidate_top1"],
        },
        {
            "name": "credit-cancel-reconnect",
            "ok": loopback["no_credit_observed"]
            and loopback["cancel_acknowledged"]
            and loopback["cancel_status"] == "cancelled"
            and loopback["reconnect_status"] == "ok",
        },
        {
            "name": "trace-resource-evidence-ledger",
            "ok": loopback["cleanup_completed"]
            and loopback["evidence_ledger"]["evidence_class"] == EVIDENCE_CLASS
            and loopback["evidence_ledger"]["plan_id"] == PLAN_ID
            and not loopback["evidence_ledger"]["activation_contents_logged"]
            and all(
                row.get("request_id")
                and row.get("plan_id")
                and row.get("stage_id")
                and row.get("trace_id")
                and row.get("span_id")
                for row in loopback["evidence_ledger"]["trace_samples"]
            )
            and sum(
                row["payload_bytes_sent"]
                for row in loopback["evidence_ledger"]["channel_metrics"]
            )
            > 0,
        },
        {
            "name": "named-scenario-matrix",
            "ok": len(scenarios) == expected_scenario_rows
            and {row["scenario_id"] for row in scenarios} == set(SCENARIOS)
            and {row["context_tokens"] for row in scenarios} == set(CONTEXT_POINTS)
            and {row["concurrency"] for row in scenarios} == set(CONCURRENCY_POINTS)
            and all(row["output_finite"] for row in scenarios),
        },
        {"name": "fault-matrix", "ok": _fault_matrix_ok(faults)},
        {
            "name": "scheduler-concurrency-sweep",
            "ok": {row["concurrency"] for row in scheduler} == set(CONCURRENCY_POINTS)
            and all(
                row["all_admitted"]
                and row["fifo"]
                and row["overflow_rejected"]
                and row["deadline_removed"]
                and row["cancelled"]
                and row["final_stats"]["queued"] == 0
                and row["final_stats"]["inflight"] == 0
                for row in scheduler
            ),
        },
        {
            "name": "sustained-bounds",
            "ok": loopback["sustained"]["configured_wall_seconds"] >= 1800
            and loopback["sustained"]["wall_elapsed_ns"] >= 1_800_000_000_000
            and loopback["sustained"]["actual_loopback_iterations"]
            >= loopback["sustained"]["minimum_loopback_iterations"]
            and loopback["sustained"]["concurrency"] == 8
            and loopback["sustained"]["max_observed_concurrency"] == 8
            and not loopback["sustained"]["queue_bound_exceeded"]
            and not loopback["sustained"]["memory_bound_exceeded"]
            and loopback["sustained"]["min_message_credit"] >= 1,
        },
        {
            "name": "physical-spike-dispositioned",
            "ok": physical["status"] in {"available", "unavailable"}
            and physical["formal_g2_passed"] is False,
        },
    ]
    passed = all(check["ok"] for check in checks)
    artifact = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "plan_version": "v4",
        "phase": "0.5",
        "evidence_class": EVIDENCE_CLASS,
        "measurement_kind": "simulation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "assumption_ids": [f"SA-{index:03d}" for index in range(1, 11)],
        "manifests": [manifest.to_dict() | {"manifest_hash": manifest.manifest_hash} for manifest in phase05_manifests()],
        "conformance": conformance,
        "loopback": loopback,
        "scenario_matrix": scenarios,
        "fault_matrix": faults,
        "scheduler_sweep": scheduler,
        "physical_backend_spike": physical,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_count": sum(1 for check in checks if check["ok"]),
            "phase05_exit_passed": passed,
            "scenario_row_count": len(scenarios),
            "fault_count": len(faults),
            "formal_g2_passed": False,
            "physical_claim": False,
        },
        "limitations": [
            "The 30-minute sustained interval is real wall-clock T1 loopback with concurrent simulated workers; it is not physical-hardware stability evidence.",
            "The mechanism backend uses DeepSeek-V2-Lite boundary shapes and stage cuts with a deterministic reference transform; it is not compiled DeepSeek-V2-Lite MAX graph execution.",
            "The two-stage orchestrator is lockstep; admission and continuous batching are evaluated in separate simulations.",
            "Wire replay is bounded, but this artifact does not establish an end-of-request KV/state release lifecycle or indefinite-service memory boundedness.",
            "No Apple/NVIDIA supported-platform or throughput claim is made.",
        ],
    }
    if out is not None:
        write_json(out, artifact)
    return artifact


def validate_phase05_engine_v0_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1 or data.get("record_kind") != RECORD_KIND:
        errors.append("invalid Phase 0.5 evidence version/record_kind")
    if data.get("plan_version") != "v4" or data.get("phase") != "0.5":
        errors.append("evidence must bind plan v4 Phase 0.5")
    if data.get("evidence_class") != EVIDENCE_CLASS or data.get("measurement_kind") != "simulation":
        errors.append("Phase 0.5 evidence must be classified T1-simulation")
    assumption_ids = data.get("assumption_ids")
    if assumption_ids != [f"SA-{index:03d}" for index in range(1, 11)]:
        errors.append("assumption_ids must contain SA-001 through SA-010")
    manifests = data.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != 2:
        errors.append("exactly two stage manifests are required")
    else:
        expected_manifests = phase05_manifests()
        for index, row in enumerate(manifests):
            try:
                manifest_data = dict(row)
                expected_hash = manifest_data.pop("manifest_hash")
                parsed = StageManifest.from_dict(manifest_data)
                if parsed.manifest_hash != expected_hash:
                    errors.append(f"manifests[{index}] hash mismatch")
                if (
                    parsed.to_dict() != expected_manifests[index].to_dict()
                    or expected_hash != expected_manifests[index].manifest_hash
                ):
                    errors.append(
                        f"manifests[{index}] does not match the Phase 0.5 mechanism target"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"manifests[{index}] invalid: {exc}")
    conformance = data.get("conformance") if isinstance(data.get("conformance"), dict) else {}
    if conformance.get("check_count") != conformance.get("passed_count") or not conformance.get("checks"):
        errors.append("Stage ABI/backend conformance is incomplete")
    loopback = data.get("loopback") if isinstance(data.get("loopback"), dict) else {}
    if not loopback.get("independent_processes") or loopback.get("worker_count") != 2:
        errors.append("loopback evidence requires two independent processes")
    if loopback.get("prefill_status") != "ok" or loopback.get("decode_status") != "ok":
        errors.append("loopback prefill/decode must pass")
    control_plane = loopback.get("control_plane")
    if not isinstance(control_plane, list) or len(control_plane) != 2 or not all(
        isinstance(row, dict)
        and row.get("health", {}).get("state") == "READY"
        and row.get("capabilities", {}).get("backend") == "simulated-max"
        and row.get("install", {}).get("unchanged") is True
        for row in control_plane
    ):
        errors.append("two-plane HTTP control evidence is incomplete")
    current_capability_attestation = isinstance(control_plane, list) and len(
        control_plane
    ) == 2 and all(
        isinstance(row, dict)
        and isinstance(row.get("capabilities"), dict)
        and row["capabilities"].get("capability_source") == "backend"
        and isinstance(row["capabilities"].get("attestation"), dict)
        and row["capabilities"]["attestation"].get("compatible") is True
        and row["capabilities"]["attestation"].get("checked_before_load") is True
        and isinstance(
            row["capabilities"]["attestation"].get("observed"), dict
        )
        and row["capabilities"]["attestation"]["observed"].get("source")
        == "backend"
        for row in control_plane
    )
    if float(loopback.get("max_abs_error", math.inf)) > 0.02:
        errors.append("loopback output exceeds BF16 tolerance")
    if loopback.get("reference_top1") != loopback.get("candidate_top1"):
        errors.append("loopback top-1 differs from reference")
    if loopback.get("cleanup_completed") is not True:
        errors.append("worker/channel cleanup did not complete")
    ledger = loopback.get("evidence_ledger") if isinstance(loopback.get("evidence_ledger"), dict) else {}
    trace_samples = ledger.get("trace_samples") if isinstance(ledger.get("trace_samples"), list) else []
    channel_metrics = ledger.get("channel_metrics") if isinstance(ledger.get("channel_metrics"), list) else []
    if (
        ledger.get("evidence_class") != EVIDENCE_CLASS
        or ledger.get("plan_id") != PLAN_ID
        or ledger.get("plan_hash") != PLAN_HASH
        or ledger.get("activation_contents_logged") is not False
        or not trace_samples
        or not all(
            isinstance(row, dict)
            and row.get("request_id")
            and row.get("plan_id")
            and row.get("stage_id")
            and row.get("trace_id")
            and row.get("span_id")
            for row in trace_samples
        )
        or not channel_metrics
        or sum(
            int(row.get("payload_bytes_sent", 0))
            for row in channel_metrics
            if isinstance(row, dict)
        )
        <= 0
    ):
        errors.append("trace/resource/evidence ledger is incomplete")
    sustained = loopback.get("sustained") if isinstance(loopback.get("sustained"), dict) else {}
    if sustained.get("duration_kind") != "wall-clock-with-real-concurrent-loopback":
        errors.append("sustained duration kind must remain explicit")
    if (
        int(sustained.get("configured_wall_seconds", 0)) < 1800
        or int(sustained.get("wall_elapsed_ns", 0)) < 1_800_000_000_000
        or sustained.get("concurrency") != 8
        or sustained.get("max_observed_concurrency") != 8
    ):
        errors.append(
            "sustained run must cover 1800 wall-clock seconds at observed concurrency 8"
        )
    if int(sustained.get("actual_loopback_iterations", 0)) < int(
        sustained.get("minimum_loopback_iterations", 0)
    ):
        errors.append("sustained run did not reach its minimum real loopback iterations")
    if sustained.get("queue_bound_exceeded") or sustained.get("memory_bound_exceeded"):
        errors.append("sustained run exceeded configured bounds")
    if int(sustained.get("max_message_credit", -1)) > int(
        sustained.get("configured_message_credit", -1)
    ):
        errors.append("message credit grew beyond configured capacity")
    if int(sustained.get("max_byte_credit", -1)) > int(
        sustained.get("configured_byte_credit", -1)
    ):
        errors.append("byte credit grew beyond configured capacity")
    if int(sustained.get("max_observed_process_rss_bytes", -1)) > int(
        sustained.get("configured_process_rss_limit_bytes", -1)
    ):
        errors.append("process RSS grew beyond configured capacity")
    worker_rss_samples = sustained.get("worker_rss_samples")
    if not isinstance(worker_rss_samples, list) or len(worker_rss_samples) < 4:
        errors.append("sustained run lacks worker RSS samples")
    scenarios = data.get("scenario_matrix")
    if not isinstance(scenarios, list):
        errors.append("scenario_matrix must be a list")
        scenarios = []
    expected_rows = len(SCENARIOS) * len(CONTEXT_POINTS) * len(CONCURRENCY_POINTS)
    if len(scenarios) != expected_rows:
        errors.append(f"scenario_matrix must contain {expected_rows} rows")
    if {row.get("scenario_id") for row in scenarios if isinstance(row, dict)} != set(SCENARIOS):
        errors.append("scenario_matrix is missing named scenarios")
    if {row.get("context_tokens") for row in scenarios if isinstance(row, dict)} != set(CONTEXT_POINTS):
        errors.append("scenario_matrix is missing context points")
    if {row.get("concurrency") for row in scenarios if isinstance(row, dict)} != set(CONCURRENCY_POINTS):
        errors.append("scenario_matrix is missing concurrency points")
    faults = data.get("fault_matrix")
    if not isinstance(faults, list) or {row.get("fault") for row in faults if isinstance(row, dict)} != set(FAULTS):
        errors.append("fault_matrix is incomplete")
    elif not _fault_matrix_ok(faults):
        errors.append("fault_matrix outcome mismatch")
    scheduler = data.get("scheduler_sweep")
    if not isinstance(scheduler, list) or {
        row.get("concurrency") for row in scheduler if isinstance(row, dict)
    } != set(CONCURRENCY_POINTS):
        errors.append("scheduler_sweep is missing concurrency 1/4/8")
    elif not all(
        row.get("all_admitted")
        and row.get("fifo")
        and row.get("overflow_rejected")
        and row.get("deadline_removed")
        and row.get("cancelled")
        and row.get("final_stats", {}).get("queued") == 0
        and row.get("final_stats", {}).get("inflight") == 0
        for row in scheduler
    ):
        errors.append("scheduler_sweep fairness/bounds/cleanup failed")
    physical = data.get("physical_backend_spike") if isinstance(data.get("physical_backend_spike"), dict) else {}
    if physical.get("status") not in {"available", "unavailable"}:
        errors.append("physical backend spike requires explicit disposition")
    if physical.get("formal_g2_passed") is not False:
        errors.append("Phase 0.5 evidence cannot claim G2")
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks or not all(check.get("ok") is True for check in checks if isinstance(check, dict)):
        errors.append("all Phase 0.5 exit checks must pass")
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    if summary.get("phase05_exit_passed") is not True:
        errors.append("summary.phase05_exit_passed must be true")
    if summary.get("formal_g2_passed") is not False or summary.get("physical_claim") is not False:
        errors.append("summary must not make physical/G2 claims")
    warnings.append(
        "Phase 0.5 is T1 simulation/loopback evidence; physical MAX correctness and G2 remain open."
    )
    conformance_names = {
        check.get("name")
        for check in conformance.get("checks", [])
        if isinstance(check, dict)
    }
    current_contract_authority = (
        CURRENT_STAGE_CONFORMANCE_CHECKS.issubset(conformance_names)
        and current_capability_attestation
    )
    if not current_contract_authority:
        warnings.append(
            "Historical artifact remains valid for its recorded T1 closure, but it "
            "predates the current backend-attestation and bounded-replay contract; "
            "it is not current-contract authority."
        )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "phase05_exit_passed": summary.get("phase05_exit_passed") is True,
            "check_count": summary.get("check_count"),
            "scenario_row_count": len(scenarios),
            "fault_count": len(faults) if isinstance(faults, list) else 0,
            "worker_count": loopback.get("worker_count"),
            "max_abs_error": loopback.get("max_abs_error"),
            "formal_g2_passed": summary.get("formal_g2_passed") is True,
            "current_contract_authority": current_contract_authority,
        },
    }


def validate_phase05_engine_v0(path: str | Path) -> dict[str, Any]:
    try:
        data = read_json(path)
    except Exception as exc:  # noqa: BLE001 - validator reports artifact errors.
        return {
            "ok": False,
            "errors": [f"invalid Phase 0.5 artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["Phase 0.5 artifact must be an object"],
            "warnings": [],
            "summary": {},
            "fixture": str(path),
        }
    result = validate_phase05_engine_v0_fixture(data)
    result["fixture"] = str(path)
    return result


def validate_stage_abi_golden(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    errors: list[str] = []
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture errors.
        return {
            "ok": False,
            "errors": [f"invalid Stage ABI golden: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("record_kind") != "stage-abi-v1-golden":
        errors.append("invalid Stage ABI golden version/record_kind")
        data = data if isinstance(data, dict) else {}
    try:
        shape = tuple(int(item) for item in data.get("shape", []))
    except (TypeError, ValueError):
        shape = ()
        errors.append("shape must contain integers")
    try:
        if data.get("input_generator") == "phase05_tensor_v1":
            if len(shape) != 2:
                raise ValueError("phase05_tensor_v1 requires a rank-2 shape")
            tensor = phase05_tensor(shape[0], shape[1])
            if data.get("dtype") != tensor.descriptor.dtype:
                raise ValueError("golden dtype differs from phase05_tensor_v1")
        elif "input_values" in data:
            tensor = Tensor.from_values(
                data.get("input_values", []),
                kind="activation",
                dtype=str(data.get("dtype", "")),
                shape=shape,
            )
        else:
            raise ValueError("unknown or missing input_generator")
    except (TypeError, ValueError) as exc:
        errors.append(f"golden input tensor invalid: {exc}")
        tensor = phase05_tensor(2)
    input_payload_hash = "sha256:" + hashlib.sha256(tensor.payload).hexdigest()
    if input_payload_hash != data.get("expected_input_payload_sha256"):
        errors.append("input tensor payload differs from golden")
    row_mapping = data.get("row_mapping")
    if not isinstance(row_mapping, list) or len(row_mapping) != shape[0] or any(
        not isinstance(row, dict)
        or not row.get("request_id")
        or not isinstance(row.get("token_position"), int)
        for row in row_mapping or []
    ):
        errors.append("row_mapping must bind every tensor row")
    manifests = phase05_manifests(shape[1] if len(shape) == 2 else 8)
    manifest_hashes = [manifest.manifest_hash for manifest in manifests]
    if manifest_hashes != data.get("expected_manifest_hashes"):
        errors.append("manifest hashes differ from golden")
    metadata = _frame_metadata(manifests[0])
    metadata["deadline_ns"] = int(data.get("fixed_deadline_ns", 0))
    frame = Frame.from_tensor(tensor, sequence_no=0, metadata=metadata)
    frame_hash = "sha256:" + hashlib.sha256(encode_frame(frame)).hexdigest()
    if frame_hash != data.get("expected_activation_frame_sha256"):
        errors.append("activation frame bytes differ from golden")
    final = _reference_pipeline(manifests, tensor)
    final_payload_hash = "sha256:" + hashlib.sha256(final.payload).hexdigest()
    if final_payload_hash != data.get("expected_final_payload_sha256"):
        errors.append("reference final payload differs from golden")
    expected_first16 = tuple(
        float(value) for value in data.get("expected_final_first16", [])
    )
    if len(expected_first16) != 16 or final.values()[:16] != expected_first16:
        errors.append("reference final first16 values differ from golden")
    if _top1(final) != data.get("expected_top1"):
        errors.append("reference top-1 differs from golden")
    conformance = run_stage_conformance()
    by_name = {row["name"]: row for row in conformance["checks"]}
    required = data.get("required_conformance_checks")
    if not isinstance(required, list) or not required:
        errors.append("required_conformance_checks must be a non-empty list")
        required = []
    missing = [name for name in required if name not in by_name]
    failing = [name for name in required if name in by_name and not by_name[name]["ok"]]
    if missing:
        errors.append(f"missing conformance checks: {missing}")
    if failing:
        errors.append(f"failing conformance checks: {failing}")
    related = data.get("related_logical_contracts")

    def related_contract_exists(item: Any) -> bool:
        if not isinstance(item, str):
            return False
        declared = Path(item)
        if declared.exists():
            return True
        parts = declared.parts
        if parts and parts[0] == "fornax":
            return Path(__file__).resolve().parent.joinpath(*parts[1:]).exists()
        return False

    if not isinstance(related, list) or not all(
        related_contract_exists(item) for item in related
    ):
        errors.append("related logical contract paths must exist")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": [
            "Stage ABI golden is T0/T1 contract evidence, not physical MAX conformance."
        ],
        "summary": {
            "manifest_count": len(manifests),
            "tensor_rows": shape[0] if shape else 0,
            "conformance_check_count": len(required),
            "input_payload_sha256": input_payload_hash,
            "frame_sha256": frame_hash,
            "final_payload_sha256": final_payload_hash,
            "first16_max_abs_error": _max_abs_error(
                final.values()[:16], expected_first16
            ),
        },
        "fixture": str(fixture_path),
    }
