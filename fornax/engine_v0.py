from __future__ import annotations

import multiprocessing
import http.client
import http.server
import json
import os
import resource
import socket
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from typing import Any

from .stage_abi import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    Frame,
    FrameError,
    MessageKind,
    SequenceTracker,
    read_frame,
    send_frame,
    validate_frame_identity,
)
from .stage_runtime import (
    SimulationProfile,
    SimulatedMaxStageBackend,
    StageManifest,
    StageRequest,
    StageResult,
    StageRuntimeError,
    Tensor,
)


class EngineV0Error(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def _process_max_rss_bytes() -> int:
    """Return this process's observed peak RSS using platform-normalized bytes."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _control_frame(kind: MessageKind, sequence_no: int, **metadata: Any) -> Frame:
    value = {"sequence_no": sequence_no}
    value.update(metadata)
    return Frame(kind=kind, sequence_no=sequence_no, metadata=value)


def _result_error_frame(result: StageResult, sequence_no: int) -> Frame:
    error = result.error or {"code": "EXECUTION", "message": result.status}
    return _control_frame(
        MessageKind.ERROR,
        sequence_no,
        request_id=result.request_id,
        microbatch_id=result.microbatch_id,
        status=result.status,
        code=error["code"],
        message=error["message"],
        kv_epoch_before=result.kv_epoch_before,
        kv_epoch_after=result.kv_epoch_after,
    )


def _serve_connection(
    connection: socket.socket,
    *,
    backend: SimulatedMaxStageBackend,
    handle: Any,
    manifest: StageManifest,
    max_payload_bytes: int,
) -> bool:
    incoming = SequenceTracker()
    outgoing_sequence = 0
    negotiated = False
    connection.settimeout(10.0)
    while True:
        try:
            frame = read_frame(connection, max_payload_bytes=max_payload_bytes)
            incoming.accept(frame)
        except (FrameError, OSError, socket.timeout) as exc:
            if isinstance(exc, FrameError):
                try:
                    send_frame(
                        connection,
                        _control_frame(
                            MessageKind.ERROR,
                            outgoing_sequence,
                            request_id="channel",
                            microbatch_id="channel",
                            status="failed",
                            code=exc.code,
                            message=exc.message,
                        ),
                        max_payload_bytes=max_payload_bytes,
                    )
                except (FrameError, OSError):
                    pass
            return True

        control = frame.metadata.get("control")
        if not negotiated:
            if frame.kind != MessageKind.HEARTBEAT or control != "negotiate":
                return True
            if (
                frame.metadata.get("plan_id") != manifest.plan_id
                or frame.metadata.get("plan_hash") != manifest.plan_hash
                or frame.metadata.get("manifest_hash") != manifest.manifest_hash
                or frame.metadata.get("destination_stage") != manifest.stage_id
                or frame.metadata.get("abi_major") != 1
            ):
                send_frame(
                    connection,
                    _control_frame(
                        MessageKind.ERROR,
                        outgoing_sequence,
                        request_id="channel",
                        microbatch_id="channel",
                        status="rejected",
                        code="STALE_PLAN",
                        message="channel negotiation identity mismatch",
                    ),
                    max_payload_bytes=max_payload_bytes,
                )
                return True
            send_frame(
                connection,
                _control_frame(
                    MessageKind.HEARTBEAT,
                    outgoing_sequence,
                    control="ready",
                    stage_id=manifest.stage_id,
                    manifest_hash=manifest.manifest_hash,
                    pid=os.getpid(),
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
            send_frame(
                connection,
                _control_frame(
                    MessageKind.CREDIT,
                    outgoing_sequence,
                    messages=1,
                    bytes=max_payload_bytes,
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
            negotiated = True
            continue

        if frame.kind == MessageKind.HEARTBEAT:
            if control == "health":
                health = backend.health(handle)
                send_frame(
                    connection,
                    _control_frame(
                        MessageKind.HEARTBEAT,
                        outgoing_sequence,
                        control="health",
                        state=health.state,
                        inflight=health.inflight,
                        stage_id=health.stage_id,
                    ),
                    max_payload_bytes=max_payload_bytes,
                )
                outgoing_sequence += 1
                continue
            if control == "shutdown":
                backend.drain(handle, time.monotonic_ns() + 1_000_000_000)
                backend.unload(handle)
                send_frame(
                    connection,
                    _control_frame(
                        MessageKind.HEARTBEAT,
                        outgoing_sequence,
                        control="shutdown-complete",
                        stage_id=manifest.stage_id,
                    ),
                    max_payload_bytes=max_payload_bytes,
                )
                return False

        if frame.kind == MessageKind.CANCEL:
            request_id = str(frame.metadata["request_id"])
            result = backend.cancel(
                handle, request_id, str(frame.metadata.get("reason", "remote cancel"))
            )
            send_frame(
                connection,
                _control_frame(
                    MessageKind.ACK,
                    outgoing_sequence,
                    request_id=request_id,
                    microbatch_id=str(frame.metadata["microbatch_id"]),
                    ack_sequence=frame.sequence_no,
                    cancelled=result.cancelled,
                    execution_started=result.execution_started,
                    kv_mutated=result.kv_mutated,
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
            continue

        if frame.kind not in {MessageKind.ACTIVATION, MessageKind.LOGITS}:
            send_frame(
                connection,
                _control_frame(
                    MessageKind.ERROR,
                    outgoing_sequence,
                    request_id=str(frame.metadata.get("request_id", "channel")),
                    microbatch_id=str(frame.metadata.get("microbatch_id", "channel")),
                    status="rejected",
                    code="ABI_VERSION",
                    message=f"unexpected application frame {frame.kind.name}",
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
            continue

        try:
            validate_frame_identity(
                frame,
                plan_id=manifest.plan_id,
                plan_hash=manifest.plan_hash,
                manifest_hash=manifest.manifest_hash,
                destination_stage=manifest.stage_id,
            )
            tensor = frame.tensor()
            request = StageRequest(
                plan_id=str(frame.metadata["plan_id"]),
                plan_hash=str(frame.metadata["plan_hash"]),
                request_id=str(frame.metadata["request_id"]),
                microbatch_id=str(frame.metadata["microbatch_id"]),
                sequence_no=int(frame.metadata["request_sequence_no"]),
                phase=str(frame.metadata["phase"]),
                token_start=int(frame.metadata["token_start"]),
                token_count=int(frame.metadata["token_count"]),
                input_activation=tensor,
                kv_epoch=int(frame.metadata["kv_epoch"]),
                deadline_ns=int(frame.metadata["deadline_ns"]),
                trace_context={
                    "trace_id": str(frame.metadata["trace_id"]),
                    "span_id": str(frame.metadata["span_id"]),
                },
            )
            result = backend.execute(handle, request)
        except FrameError as exc:
            send_frame(
                connection,
                _control_frame(
                    MessageKind.ERROR,
                    outgoing_sequence,
                    request_id=str(frame.metadata.get("request_id", "channel")),
                    microbatch_id=str(frame.metadata.get("microbatch_id", "channel")),
                    status="rejected",
                    code=exc.code,
                    message=exc.message,
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
            send_frame(
                connection,
                _control_frame(
                    MessageKind.CREDIT,
                    outgoing_sequence,
                    messages=1,
                    bytes=len(frame.payload),
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
            continue
        except StageRuntimeError as exc:
            send_frame(
                connection,
                _control_frame(
                    MessageKind.ERROR,
                    outgoing_sequence,
                    request_id=str(frame.metadata.get("request_id", "channel")),
                    microbatch_id=str(frame.metadata.get("microbatch_id", "channel")),
                    status="failed",
                    code=exc.code,
                    message=exc.message,
                ),
                max_payload_bytes=max_payload_bytes,
            )
            return True

        send_frame(
            connection,
            _control_frame(
                MessageKind.ACK,
                outgoing_sequence,
                request_id=result.request_id,
                microbatch_id=result.microbatch_id,
                ack_sequence=frame.sequence_no,
                payload_crc=int(frame.crc or 0),
            ),
            max_payload_bytes=max_payload_bytes,
        )
        outgoing_sequence += 1
        if result.status != "ok" or result.output_tensor is None:
            send_frame(
                connection,
                _result_error_frame(result, outgoing_sequence),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
        else:
            metadata = {
                "plan_id": result.plan_id,
                "plan_hash": result.plan_hash,
                "manifest_hash": manifest.manifest_hash,
                "request_id": result.request_id,
                "microbatch_id": result.microbatch_id,
                "source_stage": manifest.stage_id,
                "destination_stage": "orchestrator",
                "phase": request.phase,
                "token_start": request.token_start,
                "token_count": request.token_count,
                "kv_epoch": result.kv_epoch_after,
                "deadline_ns": request.deadline_ns,
                "trace_id": request.trace_context["trace_id"],
                "span_id": request.trace_context["span_id"],
                "request_sequence_no": result.sequence_no,
                "status": result.status,
                "kv_epoch_before": result.kv_epoch_before,
                "kv_epoch_after": result.kv_epoch_after,
                "timings_ns": result.timings_ns,
            }
            send_frame(
                connection,
                Frame.from_tensor(
                    result.output_tensor,
                    sequence_no=outgoing_sequence,
                    metadata=metadata,
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing_sequence += 1
        send_frame(
            connection,
            _control_frame(
                MessageKind.CREDIT,
                outgoing_sequence,
                messages=1,
                bytes=len(frame.payload),
            ),
            max_payload_bytes=max_payload_bytes,
        )
        outgoing_sequence += 1


def _control_handler(
    backend: SimulatedMaxStageBackend,
    handle: Any,
    manifest: StageManifest,
) -> type[http.server.BaseHTTPRequestHandler]:
    class ControlHandler(http.server.BaseHTTPRequestHandler):
        server_version = "FornaxEngineV0/1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _write(self, status: int, value: dict[str, Any]) -> None:
            payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            if self.headers.get("X-Fornax-Node-ID") != "orchestrator":
                self._write(403, {"ok": False, "code": "NODE_IDENTITY"})
                return False
            if self.headers.get("X-Fornax-Plan-Hash") != manifest.plan_hash:
                self._write(409, {"ok": False, "code": "STALE_PLAN"})
                return False
            return True

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0 or length > 64 * 1024:
                raise ValueError("control body exceeds 64 KiB")
            payload = self.rfile.read(length)
            if not payload:
                return {}
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("control body must be an object")
            return value

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            if not self._authorized():
                return
            if self.path == "/health":
                health = backend.health(handle)
                self._write(
                    200,
                    {
                        "ok": True,
                        "state": health.state,
                        "stage_id": health.stage_id,
                        "manifest_hash": health.manifest_hash,
                        "inflight": health.inflight,
                        "degraded": health.degraded,
                    },
                )
                return
            if self.path == "/capabilities":
                self._write(
                    200,
                    {
                        "ok": True,
                        "node_id": manifest.device_requirement.get("device_identity"),
                        "stage_id": manifest.stage_id,
                        "backend": backend.backend_name,
                        "build_id": manifest.max_build_id,
                        "abi_major": manifest.fornax_abi_major,
                        "abi_minor": manifest.fornax_abi_minor,
                        "dtypes": manifest.device_requirement.get("dtypes", []),
                        "max_frame_bytes": DEFAULT_MAX_PAYLOAD_BYTES,
                    },
                )
                return
            if self.path == "/status":
                self._write(
                    200,
                    {
                        "ok": True,
                        "plan_id": manifest.plan_id,
                        "plan_hash": manifest.plan_hash,
                        "stage_id": manifest.stage_id,
                        "manifest_hash": manifest.manifest_hash,
                        "process_max_rss_bytes": _process_max_rss_bytes(),
                    },
                )
                return
            self._write(404, {"ok": False, "code": "NOT_FOUND"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            if not self._authorized():
                return
            try:
                body = self._body()
                if self.path == "/plan/install":
                    if body.get("manifest_hash") != manifest.manifest_hash:
                        self._write(409, {"ok": False, "code": "STALE_PLAN"})
                        return
                    self._write(
                        200,
                        {
                            "ok": True,
                            "installed": True,
                            "unchanged": True,
                            "manifest_hash": manifest.manifest_hash,
                        },
                    )
                    return
                if self.path == "/cancel":
                    result = backend.cancel(
                        handle,
                        str(body.get("request_id", "")),
                        str(body.get("reason", "control cancellation")),
                    )
                    self._write(200, {"ok": True, **asdict(result)})
                    return
                if self.path == "/drain":
                    result = backend.drain(
                        handle, int(body.get("deadline_ns", time.monotonic_ns()))
                    )
                    self._write(200, {"ok": result.drained, **asdict(result)})
                    return
            except (ValueError, StageRuntimeError, json.JSONDecodeError) as exc:
                self._write(
                    400,
                    {
                        "ok": False,
                        "code": getattr(exc, "code", "METADATA"),
                        "message": str(exc),
                    },
                )
                return
            self._write(404, {"ok": False, "code": "NOT_FOUND"})

    return ControlHandler


def _worker_main(
    manifest_data: dict[str, Any],
    profile_data: dict[str, Any],
    ready: Connection,
    host: str,
    max_payload_bytes: int,
) -> None:
    listener: socket.socket | None = None
    control_server: http.server.ThreadingHTTPServer | None = None
    control_thread: threading.Thread | None = None
    try:
        manifest = StageManifest.from_dict(manifest_data)
        profile = SimulationProfile(**profile_data)
        backend = SimulatedMaxStageBackend(profile)
        handle = backend.load(manifest)
        control_server = http.server.ThreadingHTTPServer(
            (host, 0), _control_handler(backend, handle, manifest)
        )
        control_thread = threading.Thread(
            target=control_server.serve_forever,
            name=f"fornax-control-{manifest.stage_id}",
            daemon=True,
        )
        control_thread.start()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        listener.listen(4)
        listener.settimeout(0.5)
        port = int(listener.getsockname()[1])
        ready.send(
            {
                "ok": True,
                "host": host,
                "port": port,
                "control_port": int(control_server.server_address[1]),
                "pid": os.getpid(),
                "stage_id": manifest.stage_id,
                "manifest_hash": manifest.manifest_hash,
            }
        )
        keep_running = True
        while keep_running:
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            with connection:
                keep_running = _serve_connection(
                    connection,
                    backend=backend,
                    handle=handle,
                    manifest=manifest,
                    max_payload_bytes=max_payload_bytes,
                )
    except Exception as exc:  # noqa: BLE001 - startup errors cross process boundary.
        try:
            ready.send({"ok": False, "error": repr(exc)})
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        if control_server is not None:
            control_server.shutdown()
            control_server.server_close()
        if control_thread is not None:
            control_thread.join(timeout=2.0)
        if listener is not None:
            listener.close()
        ready.close()


@dataclass
class WorkerProcess:
    manifest: StageManifest
    profile: SimulationProfile
    host: str = "127.0.0.1"
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    process: multiprocessing.Process | None = field(init=False, default=None)
    endpoint: dict[str, Any] | None = field(init=False, default=None)

    def start(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        if self.process is not None:
            raise EngineV0Error("WORKER_STATE", "worker is already started")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_main,
            args=(
                self.manifest.to_dict(),
                asdict(self.profile),
                child,
                self.host,
                self.max_payload_bytes,
            ),
            name=f"fornax-{self.manifest.stage_id}",
        )
        process.start()
        child.close()
        if not parent.poll(timeout_s):
            process.terminate()
            process.join(timeout=2.0)
            raise EngineV0Error("WORKER_START", "worker startup timed out")
        endpoint = parent.recv()
        parent.close()
        if not endpoint.get("ok"):
            process.join(timeout=2.0)
            raise EngineV0Error("WORKER_START", str(endpoint.get("error")))
        self.process = process
        self.endpoint = endpoint
        return endpoint

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def join(self, timeout_s: float = 5.0) -> None:
        if self.process is None:
            return
        self.process.join(timeout_s)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
        self.process = None


@dataclass
class StageChannel:
    manifest: StageManifest
    host: str
    port: int
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    timeout_s: float = 10.0
    channel: socket.socket | None = field(init=False, default=None)
    outgoing_sequence: int = field(init=False, default=0)
    incoming: SequenceTracker = field(init=False, default_factory=SequenceTracker)
    message_credit: int = field(init=False, default=0)
    byte_credit: int = field(init=False, default=0)
    events: list[dict[str, Any]] = field(init=False, default_factory=list)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def _record(self, kind: str, **fields: Any) -> None:
        event = {"kind": kind, "timestamp_ns": time.monotonic_ns()}
        event.update(fields)
        self.events.append(event)

    def connect(self) -> None:
        if self.channel is not None:
            raise EngineV0Error("CHANNEL_STATE", "channel is already connected")
        channel = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        channel.settimeout(self.timeout_s)
        self.channel = channel
        self.outgoing_sequence = 0
        self.incoming = SequenceTracker()
        negotiate = _control_frame(
            MessageKind.HEARTBEAT,
            self.outgoing_sequence,
            control="negotiate",
            abi_major=1,
            abi_minor=0,
            plan_id=self.manifest.plan_id,
            plan_hash=self.manifest.plan_hash,
            manifest_hash=self.manifest.manifest_hash,
            destination_stage=self.manifest.stage_id,
            node_id="orchestrator",
            build_id="fornax-python-engine-v0",
        )
        send_frame(channel, negotiate, max_payload_bytes=self.max_payload_bytes)
        self.outgoing_sequence += 1
        ready = self._read()
        if ready.kind == MessageKind.ERROR:
            raise EngineV0Error(str(ready.metadata.get("code")), str(ready.metadata.get("message")))
        if ready.kind != MessageKind.HEARTBEAT or ready.metadata.get("control") != "ready":
            raise EngineV0Error("NEGOTIATION", "worker did not enter READY")
        credit = self._read()
        self._apply_credit(credit)
        self._record("channel_ready", stage_id=self.manifest.stage_id)

    def _require_channel(self) -> socket.socket:
        if self.channel is None:
            raise EngineV0Error("CHANNEL_STATE", "channel is not connected")
        return self.channel

    def _read(self) -> Frame:
        frame = read_frame(self._require_channel(), max_payload_bytes=self.max_payload_bytes)
        self.incoming.accept(frame)
        self._record(
            "frame_received",
            stage_id=self.manifest.stage_id,
            frame_kind=frame.kind.name,
            sequence_no=frame.sequence_no,
            payload_bytes=len(frame.payload),
        )
        return frame

    def _apply_credit(self, frame: Frame) -> None:
        if frame.kind != MessageKind.CREDIT:
            raise EngineV0Error("NO_CREDIT", "expected worker credit frame")
        self.message_credit += int(frame.metadata["messages"])
        self.byte_credit += int(frame.metadata["bytes"])
        self._record(
            "credit_received",
            stage_id=self.manifest.stage_id,
            message_credit=self.message_credit,
            byte_credit=self.byte_credit,
        )

    def execute(self, request: StageRequest, *, source_stage: str = "orchestrator") -> StageResult:
        with self._lock:
            payload_bytes = len(request.input_activation.payload)
            if self.message_credit < 1 or self.byte_credit < payload_bytes:
                raise EngineV0Error("NO_CREDIT", "channel credit exhausted", retryable=True)
            metadata = {
                "plan_id": request.plan_id,
                "plan_hash": request.plan_hash,
                "manifest_hash": self.manifest.manifest_hash,
                "request_id": request.request_id,
                "microbatch_id": request.microbatch_id,
                "source_stage": source_stage,
                "destination_stage": self.manifest.stage_id,
                "phase": request.phase,
                "token_start": request.token_start,
                "token_count": request.token_count,
                "kv_epoch": request.kv_epoch,
                "deadline_ns": request.deadline_ns,
                "trace_id": request.trace_context["trace_id"],
                "span_id": request.trace_context["span_id"],
                "request_sequence_no": request.sequence_no,
            }
            frame = Frame.from_tensor(
                request.input_activation,
                sequence_no=self.outgoing_sequence,
                metadata=metadata,
            )
            self.message_credit -= 1
            self.byte_credit -= payload_bytes
            sent_bytes = send_frame(
                self._require_channel(), frame, max_payload_bytes=self.max_payload_bytes
            )
            self._record(
                "frame_sent",
                stage_id=self.manifest.stage_id,
                frame_kind=frame.kind.name,
                sequence_no=frame.sequence_no,
                payload_bytes=payload_bytes,
                wire_bytes=sent_bytes,
            )
            self.outgoing_sequence += 1

            ack = self._read()
            if ack.kind == MessageKind.ERROR:
                result = StageResult(
                    plan_id=request.plan_id,
                    plan_hash=request.plan_hash,
                    request_id=request.request_id,
                    microbatch_id=request.microbatch_id,
                    sequence_no=request.sequence_no,
                    status=str(ack.metadata.get("status", "failed")),
                    output_kind=None,
                    output_tensor=None,
                    kv_epoch_before=request.kv_epoch,
                    kv_epoch_after=request.kv_epoch,
                    timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                    error={
                        "code": str(ack.metadata.get("code", "EXECUTION")),
                        "message": str(ack.metadata.get("message", "worker error")),
                    },
                )
                if result.error["code"] in {"STALE_PLAN", "TENSOR_CONTRACT", "METADATA"}:
                    self._apply_credit(self._read())
                return result
            if ack.kind != MessageKind.ACK or ack.metadata.get("ack_sequence") != frame.sequence_no:
                raise EngineV0Error("SEQUENCE", "worker did not ACK exact input frame")
            response = self._read()
            if response.kind == MessageKind.ERROR:
                result = StageResult(
                    plan_id=request.plan_id,
                    plan_hash=request.plan_hash,
                    request_id=request.request_id,
                    microbatch_id=request.microbatch_id,
                    sequence_no=request.sequence_no,
                    status=str(response.metadata.get("status", "failed")),
                    output_kind=None,
                    output_tensor=None,
                    kv_epoch_before=int(response.metadata.get("kv_epoch_before", request.kv_epoch)),
                    kv_epoch_after=int(response.metadata.get("kv_epoch_after", request.kv_epoch)),
                    timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                    error={
                        "code": str(response.metadata.get("code", "EXECUTION")),
                        "message": str(response.metadata.get("message", "worker error")),
                    },
                )
            else:
                validate_frame_identity(
                    response,
                    plan_id=self.manifest.plan_id,
                    plan_hash=self.manifest.plan_hash,
                    manifest_hash=self.manifest.manifest_hash,
                    destination_stage="orchestrator",
                )
                output = response.tensor()
                result = StageResult(
                    plan_id=request.plan_id,
                    plan_hash=request.plan_hash,
                    request_id=request.request_id,
                    microbatch_id=request.microbatch_id,
                    sequence_no=request.sequence_no,
                    status="ok",
                    output_kind=output.descriptor.kind,
                    output_tensor=output,
                    kv_epoch_before=int(response.metadata["kv_epoch_before"]),
                    kv_epoch_after=int(response.metadata["kv_epoch_after"]),
                    timings_ns={
                        key: int(value)
                        for key, value in dict(response.metadata["timings_ns"]).items()
                    },
                )
            self._apply_credit(self._read())
            return result

    def cancel(self, request_id: str, microbatch_id: str, reason: str) -> dict[str, Any]:
        with self._lock:
            frame = _control_frame(
                MessageKind.CANCEL,
                self.outgoing_sequence,
                request_id=request_id,
                microbatch_id=microbatch_id,
                reason=reason,
            )
            send_frame(self._require_channel(), frame, max_payload_bytes=self.max_payload_bytes)
            self.outgoing_sequence += 1
            response = self._read()
            if response.kind != MessageKind.ACK:
                raise EngineV0Error("CANCELLED", "worker did not acknowledge cancellation")
            return dict(response.metadata)

    def shutdown(self) -> None:
        if self.channel is None:
            return
        with self._lock:
            try:
                frame = _control_frame(
                    MessageKind.HEARTBEAT,
                    self.outgoing_sequence,
                    control="shutdown",
                )
                send_frame(
                    self.channel, frame, max_payload_bytes=self.max_payload_bytes
                )
                self.outgoing_sequence += 1
                response = self._read()
                if response.metadata.get("control") != "shutdown-complete":
                    raise EngineV0Error("WORKER_STATE", "worker did not shut down cleanly")
            finally:
                self.channel.close()
                self.channel = None

    def disconnect(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None


@dataclass(frozen=True)
class WorkerControlClient:
    manifest: StageManifest
    host: str
    port: int
    timeout_s: float = 5.0

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        payload = (
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if body is not None
            else None
        )
        headers = {
            "X-Fornax-Node-ID": "orchestrator",
            "X-Fornax-Plan-Hash": self.manifest.plan_hash,
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout_s
        )
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(value, dict):
                raise EngineV0Error("METADATA", "control response is not an object")
            return response.status, value
        finally:
            connection.close()

    def health(self) -> dict[str, Any]:
        status, value = self._request("GET", "/health")
        if status != 200:
            raise EngineV0Error(str(value.get("code")), "health request failed")
        return value

    def capabilities(self) -> dict[str, Any]:
        status, value = self._request("GET", "/capabilities")
        if status != 200:
            raise EngineV0Error(str(value.get("code")), "capability request failed")
        return value

    def status(self) -> dict[str, Any]:
        status, value = self._request("GET", "/status")
        if status != 200:
            raise EngineV0Error(str(value.get("code")), "status request failed")
        return value

    def install(self) -> dict[str, Any]:
        status, value = self._request(
            "POST", "/plan/install", {"manifest_hash": self.manifest.manifest_hash}
        )
        if status != 200:
            raise EngineV0Error(str(value.get("code")), "plan install failed")
        return value

    def cancel(self, request_id: str, reason: str) -> dict[str, Any]:
        status, value = self._request(
            "POST", "/cancel", {"request_id": request_id, "reason": reason}
        )
        if status != 200:
            raise EngineV0Error(str(value.get("code")), "cancel request failed")
        return value

    def drain(self, deadline_ns: int) -> dict[str, Any]:
        status, value = self._request(
            "POST", "/drain", {"deadline_ns": deadline_ns}
        )
        if status != 200:
            raise EngineV0Error(str(value.get("code")), "drain request failed")
        return value


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    submitted_order: int
    deadline_ns: int
    payload: Any


class AdmissionScheduler:
    def __init__(self, *, max_inflight: int, max_queued: int, microbatch_size: int) -> None:
        if min(max_inflight, max_queued, microbatch_size) <= 0:
            raise ValueError("scheduler limits must be positive")
        self.max_inflight = max_inflight
        self.max_queued = max_queued
        self.microbatch_size = microbatch_size
        self._queue: deque[ScheduledRequest] = deque()
        self._inflight: set[str] = set()
        self._order = 0
        self.events: list[dict[str, Any]] = []

    def submit(self, request_id: str, payload: Any, deadline_ns: int) -> bool:
        if request_id in self._inflight or any(item.request_id == request_id for item in self._queue):
            raise EngineV0Error("SEQUENCE", "duplicate scheduled request")
        if len(self._queue) >= self.max_queued:
            self.events.append({"kind": "admission_rejected", "request_id": request_id})
            return False
        self._queue.append(ScheduledRequest(request_id, self._order, deadline_ns, payload))
        self._order += 1
        self.events.append({"kind": "request_queued", "request_id": request_id})
        return True

    def next_microbatch(self, now_ns: int) -> tuple[ScheduledRequest, ...]:
        admitted: list[ScheduledRequest] = []
        while self._queue and len(self._inflight) < self.max_inflight and len(admitted) < self.microbatch_size:
            item = self._queue.popleft()
            if item.deadline_ns <= now_ns:
                self.events.append({"kind": "request_deadline", "request_id": item.request_id})
                continue
            self._inflight.add(item.request_id)
            admitted.append(item)
            self.events.append({"kind": "request_admitted", "request_id": item.request_id})
        return tuple(admitted)

    def complete(self, request_id: str) -> None:
        if request_id not in self._inflight:
            raise EngineV0Error("SEQUENCE", "request is not inflight")
        self._inflight.remove(request_id)
        self.events.append({"kind": "request_completed", "request_id": request_id})

    def cancel(self, request_id: str) -> bool:
        if request_id in self._inflight:
            self._inflight.remove(request_id)
            self.events.append({"kind": "request_cancelled", "request_id": request_id})
            return True
        for item in tuple(self._queue):
            if item.request_id == request_id:
                self._queue.remove(item)
                self.events.append({"kind": "request_cancelled", "request_id": request_id})
                return True
        return False

    @property
    def stats(self) -> dict[str, int]:
        return {
            "queued": len(self._queue),
            "inflight": len(self._inflight),
            "max_queued": self.max_queued,
            "max_inflight": self.max_inflight,
        }


class EngineV0Orchestrator:
    def __init__(self, stages: list[tuple[StageManifest, StageChannel]]) -> None:
        if len(stages) < 2:
            raise ValueError("Engine v0 requires at least two stages")
        indices = [manifest.stage_index for manifest, _ in stages]
        if indices != list(range(len(stages))):
            raise ValueError("stages must be contiguous and ordered from zero")
        self.stages = stages
        self.kv_epochs: dict[tuple[str, str], int] = {}
        self.events: list[dict[str, Any]] = []

    def execute(
        self,
        tensor: Tensor,
        *,
        request_id: str,
        phase: str,
        request_sequence_no: int,
        deadline_ns: int,
        microbatch_id: str = "microbatch-0",
    ) -> StageResult:
        current = tensor
        last_result: StageResult | None = None
        source_stage = "gateway"
        for manifest, channel in self.stages:
            epoch_key = (request_id, manifest.stage_id)
            request = StageRequest(
                plan_id=manifest.plan_id,
                plan_hash=manifest.plan_hash,
                request_id=request_id,
                microbatch_id=microbatch_id,
                sequence_no=request_sequence_no,
                phase=phase,
                token_start=0,
                token_count=current.descriptor.shape[0],
                input_activation=current,
                kv_epoch=self.kv_epochs.get(epoch_key, 0),
                deadline_ns=deadline_ns,
                trace_context={
                    "trace_id": f"trace-{request_id}",
                    "span_id": f"span-{manifest.stage_id}-{request_sequence_no}",
                },
            )
            started = time.monotonic_ns()
            result = channel.execute(request, source_stage=source_stage)
            finished = time.monotonic_ns()
            self.events.append(
                {
                    "kind": "stage_result",
                    "request_id": request_id,
                    "plan_id": manifest.plan_id,
                    "plan_hash": manifest.plan_hash,
                    "stage_id": manifest.stage_id,
                    "phase": phase,
                    "status": result.status,
                    "trace_id": request.trace_context["trace_id"],
                    "span_id": request.trace_context["span_id"],
                    "elapsed_ns": max(0, finished - started),
                    "kv_epoch_before": result.kv_epoch_before,
                    "kv_epoch_after": result.kv_epoch_after,
                }
            )
            if result.status != "ok" or result.output_tensor is None:
                return result
            self.kv_epochs[epoch_key] = result.kv_epoch_after
            current = result.output_tensor
            source_stage = manifest.stage_id
            if current.descriptor.kind == "logits" and manifest.stage_index != len(self.stages) - 1:
                raise EngineV0Error("TENSOR_CONTRACT", "non-final stage returned logits")
            if current.descriptor.kind == "logits" and manifest.stage_index == len(self.stages) - 1:
                last_result = result
            else:
                last_result = result
        assert last_result is not None
        return last_result


def start_two_worker_engine(
    manifests: tuple[StageManifest, StageManifest],
    profiles: tuple[SimulationProfile, SimulationProfile],
) -> tuple[list[WorkerProcess], list[StageChannel], EngineV0Orchestrator]:
    workers = [
        WorkerProcess(manifest=stage_manifest, profile=profile)
        for stage_manifest, profile in zip(manifests, profiles)
    ]
    channels: list[StageChannel] = []
    try:
        endpoints = [worker.start() for worker in workers]
        if len({endpoint["pid"] for endpoint in endpoints}) != 2:
            raise EngineV0Error("WORKER_START", "workers are not independent processes")
        for stage_manifest, endpoint in zip(manifests, endpoints):
            channel = StageChannel(
                manifest=stage_manifest,
                host=str(endpoint["host"]),
                port=int(endpoint["port"]),
            )
            channel.connect()
            channels.append(channel)
        orchestrator = EngineV0Orchestrator(list(zip(manifests, channels)))
        return workers, channels, orchestrator
    except Exception:
        for channel in channels:
            channel.disconnect()
        for worker in workers:
            worker.join()
        raise


def stop_two_worker_engine(workers: list[WorkerProcess], channels: list[StageChannel]) -> None:
    for channel in channels:
        try:
            channel.shutdown()
        except (EngineV0Error, FrameError, OSError):
            channel.disconnect()
    for worker in workers:
        worker.join()
