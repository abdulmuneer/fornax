from __future__ import annotations

import json
import os
import socket
import threading
import time
import unittest
import uuid
from pathlib import Path

from fornax.stage_abi import (
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
from fornax.engine_v0 import (
    AdmissionScheduler,
    EngineV0Orchestrator,
    EngineV0Error,
    StageChannel,
    WorkerControlClient,
    _serve_connection,
    start_two_worker_engine,
    stop_two_worker_engine,
)
from fornax.stage_runtime import (
    StageBackendSpec,
    MaxStageBackend,
    ReferenceStageBackend,
    SimulationProfile,
    SimulatedMaxStageBackend,
    StageManifest,
    StageRequest,
    StageRuntimeError,
    Tensor,
    attest_backend_capabilities,
    create_stage_backend,
)
from fornax.phase05 import (
    phase05_manifests,
    run_loopback_engine,
    validate_phase05_engine_v0_fixture,
    validate_stage_abi_golden,
)


PLAN_ID = "11111111-1111-4111-8111-111111111111"
PLAN_HASH = "sha256:" + "1" * 64
MODEL_HASH = "sha256:" + "2" * 64
TOKENIZER_HASH = "sha256:" + "3" * 64
TEMPLATE_HASH = "sha256:" + "4" * 64


def manifest(stage_index: int, *, output_kind: str = "activation") -> StageManifest:
    return StageManifest(
        manifest_version=1,
        model_id="fornax/phase05-fixture",
        model_snapshot="fixture-snapshot-v1",
        model_config_hash=MODEL_HASH,
        tokenizer_hash=TOKENIZER_HASH,
        template_hash=TEMPLATE_HASH,
        max_build_id="simulated-max-v1",
        fornax_abi_major=1,
        fornax_abi_minor=0,
        plan_id=PLAN_ID,
        plan_hash=PLAN_HASH,
        stage_id=f"stage-{stage_index}",
        stage_index=stage_index,
        layer_start=stage_index * 2,
        layer_end=stage_index * 2 + 1,
        input_contract={
            "kind": "activation",
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": 4,
        },
        output_contract={
            "kind": output_kind,
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": 4,
        },
        kv_policy="stage_local",
        weight_artifacts=(
            {
                "path": "fixture.bin",
                "size": 16,
                "sha256": "sha256:" + "5" * 64,
                "layer_start": stage_index * 2,
                "layer_end": stage_index * 2 + 1,
            },
        ),
        device_requirement={
            "backend": "simulated-max",
            "device_identity": f"sim-device-{stage_index}",
            "minimum_memory_bytes": 1024,
            "dtypes": ["bf16"],
        },
    )


def request(
    tensor: Tensor,
    *,
    request_id: str | None = None,
    sequence_no: int = 0,
    kv_epoch: int = 0,
    phase: str = "prefill",
) -> StageRequest:
    return StageRequest(
        plan_id=PLAN_ID,
        plan_hash=PLAN_HASH,
        request_id=request_id or str(uuid.uuid4()),
        microbatch_id="microbatch-0",
        sequence_no=sequence_no,
        phase=phase,
        token_start=0,
        token_count=tensor.descriptor.shape[0],
        input_activation=tensor,
        kv_epoch=kv_epoch,
        deadline_ns=time.monotonic_ns() + 10_000_000_000,
        trace_context={"trace_id": "trace-0", "span_id": "span-0"},
    )


def frame_metadata(stage: StageManifest, *, sequence_no: int = 0) -> dict[str, object]:
    return {
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "manifest_hash": stage.manifest_hash,
        "request_id": "22222222-2222-4222-8222-222222222222",
        "microbatch_id": "microbatch-0",
        "sequence_no": sequence_no,
        "source_stage": "gateway",
        "destination_stage": stage.stage_id,
        "phase": "prefill",
        "token_start": 0,
        "token_count": 2,
        "kv_epoch": 0,
        "deadline_ns": time.monotonic_ns() + 10_000_000_000,
        "trace_id": "trace-0",
        "span_id": "span-0",
        "request_sequence_no": 0,
    }


class Phase05StageRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor = Tensor.from_values(
            [0.0, 0.25, -0.5, 0.75, 1.0, -1.0, 0.5, -0.25],
            kind="activation",
            dtype="bf16",
            shape=(2, 4),
        )

    def test_manifest_hash_is_canonical_and_validated(self) -> None:
        first = manifest(0)
        second = StageManifest.from_dict(first.to_dict())
        self.assertEqual(first.manifest_hash, second.manifest_hash)
        self.assertTrue(first.manifest_hash.startswith("sha256:"))
        bad = first.to_dict()
        bad["plan_id"] = "NOT-A-UUID"
        with self.assertRaisesRegex(ValueError, "canonical UUID"):
            StageManifest.from_dict(bad)

    def test_bf16_tensor_round_trip_is_finite_and_bounded(self) -> None:
        decoded = self.tensor.values()
        self.assertEqual(8, len(decoded))
        for actual, expected in zip(decoded, [0.0, 0.25, -0.5, 0.75, 1.0, -1.0, 0.5, -0.25]):
            self.assertLessEqual(abs(actual - expected), 0.01)

    def test_reference_and_simulated_backends_share_contract(self) -> None:
        stage = manifest(0)
        reference = ReferenceStageBackend()
        simulated = SimulatedMaxStageBackend(
            SimulationProfile(
                scenario_id="S-DESKTOP",
                stage_service_ns=500_000,
                pack_ns=10_000,
                unpack_ns=12_000,
                jitter_fraction=0.0,
                seed=7,
            )
        )
        request_id = "33333333-3333-4333-8333-333333333333"
        outputs = []
        for backend in (reference, simulated):
            handle = backend.load(stage)
            first_request = request(self.tensor, request_id=request_id)
            first = backend.execute(handle, first_request)
            self.assertEqual("ok", first.status)
            self.assertEqual((0, 1), (first.kv_epoch_before, first.kv_epoch_after))
            duplicate = backend.execute(handle, first_request)
            self.assertEqual(first, duplicate)
            assert first.output_tensor is not None
            outputs.append(first.output_tensor.values())

            decode_request = request(
                self.tensor,
                request_id=request_id,
                sequence_no=1,
                kv_epoch=1,
                phase="decode",
            )
            decode = backend.execute(handle, decode_request)
            self.assertEqual("ok", decode.status)
            self.assertEqual((1, 2), (decode.kv_epoch_before, decode.kv_epoch_after))
            self.assertTrue(backend.drain(handle, time.monotonic_ns() + 1_000_000).drained)
            self.assertEqual("DRAINING", backend.health(handle).state)
            self.assertTrue(backend.unload(handle).unloaded)
        self.assertEqual(outputs[0], outputs[1])

    def test_cancel_stale_plan_deadline_and_faults_fail_closed(self) -> None:
        stage = manifest(0)
        backend = ReferenceStageBackend()
        handle = backend.load(stage)
        cancelled_id = "44444444-4444-4444-8444-444444444444"
        cancel = backend.cancel(handle, cancelled_id, "unit cancellation")
        self.assertTrue(cancel.cancelled)
        cancelled = backend.execute(
            handle, request(self.tensor, request_id=cancelled_id)
        )
        self.assertEqual("cancelled", cancelled.status)

        stale_request = request(self.tensor)
        object.__setattr__(stale_request, "plan_hash", "sha256:" + "f" * 64)
        stale = backend.execute(handle, stale_request)
        self.assertEqual("STALE_PLAN", stale.error["code"])

        timeout_backend = SimulatedMaxStageBackend(
            SimulationProfile(scenario_id="fault-timeout", fault="timeout")
        )
        timeout_handle = timeout_backend.load(stage)
        timed_out = timeout_backend.execute(timeout_handle, request(self.tensor))
        self.assertEqual("deadline", timed_out.status)

        disconnect_backend = SimulatedMaxStageBackend(
            SimulationProfile(scenario_id="fault-disconnect", fault="disconnect")
        )
        disconnect_handle = disconnect_backend.load(stage)
        with self.assertRaisesRegex(StageRuntimeError, "disconnect"):
            disconnect_backend.execute(disconnect_handle, request(self.tensor))

    def test_physical_backend_never_silently_falls_back(self) -> None:
        backend = MaxStageBackend()
        self.assertFalse(backend.available)
        with self.assertRaisesRegex(StageRuntimeError, "unavailable"):
            backend.load(manifest(0))

        disguised_reference = MaxStageBackend(ReferenceStageBackend())
        self.assertTrue(disguised_reference.available)
        with self.assertRaisesRegex(StageRuntimeError, "backend requested"):
            attest_backend_capabilities(disguised_reference, manifest(0))


class Phase05StageAbiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = manifest(0)
        self.tensor = Tensor.from_values(
            [0.0, 0.25, -0.5, 0.75, 1.0, -1.0, 0.5, -0.25],
            kind="activation",
            dtype="bf16",
            shape=(2, 4),
        )

    def test_crc32c_known_vector(self) -> None:
        self.assertEqual(0xE3069283, crc32c(b"123456789"))

    def test_stage_abi_golden_passes(self) -> None:
        result = validate_stage_abi_golden("fornax/golden_vectors/stage_abi_v1")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(31, result["summary"]["conformance_check_count"])

    def test_phase05_validator_rejects_toy_shape_manifests(self) -> None:
        toy_manifests = [
            stage.to_dict() | {"manifest_hash": stage.manifest_hash}
            for stage in phase05_manifests(8)
        ]
        result = validate_phase05_engine_v0_fixture(
            {
                "version": 1,
                "record_kind": "phase05-engine-v0-evidence",
                "plan_version": "v4",
                "phase": "0.5",
                "evidence_class": "T1-simulation",
                "measurement_kind": "simulation",
                "assumption_ids": [f"SA-{index:03d}" for index in range(1, 11)],
                "manifests": toy_manifests,
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("mechanism target", " ".join(result["errors"]))

    def test_historical_phase05_artifact_is_not_current_contract_authority(
        self,
    ) -> None:
        fixture = (
            Path(__file__).resolve().parents[1]
            / "docs/fornax/evidence/phase05-engine-v0-2026-07-10.json"
        )
        data = json.loads(fixture.read_text(encoding="utf-8"))
        result = validate_phase05_engine_v0_fixture(data)
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(result["summary"]["current_contract_authority"])
        self.assertIn("not current-contract authority", " ".join(result["warnings"]))

    def test_activation_frame_round_trip_and_identity(self) -> None:
        frame = Frame.from_tensor(
            self.tensor,
            sequence_no=0,
            metadata=frame_metadata(self.manifest),
        )
        encoded = encode_frame(frame)
        self.assertEqual(MAGIC, encoded[:4])
        decoded = decode_frame(encoded)
        self.assertEqual(frame.kind, decoded.kind)
        self.assertEqual(self.tensor.values(), decoded.tensor().values())
        validate_frame_identity(
            decoded,
            plan_id=PLAN_ID,
            plan_hash=PLAN_HASH,
            manifest_hash=self.manifest.manifest_hash,
            destination_stage=self.manifest.stage_id,
        )
        with self.assertRaisesRegex(FrameError, "destination_stage"):
            validate_frame_identity(
                decoded,
                plan_id=PLAN_ID,
                plan_hash=PLAN_HASH,
                manifest_hash=self.manifest.manifest_hash,
                destination_stage="stage-9",
            )

    def test_missing_request_sequence_extension_remains_v1_compatible(self) -> None:
        metadata = frame_metadata(self.manifest)
        del metadata["request_sequence_no"]
        decoded = decode_frame(
            encode_frame(
                Frame.from_tensor(self.tensor, sequence_no=0, metadata=metadata)
            )
        )
        self.assertNotIn("request_sequence_no", decoded.metadata)

    def test_channel_negotiation_rejects_future_minor(self) -> None:
        backend = ReferenceStageBackend()
        handle = backend.load(self.manifest)
        server, client = socket.socketpair()

        def serve() -> None:
            with server:
                _serve_connection(
                    server,
                    backend=backend,
                    handle=handle,
                    manifest=self.manifest,
                    max_payload_bytes=256 * 1024 * 1024,
                )

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            send_frame(
                client,
                Frame(
                    MessageKind.HEARTBEAT,
                    0,
                    {
                        "sequence_no": 0,
                        "control": "negotiate",
                        "abi_major": 1,
                        "abi_minor": 1,
                        "plan_id": self.manifest.plan_id,
                        "plan_hash": self.manifest.plan_hash,
                        "manifest_hash": self.manifest.manifest_hash,
                        "destination_stage": self.manifest.stage_id,
                    },
                ),
            )
            response = read_frame(client)
            self.assertEqual(MessageKind.ERROR, response.kind)
            self.assertEqual("ABI_VERSION", response.metadata["code"])
        finally:
            client.close()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())

    def test_request_sequence_extension_requires_nonnegative_integer(self) -> None:
        for invalid in (None, True, -1, "0"):
            with self.subTest(invalid=invalid):
                metadata = frame_metadata(self.manifest)
                metadata["request_sequence_no"] = invalid
                with self.assertRaisesRegex(
                    FrameError, "request_sequence_no must be a non-negative integer"
                ):
                    encode_frame(
                        Frame.from_tensor(
                            self.tensor, sequence_no=0, metadata=metadata
                        )
                    )

    def _connected_reference_worker(
        self, backend: ReferenceStageBackend
    ) -> tuple[socket.socket, threading.Thread]:
        handle = backend.load(self.manifest)
        server, client = socket.socketpair()
        def serve() -> None:
            with server:
                _serve_connection(
                    server,
                    backend=backend,
                    handle=handle,
                    manifest=self.manifest,
                    max_payload_bytes=256 * 1024 * 1024,
                )

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        negotiate = Frame(
            kind=MessageKind.HEARTBEAT,
            sequence_no=0,
            metadata={
                "sequence_no": 0,
                "control": "negotiate",
                "plan_id": self.manifest.plan_id,
                "plan_hash": self.manifest.plan_hash,
                "manifest_hash": self.manifest.manifest_hash,
                "destination_stage": self.manifest.stage_id,
                "abi_major": 1,
                "abi_minor": 0,
            },
        )
        send_frame(client, negotiate)
        self.assertEqual("ready", read_frame(client).metadata["control"])
        self.assertEqual(MessageKind.CREDIT, read_frame(client).kind)
        return client, thread

    def _shutdown_reference_worker(
        self, client: socket.socket, thread: threading.Thread, sequence_no: int
    ) -> None:
        send_frame(
            client,
            Frame(
                kind=MessageKind.HEARTBEAT,
                sequence_no=sequence_no,
                metadata={"sequence_no": sequence_no, "control": "shutdown"},
            ),
        )
        self.assertEqual("shutdown-complete", read_frame(client).metadata["control"])
        client.close()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_duplicate_data_frame_replays_wire_response_without_execution(self) -> None:
        class CountingBackend(ReferenceStageBackend):
            def __init__(self) -> None:
                super().__init__()
                self.execute_calls = 0

            def execute(self, handle, stage_request):  # type: ignore[no-untyped-def]
                self.execute_calls += 1
                return super().execute(handle, stage_request)

        backend = CountingBackend()
        client, thread = self._connected_reference_worker(backend)
        frame = Frame.from_tensor(
            self.tensor,
            sequence_no=1,
            metadata=frame_metadata(self.manifest, sequence_no=1),
        )
        try:
            for _ in range(2):
                send_frame(client, frame)
                responses = tuple(read_frame(client) for _ in range(3))
                self.assertEqual(
                    (MessageKind.ACK, MessageKind.ACTIVATION, MessageKind.CREDIT),
                    tuple(response.kind for response in responses),
                )
            self.assertEqual(1, backend.execute_calls)
            self._shutdown_reference_worker(client, thread, 2)
        finally:
            if thread.is_alive():
                client.close()

    def test_duplicate_replay_window_evicts_old_tensor_response(self) -> None:
        class CountingBackend(ReferenceStageBackend):
            def __init__(self) -> None:
                super().__init__()
                self.execute_calls = 0

            def execute(self, handle, stage_request):  # type: ignore[no-untyped-def]
                self.execute_calls += 1
                return super().execute(handle, stage_request)

        backend = CountingBackend()
        client, thread = self._connected_reference_worker(backend)
        first = Frame.from_tensor(
            self.tensor,
            sequence_no=1,
            metadata=frame_metadata(self.manifest, sequence_no=1),
        )
        second_metadata = frame_metadata(self.manifest, sequence_no=2)
        second_metadata["request_id"] = "33333333-3333-4333-8333-333333333333"
        second = Frame.from_tensor(
            self.tensor,
            sequence_no=2,
            metadata=second_metadata,
        )
        try:
            for frame in (first, second):
                send_frame(client, frame)
                self.assertEqual(
                    (MessageKind.ACK, MessageKind.ACTIVATION, MessageKind.CREDIT),
                    tuple(read_frame(client).kind for _ in range(3)),
                )
            send_frame(client, first)
            rejected = read_frame(client)
            self.assertEqual(MessageKind.ERROR, rejected.kind)
            self.assertEqual("SEQUENCE", rejected.metadata["code"])
            self.assertEqual(2, backend.execute_calls)
            client.close()
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
        finally:
            if thread.is_alive():
                client.close()

    def test_non_finite_tensor_is_rejected_without_killing_worker(self) -> None:
        class CountingBackend(ReferenceStageBackend):
            def __init__(self) -> None:
                super().__init__()
                self.execute_calls = 0

            def execute(self, handle, stage_request):  # type: ignore[no-untyped-def]
                self.execute_calls += 1
                return super().execute(handle, stage_request)

        backend = CountingBackend()
        client, thread = self._connected_reference_worker(backend)
        metadata = frame_metadata(self.manifest, sequence_no=1)
        metadata["tensor"] = self.tensor.descriptor.to_dict()
        non_finite = Frame(
            kind=MessageKind.ACTIVATION,
            sequence_no=1,
            metadata=metadata,
            payload=b"\xc0\x7f" + self.tensor.payload[2:],
        )
        try:
            send_frame(client, non_finite)
            error = read_frame(client)
            self.assertEqual(MessageKind.ERROR, error.kind)
            self.assertEqual("TENSOR_CONTRACT", error.metadata["code"])
            self.assertEqual(MessageKind.CREDIT, read_frame(client).kind)

            valid = Frame.from_tensor(
                self.tensor,
                sequence_no=2,
                metadata=frame_metadata(self.manifest, sequence_no=2),
            )
            send_frame(client, valid)
            responses = tuple(read_frame(client) for _ in range(3))
            self.assertEqual(MessageKind.ACK, responses[0].kind)
            self.assertEqual(1, backend.execute_calls)
            self._shutdown_reference_worker(client, thread, 3)
        finally:
            if thread.is_alive():
                client.close()

    def test_frame_rejects_crc_truncation_reserved_kind_and_control_payload(self) -> None:
        frame = Frame.from_tensor(
            self.tensor,
            sequence_no=0,
            metadata=frame_metadata(self.manifest),
        )
        encoded = bytearray(encode_frame(frame))
        encoded[-1] ^= 0x01
        with self.assertRaisesRegex(FrameError, "CRC32C"):
            decode_frame(bytes(encoded))
        with self.assertRaisesRegex(FrameError, "short frame"):
            decode_frame(b"FNX1")

        reserved = Frame(
            kind=MessageKind.KV_PAGE,
            sequence_no=0,
            metadata={"sequence_no": 0},
        )
        with self.assertRaisesRegex(FrameError, "reserved"):
            encode_frame(reserved)
        credit = Frame(
            kind=MessageKind.CREDIT,
            sequence_no=0,
            metadata={"sequence_no": 0, "messages": 1, "bytes": 16},
            payload=b"x",
        )
        with self.assertRaisesRegex(FrameError, "must not contain"):
            encode_frame(credit)

    def test_duplicate_metadata_and_sequence_conflict_are_rejected(self) -> None:
        metadata = b'{"sequence_no":0,"sequence_no":0}'
        checksum = crc32c(metadata)
        encoded = PRELUDE.pack(
            MAGIC,
            ABI_MAJOR,
            ABI_MINOR,
            int(MessageKind.HEARTBEAT),
            0,
            len(metadata),
            0,
            0,
            checksum,
            0,
        ) + metadata
        with self.assertRaisesRegex(FrameError, "duplicate"):
            decode_frame(encoded)

        first = decode_frame(
            encode_frame(
                Frame.from_tensor(
                    self.tensor,
                    sequence_no=0,
                    metadata=frame_metadata(self.manifest),
                )
            )
        )
        tracker = SequenceTracker()
        self.assertEqual("accepted", tracker.accept(first))
        self.assertEqual("duplicate", tracker.accept(first))
        altered_tensor = Tensor.from_values(
            [0.0, 0.25, -0.5, 0.75, 1.0, -1.0, 0.5, 0.0],
            kind="activation",
            dtype="bf16",
            shape=(2, 4),
        )
        altered = decode_frame(
            encode_frame(
                Frame.from_tensor(
                    altered_tensor,
                    sequence_no=0,
                    metadata=frame_metadata(self.manifest),
                )
            )
        )
        with self.assertRaisesRegex(FrameError, "conflicting duplicate"):
            tracker.accept(altered)

    def test_sequence_tracker_history_is_bounded(self) -> None:
        first = decode_frame(
            encode_frame(
                Frame.from_tensor(
                    self.tensor,
                    sequence_no=0,
                    metadata=frame_metadata(self.manifest, sequence_no=0),
                )
            )
        )
        second = decode_frame(
            encode_frame(
                Frame.from_tensor(
                    self.tensor,
                    sequence_no=1,
                    metadata=frame_metadata(self.manifest, sequence_no=1),
                )
            )
        )
        tracker = SequenceTracker(max_entries=1)
        tracker.accept(first)
        tracker.accept(second)
        self.assertEqual({1}, set(tracker.acknowledged or {}))
        with self.assertRaisesRegex(FrameError, "expected sequence 2, got 0"):
            tracker.accept(first)


class Phase05EngineV0Test(unittest.TestCase):
    def setUp(self) -> None:
        self.manifests = (manifest(0), manifest(1, output_kind="logits"))
        self.profiles = (
            SimulationProfile(
                scenario_id="S-DESKTOP-stage-0",
                stage_service_ns=500_000,
                seed=1,
                device_identity="sim-device-0",
            ),
            SimulationProfile(
                scenario_id="S-DESKTOP-stage-1",
                stage_service_ns=750_000,
                seed=2,
                device_identity="sim-device-1",
            ),
        )
        self.tensor = Tensor.from_values(
            [0.0, 0.25, -0.5, 0.75, 1.0, -1.0, 0.5, -0.25],
            kind="activation",
            dtype="bf16",
            shape=(2, 4),
        )

    def test_two_independent_worker_processes_execute_prefill_and_decode(self) -> None:
        workers, channels, orchestrator = start_two_worker_engine(
            self.manifests, self.profiles
        )
        try:
            pids = [worker.pid for worker in workers]
            self.assertEqual(2, len(set(pids)))
            self.assertNotIn(os.getpid(), pids)
            for stage_manifest, worker in zip(self.manifests, workers):
                endpoint = worker.endpoint
                assert endpoint is not None
                control = WorkerControlClient(
                    stage_manifest,
                    str(endpoint["host"]),
                    int(endpoint["control_port"]),
                )
                self.assertEqual("READY", control.health()["state"])
                capabilities = control.capabilities()
                self.assertEqual("simulated-max", capabilities["backend"])
                self.assertEqual("backend", capabilities["capability_source"])
                self.assertTrue(capabilities["attestation"]["compatible"])
                self.assertTrue(
                    capabilities["attestation"]["checked_before_load"]
                )
                self.assertEqual(
                    f"sim-device-{stage_manifest.stage_index}",
                    capabilities["node_id"],
                )
                status = control.status()
                self.assertEqual(stage_manifest.manifest_hash, status["manifest_hash"])
                self.assertGreater(status["process_max_rss_bytes"], 0)
                self.assertTrue(control.install()["unchanged"])
            request_id = "55555555-5555-4555-8555-555555555555"
            prefill = orchestrator.execute(
                self.tensor,
                request_id=request_id,
                phase="prefill",
                request_sequence_no=0,
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
            self.assertEqual("ok", prefill.status)
            self.assertEqual("logits", prefill.output_kind)
            self.assertEqual((0, 1), (prefill.kv_epoch_before, prefill.kv_epoch_after))
            decode = orchestrator.execute(
                self.tensor,
                request_id=request_id,
                phase="decode",
                request_sequence_no=1,
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
            self.assertEqual("ok", decode.status)
            self.assertEqual((1, 2), (decode.kv_epoch_before, decode.kv_epoch_after))
            self.assertTrue(all(channel.message_credit == 1 for channel in channels))
            self.assertEqual(4, len(orchestrator.events))
            releases = orchestrator.release_request(request_id)
            self.assertEqual(2, len(releases))
            self.assertTrue(all(item["released"] for item in releases))
            self.assertTrue(
                all(item["idempotency_results_released"] == 2 for item in releases)
            )
            self.assertFalse(
                any(key[0] == request_id for key in orchestrator.kv_epochs)
            )
            self.assertTrue(all(channel.message_credit == 1 for channel in channels))
            self.assertEqual("request_released", orchestrator.events[-1]["kind"])
        finally:
            stop_two_worker_engine(workers, channels)

    def test_serializable_backend_factory_and_attestation_fail_closed(self) -> None:
        profile = SimulationProfile(
            scenario_id="factory-test",
            build_id="simulated-max-v1",
            device_identity="sim-device-0",
            memory_limit_bytes=4096,
        )
        spec = StageBackendSpec.simulated(profile)
        restored = StageBackendSpec.from_dict(spec.to_dict())
        backend = create_stage_backend(restored)
        attestation = attest_backend_capabilities(backend, self.manifests[0])
        self.assertTrue(attestation["compatible"])
        self.assertEqual(
            "sim-device-0", attestation["observed"]["device_identity"]
        )

        incompatible = manifest(0).to_dict()
        incompatible["max_build_id"] = "different-build"
        with self.assertRaisesRegex(StageRuntimeError, "build_id requested"):
            attest_backend_capabilities(
                backend, StageManifest.from_dict(incompatible)
            )

    def test_orchestrator_rejects_cross_plan_and_noncontiguous_stages(self) -> None:
        first, second = self.manifests
        first_channel = StageChannel(first, "127.0.0.1", 1)

        cross_plan_data = second.to_dict()
        cross_plan_data["plan_id"] = "99999999-9999-4999-8999-999999999999"
        cross_plan = StageManifest.from_dict(cross_plan_data)
        with self.assertRaisesRegex(ValueError, "plan_id"):
            EngineV0Orchestrator(
                [
                    (first, first_channel),
                    (cross_plan, StageChannel(cross_plan, "127.0.0.1", 2)),
                ]
            )

        gap_data = second.to_dict()
        gap_data["layer_start"] = 999
        gap_data["layer_end"] = 1000
        gap = StageManifest.from_dict(gap_data)
        with self.assertRaisesRegex(ValueError, "layer ranges"):
            EngineV0Orchestrator(
                [
                    (first, first_channel),
                    (gap, StageChannel(gap, "127.0.0.1", 2)),
                ]
            )

    def test_channel_credit_cancel_and_reconnect(self) -> None:
        workers, channels, _ = start_two_worker_engine(self.manifests, self.profiles)
        stage0 = channels[0]
        replacement: StageChannel | None = None
        try:
            stage0.message_credit = 0
            with self.assertRaisesRegex(EngineV0Error, "credit exhausted"):
                stage0.execute(
                    request(
                        self.tensor,
                        request_id="66666666-6666-4666-8666-666666666666",
                    )
                )
            stage0.message_credit = 1
            cancel_id = "77777777-7777-4777-8777-777777777777"
            response = stage0.cancel(cancel_id, "microbatch-0", "unit cancel")
            self.assertTrue(response["cancelled"])
            cancelled = stage0.execute(request(self.tensor, request_id=cancel_id))
            self.assertEqual("cancelled", cancelled.status)

            endpoint = workers[0].endpoint
            assert endpoint is not None
            stage0.disconnect()
            time.sleep(0.05)
            replacement = StageChannel(
                manifest=self.manifests[0],
                host=str(endpoint["host"]),
                port=int(endpoint["port"]),
            )
            replacement.connect()
            result = replacement.execute(
                request(
                    self.tensor,
                    request_id="88888888-8888-4888-8888-888888888888",
                )
            )
            self.assertEqual("ok", result.status)
        finally:
            if replacement is not None:
                channels[0] = replacement
            stop_two_worker_engine(workers, channels)

    def test_admission_scheduler_is_fifo_bounded_and_cleans_up(self) -> None:
        scheduler = AdmissionScheduler(max_inflight=2, max_queued=3, microbatch_size=2)
        deadline = time.monotonic_ns() + 1_000_000_000
        self.assertTrue(scheduler.submit("r0", 0, deadline))
        self.assertTrue(scheduler.submit("r1", 1, deadline))
        self.assertTrue(scheduler.submit("r2", 2, deadline))
        self.assertFalse(scheduler.submit("r3", 3, deadline))
        batch = scheduler.next_microbatch(time.monotonic_ns())
        self.assertEqual(("r0", "r1"), tuple(item.request_id for item in batch))
        self.assertEqual(2, scheduler.stats["inflight"])
        scheduler.complete("r0")
        scheduler.complete("r1")
        second = scheduler.next_microbatch(time.monotonic_ns())
        self.assertEqual(("r2",), tuple(item.request_id for item in second))
        self.assertTrue(scheduler.cancel("r2"))
        self.assertEqual({"queued": 0, "inflight": 0, "max_queued": 3, "max_inflight": 2}, scheduler.stats)

    def test_sustained_runner_uses_wall_clock_and_real_concurrency(self) -> None:
        result = run_loopback_engine(
            sustained_wall_seconds=1,
            sustained_min_iterations=8,
        )
        sustained = result["sustained"]
        self.assertEqual(
            "wall-clock-with-real-concurrent-loopback",
            sustained["duration_kind"],
        )
        self.assertGreaterEqual(sustained["wall_elapsed_ns"], 1_000_000_000)
        self.assertGreaterEqual(sustained["actual_loopback_iterations"], 8)
        self.assertEqual(8, sustained["max_observed_concurrency"])
        self.assertFalse(sustained["queue_bound_exceeded"])
        self.assertFalse(sustained["memory_bound_exceeded"])
        self.assertTrue(result["cleanup_completed"])


if __name__ == "__main__":
    unittest.main()
