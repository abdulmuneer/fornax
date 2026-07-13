from __future__ import annotations

import json
import os
import time
import unittest
import uuid

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
    validate_frame_identity,
)
from fornax.engine_v0 import (
    AdmissionScheduler,
    EngineV0Error,
    StageChannel,
    WorkerControlClient,
    start_two_worker_engine,
    stop_two_worker_engine,
)
from fornax.stage_runtime import (
    MaxStageBackend,
    ReferenceStageBackend,
    SimulationProfile,
    SimulatedMaxStageBackend,
    StageManifest,
    StageRequest,
    StageRuntimeError,
    Tensor,
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
        self.assertEqual(24, result["summary"]["conformance_check_count"])

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


class Phase05EngineV0Test(unittest.TestCase):
    def setUp(self) -> None:
        self.manifests = (manifest(0), manifest(1, output_kind="logits"))
        self.profiles = (
            SimulationProfile(
                scenario_id="S-DESKTOP-stage-0",
                stage_service_ns=500_000,
                seed=1,
            ),
            SimulationProfile(
                scenario_id="S-DESKTOP-stage-1",
                stage_service_ns=750_000,
                seed=2,
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
                self.assertEqual("simulated-max", control.capabilities()["backend"])
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
        finally:
            stop_two_worker_engine(workers, channels)

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
