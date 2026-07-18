from __future__ import annotations

import socket
import threading
import time
import unittest
from unittest.mock import patch

from fornax.engine_v0 import (
    AdmissionScheduler,
    EngineV0Error,
    EngineV0Orchestrator,
    StageChannel,
    _serve_connection,
)
from fornax.phase05 import phase05_manifests, phase05_tensor
from fornax.stage_abi import Frame, MessageKind, read_frame, send_frame
from fornax.stage_runtime import (
    PythonTensorBufferAdapter,
    ReferenceStageBackend,
    SimulatedMaxStageBackend,
    SimulationProfile,
    StageRequest,
    StageResult,
    StageRuntimeError,
    Tensor,
    TensorDescriptor,
)


REQUEST_A = "10101010-1010-4010-8010-101010101010"
REQUEST_B = "20202020-2020-4020-8020-202020202020"
REQUEST_C = "30303030-3030-4030-8030-303030303030"


def stage_request(
    tensor: Tensor,
    *,
    request_id: str,
    sequence_no: int = 0,
    kv_epoch: int = 0,
    phase: str = "prefill",
) -> StageRequest:
    stage = phase05_manifests(4)[0]
    return StageRequest(
        plan_id=stage.plan_id,
        plan_hash=stage.plan_hash,
        request_id=request_id,
        microbatch_id="microbatch-0",
        sequence_no=sequence_no,
        phase=phase,
        token_start=0,
        token_count=tensor.descriptor.shape[0],
        input_activation=tensor,
        kv_epoch=kv_epoch,
        deadline_ns=time.monotonic_ns() + 10_000_000_000,
        trace_context={"trace_id": f"trace-{request_id}", "span_id": "span-0"},
    )


class RuntimeLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = phase05_manifests(4)[0]
        self.tensor = phase05_tensor(2, 4)

    def test_release_removes_request_state_and_reopens_admission(self) -> None:
        backend = ReferenceStageBackend(max_live_requests=1)
        handle = backend.load(self.manifest)
        first = backend.execute(
            handle, stage_request(self.tensor, request_id=REQUEST_A)
        )
        self.assertEqual("ok", first.status)
        rejected = backend.execute(
            handle, stage_request(self.tensor, request_id=REQUEST_B)
        )
        self.assertEqual("ADMISSION", rejected.error["code"])
        self.assertEqual(1, backend.health(handle).live_requests)

        released = backend.release(handle, REQUEST_A)
        self.assertTrue(released.released)
        self.assertEqual(1, released.kv_state_released)
        self.assertEqual(1, released.execution_state_released)
        self.assertEqual(1, released.idempotency_results_released)
        health = backend.health(handle)
        self.assertEqual(0, health.live_requests)
        self.assertEqual(0, health.completed_results)
        self.assertFalse(backend.release(handle, REQUEST_A).released)

        admitted = backend.execute(
            handle, stage_request(self.tensor, request_id=REQUEST_B)
        )
        self.assertEqual("ok", admitted.status)

    def test_release_fails_while_request_is_inflight(self) -> None:
        entered = threading.Event()
        finish = threading.Event()

        class BlockingBackend(ReferenceStageBackend):
            def _transform_tensor(self, manifest, request):  # type: ignore[no-untyped-def]
                entered.set()
                if not finish.wait(2.0):
                    raise RuntimeError("test transform timed out")
                return super()._transform_tensor(manifest, request)

        backend = BlockingBackend()
        handle = backend.load(self.manifest)
        results = []
        worker = threading.Thread(
            target=lambda: results.append(
                backend.execute(
                    handle, stage_request(self.tensor, request_id=REQUEST_A)
                )
            )
        )
        worker.start()
        self.assertTrue(entered.wait(1.0))
        with self.assertRaisesRegex(StageRuntimeError, "inflight") as caught:
            backend.release(handle, REQUEST_A)
        self.assertEqual("REQUEST_INFLIGHT", caught.exception.code)
        finish.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual("ok", results[0].status)
        self.assertTrue(backend.release(handle, REQUEST_A).released)

    def test_idle_expiry_reclaims_state_and_fences_request_identity(self) -> None:
        now_ns = [100]
        backend = ReferenceStageBackend(
            clock_ns=lambda: now_ns[0],
            request_idle_timeout_ns=10,
            execution_lease_timeout_ns=10,
            release_tombstone_ttl_ns=20,
            max_release_tombstones=2,
        )
        handle = backend.load(self.manifest)
        self.assertEqual(
            "ok",
            backend.execute(
                handle, stage_request(self.tensor, request_id=REQUEST_A)
            ).status,
        )
        self.assertEqual(1, backend.health(handle).live_requests)

        now_ns[0] = 111
        swept = backend.sweep_expired(handle)
        self.assertEqual(1, swept.expired_requests)
        health = backend.health(handle)
        self.assertEqual(0, health.live_requests)
        self.assertEqual(1, health.release_tombstones)
        fenced = backend.execute(
            handle, stage_request(self.tensor, request_id=REQUEST_A)
        )
        self.assertEqual("rejected", fenced.status)
        self.assertEqual("REQUEST_TOMBSTONED", fenced.error["code"])
        repeated_release = backend.release(handle, REQUEST_A)
        self.assertFalse(repeated_release.released)
        self.assertTrue(repeated_release.tombstoned)
        self.assertEqual("request_idle_expired", repeated_release.reason)

        now_ns[0] = 132
        self.assertEqual(0, backend.health(handle).release_tombstones)
        self.assertEqual(
            "ok",
            backend.execute(
                handle, stage_request(self.tensor, request_id=REQUEST_A)
            ).status,
        )

    def test_release_tombstones_are_count_and_time_bounded(self) -> None:
        now_ns = [100]
        backend = ReferenceStageBackend(
            clock_ns=lambda: now_ns[0],
            max_release_tombstones=2,
            release_tombstone_ttl_ns=10,
        )
        handle = backend.load(self.manifest)
        for request_id in (REQUEST_A, REQUEST_B):
            released = backend.release(handle, request_id)
            self.assertFalse(released.released)
            self.assertTrue(released.tombstoned)
        with self.assertRaises(StageRuntimeError) as caught:
            backend.release(handle, REQUEST_C)
        self.assertEqual("TOMBSTONE_CAPACITY", caught.exception.code)
        health = backend.health(handle)
        self.assertEqual(2, health.release_tombstones)
        self.assertEqual(2, health.max_release_tombstones)

        for request_id in (REQUEST_A, REQUEST_B):
            fenced = backend.execute(
                handle, stage_request(self.tensor, request_id=request_id)
            )
            self.assertEqual("rejected", fenced.status)
            self.assertEqual("REQUEST_TOMBSTONED", fenced.error["code"])

        now_ns[0] = 111
        swept = backend.sweep_expired(handle)
        self.assertEqual(2, swept.pruned_tombstones)
        self.assertEqual(0, backend.health(handle).release_tombstones)
        self.assertTrue(backend.release(handle, REQUEST_C).tombstoned)

    def test_tombstone_capacity_never_discards_live_state_or_replay_fence(self) -> None:
        now_ns = [100]
        backend = ReferenceStageBackend(
            clock_ns=lambda: now_ns[0],
            max_release_tombstones=1,
            release_tombstone_ttl_ns=100,
            request_idle_timeout_ns=10,
        )
        handle = backend.load(self.manifest)
        first = backend.execute(
            handle, stage_request(self.tensor, request_id=REQUEST_A)
        )
        self.assertEqual("ok", first.status)
        self.assertTrue(backend.release(handle, REQUEST_A).released)

        second = backend.execute(
            handle, stage_request(self.tensor, request_id=REQUEST_B)
        )
        self.assertEqual("ok", second.status)
        now_ns[0] = 111
        with self.assertRaises(StageRuntimeError) as caught:
            backend.sweep_expired(handle)
        self.assertEqual("TOMBSTONE_CAPACITY", caught.exception.code)

        stage = backend._stage(handle)
        self.assertIn(REQUEST_A, stage.release_tombstones)
        self.assertIn(REQUEST_B, stage.requests)

    def test_expired_execution_lease_discards_late_result(self) -> None:
        now_ns = [100]
        entered = threading.Event()
        finish = threading.Event()

        class BlockingBackend(ReferenceStageBackend):
            def _transform_tensor(self, manifest, request):  # type: ignore[no-untyped-def]
                entered.set()
                if not finish.wait(2.0):
                    raise RuntimeError("test transform timed out")
                return super()._transform_tensor(manifest, request)

        backend = BlockingBackend(
            clock_ns=lambda: now_ns[0],
            execution_lease_timeout_ns=10,
            request_idle_timeout_ns=100,
            release_tombstone_ttl_ns=100,
        )
        handle = backend.load(self.manifest)
        results: list[StageResult] = []
        worker = threading.Thread(
            target=lambda: results.append(
                backend.execute(
                    handle, stage_request(self.tensor, request_id=REQUEST_A)
                )
            )
        )
        worker.start()
        self.assertTrue(entered.wait(1.0))
        now_ns[0] = 111
        health = backend.health(handle)
        self.assertEqual(0, health.inflight)
        self.assertEqual(0, health.live_requests)
        self.assertEqual(1, health.expired_execution_leases)
        finish.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual("failed", results[0].status)
        self.assertEqual("LEASE_EXPIRED", results[0].error["code"])
        self.assertEqual(0, backend.health(handle).completed_results)

    def test_python_native_buffer_seam_is_copy_explicit_and_bounded(self) -> None:
        adapter = PythonTensorBufferAdapter(max_imported_bytes=64)
        imported = adapter.import_tensor(
            self.tensor,
            expected=self.tensor.descriptor,
            purpose="unit-input",
        )
        self.assertTrue(imported.copy_performed)
        self.assertTrue(imported.descriptor_validated)
        self.assertTrue(imported.payload_validated)
        self.assertEqual(len(self.tensor.payload), adapter.health().inflight_bytes)
        exported = adapter.export_tensor(
            imported, expected=self.tensor.descriptor
        )
        self.assertEqual(self.tensor, exported)
        adapter.release(imported)
        adapter.release(imported)
        self.assertEqual(0, adapter.health().inflight_bytes)

        wrong = TensorDescriptor(
            kind="logits",
            dtype=self.tensor.descriptor.dtype,
            shape=self.tensor.descriptor.shape,
        )
        with self.assertRaisesRegex(StageRuntimeError, "descriptor") as caught:
            adapter.import_tensor(
                self.tensor,
                expected=wrong,
                purpose="wrong-contract",
            )
        self.assertEqual("TENSOR_CONTRACT", caught.exception.code)

        backend = ReferenceStageBackend(
            buffer_adapter=adapter,
            max_native_buffer_bytes=64,
        )
        handle = backend.load(self.manifest)
        result = backend.execute(
            handle, stage_request(self.tensor, request_id=REQUEST_A)
        )
        self.assertEqual("ok", result.status)
        health = backend.health(handle)
        self.assertEqual(0, health.native_buffer_imports)
        self.assertEqual(0, health.native_buffer_bytes)
        self.assertLessEqual(
            health.native_buffer_high_water_bytes,
            health.max_native_buffer_bytes,
        )
        self.assertGreaterEqual(health.native_buffer_copy_operations, 6)

    def test_evicted_and_conflicting_duplicates_fail_without_kv_mutation(self) -> None:
        backend = ReferenceStageBackend(max_completed_results_per_request=1)
        handle = backend.load(self.manifest)
        first_request = stage_request(self.tensor, request_id=REQUEST_A)
        first = backend.execute(handle, first_request)
        self.assertEqual("ok", first.status)
        second = backend.execute(
            handle,
            stage_request(
                self.tensor,
                request_id=REQUEST_A,
                sequence_no=1,
                kv_epoch=1,
                phase="decode",
            ),
        )
        self.assertEqual((1, 2), (second.kv_epoch_before, second.kv_epoch_after))

        token_conflict = stage_request(
            self.tensor,
            request_id=REQUEST_A,
            sequence_no=1,
            kv_epoch=1,
            phase="decode",
        )
        object.__setattr__(token_conflict, "token_start", 1)
        with self.assertRaisesRegex(StageRuntimeError, "replay window") as conflict:
            backend.execute(handle, token_conflict)
        self.assertEqual("SEQUENCE", conflict.exception.code)

        with self.assertRaisesRegex(StageRuntimeError, "replay window") as evicted:
            backend.execute(handle, first_request)
        self.assertEqual("SEQUENCE", evicted.exception.code)

        altered = Tensor.from_values(
            [value + 0.125 for value in self.tensor.values()],
            kind="activation",
            dtype="bf16",
            shape=self.tensor.descriptor.shape,
        )
        with self.assertRaisesRegex(StageRuntimeError, "replay window") as conflict:
            backend.execute(
                handle,
                stage_request(
                    altered,
                    request_id=REQUEST_A,
                    sequence_no=1,
                    kv_epoch=1,
                    phase="decode",
                ),
            )
        self.assertEqual("SEQUENCE", conflict.exception.code)

        third = backend.execute(
            handle,
            stage_request(
                self.tensor,
                request_id=REQUEST_A,
                sequence_no=2,
                kv_epoch=2,
                phase="decode",
            ),
        )
        self.assertEqual((2, 3), (third.kv_epoch_before, third.kv_epoch_after))
        self.assertEqual(1, backend.health(handle).completed_results)

    def test_simulated_duplicate_replays_exact_final_result_with_jitter(self) -> None:
        backend = SimulatedMaxStageBackend(
            SimulationProfile(
                scenario_id="jitter-idempotency",
                stage_service_ns=1_000_000,
                jitter_fraction=0.5,
                seed=7,
            )
        )
        handle = backend.load(self.manifest)
        request = stage_request(self.tensor, request_id=REQUEST_A)
        first = backend.execute(handle, request)
        duplicate = backend.execute(handle, request)
        self.assertEqual(first, duplicate)

    def test_simulated_result_is_decorated_before_becoming_replay_visible(self) -> None:
        entered = threading.Event()
        finish = threading.Event()

        class BlockingRandom:
            def uniform(self, lower, upper):  # type: ignore[no-untyped-def]
                entered.set()
                if not finish.wait(2.0):
                    raise RuntimeError("test jitter timed out")
                return 0.25

        backend = SimulatedMaxStageBackend(
            SimulationProfile(
                scenario_id="atomic-final-result",
                stage_service_ns=1_000_000,
                jitter_fraction=0.5,
                fault="corruption",
            )
        )
        backend._random = BlockingRandom()  # type: ignore[assignment]
        handle = backend.load(self.manifest)
        request = stage_request(self.tensor, request_id=REQUEST_A)
        completed: list[StageResult] = []
        worker = threading.Thread(
            target=lambda: completed.append(backend.execute(handle, request))
        )
        worker.start()
        self.assertTrue(entered.wait(1.0))

        concurrent = backend.execute(handle, request)
        self.assertEqual("rejected", concurrent.status)
        self.assertEqual("ADMISSION", concurrent.error["code"])
        finish.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual("failed", completed[0].status)
        self.assertEqual("CHECKSUM", completed[0].error["code"])
        self.assertEqual(completed[0], backend.execute(handle, request))

    def test_pipeline_retry_replays_upstream_after_downstream_failure(self) -> None:
        manifests = phase05_manifests(4)

        class CountingBackend(ReferenceStageBackend):
            def __init__(self) -> None:
                super().__init__()
                self.transform_calls = 0

            def _transform_tensor(self, manifest, request):  # type: ignore[no-untyped-def]
                self.transform_calls += 1
                return super()._transform_tensor(manifest, request)

        class LocalChannel:
            def __init__(self, stage_manifest, *, fail_once=False):  # type: ignore[no-untyped-def]
                self.manifest = stage_manifest
                self.backend = CountingBackend()
                self.handle = self.backend.load(stage_manifest)
                self.fail_once = fail_once

            def execute(self, request, *, source_stage):  # type: ignore[no-untyped-def]
                if self.fail_once:
                    self.fail_once = False
                    return StageResult(
                        plan_id=request.plan_id,
                        plan_hash=request.plan_hash,
                        request_id=request.request_id,
                        microbatch_id=request.microbatch_id,
                        sequence_no=request.sequence_no,
                        status="failed",
                        output_kind=None,
                        output_tensor=None,
                        kv_epoch_before=request.kv_epoch,
                        kv_epoch_after=request.kv_epoch,
                        timings_ns={
                            "queue": 0,
                            "execute": 0,
                            "pack": 0,
                            "unpack": 0,
                        },
                        error={"code": "EXECUTION", "message": "fail once"},
                    )
                return self.backend.execute(self.handle, request)

            def release_request(self, request_id):  # type: ignore[no-untyped-def]
                released = self.backend.release(self.handle, request_id)
                return {"request_id": request_id, "released": released.released}

        channels = [LocalChannel(manifests[0]), LocalChannel(manifests[1], fail_once=True)]
        orchestrator = EngineV0Orchestrator(
            list(zip(manifests, channels))  # type: ignore[arg-type]
        )
        kwargs = {
            "request_id": REQUEST_A,
            "phase": "prefill",
            "request_sequence_no": 0,
            "deadline_ns": time.monotonic_ns() + 10_000_000_000,
        }
        first = orchestrator.execute(self.tensor, **kwargs)
        self.assertEqual("failed", first.status)
        self.assertEqual({}, orchestrator.kv_epochs)

        retried = orchestrator.execute(self.tensor, **kwargs)
        self.assertEqual("ok", retried.status)
        self.assertEqual(1, channels[0].backend.transform_calls)
        self.assertEqual(1, channels[1].backend.transform_calls)
        self.assertEqual(2, len(orchestrator.kv_epochs))
        orchestrator.release_request(REQUEST_A)

    def test_transform_cache_and_event_history_are_bounded(self) -> None:
        backend = ReferenceStageBackend(max_transform_cache_entries=1)
        handle = backend.load(self.manifest)
        self.assertEqual(
            "ok",
            backend.execute(
                handle, stage_request(self.tensor, request_id=REQUEST_A)
            ).status,
        )
        backend.release(handle, REQUEST_A)
        altered = Tensor.from_values(
            [value + 0.25 for value in self.tensor.values()],
            kind="activation",
            dtype="bf16",
            shape=self.tensor.descriptor.shape,
        )
        self.assertEqual(
            "ok",
            backend.execute(
                handle, stage_request(altered, request_id=REQUEST_B)
            ).status,
        )
        self.assertEqual(1, backend.health(handle).transform_cache_entries)

        scheduler = AdmissionScheduler(
            max_inflight=1,
            max_queued=1,
            microbatch_size=1,
            max_event_entries=2,
        )
        deadline = time.monotonic_ns() + 1_000_000_000
        scheduler.submit("r0", None, deadline)
        scheduler.submit("r1", None, deadline)
        scheduler.next_microbatch(time.monotonic_ns())
        self.assertEqual(2, len(scheduler.events))
        self.assertIsInstance(scheduler.events[:], list)

    def test_result_and_transform_byte_caps_evict_only_replayable_state(self) -> None:
        backend = ReferenceStageBackend(
            max_completed_result_bytes=512,
            max_transform_cache_bytes=128,
        )
        handle = backend.load(self.manifest)
        first_request = stage_request(self.tensor, request_id=REQUEST_A)
        first = backend.execute(handle, first_request)
        self.assertEqual("ok", first.status)
        health = backend.health(handle)
        self.assertEqual(0, health.completed_results)
        self.assertEqual(0, health.completed_result_bytes)
        self.assertEqual(0, health.transform_cache_entries)
        self.assertEqual(0, health.transform_cache_bytes)
        self.assertLessEqual(
            health.completed_result_high_water_bytes,
            health.max_completed_result_bytes,
        )
        self.assertLessEqual(
            health.transform_cache_high_water_bytes,
            health.max_transform_cache_bytes,
        )

        with self.assertRaisesRegex(StageRuntimeError, "replay window"):
            backend.execute(handle, first_request)
        decode = backend.execute(
            handle,
            stage_request(
                self.tensor,
                request_id=REQUEST_A,
                sequence_no=1,
                kv_epoch=1,
                phase="decode",
            ),
        )
        self.assertEqual((1, 2), (decode.kv_epoch_before, decode.kv_epoch_after))

    def test_orchestrator_admission_is_bounded_until_explicit_release(self) -> None:
        manifests = phase05_manifests(4)

        class FakeChannel:
            def __init__(self, manifest):  # type: ignore[no-untyped-def]
                self.manifest = manifest

            def execute(self, request, *, source_stage):  # type: ignore[no-untyped-def]
                output_kind = str(
                    self.manifest.output_contract.get("kind", "activation")
                )
                output = Tensor.from_values(
                    request.input_activation.values(),
                    kind=output_kind,
                    dtype=str(self.manifest.output_contract["dtype"]),
                    shape=request.input_activation.descriptor.shape,
                )
                return StageResult(
                    plan_id=request.plan_id,
                    plan_hash=request.plan_hash,
                    request_id=request.request_id,
                    microbatch_id=request.microbatch_id,
                    sequence_no=request.sequence_no,
                    status="ok",
                    output_kind=output_kind,
                    output_tensor=output,
                    kv_epoch_before=request.kv_epoch,
                    kv_epoch_after=request.kv_epoch + 1,
                    timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                )

            def release_request(self, request_id):  # type: ignore[no-untyped-def]
                return {"request_id": request_id, "released": True}

        orchestrator = EngineV0Orchestrator(
            [(item, FakeChannel(item)) for item in manifests],  # type: ignore[list-item]
            max_live_requests=1,
        )
        first = orchestrator.execute(
            self.tensor,
            request_id=REQUEST_A,
            phase="prefill",
            request_sequence_no=0,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
        )
        self.assertEqual("ok", first.status)
        self.assertEqual(1, orchestrator.active_request_count)
        with self.assertRaisesRegex(EngineV0Error, "capacity") as rejected:
            orchestrator.execute(
                self.tensor,
                request_id=REQUEST_B,
                phase="prefill",
                request_sequence_no=0,
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
        self.assertEqual("ADMISSION", rejected.exception.code)
        orchestrator.release_request(REQUEST_A)
        self.assertEqual(0, orchestrator.active_request_count)
        second = orchestrator.execute(
            self.tensor,
            request_id=REQUEST_B,
            phase="prefill",
            request_sequence_no=0,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
        )
        self.assertEqual("ok", second.status)

    def test_invalid_pipeline_request_does_not_consume_admission(self) -> None:
        manifests = phase05_manifests(4)

        class NeverCalledChannel:
            def __init__(self, stage_manifest):  # type: ignore[no-untyped-def]
                self.manifest = stage_manifest
                self.execute_calls = 0

            def execute(self, request, *, source_stage):  # type: ignore[no-untyped-def]
                self.execute_calls += 1
                raise AssertionError("invalid request reached a stage")

            def release_request(self, request_id):  # type: ignore[no-untyped-def]
                return {"request_id": request_id, "released": False}

        channels = [NeverCalledChannel(item) for item in manifests]
        orchestrator = EngineV0Orchestrator(
            list(zip(manifests, channels)),  # type: ignore[arg-type]
            max_live_requests=1,
        )
        deadline = time.monotonic_ns() + 10_000_000_000
        with self.assertRaisesRegex(ValueError, "canonical UUID"):
            orchestrator.execute(
                self.tensor,
                request_id="not-a-request-id",
                phase="prefill",
                request_sequence_no=0,
                deadline_ns=deadline,
            )
        with self.assertRaisesRegex(ValueError, "phase"):
            orchestrator.execute(
                self.tensor,
                request_id=REQUEST_A,
                phase="invalid",
                request_sequence_no=0,
                deadline_ns=deadline,
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            orchestrator.execute(
                self.tensor,
                request_id=REQUEST_A,
                phase="prefill",
                request_sequence_no=True,
                deadline_ns=deadline,
            )
        self.assertEqual(0, orchestrator.active_request_count)
        self.assertTrue(all(channel.execute_calls == 0 for channel in channels))

    def test_partial_release_blocks_execute_until_release_retry_completes(self) -> None:
        manifests = phase05_manifests(4)

        class ReleaseChannel:
            def __init__(self, stage_manifest, *, fail_release_once=False):  # type: ignore[no-untyped-def]
                self.manifest = stage_manifest
                self.fail_release_once = fail_release_once
                self.release_calls = 0

            def execute(self, request, *, source_stage):  # type: ignore[no-untyped-def]
                output_kind = str(
                    self.manifest.output_contract.get("kind", "activation")
                )
                output = Tensor.from_values(
                    request.input_activation.values(),
                    kind=output_kind,
                    dtype=str(self.manifest.output_contract["dtype"]),
                    shape=request.input_activation.descriptor.shape,
                )
                return StageResult(
                    plan_id=request.plan_id,
                    plan_hash=request.plan_hash,
                    request_id=request.request_id,
                    microbatch_id=request.microbatch_id,
                    sequence_no=request.sequence_no,
                    status="ok",
                    output_kind=output_kind,
                    output_tensor=output,
                    kv_epoch_before=request.kv_epoch,
                    kv_epoch_after=request.kv_epoch + 1,
                    timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                )

            def release_request(self, request_id):  # type: ignore[no-untyped-def]
                self.release_calls += 1
                if self.fail_release_once:
                    self.fail_release_once = False
                    raise EngineV0Error("TRANSPORT", "release transport failed")
                return {
                    "request_id": request_id,
                    "released": self.release_calls == 1,
                }

        channels = [
            ReleaseChannel(manifests[0]),
            ReleaseChannel(manifests[1], fail_release_once=True),
        ]
        orchestrator = EngineV0Orchestrator(
            list(zip(manifests, channels))  # type: ignore[arg-type]
        )
        deadline = time.monotonic_ns() + 10_000_000_000
        result = orchestrator.execute(
            self.tensor,
            request_id=REQUEST_A,
            phase="prefill",
            request_sequence_no=0,
            deadline_ns=deadline,
        )
        self.assertEqual("ok", result.status)
        with self.assertRaisesRegex(EngineV0Error, "release transport failed"):
            orchestrator.release_request(REQUEST_A)
        with self.assertRaisesRegex(EngineV0Error, "release is in progress") as caught:
            orchestrator.execute(
                self.tensor,
                request_id=REQUEST_A,
                phase="decode",
                request_sequence_no=1,
                deadline_ns=deadline,
            )
        self.assertEqual("REQUEST_INFLIGHT", caught.exception.code)
        orchestrator.release_request(REQUEST_A)
        self.assertEqual(0, orchestrator.active_request_count)

    def test_sequence_error_preserves_channel_for_final_release(self) -> None:
        backend = ReferenceStageBackend(max_completed_results_per_request=1)
        handle = backend.load(self.manifest)
        server, client = socket.socketpair()
        worker = threading.Thread(
            target=lambda: _serve_connection(
                server,
                backend=backend,
                handle=handle,
                manifest=self.manifest,
                max_payload_bytes=256 * 1024 * 1024,
            ),
            daemon=True,
        )
        worker.start()
        channel = StageChannel(
            self.manifest,
            "127.0.0.1",
            1,
        )
        try:
            with patch(
                "fornax.engine_v0.socket.create_connection",
                return_value=client,
            ) as create_connection:
                channel.connect()
                first_request = stage_request(self.tensor, request_id=REQUEST_A)
                self.assertEqual("ok", channel.execute(first_request).status)
                second_request = stage_request(
                    self.tensor,
                    request_id=REQUEST_A,
                    sequence_no=1,
                    kv_epoch=1,
                    phase="decode",
                )
                self.assertEqual("ok", channel.execute(second_request).status)

                rejected = channel.execute(first_request)
                self.assertEqual("SEQUENCE", rejected.error["code"])
                self.assertIsNotNone(channel.channel)
                self.assertEqual(1, channel.message_credit)
                released = channel.release_request(REQUEST_A)
                self.assertTrue(released["released"])
                self.assertIsNotNone(channel.channel)
                self.assertEqual(1, channel.message_credit)
                self.assertEqual(1, create_connection.call_count)
                channel.shutdown()
        finally:
            channel.disconnect()
            server.close()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())

    def test_orchestrator_rejects_execution_while_release_is_in_progress(self) -> None:
        manifests = phase05_manifests(4)
        release_entered = threading.Event()
        allow_release = threading.Event()

        class BlockingReleaseChannel:
            def __init__(self, manifest):  # type: ignore[no-untyped-def]
                self.manifest = manifest

            def execute(self, request, *, source_stage):  # type: ignore[no-untyped-def]
                raise AssertionError("execute must be fenced during release")

            def release_request(self, request_id):  # type: ignore[no-untyped-def]
                if self.manifest.stage_index == 0:
                    release_entered.set()
                    if not allow_release.wait(2.0):
                        raise RuntimeError("test release timed out")
                return {"request_id": request_id, "released": False}

        orchestrator = EngineV0Orchestrator(
            [
                (manifest, BlockingReleaseChannel(manifest))
                for manifest in manifests
            ],  # type: ignore[list-item]
        )
        errors: list[Exception] = []
        release_thread = threading.Thread(
            target=lambda: self._capture_release_error(
                orchestrator, REQUEST_A, errors
            )
        )
        release_thread.start()
        self.assertTrue(release_entered.wait(1.0))
        try:
            with self.assertRaisesRegex(
                EngineV0Error, "release is in progress"
            ) as rejected:
                orchestrator.execute(
                    self.tensor,
                    request_id=REQUEST_A,
                    phase="prefill",
                    request_sequence_no=0,
                    deadline_ns=time.monotonic_ns() + 10_000_000_000,
                )
            self.assertEqual("REQUEST_INFLIGHT", rejected.exception.code)
        finally:
            allow_release.set()
            release_thread.join(timeout=2.0)
        self.assertFalse(release_thread.is_alive())
        self.assertEqual([], errors)

    @staticmethod
    def _capture_release_error(
        orchestrator: EngineV0Orchestrator,
        request_id: str,
        errors: list[Exception],
    ) -> None:
        try:
            orchestrator.release_request(request_id)
        except Exception as exc:  # noqa: BLE001 - surfaced in the calling test.
            errors.append(exc)

    def test_wire_release_drops_cached_tensor_replay(self) -> None:
        class CountingBackend(ReferenceStageBackend):
            def __init__(self) -> None:
                super().__init__()
                self.execute_calls = 0

            def execute(self, handle, request):  # type: ignore[no-untyped-def]
                self.execute_calls += 1
                return super().execute(handle, request)

        backend = CountingBackend()
        handle = backend.load(self.manifest)
        server, client = socket.socketpair()
        worker = threading.Thread(
            target=lambda: _serve_connection(
                server,
                backend=backend,
                handle=handle,
                manifest=self.manifest,
                max_payload_bytes=256 * 1024 * 1024,
            ),
            daemon=True,
        )
        worker.start()
        try:
            negotiate = Frame(
                MessageKind.HEARTBEAT,
                0,
                {
                    "sequence_no": 0,
                    "control": "negotiate",
                    "abi_major": 1,
                    "abi_minor": 0,
                    "plan_id": self.manifest.plan_id,
                    "plan_hash": self.manifest.plan_hash,
                    "manifest_hash": self.manifest.manifest_hash,
                    "destination_stage": self.manifest.stage_id,
                },
            )
            send_frame(client, negotiate)
            self.assertEqual("ready", read_frame(client).metadata["control"])
            self.assertEqual(MessageKind.CREDIT, read_frame(client).kind)

            metadata = {
                "plan_id": self.manifest.plan_id,
                "plan_hash": self.manifest.plan_hash,
                "manifest_hash": self.manifest.manifest_hash,
                "request_id": REQUEST_A,
                "microbatch_id": "microbatch-0",
                "source_stage": "orchestrator",
                "destination_stage": self.manifest.stage_id,
                "phase": "prefill",
                "token_start": 0,
                "token_count": 2,
                "kv_epoch": 0,
                "deadline_ns": time.monotonic_ns() + 10_000_000_000,
                "trace_id": "trace-release",
                "span_id": "span-release",
                "request_sequence_no": 0,
            }
            data = Frame.from_tensor(self.tensor, sequence_no=1, metadata=metadata)
            send_frame(client, data)
            self.assertEqual(
                (MessageKind.ACK, MessageKind.ACTIVATION, MessageKind.CREDIT),
                tuple(read_frame(client).kind for _ in range(3)),
            )

            release = Frame(
                MessageKind.HEARTBEAT,
                2,
                {
                    "sequence_no": 2,
                    "control": "release-request",
                    "request_id": REQUEST_A,
                },
            )
            send_frame(client, release)
            release_ack = read_frame(client)
            self.assertEqual(MessageKind.ACK, release_ack.kind)
            self.assertTrue(release_ack.metadata["released"])
            self.assertEqual(0, backend.health(handle).live_requests)

            send_frame(client, data)
            replay = read_frame(client)
            self.assertEqual(MessageKind.ERROR, replay.kind)
            self.assertEqual("SEQUENCE", replay.metadata["code"])
            self.assertEqual(1, backend.execute_calls)
        finally:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            worker.join(timeout=2.0)
            server.close()

    def test_release_tombstone_fences_request_after_wire_reconnect(self) -> None:
        backend = ReferenceStageBackend()
        handle = backend.load(self.manifest)

        def start_connection():  # type: ignore[no-untyped-def]
            server, client = socket.socketpair()
            worker = threading.Thread(
                target=lambda: _serve_connection(
                    server,
                    backend=backend,
                    handle=handle,
                    manifest=self.manifest,
                    max_payload_bytes=256 * 1024 * 1024,
                ),
                daemon=True,
            )
            worker.start()
            negotiate = Frame(
                MessageKind.HEARTBEAT,
                0,
                {
                    "sequence_no": 0,
                    "control": "negotiate",
                    "abi_major": 1,
                    "abi_minor": 0,
                    "plan_id": self.manifest.plan_id,
                    "plan_hash": self.manifest.plan_hash,
                    "manifest_hash": self.manifest.manifest_hash,
                    "destination_stage": self.manifest.stage_id,
                },
            )
            send_frame(client, negotiate)
            self.assertEqual("ready", read_frame(client).metadata["control"])
            self.assertEqual(MessageKind.CREDIT, read_frame(client).kind)
            return server, client, worker

        metadata = {
            "plan_id": self.manifest.plan_id,
            "plan_hash": self.manifest.plan_hash,
            "manifest_hash": self.manifest.manifest_hash,
            "request_id": REQUEST_A,
            "microbatch_id": "microbatch-reconnect",
            "source_stage": "orchestrator",
            "destination_stage": self.manifest.stage_id,
            "phase": "prefill",
            "token_start": 0,
            "token_count": 2,
            "kv_epoch": 0,
            "deadline_ns": time.monotonic_ns() + 10_000_000_000,
            "trace_id": "trace-reconnect",
            "span_id": "span-reconnect",
            "request_sequence_no": 0,
        }
        data = Frame.from_tensor(self.tensor, sequence_no=1, metadata=metadata)

        server, client, worker = start_connection()
        try:
            send_frame(client, data)
            self.assertEqual(
                (MessageKind.ACK, MessageKind.ACTIVATION, MessageKind.CREDIT),
                tuple(read_frame(client).kind for _ in range(3)),
            )
            release = Frame(
                MessageKind.HEARTBEAT,
                2,
                {
                    "sequence_no": 2,
                    "control": "release-request",
                    "request_id": REQUEST_A,
                },
            )
            send_frame(client, release)
            released = read_frame(client)
            self.assertEqual(MessageKind.ACK, released.kind)
            self.assertTrue(released.metadata["tombstoned"])
        finally:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            worker.join(timeout=2.0)
            server.close()
        self.assertFalse(worker.is_alive())

        server, client, worker = start_connection()
        try:
            send_frame(client, data)
            ack = read_frame(client)
            fenced = read_frame(client)
            credit = read_frame(client)
            self.assertEqual(MessageKind.ACK, ack.kind)
            self.assertEqual(MessageKind.ERROR, fenced.kind)
            self.assertEqual("REQUEST_TOMBSTONED", fenced.metadata["code"])
            self.assertEqual(MessageKind.CREDIT, credit.kind)
        finally:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            worker.join(timeout=2.0)
            server.close()
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
