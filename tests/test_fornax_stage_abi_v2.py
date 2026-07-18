from __future__ import annotations

import socket
import threading
import time
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from fornax.ragged_runtime import (
    IntegratedRaggedScheduler,
    RaggedGenerationRequest,
    RaggedReferenceStageBackend,
    RaggedRuntimeError,
    RaggedOrchestrator,
    RaggedWorkerChannel,
    RaggedWorkerProcess,
    start_ragged_engine,
    stop_ragged_engine,
)
from fornax.stage_abi_v2 import (
    ABI_MAJOR,
    ABI_MINOR,
    MAGIC,
    PRELUDE,
    BatchDescriptor,
    LogicalTensor,
    RaggedBatchRequest,
    RaggedBatchResult,
    RaggedFrame,
    RaggedFrameError,
    RaggedMessageKind,
    RaggedStageManifest,
    SequenceSlice,
    SequenceResult,
    decode_ragged_frame,
    encode_ragged_frame,
    derive_execution_lease_id,
    read_ragged_frame,
    send_ragged_frame,
)
from fornax.stage_abi import crc32c
from fornax.stage_runtime import canonical_json_bytes


PLAN_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
PLAN_HASH = "sha256:" + "b" * 64
REQUEST_A = "11111111-1111-4111-8111-111111111111"
REQUEST_B = "22222222-2222-4222-8222-222222222222"
REQUEST_C = "33333333-3333-4333-8333-333333333333"
PLAN_GENERATION_ID = "99999999-9999-4999-8999-999999999999"
LEASE_A = derive_execution_lease_id(PLAN_GENERATION_ID, PLAN_ID, PLAN_HASH, REQUEST_A)
LEASE_B = derive_execution_lease_id(PLAN_GENERATION_ID, PLAN_ID, PLAN_HASH, REQUEST_B)
LEASE_C = derive_execution_lease_id(PLAN_GENERATION_ID, PLAN_ID, PLAN_HASH, REQUEST_C)


def manifests() -> tuple[RaggedStageManifest, RaggedStageManifest]:
    return (
        RaggedStageManifest(
            plan_id=PLAN_ID,
            plan_hash=PLAN_HASH,
            stage_id="stage-0",
            stage_index=0,
            layer_start=0,
            layer_end=0,
            stage_role="first",
            input_kind="token_ids",
            output_kind="activation",
            dtype="bf16",
            hidden_size=4,
        ),
        RaggedStageManifest(
            plan_id=PLAN_ID,
            plan_hash=PLAN_HASH,
            stage_id="stage-1",
            stage_index=1,
            layer_start=1,
            layer_end=1,
            stage_role="final",
            input_kind="activation",
            output_kind="logits",
            dtype="bf16",
            hidden_size=4,
            vocabulary_size=8,
        ),
    )


def sequence(
    request_id: str,
    lease_id: str,
    *,
    start: int,
    count: int,
    sequence_no: int = 0,
    kv_epoch: int = 0,
    phase: str = "prefill",
    deadline_budget_ns: int | None = None,
) -> SequenceSlice:
    return SequenceSlice(
        request_id=request_id,
        request_sequence_no=sequence_no,
        input_row_start=start,
        input_row_count=count,
        token_position_start=0 if phase == "prefill" else sequence_no,
        kv_epoch=kv_epoch,
        deadline_budget_ns=(
            60_000_000_000
            if deadline_budget_ns is None
            else deadline_budget_ns
        ),
        execution_lease_id=lease_id,
        trace_id=f"trace-{request_id}",
        span_id=f"span-{sequence_no}",
    )


def prefill_request(
    manifest: RaggedStageManifest,
    *,
    batch_id: str = "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
) -> RaggedBatchRequest:
    descriptor = BatchDescriptor(
        batch_id=batch_id,
        batch_sequence_no=0,
        phase="prefill",
        input_row_count=5,
        sequences=(
            sequence(REQUEST_A, LEASE_A, start=0, count=2),
            sequence(REQUEST_B, LEASE_B, start=2, count=3),
        ),
    )
    return RaggedBatchRequest(
        plan_id=PLAN_ID,
        plan_hash=PLAN_HASH,
        manifest_hash=manifest.manifest_hash,
        descriptor=descriptor,
        input_tensor=LogicalTensor.from_values(
            [10, 11, 20, 21, 22], kind="token_ids", dtype="int32", shape=(5,)
        ),
    )


class StageAbiV2ContractTest(unittest.TestCase):
    def test_public_ragged_module_exports_exact_version(self) -> None:
        from fornax.ragged import (
            FNX2_ABI_VERSION,
            IntegratedRaggedScheduler as PublicScheduler,
            derive_execution_lease_id as public_derive_lease,
        )

        self.assertEqual((2, 0), FNX2_ABI_VERSION)
        self.assertIs(PublicScheduler, IntegratedRaggedScheduler)
        self.assertIs(public_derive_lease, derive_execution_lease_id)

    def test_reference_only_validation_does_not_claim_loopback_evidence(self) -> None:
        from fornax.fnx2_validation import validate_stage_abi_v2_golden

        fixture = (
            Path(__file__).parents[1]
            / "fornax"
            / "golden_vectors"
            / "stage_abi_v2"
            / "fixture.json"
        )
        report = validate_stage_abi_v2_golden(fixture, run_loopback=False)
        self.assertFalse(report["summary"]["loopback_executed"])
        self.assertFalse(report["summary"]["loopback_validated"])
        self.assertEqual("t0_reference_only", report["summary"]["evidence_class"])

    def test_unequal_prefill_and_decode_descriptor_rules(self) -> None:
        request = prefill_request(manifests()[0])
        self.assertEqual([2, 3], [item.input_row_count for item in request.descriptor.sequences])
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            BatchDescriptor(
                batch_id=str(uuid.uuid4()),
                batch_sequence_no=1,
                phase="decode",
                input_row_count=2,
                sequences=(
                    sequence(
                        REQUEST_A,
                        LEASE_A,
                        start=0,
                        count=2,
                        phase="decode",
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "without gaps or overlap"):
            BatchDescriptor(
                batch_id=str(uuid.uuid4()),
                batch_sequence_no=1,
                phase="prefill",
                input_row_count=2,
                sequences=(sequence(REQUEST_A, LEASE_A, start=1, count=1),),
            )

    def test_exact_fnx2_codec_round_trip_and_rejects_future_minor(self) -> None:
        request = prefill_request(manifests()[0])
        frame = RaggedFrame.from_batch(
            request,
            sequence_no=7,
            source_stage="gateway",
            destination_stage="stage-0",
        )
        encoded = encode_ragged_frame(frame)
        decoded = decode_ragged_frame(encoded)
        self.assertEqual(b"FNX2", encoded[:4])
        self.assertEqual(ABI_MAJOR, decoded.abi_major)
        self.assertEqual(ABI_MINOR, decoded.abi_minor)
        self.assertEqual(request, decoded.batch_request())

        fields = list(PRELUDE.unpack(encoded[: PRELUDE.size]))
        fields[2] = 1
        future_minor = PRELUDE.pack(*fields) + encoded[PRELUDE.size :]
        with self.assertRaisesRegex(RaggedFrameError, "unsupported frame"):
            decode_ragged_frame(future_minor)

        corrupted = bytearray(encoded)
        corrupted[-1] ^= 0x01
        with self.assertRaisesRegex(RaggedFrameError, "CRC32C"):
            decode_ragged_frame(bytes(corrupted))

    def test_reference_oracle_replays_after_repacking_without_second_kv_mutation(self) -> None:
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest, plan_generation_id=PLAN_GENERATION_ID
        )
        request = prefill_request(manifest)
        first = backend.execute(request)
        self.assertTrue(all(item.status == "ok" for item in first.results))
        self.assertTrue(all(item.kv_epoch_after == 1 for item in first.results))

        repacked_descriptor = BatchDescriptor(
            batch_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            batch_sequence_no=1,
            phase="prefill",
            input_row_count=5,
            sequences=(
                sequence(REQUEST_B, LEASE_B, start=0, count=3),
                sequence(REQUEST_A, LEASE_A, start=3, count=2),
            ),
        )
        repacked = RaggedBatchRequest(
            plan_id=PLAN_ID,
            plan_hash=PLAN_HASH,
            manifest_hash=manifest.manifest_hash,
            descriptor=repacked_descriptor,
            input_tensor=LogicalTensor.from_values(
                [20, 21, 22, 10, 11],
                kind="token_ids",
                dtype="int32",
                shape=(5,),
            ),
        )
        replay = backend.execute(repacked)
        self.assertTrue(all(item.status == "ok" for item in replay.results))
        self.assertTrue(all(item.kv_epoch_after == 1 for item in replay.results))
        self.assertEqual(first.output_tensor.values(), replay.output_tensor.values())

    def test_conflicting_replay_fails_only_that_sequence(self) -> None:
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest, plan_generation_id=PLAN_GENERATION_ID
        )
        backend.execute(prefill_request(manifest))
        conflict = prefill_request(
            manifest, batch_id="ffffffff-ffff-4fff-8fff-ffffffffffff"
        )
        values = list(conflict.input_tensor.values())
        values[0] = 999
        conflict = RaggedBatchRequest(
            plan_id=conflict.plan_id,
            plan_hash=conflict.plan_hash,
            manifest_hash=conflict.manifest_hash,
            descriptor=conflict.descriptor,
            input_tensor=LogicalTensor.from_values(
                values, kind="token_ids", dtype="int32", shape=(5,)
            ),
        )
        result = backend.execute(conflict)
        by_id = {item.request_id: item for item in result.results}
        self.assertEqual("rejected", by_id[REQUEST_A].status)
        self.assertEqual("SEQUENCE_CONFLICT", by_id[REQUEST_A].error["code"])
        self.assertEqual("ok", by_id[REQUEST_B].status)

    def test_partial_decode_failure_does_not_mutate_other_sequence_kv(self) -> None:
        now = [5_000_000]
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest,
            clock_ns=lambda: now[0],
            plan_generation_id=PLAN_GENERATION_ID,
        )
        initial_descriptor = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=0,
            phase="prefill",
            input_row_count=3,
            sequences=(
                sequence(REQUEST_A, LEASE_A, start=0, count=1, deadline_budget_ns=100),
                sequence(REQUEST_B, LEASE_B, start=1, count=1, deadline_budget_ns=100),
                sequence(REQUEST_C, LEASE_C, start=2, count=1, deadline_budget_ns=10),
            ),
        )
        initial = RaggedBatchRequest(
            PLAN_ID,
            PLAN_HASH,
            manifest.manifest_hash,
            initial_descriptor,
            LogicalTensor.from_values(
                [1, 2, 3], kind="token_ids", dtype="int32", shape=(3,)
            ),
        )
        self.assertTrue(all(item.status == "ok" for item in backend.execute(initial).results))
        backend.cancel(REQUEST_B, LEASE_B)
        decode_descriptor = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=1,
            phase="decode",
            input_row_count=3,
            sequences=(
                sequence(
                    REQUEST_A,
                    LEASE_A,
                    start=0,
                    count=1,
                    sequence_no=1,
                    kv_epoch=1,
                    phase="decode",
                    deadline_budget_ns=100,
                ),
                sequence(
                    REQUEST_B,
                    LEASE_B,
                    start=1,
                    count=1,
                    sequence_no=1,
                    kv_epoch=1,
                    phase="decode",
                    deadline_budget_ns=100,
                ),
                sequence(
                    REQUEST_C,
                    LEASE_C,
                    start=2,
                    count=1,
                    sequence_no=1,
                    kv_epoch=1,
                    phase="decode",
                    deadline_budget_ns=0,
                ),
            ),
        )
        decoded = backend.execute(
            RaggedBatchRequest(
                PLAN_ID,
                PLAN_HASH,
                manifest.manifest_hash,
                decode_descriptor,
                LogicalTensor.from_values(
                    [4, 5, 6], kind="token_ids", dtype="int32", shape=(3,)
                ),
            )
        )
        by_id = {item.request_id: item for item in decoded.results}
        self.assertEqual("ok", by_id[REQUEST_A].status)
        self.assertEqual("cancelled", by_id[REQUEST_B].status)
        self.assertEqual("deadline", by_id[REQUEST_C].status)
        self.assertEqual(2, by_id[REQUEST_A].kv_epoch_after)
        self.assertEqual(1, by_id[REQUEST_B].kv_epoch_after)
        self.assertEqual(1, by_id[REQUEST_C].kv_epoch_after)
        self.assertEqual((1, 4), decoded.output_tensor.descriptor.shape)

    def test_release_tombstone_fences_stale_replay_for_generation(self) -> None:
        now = [1_000_000]
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest,
            clock_ns=lambda: now[0],
            max_live_requests=2,
            max_completed_results=2,
            max_tombstones=2,
            plan_generation_id=PLAN_GENERATION_ID,
        )
        request = prefill_request(manifest)
        # Replace real deadlines with the fake clock domain.
        descriptor = BatchDescriptor(
            batch_id=request.descriptor.batch_id,
            batch_sequence_no=0,
            phase="prefill",
            input_row_count=5,
            sequences=tuple(
                replace_deadline(item, 50) for item in request.descriptor.sequences
            ),
        )
        request = RaggedBatchRequest(
            request.plan_id,
            request.plan_hash,
            request.manifest_hash,
            descriptor,
            request.input_tensor,
        )
        backend.execute(request)
        released = backend.release(REQUEST_A, LEASE_A, expected_final_epoch=1)
        self.assertTrue(released["released"])
        replay = backend.execute(request)
        by_id = {item.request_id: item for item in replay.results}
        self.assertEqual("RELEASED", by_id[REQUEST_A].error["code"])
        now[0] += 60
        backend.sweep_expired()
        health = backend.health()
        self.assertEqual(0, health.live_requests)
        self.assertLessEqual(health.tombstones, health.max_tombstones)

    def test_many_unique_request_release_stays_within_all_caps(self) -> None:
        now = [1_000_000]
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest,
            clock_ns=lambda: now[0],
            max_live_requests=1,
            max_completed_results=1,
            max_tombstones=40,
            plan_generation_id=PLAN_GENERATION_ID,
        )
        for index in range(40):
            request_id = str(uuid.uuid5(uuid.UUID(PLAN_ID), f"request-{index}"))
            lease_id = derive_execution_lease_id(
                PLAN_GENERATION_ID, PLAN_ID, PLAN_HASH, request_id
            )
            descriptor = BatchDescriptor(
                batch_id=str(uuid.uuid5(uuid.UUID(PLAN_ID), f"batch-{index}")),
                batch_sequence_no=index,
                phase="prefill",
                input_row_count=1,
                sequences=(
                    sequence(
                        request_id,
                        lease_id,
                        start=0,
                        count=1,
                        deadline_budget_ns=1,
                    ),
                ),
            )
            request = RaggedBatchRequest(
                PLAN_ID,
                PLAN_HASH,
                manifest.manifest_hash,
                descriptor,
                LogicalTensor.from_values(
                    [index], kind="token_ids", dtype="int32", shape=(1,)
                ),
            )
            result = backend.execute(request)
            self.assertEqual("ok", result.results[0].status)
            backend.release(request_id, lease_id, expected_final_epoch=1)
            now[0] += 2
        health = backend.health()
        self.assertEqual(0, health.live_requests)
        self.assertEqual(0, health.completed_results)
        self.assertEqual(0, health.kv_bytes)
        self.assertGreater(health.high_water_kv_bytes, 0)
        self.assertLessEqual(health.high_water_kv_bytes, health.max_kv_bytes)
        self.assertEqual(health.tombstones, 40)

    def test_generation_bound_retired_lease_never_revives_and_restart_fails_closed(self) -> None:
        now = [1_000]
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest,
            clock_ns=lambda: now[0],
            plan_generation_id=PLAN_GENERATION_ID,
            max_tombstones=1,
        )
        descriptor = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=0,
            phase="prefill",
            input_row_count=1,
            sequences=(
                sequence(
                    REQUEST_A,
                    LEASE_A,
                    start=0,
                    count=1,
                    deadline_budget_ns=100,
                ),
            ),
        )
        request = RaggedBatchRequest(
            PLAN_ID,
            PLAN_HASH,
            manifest.manifest_hash,
            descriptor,
            LogicalTensor.from_values([1], kind="token_ids", dtype="int32", shape=(1,)),
        )
        self.assertEqual("ok", backend.execute(request).results[0].status)
        backend.release(REQUEST_A, LEASE_A, expected_final_epoch=1)
        now[0] += 1_000_000
        self.assertEqual("RELEASED", backend.execute(request).results[0].error["code"])

        request_b = str(uuid.uuid5(uuid.UUID(PLAN_ID), "capacity-request"))
        lease_b = derive_execution_lease_id(
            PLAN_GENERATION_ID, PLAN_ID, PLAN_HASH, request_b
        )
        descriptor_b = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=1,
            phase="prefill",
            input_row_count=1,
            sequences=(
                sequence(
                    request_b,
                    lease_b,
                    start=0,
                    count=1,
                    deadline_budget_ns=100,
                ),
            ),
        )
        blocked = backend.execute(
            RaggedBatchRequest(
                PLAN_ID,
                PLAN_HASH,
                manifest.manifest_hash,
                descriptor_b,
                LogicalTensor.from_values(
                    [2], kind="token_ids", dtype="int32", shape=(1,)
                ),
            )
        )
        self.assertEqual("ADMISSION", blocked.results[0].error["code"])

        restarted = RaggedReferenceStageBackend(
            manifest, plan_generation_id="88888888-8888-4888-8888-888888888888"
        )
        self.assertEqual("UNKNOWN_LEASE", restarted.execute(request).results[0].error["code"])

    def test_phase_position_deadline_and_cancelled_replay_invariants(self) -> None:
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest, plan_generation_id=PLAN_GENERATION_ID
        )
        request = prefill_request(manifest)
        self.assertTrue(all(item.status == "ok" for item in backend.execute(request).results))

        changed_deadline = replace(
            request,
            descriptor=replace(
                request.descriptor,
                batch_id=str(uuid.uuid4()),
                sequences=tuple(
                    replace(
                        item,
                        deadline_budget_ns=item.deadline_budget_ns - 1,
                    )
                    for item in request.descriptor.sequences
                ),
            ),
        )
        tightened_replay = backend.execute(changed_deadline)
        self.assertTrue(all(item.status == "ok" for item in tightened_replay.results))

        wrong_position = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=1,
            phase="decode",
            input_row_count=1,
            sequences=(
                sequence(
                    REQUEST_A,
                    LEASE_A,
                    start=0,
                    count=1,
                    sequence_no=1,
                    kv_epoch=1,
                    phase="decode",
                    deadline_budget_ns=request.descriptor.sequences[
                        0
                    ].deadline_budget_ns,
                ),
            ),
        )
        wrong_position = replace(
            wrong_position,
            sequences=(replace(wrong_position.sequences[0], token_position_start=1),),
        )
        rejected = backend.execute(
            RaggedBatchRequest(
                PLAN_ID,
                PLAN_HASH,
                manifest.manifest_hash,
                wrong_position,
                LogicalTensor.from_values(
                    [7], kind="token_ids", dtype="int32", shape=(1,)
                ),
            )
        )
        self.assertEqual("POSITION", rejected.results[0].error["code"])

        backend.cancel(REQUEST_A, LEASE_A)
        replay = backend.execute(request)
        by_id = {item.request_id: item for item in replay.results}
        self.assertEqual("ok", by_id[REQUEST_A].status)
        self.assertEqual(1, by_id[REQUEST_A].kv_epoch_after)

    def test_relative_deadline_is_receiver_local_and_retry_can_only_tighten(self) -> None:
        manifest = manifests()[0]
        for initial_clock in (10, 10**18):
            now = [initial_clock]
            backend = RaggedReferenceStageBackend(
                manifest,
                clock_ns=lambda: now[0],
                plan_generation_id=PLAN_GENERATION_ID,
            )
            descriptor = BatchDescriptor(
                batch_id=str(uuid.uuid4()),
                batch_sequence_no=0,
                phase="prefill",
                input_row_count=1,
                sequences=(
                    sequence(
                        REQUEST_A,
                        LEASE_A,
                        start=0,
                        count=1,
                        deadline_budget_ns=100,
                    ),
                ),
            )
            request = RaggedBatchRequest(
                PLAN_ID,
                PLAN_HASH,
                manifest.manifest_hash,
                descriptor,
                LogicalTensor.from_values(
                    [1], kind="token_ids", dtype="int32", shape=(1,)
                ),
            )
            self.assertEqual("ok", backend.execute(request).results[0].status)

            now[0] += 20
            tightened = replace(
                request,
                descriptor=replace(
                    request.descriptor,
                    batch_id=str(uuid.uuid4()),
                    sequences=(
                        replace(
                            request.descriptor.sequences[0],
                            deadline_budget_ns=30,
                        ),
                    ),
                ),
            )
            self.assertEqual("ok", backend.execute(tightened).results[0].status)

            # The receiver stored min(initial+100, retry_time+30), so this is
            # expired at both a tiny and a highly skewed local clock value.
            now[0] += 31
            expired = backend.execute(tightened).results[0]
            self.assertEqual("RELEASED", expired.error["code"])

    def test_receiver_credit_is_atomic_and_exact_replay_needs_no_free_kv(self) -> None:
        manifest = manifests()[0]
        backend = RaggedReferenceStageBackend(
            manifest,
            plan_generation_id=PLAN_GENERATION_ID,
            max_live_requests=2,
            max_kv_bytes=40,
            max_completed_bytes=40,
        )
        request = prefill_request(manifest)
        initial_credit = backend.credit_snapshot(max_payload_bytes=1 << 20)
        first = backend.execute_with_credit(
            request, initial_credit, max_payload_bytes=1 << 20
        )
        self.assertTrue(all(item.status == "ok" for item in first.results))
        exhausted = backend.credit_snapshot(max_payload_bytes=1 << 20)
        self.assertEqual(0, exhausted["live_requests"])
        self.assertEqual(0, exhausted["kv_bytes"])
        self.assertEqual(0, exhausted["replay_bytes"])

        replay = backend.execute_with_credit(
            request, exhausted, max_payload_bytes=1 << 20
        )
        self.assertTrue(all(item.status == "ok" for item in replay.results))
        before = backend.health()

        new_descriptor = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=1,
            phase="prefill",
            input_row_count=1,
            sequences=(
                sequence(REQUEST_C, LEASE_C, start=0, count=1),
            ),
        )
        with self.assertRaisesRegex(RaggedRuntimeError, "receiver rejected") as error:
            backend.execute_with_credit(
                RaggedBatchRequest(
                    PLAN_ID,
                    PLAN_HASH,
                    manifest.manifest_hash,
                    new_descriptor,
                    LogicalTensor.from_values(
                        [3], kind="token_ids", dtype="int32", shape=(1,)
                    ),
                ),
                exhausted,
                max_payload_bytes=1 << 20,
            )
        self.assertEqual("CREDIT", error.exception.code)
        after = backend.health()
        self.assertEqual(before.kv_bytes, after.kv_bytes)
        self.assertEqual(before.completed_bytes, after.completed_bytes)
        self.assertEqual(before.live_requests, after.live_requests)

    def test_strict_metadata_integer_dtype_and_replay_byte_bounds(self) -> None:
        manifest = manifests()[0]
        with self.assertRaisesRegex(ValueError, "signed 32-bit"):
            LogicalTensor.from_values(
                [1.0], kind="token_ids", dtype="int32", shape=(1,)
            )
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            BatchDescriptor.from_dict(
                {**prefill_request(manifest).descriptor.to_dict(), "future": 1}
            )
        with self.assertRaisesRegex(ValueError, "no greater"):
            replace(
                prefill_request(manifest).descriptor.sequences[0],
                token_position_start=1 << 64,
            )
        frame = RaggedFrame.from_batch(
            prefill_request(manifest),
            sequence_no=0,
            source_stage="gateway",
            destination_stage="stage-0",
        )
        with self.assertRaisesRegex(RaggedFrameError, "unknown fields"):
            encode_ragged_frame(replace(frame, metadata={**frame.metadata, "future": 1}))

        bounded = RaggedReferenceStageBackend(
            manifest,
            plan_generation_id=PLAN_GENERATION_ID,
            max_completed_bytes=8,
        )
        single_descriptor = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=0,
            phase="prefill",
            input_row_count=2,
            sequences=(sequence(REQUEST_A, LEASE_A, start=0, count=2),),
        )
        with self.assertRaisesRegex(RaggedRuntimeError, "atomically reserve") as error:
            bounded.execute(
                RaggedBatchRequest(
                    PLAN_ID,
                    PLAN_HASH,
                    manifest.manifest_hash,
                    single_descriptor,
                    LogicalTensor.from_values(
                        [1, 2], kind="token_ids", dtype="int32", shape=(2,)
                    ),
                )
            )
        self.assertEqual("ADMISSION", error.exception.code)
        self.assertEqual(0, bounded.health().kv_bytes)
        self.assertLessEqual(
            bounded.health().high_water_completed_bytes,
            bounded.health().max_completed_bytes,
        )

        result = RaggedReferenceStageBackend(
            manifest, plan_generation_id=PLAN_GENERATION_ID
        ).execute(prefill_request(manifest))
        result_frame = RaggedFrame.from_result(result, sequence_no=0)
        with self.assertRaisesRegex(RaggedFrameError, "batch_sequence_no"):
            encode_ragged_frame(
                replace(
                    result_frame,
                    metadata={**result_frame.metadata, "batch_sequence_no": True},
                )
            )
        bad_transition = replace(
            result,
            results=(
                replace(result.results[0], kv_epoch_after=2),
                result.results[1],
            ),
        )
        with self.assertRaisesRegex(ValueError, "KV/output transition"):
            bad_transition.validate_for(prefill_request(manifest))

    def test_final_logits_batch_replay_bytes_are_reserved_atomically(self) -> None:
        manifest = manifests()[1]
        backend = RaggedReferenceStageBackend(
            manifest,
            plan_generation_id=PLAN_GENERATION_ID,
            max_completed_bytes=16,
        )
        descriptor = BatchDescriptor(
            batch_id=str(uuid.uuid4()),
            batch_sequence_no=0,
            phase="prefill",
            input_row_count=2,
            sequences=(
                sequence(REQUEST_A, LEASE_A, start=0, count=1),
                sequence(REQUEST_B, LEASE_B, start=1, count=1),
            ),
        )
        request = RaggedBatchRequest(
            PLAN_ID,
            PLAN_HASH,
            manifest.manifest_hash,
            descriptor,
            LogicalTensor.from_values(
                [0.0] * 8,
                kind="activation",
                dtype="bf16",
                shape=(2, 4),
            ),
        )
        with self.assertRaisesRegex(RaggedRuntimeError, "atomically reserve"):
            backend.execute(request)
        health = backend.health()
        self.assertEqual(0, health.kv_bytes)
        self.assertEqual(0, health.live_requests)
        self.assertEqual(0, health.completed_results)

    def test_result_correlation_mismatch_quarantines_channel(self) -> None:
        manifest = manifests()[0]
        request = prefill_request(manifest)
        backend = RaggedReferenceStageBackend(
            manifest, plan_generation_id=PLAN_GENERATION_ID
        )
        client_socket, server_socket = socket.socketpair()
        channel = RaggedWorkerChannel(
            manifest,
            "unused",
            0,
            plan_generation_id=PLAN_GENERATION_ID,
        )
        channel._socket = client_socket
        channel._credits = {name: 1 << 30 for name in channel._credits}

        def serve_wrong_result() -> None:
            try:
                incoming = read_ragged_frame(server_socket)
                result = backend.execute(incoming.batch_request())
                wrong = replace(result, batch_id=str(uuid.uuid4()))
                send_ragged_frame(
                    server_socket, RaggedFrame.from_result(wrong, sequence_no=0)
                )
            finally:
                server_socket.close()

        thread = threading.Thread(target=serve_wrong_result)
        thread.start()
        with self.assertRaisesRegex(RaggedRuntimeError, "uncorrelated"):
            channel.execute(request, source_stage="gateway")
        thread.join(timeout=2)
        self.assertTrue(channel.quarantined)
        with self.assertRaisesRegex(RaggedRuntimeError, "quarantined|correlation"):
            channel.execute(request, source_stage="gateway")

    def test_hello_generation_mismatch_quarantines_before_credit(self) -> None:
        manifest = manifests()[0]
        client_socket, server_socket = socket.socketpair()
        channel = RaggedWorkerChannel(
            manifest,
            "unused",
            0,
            plan_generation_id=PLAN_GENERATION_ID,
        )
        channel._socket = client_socket
        hello = RaggedFrame(
            kind=RaggedMessageKind.CONTROL,
            sequence_no=0,
            metadata={
                "sequence_no": 0,
                "control_id": str(uuid.uuid4()),
                "control": "hello",
                "plan_id": manifest.plan_id,
                "plan_hash": manifest.plan_hash,
                "manifest_hash": manifest.manifest_hash,
                "stage_id": manifest.stage_id,
                "plan_generation_id": "88888888-8888-4888-8888-888888888888",
                "supported_versions": [[2, 0]],
            },
        )
        send_ragged_frame(server_socket, hello)
        with self.assertRaisesRegex(RaggedRuntimeError, "hello does not match"):
            channel._perform_handshake()
        self.assertTrue(channel.quarantined)
        server_socket.close()

    def test_length_delimited_semantic_error_keeps_worker_channel_live(self) -> None:
        manifest = manifests()[0]
        worker = RaggedWorkerProcess(
            manifest, plan_generation_id=PLAN_GENERATION_ID
        )
        endpoint = worker.start()
        channel = socket.create_connection((endpoint["host"], endpoint["port"]))
        self.addCleanup(channel.close)
        self.addCleanup(worker.join)

        hello = read_ragged_frame(channel)
        self.assertEqual("hello", hello.metadata["control"])
        ack_metadata = {
            "sequence_no": 0,
            "control_id": hello.metadata["control_id"],
            "operation": "hello",
            "plan_id": manifest.plan_id,
            "plan_hash": manifest.plan_hash,
            "manifest_hash": manifest.manifest_hash,
            "stage_id": manifest.stage_id,
            "plan_generation_id": PLAN_GENERATION_ID,
            "selected_version": [2, 0],
        }
        send_ragged_frame(
            channel,
            RaggedFrame(RaggedMessageKind.ACK, 0, ack_metadata),
        )
        self.assertEqual(RaggedMessageKind.CREDIT, read_ragged_frame(channel).kind)

        request = prefill_request(manifest)
        valid_frame = RaggedFrame.from_batch(
            request,
            sequence_no=1,
            source_stage="gateway",
            destination_stage=manifest.stage_id,
        )
        malformed_metadata = {**valid_frame.metadata, "unknown_future_field": True}
        metadata_bytes = canonical_json_bytes(malformed_metadata)
        malformed = PRELUDE.pack(
            MAGIC,
            ABI_MAJOR,
            ABI_MINOR,
            int(RaggedMessageKind.BATCH),
            0,
            len(metadata_bytes),
            len(valid_frame.payload),
            1,
            crc32c(metadata_bytes + valid_frame.payload),
            0,
        ) + metadata_bytes + valid_frame.payload
        channel.sendall(malformed)
        semantic_error = read_ragged_frame(channel)
        self.assertEqual(RaggedMessageKind.ERROR, semantic_error.kind)
        self.assertEqual("BATCH_CONTRACT", semantic_error.metadata["code"])

        send_ragged_frame(
            channel,
            RaggedFrame.from_batch(
                request,
                sequence_no=2,
                source_stage="wrong-upstream",
                destination_stage=manifest.stage_id,
            ),
        )
        source_error = read_ragged_frame(channel)
        self.assertEqual(RaggedMessageKind.ERROR, source_error.kind)
        self.assertEqual("STALE_PLAN", source_error.metadata["code"])

        refresh_id = str(uuid.uuid4())
        send_ragged_frame(
            channel,
            RaggedFrame(
                RaggedMessageKind.CONTROL,
                3,
                {
                    "sequence_no": 3,
                    "control_id": refresh_id,
                    "control": "credit_refresh",
                    "plan_id": manifest.plan_id,
                    "plan_hash": manifest.plan_hash,
                    "manifest_hash": manifest.manifest_hash,
                    "stage_id": manifest.stage_id,
                    "plan_generation_id": PLAN_GENERATION_ID,
                },
            ),
        )
        refresh_ack = read_ragged_frame(channel)
        self.assertEqual("credit_refresh", refresh_ack.metadata["operation"])
        self.assertEqual(RaggedMessageKind.CREDIT, read_ragged_frame(channel).kind)

        send_ragged_frame(
            channel,
            RaggedFrame.from_batch(
                request,
                sequence_no=4,
                source_stage="gateway",
                destination_stage=manifest.stage_id,
            ),
        )
        result = read_ragged_frame(channel)
        self.assertEqual(RaggedMessageKind.RESULT, result.kind)
        self.assertTrue(all(item.status == "ok" for item in result.batch_result().results))
        self.assertEqual(RaggedMessageKind.CREDIT, read_ragged_frame(channel).kind)

        control_id = str(uuid.uuid4())
        send_ragged_frame(
            channel,
            RaggedFrame(
                RaggedMessageKind.CONTROL,
                5,
                {
                    "sequence_no": 5,
                    "control_id": control_id,
                    "control": "shutdown",
                    "plan_id": manifest.plan_id,
                    "plan_hash": manifest.plan_hash,
                    "manifest_hash": manifest.manifest_hash,
                    "stage_id": manifest.stage_id,
                    "plan_generation_id": PLAN_GENERATION_ID,
                },
            ),
        )
        shutdown = read_ragged_frame(channel)
        self.assertEqual(control_id, shutdown.metadata["control_id"])

    def test_expiry_credit_refresh_returns_live_kv_and_replay_capacity(self) -> None:
        manifest = manifests()[0]
        worker = RaggedWorkerProcess(
            manifest, plan_generation_id=PLAN_GENERATION_ID
        )
        endpoint = worker.start()
        channel = RaggedWorkerChannel(
            manifest,
            endpoint["host"],
            endpoint["port"],
            plan_generation_id=PLAN_GENERATION_ID,
        )
        try:
            channel.connect()
            request = prefill_request(manifest)
            request = replace(
                request,
                descriptor=replace(
                    request.descriptor,
                    sequences=tuple(
                        replace(item, deadline_budget_ns=100_000_000)
                        for item in request.descriptor.sequences
                    ),
                ),
            )
            result = channel.execute(request, source_stage="gateway")
            self.assertTrue(all(item.status == "ok" for item in result.results))
            before = dict(channel._credits)
            time.sleep(0.2)
            after = channel.refresh_credits()
            self.assertGreater(after["live_requests"], before["live_requests"])
            self.assertGreater(after["kv_bytes"], before["kv_bytes"])
            self.assertGreater(after["replay_bytes"], before["replay_bytes"])
        finally:
            try:
                channel.shutdown()
            finally:
                worker.join()

    def test_cleanup_saga_continues_after_partial_stage_failure(self) -> None:
        stage_manifests = manifests()

        class StubChannel:
            def __init__(self, manifest: RaggedStageManifest, fail: bool) -> None:
                self.manifest = manifest
                self.plan_generation_id = PLAN_GENERATION_ID
                self.fail = fail

            def release(self, request_id: str, lease_id: str, **_: object) -> dict[str, object]:
                if self.fail:
                    raise OSError("injected cleanup transport failure")
                return {
                    "released": True,
                    "request_id": request_id,
                    "execution_lease_id": lease_id,
                    "kv_epoch": 1,
                    "completed_results_released": 1,
                    "tombstone_present": True,
                }

        first = StubChannel(stage_manifests[0], True)
        second = StubChannel(stage_manifests[1], False)
        orchestrator = RaggedOrchestrator(
            [(stage_manifests[0], first), (stage_manifests[1], second)]  # type: ignore[list-item]
        )
        partial = orchestrator.release(REQUEST_A, LEASE_A)
        self.assertFalse(partial[0]["cleanup_complete"])
        self.assertEqual("TRANSPORT", partial[0]["error"]["code"])
        self.assertTrue(partial[1]["cleanup_complete"])
        self.assertEqual(("stage-0",), orchestrator.cleanup_pending[(REQUEST_A, LEASE_A)])
        first.fail = False
        retried = orchestrator.release(REQUEST_A, LEASE_A)
        self.assertTrue(all(item["cleanup_complete"] for item in retried))
        self.assertNotIn((REQUEST_A, LEASE_A), orchestrator.cleanup_pending)

    def test_cleanup_pending_capacity_is_reserved_before_stage_mutation(self) -> None:
        stage_manifests = manifests()

        class FailingChannel:
            def __init__(self, manifest: RaggedStageManifest) -> None:
                self.manifest = manifest
                self.plan_generation_id = PLAN_GENERATION_ID
                self.release_calls = 0

            def release(self, *_: object, **__: object) -> dict[str, object]:
                self.release_calls += 1
                raise OSError("offline")

        channels = [FailingChannel(item) for item in stage_manifests]
        orchestrator = RaggedOrchestrator(
            list(zip(stage_manifests, channels)),  # type: ignore[arg-type]
            max_cleanup_pending=1,
        )
        orchestrator.release(REQUEST_A, LEASE_A)
        calls_before = [item.release_calls for item in channels]
        with self.assertRaisesRegex(RaggedRuntimeError, "cleanup-pending") as error:
            orchestrator.release(REQUEST_B, LEASE_B)
        self.assertEqual("CLEANUP_CAPACITY", error.exception.code)
        self.assertEqual(calls_before, [item.release_calls for item in channels])
        self.assertEqual(1, len(orchestrator.cleanup_pending))

    def test_orchestrator_normalizes_compacted_results_to_original_rows(self) -> None:
        stage_manifests = manifests()

        class StubChannel:
            def __init__(self, manifest: RaggedStageManifest) -> None:
                self.manifest = manifest
                self.plan_generation_id = PLAN_GENERATION_ID

            def execute(
                self, request: RaggedBatchRequest, *, source_stage: str
            ) -> RaggedBatchResult:
                self.assert_source(source_stage)
                rows = request.descriptor.sequences
                if self.manifest.stage_index == 0:
                    first, second = rows
                    results = (
                        SequenceResult(
                            request_id=first.request_id,
                            execution_lease_id=first.execution_lease_id,
                            request_sequence_no=first.request_sequence_no,
                            status="failed",
                            input_row_start=first.input_row_start,
                            input_row_count=first.input_row_count,
                            output_row_start=None,
                            output_row_count=0,
                            kv_epoch_before=first.kv_epoch,
                            kv_epoch_after=first.kv_epoch,
                            error={"code": "INJECTED", "message": "injected"},
                            timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                        ),
                        SequenceResult(
                            request_id=second.request_id,
                            execution_lease_id=second.execution_lease_id,
                            request_sequence_no=second.request_sequence_no,
                            status="ok",
                            input_row_start=second.input_row_start,
                            input_row_count=second.input_row_count,
                            output_row_start=0,
                            output_row_count=second.input_row_count,
                            kv_epoch_before=second.kv_epoch,
                            kv_epoch_after=second.kv_epoch + 1,
                            error=None,
                            timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                        ),
                    )
                    output = LogicalTensor.from_values(
                        [0.0] * (second.input_row_count * 4),
                        kind="activation",
                        dtype="bf16",
                        shape=(second.input_row_count, 4),
                    )
                else:
                    only = rows[0]
                    results = (
                        SequenceResult(
                            request_id=only.request_id,
                            execution_lease_id=only.execution_lease_id,
                            request_sequence_no=only.request_sequence_no,
                            status="ok",
                            input_row_start=only.input_row_start,
                            input_row_count=only.input_row_count,
                            output_row_start=0,
                            output_row_count=only.input_row_count,
                            kv_epoch_before=only.kv_epoch,
                            kv_epoch_after=only.kv_epoch + 1,
                            error=None,
                            timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                        ),
                    )
                    output = LogicalTensor.from_values(
                        [0.0] * (only.input_row_count * 8),
                        kind="logits",
                        dtype="bf16",
                        shape=(only.input_row_count, 8),
                    )
                return RaggedBatchResult(
                    batch_id=request.descriptor.batch_id,
                    batch_sequence_no=request.descriptor.batch_sequence_no,
                    manifest_hash=request.manifest_hash,
                    results=results,
                    output_tensor=output,
                )

            def assert_source(self, value: str) -> None:
                expected = "gateway" if self.manifest.stage_index == 0 else "stage-0"
                if value != expected:
                    raise AssertionError(f"unexpected source {value}")

        channels = [StubChannel(item) for item in stage_manifests]
        orchestrator = RaggedOrchestrator(
            list(zip(stage_manifests, channels))  # type: ignore[arg-type]
        )
        request = prefill_request(stage_manifests[0])
        result = orchestrator.execute(request.descriptor, request.input_tensor)
        by_id = {item.request_id: item for item in result.results}
        self.assertEqual(0, by_id[REQUEST_A].input_row_start)
        self.assertEqual(2, by_id[REQUEST_A].input_row_count)
        self.assertEqual(2, by_id[REQUEST_B].input_row_start)
        self.assertEqual(3, by_id[REQUEST_B].input_row_count)
        self.assertEqual(0, by_id[REQUEST_B].output_row_start)
        self.assertEqual((3, 8), result.output_tensor.descriptor.shape)

    def test_scheduler_total_bounds_ready_subset_and_decode_max_rows(self) -> None:
        class StubOrchestrator:
            def issue_lease(self, request_id: str) -> str:
                return derive_execution_lease_id(
                    PLAN_GENERATION_ID, PLAN_ID, PLAN_HASH, request_id
                )

            def execute(
                self, descriptor: BatchDescriptor, tensor: LogicalTensor
            ) -> RaggedBatchResult:
                results: list[SequenceResult] = []
                cursor = 0
                for item in sorted(
                    descriptor.sequences,
                    key=lambda value: (value.request_id, value.execution_lease_id),
                ):
                    results.append(
                        SequenceResult(
                            request_id=item.request_id,
                            execution_lease_id=item.execution_lease_id,
                            request_sequence_no=item.request_sequence_no,
                            status="ok",
                            input_row_start=item.input_row_start,
                            input_row_count=item.input_row_count,
                            output_row_start=cursor,
                            output_row_count=item.input_row_count,
                            kv_epoch_before=item.kv_epoch,
                            kv_epoch_after=item.kv_epoch + 1,
                            error=None,
                            timings_ns={
                                "queue": 0,
                                "execute": 0,
                                "pack": 0,
                                "unpack": 0,
                            },
                        )
                    )
                    cursor += item.input_row_count
                return RaggedBatchResult(
                    batch_id=descriptor.batch_id,
                    batch_sequence_no=descriptor.batch_sequence_no,
                    manifest_hash="sha256:" + "c" * 64,
                    results=tuple(results),
                    output_tensor=LogicalTensor.from_values(
                        [0.0] * (cursor * 2),
                        kind="logits",
                        dtype="bf16",
                        shape=(cursor, 2),
                    ),
                )

        orchestrator = StubOrchestrator()
        scheduler = IntegratedRaggedScheduler(
            PLAN_ID,
            max_sequences=3,
            max_rows=4,
            max_queued=3,
            max_active=3,
            max_terminal=3,
            max_total=3,
        )
        deadline = time.monotonic_ns() + 60_000_000_000
        requests = [
            RaggedGenerationRequest(REQUEST_A, LEASE_A, (1, 2, 3), deadline, 3),
            RaggedGenerationRequest(REQUEST_B, LEASE_B, (4, 5, 6), deadline, 3),
            RaggedGenerationRequest(REQUEST_C, LEASE_C, (7,), deadline, 3),
        ]
        self.assertTrue(all(scheduler.submit(item) for item in requests))
        prefill = scheduler.run_prefill(orchestrator)  # type: ignore[arg-type]
        assert prefill is not None
        self.assertEqual({REQUEST_A, REQUEST_C}, {item.request_id for item in prefill.results})
        self.assertEqual(1, scheduler.stats["queued"])
        decode = scheduler.run_decode(  # type: ignore[arg-type]
            orchestrator, {REQUEST_C: 9}
        )
        assert decode is not None
        self.assertEqual([REQUEST_C], [item.request_id for item in decode.results])
        self.assertEqual(1, decode.output_tensor.descriptor.row_count)

        bounded = IntegratedRaggedScheduler(
            PLAN_ID, max_queued=2, max_active=1, max_terminal=1, max_total=2
        )
        self.assertTrue(bounded.submit(requests[0]))
        self.assertTrue(bounded.submit(requests[1]))
        self.assertFalse(bounded.submit(requests[2]))

    def test_scheduler_release_validates_lease_before_removing_state(self) -> None:
        class ReleaseStub:
            release_calls = 0

            def release(self, *_: object) -> tuple[dict[str, object], ...]:
                self.release_calls += 1
                return ()

        scheduler = IntegratedRaggedScheduler(PLAN_ID)
        valid = RaggedGenerationRequest(
            REQUEST_A,
            LEASE_A,
            (1,),
            time.monotonic_ns() + 60_000_000_000,
            1,
        )
        stale = replace(
            valid,
            execution_lease_id="88888888-8888-4888-8888-888888888888",
        )
        self.assertTrue(scheduler.submit(valid))
        orchestrator = ReleaseStub()
        with self.assertRaisesRegex(RaggedRuntimeError, "release lease"):
            scheduler.release(orchestrator, stale)  # type: ignore[arg-type]
        self.assertEqual(0, orchestrator.release_calls)
        self.assertEqual(1, scheduler.stats["queued"])

    def test_scheduler_auto_cleanup_retains_only_partial_saga_state(self) -> None:
        class CleanupStub:
            fail_cleanup = True
            release_calls = 0

            def issue_lease(self, request_id: str) -> str:
                return derive_execution_lease_id(
                    PLAN_GENERATION_ID, PLAN_ID, PLAN_HASH, request_id
                )

            def execute(
                self, descriptor: BatchDescriptor, tensor: LogicalTensor
            ) -> RaggedBatchResult:
                item = descriptor.sequences[0]
                return RaggedBatchResult(
                    batch_id=descriptor.batch_id,
                    batch_sequence_no=descriptor.batch_sequence_no,
                    manifest_hash="sha256:" + "d" * 64,
                    results=(
                        SequenceResult(
                            request_id=item.request_id,
                            execution_lease_id=item.execution_lease_id,
                            request_sequence_no=item.request_sequence_no,
                            status="failed",
                            input_row_start=item.input_row_start,
                            input_row_count=item.input_row_count,
                            output_row_start=None,
                            output_row_count=0,
                            kv_epoch_before=item.kv_epoch,
                            kv_epoch_after=item.kv_epoch,
                            error={"code": "INJECTED", "message": "injected"},
                            timings_ns={
                                "queue": 0,
                                "execute": 0,
                                "pack": 0,
                                "unpack": 0,
                            },
                        ),
                    ),
                    output_tensor=None,
                )

            def release(self, request_id: str, lease_id: str) -> tuple[dict[str, object], ...]:
                self.release_calls += 1
                return (
                    {
                        "stage_id": "stage-0",
                        "released": not self.fail_cleanup,
                        "cleanup_complete": not self.fail_cleanup,
                    },
                    {
                        "stage_id": "stage-1",
                        "released": True,
                        "cleanup_complete": True,
                    },
                )

        orchestrator = CleanupStub()
        scheduler = IntegratedRaggedScheduler(
            PLAN_ID,
            max_sequences=1,
            max_rows=4,
            max_queued=1,
            max_active=1,
            max_terminal=1,
            max_total=1,
        )
        request = RaggedGenerationRequest(
            REQUEST_A,
            LEASE_A,
            (1,),
            time.monotonic_ns() + 60_000_000_000,
            1,
        )
        self.assertTrue(scheduler.submit(request))
        self.assertIsNotNone(scheduler.run_prefill(orchestrator))  # type: ignore[arg-type]
        self.assertEqual(1, orchestrator.release_calls)
        self.assertEqual(1, scheduler.stats["terminal"])
        self.assertEqual(0, scheduler.stats["queued"])
        orchestrator.fail_cleanup = False
        scheduler.release(orchestrator, request)  # type: ignore[arg-type]
        self.assertEqual(2, orchestrator.release_calls)
        self.assertEqual(0, scheduler.stats["terminal"])


    def test_two_independent_workers_execute_integrated_ragged_scheduler(self) -> None:
        stage_manifests = manifests()
        workers, channels, orchestrator = start_ragged_engine(
            stage_manifests, plan_generation_id=PLAN_GENERATION_ID
        )
        self.addCleanup(stop_ragged_engine, workers, channels)
        self.assertEqual(2, len({worker.pid for worker in workers}))

        scheduler = IntegratedRaggedScheduler(PLAN_ID, max_sequences=3, max_rows=16)
        deadline = time.monotonic_ns() + 60_000_000_000
        request_a = RaggedGenerationRequest(
            REQUEST_A, LEASE_A, (1, 2), deadline, 2
        )
        request_b = RaggedGenerationRequest(
            REQUEST_B, LEASE_B, (3, 4, 5, 6), deadline, 3
        )
        self.assertTrue(scheduler.submit(request_a))
        self.assertTrue(scheduler.submit(request_b))
        prefill = scheduler.run_prefill(orchestrator)
        self.assertIsNotNone(prefill)
        assert prefill is not None
        self.assertEqual([2, 4], [
            item.input_row_count
            for item in sorted(prefill.results, key=lambda value: value.request_id)
        ])
        self.assertEqual((6, 8), prefill.output_tensor.descriptor.shape)
        refreshed = [channel.refresh_credits() for channel in channels]
        self.assertTrue(all(item["frames"] == 1 for item in refreshed))

        decode = scheduler.run_decode(
            orchestrator, {REQUEST_A: 7, REQUEST_B: 8}
        )
        self.assertIsNotNone(decode)
        assert decode is not None
        self.assertEqual((2, 8), decode.output_tensor.descriptor.shape)
        self.assertTrue(all(item.kv_epoch_after == 2 for item in decode.results))

        cancel_results = scheduler.cancel(orchestrator, REQUEST_B)
        self.assertEqual(2, len(cancel_results))
        decode_a = scheduler.run_decode(orchestrator, {REQUEST_A: 9})
        self.assertIsNotNone(decode_a)
        assert decode_a is not None
        self.assertEqual([REQUEST_A], [item.request_id for item in decode_a.results])
        self.assertEqual((1, 8), decode_a.output_tensor.descriptor.shape)

        released_a = scheduler.release(orchestrator, request_a)
        released_b = scheduler.release(orchestrator, request_b)
        self.assertTrue(all(item["cleanup_complete"] for item in released_a))
        self.assertTrue(all(item["cleanup_complete"] for item in released_b))
        self.assertTrue(
            all(item.get("released") or item.get("tombstone_present") for item in released_a)
        )
        self.assertTrue(
            all(item.get("released") or item.get("tombstone_present") for item in released_b)
        )
        self.assertEqual(0, scheduler.stats["active"])
        self.assertIn(
            "ragged_stage_result", {event["kind"] for event in orchestrator.events}
        )


def replace_deadline(
    item: SequenceSlice, deadline_budget_ns: int
) -> SequenceSlice:
    return SequenceSlice(
        request_id=item.request_id,
        request_sequence_no=item.request_sequence_no,
        input_row_start=item.input_row_start,
        input_row_count=item.input_row_count,
        token_position_start=item.token_position_start,
        kv_epoch=item.kv_epoch,
        deadline_budget_ns=deadline_budget_ns,
        execution_lease_id=item.execution_lease_id,
        trace_id=item.trace_id,
        span_id=item.span_id,
    )


if __name__ == "__main__":
    unittest.main()
