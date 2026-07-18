"""Slow-correct FNX2 ragged runtime, loopback workers, and integrated scheduler.

The implementation is intentionally model-free.  It is the executable oracle
that physical MAX adapters must match, not a production tensor hot path or a
performance result.
"""

from __future__ import annotations

import multiprocessing
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from typing import Any, Callable

from .stage_abi_v2 import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    DTYPE_BYTES,
    BatchDescriptor,
    LogicalTensor,
    RaggedBatchRequest,
    RaggedBatchResult,
    RaggedFrame,
    RaggedFrameError,
    RaggedMessageKind,
    RaggedStageManifest,
    SequenceResult,
    SequenceSlice,
    derive_execution_lease_id,
    read_ragged_frame,
    send_ragged_frame,
)
from .stage_runtime import deterministic_stage_values


class RaggedRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


ReplayBaseKey = tuple[str, str, str, str, int, str, int, int]
WORKER_MAX_ROWS = 65_536
WORKER_MAX_SEQUENCES = 1_024


def _semantic_replay_key(
    request: RaggedBatchRequest, sequence: SequenceSlice
) -> ReplayBaseKey:
    return (
        request.plan_hash,
        request.manifest_hash,
        sequence.request_id,
        sequence.execution_lease_id,
        sequence.request_sequence_no,
        request.descriptor.phase,
        sequence.token_position_start,
        sequence.kv_epoch,
    )


@dataclass(frozen=True)
class _CachedSequence:
    input_digest: str
    output_tensor: LogicalTensor
    kv_epoch_before: int
    kv_epoch_after: int


@dataclass
class _SequenceState:
    request_id: str
    execution_lease_id: str
    lease_expiry_ns: int
    kv_epoch: int = 0
    highest_sequence_no: int = -1
    next_token_position: int = 0
    cancelled: bool = False
    inflight: int = 0
    kv_bytes: int = 0
    completed: OrderedDict[ReplayBaseKey, _CachedSequence] = field(
        default_factory=OrderedDict
    )


@dataclass(frozen=True)
class RaggedBackendHealth:
    state: str
    stage_id: str
    live_requests: int
    completed_results: int
    completed_bytes: int
    tombstones: int
    kv_bytes: int
    max_live_requests: int
    max_completed_results: int
    max_completed_bytes: int
    max_tombstones: int
    max_kv_bytes: int
    high_water_live_requests: int
    high_water_completed_results: int
    high_water_completed_bytes: int
    high_water_tombstones: int
    high_water_kv_bytes: int


class RaggedReferenceStageBackend:
    """Deterministic, bounded FNX2 oracle with per-sequence isolation."""

    def __init__(
        self,
        manifest: RaggedStageManifest,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_live_requests: int = 4096,
        max_completed_results: int = 8192,
        max_completed_bytes: int = 256 * 1024 * 1024,
        max_tombstones: int = 8192,
        max_kv_bytes: int = 512 * 1024 * 1024,
        plan_generation_id: str | None = None,
    ) -> None:
        for name, value in {
            "max_live_requests": max_live_requests,
            "max_completed_results": max_completed_results,
            "max_completed_bytes": max_completed_bytes,
            "max_tombstones": max_tombstones,
            "max_kv_bytes": max_kv_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.manifest = manifest
        self._clock_ns = clock_ns
        self.max_live_requests = max_live_requests
        self.max_completed_results = max_completed_results
        self.max_completed_bytes = max_completed_bytes
        self.max_tombstones = max_tombstones
        self.max_kv_bytes = max_kv_bytes
        self.plan_generation_id = plan_generation_id or str(uuid.uuid4())
        try:
            uuid.UUID(self.plan_generation_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("plan_generation_id must be a canonical UUID") from exc
        if str(uuid.UUID(self.plan_generation_id)) != self.plan_generation_id:
            raise ValueError("plan_generation_id must be a canonical UUID")
        self._requests: dict[tuple[str, str], _SequenceState] = {}
        self._request_leases: dict[str, str] = {}
        self._completed_lru: OrderedDict[tuple[tuple[str, str], ReplayBaseKey], None] = (
            OrderedDict()
        )
        self._tombstones: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._kv_bytes = 0
        self._completed_bytes = 0
        self._lock = threading.RLock()
        self._high_water_live = 0
        self._high_water_completed = 0
        self._high_water_completed_bytes = 0
        self._high_water_tombstones = 0
        self._high_water_kv_bytes = 0

    def _sweep(self, now_ns: int) -> None:
        # Tombstones deliberately live for the whole immutable plan generation.
        # Expiring them would allow a valid old lease to become a new request.
        expired = [
            key
            for key, state in self._requests.items()
            if state.lease_expiry_ns <= now_ns and state.inflight == 0
        ]
        for key in expired:
            try:
                self._drop_state(key, now_ns=now_ns, tombstone=True)
            except RaggedRuntimeError as exc:
                if exc.code != "TOMBSTONE_CAPACITY":
                    raise
                # Fail closed: retain the bounded live/KV state until the plan
                # generation drains instead of losing the anti-replay fence.

    def sweep_expired(self, now_ns: int | None = None) -> int:
        with self._lock:
            before = len(self._requests)
            self._sweep(self._clock_ns() if now_ns is None else now_ns)
            return before - len(self._requests)

    def _install_tombstone(
        self, key: tuple[str, str], now_ns: int, lease_expiry_ns: int
    ) -> None:
        self._tombstones.pop(key, None)
        if len(self._tombstones) >= self.max_tombstones:
            raise RaggedRuntimeError(
                "TOMBSTONE_CAPACITY",
                "release tombstone capacity is exhausted; admission remains fail-closed",
                retryable=True,
            )
        self._tombstones[key] = now_ns
        self._high_water_tombstones = max(
            self._high_water_tombstones, len(self._tombstones)
        )

    def _drop_state(
        self, key: tuple[str, str], *, now_ns: int, tombstone: bool
    ) -> _SequenceState | None:
        state = self._requests.pop(key, None)
        if state is None:
            return None
        if tombstone:
            # Install the fence before deleting heavy state.  If the bounded
            # tombstone table is full, release fails closed and the live state
            # remains available for a later retry.
            self._requests[key] = state
            self._install_tombstone(key, now_ns, state.lease_expiry_ns)
            self._requests.pop(key, None)
        if self._request_leases.get(state.request_id) == state.execution_lease_id:
            self._request_leases.pop(state.request_id, None)
        for replay_key in tuple(state.completed):
            self._completed_lru.pop((key, replay_key), None)
            self._completed_bytes -= state.completed[replay_key].output_tensor.descriptor.payload_bytes
        self._kv_bytes -= state.kv_bytes
        return state

    def issue_lease(self, request_id: str) -> str:
        return derive_execution_lease_id(
            self.plan_generation_id,
            self.manifest.plan_id,
            self.manifest.plan_hash,
            request_id,
        )

    def _admit(
        self, sequence: SequenceSlice, now_ns: int, phase: str
    ) -> tuple[_SequenceState | None, tuple[str, str] | None, bool]:
        key = (sequence.request_id, sequence.execution_lease_id)
        if sequence.execution_lease_id != self.issue_lease(sequence.request_id):
            return None, ("UNKNOWN_LEASE", "lease is not valid for this plan generation"), False
        state = self._requests.get(key)
        if state is not None:
            # Convert the portable remaining budget into this receiver's
            # monotonic clock domain.  A retry may tighten the deadline but can
            # never extend the receiver-local deadline established earlier.
            candidate_expiry_ns = now_ns + sequence.deadline_budget_ns
            state.lease_expiry_ns = min(
                state.lease_expiry_ns, candidate_expiry_ns
            )
            if state.lease_expiry_ns <= now_ns:
                return None, ("DEADLINE", "request deadline expired"), False
            return state, None, False
        if key in self._tombstones:
            return None, ("RELEASED", "request lease has been released or expired"), False
        active_lease = self._request_leases.get(sequence.request_id)
        if active_lease is not None and active_lease != sequence.execution_lease_id:
            return None, ("LEASE_CONFLICT", "request already has a different live lease"), False
        if len(self._requests) >= self.max_live_requests:
            return None, ("ADMISSION", "live request capacity is exhausted"), False
        if len(self._tombstones) >= self.max_tombstones:
            return None, (
                "ADMISSION",
                "plan generation tombstone capacity is exhausted; rotate only after drain",
            ), False
        if (
            phase != "prefill"
            or sequence.request_sequence_no != 0
            or sequence.kv_epoch != 0
            or sequence.token_position_start != 0
        ):
            return None, (
                "SEQUENCE",
                "new lease must begin with prefill sequence 0 at position/KV epoch 0",
            ), False
        state = _SequenceState(
            request_id=sequence.request_id,
            execution_lease_id=sequence.execution_lease_id,
            lease_expiry_ns=now_ns + sequence.deadline_budget_ns,
        )
        self._requests[key] = state
        self._request_leases[sequence.request_id] = sequence.execution_lease_id
        self._high_water_live = max(self._high_water_live, len(self._requests))
        return state, None, True

    @staticmethod
    def _replay_base(
        request: RaggedBatchRequest, sequence: SequenceSlice
    ) -> ReplayBaseKey:
        return _semantic_replay_key(request, sequence)

    def _credit_snapshot_locked(
        self,
        *,
        max_payload_bytes: int,
        max_rows: int = WORKER_MAX_ROWS,
        max_sequences: int = WORKER_MAX_SEQUENCES,
    ) -> dict[str, int]:
        return {
            "frames": 1,
            "payload_bytes": max_payload_bytes,
            "rows": max_rows,
            "sequences": max_sequences,
            "live_requests": max(0, self.max_live_requests - len(self._requests)),
            "kv_bytes": max(0, self.max_kv_bytes - self._kv_bytes),
            "replay_bytes": max(
                0, self.max_completed_bytes - self._completed_bytes
            ),
        }

    def credit_snapshot(
        self,
        *,
        max_payload_bytes: int,
        max_rows: int = WORKER_MAX_ROWS,
        max_sequences: int = WORKER_MAX_SEQUENCES,
    ) -> dict[str, int]:
        """Return receiver-authoritative credits after applying expiry."""

        with self._lock:
            self._sweep(self._clock_ns())
            return self._credit_snapshot_locked(
                max_payload_bytes=max_payload_bytes,
                max_rows=max_rows,
                max_sequences=max_sequences,
            )

    def _credit_demand_locked(
        self, request: RaggedBatchRequest, now_ns: int
    ) -> dict[str, int]:
        """Compute only resources that this batch can newly retain.

        Exact semantic replay consumes wire/row capacity but no new live, KV,
        or replay-retention capacity.  Invalid sequences also consume no
        retained resources; :meth:`execute` returns their typed result.
        """

        demand = {
            "frames": 1,
            "payload_bytes": len(request.input_tensor.payload),
            "rows": request.descriptor.input_row_count,
            "sequences": len(request.descriptor.sequences),
            "live_requests": 0,
            "kv_bytes": 0,
            "replay_bytes": 0,
        }
        output_width = (
            self.manifest.hidden_size
            if self.manifest.output_kind == "activation"
            else self.manifest.vocabulary_size
        )
        assert output_width is not None
        for sequence in request.descriptor.sequences:
            state_key = (sequence.request_id, sequence.execution_lease_id)
            if (
                sequence.execution_lease_id != self.issue_lease(sequence.request_id)
                or state_key in self._tombstones
                or sequence.deadline_budget_ns == 0
            ):
                continue
            state = self._requests.get(state_key)
            replay_key = self._replay_base(request, sequence)
            source = request.input_tensor.rows(
                sequence.input_row_start, sequence.input_row_count
            )
            cached = None if state is None else state.completed.get(replay_key)
            if cached is not None:
                # A conflicting payload is rejected and also retains nothing.
                if cached.input_digest == source.digest:
                    continue
                continue
            if state is None:
                if (
                    request.descriptor.phase != "prefill"
                    or sequence.request_sequence_no != 0
                    or sequence.kv_epoch != 0
                    or sequence.token_position_start != 0
                ):
                    continue
                demand["live_requests"] += 1
            elif (
                state.lease_expiry_ns <= now_ns
                or state.cancelled
                or sequence.kv_epoch != state.kv_epoch
                or sequence.request_sequence_no != state.highest_sequence_no + 1
                or request.descriptor.phase != "decode"
                or sequence.token_position_start != state.next_token_position
            ):
                continue
            demand["kv_bytes"] += (
                sequence.input_row_count
                * self.manifest.hidden_size
                * DTYPE_BYTES[self.manifest.dtype]
            )
            demand["replay_bytes"] += (
                sequence.input_row_count
                * output_width
                * DTYPE_BYTES[self.manifest.dtype]
            )
        return demand

    def execute_with_credit(
        self,
        request: RaggedBatchRequest,
        advertised_credit: dict[str, int],
        *,
        max_payload_bytes: int,
    ) -> RaggedBatchResult:
        """Atomically enforce the receiver's last advertised credit and run."""

        with self._lock:
            now_ns = self._clock_ns()
            self._sweep(now_ns)
            demand = self._credit_demand_locked(request, now_ns)
            current = self._credit_snapshot_locked(
                max_payload_bytes=max_payload_bytes
            )
            for name, value in demand.items():
                available = min(advertised_credit.get(name, 0), current[name])
                if value > available:
                    raise RaggedRuntimeError(
                        "CREDIT",
                        f"receiver rejected batch exceeding {name} credit",
                        retryable=True,
                    )
            # The RLock keeps the check and all mutations in execute atomic.
            return self.execute(request)

    def _remember(
        self,
        state_key: tuple[str, str],
        state: _SequenceState,
        replay_key: ReplayBaseKey,
        cached: _CachedSequence,
    ) -> None:
        state.completed[replay_key] = cached
        state.completed.move_to_end(replay_key)
        lru_key = (state_key, replay_key)
        self._completed_lru[lru_key] = None
        self._completed_bytes += cached.output_tensor.descriptor.payload_bytes
        self._completed_lru.move_to_end(lru_key)
        while (
            len(self._completed_lru) > self.max_completed_results
            or self._completed_bytes > self.max_completed_bytes
        ):
            (old_state_key, old_replay_key), _ = self._completed_lru.popitem(last=False)
            old_state = self._requests.get(old_state_key)
            if old_state is not None:
                old_cached = old_state.completed.pop(old_replay_key, None)
                if old_cached is not None:
                    self._completed_bytes -= old_cached.output_tensor.descriptor.payload_bytes
        self._high_water_completed = max(
            self._high_water_completed, len(self._completed_lru)
        )
        self._high_water_completed_bytes = max(
            self._high_water_completed_bytes, self._completed_bytes
        )

    def _evict_completed(self, lru_key: tuple[tuple[str, str], ReplayBaseKey]) -> None:
        self._completed_lru.pop(lru_key, None)
        state = self._requests.get(lru_key[0])
        if state is None:
            return
        cached = state.completed.pop(lru_key[1], None)
        if cached is not None:
            self._completed_bytes -= cached.output_tensor.descriptor.payload_bytes

    def _reserve_batch_replay(
        self, request: RaggedBatchRequest, now_ns: int
    ) -> None:
        """Atomically reserve retained-result count/bytes before any KV mutation."""

        protected: set[tuple[tuple[str, str], ReplayBaseKey]] = set()
        new_count = 0
        new_bytes = 0
        output_width = (
            self.manifest.hidden_size
            if self.manifest.output_kind == "activation"
            else self.manifest.vocabulary_size
        )
        assert output_width is not None
        for sequence in request.descriptor.sequences:
            state_key = (sequence.request_id, sequence.execution_lease_id)
            if (
                sequence.execution_lease_id != self.issue_lease(sequence.request_id)
                or state_key in self._tombstones
                or sequence.deadline_budget_ns == 0
            ):
                continue
            state = self._requests.get(state_key)
            replay_key = self._replay_base(request, sequence)
            cached = None if state is None else state.completed.get(replay_key)
            if cached is not None:
                protected.add((state_key, replay_key))
                continue
            if state is None:
                if (
                    request.descriptor.phase != "prefill"
                    or sequence.request_sequence_no != 0
                    or sequence.kv_epoch != 0
                    or sequence.token_position_start != 0
                ):
                    continue
            elif (
                state.lease_expiry_ns <= now_ns
                or state.cancelled
                or sequence.kv_epoch != state.kv_epoch
                or sequence.request_sequence_no != state.highest_sequence_no + 1
                or request.descriptor.phase != "decode"
                or sequence.token_position_start != state.next_token_position
            ):
                continue
            new_count += 1
            new_bytes += (
                sequence.input_row_count
                * output_width
                * DTYPE_BYTES[self.manifest.dtype]
            )
        protected_bytes = sum(
            self._requests[state_key]
            .completed[replay_key]
            .output_tensor.descriptor.payload_bytes
            for state_key, replay_key in protected
        )
        if (
            new_count > self.max_completed_results
            or new_bytes + protected_bytes > self.max_completed_bytes
        ):
            raise RaggedRuntimeError(
                "ADMISSION",
                "batch cannot atomically reserve retained replay count/bytes",
                retryable=True,
            )
        while (
            len(self._completed_lru) + new_count > self.max_completed_results
            or self._completed_bytes + new_bytes > self.max_completed_bytes
        ):
            victim = next(
                (key for key in self._completed_lru if key not in protected), None
            )
            if victim is None:
                raise RaggedRuntimeError(
                    "ADMISSION",
                    "retained replay capacity is pinned by this batch",
                    retryable=True,
                )
            self._evict_completed(victim)

    def _transform(self, tensor: LogicalTensor) -> LogicalTensor:
        manifest = self.manifest
        rows = tensor.descriptor.row_count
        if manifest.input_kind == "token_ids":
            token_ids = tuple(int(value) for value in tensor.values())
            values: list[float] = []
            for token_id in token_ids:
                for hidden in range(manifest.hidden_size):
                    raw = (
                        token_id * 17
                        + hidden * 13
                        + manifest.layer_start * 7
                        + manifest.layer_end * 5
                    )
                    values.append(round(((raw % 127) - 63) / 64.0, 7))
            return LogicalTensor.from_values(
                values,
                kind="activation",
                dtype=manifest.dtype,
                shape=(rows, manifest.hidden_size),
            )

        input_width = tensor.descriptor.width
        if input_width != manifest.hidden_size:
            raise RaggedRuntimeError(
                "TENSOR_CONTRACT", "activation width does not match manifest"
            )
        float_values = tuple(float(value) for value in tensor.values())
        if manifest.output_kind == "activation":
            transformed = deterministic_stage_values(
                float_values,
                rows=rows,
                width=manifest.hidden_size,
                layer_start=manifest.layer_start,
                layer_end=manifest.layer_end,
            )
            return LogicalTensor.from_values(
                transformed,
                kind="activation",
                dtype=manifest.dtype,
                shape=(rows, manifest.hidden_size),
            )

        assert manifest.vocabulary_size is not None
        logits: list[float] = []
        for row in range(rows):
            row_values = float_values[
                row * manifest.hidden_size : (row + 1) * manifest.hidden_size
            ]
            for vocab in range(manifest.vocabulary_size):
                acc = ((vocab * 11 + manifest.layer_end * 7) % 37 - 18) / 37.0
                for hidden, value in enumerate(row_values):
                    weight = ((hidden * 13 + vocab * 5 + 3) % 29 - 14) / 29.0
                    acc += value * weight
                logits.append(round(acc, 7))
        return LogicalTensor.from_values(
            logits,
            kind="logits",
            dtype=manifest.dtype,
            shape=(rows, manifest.vocabulary_size),
        )

    @staticmethod
    def _failed(
        sequence: SequenceSlice, status: str, code: str, message: str
    ) -> SequenceResult:
        return SequenceResult(
            request_id=sequence.request_id,
            execution_lease_id=sequence.execution_lease_id,
            request_sequence_no=sequence.request_sequence_no,
            status=status,
            input_row_start=sequence.input_row_start,
            input_row_count=sequence.input_row_count,
            output_row_start=None,
            output_row_count=0,
            kv_epoch_before=sequence.kv_epoch,
            kv_epoch_after=sequence.kv_epoch,
            error={"code": code, "message": message[:256]},
            timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
        )

    def execute(self, request: RaggedBatchRequest) -> RaggedBatchResult:
        with self._lock:
            now_ns = self._clock_ns()
            self._sweep(now_ns)
            manifest = self.manifest
            if (
                request.plan_id != manifest.plan_id
                or request.plan_hash != manifest.plan_hash
                or request.manifest_hash != manifest.manifest_hash
            ):
                raise RaggedRuntimeError("STALE_PLAN", "batch does not match loaded manifest")
            tensor = request.input_tensor
            if tensor.descriptor.kind != manifest.input_kind:
                raise RaggedRuntimeError("TENSOR_CONTRACT", "input kind does not match manifest")
            if manifest.input_kind == "token_ids":
                if tensor.descriptor.dtype != "int32":
                    raise RaggedRuntimeError(
                        "TENSOR_CONTRACT", "token input dtype must be int32"
                    )
            elif (
                tensor.descriptor.width != manifest.hidden_size
                or tensor.descriptor.dtype != manifest.dtype
            ):
                raise RaggedRuntimeError(
                    "TENSOR_CONTRACT",
                    "activation width/dtype does not match manifest",
                )
            self._reserve_batch_replay(request, now_ns)

            outputs: list[LogicalTensor] = []
            results: list[SequenceResult] = []
            output_cursor = 0
            for sequence in sorted(
                request.descriptor.sequences,
                key=lambda item: (item.request_id, item.execution_lease_id),
            ):
                state_key = (sequence.request_id, sequence.execution_lease_id)
                if sequence.execution_lease_id != self.issue_lease(sequence.request_id):
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "UNKNOWN_LEASE",
                            "lease is not valid for this plan generation",
                        )
                    )
                    continue
                if state_key in self._tombstones:
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "RELEASED",
                            "request lease has been released or expired",
                        )
                    )
                    continue
                if sequence.deadline_budget_ns == 0:
                    results.append(
                        self._failed(
                            sequence,
                            "deadline",
                            "DEADLINE",
                            "sequence deadline expired before execution",
                        )
                    )
                    continue
                state, admission_error, newly_admitted = self._admit(
                    sequence, now_ns, request.descriptor.phase
                )
                if state is None:
                    assert admission_error is not None
                    results.append(
                        self._failed(
                            sequence,
                            "deadline"
                            if admission_error[0] == "DEADLINE"
                            else "rejected",
                            *admission_error,
                        )
                    )
                    continue
                source = tensor.rows(sequence.input_row_start, sequence.input_row_count)
                replay_key = self._replay_base(request, sequence)
                cached = state.completed.get(replay_key)
                if cached is not None:
                    if cached.input_digest != source.digest:
                        results.append(
                            self._failed(
                                sequence,
                                "rejected",
                                "SEQUENCE_CONFLICT",
                                "semantic replay identity has a different tensor digest",
                            )
                        )
                        continue
                    output = cached.output_tensor
                    outputs.append(output)
                    results.append(
                        SequenceResult(
                            request_id=sequence.request_id,
                            execution_lease_id=sequence.execution_lease_id,
                            request_sequence_no=sequence.request_sequence_no,
                            status="ok",
                            input_row_start=sequence.input_row_start,
                            input_row_count=sequence.input_row_count,
                            output_row_start=output_cursor,
                            output_row_count=output.descriptor.row_count,
                            kv_epoch_before=cached.kv_epoch_before,
                            kv_epoch_after=cached.kv_epoch_after,
                            error=None,
                            timings_ns={
                                "queue": 0,
                                "execute": 0,
                                "pack": 0,
                                "unpack": 0,
                            },
                        )
                    )
                    output_cursor += output.descriptor.row_count
                    self._completed_lru.move_to_end((state_key, replay_key))
                    continue

                # Exact committed replays win over later cancellation.  The
                # cancellation fence applies only to new work.
                if state.cancelled:
                    results.append(
                        self._failed(
                            sequence, "cancelled", "CANCELLED", "request lease is cancelled"
                        )
                    )
                    continue

                if sequence.kv_epoch != state.kv_epoch:
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "STALE_KV",
                            f"expected KV epoch {state.kv_epoch}",
                        )
                    )
                    continue

                expected_sequence = state.highest_sequence_no + 1
                if sequence.request_sequence_no != expected_sequence:
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "SEQUENCE",
                            f"expected request sequence {expected_sequence}",
                        )
                    )
                    continue
                if request.descriptor.phase == "prefill" and not newly_admitted:
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "PHASE",
                            "prefill may commit only once for a lease",
                        )
                    )
                    continue
                if sequence.token_position_start != state.next_token_position:
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "POSITION",
                            f"expected token position {state.next_token_position}",
                        )
                    )
                    continue

                kv_increment = (
                    sequence.input_row_count
                    * manifest.hidden_size
                    * DTYPE_BYTES[manifest.dtype]
                )
                if self._kv_bytes + kv_increment > self.max_kv_bytes:
                    if newly_admitted:
                        self._drop_state(state_key, now_ns=now_ns, tombstone=False)
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "ADMISSION",
                            "logical KV byte capacity is exhausted",
                        )
                    )
                    continue

                output_width = (
                    manifest.hidden_size
                    if manifest.output_kind == "activation"
                    else manifest.vocabulary_size
                )
                assert output_width is not None
                replay_bytes = (
                    sequence.input_row_count
                    * output_width
                    * DTYPE_BYTES[manifest.dtype]
                )
                if replay_bytes > self.max_completed_bytes:
                    if newly_admitted:
                        self._drop_state(state_key, now_ns=now_ns, tombstone=False)
                    results.append(
                        self._failed(
                            sequence,
                            "rejected",
                            "ADMISSION",
                            "single replay output exceeds retained-byte capacity",
                        )
                    )
                    continue

                state.inflight += 1
                try:
                    output = self._transform(source)
                finally:
                    state.inflight -= 1
                before = state.kv_epoch
                after = before + 1
                state.kv_epoch = after
                state.highest_sequence_no = sequence.request_sequence_no
                state.next_token_position = (
                    sequence.token_position_start + sequence.input_row_count
                )
                state.kv_bytes += kv_increment
                self._kv_bytes += kv_increment
                self._high_water_kv_bytes = max(
                    self._high_water_kv_bytes, self._kv_bytes
                )
                cached = _CachedSequence(
                    input_digest=source.digest,
                    output_tensor=output,
                    kv_epoch_before=before,
                    kv_epoch_after=after,
                )
                self._remember(state_key, state, replay_key, cached)
                outputs.append(output)
                results.append(
                    SequenceResult(
                        request_id=sequence.request_id,
                        execution_lease_id=sequence.execution_lease_id,
                        request_sequence_no=sequence.request_sequence_no,
                        status="ok",
                        input_row_start=sequence.input_row_start,
                        input_row_count=sequence.input_row_count,
                        output_row_start=output_cursor,
                        output_row_count=output.descriptor.row_count,
                        kv_epoch_before=before,
                        kv_epoch_after=after,
                        error=None,
                        timings_ns={"queue": 0, "execute": 0, "pack": 0, "unpack": 0},
                    )
                )
                output_cursor += output.descriptor.row_count

            output_tensor: LogicalTensor | None = None
            if outputs:
                first = outputs[0]
                width = first.descriptor.width
                values: list[int | float] = []
                for output in outputs:
                    if (
                        output.descriptor.kind != first.descriptor.kind
                        or output.descriptor.dtype != first.descriptor.dtype
                        or output.descriptor.width != width
                    ):
                        raise AssertionError("reference backend produced incompatible outputs")
                    values.extend(output.values())
                assert width is not None
                output_tensor = LogicalTensor.from_values(
                    values,
                    kind=first.descriptor.kind,
                    dtype=first.descriptor.dtype,
                    shape=(output_cursor, width),
                )
            return RaggedBatchResult(
                batch_id=request.descriptor.batch_id,
                batch_sequence_no=request.descriptor.batch_sequence_no,
                manifest_hash=manifest.manifest_hash,
                results=tuple(results),
                output_tensor=output_tensor,
            )

    def cancel(self, request_id: str, execution_lease_id: str) -> dict[str, Any]:
        with self._lock:
            if execution_lease_id != self.issue_lease(request_id):
                raise RaggedRuntimeError(
                    "UNKNOWN_LEASE", "lease is not valid for this plan generation"
                )
            self._sweep(self._clock_ns())
            state = self._requests.get((request_id, execution_lease_id))
            if state is None:
                return {"cancelled": False, "inflight": False, "kv_mutated": False}
            state.cancelled = True
            return {
                "cancelled": True,
                "inflight": state.inflight > 0,
                "kv_mutated": state.kv_epoch > 0,
            }

    def release(
        self,
        request_id: str,
        execution_lease_id: str,
        *,
        expected_final_epoch: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if execution_lease_id != self.issue_lease(request_id):
                raise RaggedRuntimeError(
                    "UNKNOWN_LEASE", "lease is not valid for this plan generation"
                )
            now_ns = self._clock_ns()
            self._sweep(now_ns)
            key = (request_id, execution_lease_id)
            state = self._requests.get(key)
            if state is None:
                return {
                    "released": False,
                    "request_id": request_id,
                    "execution_lease_id": execution_lease_id,
                    "kv_epoch": None,
                    "completed_results_released": 0,
                    "tombstone_present": key in self._tombstones,
                }
            if state.inflight:
                raise RaggedRuntimeError(
                    "REQUEST_INFLIGHT",
                    "cannot release a request with inflight execution",
                    retryable=True,
                )
            if expected_final_epoch is not None and expected_final_epoch != state.kv_epoch:
                raise RaggedRuntimeError("STALE_KV", "release expected final epoch mismatch")
            completed = len(state.completed)
            epoch = state.kv_epoch
            self._drop_state(key, now_ns=now_ns, tombstone=True)
            return {
                "released": True,
                "request_id": request_id,
                "execution_lease_id": execution_lease_id,
                "kv_epoch": epoch,
                "completed_results_released": completed,
                "tombstone_present": True,
            }

    def health(self) -> RaggedBackendHealth:
        with self._lock:
            self._sweep(self._clock_ns())
            return RaggedBackendHealth(
                state="READY",
                stage_id=self.manifest.stage_id,
                live_requests=len(self._requests),
                completed_results=len(self._completed_lru),
                completed_bytes=self._completed_bytes,
                tombstones=len(self._tombstones),
                kv_bytes=self._kv_bytes,
                max_live_requests=self.max_live_requests,
                max_completed_results=self.max_completed_results,
                max_completed_bytes=self.max_completed_bytes,
                max_tombstones=self.max_tombstones,
                max_kv_bytes=self.max_kv_bytes,
                high_water_live_requests=self._high_water_live,
                high_water_completed_results=self._high_water_completed,
                high_water_completed_bytes=self._high_water_completed_bytes,
                high_water_tombstones=self._high_water_tombstones,
                high_water_kv_bytes=self._high_water_kv_bytes,
            )


def _control_frame(
    kind: RaggedMessageKind, sequence_no: int, **metadata: Any
) -> RaggedFrame:
    value = {"sequence_no": sequence_no}
    value.update(metadata)
    return RaggedFrame(kind=kind, sequence_no=sequence_no, metadata=value)


def _validate_control_binding(
    frame: RaggedFrame,
    manifest: RaggedStageManifest,
    plan_generation_id: str,
) -> None:
    expected = {
        "plan_id": manifest.plan_id,
        "plan_hash": manifest.plan_hash,
        "manifest_hash": manifest.manifest_hash,
        "stage_id": manifest.stage_id,
        "plan_generation_id": plan_generation_id,
    }
    if any(frame.metadata.get(name) != value for name, value in expected.items()):
        raise RaggedRuntimeError(
            "STALE_PLAN", "control frame does not match the loaded plan/stage"
        )


def _ragged_worker_main(
    manifest_data: dict[str, Any],
    ready: Connection,
    host: str,
    max_payload_bytes: int,
    plan_generation_id: str,
    expected_source_stage: str,
) -> None:
    listener: socket.socket | None = None
    try:
        manifest = RaggedStageManifest.from_dict(manifest_data)
        backend = RaggedReferenceStageBackend(
            manifest, plan_generation_id=plan_generation_id
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        listener.listen(1)
        ready.send(
            {
                "ok": True,
                "host": host,
                "port": listener.getsockname()[1],
                "pid": multiprocessing.current_process().pid,
                "stage_id": manifest.stage_id,
                "manifest_hash": manifest.manifest_hash,
                "plan_generation_id": plan_generation_id,
            }
        )
        connection, _ = listener.accept()
        with connection:
            outgoing = 0
            incoming = 0
            advertised_credit: dict[str, int] = {}

            def send_credit() -> None:
                nonlocal outgoing, advertised_credit
                advertised_credit = backend.credit_snapshot(
                    max_payload_bytes=max_payload_bytes
                )
                send_ragged_frame(
                    connection,
                    _control_frame(
                        RaggedMessageKind.CREDIT,
                        outgoing,
                        **advertised_credit,
                    ),
                    max_payload_bytes=max_payload_bytes,
                )
                outgoing += 1

            hello_id = str(uuid.uuid4())
            send_ragged_frame(
                connection,
                _control_frame(
                    RaggedMessageKind.CONTROL,
                    outgoing,
                    control_id=hello_id,
                    control="hello",
                    plan_id=manifest.plan_id,
                    plan_hash=manifest.plan_hash,
                    manifest_hash=manifest.manifest_hash,
                    stage_id=manifest.stage_id,
                    plan_generation_id=plan_generation_id,
                    supported_versions=[[2, 0]],
                ),
                max_payload_bytes=max_payload_bytes,
            )
            outgoing += 1
            hello_ack = read_ragged_frame(
                connection, max_payload_bytes=max_payload_bytes
            )
            if hello_ack.sequence_no != incoming:
                raise RaggedFrameError("SEQUENCE", "invalid hello ACK sequence")
            incoming += 1
            if (
                hello_ack.kind != RaggedMessageKind.ACK
                or hello_ack.metadata.get("operation") != "hello"
                or hello_ack.metadata.get("control_id") != hello_id
            ):
                raise RaggedFrameError("METADATA", "invalid hello ACK correlation")
            _validate_control_binding(hello_ack, manifest, plan_generation_id)
            send_credit()
            while True:
                try:
                    frame = read_ragged_frame(
                        connection, max_payload_bytes=max_payload_bytes
                    )
                except RaggedFrameError as exc:
                    if not exc.recoverable or exc.sequence_no != incoming:
                        raise
                    incoming += 1
                    send_ragged_frame(
                        connection,
                        _control_frame(
                            RaggedMessageKind.ERROR,
                            outgoing,
                            code=exc.code,
                            message=exc.message[:256],
                            control_id=None,
                        ),
                        max_payload_bytes=max_payload_bytes,
                    )
                    outgoing += 1
                    continue
                if frame.sequence_no != incoming:
                    raise RaggedFrameError(
                        "SEQUENCE", f"expected channel sequence {incoming}"
                    )
                incoming += 1
                try:
                    if frame.kind == RaggedMessageKind.BATCH:
                        if frame.metadata.get("destination_stage") != manifest.stage_id:
                            raise RaggedRuntimeError(
                                "STALE_PLAN", "batch destination does not match worker"
                            )
                        if frame.metadata.get("source_stage") != expected_source_stage:
                            raise RaggedRuntimeError(
                                "STALE_PLAN",
                                "batch source does not match the configured upstream stage",
                            )
                        request = frame.batch_request()
                        result = backend.execute_with_credit(
                            request,
                            advertised_credit,
                            max_payload_bytes=max_payload_bytes,
                        )
                        send_ragged_frame(
                            connection,
                            RaggedFrame.from_result(result, sequence_no=outgoing),
                            max_payload_bytes=max_payload_bytes,
                        )
                        outgoing += 1
                        send_credit()
                        continue
                    if frame.kind == RaggedMessageKind.CANCEL:
                        _validate_control_binding(
                            frame, manifest, plan_generation_id
                        )
                        result = backend.cancel(
                            frame.metadata["request_id"],
                            frame.metadata["execution_lease_id"],
                        )
                        send_ragged_frame(
                            connection,
                            _control_frame(
                                RaggedMessageKind.ACK,
                                outgoing,
                                control_id=frame.metadata["control_id"],
                                operation="cancel",
                                request_id=frame.metadata["request_id"],
                                execution_lease_id=frame.metadata[
                                    "execution_lease_id"
                                ],
                                **result,
                            ),
                            max_payload_bytes=max_payload_bytes,
                        )
                        outgoing += 1
                        continue
                    if frame.kind == RaggedMessageKind.RELEASE:
                        _validate_control_binding(
                            frame, manifest, plan_generation_id
                        )
                        result = backend.release(
                            frame.metadata["request_id"],
                            frame.metadata["execution_lease_id"],
                            expected_final_epoch=frame.metadata[
                                "expected_final_epoch"
                            ],
                        )
                        send_ragged_frame(
                            connection,
                            _control_frame(
                                RaggedMessageKind.ACK,
                                outgoing,
                                control_id=frame.metadata["control_id"],
                                operation="release",
                                **result,
                            ),
                            max_payload_bytes=max_payload_bytes,
                        )
                        outgoing += 1
                        send_credit()
                        continue
                    if frame.kind == RaggedMessageKind.CONTROL:
                        _validate_control_binding(
                            frame, manifest, plan_generation_id
                        )
                        operation = frame.metadata.get("control")
                        if operation == "credit_refresh":
                            send_ragged_frame(
                                connection,
                                _control_frame(
                                    RaggedMessageKind.ACK,
                                    outgoing,
                                    control_id=frame.metadata["control_id"],
                                    operation="credit_refresh",
                                    stage_id=manifest.stage_id,
                                ),
                                max_payload_bytes=max_payload_bytes,
                            )
                            outgoing += 1
                            send_credit()
                            continue
                        if operation != "shutdown":
                            raise RaggedRuntimeError(
                                "ABI_VERSION", "unsupported control operation"
                            )
                        send_ragged_frame(
                            connection,
                            _control_frame(
                                RaggedMessageKind.ACK,
                                outgoing,
                                control_id=frame.metadata["control_id"],
                                operation="shutdown",
                                stage_id=manifest.stage_id,
                            ),
                            max_payload_bytes=max_payload_bytes,
                        )
                        outgoing += 1
                        return
                    raise RaggedRuntimeError("ABI_VERSION", "unexpected frame kind")
                except (RaggedRuntimeError, RaggedFrameError, ValueError) as exc:
                    send_ragged_frame(
                        connection,
                        _control_frame(
                            RaggedMessageKind.ERROR,
                            outgoing,
                            code=getattr(exc, "code", "BATCH_CONTRACT"),
                            message=str(exc)[:256],
                            control_id=frame.metadata.get("control_id"),
                        ),
                        max_payload_bytes=max_payload_bytes,
                    )
                    outgoing += 1
    except BaseException as exc:  # noqa: BLE001 - child reports startup failure.
        try:
            ready.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        ready.close()
        if listener is not None:
            listener.close()


class RaggedWorkerProcess:
    def __init__(
        self,
        manifest: RaggedStageManifest,
        *,
        host: str = "127.0.0.1",
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        plan_generation_id: str | None = None,
        expected_source_stage: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.host = host
        self.max_payload_bytes = max_payload_bytes
        self.plan_generation_id = plan_generation_id or str(uuid.uuid4())
        if str(uuid.UUID(self.plan_generation_id)) != self.plan_generation_id:
            raise ValueError("plan_generation_id must be a canonical UUID")
        if expected_source_stage is None:
            if manifest.stage_index != 0:
                raise ValueError(
                    "a non-first FNX2 worker requires its exact upstream stage ID"
                )
            expected_source_stage = "gateway"
        if not isinstance(expected_source_stage, str) or not expected_source_stage:
            raise ValueError("expected_source_stage must be a non-empty string")
        self.expected_source_stage = expected_source_stage
        self._process: multiprocessing.Process | None = None
        self._parent: Connection | None = None

    def start(self, timeout_s: float = 10.0) -> dict[str, Any]:
        if self._process is not None:
            raise RaggedRuntimeError("WORKER_START", "worker already started")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_ragged_worker_main,
            args=(
                self.manifest.to_dict(),
                child,
                self.host,
                self.max_payload_bytes,
                self.plan_generation_id,
                self.expected_source_stage,
            ),
            name=f"fornax-fnx2-{self.manifest.stage_id}",
        )
        process.start()
        child.close()
        self._process = process
        self._parent = parent
        if not parent.poll(timeout_s):
            process.terminate()
            process.join(timeout=2.0)
            raise RaggedRuntimeError("WORKER_START", "worker startup timed out")
        value = parent.recv()
        if not value.get("ok"):
            process.join(timeout=2.0)
            raise RaggedRuntimeError(
                "WORKER_START", str(value.get("error", "worker startup failed"))
            )
        return value

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def join(self, timeout_s: float = 5.0) -> None:
        if self._parent is not None:
            self._parent.close()
            self._parent = None
        if self._process is not None:
            self._process.join(timeout=timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None


class RaggedWorkerChannel:
    def __init__(
        self,
        manifest: RaggedStageManifest,
        host: str,
        port: int,
        *,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        plan_generation_id: str | None = None,
        max_known_replays: int = 8192,
    ) -> None:
        self.manifest = manifest
        self.host = host
        self.port = port
        self.max_payload_bytes = max_payload_bytes
        self.plan_generation_id = plan_generation_id or str(uuid.uuid4())
        if str(uuid.UUID(self.plan_generation_id)) != self.plan_generation_id:
            raise ValueError("plan_generation_id must be a canonical UUID")
        if (
            isinstance(max_known_replays, bool)
            or not isinstance(max_known_replays, int)
            or max_known_replays <= 0
        ):
            raise ValueError("max_known_replays must be a positive integer")
        self.max_known_replays = max_known_replays
        self._socket: socket.socket | None = None
        self._outgoing = 0
        self._incoming = 0
        self._credits: dict[str, int] = {
            "frames": 0,
            "payload_bytes": 0,
            "rows": 0,
            "sequences": 0,
            "live_requests": 0,
            "kv_bytes": 0,
            "replay_bytes": 0,
        }
        # Credits advertise *new* request slots.  A decode for an identity
        # already admitted on this channel must not consume another slot.
        # This cache is only an optimization: the worker remains the authority
        # and independently fails closed on admission, expiry, and lease rules.
        self._known_requests: set[tuple[str, str]] = set()
        self._known_replays: OrderedDict[tuple[ReplayBaseKey, str], None] = (
            OrderedDict()
        )
        self._quarantined_reason: str | None = None

    def issue_lease(self, request_id: str) -> str:
        return derive_execution_lease_id(
            self.plan_generation_id,
            self.manifest.plan_id,
            self.manifest.plan_hash,
            request_id,
        )

    def connect(self, timeout_s: float = 5.0) -> None:
        if self._quarantined_reason is not None:
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", self._quarantined_reason
            )
        if self._socket is not None:
            raise RaggedRuntimeError("TRANSPORT", "channel is already connected")
        channel = socket.create_connection((self.host, self.port), timeout=timeout_s)
        channel.settimeout(timeout_s)
        self._socket = channel
        try:
            self._perform_handshake()
        except (OSError, RaggedFrameError, RaggedRuntimeError, ValueError) as exc:
            self._quarantine(exc)
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", f"FNX2 hello failed: {exc}"
            ) from exc

    def _perform_handshake(self) -> None:
        try:
            hello = self._read()
            expected = {
                "plan_id": self.manifest.plan_id,
                "plan_hash": self.manifest.plan_hash,
                "manifest_hash": self.manifest.manifest_hash,
                "stage_id": self.manifest.stage_id,
                "plan_generation_id": self.plan_generation_id,
            }
            if (
                hello.kind != RaggedMessageKind.CONTROL
                or hello.metadata.get("control") != "hello"
                or hello.metadata.get("supported_versions") != [[2, 0]]
                or any(
                    hello.metadata.get(name) != value
                    for name, value in expected.items()
                )
            ):
                raise RaggedRuntimeError(
                    "STALE_PLAN",
                    "worker hello does not match exact plan/stage/generation",
                )
            self._send(
                _control_frame(
                    RaggedMessageKind.ACK,
                    self._outgoing,
                    control_id=hello.metadata["control_id"],
                    operation="hello",
                    **expected,
                    selected_version=[2, 0],
                )
            )
            self._consume_credit(self._read())
        except (OSError, RaggedFrameError, RaggedRuntimeError, ValueError) as exc:
            self._quarantine(exc)
            raise

    def _require_socket(self) -> socket.socket:
        if self._quarantined_reason is not None:
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", self._quarantined_reason
            )
        if self._socket is None:
            raise RaggedRuntimeError("TRANSPORT", "channel is not connected")
        return self._socket

    def _quarantine(self, reason: BaseException | str) -> None:
        self._quarantined_reason = str(reason)[:256]
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def _read(self) -> RaggedFrame:
        frame = read_ragged_frame(
            self._require_socket(), max_payload_bytes=self.max_payload_bytes
        )
        if frame.sequence_no != self._incoming:
            raise RaggedRuntimeError(
                "SEQUENCE", f"expected worker sequence {self._incoming}"
            )
        self._incoming += 1
        return frame

    def _send(self, frame: RaggedFrame) -> None:
        if frame.sequence_no != self._outgoing:
            raise AssertionError("outgoing sequence mismatch")
        send_ragged_frame(
            self._require_socket(), frame, max_payload_bytes=self.max_payload_bytes
        )
        self._outgoing += 1

    def _consume_credit(self, frame: RaggedFrame) -> None:
        if frame.kind != RaggedMessageKind.CREDIT:
            raise RaggedRuntimeError("CREDIT", "expected bounded credit frame")
        for name in self._credits:
            self._credits[name] = frame.metadata[name]

    def execute(
        self, request: RaggedBatchRequest, *, source_stage: str
    ) -> RaggedBatchResult:
        identities = {
            (item.request_id, item.execution_lease_id)
            for item in request.descriptor.sequences
        }
        exact_replays: set[tuple[str, str]] = set()
        new_work_rows = 0
        replay_identities: dict[tuple[str, str], tuple[ReplayBaseKey, str]] = {}
        for item in request.descriptor.sequences:
            identity = (item.request_id, item.execution_lease_id)
            replay_identity = (
                _semantic_replay_key(request, item),
                request.input_tensor.rows(
                    item.input_row_start, item.input_row_count
                ).digest,
            )
            replay_identities[identity] = replay_identity
            if replay_identity in self._known_replays:
                exact_replays.add(identity)
            else:
                new_work_rows += item.input_row_count
        needed = {
            "frames": 1,
            "payload_bytes": len(request.input_tensor.payload),
            "rows": request.descriptor.input_row_count,
            "sequences": len(request.descriptor.sequences),
            "live_requests": len(
                identities - self._known_requests - exact_replays
            ),
            "kv_bytes": (
                new_work_rows
                * self.manifest.hidden_size
                * DTYPE_BYTES[self.manifest.dtype]
            ),
            "replay_bytes": (
                new_work_rows
                * (
                    self.manifest.hidden_size
                    if self.manifest.output_kind == "activation"
                    else int(self.manifest.vocabulary_size or 0)
                )
                * DTYPE_BYTES[self.manifest.dtype]
            ),
        }
        for name, value in needed.items():
            if value > self._credits[name]:
                raise RaggedRuntimeError(
                    "CREDIT", f"batch exceeds {name} credit", retryable=True
                )
        try:
            self._send(
                RaggedFrame.from_batch(
                    request,
                    sequence_no=self._outgoing,
                    source_stage=source_stage,
                    destination_stage=self.manifest.stage_id,
                )
            )
            response = self._read()
        except (OSError, RaggedFrameError, RaggedRuntimeError, ValueError) as exc:
            self._quarantine(exc)
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", f"result read failed: {exc}"
            ) from exc
        if response.kind == RaggedMessageKind.ERROR:
            raise RaggedRuntimeError(
                response.metadata["code"],
                response.metadata["message"],
            )
        if response.kind != RaggedMessageKind.RESULT:
            self._quarantine("expected ragged result")
            raise RaggedRuntimeError("CHANNEL_QUARANTINED", "expected ragged result")
        try:
            result = response.batch_result()
            result.validate_for(request)
        except (ValueError, RaggedFrameError) as exc:
            self._quarantine(exc)
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", f"uncorrelated worker result: {exc}"
            ) from exc
        if result.manifest_hash != self.manifest.manifest_hash:
            self._quarantine("worker result manifest mismatch")
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", "worker result manifest mismatch"
            )
        if result.output_tensor is not None:
            expected_width = (
                self.manifest.hidden_size
                if self.manifest.output_kind == "activation"
                else self.manifest.vocabulary_size
            )
            if (
                result.output_tensor.descriptor.kind != self.manifest.output_kind
                or result.output_tensor.descriptor.dtype != self.manifest.dtype
                or result.output_tensor.descriptor.width != expected_width
            ):
                self._quarantine("worker result tensor contract mismatch")
                raise RaggedRuntimeError(
                    "CHANNEL_QUARANTINED", "worker result tensor contract mismatch"
                )
        # These outcomes prove that the backend found live state for the
        # identity.  Admission/deadline/lease/released failures deliberately do
        # not populate the cache.
        stateful_error_codes = {"SEQUENCE", "SEQUENCE_CONFLICT", "STALE_KV"}
        for item in result.results:
            error_code = None if item.error is None else item.error.get("code")
            if (
                item.status in {"ok", "cancelled"}
                or error_code in stateful_error_codes
            ):
                self._known_requests.add(
                    (item.request_id, item.execution_lease_id)
                )
            if item.status == "ok":
                replay_identity = replay_identities[
                    (item.request_id, item.execution_lease_id)
                ]
                self._known_replays[replay_identity] = None
                self._known_replays.move_to_end(replay_identity)
                while len(self._known_replays) > self.max_known_replays:
                    self._known_replays.popitem(last=False)
        try:
            self._consume_credit(self._read())
        except (OSError, RaggedFrameError, RaggedRuntimeError, ValueError) as exc:
            self._quarantine(exc)
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", f"post-result credit failed: {exc}"
            ) from exc
        return result

    def _request_control(
        self,
        kind: RaggedMessageKind,
        request_id: str,
        execution_lease_id: str,
        **metadata: Any,
    ) -> dict[str, Any]:
        control_id = str(uuid.uuid4())
        try:
            self._send(
                _control_frame(
                    kind,
                    self._outgoing,
                    control_id=control_id,
                    plan_id=self.manifest.plan_id,
                    plan_hash=self.manifest.plan_hash,
                    manifest_hash=self.manifest.manifest_hash,
                    stage_id=self.manifest.stage_id,
                    plan_generation_id=self.plan_generation_id,
                    request_id=request_id,
                    execution_lease_id=execution_lease_id,
                    **metadata,
                )
            )
            response = self._read()
        except (OSError, RaggedFrameError, RaggedRuntimeError, ValueError) as exc:
            self._quarantine(exc)
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", f"control response failed: {exc}"
            ) from exc
        if response.kind == RaggedMessageKind.ERROR:
            if response.metadata.get("control_id") != control_id:
                self._quarantine("uncorrelated control error")
                raise RaggedRuntimeError(
                    "CHANNEL_QUARANTINED", "uncorrelated control error"
                )
            raise RaggedRuntimeError(
                response.metadata["code"],
                response.metadata["message"],
            )
        if response.kind != RaggedMessageKind.ACK:
            self._quarantine("expected control acknowledgement")
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", "expected control acknowledgement"
            )
        expected_operation = "cancel" if kind == RaggedMessageKind.CANCEL else "release"
        if (
            response.metadata.get("control_id") != control_id
            or response.metadata.get("operation") != expected_operation
            or response.metadata.get("request_id") != request_id
            or response.metadata.get("execution_lease_id") != execution_lease_id
        ):
            self._quarantine("uncorrelated control acknowledgement")
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", "uncorrelated control acknowledgement"
            )
        return dict(response.metadata)

    def cancel(self, request_id: str, execution_lease_id: str) -> dict[str, Any]:
        return self._request_control(
            RaggedMessageKind.CANCEL, request_id, execution_lease_id
        )

    def release(
        self,
        request_id: str,
        execution_lease_id: str,
        *,
        expected_final_epoch: int | None = None,
    ) -> dict[str, Any]:
        result = self._request_control(
            RaggedMessageKind.RELEASE,
            request_id,
            execution_lease_id,
            expected_final_epoch=expected_final_epoch,
        )
        try:
            self._consume_credit(self._read())
        except (OSError, RaggedFrameError, RaggedRuntimeError, ValueError) as exc:
            self._quarantine(exc)
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", f"post-release credit failed: {exc}"
            ) from exc
        self._known_requests.discard((request_id, execution_lease_id))
        for replay_identity in tuple(self._known_replays):
            replay_key, _ = replay_identity
            if replay_key[2:4] == (request_id, execution_lease_id):
                self._known_replays.pop(replay_identity, None)
        return result

    def refresh_credits(self) -> dict[str, int]:
        """Ask the receiver to sweep expiry and publish a fresh credit vector."""

        control_id = str(uuid.uuid4())
        try:
            self._send(
                _control_frame(
                    RaggedMessageKind.CONTROL,
                    self._outgoing,
                    control_id=control_id,
                    control="credit_refresh",
                    plan_id=self.manifest.plan_id,
                    plan_hash=self.manifest.plan_hash,
                    manifest_hash=self.manifest.manifest_hash,
                    stage_id=self.manifest.stage_id,
                    plan_generation_id=self.plan_generation_id,
                )
            )
            response = self._read()
            if (
                response.kind != RaggedMessageKind.ACK
                or response.metadata.get("control_id") != control_id
                or response.metadata.get("operation") != "credit_refresh"
                or response.metadata.get("stage_id") != self.manifest.stage_id
            ):
                raise RaggedRuntimeError(
                    "TRANSPORT", "credit refresh was not acknowledged"
                )
            self._consume_credit(self._read())
            return dict(self._credits)
        except (OSError, RaggedFrameError, RaggedRuntimeError, ValueError) as exc:
            self._quarantine(exc)
            if isinstance(exc, RaggedRuntimeError):
                raise
            raise RaggedRuntimeError(
                "CHANNEL_QUARANTINED", f"credit refresh failed: {exc}"
            ) from exc

    def shutdown(self) -> None:
        if self._socket is None:
            return
        try:
            control_id = str(uuid.uuid4())
            self._send(
                _control_frame(
                    RaggedMessageKind.CONTROL,
                    self._outgoing,
                    control_id=control_id,
                    control="shutdown",
                    plan_id=self.manifest.plan_id,
                    plan_hash=self.manifest.plan_hash,
                    manifest_hash=self.manifest.manifest_hash,
                    stage_id=self.manifest.stage_id,
                    plan_generation_id=self.plan_generation_id,
                )
            )
            response = self._read()
            if (
                response.kind != RaggedMessageKind.ACK
                or response.metadata.get("control_id") != control_id
                or response.metadata.get("operation") != "shutdown"
            ):
                raise RaggedRuntimeError("TRANSPORT", "shutdown was not acknowledged")
        finally:
            self.disconnect()

    def disconnect(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    @property
    def quarantined(self) -> bool:
        return self._quarantined_reason is not None


class RaggedOrchestrator:
    def __init__(
        self,
        stages: list[tuple[RaggedStageManifest, RaggedWorkerChannel]],
        *,
        max_event_entries: int = 16_384,
        max_cleanup_pending: int = 4096,
    ) -> None:
        if len(stages) < 2:
            raise ValueError("ragged orchestrator requires at least two stages")
        manifests = [item[0] for item in stages]
        if [item.stage_index for item in manifests] != list(range(len(manifests))):
            raise ValueError("ragged stages must be contiguous and ordered")
        if manifests[0].stage_role != "first" or manifests[-1].stage_role != "final":
            raise ValueError("ragged route requires first and final stage roles")
        if manifests[0].layer_start != 0 or any(
            right.layer_start != left.layer_end + 1
            for left, right in zip(manifests, manifests[1:])
        ):
            raise ValueError("ragged stage layer ranges must be contiguous")
        for left, right in zip(manifests, manifests[1:]):
            if left.output_kind != right.input_kind:
                raise ValueError("adjacent ragged tensor kinds must match")
            if left.hidden_size != right.hidden_size or left.dtype != right.dtype:
                raise ValueError("adjacent ragged tensor contracts must match")
        first = manifests[0]
        if any(
            item.plan_id != first.plan_id or item.plan_hash != first.plan_hash
            for item in manifests[1:]
        ):
            raise ValueError("ragged stages must identify one plan")
        for manifest, channel in stages:
            if channel.manifest.manifest_hash != manifest.manifest_hash:
                raise ValueError("ragged channel manifest mismatch")
        generations = {channel.plan_generation_id for _, channel in stages}
        if len(generations) != 1:
            raise ValueError("ragged stages must share one worker-route generation")
        if max_event_entries <= 0:
            raise ValueError("max_event_entries must be positive")
        if (
            isinstance(max_cleanup_pending, bool)
            or not isinstance(max_cleanup_pending, int)
            or max_cleanup_pending <= 0
        ):
            raise ValueError("max_cleanup_pending must be positive")
        self.stages = stages
        self.plan_generation_id = next(iter(generations))
        self.max_cleanup_pending = max_cleanup_pending
        self._kv_epochs: dict[tuple[str, str, str], int] = {}
        self._cleanup_pending: dict[
            tuple[str, str], set[str]
        ] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=max_event_entries)

    def issue_lease(self, request_id: str) -> str:
        first = self.stages[0][0]
        return derive_execution_lease_id(
            self.plan_generation_id,
            first.plan_id,
            first.plan_hash,
            request_id,
        )

    def execute(
        self, descriptor: BatchDescriptor, tensor: LogicalTensor
    ) -> RaggedBatchResult:
        if tensor.descriptor.row_count != descriptor.input_row_count:
            raise ValueError("scheduler tensor rows do not match batch descriptor")
        for sequence in descriptor.sequences:
            if sequence.execution_lease_id != self.issue_lease(sequence.request_id):
                raise RaggedRuntimeError(
                    "UNKNOWN_LEASE",
                    "descriptor lease is not bound to this worker-route generation",
                )
        current_descriptor = descriptor
        current_tensor = tensor
        original = {
            (item.request_id, item.execution_lease_id): item
            for item in descriptor.sequences
        }
        terminal: dict[tuple[str, str], SequenceResult] = {}
        final_result: RaggedBatchResult | None = None
        source_stage = "gateway"
        route_started_ns = time.monotonic_ns()
        for stage_number, (manifest, channel) in enumerate(self.stages):
            stage_descriptor = current_descriptor.with_stage_epochs(
                self._kv_epochs, stage_id=manifest.stage_id
            )
            elapsed_route_ns = max(0, time.monotonic_ns() - route_started_ns)
            stage_descriptor = replace(
                stage_descriptor,
                sequences=tuple(
                    replace(
                        item,
                        deadline_budget_ns=max(
                            0, item.deadline_budget_ns - elapsed_route_ns
                        ),
                    )
                    for item in stage_descriptor.sequences
                ),
            )
            request = RaggedBatchRequest(
                plan_id=manifest.plan_id,
                plan_hash=manifest.plan_hash,
                manifest_hash=manifest.manifest_hash,
                descriptor=stage_descriptor,
                input_tensor=current_tensor,
            )
            started = time.monotonic_ns()
            stage_result = channel.execute(request, source_stage=source_stage)
            elapsed = max(0, time.monotonic_ns() - started)
            self.events.append(
                {
                    "kind": "ragged_stage_result",
                    "stage_id": manifest.stage_id,
                    "batch_id": descriptor.batch_id,
                    "sequence_count": len(stage_result.results),
                    "successful_count": sum(
                        item.status == "ok" for item in stage_result.results
                    ),
                    "elapsed_ns": elapsed,
                    "evidence_class": "t1_loopback",
                }
            )
            successes: list[SequenceResult] = []
            for item in stage_result.results:
                key = (item.request_id, item.execution_lease_id)
                original_slice = original[key]
                normalized_item = replace(
                    item,
                    input_row_start=original_slice.input_row_start,
                    input_row_count=original_slice.input_row_count,
                )
                if item.status == "ok":
                    self._kv_epochs[(manifest.stage_id, *key)] = item.kv_epoch_after
                    successes.append(normalized_item)
                else:
                    terminal[key] = normalized_item
            if stage_number == len(self.stages) - 1:
                for item in successes:
                    terminal[(item.request_id, item.execution_lease_id)] = item
                combined = tuple(terminal[key] for key in sorted(terminal))
                final_result = RaggedBatchResult(
                    batch_id=descriptor.batch_id,
                    batch_sequence_no=descriptor.batch_sequence_no,
                    manifest_hash=manifest.manifest_hash,
                    results=combined,
                    output_tensor=stage_result.output_tensor,
                )
                break
            if not successes or stage_result.output_tensor is None:
                combined = tuple(terminal[key] for key in sorted(terminal))
                final_result = RaggedBatchResult(
                    batch_id=descriptor.batch_id,
                    batch_sequence_no=descriptor.batch_sequence_no,
                    manifest_hash=manifest.manifest_hash,
                    results=combined,
                    output_tensor=None,
                )
                break
            next_sequences: list[SequenceSlice] = []
            cursor = 0
            for item in successes:
                key = (item.request_id, item.execution_lease_id)
                previous = original[key]
                next_sequences.append(
                    replace(
                        previous,
                        input_row_start=cursor,
                        input_row_count=item.output_row_count,
                        kv_epoch=0,
                    )
                )
                cursor += item.output_row_count
            current_descriptor = BatchDescriptor(
                batch_id=str(
                    uuid.uuid5(
                        uuid.UUID(descriptor.batch_id), f"stage-{stage_number + 1}"
                    )
                ),
                batch_sequence_no=descriptor.batch_sequence_no,
                phase=descriptor.phase,
                input_row_count=cursor,
                sequences=tuple(next_sequences),
            )
            current_tensor = stage_result.output_tensor
            source_stage = manifest.stage_id
        assert final_result is not None
        return final_result

    def cancel(self, request_id: str, execution_lease_id: str) -> tuple[dict[str, Any], ...]:
        results: list[dict[str, Any]] = []
        for manifest, channel in self.stages:
            try:
                value = channel.cancel(request_id, execution_lease_id)
                value["stage_id"] = manifest.stage_id
                value["cleanup_complete"] = True
            except (RaggedRuntimeError, RaggedFrameError, OSError) as exc:
                value = {
                    "stage_id": manifest.stage_id,
                    "cancelled": False,
                    "cleanup_complete": False,
                    "error": {
                        "code": getattr(exc, "code", "TRANSPORT"),
                        "message": getattr(exc, "message", str(exc))[:256],
                    },
                }
            results.append(value)
        return tuple(results)

    def release(self, request_id: str, execution_lease_id: str) -> tuple[dict[str, Any], ...]:
        results: list[dict[str, Any]] = []
        pending: set[str] = set()
        key = (request_id, execution_lease_id)
        if (
            key not in self._cleanup_pending
            and len(self._cleanup_pending) >= self.max_cleanup_pending
        ):
            raise RaggedRuntimeError(
                "CLEANUP_CAPACITY",
                "cleanup-pending capacity is exhausted before release",
                retryable=True,
            )
        # Reserve the bounded saga slot before any stage can mutate.  Success
        # removes it below; partial transport failure keeps exact ownership.
        self._cleanup_pending.setdefault(
            key, {manifest.stage_id for manifest, _ in self.stages}
        )
        for manifest, channel in self.stages:
            try:
                value = channel.release(
                    request_id,
                    execution_lease_id,
                    expected_final_epoch=self._kv_epochs.get(
                        (manifest.stage_id, request_id, execution_lease_id)
                    ),
                )
                value["stage_id"] = manifest.stage_id
                value["cleanup_complete"] = True
                self._kv_epochs.pop(
                    (manifest.stage_id, request_id, execution_lease_id), None
                )
            except (RaggedRuntimeError, RaggedFrameError, OSError) as exc:
                pending.add(manifest.stage_id)
                value = {
                    "stage_id": manifest.stage_id,
                    "released": False,
                    "cleanup_complete": False,
                    "error": {
                        "code": getattr(exc, "code", "TRANSPORT"),
                        "message": getattr(exc, "message", str(exc))[:256],
                    },
                }
            results.append(value)
        if pending:
            self._cleanup_pending[key] = pending
        else:
            self._cleanup_pending.pop(key, None)
        self.events.append(
            {
                "kind": "ragged_request_released",
                "request_id": request_id,
                "execution_lease_id": execution_lease_id,
                "stage_count": len(results),
                "pending_stages": sorted(pending),
            }
        )
        return tuple(results)

    @property
    def cleanup_pending(self) -> dict[tuple[str, str], tuple[str, ...]]:
        return {
            key: tuple(sorted(value)) for key, value in self._cleanup_pending.items()
        }


@dataclass(frozen=True)
class RaggedGenerationRequest:
    request_id: str
    execution_lease_id: str
    prompt_token_ids: tuple[int, ...]
    deadline_ns: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if str(uuid.UUID(self.request_id)) != self.request_id:
            raise ValueError("request_id must be a canonical UUID")
        if str(uuid.UUID(self.execution_lease_id)) != self.execution_lease_id:
            raise ValueError("execution_lease_id must be a canonical UUID")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.prompt_token_ids):
            raise ValueError("prompt token IDs must be non-negative integers")
        if self.deadline_ns <= 0 or self.max_new_tokens <= 0:
            raise ValueError("deadline and max_new_tokens must be positive")


@dataclass
class _ScheduledState:
    request: RaggedGenerationRequest
    request_sequence_no: int = 0
    token_position: int = 0
    decoded_tokens: int = 0


class IntegratedRaggedScheduler:
    """Forms real FNX2 batches and executes them through the worker route."""

    def __init__(
        self,
        plan_id: str,
        *,
        max_sequences: int = 32,
        max_rows: int = 4096,
        max_queued: int = 4096,
        max_active: int = 4096,
        max_terminal: int = 4096,
        max_total: int = 8192,
        max_event_entries: int = 16_384,
    ) -> None:
        uuid.UUID(plan_id)
        for name, value in {
            "max_sequences": max_sequences,
            "max_rows": max_rows,
            "max_queued": max_queued,
            "max_active": max_active,
            "max_terminal": max_terminal,
            "max_total": max_total,
            "max_event_entries": max_event_entries,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.plan_id = plan_id
        self.max_sequences = max_sequences
        self.max_rows = max_rows
        self.max_queued = max_queued
        self.max_active = max_active
        self.max_terminal = max_terminal
        self.max_total = max_total
        self._batch_sequence = 0
        self._queued: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._active: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._terminal: OrderedDict[str, _ScheduledState] = OrderedDict()
        self.events: deque[dict[str, Any]] = deque(maxlen=max_event_entries)

    def submit(self, request: RaggedGenerationRequest) -> bool:
        if (
            request.request_id in self._queued
            or request.request_id in self._active
            or request.request_id in self._terminal
        ):
            raise RaggedRuntimeError("SEQUENCE", "request is already scheduled")
        if (
            len(self._queued) >= self.max_queued
            or len(self._queued) + len(self._active) + len(self._terminal)
            >= self.max_total
        ):
            self.events.append({"kind": "ragged_admission_rejected", "request_id": request.request_id})
            return False
        self._queued[request.request_id] = _ScheduledState(request=request)
        self.events.append({"kind": "ragged_request_queued", "request_id": request.request_id})
        return True

    def _next_batch_id(self, phase: str) -> str:
        return str(
            uuid.uuid5(
                uuid.UUID(self.plan_id), f"ragged-{self._batch_sequence}-{phase}"
            )
        )

    def _cleanup_terminal_state(
        self,
        orchestrator: RaggedOrchestrator,
        state: _ScheduledState,
    ) -> tuple[dict[str, Any], ...]:
        request = state.request
        result = orchestrator.release(
            request.request_id, request.execution_lease_id
        )
        if all(item.get("cleanup_complete") for item in result):
            self._queued.pop(request.request_id, None)
            self._active.pop(request.request_id, None)
            self._terminal.pop(request.request_id, None)
        else:
            if (
                request.request_id not in self._terminal
                and len(self._terminal) >= self.max_terminal
            ):
                # The caller reserves worst-case terminal slots before batch
                # formation. Reaching this path outside a batch fails before
                # local ownership is discarded.
                raise RaggedRuntimeError(
                    "SCHEDULER", "terminal cleanup capacity is exhausted"
                )
            self._queued.pop(request.request_id, None)
            self._active.pop(request.request_id, None)
            self._terminal[request.request_id] = state
        self.events.append(
            {
                "kind": "ragged_terminal_cleanup",
                "request_id": request.request_id,
                "complete": all(item.get("cleanup_complete") for item in result),
            }
        )
        return result

    def run_prefill(self, orchestrator: RaggedOrchestrator) -> RaggedBatchResult | None:
        selected: list[_ScheduledState] = []
        rows = 0
        now_ns = time.monotonic_ns()
        active_slots = self.max_active - len(self._active)
        terminal_slots = self.max_terminal - len(self._terminal)
        if active_slots <= 0 or terminal_slots <= 0:
            return None
        for state in tuple(self._queued.values()):
            if state.request.deadline_ns <= now_ns:
                self._queued.pop(state.request.request_id, None)
                self.events.append(
                    {
                        "kind": "ragged_admission_rejected",
                        "request_id": state.request.request_id,
                        "reason": "deadline",
                    }
                )
                continue
            count = len(state.request.prompt_token_ids)
            if len(selected) >= min(
                self.max_sequences, active_slots, terminal_slots
            ):
                break
            if count > self.max_rows:
                self._queued.pop(state.request.request_id)
                self.events.append({"kind": "ragged_admission_rejected", "request_id": state.request.request_id, "reason": "row_limit"})
                continue
            if rows + count > self.max_rows:
                # Skip a large head item and admit a later ready subset that fits.
                continue
            if state.request.execution_lease_id != orchestrator.issue_lease(
                state.request.request_id
            ):
                self._queued.pop(state.request.request_id)
                self.events.append(
                    {
                        "kind": "ragged_admission_rejected",
                        "request_id": state.request.request_id,
                        "reason": "unknown_generation_lease",
                    }
                )
                continue
            selected.append(state)
            rows += count
        if not selected:
            return None
        values: list[int] = []
        sequences: list[SequenceSlice] = []
        cursor = 0
        for state in selected:
            request = state.request
            count = len(request.prompt_token_ids)
            values.extend(request.prompt_token_ids)
            sequences.append(
                SequenceSlice(
                    request_id=request.request_id,
                    request_sequence_no=state.request_sequence_no,
                    input_row_start=cursor,
                    input_row_count=count,
                    token_position_start=0,
                    kv_epoch=0,
                    deadline_budget_ns=max(0, request.deadline_ns - now_ns),
                    execution_lease_id=request.execution_lease_id,
                    trace_id=f"trace-{request.request_id}",
                    span_id=f"prefill-{state.request_sequence_no}",
                )
            )
            cursor += count
        descriptor = BatchDescriptor(
            batch_id=self._next_batch_id("prefill"),
            batch_sequence_no=self._batch_sequence,
            phase="prefill",
            input_row_count=rows,
            sequences=tuple(sequences),
        )
        self._batch_sequence += 1
        try:
            result = orchestrator.execute(
                descriptor,
                LogicalTensor.from_values(
                    values, kind="token_ids", dtype="int32", shape=(rows,)
                ),
            )
        except Exception:
            for state in selected:
                self._cleanup_terminal_state(orchestrator, state)
            raise
        statuses = {
            (item.request_id, item.execution_lease_id): item.status for item in result.results
        }
        for state in selected:
            request = state.request
            if statuses.get((request.request_id, request.execution_lease_id)) == "ok":
                self._queued.pop(request.request_id, None)
                state.request_sequence_no += 1
                state.token_position = len(request.prompt_token_ids)
                self._active[request.request_id] = state
            else:
                self._cleanup_terminal_state(orchestrator, state)
        self.events.append({"kind": "ragged_prefill_batch", "batch_id": descriptor.batch_id, "sequence_count": len(selected), "row_count": rows})
        return result

    def run_decode(
        self,
        orchestrator: RaggedOrchestrator,
        next_token_ids: dict[str, int],
    ) -> RaggedBatchResult | None:
        now_ns = time.monotonic_ns()
        for request_id, state in tuple(self._active.items()):
            if state.request.deadline_ns <= now_ns:
                if len(self._terminal) >= self.max_terminal:
                    continue
                self._cleanup_terminal_state(orchestrator, state)
        terminal_slots = self.max_terminal - len(self._terminal)
        if terminal_slots <= 0:
            return None
        selected = [
            state
            for state in self._active.values()
            if state.request.request_id in next_token_ids
            and state.request.deadline_ns > now_ns
        ][: min(self.max_sequences, self.max_rows, terminal_slots)]
        if not selected:
            return None
        values: list[int] = []
        sequences: list[SequenceSlice] = []
        for row, state in enumerate(selected):
            request = state.request
            token = next_token_ids[request.request_id]
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("decode token IDs must be non-negative integers")
            values.append(token)
            sequences.append(
                SequenceSlice(
                    request_id=request.request_id,
                    request_sequence_no=state.request_sequence_no,
                    input_row_start=row,
                    input_row_count=1,
                    token_position_start=state.token_position,
                    kv_epoch=0,
                    deadline_budget_ns=max(0, request.deadline_ns - now_ns),
                    execution_lease_id=request.execution_lease_id,
                    trace_id=f"trace-{request.request_id}",
                    span_id=f"decode-{state.request_sequence_no}",
                )
            )
        descriptor = BatchDescriptor(
            batch_id=self._next_batch_id("decode"),
            batch_sequence_no=self._batch_sequence,
            phase="decode",
            input_row_count=len(selected),
            sequences=tuple(sequences),
        )
        self._batch_sequence += 1
        try:
            result = orchestrator.execute(
                descriptor,
                LogicalTensor.from_values(
                    values, kind="token_ids", dtype="int32", shape=(len(values),)
                ),
            )
        except Exception:
            for state in selected:
                self._cleanup_terminal_state(orchestrator, state)
            raise
        statuses = {
            (item.request_id, item.execution_lease_id): item.status for item in result.results
        }
        for state in selected:
            request = state.request
            if statuses.get((request.request_id, request.execution_lease_id)) != "ok":
                self._cleanup_terminal_state(orchestrator, state)
                continue
            state.request_sequence_no += 1
            state.token_position += 1
            state.decoded_tokens += 1
            if state.decoded_tokens >= request.max_new_tokens:
                self._cleanup_terminal_state(orchestrator, state)
        self.events.append({"kind": "ragged_decode_batch", "batch_id": descriptor.batch_id, "sequence_count": len(selected), "row_count": len(selected)})
        return result

    def cancel(
        self,
        orchestrator: RaggedOrchestrator,
        request_id: str,
    ) -> tuple[dict[str, Any], ...]:
        queued = self._queued.get(request_id)
        if queued is not None:
            self._queued.pop(request_id, None)
            self.events.append({"kind": "ragged_request_cancelled", "request_id": request_id})
            return ()
        state = self._active.get(request_id) or self._terminal.get(request_id)
        if state is None:
            return ()
        self.events.append({"kind": "ragged_request_cancelled", "request_id": request_id})
        cancelled = orchestrator.cancel(request_id, state.request.execution_lease_id)
        self._cleanup_terminal_state(orchestrator, state)
        return cancelled

    def release(
        self,
        orchestrator: RaggedOrchestrator,
        request: RaggedGenerationRequest,
    ) -> tuple[dict[str, Any], ...]:
        # Lookup and validate ownership before mutating any scheduler table.
        # A stale lease must not be able to pop the live request it conflicts
        # with.
        state = (
            self._active.get(request.request_id)
            or self._queued.get(request.request_id)
            or self._terminal.get(request.request_id)
        )
        if state is not None and (
            state.request.execution_lease_id != request.execution_lease_id
        ):
            raise RaggedRuntimeError("LEASE_CONFLICT", "release lease does not match scheduled request")
        if state is not None:
            result = self._cleanup_terminal_state(orchestrator, state)
        else:
            result = orchestrator.release(
                request.request_id, request.execution_lease_id
            )
        self.events.append({"kind": "ragged_request_released", "request_id": request.request_id})
        return result

    @property
    def stats(self) -> dict[str, int]:
        return {
            "queued": len(self._queued),
            "active": len(self._active),
            "terminal": len(self._terminal),
            "total": len(self._queued) + len(self._active) + len(self._terminal),
            "max_sequences": self.max_sequences,
            "max_rows": self.max_rows,
            "max_queued": self.max_queued,
            "max_active": self.max_active,
            "max_terminal": self.max_terminal,
            "max_total": self.max_total,
        }


def start_ragged_engine(
    manifests: tuple[RaggedStageManifest, ...],
    *,
    plan_generation_id: str | None = None,
) -> tuple[list[RaggedWorkerProcess], list[RaggedWorkerChannel], RaggedOrchestrator]:
    if len(manifests) < 2:
        raise ValueError("ragged engine requires at least two manifests")
    generation = plan_generation_id or str(uuid.uuid4())
    if str(uuid.UUID(generation)) != generation:
        raise ValueError("plan_generation_id must be a canonical UUID")
    workers = [
        RaggedWorkerProcess(
            manifest,
            plan_generation_id=generation,
            expected_source_stage=(
                "gateway" if index == 0 else manifests[index - 1].stage_id
            ),
        )
        for index, manifest in enumerate(manifests)
    ]
    channels: list[RaggedWorkerChannel] = []
    try:
        endpoints = [worker.start() for worker in workers]
        if len({item["pid"] for item in endpoints}) != len(endpoints):
            raise RaggedRuntimeError("WORKER_START", "workers are not independent processes")
        for manifest, endpoint in zip(manifests, endpoints):
            channel = RaggedWorkerChannel(
                manifest,
                str(endpoint["host"]),
                int(endpoint["port"]),
                plan_generation_id=generation,
            )
            channel.connect()
            channels.append(channel)
        return workers, channels, RaggedOrchestrator(list(zip(manifests, channels)))
    except Exception:
        for channel in channels:
            channel.disconnect()
        for worker in workers:
            worker.join()
        raise


def stop_ragged_engine(
    workers: list[RaggedWorkerProcess], channels: list[RaggedWorkerChannel]
) -> None:
    for channel in channels:
        try:
            channel.shutdown()
        except (OSError, RaggedFrameError, RaggedRuntimeError):
            channel.disconnect()
    for worker in workers:
        worker.join()
