# Fornax Stage Runtime and Wire ABI

Version: 1.0-draft  
Plan: `project-plan-v4.md` §4  
Status: Implemented and conformant at T0/T1; physical `MaxStageBackend` conformance pending G2

## 1. Scope

This specification defines the experimental v1 boundary between Fornax
orchestration and a future MAX-backed stage worker, plus the v1 frame currently
exercised over T1 loopback. Physical-host conformance remains open.

It does not define MAX internal graph APIs, remote-expert RPC, production service
discovery, RDMA registration, or distributed KV migration.

## 2. Design rules

1. Network operations occur outside compiled MAX graphs.
2. A stage is a complete contiguous model layer range.
3. All execution is identified by immutable model, build, plan, and stage hashes.
4. Buffers have one owner at a time and explicit release/acknowledgement points.
5. V1 sends one contiguous logical tensor per data frame.
6. Unknown major versions, stale plans, invalid shapes, and checksum failures fail
   closed before graph execution.
7. Retries occur only at explicitly replay-safe boundaries.

## 3. Stage manifest

Each worker loads a signed-or-hashed manifest before accepting requests.

| Field | Type | Rule |
|---|---|---|
| `manifest_version` | integer | Must equal 1 |
| `model_id` | string | Exact repository or local identifier |
| `model_snapshot` | string | Immutable revision/hash |
| `model_config_hash` | SHA-256 | Canonical model configuration |
| `tokenizer_hash` | SHA-256 | Required even when tokenizer runs only at gateway |
| `template_hash` | SHA-256 | Required for request reproducibility |
| `max_build_id` | string | Exact accepted MAX build identity |
| `fornax_abi_major/minor` | integers | Worker compatibility |
| `plan_id` | UUID | Immutable installed placement plan |
| `plan_hash` | SHA-256 | Canonical plan bytes |
| `stage_id` | string | Unique within plan |
| `stage_index` | integer | Zero-based pipeline order |
| `layer_start/end` | integers | Inclusive contiguous range |
| `input/output_contract` | objects | Dtype, rank, dimensions, logical layout |
| `kv_policy` | enum | `stage_local` in Phase 0.5 |
| `weight_artifacts` | list | Path/URI, size, SHA-256, assigned layer range |
| `device_requirement` | object | Backend, device identity, minimum memory, dtype capabilities |

The worker computes a canonical manifest hash. `plan_hash`, `stage_id`, and
manifest hash are attached to every execution.

## 4. StageExecutable interface

The language-neutral contract is:

```text
capabilities() -> BackendCapabilities
load(manifest) -> StageHandle
health(StageHandle) -> StageHealth
execute(StageHandle, StageRequest) -> StageResult
cancel(StageHandle, RequestId, reason) -> CancelResult
release(StageHandle, RequestId) -> ReleaseResult
drain(StageHandle, deadline) -> DrainResult
unload(StageHandle) -> UnloadResult
```

`capabilities()` is evaluated before `load`. Its backend, build, device,
memory, dtype, operation, quantization, ABI, and frame-limit fields originate
from the constructed backend—not from the requested manifest. The worker stores
requested and observed values in a capability attestation and fails startup with
`CAPABILITY_MISMATCH` when a known requirement is not met.

Worker construction uses a process-serializable backend specification. The
reference and simulated backends are built in; a physical MAX backend must name
an importable adapter factory. A missing or invalid factory fails closed and may
never select the simulator implicitly.

The versioned experimental Python imports, factory example, lifecycle smoke, and API-versioning
policy are documented in
[`stage-backend-adapters.md`](stage-backend-adapters.md).
The incompatible ragged multi-sequence successor is an unimplemented T0 design
in [`stage-abi-v2-ragged-design.md`](stage-abi-v2-ragged-design.md).

### StageRequest

| Field | Required | Meaning |
|---|---|---|
| `plan_id`, `plan_hash` | yes | Installed immutable placement |
| `request_id` | yes | End-to-end request UUID |
| `microbatch_id` | yes | Globally unique within request set |
| `sequence_no` | yes | Monotonic stage-message sequence |
| `phase` | yes | `prefill` or `decode` |
| `token_start`, `token_count` | yes | Logical positions represented by input |
| `input_activation` | except stage 0 token input adapter | Logical tensor descriptor plus buffer |
| `kv_epoch` | yes | Expected stage-local KV state version |
| `deadline_ns` | yes | Absolute orchestration deadline |
| `trace_context` | yes | Request/plan/span correlation |

### StageResult

| Field | Required | Meaning |
|---|---|---|
| Identity fields | yes | Echo accepted plan/request/microbatch/sequence |
| `status` | yes | `ok`, `cancelled`, `deadline`, `rejected`, `failed` |
| `output_kind` | on success | `activation` or `logits` |
| `output_tensor` | on success | Descriptor and buffer |
| `kv_epoch_before/after` | yes | Ownership/version transition |
| `timings_ns` | yes | queue, execute, pack/unpack as locally observable |
| `error` | on failure | Stable code plus bounded diagnostic text |

## 5. Execution state machine

```text
UNLOADED -> LOADING -> READY -> DRAINING -> UNLOADING -> UNLOADED
                     |   |
                     |   +-> FAILED
                     +-> EXECUTING -> READY
```

- A worker executes only the currently installed manifest.
- Plan replacement requires drain, unload, and load; V1 has no in-place mutation.
- Per-request stage execution replays an exact retained result without executing
  twice inside the backend's bounded replay window. An evicted, conflicting, or
  older sequence fails closed. Reference/simulated backend state survives an
  FNX1 channel reconnect to the same worker: retained results still replay and a
  bounded release tombstone rejects a finalized request ID. FNX1 has no durable
  replay epoch, so no guarantee survives worker restart or configured tombstone
  expiry. Non-expired fences are never evicted: tombstone-capacity exhaustion
  fails closed while retaining live request state.
- Cancellation is best effort after execution starts. The result states whether
  KV mutation occurred.
- API v2 and the implemented FNX1 control paths define explicit final
  `release(request_id)`. Release fails with `REQUEST_INFLIGHT`; after terminal
  work it clears request-owned KV, cancellation, execution, idempotency, wire
  replay, and orchestrator epoch/admission state. Repeating release for absent
  state is safe and reports `released=false`.
- The reference/simulated runtime caps live requests, results per request,
  retained-result bytes, transform entries/bytes, and event history. Admission
  and old replay fail closed when those bounds are reached or evicted; live KV is
  never evicted merely to make room for replay data.
- A newly executed result becomes replay-visible only after backend-specific
  finalization. The orchestrator commits its stage KV epochs only after every
  stage succeeds, so an exact retry after a downstream failure can replay the
  already-completed upstream stage instead of advancing it twice.
- Once any part of final release starts, execution for that request stays fenced
  until every stage acknowledges a release attempt. A failed partial release is
  retried; it does not reopen execution against partially destroyed state.
- The reference/simulated loaded stage opportunistically expires idle request
  state, fences late results with an internal execution lease, and retains
  explicit/automatic release tombstones within configured count and time bounds.
  Callers should still release promptly; expiry is a bounded safety net.
- These are T0/T1 Python mechanism guarantees, not an indefinite-service or
  physical-memory result. There is a runnable many-unique-request pressure mode,
  but physical native-KV/RSS stress and restart-durable replay remain open. FNX1
  framing did not change and carries no durable replay epoch.

## 6. V1 frame format

Every data-plane message is:

```text
[40-byte fixed prelude][metadata JSON][raw tensor payload]
```

All fixed-prelude integers use network byte order. Tensor scalar bytes use the
little-endian representation declared by the dtype contract.

### Fixed prelude

| Offset | Bytes | Field | Rule |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `FNX1` |
| 4 | 2 | ABI major | `1` |
| 6 | 2 | ABI minor | `0` initially |
| 8 | 2 | message kind | Enum below |
| 10 | 2 | flags | Reserved bits must be zero |
| 12 | 4 | metadata bytes | Maximum 64 KiB |
| 16 | 8 | payload bytes | Must equal tensor descriptor byte count |
| 24 | 8 | sequence number | Monotonic per channel |
| 32 | 4 | CRC32C | Over metadata followed by payload |
| 36 | 4 | reserved | Must be zero |

### Message kinds

| Value | Kind | Payload |
|---:|---|---|
| 1 | `ACTIVATION` | Raw activation tensor |
| 2 | `LOGITS` | Raw logits tensor |
| 3 | `KV_PAGE` | Reserved; rejected in Phase 0.5 baseline |
| 4 | `EXPERT_BATCH` | Reserved; rejected in Phase 0.5 baseline |
| 5 | `CREDIT` | No tensor payload; flow-control metadata |
| 6 | `ACK` | No tensor payload |
| 7 | `CANCEL` | No tensor payload |
| 8 | `ERROR` | No tensor payload |
| 9 | `HEARTBEAT` | No tensor payload |

### Required metadata

Metadata is UTF-8 JSON with sorted keys and no duplicate keys. V1 requires:

- `plan_id`, `plan_hash`, `manifest_hash`;
- `request_id`, `microbatch_id`, `sequence_no`;
- `source_stage`, `destination_stage`;
- `phase`, `token_start`, `token_count`, `kv_epoch`;
- `tensor.kind`, `tensor.dtype`, `tensor.shape`, `tensor.layout`;
- `tensor.logical_elements`, `tensor.payload_bytes`;
- `deadline_ns`, `trace_id`, `span_id`.

The metadata `sequence_no` must match the prelude. UUIDs use lower-case canonical
text. Hashes use `sha256:<64 lowercase hex>`.

`request_sequence_no` is an optional v1 extension used to keep a request's
prefill/decode sequence independent of the channel sequence. A v1 sender that
omits it remains valid; the receiver uses the frame sequence as the execution
sequence. When present it is a non-boolean, non-negative JSON integer. New Fornax
workers emit it. Making it mandatory or changing that fallback requires a future
ABI major.

## 7. Tensor representation

- V1 accepts only dense contiguous row-major payloads.
- Activation shape is `[token_rows, hidden_size]`.
- Logits shape is `[token_rows, vocabulary_size]` unless the manifest binds a
  final-token-only result.
- BF16 uses IEEE bfloat16 bit representation; FP16 uses IEEE binary16.
- Payloads contain no implicit padding. Any transport padding follows the payload
  and is excluded from `payload_bytes` and CRC.
- Compression and quantized activation transport are unsupported in V1.
- A receiver may copy into a backend-native buffer; zero-copy is not required.

## 8. Ordering, ownership, and acknowledgement

1. The sender retains the source buffer until a valid `ACK` or terminal `ERROR`.
2. The receiver owns its copied/backend buffer after CRC, metadata, and manifest
   validation.
3. An `ACK` identifies the exact request, microbatch, sequence, and payload CRC.
4. V1 advertises one input credit and retains the exact response for the newest
   completed data frame. An immediate exact duplicate replays that response and
   is never executed again.
5. Duplicate frames whose identity matches but CRC differs terminate the channel.
6. Out-of-order frames are rejected in V1; the sender reconnects and replays only
   from an orchestrator-approved boundary.
7. Accepting a newer input sequence evicts the prior replay response and digest.
   A retry older than the bounded window fails with `SEQUENCE`; it is never
   re-executed. This keeps persistent-channel replay memory bounded by one tensor
   response under the one-credit protocol.
8. A backend-level logical `SEQUENCE` rejection returns the consumed credit and
   keeps the negotiated channel available for final request release. A wire-level
   conflicting duplicate/integrity failure remains channel-terminal.

## 9. Compatibility

- Major or minor mismatch: reject channel. The implemented v1 contract is exactly
  `1.0`; optional-field negotiation must be specified and tested before any
  minor-version bump.
- Dtype/layout/model/plan mismatch: reject frame without graph execution.
- A running plan is immutable. Rolling upgrade uses parallel worker/channel sets
  and a gateway drain; it is not part of Phase 0.5.

## 10. Stable error codes

| Code | Meaning |
|---|---|
| `ABI_VERSION` | Unsupported protocol version |
| `FRAME_SIZE` | Length exceeds contract or configured limit |
| `CHECKSUM` | CRC failure |
| `METADATA` | Invalid/duplicate/missing metadata |
| `STALE_PLAN` | Plan or manifest mismatch |
| `TENSOR_CONTRACT` | Dtype, rank, shape, layout, or byte-count mismatch |
| `SEQUENCE` | Invalid ordering or conflicting duplicate |
| `NO_CREDIT` | Sender exceeded advertised flow-control credit |
| `DEADLINE` | Deadline expired before execution |
| `CANCELLED` | Request cancelled |
| `EXECUTION` | MAX stage execution failed |
| `CAPABILITY_MISMATCH` | Backend-observed facts do not satisfy the requested manifest |
| `ADMISSION` | Configured live-request state capacity is exhausted or the request already has work in flight |
| `REQUEST_INFLIGHT` | Final release was requested while stage or pipeline work is still in flight |
| `REQUEST_TOMBSTONED` | Request ID is fenced by a bounded explicit/automatic release tombstone |
| `TOMBSTONE_CAPACITY` | A new release/expiry fence cannot be installed; existing fences and live state are retained and the operation fails closed |
| `LEASE_EXPIRED` | A T0/T1 execution completed after its internal lease was fenced; its result/KV state was discarded |

## 11. Required conformance corpus

- Valid BF16 activation and logits frames.
- Every supported message kind with zero payload rules.
- Short prelude, oversized metadata, oversized payload, truncated payload.
- Wrong CRC, wrong byte order marker through impossible version, duplicate keys.
- Stale plan, wrong manifest, wrong destination, deadline expired.
- Shape/byte-count mismatch and unsupported dtype/layout.
- Duplicate same CRC, duplicate different CRC, and out-of-order sequence.
- Cancellation before queue, while queued, and after execution starts.
- Credit exhaustion and recovery.
- Exact duplicate data frame replays the cached wire response without calling
  the backend again.
- Replay and sequence histories stay bounded; after the next input is accepted,
  an older retry fails closed without backend execution.
- A malformed optional `request_sequence_no` is rejected as `METADATA` and does
  not terminate the worker process.
- A valid-CRC non-finite tensor returns `TENSOR_CONTRACT`, restores credit, and
  leaves the worker able to process the next valid frame.
- Backend capability attestation is recorded before manifest load; requested
  values are never reported as observed facts.
- Final request release clears request-owned backend, wire-replay, KV-epoch, and
  orchestrator admission state; repeat release is idempotent and in-flight
  release fails closed.
- Partial-pipeline exact retry reuses the upstream retained result, simulated
  result decoration is atomic with replay visibility, logical sequence rejection
  preserves same-channel release, and partial release fences new execution until
  a retry finishes cleanup.
- Live-request, completed-result, transform-cache, retained-byte, and event
  histories stay within configured bounds, with health counters exposing the
  current and high-water values.
- Idle expiry reclaims request-owned state, late execution-lease completion
  cannot resurrect it, and count/time-bounded tombstones fence request-ID reuse
  after a data-channel reconnect to the same worker.
- Logical/native imports validate descriptors and bytes, expose copy and
  high-water accounting, and release every staging allocation on success and
  failure paths.

Conformance must run in T0/T1 and against the physical T3 channel before G2
`PROCEED`. G1 authorized Engine v0 against reference/simulated backends; it did
not close physical channel conformance.

### Golden change record — 2026-07-17

The Stage ABI v1 golden conformance list grew from 24 to 31 named checks for the
optional request-sequence fallback/type rule, exact minor-version rejection,
bounded sequence history, and backend-factory capability attestation, plus
socketpair proof that duplicate frames do not execute twice and non-finite input
does not kill the worker. The
FNX1 frame bytes, manifest hashes, reference tensor payload, and ABI major/minor
did not change. This is a stronger validation of the existing v1 contract, not a
wire-format revision. The separate Stage Backend API moved from v1 to v2 when
`release` and retention health became required; FNX1 remains exactly wire 1.0.
