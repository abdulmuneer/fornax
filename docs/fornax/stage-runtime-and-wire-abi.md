# Fornax Stage Runtime and Wire ABI

Version: 1.0-draft  
Plan: `project-plan-v4.md` §4  
Status: Implemented and conformant at T0/T1; physical `MaxStageBackend` conformance pending G2

## 1. Scope

This specification defines the stable boundary between Fornax orchestration and
a MAX-backed stage worker, plus the v1 frame used to exchange stage-boundary
tensors between physical hosts.

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
load(manifest) -> StageHandle
health(StageHandle) -> StageHealth
execute(StageHandle, StageRequest) -> StageResult
cancel(StageHandle, RequestId, reason) -> CancelResult
drain(StageHandle, deadline) -> DrainResult
unload(StageHandle) -> UnloadResult
```

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
- Per-request stage execution is at-most-once unless the orchestrator supplies a
  new replay epoch at a documented replay-safe boundary.
- Cancellation is best effort after execution starts. The result states whether
  KV mutation occurred.

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
4. Duplicate frames with an already-acked sequence are acknowledged but never
   executed again.
5. Duplicate frames whose identity matches but CRC differs terminate the channel.
6. Out-of-order frames are rejected in V1; the sender reconnects and replays only
   from an orchestrator-approved boundary.

## 9. Compatibility

- Major mismatch: reject channel.
- Receiver minor lower than sender minor: accept only when all unknown metadata
  fields are optional and flags are zero; otherwise reject.
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

Conformance must run in T0/T1 and against the physical T3 channel before G1
`PROCEED`.
