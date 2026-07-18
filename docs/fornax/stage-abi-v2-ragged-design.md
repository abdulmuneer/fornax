# Stage ABI v2 ragged-batching design

Status: **implemented candidate at T0/T1; physical conformance open**

Decision authority: RT + DIST + NET review and a recorded ABI-major decision are
still required before FNX2 becomes a supported product-path ABI

Gate effect: reference, exact-wire, two-worker, integrated-scheduler, replay,
partial-failure, generation-bound lease/release, result-correlation, cleanup, and
bounded-state regressions now pass at T0/T1. This is not the full negative
conformance matrix and has no physical G2 authority.

This design addresses the expressiveness gap recorded as R-14/I-13: FNX1 v1 binds
one `StageRequest` to one request ID and KV epoch, while its tensor descriptor has
no request-to-row mapping. The candidate FNX2 implementation now wires a real
bounded scheduler through two independently spawned loopback workers without
inventing those semantics.

The implementation is the slow, model-free correctness oracle in
`fornax.stage_abi_v2`, `fornax.ragged_runtime`, and the public `fornax.ragged`
module. It does not claim physical MAX support, real-model logits, performance,
numerical parity, or G2 closure. FNX1 remains the compatibility contract for the
historical Engine-v0 evidence in
[`stage-runtime-and-wire-abi.md`](stage-runtime-and-wire-abi.md).

## 1. Required outcomes

V2 must represent and validate:

- token-input, hidden-activation, and final-logit stage boundaries;
- unequal prefill chunks from multiple requests in one packed tensor;
- independent one-row decode progress for a changing subset of requests;
- explicit request-to-row and absolute-position mappings;
- per-sequence KV ownership, epoch transitions, deadline, cancellation, status,
  and error;
- partial batch outcomes without mutating failed sequences;
- at-most-once replay per sequence, even when batches are repacked;
- explicit request release/expiry and bounded retention; and
- logical values independent of backend-native MAX layouts.

Streaming text, sampling policy, remote-expert transport, cross-stage KV
migration, and a specific MAX paged-cache layout are separate contracts.

## 2. Major-version boundary

V2 is a new wire major, not a v1 optional-field extension. The candidate frame
identity is `FNX2`, ABI `2.0`. The first golden vector now freezes the canonical
prefill frame at 1,292 bytes and
`sha256:1f4ba161723e850234198404deaae18aa0976ff321e8ce9d312ecf7cf816bb6b`.
The digest changed deliberately when the non-portable sender monotonic timestamp
was replaced by a relative deadline budget. V1 and v2 channels never mix frames.
Before issuing any credit, the
worker sends a mandatory `CONTROL hello` and requires a correlated ACK binding
the exact plan, manifest, stage, worker-route generation, and FNX2 2.0. Every
other minor fails closed. Selection among multiple supported versions remains an
open upgrade-path contract before a mixed-version product endpoint.

The Python Stage Backend API and wire ABI remain independent. An adapter may
support backend API v2 while advertising only FNX1, but it cannot claim ragged
execution until it advertises and passes FNX2.

## 3. Stage and tensor roles

Every manifest binds one stage role and exact boundary kinds:

| Field | Values | Rule |
|---|---|---|
| `stage_role` | `first`, `middle`, `final` | Fixed for the manifest |
| `input_kind` | `token_ids`, `activation` | `first` may accept tokens; others accept activations |
| `output_kind` | `activation`, `logits` | Only `final` may emit logits |
| `hidden_size` | positive integer when activation is present | Must match adjacent stages |
| `vocabulary_size` | positive integer for logits | Output width must equal it |
| `logical_layout` | `contiguous_row_major` initially | Native layouts convert explicitly |

`token_ids` use `[row_count]` integer storage. Activations use
`[row_count, hidden_size]`. Logits use `[output_row_count, vocabulary_size]`, or a
manifest-declared final-token-only subset whose row mapping remains explicit.
Calling a hidden-width mechanism tensor “logits” is a contract failure.

## 4. Batch descriptor

Each data frame carries a canonical `BatchDescriptor` in metadata and its
SHA-256. The payload descriptor and batch descriptor are hashed together for
replay identity.

```text
BatchDescriptor {
  batch_id: UUID                       // transport correlation only
  batch_sequence_no: uint64            // monotonic on the logical stage route
  phase: prefill | decode
  input_row_count: uint32
  sequences: [SequenceSlice, ...]
}

SequenceSlice {
  request_id: UUID
  request_sequence_no: uint64          // monotonic within request
  input_row_start: uint32
  input_row_count: uint32
  token_position_start: uint64
  kv_epoch: uint64
  deadline_budget_ns: uint64           // remaining budget, never a clock value
  execution_lease_id: UUID
  trace_id: string
  span_id: string
}
```

Initial v2 requires each sequence's rows and token positions to be contiguous.
Slices are sorted by `input_row_start`, do not overlap, cover every input row
exactly once, and have positive row counts. Prefill may use different row counts;
decode uses one row per active sequence. Repacking may change `batch_id` and row
offsets, but never request identity, position, KV epoch, or logical values.

The execution lease is deterministically derived by the orchestrator from a fresh worker-route
generation plus plan and request identity. All stages in one route share that
generation. On receipt, each worker converts the remaining budget to its own
monotonic clock domain. The first receipt establishes a local deadline; a retry
uses `min(stored_deadline, receiver_now + remaining_budget)`, so it may tighten
but never extend that deadline. The orchestrator subtracts route elapsed time
before forwarding to each stage, and a zero budget is an already-expired
request. No sender monotonic timestamp crosses a process or machine boundary. A restart
uses a new generation and rejects old leases. Within a generation, release and
expiry tombstones do not time-expire: the bounded tombstone table fails closed
when full, and the generation may be rotated only after drain. This makes the
boundedness tradeoff explicit instead of allowing an old arbitrary UUID to
revive after a time window. This is stale-generation fencing, not authentication
or authorization; clients should call `orchestrator.issue_lease(request_id)`
(the deterministic helper is exported from `fornax.ragged`).

## 5. Per-sequence result

A batch result contains one `SequenceResult` for every input slice, in request
identity order, plus a compacted tensor for successful outputs:

```text
SequenceResult {
  request_id: UUID
  request_sequence_no: uint64
  status: ok | cancelled | deadline | rejected | failed
  input_row_start: uint32
  input_row_count: uint32
  output_row_start: uint32 | null
  output_row_count: uint32
  kv_epoch_before: uint64
  kv_epoch_after: uint64
  error: {code, message} | null
  timings_ns: {queue, execute, pack, unpack}
}
```

Successful output ranges are contiguous, non-overlapping, and cover the output
tensor exactly once. A failed sequence has zero output rows and cannot advance
KV. Other sequences may succeed; the batch-level transport status is not a
substitute for per-sequence status. A backend that cannot isolate partial
failure must fail every sequence before any mutation. When an upstream failure
compacts later-stage input, the orchestrator normalizes final
`input_row_start/input_row_count` fields back to the original scheduler batch;
`output_row_start` continues to address the compacted final tensor.

## 6. KV, cancellation, and release

- KV has one writer per `(plan, stage, request, execution_lease_id)`.
- Successful prefill/decode advances that sequence's epoch exactly once.
- Deadline, pre-execution cancellation, validation failure, and conflicting
  replay leave its epoch unchanged.
- Cancellation names request plus lease and is idempotent. It reports whether
  execution started and whether KV mutated.
- `RELEASE_REQUEST` names plan, stage, request, lease, and expected final epoch.
  It is accepted only with no inflight execution, clears KV and heavy replay
  state, and returns the released epoch/counts.
- Release installs a lightweight tombstone for the remainder of the immutable
  plan generation. Tombstones, live requests, completed results, retained replay
  bytes, KV bytes, scheduler states, transforms, and events have configured hard
  caps and high-water marks. Tombstone exhaustion rejects new admission and a
  failed release keeps live state; it never evicts the anti-replay fence, loses
  live KV, or permits stale work to execute.
- Plan drain rejects new leases, completes/cancels inflight work, releases all
  request state, then unloads weights.
- Terminal scheduler failures run release as a cleanup saga across every stage.
  Successfully cleaned stages are not rolled back; failed cleanup stages remain
  in a bounded retry set, and local ownership is retained until cleanup completes.

## 7. At-most-once identity

The semantic replay key is:

```text
(plan_hash, manifest_hash, request_id, execution_lease_id,
 request_sequence_no, phase, token_position_start, kv_epoch,
 input_tensor_digest)
```

An identical retry returns the exact cached per-sequence result. The same logical
identity with a different digest or mapping is `SEQUENCE_CONFLICT`. A retry older
than retained result history is rejected using current KV epoch, lease state, or
release tombstone; it is never re-executed. Batch packing and channel sequence
numbers are not part of semantic identity.

## 8. Credits and bounds

The receiver advertises simultaneous limits for frames, payload bytes, input rows,
sequences, live request states, KV bytes, and retained replay-output bytes.
Admission requires all credits. The worker rechecks its own last advertised
vector and current free resources atomically with execution; the sender-side
check is only an early rejection. Exact cached semantic replay requires no new
live-request, KV, or replay-byte credit and remains legal when those three free
credits are zero. Metadata and the complete frame are separately
bounded by the codec's fixed metadata and payload limits; `payload_bytes` is not
presented as a total wire-byte credit.
Transient frame/payload/row/sequence credit is refreshed after each result.
Live-request, KV, and retained-replay credits return only on release or expiry;
release sends a fresh vector after its ACK, and `CONTROL credit_refresh` provides
an explicit expiry sweep/refresh path. Every limit and high-water mark is
observable. No implementation may call a run memory-bounded based only on socket
credit or process RSS.

## 9. Validation order

Before backend execution, validate in this order:

1. frame length, exact ABI, flags, checksum, and canonical metadata;
2. plan, manifest, exact source/destination stage, route generation, stage role,
   and tensor kind;
3. descriptor rank/dtype/layout/width and finite values;
4. slice coverage, uniqueness, positions, row counts, deadlines, and leases;
5. request/KV epoch, cancellation/release state, replay identity, and credits;
6. backend capability attestation for the entire batch envelope.

Failure before step 6 cannot call the backend. Diagnostics are bounded and do
not include activation, token, or KV contents.

A fully consumed, length-delimited semantic metadata/tensor/descriptor error
consumes its channel sequence, returns one bounded `ERROR`, and permits the next
frame. ABI/version, oversize, truncation, and checksum failures are terminal
because safe resynchronization or integrity cannot be established.

## 10. Required T0/T1 conformance

Positive cases:

- two-request prefill with unequal row counts and exact row/value preservation;
- three-request decode, then cancellation of the middle request and independent
  progress of the other two;
- first-stage token input, middle activation, and vocabulary-shaped final logits;
- mixed per-sequence success/deadline/cancel with correct compacted output map;
- exact retry after repacking with no second execution or KV mutation;
- release followed by zero heavy state and idempotent repeated release; and
- long unique-request soak that remains within every configured state/event cap.

The focused executable regressions cover overlap, gap, duplicate request/lease,
zero rows, out-of-range positions, wrong output width, stale KV, conflicting digest,
expired/unknown lease, released request, partial mutation on failure, credit
overflow, malformed per-sequence error, future minor version, and old replay
after cache eviction. They also cover strict JSON types/unknown fields,
receiver-local deadline budgets under skewed clocks and retry tightening,
phase/position progression, replay after cancellation, generation
restart fencing, tombstone-capacity fail-closed behavior, exact result
correlation with channel quarantine, receiver-side credit enforcement, exact
replay at zero free KV/replay credit, retained-byte caps, transport-normalized
cleanup continuation, bounded cleanup-pending state, lookup-before-mutation
release, source-stage binding, non-shutdown control continuity, original-row
result normalization, and total scheduler bounds. A reference-only validator run
reports `t0_reference_only`; it cannot claim loopback evidence. Additional
malformed-frame, negotiated-upgrade, and
physical-adapter cases remain required for the complete matrix.

The same semantic cases must pass:

1. direct slow reference backend;
2. serialized FNX2 over two independent loopback workers;
3. simulated scheduling/fault envelopes; and
4. each physical MAX adapter before any G2 batching claim.

The packaged command is:

```bash
python3 -m fornax test stage-abi-v2
```

It currently replays the exact codec, unequal prefill, vocabulary-shaped final
output, independent decode, two independent worker processes, bounded
multi-dimensional credits, and final release. Its report deliberately states
`physical_g2_passed=False`.

## 11. Physical acceptance

Physical conformance adds exact model/build/device identity, reference activation
and final-logit tolerances, row/position/KV parity, component timings, bounded
native buffers and KV, cancellation/release under load, and repeated
NVIDIA-to-Apple execution. A passing functional adapter smoke or finite output is
insufficient.

## 12. Migration sequence

1. **Open:** freeze this candidate through RT/DIST/NET review and record the
   ABI-major decision.
2. **Complete at T0:** add v2 dataclasses and canonical JSON/hash tests without
   changing v1 bytes.
3. **Partial at T0:** implement the slow reference oracle and focused P0/P1
   negative regressions. The full transport/upgrade/physical matrix remains open.
4. **Complete at T0/T1:** extend v1 final release/count/byte bounds with v2 per-sequence semantic
   replay plus automatic expiry, leases, and tombstones.
5. **Complete at T0/T1:** implement the FNX2 codec and two-worker loopback
   golden vector.
6. **Complete at T1 reference scope:** integrate bounded admission, unequal
   prefill, changing-subset decode, cancellation, and release with the real FNX2
   orchestrator. This is model-free reference execution, not a MAX hot path.
7. **Open:** implement physical MAX adapters and acquire T2/T3 evidence.
8. **Open:** retire v1 from product paths only after compatibility and rollback
   evidence.

Fornax may now claim an integrated **T1 reference ragged-batching engine**. Until
step 7 passes, it may not claim native MAX ragged batching, physical
heterogeneous batching, production continuous-batching throughput, or G2.
