# Fornax Runtime Format and Invariants

Version: 1.1-draft

Plan: `project-plan-v4.md` §4.4

Status: Logical target; dense single-request subset is implemented at T0/T1.
Ragged row mapping, real vocabulary logits, KV-page transfer, and expert batches
remain unimplemented and block the corresponding G2/product claims.

## 1. Contract layers

This document defines logical tensor meaning. The wire representation is defined
by `stage-runtime-and-wire-abi.md`. Backend-native MAX layouts may differ, but they
must convert to and from this logical contract without changing values, routing,
ownership, or token order.

## 2. Global invariants

- Every payload is bound to one model snapshot, plan hash, stage, request,
  microbatch, phase, token interval, and sequence number.
- Logical tensors are dense, contiguous, and row-major in ABI v1.
- Payload scalar bytes are little-endian.
- No implicit padding participates in logical shape or comparisons.
- NaN or infinity is invalid unless a future operation-specific contract permits
  it. No Phase 0.5 payload permits it.
- Backend conversion is explicit; silent dtype reinterpretation is forbidden.
- Logical/native import and export record descriptor validation, byte count,
  ownership, and whether a copy occurred; opaque native handles never weaken
  logical validation.
- The producer owns a source buffer until handoff acknowledgement.
- Plan changes never mutate an in-flight payload's interpretation.

## 3. Dtypes

| Logical dtype | ABI v1 | Scalar bytes | Phase 0.5 use |
|---|---|---:|---|
| BF16 | Required | 2 | Default activation/logit comparison path |
| FP16 | Optional negotiated | 2 | Fallback only after recorded reason |
| FP32 | Reference only | 4 | Accumulation/reference outputs |
| FP8 | Not wire-v1 | 1 | Deferred |
| Q4/Q8 | At-rest weights only | format-specific | Never interpreted as activation bytes |

Quantized weights must name encoding, group size, scale dtype/layout, zero-point
policy, packing order, and backend compatibility in the model manifest.

## 4. Activation tensor

| Field | Rule |
|---|---|
| Logical shape | `[token_rows, hidden_size]` |
| Layout | Contiguous row-major |
| Row order | Request order within microbatch, then token position order |
| Dtype | Manifest-bound BF16 or negotiated FP16 |
| Strides | `[hidden_size, 1]` logical elements |
| Padding | None in logical payload |
| Producer | Previous stage or gateway adapter for stage 0 |
| Consumer | Exactly one next stage in Phase 0.5 |
| Lifetime | Through receiver acknowledgement or terminal error |

The target activation descriptor includes the original request-to-row mapping.
Current FNX1 v1 does not carry that mapping in `TensorDescriptor`; its golden
fixture has only a sidecar `row_mapping`. Therefore v1 cannot execute a real
ragged multi-request stage batch, and a microbatching claim must wait for ABI v2.
Once implemented, a microbatch may not reorder rows without updating the mapping
and reference gather order.

## 5. Logits tensor

| Field | Rule |
|---|---|
| Logical shape | `[token_rows, vocabulary_size]`, or `[1, vocabulary_size]` when manifest binds final-token-only |
| Dtype | BF16/FP16 output plus FP32 reference comparison where available |
| Layout | Contiguous row-major |
| Token mapping | Explicit in descriptor |
| Sampling | Outside the stage unless final-stage manifest explicitly includes sampler |

Greedy validation records maximum absolute/relative error, top-1 identity, and
the top-k overlap chosen by the validation plan.

The current Phase 0.5 mechanism fixture labels its final hidden-width tensor
`logits`; it does not execute a vocabulary projection and is not model-logit
evidence. ABI v2/final-stage conformance must bind `vocabulary_size` and reject a
hidden-width substitute.

## 6. KV page logical format

KV remains stage-local during Phase 0.5. This contract exists so ownership is not
invented later.

| Field | Rule |
|---|---|
| `cache_id` | Derived from plan, request, and stage |
| `kv_epoch` | Monotonic per request/stage |
| `owner_stage` | Single writer |
| `layer_id` | Must belong to owner stage |
| `page_index` | Non-negative, unique within layer/cache |
| `page_size_tokens` | Manifest-bound |
| `valid_tokens` | `0..page_size_tokens` |
| `k_shape`, `v_shape` | Model/attention-specific and manifest-bound |
| `dtype` | Manifest-bound |
| `position_start` | Absolute logical token position |
| `payload_hash` | Required for persistence/transfer evidence |

An epoch change follows a successful mutation. Cancellation results state whether
the epoch changed. Cross-stage KV transfer is rejected by the Phase 0.5 runtime.

## 7. Routed expert batch logical format

Remote experts are deferred, but the preserved logical contract is:

| Field | Rule |
|---|---|
| `layer_id` | MoE layer in the source stage |
| `expert_ids` | One entry per packed route |
| `token_indices` | Original activation row for each route |
| `topk_slot` | Router slot for deterministic gather |
| `topk_weights` | Finite, non-negative, reference-normalized as model requires |
| `hidden` | `[route_count, hidden_size]` |
| `gather_order` | Permutation that restores model-defined route order |
| `router_hash` | Hash of routing identity fields and weights |

Duplicate routes are allowed only if the model's router emits them and the
reference contract records that behavior. Missing, invented, or reordered routes
are correctness failures.

## 8. Ownership and lifetime

| Object | Owner | Release point |
|---|---|---|
| Input activation source buffer | Sending stage | Matching ACK or terminal channel failure |
| Received activation buffer | Receiving stage | Stage execution and replay policy complete |
| Stage-local KV | Owning stage | Explicit final release, bounded idle expiry, expired execution-lease fencing, or plan drain in the reference/simulated runtime; physical equivalence remains G2 work |
| Output/logits source buffer | Producing stage | Matching ACK |
| Weight shard | Loaded stage manifest | Plan unload |
| Serialization staging buffer | Channel endpoint | ACK/error and no replay reference |

Reference-count or pool implementations are permitted, but observable ownership
must match this table.

### 8.1 Reference native-buffer seam

The T0/T1 reference backend imports a validated logical `Tensor` into a bounded
adapter-owned staging allocation, exports a separately validated logical tensor
to the deterministic Python oracle, and releases staging in `finally` paths. It
does the same at the output boundary. `StageHealth` exposes current/high-water
staging bytes and copy operations. The default adapter deliberately copies; no
zero-copy conclusion follows.

A physical backend may replace the opaque staging handle with a MAX/runtime
buffer. It must still validate kind, dtype, shape, layout, logical elements, byte
count, and finite-value policy before execution and again before publishing a
logical output. Device residency or zero-copy claims require physical evidence.

### 8.2 Bounded request lifecycle

Reference and simulated loaded stages use three independent bounds:

- idle request state is reclaimed on the next backend operation after its
  configured timeout;
- an internal execution lease prevents a late T0/T1 computation from committing
  KV or replay state after its deadline;
- explicit/automatic release records a count- and time-bounded request-ID
  tombstone.

The tombstone persists across FNX1 channel reconnects to the same loaded worker
and rejects reuse with `REQUEST_TOMBSTONED`. It does not persist across worker
restart, is not replicated, and is not an indefinite exactly-once guarantee.
Non-expired fences are not evicted to satisfy the count bound; a full table
returns `TOMBSTONE_CAPACITY` and retains the request state until a fence expires.
Physical backends must prove safe native cancellation/fencing separately.

## 9. Reference path and tolerances

The reference hierarchy is:

1. CPU/FP32 operator reference for focused kernels where feasible.
2. Whole-model or stage reference on a known-good backend/build.
3. Cross-backend comparison using identical model snapshot, prompt tokens,
   positions, masks, stage cut, and routing.

For every comparison record:

- reference and candidate build/hardware;
- shape, dtype, operation/stage/layer;
- maximum and percentile absolute/relative error;
- non-finite counts;
- token/routing/top-k identity checks;
- tolerance source and decision authority.

Initial Phase 0.5 BF16 thresholds are `atol=0.02`, `rtol=0.02`, subject to the
reference-error study in the target contract. FP16 uses `atol=0.005`,
`rtol=0.005` provisionally. These are acceptance thresholds, not measured error.

## 10. Malformed and stale payload behavior

Reject before execution when any of the following occurs:

- model, plan, manifest, stage, request, or sequence identity mismatch;
- unsupported dtype, rank, shape, layout, or byte count;
- non-finite activation, logits, or routing weight;
- invalid token interval or row mapping;
- stale/incorrect KV owner or epoch;
- invalid expert ID, top-k slot, gather permutation, or route count;
- checksum failure, duplicate conflict, or expired deadline.

No receiver may truncate, reshape, cast, renormalize, or repair a malformed
payload silently.

## 11. Golden-vector corpus required before ragged/physical claims

- BF16 activation vectors at hidden size 2048; add FP16 vectors if FP16 is
  enabled by the target contract.
- Row mapping across at least two requests and multiple token positions.
- Final-token and multi-row logits fixtures.
- KV page boundary, partial page, stale epoch, and wrong owner fixtures.
- Expert top-k routing/gather fixture for 64 experts and top-k 6.
- NaN, infinity, wrong-shape, wrong-byte-count, stale-plan, and duplicate-conflict
  negatives.
- Candidate outputs from the reference and simulated MAX backends for the
  accepted model snapshot.
- Physical Linux/NVIDIA and macOS/Apple candidates captured as G2 evidence when
  those backends are available; their absence does not block Phase 0.5.

The existing JSON golden manifest is retained as a small logical-schema fixture.
The FNX1 v1 corpus closes only the recorded dense T0/T1 mechanism scope; neither
fixture is sufficient for ragged batching, physical MAX conformance, real logits,
or G2 closure.
