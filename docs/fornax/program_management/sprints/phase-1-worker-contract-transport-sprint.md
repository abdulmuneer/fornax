# Phase-1 Worker Contract and Transport Sprint

> **V4 status:** active physical-validation lane after DEC-008. The Stage ABI,
> worker, loopback transport, and orchestrator are complete at T0/T1. This sprint
> now owns physical `MaxStageBackend` integration and T2/T3 evidence.

**Goal:** replay the completed Engine v0 contracts through real MAX backends and
a qualifying 2–3-node route, while correcting the ABI where native ragged MAX
batching proves v1 insufficient.

**Duration:** notional W5-W10.
**Milestone:** M2 Pipeline correctness (simulation, then 2-3 nodes).
**Gate contribution:** feeds G2, but does not close G2 without real 2-3 node
pipeline evidence.

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S1-1 | Stage host executes a layer-group boundary | RT | B2 | stage-host contract passes reference and boundary checks |
| S1-2 | Boundary activation/KV handoff semantics | RT + NET | B3 | golden vectors cover activation, KV pages, ownership, dtype, and shape |
| S1-3 | Slow-correct reference path | RT | B4 | optimized/simulated path has a reference comparator |
| S1-4 | Activation/KV transport contract | NET | E2 | simulated logical-host transport covers latency, bandwidth, ordering, and failure metadata |
| S1-5 | Engine backend skeleton | API + DIST | H3 | `FornaxBackend` seam records request, plan, tokenizer/template, and result contract fields |
| S1-6 | T1 worker-contract bundle | SRE + QA | B2-B4, E2, H3 | `fornax program simulate-t1` includes worker, transport, and serving checks |

## Sprint Board

| Deliverable | Status |
|---|---|
| S1-1 | T1 complete; live `MaxStageBackend` layer-group execution is active/open. |
| S1-2 | T1 activation/KV ownership semantics complete; physical custom-op/layout parity open. |
| S1-3 | Complete at T1: slow-correct/reference backend and golden oracle exist. |
| S1-4 | T1 persistent TCP complete; real physical route measurement and failure replay open. |
| S1-5 | Engine/serving seam complete at T1; physical MAX endpoint integration open. |
| S1-6 | T1 bundle closed by EV-009/DEC-008; G2 still needs T2/T3 proof. |

## Validation

- `python3 -m fornax test runtime-format --golden fornax/golden_vectors/runtime_format`
- `python3 -m fornax test stage-host --golden fornax/golden_vectors/stage_host`
- `python3 -m fornax test network-contract --mode simulated`
- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`

## Exit Criteria

- Worker boundary, transport, and serving seam are deterministic under T1.
- Two local GPUs can be treated as two logical hosts for smoke evidence when
  accelerator validation is available.
- Remaining G2 gap is explicit: real 2-3 node pipeline correctness and planner
  accuracy evidence.

## Active actions

1. **Source-lineage mechanism implemented in the working tree:**
   `dependencies/max-lineage.json` records and verifies the accepted MAX
   base/patch/diff. I-11 remains open until these files are committed and a fresh
   clean rebuild packet exists for each participating platform.
2. Implement `MaxStageBackend` behind the existing `StageExecutable` seam.
3. Run T2 operator/stage parity on Apple and NVIDIA candidates.
4. Run the T3 two-node prefill/decode, fault, calibration, and batching packet.
5. Record the measured Apple role and convene G2 only when all exit evidence is durable.
6. **Complete at T0/T1:** candidate FNX2 2.0 implements stage roles/input kinds,
   row offsets, per-sequence identity/position/KV/error, unequal prefill,
   independent decode, cancellation/release/expiry, leases/tombstones,
   multi-dimensional credits, an integrated scheduler, frozen golden vectors,
   and two independent loopback workers. Replay the corpus on physical MAX
   adapters before any batching/throughput claim.
7. **Complete at T0/T1:** replace simulator-hardcoded worker construction with
   an explicit serializable, fail-closed backend factory and backend-discovered
   pre-load capability attestation. Physical factory adapters still require T2
   evidence.
8. Run the parallel WS-I discovery lane: qualified interviews, privacy-safe
   workload traces, design-partner qualification, and pilot scorecard. This lane
   cannot close G2 or authorize product claims.
9. **Mechanism complete at T0/T1:** explicit release, in-flight rejection,
   idempotent repeat release, idle expiry, internal execution leases,
   same-worker tombstone fencing, bounded count/time/byte/event state, and a
   unique-request pressure runner. Complete I-22's evidence boundary with a
   reviewed long-duration run, restart-durable fences, and physical/native-KV
   validation.
10. Use `fornax program g2-validate` as the fail-closed packet builder: V1-V5
    must pass before an authorized physical manifest can run V6-V10. A
    hardware-free `BLOCKED` packet is readiness evidence, never G2 closure.

## 2026-07-17 T0/T1 hardening disposition

- Backend identity/capabilities now originate in the selected backend and are
  attested against the requested stage manifest before load; there is no silent
  physical-to-simulation fallback.
- Exact duplicate data frames replay cached wire responses without re-executing
  the stage; conflicting duplicates still fail.
- Non-finite tensor input produces a typed contract error without killing the
  worker connection, and the request-sequence extension remains optional under
  the frozen FNX1 golden contract.
- Orchestrator admission rejects mixed plan/model/ABI chains, noncontiguous stage
  indices/layers, and adjacent tensor-contract mismatches.
- Stage Backend API v2 requires final `release`; reference/simulated workers
  clear request-owned KV/cancel/execution/replay state and expose configured
  count/byte retention plus high-water health. The worker wire replay and
  orchestrator epoch/admission state are also cleared only after acknowledged
  release.

These are reference/loopback guarantees. They close I-14 and the
hardware-independent portions of I-13/I-16/I-22 at T0/T1, but do not close the
physical `MaxStageBackend`, physical ragged conformance, zero-copy/native hot
path, authenticated calibration evidence, or G2. Restart-durable fencing, a
reviewed long-duration current artifact, and physical/native-KV lifecycle proof
remain open.
