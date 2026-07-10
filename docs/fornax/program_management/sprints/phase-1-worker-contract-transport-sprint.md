# Phase-1 Worker Contract and Transport Sprint

> **V4 status:** active physical-validation lane after DEC-008. The Stage ABI,
> worker, loopback transport, and orchestrator are complete at T0/T1. This sprint
> now owns physical `MaxStageBackend` integration and T2/T3 evidence.

**Goal:** replay the completed Engine v0 contracts through real MAX backends and
a qualifying 2–3-node route without changing the Stage ABI.

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

1. Close I-11 with a root-pinned reproducible MAX lineage.
2. Implement `MaxStageBackend` behind the existing `StageExecutable` seam.
3. Run T2 operator/stage parity on Apple and NVIDIA candidates.
4. Run the T3 two-node prefill/decode, fault, calibration, and batching packet.
5. Record the measured Apple role and convene G2 only when all exit evidence is durable.
