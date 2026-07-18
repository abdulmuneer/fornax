# Phase 0.5 — Assumption-Driven Engine v0 Sprint

Plan: `../../project-plan-v4.md` Phase 0.5  
Gate posture: G1 `PROCEED`; feeds M1 and G2 entry  
Duration: W1–W6 notional; re-estimate after first executable worker pair
Status: **Complete at T0/T1 scope; closed by DEC-008 on 2026-07-10**

## Goal

Build the first executable Fornax engine without waiting for the complete
heterogeneous fleet. Two independent workers execute through experimental FNX1
v1 using reference/simulated backends and real TCP framing over loopback under a
lockstep orchestrator. Admission/batching bounds are tested separately with the
same trace/evidence schemas intended for later physical validation.

## Deliverables

| ID | Deliverable | Owner | Definition of done | Status |
|---|---|---|---|---|
| S05-1 | Stage manifest, request/result types, v1 frame codec | RT + NET | Positive/negative conformance corpus passes | Complete — 24 at closure; current corpus 31 |
| S05-2 | `ReferenceStageBackend` | RT + LLM | Deterministic prefill/decode transform, KV epochs, activation/logit oracle | Complete |
| S05-3 | `SimulatedMaxStageBackend` | RT + DIST | Reference-equivalent output plus scenario service/memory/fault injection | Complete |
| S05-4 | Worker process and lifecycle | RT | load/health/execute/cancel/drain/unload state machine passes | Complete |
| S05-5 | Persistent TCP channel | NET | handshake, frame, ACK, credit, heartbeat, error, reconnect tests pass | Complete |
| S05-6 | Engine orchestrator | DIST | One request completes prefill/decode across two workers | Complete — prefill/decode |
| S05-7 | Admission and microbatch scheduler | DIST | Concurrency 1/4/8, bounded queues, timeout/cancel, fairness traces | Complete |
| S05-8 | Planner defect repairs | DIST | I-7 and I-8 reproduced regressions pass | Complete |
| S05-9 | Observability/resource/evidence ledger | SRE + PM | Scenario/SA IDs, spans, queue/bytes/memory, evidence class recorded | Complete |
| S05-10 | Named scenario and fault sweep | WS-J | Required matrix runs deterministically; blockers recorded | Complete — 60 scenarios, seven faults |
| S05-11 | Sustained loopback run | WS-J + SRE | 30 minutes at highest supported scenario concurrency with bounds | Complete — 1,800.010 wall seconds, observed concurrency 8 |
| S05-12 | Physical backend spike | RT + KER + LLM | Opportunistic T2/T3 result or explicit unavailable status; does not block S05-1..11 | Complete by explicit `unavailable` disposition; G2 remains open |

## Critical sequence

```text
S05-1 -> S05-2/S05-3 -> S05-4/S05-5 -> S05-6 -> S05-7 -> S05-10/S05-11
                  S05-8 and S05-9 run in parallel
                  S05-12 runs whenever hardware is available
```

## Required scenario coverage

- Named scenarios from `../../simulation-and-assumption-contract.md`.
- Context 16/128/512/4096 and concurrency 1/4/8, with broader sensitivity where
  runtime permits.
- No fault, slow stage, no credit, timeout, cancellation, stale plan, corruption,
  disconnect, conflicting duplicate, and reconnect.

## Exit criteria

- Two independent loopback worker processes complete one deterministic request
  for prefill and decode through experimental FNX1 v1 under a lockstep
  orchestrator.
- Reference and simulated backends pass the same conformance suite.
- Planner I-7/I-8 regressions pass.
- Queue, byte-credit, resource, cancellation, timeout, and cleanup invariants pass.
- Named scenarios produce deterministic artifacts with SA-* provenance.
- Sustained run stays within configured channel-credit, scheduler-queue, and RSS
  bounds; this does not establish bounded request/KV lifecycle state.
- Every unvalidated physical claim is still mapped to G2 evidence.

## Guardrails

- No new Phase 3-5 proxy features.
- No physical throughput/support claim from simulation.
- No backend-specific bypass around the Stage ABI.
- Remote expert placement remains disabled.
- Hardware unavailability is reported but does not block the engine sequence.

## Closure record

- Review: [Phase 0.5 exit / M1](../gate-reviews/phase-0-5-exit-2026-07-10.md)
- Evidence: [`phase05-engine-v0-2026-07-10.json`](../../evidence/phase05-engine-v0-2026-07-10.json)
- SHA-256: `d9f57d940306568959fd87139c0e95b8dcdd770166eabc2c31e9d425d40d1e37`
- Regression: `make test` — all contract/golden suites and 275 unit tests pass.
- Scope: T0/T1 only; no physical backend, supported-platform, throughput, or G2 claim.
