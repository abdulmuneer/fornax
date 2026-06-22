# Phase-4 Resilience and Elasticity Sprint

**Goal:** prove the runtime can handle backpressure, replay, and node changes in
simulation before collecting T4 failure evidence.

**Duration:** notional W23-W27.
**Milestone:** M6 Resilience / elasticity.
**Gate:** G4 Resilience.

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S4-1 | Timeout/retry/cancel/partition semantics | NET + SRE | E4 | simulated failures produce deterministic request outcomes |
| S4-2 | In-flight replay model | RT + SRE | replay | replay preserves request state and KV ownership where legal |
| S4-3 | Added-node scaling model | DIST | F3 | new replicated capacity improves modeled throughput without correctness regression |
| S4-4 | Drain/restart rollback hooks | SRE + API | I2 input | lifecycle events are recorded and auditable |
| S4-5 | T4 resilience runbook | SRE + PM | G4 input | real lab runbook defines node-loss and added-node evidence collection |

## Sprint Board

| Deliverable | Status |
|---|---|
| S4-1 | Partial: failure/backpressure simulation exists; real partition proof remains open. |
| S4-2 | Partial: resilience replay simulation exists; live replay remains open. |
| S4-3 | Partial: stage-replication simulation exists; real added-node scaling remains open. |
| S4-4 | Partial: ops lifecycle simulation exists; production hooks remain open. |
| S4-5 | Open: real T4 runbook and execution evidence remain to be collected. |

## Validation

- `python3 -m fornax test resilience-replay`
- `python3 -m fornax test stage-replication`
- `python3 -m fornax test ops-lifecycle`
- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`

## Exit Criteria

- Simulated single-node loss and replay behavior are deterministic.
- Added-node scaling assumptions are testable.
- G4 remains open until T4 verifies zero dropped in-flight requests on real
  single-node loss.
