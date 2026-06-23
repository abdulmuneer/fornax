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
| S4-1 | Complete at two-H100 proxy scope: local endpoint failure semantics plus replay simulation are deterministic; real distributed partition proof remains deferred. |
| S4-2 | Complete at two-H100 proxy scope: resilience replay simulation preserves all in-flight requests with zero dropped or duplicate tokens; live replay remains deferred. |
| S4-3 | Complete at two-H100 proxy scope: stage-replication simulation validates added capacity, speedup, and output parity; real added-node scaling remains deferred. |
| S4-4 | Complete at two-H100 proxy scope: ops lifecycle simulation records drain, restart, rollback, and node replacement hooks with zero dropped in-flight work; production hooks remain deferred. |
| S4-5 | Complete at runbook/proxy scope: `docs/fornax/program_management/t4-resilience-runbook.md` and `fornax.phase4_resilience_gate` define T4 node-loss, added-node, lifecycle, and heterogeneous follow-up evidence collection; real T4 execution evidence remains deferred. |

## Validation

- `python3 -m fornax test resilience-replay`
- `python3 -m fornax test stage-replication`
- `python3 -m fornax test ops-lifecycle`
- `python3 -m fornax test phase4-resilience-gate`
- `python3 -m fornax program phase4-resilience-gate --resilience-artifact fornax/golden_vectors/resilience_replay --replication-artifact fornax/golden_vectors/stage_replication --ops-artifact fornax/golden_vectors/ops_lifecycle --out /tmp/fornax_phase4_g4_two_h100_proxy_gate_20260623.json --runbook-out /tmp/fornax_phase4_t4_resilience_runbook_20260623.md --date 2026-06-23 --outcome PROCEED --accepted-by operator`
- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`

## Exit Criteria

- Simulated single-node loss and replay behavior are deterministic.
- Added-node scaling assumptions are testable and preserve correctness.
- Drain, restart, rollback, and node replacement hooks are auditable.
- Current two-H100 proxy gate passes with `phase4_proxy_passed=true` and
  `formal_g4_passed=false`.
- Formal T4/G4 remains deferred until real heterogeneous lab validation verifies
  zero dropped in-flight requests on real single-node loss and real added-node
  scaling.
