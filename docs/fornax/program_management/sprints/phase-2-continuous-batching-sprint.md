# Phase-2 Continuous Batching Sprint

**Goal:** prove the scheduler can admit, batch, overlap, and explain work under
synthetic load before hardware scale-out evidence is available.

**Duration:** notional W9-W13.
**Milestone:** M3 Continuous batching scales.
**Gate contribution:** feeds G2 with scheduler and telemetry evidence.

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S2-1 | Microbatch scheduler + admission policy | DIST | F1 | deterministic admission, batching, cancellation, and fairness tests |
| S2-2 | 1F1B overlap simulation | DIST + RT | F2 | bubble-fraction telemetry improves with legal overlap |
| S2-3 | Request/plan/span trace propagation | SRE + DIST | G1 | request IDs, plan IDs, stage timings, and router/expert traces share one trace ledger |
| S2-4 | Queue, backpressure, and memory/KV metrics | SRE + NET | G2 | metric ledger validates queue depth, backpressure state, KV pressure, and timing |
| S2-5 | Continuous-batching T1 bundle | QA | F1-F2, G1-G2 | synthetic traces pass scheduler, telemetry, and observability checks |

## Sprint Board

| Deliverable | Status |
|---|---|
| S2-1 | Partial: T1 scheduler contract exists; production runtime integration remains open. |
| S2-2 | Partial: continuous-batching simulation exists; real overlap telemetry remains open. |
| S2-3 | Partial: observability, metrics-ledger, and trace-ledger fixtures exist; live runtime telemetry remains open. |
| S2-4 | Partial: metrics ledger exists; live exporter/dashboard evidence remains open. |
| S2-5 | Partial: T1 bundle runs; real scale evidence remains G2 work. |

## Validation

- `python3 -m fornax test scheduler-contract`
- `python3 -m fornax test continuous-batching`
- `python3 -m fornax test metrics-ledger`
- `python3 -m fornax test trace-ledger`
- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`

## Exit Criteria

- Scheduler behavior is deterministic and replayable under synthetic traces.
- Telemetry can explain queueing, backpressure, KV pressure, and stage timing.
- G2 gap is limited to real 2-3 node throughput, planner accuracy, and scale
  evidence.
