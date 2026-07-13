# Metrics and Reporting

## Engine v0 metrics

| Metric | Target/rule | Evidence class |
|---|---|---|
| Executable path | One request crosses two independent workers through production Stage ABI | T1 |
| Contract correctness | Reference/simulated outputs and ownership events match | T0/T1 |
| Scenario coverage | Every named compute/link/fault scenario runs or has a recorded blocker | T1 |
| Assumption traceability | 100% injected hardware parameters cite SA-* and scenario IDs | T1 |
| Queue/memory bounds | No configured limit exceeded or unbounded growth in sustained run | T1 |
| Failure semantics | Cancel/timeout/stale/corrupt/disconnect outcomes match contract | T1 |
| Planner regressions | Both reproduced false-feasible/false-infeasible cases fixed | T0 |

Simulation throughput is useful for sensitivity and scheduler regression only. It
is not reported as hardware tokens/s.

### Phase 0.5 closure snapshot

| Metric | Recorded result | Artifact |
|---|---:|---|
| Stage conformance | 24/24 | EV-008 / EV-009 |
| Exit checks | 12/12 | EV-009 |
| Scenario/fault/scheduler rows | 60 / 7 / 3 | EV-009 |
| Sustained wall time | 1,800.010 seconds | EV-009 |
| Real loopback requests / observed concurrency | 14,304 / 8 | EV-009 |
| Message credit min/max/configured | 1 / 1 / 1 | EV-009 |
| Byte credit min/max/configured | 268,435,456 / 268,435,456 / 268,435,456 | EV-009 |
| Peak process RSS / bound | 124,043,264 / 1,073,741,824 bytes | EV-009 |
| Full unit tests | 275 passed | EV-001 |

These are local T0/T1 measurements and counts, not hardware throughput or G2
evidence.

## Physical/product metrics

| Metric | Provisional target | First authoritative tier |
|---|---|---|
| Backend correctness | Per-dtype operator/stage/logit tolerance | T2 |
| Physical distributed correctness | Prefill/decode across 2–3 nodes | T3 |
| Planner accuracy | +/-20% G2, +/-10% G3 | T3/T4 |
| Capacity | Model exceeds one node and fleet closes with >=10% headroom/node | T4 |
| Throughput efficiency | >=60% of contract-defined sum-of-node ideal | T4 |
| Concurrency | Saturation within persona supply | T3/T4 |
| Apple role | Highest role proven by correctness/stability/throughput | T2/T3 |
| Resilience | G4 target-contract request-loss rule | T4 |
| Operator success | Fresh install/operate/upgrade/rollback | T5 |

## Program-health metrics

| Metric | Cadence |
|---|---|
| Engine v0 critical-path deliverables complete/blocked | Weekly |
| SA-* validated, invalidated, and open by blocked gate | Weekly |
| Named scenarios passing | Weekly |
| Planner defect regressions | Every change/weekly summary |
| Stage ABI conformance cases | Every change/weekly summary |
| Available/scheduled physical validation windows | Weekly |
| Top risks R-11/R-12/R-13 trend | Weekly |
| Evidence by class and freshness | Weekly + gate |

## Reporting

- Weekly status uses `templates/status-report.md`.
- Gate packets use `templates/gate-review.md`.
- Evidence details live in `../evidence-register.md`.
- Assumption details live in `../simulation-and-assumption-contract.md` and RAID.

## Honesty guardrail

Every number carries `measurement_kind` (`reference`, `simulation`,
`same-host-proxy`, or `physical`), environment/scenario identity, and source
artifact. A number without provenance is a reportable issue, not a metric.
