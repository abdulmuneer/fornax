# Metrics & Reporting

Two metric families: **product metrics** (does the engine meet its targets) and
**program-health metrics** (is the program on track to decide/deliver).

## Product metrics (from plan §8 — provisional until v0-contract binds them)

| Metric | Target (provisional) | Source tier |
|---|---|---|
| Capacity (N× largest node) | N ≥ 2× (v0 proof), fits fleet w/ memory margin | T4 |
| Throughput efficiency | ≥ 60% of sum-of-nodes ideal | T3/T4 |
| Planner accuracy | ±20% → ±10% by Phase 3 | T3 |
| Concurrency efficiency | saturates ≤ contracted min concurrency (seed ≤ 32) | T3 |
| Expert locality | remote-expert wait < SLO | T4 |
| Correctness | logit divergence < per-dtype tol at every seam | T2–T4 |
| Cost | $/token, $/capacity vs 8×H100 | T4 + contract |
| Elasticity | 0 dropped in-flight on node loss | T4 |
| Honesty | every metric traceable to a measurement | all |

## Program-health metrics

| Metric | Why | Cadence |
|---|---|---|
| Gate posture (current gate, days-in-gate) | detect stalls | weekly |
| Milestone burn vs notional schedule | drift | weekly |
| Top risk score trend (R-4, R-8) | the two kill-risks | weekly |
| Planned-vs-proven status (R-10) | guard against status drift; every artifact's gate status is current | weekly + gate |
| Open issues (I-\*) vs gate | readiness for next gate | weekly |
| External-watch status (D-1) | the exogenous dependency | per nightly |
| Assumption validation rate (A-\*) | turning assumptions into facts | weekly |

## Reporting

- **Weekly:** [templates/status-report.md](templates/status-report.md) → Sponsor.
- **Per gate:** [templates/gate-review.md](templates/gate-review.md) → DEC-\*.
- **Dashboards** are deferred; until then the weekly report is the dashboard.

## Honesty guardrail

The plan's "no fabricated metrics" invariant is a **program rule**: a reported
number with no measurement behind it is a reportable issue (I-\*), not a rounding
choice. Provisional/estimated numbers are labeled as such.
