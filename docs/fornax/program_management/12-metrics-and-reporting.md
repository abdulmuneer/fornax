# Metrics and Reporting

## Engine v0 metrics

| Metric | Target/rule | Evidence class |
|---|---|---|
| Executable path | One request crosses two independent loopback workers through experimental FNX1 v1 under a lockstep orchestrator | T1 |
| Contract correctness | Reference/simulated outputs and ownership events match | T0/T1 |
| Scenario coverage | Every named compute/link/fault scenario runs or has a recorded blocker | T1 |
| Assumption traceability | 100% injected hardware parameters cite SA-* and scenario IDs | T1 |
| Scheduler/channel soak bounds | No configured queue, credit, or process-RSS limit exceeded in the recorded sustained run | T1 |
| Candidate FNX2 ragged correctness | Unequal prefill, independent decode, compacted results, per-sequence errors/KV, replay, cancellation/release, and two independent loopback workers match the frozen golden corpus | T0/T1 only; physical MAX conformance open |
| Request/KV lifecycle bounds | Explicit release, idle expiry, execution leases, same-worker tombstones, and configured count/time/byte/event limits | Implemented/tested at T0/T1; restart durability, physical validation, and reviewed long-duration current evidence remain open before a production memory claim |
| Planner authority | Exploratory plans labeled; deployment mode requires exact capabilities plus provenance/confidence/error and fails closed | T0 schema/admission only; source authenticity and physical calibration remain G2 evidence |
| Failure semantics | Cancel/timeout/stale/corrupt/disconnect outcomes match contract | T1 |
| Planner regressions | Both reproduced false-feasible/false-infeasible cases fixed | T0 |

Simulation throughput is useful for sensitivity and scheduler regression only. It
is not reported as hardware tokens/s.

### Phase 0.5 closure snapshot

| Metric | Recorded result | Artifact |
|---|---:|---|
| Stage conformance at historical closure | 24/24 | EV-009 |
| Current Stage ABI v1 conformance | 31/31 | EV-008; working-tree T0 suite |
| Exit checks | 12/12 | EV-009 |
| Scenario/fault/scheduler rows | 60 / 7 / 3 | EV-009 |
| Sustained wall time | 1,800.010 seconds | EV-009 |
| Real loopback requests / observed concurrency | 14,304 / 8 | EV-009 |
| Message credit min/max/configured | 1 / 1 / 1 | EV-009 |
| Byte credit min/max/configured | 268,435,456 / 268,435,456 / 268,435,456 | EV-009 |
| Peak process RSS / bound | 124,043,264 / 1,073,741,824 bytes | EV-009 |
| Full unit tests at historical closure | 275 passed | EV-001 |
| Current full unit tests | 386 passed (2026-07-18 socket-enabled working-tree run) | T0/T1 verification; not physical evidence |
| Lifecycle pressure candidate | 113,718 requests / 1,800.004556833 monotonic seconds; bounds pass; source authority false; continuity not established | EV-016; T0/T1 reference only |

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

## Customer and pilot metrics

These are evidence fields, not current results or GA gates. Discovery may run in
parallel with G2 while all product capability claims remain gated.

| Metric | Evidence required |
|---|---|
| Qualified interviews | Interview record with buyer, workload, estate, security, budget owner, and buying path |
| Workload/concurrency fit | Anonymized arrival/concurrency/context/output trace and measured saturation comparison |
| Design-partner pull | Written commitment with target pilot and success definition |
| Commercial proof | Paid pilot; general interest is not purchase evidence |
| Time to first success | Clean-operator elapsed time and intervention log |
| Delivery/support burden | Deployment, model-enablement, incident, and support hours |
| Buyer economics | Fleet/fabric/power/labor/support TCO and cost per useful output token |
| Service repeatability | Reusable profiles, automation, conformance, certified configs, or upstreamable work created |

## Program-health metrics

| Metric | Cadence |
|---|---|
| Engine v0 critical-path deliverables complete/blocked | Weekly |
| SA-* validated, invalidated, and open by blocked gate | Weekly |
| Named scenarios passing | Weekly |
| Planner defect regressions | Every change/weekly summary |
| Stage ABI conformance cases | Every change/weekly summary |
| Available/scheduled physical validation windows | Weekly |
| Top risks R-11–R-18 trend | Weekly |
| Customer discovery/traces/design-partner evidence | Weekly, with privacy-safe summary |
| External readiness XG-1…XG-8 and disclosure-owner status | Weekly while fundraising materials are active |
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
