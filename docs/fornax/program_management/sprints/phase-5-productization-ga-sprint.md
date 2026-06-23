# Phase-5 Productization and GA Sprint

**Goal:** make Fornax installable, operable, upgradeable, and explainable for
internal users after the core technical gates are proven.

**Duration:** W27+.
**Milestone:** M7 Productization / GA.
**Gate:** G5 Product GA.

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S5-1 | Operator UX | SRE + API | I1 | `cluster.yaml`, `model.yaml`, `placement.json`, and `fornax doctor` are documented and validated |
| S5-2 | Deploy/upgrade/drain/restart/rollback lifecycle | SRE | I2 | lifecycle workflows are repeatable and audited |
| S5-3 | Node replacement workflow | SRE + NET | I2 | replacement preserves identity, auth, placement, and state cleanup invariants |
| S5-4 | Onboarding tracks and glossary | PM + API | I3 | new operator and developer paths are documented |
| S5-5 | Benchmark methodology of record | QA + PM | I3 | reproducible benchmark commands, traces, versions, and evidence locations exist |
| S5-6 | GA gate pack | PM + TL | G5 | install, operate, upgrade, and serve evidence is ready for Sponsor review |

## Sprint Board

| Deliverable | Status |
|---|---|
| S5-1 | Passed at two-H100 proxy scope: ops lifecycle artifact validates `cluster.yaml`, `model.yaml`, `placement.json`, auth, plan integrity, and `/v1/chat/completions`; polished installable UX remains formal GA work. |
| S5-2 | Passed at two-H100 proxy scope: lifecycle simulation covers deploy, drain, upgrade, restart, rollback, and zero dropped in-flight requests. |
| S5-3 | Passed at two-H100 proxy scope: node replacement is drained, removed, replaced, health-checked, and traffic-restored in the lifecycle artifact. |
| S5-4 | Passed at two-H100 proxy scope: onboarding methodology validates operator, developer, benchmark-owner, reviewer, required docs, and glossary. |
| S5-5 | Passed at two-H100 proxy scope: benchmark methodology and measured ledger fixture validate reproducibility fields and correctness-first boundaries. |
| S5-6 | Passed at two-H100 proxy scope: `phase5-ga-gate` composes lifecycle, onboarding, benchmark ledger, G4 proxy, and G5 runbook into an 11/11 packet with formal G5 deferred. |

## Proxy Gate Evidence

- Gate packet: `/tmp/fornax_phase5_g5_two_h100_proxy_gate_20260623.json`.
- Runbook: [`../g5-productization-runbook.md`](../g5-productization-runbook.md).
- Golden fixture: `fornax/golden_vectors/phase5_ga_gate/fixture.json`.
- Scope: accepted local proxy using `cuda:0`, `cuda:1`, `logical-host-0`, and `logical-host-1` on H100 hardware.
- Boundary: `phase5_proxy_passed=true`, `formal_g5_passed=false`, and `formal_g5_validation_deferred=true`.

## Validation

- `python3 -m fornax test ops-lifecycle`
- `python3 -m fornax test onboarding-methodology`
- `python3 -m fornax test benchmark-ledger`
- `python3 -m fornax test phase4-resilience-gate`
- `python3 -m fornax test phase5-ga-gate`
- `python3 -m fornax program phase5-ga-gate --ops-artifact fornax/golden_vectors/ops_lifecycle --onboarding-artifact fornax/golden_vectors/onboarding_methodology --benchmark-ledger fornax/golden_vectors/benchmark_ledger --phase4-artifact fornax/golden_vectors/phase4_resilience_gate --out /tmp/fornax_phase5_g5_two_h100_proxy_gate_20260623.json --runbook-out /tmp/fornax_phase5_g5_productization_runbook_20260623.md --date 2026-06-23 --outcome PROCEED --accepted-by operator --proxy-devices cuda:0,cuda:1 --proxy-logical-hosts logical-host-0,logical-host-1`
- `python3 -m fornax preflight --help`
- `python3 -m fornax doctor --help`

## Exit Criteria

- Proxy exit: local two-H100 productization package is machine-checkable, documented, and gate-packaged.
- Formal G5 exit: a firm can install, operate, upgrade, and serve internal users with Fornax, and Sponsor accepts the real GA evidence.
- Runbooks and onboarding are repeatable without oral context.
- G5 evidence is packaged for Sponsor decision without overstating proxy evidence as formal GA.
