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
| S5-1 | Partial: preflight/doctor artifacts exist; polished operator UX remains open. |
| S5-2 | Partial: lifecycle simulation exists; production workflow remains open. |
| S5-3 | Open: real node replacement workflow remains open. |
| S5-4 | Partial: onboarding simulation/methodology exists; product docs remain open. |
| S5-5 | Partial: benchmark discipline is defined; benchmark of record remains open. |
| S5-6 | Open: GA evidence requires completed prior gates. |

## Validation

- `python3 -m fornax test ops-lifecycle`
- `python3 -m fornax test onboarding`
- `python3 -m fornax preflight --help`
- `python3 -m fornax doctor --help`

## Exit Criteria

- A firm can install, operate, upgrade, and serve internal users with Fornax.
- Runbooks and onboarding are repeatable without oral context.
- G5 evidence is packaged for Sponsor decision.
