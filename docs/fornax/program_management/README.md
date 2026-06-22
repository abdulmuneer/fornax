# Fornax — Program Management

This folder is the **program governance layer** for Fornax. It does not change the
engineering plan; it makes the plan *executable and governable*. The technical
source of truth is [`../project-plan-v3.md`](../project-plan-v3.md) and its
review chain ([reconciled](../fornax_plan_review_reconciled.md),
[v2-codex](../fornax_plan_v2_review_codex.md)).

> **Original authorization (plan v3 §12):** approved as a **Phase-0 evidence sprint only**.
> Phase-1 distributed-runtime engineering was **blocked until G1 passes** under the
> original gate posture.
>
> **Execution update (2026-06-22):** implementation may proceed through the full
> sprint backlog using simulation and local two-GPU logical-host validation. Gate
> closure still requires the real evidence listed in [04-stage-gates.md](04-stage-gates.md).

## Operating principle

Fornax is an **R&D program with a genuine no-go option**, an external critical
dependency (Modular/MAX Apple support), and hardware in the loop. The PM mandate
is therefore: **govern the uncertainty, not the certainty.** Reach decisions
cheaply and early, protect the critical path, and do not let the team invest in
Phase-1 engineering before the Phase-0 evidence gate (G1) clears.

Process is deliberately thin: heavy on **decisions, risks, and gates**; light on
ceremony.

## Where the program is today

- Architecture **baselined** at plan v3 (gate **G0 passed** — see
  [04-stage-gates.md](04-stage-gates.md)).
- Next formal gate is **G1 — Evidence / Go-No-Go**, fed by the
  [Phase-0 evidence sprint](sprints/phase-0-evidence-sprint.md).
- The [sprint backlog](sprints/) now materializes all phases so engineering can
  continue under simulation/local validation without claiming later gates are
  closed.

## Document tree

| Doc | Purpose |
|---|---|
| [00-charter.md](00-charter.md) | Why the program exists, objectives, scope, sponsor, guardrails |
| [01-stakeholders-and-raci.md](01-stakeholders-and-raci.md) | Workstreams, owners, RACI |
| [02-work-breakdown-structure.md](02-work-breakdown-structure.md) | WBS: workstreams → epics → deliverables |
| [03-roadmap-milestones-critical-path.md](03-roadmap-milestones-critical-path.md) | Timeline, milestones, critical path |
| [04-stage-gates.md](04-stage-gates.md) | Gate governance: entry/exit, decision authority, go/no-go |
| [05-raid-log.md](05-raid-log.md) | Risks, Assumptions, Issues, Dependencies (living) |
| [06-dependencies-and-external-watch.md](06-dependencies-and-external-watch.md) | Cross-workstream + the Modular/MAX external watch |
| [07-resourcing-and-skills.md](07-resourcing-and-skills.md) | Staffing, skills matrix, gaps |
| [08-decision-log.md](08-decision-log.md) | Decision / ADR index |
| [09-communications-and-cadence.md](09-communications-and-cadence.md) | Rituals, status, escalation |
| [10-budget-and-procurement.md](10-budget-and-procurement.md) | Hardware bundles, headcount, procurement |
| [11-quality-and-test-governance.md](11-quality-and-test-governance.md) | T0–T4 tiers, definition of done, correctness governance |
| [12-metrics-and-reporting.md](12-metrics-and-reporting.md) | KPIs, program-health metrics |
| [sprints/](sprints/) | Sprint backlog for all roadmap phases; Phase 0 was the original active sprint |
| [templates/](templates/) | Status report, gate review, decision record templates |

## Conventions

- **Single source of truth, versioned.** The plan is versioned (v1 → v2); these
  docs reference it, never fork its decisions.
- **IDs are stable.** Workstreams `WS-*`, risks/assumptions/issues/deps `R-/A-/I-/D-*`,
  decisions `DEC-*`, gates `G*`. Other docs cite these IDs.
- **Untracked working notes**, per repo convention (only `docs/extensions.md` is
  tracked) — do not `git add`.
- Dates here are **notional (week offsets from kickoff)**; absolute dates are set
  at kickoff and confirmed after the Phase-0 sprint sizes the work.
