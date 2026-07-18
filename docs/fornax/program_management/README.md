# Fornax — Program Management

This folder is the **program governance layer** for Fornax. It does not change the
engineering plan; it makes the plan *executable and governable*. The technical
source of truth is [`../project-plan-v4.md`](../project-plan-v4.md) and its
[accepted review disposition](../platform-review-action-plan-2026-07-10.md).

> **V4 authorization (2026-07-10):** G1 `PROCEED` authorizes the Engine v0
> T0/T1 scope using experimental FNX1 v1 and candidate FNX2 2.0,
> reference/simulated backends, multi-process loopback transport, and bounded
> admission/lifecycle state. Candidate FNX2 adds integrated ragged scheduling
> at T1; it is not physical conformance. Physical heterogeneous proof and a
> production-grade lifecycle remain open and gate production claims. Release,
> idle expiry, execution leases, same-worker tombstones, and bounded retention
> are implemented; restart durability, reviewed sustained evidence, and
> physical-backend validation remain open under I-22.
>
> **M1 closure (2026-07-10):** Phase 0.5 is complete at T0/T1. The active lane
> is Phase 1 physical backend integration and G2 evidence acquisition; hardware
> scarcity may delay G2 but does not invalidate the Engine v0 contract baseline.
>
> **Terminology clarification (2026-07-17):** historical “production ABI” wording
> names the intended role. Current maturity is experimental FNX1 v1 at T0/T1;
> candidate FNX2 is also T0/T1-only; see the
> [erratum](../abi-terminology-erratum-2026-07-17.md).
>
> **Proxy-review follow-up (2026-07-17):** the current closure/partial/open
> disposition for the 2026-07-13 platform and developer-experience review is
> recorded [here](../founder-proxy-review-follow-up-2026-07-17.md). It is not
> evidence of participation or endorsement by any named person.
>
> **Planner authority clarification (2026-07-17):** provenance/confidence/error,
> exact capability admission, exploratory labels, and fail-closed deployment
> mode are now implemented. Repository fixtures still lack authenticated
> physical calibration and therefore remain exploratory under I-16; see the
> [planner erratum](../planner-status-erratum-2026-07-17.md).
>
> **G2 packet automation (2026-07-17):** the current working tree contains and
> verifies an uncommitted MAX root-pin/reconstruction mechanism, and
> `fornax program g2-validate` builds a hashed, fail-closed V1-V10 packet. The
> lineage mechanism becomes durable repository state only after commit. Without
> authorized hardware results the packet records `BLOCKED`; see
> [G2-in-a-box](../g2-in-a-box.md).

## Operating principle

Fornax is an **R&D program with a genuine no-go option**, an external critical
dependency (Modular/MAX Apple support), and hardware in the loop. The PM mandate
is therefore: **govern the uncertainty, not the certainty.** Reach decisions
cheaply and early, protect the critical path, and do not let missing hardware stall
implementation when the same contract can be exercised by a deterministic
simulation backend. Assumptions stay visible until measured.

Process is deliberately thin: heavy on **decisions, risks, and gates**; light on
ceremony.

## Where the program is today

- Architecture **re-baselined** at plan v4 (gate **G0 passed** — see
  [04-stage-gates.md](04-stage-gates.md)).
- **G1 passed and Phase 0.5 / M1 is complete at T0/T1 scope.**
- The completed sprint and evidence are recorded in
  [Phase 0.5 Engine v0](sprints/phase-0-5-engine-v0-sprint.md) and its
  [exit review](gate-reviews/phase-0-5-exit-2026-07-10.md).
- The active execution lane is
  [Phase 1 physical validation](sprints/phase-1-worker-contract-transport-sprint.md),
  paced by backend/fleet availability.
- The next formal evidence gate is **G2 physical distributed correctness**.
- A parallel product-discovery lane is active for buyer/workload/concurrency and
  design-partner evidence; it does not authorize serving or performance claims.
- External fundraising materials remain internal and `ITERATE` under
  [external-readiness gates](../fundraising/external-readiness-gates.md); team,
  rights, financing, customer, market, and package claims are not inferred from
  technical progress.
- Later Phase 3-5 proxy scope is frozen; existing fixtures remain regression
  evidence without closing physical/product gates.

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
| [G1 gate review](gate-reviews/g1-2026-07-10.md) | Recorded G1 packet and outcome |
| [Phase 0.5 exit review](gate-reviews/phase-0-5-exit-2026-07-10.md) | M1 closure and exact G2 authorization boundary |
| [sprints/](sprints/) | Completed Phase 0.5 plus active physical-validation and deferred backlog |
| [internal/](internal/) | Release-disposable raw reviews, research inputs, development journal, and archived status |
| [templates/](templates/) | Status report, gate review, decision record templates |

## Release boundary

The engine, technical contracts, ADRs, current plan, and evidence remain outside
`program_management/internal/`. Development clones keep that internal subtree so
program progress moves between machines. Release packaging may omit the entire
subtree without removing runtime code or normative technical contracts.

## Conventions

- **Single source of truth, versioned.** The plan is versioned (v1 → v4); these
  docs reference it, never fork its decisions.
- **IDs are stable.** Workstreams `WS-*`, risks/assumptions/issues/deps `R-/A-/I-/D-*`,
  decisions `DEC-*`, gates `G*`. Other docs cite these IDs.
- **Untracked working notes**, per repo convention (only `docs/extensions.md` is
  tracked) — do not `git add`.
- Dates here are **notional (week offsets from kickoff)**; absolute dates are set
  at kickoff and confirmed after the Phase-0 sprint sizes the work.
