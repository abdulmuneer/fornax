# Stage-Gate Governance

Fornax runs as a **stage-gated program with a genuine kill/narrow option.** Each
gate has explicit entry/exit criteria, a single decision authority (Sponsor,
advised by PM + TL), and four possible outcomes.

## Gate outcomes (every gate)

- **PROCEED** — fund the next phase.
- **ITERATE** — stay in phase; specific criteria not yet met.
- **NARROW** — re-scope the thesis (e.g. capacity-only, homogeneous-island) and
  proceed on the smaller scope. *A valid success.*
- **KILL** — stop; capture the learning. *A valid program outcome (charter).*

## Gate definitions

### G0 — Architecture baseline ✅ (passed)
- **Entry:** plan exists and has been reviewed.
- **Exit:** plan v3 accepted (supersedes v2); reconciled review's preserved
  decisions intact (plan v3 §11). **Status: passed at M0.**

### G1 — Evidence / Go-No-Go ★ (next, the critical gate)
- **Entry:** Phase-0 evidence sprint complete
  ([sprint plan](sprints/phase-0-evidence-sprint.md)).
- **Exit (all required) — each artifact "closed" only per the plan v3 §10
  closure rules, not merely written:**
  - `v0-target-contract.md` reviewed + **signed off by TL/SP**; **memory budget
    closes with headroom**; **predicted throughput meets §8 bar** at the
    contracted concurrency; includes seed acceptance/replacement rationale (§3.2).
  - **Concurrency–market evidence** (B2): the persona supplies the minimum
    saturation concurrency (seed: ≤ 32) — or the scope is narrowed.
  - **Apple reversal trigger** (B5) evaluated **from the rank-1 local probe**
    ([06](06-dependencies-and-external-watch.md)): Apple's v0 role decided
    (compute / expert-host / capacity-only).
  - `runtime-format-and-invariants.md` (golden-vector method + failure modes
    reviewable), `networking-security-and-backpressure.md` (phase spec vs impl
    explicit), `adr/0001-max-mojo-substrate.md` (source precedence + pinned-build
    policy + rejected alts + reversal trigger) — reviewed.
  - **Phase-0 preflight workflow** (§3.4) runnable without oral context.
  - Golden-plan tests (T0) green.
  - **Owners/staffing closed:** the §0 roles have named assignees **or** the
    Sponsor explicitly accepts a narrowed scope.
- **Decision:** Sponsor. PROCEED → Phase 1 · ITERATE · NARROW · KILL.
  **If a required role is unstaffed at G1, silent PROCEED is forbidden** — the
  Sponsor must choose ITERATE/NARROW/KILL (plan v3 §0).
- **This is the gate the whole Phase-0 program exists to reach.**

### G2 — Distributed correctness
- **Entry:** Phases 1, 2, 2.5 complete.
- **Exit:** correct generation across a real 2–3 node pipeline; aggregate tok/s
  scales with concurrency and **matches planner within the provisional bound**;
  MoE layer/logit parity vs the reference path; format spec (§5.6) honored.

### G3 — Heterogeneous frontier
- **Entry:** Phase 3 complete.
- **Exit:** a real frontier MoE served across NVIDIA/AMD + Mac at **predicted
  throughput**; Apple at its gated role; security + backpressure active.

### G4 — Resilience
- **Entry:** Phase 4 complete.
- **Exit:** throughput scales with added nodes; **zero dropped in-flight requests**
  on single-node loss (replay verified).

### G5 — Product GA
- **Entry:** Phase 5 complete.
- **Exit:** a firm can install, operate, upgrade, and serve internal users; ops
  lifecycle + onboarding in place.

## Gate calendar (notional)

| Gate | Notional | Authority |
|---|---|---|
| G0 | W0 ✅ | Sponsor |
| G1 | ~W4 | Sponsor (+ PM, TL) |
| G2 | ~W17 | Sponsor (+ TL) |
| G3 | ~W24 | Sponsor |
| G4 | ~W27 | Sponsor |
| G5 | W27+ | Sponsor |

Run each gate with [templates/gate-review.md](templates/gate-review.md); record
the outcome in the [decision log](08-decision-log.md).
