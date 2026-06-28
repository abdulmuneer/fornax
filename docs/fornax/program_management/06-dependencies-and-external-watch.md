# Dependencies & External Watch

## The one external dependency that can sink the program

**D-1 — Modular / MAX Apple + MoE capability.** Fornax is a *surgery of MAX*; the
Apple/Mac role (WS-D) depends on MAX capabilities Modular ships on its own
schedule. This is the program's largest exogenous risk (R-4) and is tracked
**dated and passively** — we watch and adapt, we do not block on it.

### Source precedence ladder (plan v3 §5.4) — the gate of record

V3 makes capability adjudication explicit: when sources disagree, **capability is
unproven until the local probe passes.** Higher rank wins.

| Rank | Source | Authority |
|---|---|---|
| 1 | **Local probe on the pinned build in the target env** | **gate of record** for Fornax role assignment |
| 2 | Package docs + changelog for the pinned build | official support status |
| 3 | Supported-model catalog / model docs | model-level availability |
| 4 | Blog posts / launch announcements | directional signal only — **never a release gate** |
| 5 | Nightly behavior | unblocks only after pinned, probed, recorded; future promises never unblock |

> Live example v3 calls out: the **26.4 blog** announced expanded Apple Silicon
> MAX model support, while **package docs** still caution that large GenAI model
> inference via MAX is not yet available on Apple Silicon. Per the ladder, the
> **local expert-MLP probe decides** Apple's v0 role — not the blog.

### Watch register (update each MAX nightly / release)

| Field | Value |
|---|---|
| Pinned MAX build | _set in `adr/0001-max-mojo-substrate.md`_ |
| Capability needed (v0) | target-model **expert-MLP** on the target Mac, within tolerance/throughput bound |
| Adjudicated by | **rank-1 local probe** (ladder above) |
| Last checked | _date_ |
| Status | _probing / partial / sufficient / regressed_ |
| Reversal trigger armed? | yes — if not sufficient by end of Phase 0 → Apple demoted to capacity-only (plan §5.5) |
| Owner | KER |

Upstream anchors (plan §5.5):
- 25.6 Apple GPU direction · 26.4 MoE + Apple note · MAX package caveats · MAX
  custom ops (URLs in plan §5.5). All are **rank-3/4** signals — they inform, they
  do not gate.

**Policy:** the program never assumes a future MAX capability. It commits only to
the Apple role the *currently measured* build supports, and re-checks per nightly
(R-4 mitigation). Capability changes are logged here and, if they change a
decision, recorded as a `DEC-*` ([08](08-decision-log.md)).

## Internal dependencies

| Dep | Blocks | Owner | Note |
|---|---|---|---|
| D-4 WS-A planner | every downstream phase | DIST | Critical path; Phase 0 |
| D-2 Hardware procurement | M2 hardware tier, G3 | PM | Lead time — [10](10-budget-and-procurement.md) |
| D-3 Ignis `Engine` seam | WS-H integration | TL | Keep `generate(...)` stable |
| WS-B format spec (B1) | WS-C, WS-E, WS-F | RT | The load-bearing invariant |

## Dependency-driven sequencing rules

1. **Planner before pipeline.** No worker/transport build (WS-B/E) competes with
   WS-A for Phase-0 attention; the planner gates G1.
2. **Format spec before consumers.** `runtime-format-and-invariants.md` (B1) lands
   before WS-C/E/F build against it.
3. **Apple in parallel, gated.** WS-D starts W1, stays at its gated role.
4. **Procure early.** Start hardware procurement at kickoff so the
   `prosumer-rack`/`lab-reference` bundles exist before M2.
