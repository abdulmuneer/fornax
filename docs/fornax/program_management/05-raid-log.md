# RAID Log — Risks, Assumptions, Issues, Dependencies

Living register. Reviewed at the weekly ([09](09-communications-and-cadence.md))
and at every gate. IDs are **stable**; entries are added/updated/retired, never
renumbered. Seeded from plan v3 §7 (ranked risks) and §10 (artifacts).

## Risks (R-*)

Ranked per plan v3 §7 (rank 1 = highest). Ordered by rank.

| Rank | ID | Risk | Owner | P | I | Mitigation / trigger | Status |
|---|---|---|---|---|---|---|---|
| 1 | R-4 | **Apple/MAX readiness misses target role (critical path)** | KER | H | H | Source precedence (§5.4); pinned probes; staged roles; demotion trigger (§5.5) | Open |
| 2 | R-8 | **Concurrency–market fit** | PM | M | H | §3.3 persona + §3.2 seed sweep; NARROW at G1 | Open — resolved at G1 |
| 3 | R-5 | Remote-expert wait dominates decode | DIST | M | H | Budget remote hits; migrate hot; disable remote if unprofitable; calibrate (§5.10) | Open |
| 4 | R-3 | Heterogeneous numerics divergence | LLM | M | H | Format spec + reference path + golden vectors + per-dtype tol (§5.6) | Open |
| 5 | R-1 | Commodity network caps throughput | TL | M | H | Fabric tiers; measured links; pipeline sizing; replication | Open |
| 6 | R-6 | Surgery vs fast-moving MAX internals | RT | M | M | Pin build; thin seam; ADR source watch (§5.4) | Open |
| 7 | R-7 | Mojo toolchain maturity | RT | M | M | Lean on MAX kernels; minimal custom; fallback tests | Open |
| 8 | R-2 | Pipeline depth ↔ latency | DIST | H | M | Honest positioning; depth penalty in planner | Open |
| 9 | R-9 | **Security/backpressure slips past prototype** | NET | L(gated) | H | Spec before Phase 1a; impl before Phase 3/product (§5.8) | Open |
| 10 | R-10 | **Status drift: planned artifacts look proven** | PM | M | M | §0 gate-status table; owner/checklist per artifact; honesty rule (§12 metrics) | Open |

P/I = probability/impact (L/M/H). The two top risks (R-4, R-8) are the ones most
likely to force a NARROW/KILL at G1 — by design, confronted first. **R-10 is new
in v3**: it guards against the exact failure the v2 review caught (naming a
blocker ≠ resolving it).

## Assumptions (A-*) — each has a validation owner & method

| ID | Assumption | Validates via | Owner | Status |
|---|---|---|---|---|
| A-1 | The persona supplies pipeline-filling concurrency (seed goal: saturate ≤ 32 in-flight, §3.2) | B2 sweep in v0-contract | PM | Unvalidated → G1 |
| A-2 | MAX can run the target expert-MLP on the target Mac acceptably | D2 probe | KER | Unvalidated → G1 |
| A-3 | The named fabric tiers (§4) are procurable/available | procurement ([10](10-budget-and-procurement.md)) | PM | Unvalidated |
| A-4 | One quant format is byte-compatible across MAX backends | B1 format spec | RT | Unvalidated → G1 |
| A-5 | Required skills (esp. KER/Apple) are staffable | [07](07-resourcing-and-skills.md) | PM | Unvalidated |
| A-6 | Partitioner cost model predicts within the provisional bound | calibration (§5.10) | DIST | Unvalidated → G2 |

## Issues (I-*) — known gaps to close now

| ID | Issue | Owner | Due | Status |
|---|---|---|---|---|
| I-1 | `v0-target-contract.md` not written | DIST/PM | G1 | Open |
| I-2 | `runtime-format-and-invariants.md` not written | RT | G1 | Open |
| I-3 | `networking-security-and-backpressure.md` not written | NET | G1 | Open |
| I-4 | `adr/0001-max-mojo-substrate.md` not written | TL | G1 | Open |
| I-5 | KER/Apple staffing gap unresolved | PM | W2 | Open |
| I-6 | Phase-0 preflight workflow not written (§3.4) | DIST/SRE | G1 | Open |

## Dependencies (D-*)

| ID | Dependency | Type | Owner | Detail |
|---|---|---|---|---|
| D-1 | Modular/MAX Apple + MoE capability | **External** | KER | [06](06-dependencies-and-external-watch.md) |
| D-2 | Hardware procurement (bundles) | Internal/vendor | PM | [10](10-budget-and-procurement.md) |
| D-3 | Ignis `Engine`-trait seam stability | Internal | TL | §9 plan |
| D-4 | WS-A planner → all downstream phases | Internal | DIST | critical path |

> Refresh rule: when the plan version changes, re-seed R-* from §7 and I-* from
> §10, **preserving existing IDs and human-added entries** (see the
> `fornax-program-manager` skill).
