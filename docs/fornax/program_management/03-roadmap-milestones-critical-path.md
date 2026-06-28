# Roadmap, Milestones & Critical Path

Notional durations (week offsets from kickoff `W0`); absolute dates set at
kickoff and **re-baselined after the Phase-0 sprint sizes the work**. Durations
are T-shirt estimates, not commitments, until G1.

> **Authorization (plan v3 §12):** only **Phase 0** is authorized. M2 onward are
> planning placeholders; they are not funded until G1 passes.

## Milestone schedule (notional)

| Milestone | Phase | Notional window | Gate |
|---|---|---|---|
| M0 Architecture baseline | — | W0 | **G0 ✅** |
| M1 Evidence sprint done → go/no-go | 0 | W1–W4 | **G1** |
| M2 Pipeline correctness (sim then 2–3 nodes) | 1 | W5–W10 | — |
| M3 Continuous batching scales | 2 | W9–W13 | **G2** (with M4) |
| M4 MoE expert runtime parity | 2.5 | W11–W17 | **G2** |
| M5 Heterogeneous frontier serve | 3 | W16–W24 | **G3** |
| M6 Resilience / elasticity | 4 | W23–W27 | **G4** |
| M7 Productization / GA | 5 | W27+ | **G5** |

Phases overlap deliberately (sim tiers start before hardware tiers; WS-D runs in
parallel throughout).

## Critical path

```
G0 ─ A1→A2→A3 (planner) ─ A5 v0-contract ──┐
                                            ├─► G1 ─ B2→B3 runtime ─ F1→F2 batching ─ C1→C4 MoE ─► G3
WS-D Apple readiness (D1→D2→D4) ────────────┘            (parallel, gates G3 via D2/D3)
```

Two things sit on the critical path:

1. **WS-A → `v0-target-contract.md` (+ A6 preflight workflow) → G1.** Nothing
   downstream is funded until the contract closes. This is the shortest path to
   the most important decision.
2. **WS-D Apple/Mac readiness.** A *parallel* critical path: it does not block G1,
   but it gates G3 (heterogeneous frontier) and carries the reversal trigger. It
   must start at W1, not when hardware phases begin — because Apple can become an
   expert worker long before it is wise to place full stages there (plan §5.5).

## Schedule risks (see [05-raid-log.md](05-raid-log.md))

- **D-1 Modular/MAX Apple capability** can move the WS-D path either way — tracked
  dated, [06](06-dependencies-and-external-watch.md).
- **R-8 concurrency–market fit** can force a re-scope at G1, changing everything
  downstream — by design (cheap, early).
- Hardware procurement lead time for `prosumer-rack`/`lab-reference`
  ([10](10-budget-and-procurement.md)) must finish before M2's hardware tier.

## Re-baseline rule

The schedule is **provisional until G1**. The Phase-0 sprint produces the sized
estimates; the PM re-baselines this file at G1 and records the change in the
[decision log](08-decision-log.md).
