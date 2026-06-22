# Phase-0 Evidence Sprint

**Goal:** produce the evidence that lets the Sponsor make the **G1 go/no-go**
([04](../04-stage-gates.md)) — cheaply, before any Phase-1 spend. This sprint is
the program's current focus.

**Duration:** notional W1–W4 (re-baseline after sizing).
**Exit = G1 entry criteria met.**

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S0-1 | Partitioner + cost model + golden plans (T0) | DIST | A1–A4 | tests green; monotonicity holds |
| S0-2 | **`v0-target-contract.md`** (model + fleet + memory budget + throughput + baselines + thresholds) | DIST + PM | B1*, P1 | budget closes; throughput ≥ provisional bar at contracted concurrency |
| S0-3 | Concurrency sweep (in the contract) | DIST | B2 | min concurrency stated; persona can supply it (or NARROW) |
| S0-4 | `runtime-format-and-invariants.md` | RT | B3 | activation/KV/expert-batch format + ownership + golden-vector method |
| S0-5 | `networking-security-and-backpressure.md` | NET | B4 | v0 trust boundary + backpressure + failure semantics |
| S0-6 | `adr/0001-max-mojo-substrate.md` | TL | B5 | rationale + rejected alts + dated capability + reversal trigger |
| S0-7 | **Apple expert-MLP probe** (D2) on `desktop-minimal` | KER | A-2/R-4 | measured on **pinned build** (rank-1, §5.4); passes tolerance/throughput **or** Apple demotes to capacity-only |
| S0-8 | Roadmap re-baseline + staffing answer (I-5) | PM | A-5 | sized schedule; KER staffing decision (or Sponsor accepts narrowed scope) |
| S0-9 | **Phase-0 preflight workflow** (§3.4) | DIST + SRE | I-6 | inventory / fabric-probe / target-validate / plan / simulate / benchmark / doctor runnable without oral context |

(*B1's quantitative proof = S0-1 cost model applied in S0-2; S0-2 starts from the
§3.2 seed target and records acceptance or replacement rationale.)

## Sprint Board

| Deliverable | Status |
|---|---|
| S0-1 | Implemented at T0 scope; golden-plan gate evidence exists. |
| S0-2 | Partial: draft/validator/tooling exists; TL/SP sign-off remains open. |
| S0-3 | Partial: throughput simulation exists; persona-supplied concurrency evidence remains open. |
| S0-4 | Partial: runtime-format validator and golden vectors exist; formal review remains open. |
| S0-5 | Partial: network/security/backpressure contract exists; formal review remains open. |
| S0-6 | Partial: substrate ADR draft tooling exists; accepted ADR remains open. |
| S0-7 | Open for real Apple target probe; do not close with NVIDIA-only simulation. |
| S0-8 | Partial: rebaseline tooling exists; staffing/Sponsor scope answer remains open. |
| S0-9 | Implemented at workflow scope; gate closure still depends on reviewed artifacts. |

Detailed item status lives in
[`../../../../fornax_program_management_todo_status.md`](../../../../fornax_program_management_todo_status.md).

## The decision this sprint feeds (G1)

The sprint exists to answer four questions; the Sponsor decides on them:

1. **Does it close?** Memory budget + predicted throughput meet the bar. (S0-1/2)
2. **Is there a market?** Concurrency the persona supplies fills the pipeline. (S0-3)
3. **Can Apple play?** Expert-MLP probe → compute / expert-host / capacity-only. (S0-7)
4. **Is the substrate bet sound, with a way out?** (S0-6)

Outcome ∈ {PROCEED, ITERATE, NARROW, KILL} → recorded as **DEC-005**.

## Guardrails

- Original gate guardrail: no Phase-1 engineering (worker/transport/scheduler
  build) during this sprint because it competes with evidence work and pre-empts
  the gate. Current execution allows simulation/local logical-host implementation
  to continue, but G1/G2/G3 closure still requires the reviewed evidence in
  [04-stage-gates.md](../04-stage-gates.md).
- Hardware: only `desktop-minimal` needed; place `prosumer-rack`/`lab-reference`
  orders now for lead time ([10](../10-budget-and-procurement.md)).
