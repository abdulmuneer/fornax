# Decision Log (DEC-*) & ADR Index

Records irreversible or expensive decisions so they are reviewable after the
fact. Architecture decisions live as ADRs under `../adr/`; this log indexes them
and captures program-level decisions. Use
[templates/decision-record.md](templates/decision-record.md).

## Decision log

| ID | Date | Decision | Status | Reverses? | Ref |
|---|---|---|---|---|---|
| DEC-001 | W0 | Fornax is an **engine** (MAX surgery), not a harness over llama.cpp | Accepted | one-way-ish | plan §5.4 / ADR-0001 |
| DEC-002 | W0 | **Pipeline-parallel** spine; bounded remote experts as measured option; no default all-to-all | Accepted | reversible per-deployment | plan §5.1 |
| DEC-003 | W0 | Apple participation is **staged & gated** with a reversal trigger | Accepted | reversible | plan §5.5 |
| DEC-004 | W0 | Plan changes only by **version bump** (v1→v2→v3…) | Accepted | — | governance |
| DEC-005 | _G1_ | _Go/no-go outcome + re-baselined schedule_ | Pending | — | [04](04-stage-gates.md) |
| DEC-006 | W0 | **Speculative decoding out of v0** unless the target contract opts in | Accepted | reversible (contract opt-in) | plan v3 §3.5 |

## ADR index (`../adr/`)

| ADR | Title | Resolves | Status |
|---|---|---|---|
| 0001 | MAX/Mojo substrate (vs llama.cpp/MLX/vLLM/SGLang/hybrid) + reversal trigger | B5 | to write (Phase 0) |
| 0002 | Pipeline-parallel default | — | backfill from DEC-002 |
| 0003 | Bounded remote-expert execution | — | backfill |
| 0004 | Transport choice | — | Phase 1 |
| 0005 | Security posture | B4 | Phase 0/3 |
| 0006 | Apple participation level | B5 | G1 |
| 0007 | Prefill/decode disaggregation — rejection or deferral | — | backlog (plan v3 §10) |
| 0008 | Homogeneous intra-node tensor-parallel island policy | — | backlog (plan v3 §10) |

## Rules

- A decision that is **expensive to reverse** gets a DEC-\* before work proceeds
  (charter guardrail 3).
- Each gate outcome is a DEC-\*.
- When the plan version changes, **rejected alternatives** named in the new plan
  become ADR stubs here (don't lose them).
