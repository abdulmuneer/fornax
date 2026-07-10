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
| DEC-005 | 2026-07-10 | G1 `PROCEED` to bounded assumption-driven Engine v0; physical proof remains G2 | Accepted | reversible at G2 | [gate review](gate-reviews/g1-2026-07-10.md) |
| DEC-006 | W0 | **Speculative decoding out of v0** unless the target contract opts in | Accepted | reversible (contract opt-in) | preserved by plan v4 |
| DEC-007 | 2026-07-10 | Approve plan v4, production Stage ABI, simulated/reference backends, and freeze new Phase 3-5 proxy scope | Accepted | reversible by plan version | plan v4 / review disposition |
| DEC-008 | 2026-07-10 | Close Phase 0.5 / M1 at T0/T1 and proceed to Phase 1 physical validation; G2 and all physical claims remain open | Recorded | reversible if T1 evidence regresses | [Phase 0.5 exit review](gate-reviews/phase-0-5-exit-2026-07-10.md) |

## ADR index (`../adr/`)

| ADR | Title | Resolves | Status |
|---|---|---|---|
| 0001 | MAX/Mojo substrate and managed fork policy | B5 / I-11 | Accepted 2026-07-10 |
| 0002 | Pipeline-parallel spanning spine | DEC-002 | Accepted 2026-07-10 |
| 0003 | Remote experts deferred measured optionality | R-5 | Accepted 2026-07-10 |
| 0004 | HTTP control + framed TCP tensor transport | E2/E5 | Accepted for Engine v0 |
| 0005 | Security posture by evidence tier | B4 | Accepted for Engine v0; product posture deferred |
| 0006 | Apple simulated now, physical role measured later | B5 / SA-001 | Accepted; role open |
| 0007 | Prefill/decode disaggregation deferred | — | Accepted deferral |
| 0008 | Homogeneous islands use native MAX parallelism | — | Accepted policy |
| 0009 | MAX/Fornax ownership boundary at `StageExecutable` | F-08 | Accepted 2026-07-10 |

## Rules

- A decision that is **expensive to reverse** gets a DEC-\* before work proceeds
  (charter guardrail 3).
- Each gate outcome is a DEC-\*.
- When the plan version changes, **rejected alternatives** named in the new plan
  become ADR stubs here (don't lose them).
