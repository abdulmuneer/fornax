# Resourcing & Skills

## Skills matrix (workstream → skill → criticality)

| Skill | Workstreams | Criticality | Scarcity |
|---|---|---|---|
| Mojo + MAX graph/custom-ops | WS-B, WS-C, WS-D | Critical | High |
| **GPU/Metal kernel authoring on Apple via Mojo** | **WS-D** | **Critical for Apple G2/G3** | **Very high** |
| Distributed systems / scheduling | WS-A, WS-F, WS-E | Critical | Medium |
| LLM inference & correctness (MoE, KV, tokenizer) | WS-C, WS-H | Critical | Medium |
| Networking & security | WS-E | High | Medium |
| Observability / SRE | WS-G, WS-I | Medium | Low |
| Deterministic simulation, protocol state machines, fault injection | WS-J, WS-E, WS-F | Critical for Engine v0 | Medium |
| Numerical backend validation / reference methodology | WS-D, WS-J, WS-C | Critical for G2 | High |
| Build/release reproducibility for MAX fork | WS-B, WS-D | High | High |
| Program management | WS-X | High | Low |

## The physical-validation binding constraint

The **Apple-side kernel skill (KER/WS-D)** is the rarest *and* on the critical
path for an Apple compute claim (R-4). It is not on the Phase 0.5 Engine v0
critical path. Resourcing remains an open G2/G3 action (I-5):

- Identify whether it is in-team, hireable, or contractable before the Apple
  physical gate is scheduled.
- If unavailable, keep the Stage ABI/backend work moving and bias the eventual
  Apple role toward **capacity-only**, which needs less Apple-GPU kernel work.

## Minimal functional coverage for Phase 1 / G2

One person may hold several functions, but each accountability must be explicit:

| Role | Phase-0 load | Notes |
|---|---|---|
| DIST | Medium | Preserve engine/planner regressions; measured calibration and T3 orchestration |
| RT | High | `MaxStageBackend`, MAX lineage, physical stage integration |
| NET | High when fleet exists | Physical route, failure/backpressure replay, fabric evidence |
| SRE | Medium | T2/T3 traces, resource attribution, evidence durability |
| LLM | High | Operator/stage/logit parity and tolerance review |
| KER | High for G2 | Physical MAX/Apple parity and role decision |
| PM | Medium | Resource scheduling, assumption replacement, G2 packet honesty |
| TL/SP | Low–Medium | Architecture decisions and gate authority |

LLM ramps at Phase 1+. **v3 pulled SRE into Phase 0** (preflight `fornax doctor`/
diagnostics + observability from T1 simulation), so it is no longer purely a
Phase-1+ role.

## Capacity assumptions

- One person may hold several roles at current size. Engine v0 is complete at
  T0/T1, but physical runtime/correctness evidence should not be self-approved by
  a single functional owner.
- Missing KER hardware access delays G2 role validation; it does not invalidate
  the completed Stage ABI or simulated engine.
- Record actual assignments (names) here at kickoff; keep the RACI
  ([01](01-stakeholders-and-raci.md)) in sync.
