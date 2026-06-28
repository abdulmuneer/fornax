# Resourcing & Skills

## Skills matrix (workstream → skill → criticality)

| Skill | Workstreams | Criticality | Scarcity |
|---|---|---|---|
| Mojo + MAX graph/custom-ops | WS-B, WS-C, WS-D | Critical | High |
| **GPU/Metal kernel authoring on Apple via Mojo** | **WS-D** | **Critical (crit path)** | **Very high** |
| Distributed systems / scheduling | WS-A, WS-F, WS-E | Critical | Medium |
| LLM inference & correctness (MoE, KV, tokenizer) | WS-C, WS-H | Critical | Medium |
| Networking & security | WS-E | High | Medium |
| Observability / SRE | WS-G, WS-I | Medium | Low |
| Program management | WS-X | High | Low |

## The binding constraint

The **Apple-side kernel skill (KER/WS-D)** is the rarest *and* on the critical
path (R-4). Resourcing it is a **Phase-0 action (I-5)**:

- Identify whether it is in-team, hireable, or contractable by **W2**.
- If unavailable, that itself informs G1: bias the Apple role toward
  **capacity-only** (which needs far less Apple-GPU kernel work) and lean on MAX
  shipping the kernels.

## Minimal team to clear G1 (Phase 0)

Phase 0 is mostly model-free logic + spec + probes — it does **not** need the full
team:

| Role | Phase-0 load | Notes |
|---|---|---|
| DIST | High | Planner + cost model + v0-contract |
| RT | Medium | Format spec + substrate ADR |
| KER | Medium | Apple expert-MLP probe (D2) — gating evidence |
| NET | Low | Networking/security draft |
| SRE | Low–Med | Preflight workflow (§3.4) + observability from T1 (pulled earlier in v3) |
| PM | Medium | Gate, RAID, procurement, staffing |
| TL/SP | Low | Review + decide |

LLM ramps at Phase 1+. **v3 pulled SRE into Phase 0** (preflight `fornax doctor`/
diagnostics + observability from T1 simulation), so it is no longer purely a
Phase-1+ role.

## Capacity assumptions

- One person may hold several roles at current size (charter). The plan is
  **not** a one-person effort beyond Phase 0 — flag to Sponsor at G1 with the
  re-baselined schedule.
- Record actual assignments (names) here at kickoff; keep the RACI
  ([01](01-stakeholders-and-raci.md)) in sync.
