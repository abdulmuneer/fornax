# Stakeholders & RACI

## Functional roles

Roles are by **function**, not headcount (one person may hold several at current
size — see [07-resourcing-and-skills.md](07-resourcing-and-skills.md)).

| Code | Role |
|---|---|
| SP | Sponsor / decision authority |
| PM | Program manager |
| TL | Tech lead / architect (plan conformance) |
| RT | Runtime & MAX-surgery engineer |
| KER | Kernel engineer (GPU / Metal / Mojo) |
| DIST | Distributed-systems / scheduler engineer |
| LLM | LLM inference & correctness engineer |
| NET | Networking & security engineer |
| SRE | Observability / ops |

## Workstreams

| WS | Name | Plan refs |
|---|---|---|
| WS-A | Planner & cost model | §6 Phase 0, partitioner-spec |
| WS-B | Runtime & MAX surgery | §5.4, §5.6 |
| WS-C | MoE expert runtime | §5.1, §6 Phase 2.5 |
| WS-D | **Apple/Mac kernels & readiness (critical path)** | §5.5, §5.10 |
| WS-E | Networking, transport, security, backpressure | §5.8, §5.3 data plane |
| WS-F | Scheduler & continuous batching | §5.2, §6 Phase 2 |
| WS-G | Observability & telemetry | §5.9 |
| WS-H | Serving surface & Ignis integration | §5.7, §9 |
| WS-I | Productization & ops | §6 Phase 5, §10 P2 |
| WS-X | Program governance | this folder |

## RACI (R=responsible, A=accountable, C=consulted, I=informed)

| Workstream / decision | SP | PM | TL | RT | KER | DIST | LLM | NET | SRE |
|---|---|---|---|---|---|---|---|---|---|
| WS-A Planner | I | C | A | C | | R | C | | |
| WS-B Runtime/MAX surgery | I | I | A | R | C | C | C | | |
| WS-C MoE expert runtime | I | I | A | R | C | C | R | | |
| WS-D Apple readiness (crit) | C | C | A | C | R | | C | | |
| WS-E Net/security | I | C | C | | | C | | A/R | C |
| WS-F Scheduler/batching | I | C | A | C | | R | C | | |
| WS-G Observability | I | C | C | | | C | | | A/R |
| WS-H Serving/Ignis | I | I | A | R | | | R | | |
| WS-I Productization | C | A | C | C | | | | C | R |
| Gate go/no-go (G1–G5) | **A/R** | R | C | C | C | C | C | C | C |
| Plan version change | A | C | R | C | | | | | |
| RAID & cadence | I | **A/R** | C | C | C | C | C | C | C |

## Stakeholders (non-build)

| Stakeholder | Interest | Cadence |
|---|---|---|
| Sponsor | Go/no-go, spend, thesis | Gate reviews + weekly |
| Modular / MAX (external) | Apple/MoE capability we depend on | Dated watch (passive) — [06](06-dependencies-and-external-watch.md) |
| Early operator/design partner | Validates persona & concurrency (B2) | From Phase 3 |
| Ignis maintainers | `Engine`-trait seam stability | Phase 1 + as the seam changes |

> **Gap flagged:** WS-D (KER, Apple Metal + Mojo) is the rarest skill and on the
> critical path. Resourcing it is a Phase-0 action — [07](07-resourcing-and-skills.md).

**v3 update:** plan v3 §0 assigns explicit **Responsible/Accountable per Phase-0
artifact** (mirrored in the [sprint](sprints/phase-0-evidence-sprint.md) and
[G1 exit](04-stage-gates.md)). The new **preflight workflow** (§3.4) is owned
**DIST + SRE**, which pulls SRE into Phase 0 (previously Phase-1+). If a required
role is unstaffed at G1, silent PROCEED is forbidden (§0).
