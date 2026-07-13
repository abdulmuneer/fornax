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
| WS-A | Planner & cost model | v4 §6, partitioner-spec |
| WS-B | Runtime & MAX surgery | v4 §4–§5 |
| WS-C | MoE expert runtime | v4 §4, §8 Phase 2.5 |
| WS-D | **Apple/Mac kernels & readiness (physical critical path)** | v4 §5, §9 |
| WS-E | Networking, transport, security, backpressure | v4 §4, §7 |
| WS-F | Scheduler & continuous batching | v4 §4, §8 Phase 2 |
| WS-G | Observability & telemetry | v4 §4, §10 |
| WS-H | Serving surface & Ignis integration | v4 §4 |
| WS-I | Productization & ops | v4 §8 Phase 5 |
| WS-J | **Simulation, assumptions & physical validation** | v4 §3, §8; simulation contract |
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
| WS-J Simulation/validation | I | C | A | R | C | R | C | C | R |
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
> G2/G3 physical-validation critical path. It does not block Phase 0.5 Engine v0 —
> [07](07-resourcing-and-skills.md).

## V4 execution ownership

| Deliverable | R | A | C |
|---|---|---|---|
| Reference/simulated StageBackend | RT + DIST | TL | LLM, SRE |
| Stage ABI and framing | RT + NET | TL | DIST, LLM |
| Engine orchestrator/scheduler | DIST | TL | RT, SRE |
| Assumption/scenario register | DIST + PM | TL | KER, NET, LLM |
| Planner defect repairs | DIST | TL | RT |
| Physical MAX backend/parity | RT + KER + LLM | TL | DIST, SRE |
| Evidence classification/register | SRE + PM | PM | all WS owners |

G1 has `PROCEED`ed for this bounded scope. Missing physical hardware is an open
WS-J validation dependency, not a reason to halt Engine v0 implementation.
