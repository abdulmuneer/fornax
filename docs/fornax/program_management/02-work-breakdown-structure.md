# Work Breakdown Structure (WBS)

Derived from plan **v3** §5 (architecture) + §6 (roadmap). Each epic rolls up to a
workstream ([01](01-stakeholders-and-raci.md)) and a gate ([04](04-stage-gates.md)).
Implementation boundaries follow the **provisional module map** in plan v3 §5.3
(`fornax/planner`, `inventory`, `runtime_format`, `scheduler`, `workers`,
`transport`, `max_integration`, `moe`, `serving`, `benchmarks`).

## WS-A — Planner & cost model
- A1 Partitioner data model (ModelSpec / Inventory / Target) — partitioner-spec §1
- A2 Cost model (stage time, transfer, bubbles, remote-expert exposure) — §3
- A3 Placement/replication search — §4
- A4 Golden-plan fixtures + monotonicity tests (T0) — §5
- A5 **`v0-target-contract.md`** (closes B1/B2; seeded by §3.2) — drives G1
- A6 **Phase-0 preflight workflow** (§3.4): inventory → fabric probe → target
  validate → plan → simulate → benchmark → doctor (owner DIST+SRE) — drives G1

## WS-B — Runtime & MAX surgery
- B1 `runtime-format-and-invariants.md` (closes B3): activation/KV/expert-batch format, ownership, golden vectors
- B2 Stage host: execute a layer-group as a MAX graph/graphlet
- B3 Boundary custom ops (activation/KV handoff)
- B4 Reference (slow-correct) execution path
- B5 Substrate ADR `adr/0001-max-mojo-substrate.md` (closes B5)

## WS-C — MoE expert runtime
- C1 Router → expert bucketing → weighted gather
- C2 Local/remote dispatch + expert activation tracing
- C3 Expert placement / migration policy (lever 5)
- C4 Layer/logit parity vs reference (Phase 2.5 exit)

## WS-D — Apple/Mac kernels & readiness *(critical path)*
- D1 Capability probe harness (per-nightly MAX) — R4
- D2 Target-model Apple **expert-MLP** bring-up — §5.5
- D3 Backend op-coverage matrix + Apple demotion gate — §5.10
- D4 Reversal-trigger evaluation feeding G1 — §5.5

## WS-E — Networking, transport, security, backpressure
- E1 `networking-security-and-backpressure.md` (closes B4)
- E2 Activation/KV transport (TCP / RDMA / TB-IP / shm), topology-aware placement
- E3 Trust boundary: node identity, endpoint auth, plan-integrity tags
- E4 Backpressure + failure semantics (timeout/retry/cancel/partition)

## WS-F — Scheduler & continuous batching
- F1 Microbatch scheduler + admission
- F2 1F1B overlap; bubble-fraction telemetry
- F3 Data-parallel stage replication (lever 4)

## WS-G — Observability & telemetry
- G1 Request/plan-ID propagation; per-stage timings; router/expert traces
- G2 Queue depth / backpressure / memory-KV metrics
- G3 Placement explanations ("why excluded / why slow")

## WS-H — Serving surface & Ignis integration
- H1 End-to-end lifecycle + state ownership — §5.7
- H2 Tokenizer/chat-template/stop-token seam; model support matrix
- H3 `FornaxBackend` behind the `Engine` trait; standalone OpenAI endpoint

## WS-I — Productization & ops (post-G3)
- I1 Operator UX (`cluster.yaml` / `model.yaml` / `placement.json` / `fornax doctor`)
- I2 Ops lifecycle (deploy/upgrade/drain/restart/rollback/node-replace)
- I3 Onboarding tracks; glossary; benchmark methodology of record

## WS-X — Program governance (continuous)
- X1 Gate operation + decision log
- X2 RAID upkeep + external watch
- X3 Cadence, status, reporting

## Phase → epic rollup

| Phase | Primary epics |
|---|---|
| 0 Evidence + planner | A1–A6, B1, B5, D1–D2/D4, E1 (draft), G1 (obs draft), X1–X3 |
| 1 Worker contract + transport | B2–B4, E2, H3 (skeleton), F1(T1) |
| 2 Continuous batching | F1–F2, G1–G2 |
| 2.5 MoE surgery | C1–C4, H2 |
| 3 Heterogeneous frontier | D2–D3, E2–E4, F3, G3, H1 |
| 4 Resilience | E4, F3, replay |
| 5 Productization | I1–I3 |
