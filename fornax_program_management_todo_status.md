# Fornax Program Management Todo Status

Generated: 2026-06-22 UTC

Sources:
- `docs/fornax/program_management/02-work-breakdown-structure.md`
- `docs/fornax/program_management/03-roadmap-milestones-critical-path.md`
- `docs/fornax/program_management/04-stage-gates.md`
- `docs/fornax/program_management/sprints/phase-0-evidence-sprint.md`
- `docs/fornax/program_management/sprints/README.md` and phase sprint backlog files
- `docs/fornax/program_management/05-raid-log.md`
- `docs/fornax/program_management/08-decision-log.md`
- Current repo evidence under `fornax/`, `fornax/golden_vectors/`, `tests/`, and `fornax_development_journal.md`

Legend:
- `[x]` means current repo evidence satisfies the plan item at its stated scope.
- `[ ]` with `Partial` means useful implementation, tooling, draft, or T0/T1 simulation evidence exists, but the plan item is not closed at full scope.
- `[ ]` with `Open` means no sufficient implementation or gate evidence was found.
- Simulation-only work is not marked complete for items that require live MAX, real multi-node, real heterogeneous, Sponsor sign-off, or product GA evidence.

Current validation snapshot:
- `python3 -m fornax test golden-plans`: 3/3 passed.
- `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_trace_ledger_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65`: 31/31 passed over 2 logical hosts.
- `python3 -m unittest tests.test_fornax_planner`: 211 tests passed.
- `python3 -m fornax program local-accelerator-smoke --out-dir /tmp/fornax_local_accelerator_smoke_h100_20260622 --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --expert-device cuda:0 --transfer-source-device cuda:0 --transfer-destination-device cuda:1 ...`: 3/3 checks passed on local H100s treated as logical hosts; not G2/G3 gate evidence.
- `python3 -m fornax program simulate-phase0 --target fornax/golden_plans/v0_target_contract_fixture.md --out-dir /tmp/fornax_phase0_g1_packet_20260622 --gpu-count 2 --profile two-gpu-heterogeneous ...`: 9/9 Phase-0 deliverables machine/simulation complete or closed; recommended G1 outcome remains ITERATE.
- `python3 -m fornax program g1-evidence-packet --bundle /tmp/fornax_phase0_g1_packet_20260622 --out /tmp/fornax_phase0_g1_packet_20260622/g1-evidence-packet-cli.json --markdown-out /tmp/fornax_phase0_g1_packet_20260622/g1-evidence-packet-cli.md --date 2026-06-22 --plan-version v3`: packet valid; `machine_complete=false`, `g1_ready=false`, `closure_blockers=4`.

## Milestones

- [x] M0 Architecture baseline / G0 passed. Evidence: plan v3 accepted by `04-stage-gates.md`; program governance fixture also records G0/G1 posture.
- [ ] M1 Evidence sprint done -> G1. Partial: large T0/T1 evidence exists, but G1 remains open because TL/SP sign-off, Apple rank-1 probe, staffing answer, and Sponsor DEC-005 are not closed.
- [ ] M2 Pipeline correctness, simulation then 2-3 nodes. Partial: T1 pipeline-correctness CPU/sim fixture exists; real 2-3 node pipeline evidence is open.
- [ ] M3 Continuous batching scales. Partial: T1 continuous-batching fixture exists; real scaling evidence for G2 is open.
- [ ] M4 MoE expert runtime parity. Partial: T1/CPU MoE runtime, migration, remote expert, and parity fixtures exist; real runtime parity for G2 is open.
- [ ] M5 Heterogeneous frontier serve. Partial: planner, model support, serving adapter, trust/security, state ownership, and simulation bundle exist; real NVIDIA/AMD/Mac frontier serve for G3 is open.
- [ ] M6 Resilience / elasticity. Partial: T1 resilience replay and stage replication simulations exist; real T4 node-loss/added-node evidence is open.
- [ ] M7 Productization / GA. Partial: ops lifecycle and onboarding simulations exist; installable/operable GA evidence is open.

## Gates

- [x] G0 Architecture baseline. Evidence: `04-stage-gates.md` marks passed.
- [ ] G1 Evidence / go-no-go. Partial: T0/T1 commands, draft generation, G1 review draft, phase0 status, and G1 evidence packet exist; open blockers include Sponsor decision, TL/SP target-contract sign-off, Apple role decision from rank-1 local probe, staffing closure, and reviewed docs/ADR.
- [ ] G2 Distributed correctness. Open: requires real 2-3 node pipeline correctness, aggregate scaling, planner match, and MoE parity.
- [ ] G3 Heterogeneous frontier. Open: requires real frontier MoE across NVIDIA/AMD + Mac at predicted throughput with security/backpressure active.
- [ ] G4 Resilience. Open: requires real added-node scaling and zero dropped in-flight requests on single-node loss.
- [ ] G5 Product GA. Open: requires firm-installable, operable, upgradeable serving product with onboarding.

## Phase-0 Evidence Sprint Deliverables

- [x] S0-1 Partitioner + cost model + golden plans (A1-A4). Evidence: `fornax/planner/`, `fornax/golden_plans/`, `fornax test golden-plans` 3/3.
- [ ] S0-2 `v0-target-contract.md`. Partial: parser, validator, draft generator, fixture, and G1 evidence packet coverage exist (`fornax/target_contract.py`, `fornax/validation.py`, `fornax/golden_plans/v0_target_contract_fixture.md`, `fornax/g1_evidence_packet.py`); real signed-off contract is open.
- [ ] S0-3 Concurrency sweep in contract. Partial: throughput scaling simulation exists (`fornax/throughput_scaling.py`); persona-supplied minimum saturation concurrency and G1 market evidence are open.
- [ ] S0-4 `runtime-format-and-invariants.md`. Partial: runtime-format validator, golden vectors, and spec draft renderer exist; reviewed/accepted document is open.
- [ ] S0-5 `networking-security-and-backpressure.md`. Partial: network contract validator and spec draft renderer exist; reviewed/accepted phase spec remains open.
- [ ] S0-6 `adr/0001-max-mojo-substrate.md`. Partial: substrate ADR draft renderer exists; accepted ADR and source-watch closure remain open.
- [ ] S0-7 Apple expert-MLP probe on `desktop-minimal`. Partial: Apple probe template/validator/decision tooling exists; measured pinned-build Apple target probe is open.
- [ ] S0-8 Roadmap rebaseline + staffing answer. Partial: rebaseline draft tooling exists; KER staffing decision and Sponsor scope acceptance remain open.
- [x] S0-9 Phase-0 preflight workflow. Evidence: `fornax/preflight.py`, `fornax/doctor.py`, optional G1 draft/golden/calibration/fabric outputs, G1 evidence packet outputs, and journal validation; note that G1 closure still needs reviewed artifacts.

## Work Breakdown Structure

### WS-A — Planner & Cost Model

- [x] A1 Partitioner data model. Evidence: `fornax/planner/model.py` defines `ModelSpec`, `Inventory`, `Target`, `PlacementPlan`.
- [x] A2 Cost model. Evidence: `fornax/planner/cost.py` and unit coverage for memory, transfer, stage timing, and predicted throughput.
- [x] A3 Placement/replication search. Evidence: `fornax/planner/search.py`, `fornax plan`, slow-node and replication-related tests.
- [x] A4 Golden-plan fixtures + monotonicity tests (T0). Evidence: `fornax/golden_plans/`, `fornax/golden.py`, `python3 -m fornax test golden-plans` 3/3.
- [ ] A5 `v0-target-contract.md`. Partial: machine fixture, parser, draft, validation, and preflight integration exist; real TL/SP-reviewed contract with sign-off is open.
- [x] A6 Phase-0 preflight workflow. Evidence: `fornax preflight`, inventory/fabric/target-validate/plan/simulate/benchmark/doctor bundle generation. Gate-review inputs still need human closure.

### WS-B — Runtime & MAX Surgery

- [ ] B1 `runtime-format-and-invariants.md`. Partial: `fornax/runtime_format.py`, `fornax/runtime_format_spec.py`, and golden vectors exist; formal reviewed doc remains open.
- [ ] B2 Stage host as MAX graph/graphlet. Partial: `fornax/stage_host.py` simulates a stage host and validates boundaries; real MAX graphlet execution is open.
- [ ] B3 Boundary custom ops. Partial: activation/KV boundary semantics are simulated in stage-host/runtime-format fixtures; real custom ops are open.
- [x] B4 Reference slow-correct execution path. Evidence: slow-correct/reference parity exists in runtime-format, stage-host, pipeline, remote-expert, and MoE parity fixtures.
- [ ] B5 Substrate ADR `adr/0001-max-mojo-substrate.md`. Partial: `fornax/substrate_adr.py` can render a draft; accepted ADR is open.

### WS-C — MoE Expert Runtime

- [ ] C1 Router -> expert bucketing -> weighted gather. Partial: `fornax/moe.py` T1 simulation covers routing, dispatch, weighted gather; real runtime surgery is open.
- [ ] C2 Local/remote dispatch + expert activation tracing. Partial: `fornax/moe.py` and `fornax/remote_expert_probe.py` cover simulated/CPU paths; real distributed expert dispatch is open.
- [ ] C3 Expert placement / migration policy. Partial: `fornax/moe_migration.py` simulates hot-expert migration; real migration policy/runtime is open.
- [ ] C4 Layer/logit parity vs reference. Partial: `fornax/moe_parity.py` CPU fixture exists; real Phase 2.5 parity exit is open.

### WS-D — Apple/Mac Kernels & Readiness

- [ ] D1 Capability probe harness per-nightly MAX. Partial: `fornax/apple_probe.py` provides templates and validation; per-nightly/pinned local Apple probe runs are open.
- [ ] D2 Target-model Apple expert-MLP bring-up. Open: no measured target Apple expert-MLP artifact found.
- [ ] D3 Backend op-coverage matrix + Apple demotion gate. Partial: backend/model-support tooling exists (`fornax/backend_coverage.py`, `fornax/model_support.py`); Apple-specific measured demotion gate is open.
- [ ] D4 Reversal-trigger evaluation feeding G1. Partial: Apple role-decision draft tooling exists; final measured probe and Sponsor/G1 decision are open.

### WS-E — Networking, Transport, Security, Backpressure

- [ ] E1 `networking-security-and-backpressure.md`. Partial: `fornax/network_security_spec.py` and network contract fixture exist; reviewed/accepted doc remains open.
- [ ] E2 Activation/KV transport. Partial: `fornax/transport.py` T1 logical-host transport contract exists; real TCP/RDMA/TB-IP/shm transport is open.
- [ ] E3 Trust boundary. Partial: `fornax/trust_boundary.py` T1 simulation exists; real TLS/mTLS/auth/key distribution is open.
- [ ] E4 Backpressure + failure semantics. Partial: transport, metrics-ledger, resilience-replay simulations cover timeout/cancel/backpressure/replay; real failure semantics are open.

### WS-F — Scheduler & Continuous Batching

- [ ] F1 Microbatch scheduler + admission. Partial: `fornax/scheduler.py` T1 scheduler contract exists; production scheduler is open.
- [ ] F2 1F1B overlap and bubble-fraction telemetry. Partial: `fornax/continuous_batching.py` T1 simulation exists; real overlap telemetry is open.
- [ ] F3 Data-parallel stage replication. Partial: `fornax/stage_replication.py` T1 simulation exists; real replicated stage runtime is open.

### WS-G — Observability & Telemetry

- [ ] G1 Request/plan-ID propagation, per-stage timings, router/expert traces. Partial: observability contract, metrics ledger, and trace-ledger fixtures exist; live runtime telemetry is open.
- [ ] G2 Queue depth / backpressure / memory-KV metrics. Partial: `fornax/metrics_ledger.py` T1 metrics ledger exists; live exporter/dashboard evidence is open.
- [x] G3 Placement explanations. Evidence: planner placement explanations are implemented and propagated into target-contract drafts; live observability linkage is future work.

### WS-H — Serving Surface & Ignis Integration

- [ ] H1 End-to-end lifecycle + state ownership. Partial: `fornax/state_ownership.py` T1 simulation exists; live serving/runtime ownership is open.
- [ ] H2 Tokenizer/chat-template/stop-token seam; model support matrix. Partial: serving adapter and model-support fixtures record template/tokenizer hashes and model capability matrix; real target tokenizer/template support is open.
- [ ] H3 `FornaxBackend` behind `Engine`; standalone OpenAI endpoint. Partial: `fornax/serving.py` simulates Ignis/OpenAI-to-engine seam; real backend and live endpoint are open.

### WS-I — Productization & Ops

- [ ] I1 Operator UX. Partial: preflight, doctor, lifecycle simulations, and generated artifacts exist; installable product UX is open.
- [ ] I2 Ops lifecycle. Partial: `fornax/ops_lifecycle.py` T1 simulation exists; real deploy/upgrade/drain/restart/rollback/node-replace is open.
- [ ] I3 Onboarding tracks, glossary, benchmark methodology. Partial: `fornax/onboarding.py` T1 methodology simulation exists; real product docs and design-partner onboarding are open.

### WS-X — Program Governance

- [ ] X1 Gate operation + decision log. Partial/ongoing: decision log exists, `fornax/program_governance.py` validates controls, and `fornax/g1_evidence_packet.py` prepares a machine-checkable G1 review packet; real gate outcomes remain open.
- [ ] X2 RAID upkeep + external watch. Partial/ongoing: RAID/watch docs and governance fixture exist; live upkeep and MAX source-watch updates remain open.
- [ ] X3 Cadence, status, reporting. Partial/ongoing: templates and `phase0-status`/governance tooling exist; real cadence operation is ongoing.

## Quality/Test Tiers

- [x] T0 Planner/scheduler unit + golden plans. Evidence: `fornax test golden-plans` and unit tests pass.
- [x] T1 Simulated workers/contracts/backpressure. Evidence: `fornax program simulate-t1` reports 31/31 checks passed over two logical hosts, including trace-ledger correlation.
- [ ] T2 Single-node accelerator. Partial: `fornax program local-accelerator-smoke` exists and a local H100 run passed 3/3 checks with measured expert-MLP and same-host GPU0->GPU1 transfer evidence; target-model parity and formal gate evidence remain open.
- [ ] T3 2-3 node pipeline. Open: no real 2-3 node pipeline run found.
- [ ] T4 Full heterogeneous lab. Open: no full lab-reference heterogeneous run found.

## Decision / ADR Register

- [x] DEC-001 Fornax is an engine, not a harness. Evidence: accepted in decision log.
- [x] DEC-002 Pipeline-parallel spine. Evidence: accepted in decision log and planner/runtime direction.
- [x] DEC-003 Apple participation staged and gated. Evidence: accepted in decision log and Apple probe/rebaseline tooling.
- [x] DEC-004 Plan changes only by version bump. Evidence: accepted in decision log.
- [ ] DEC-005 G1 go/no-go + rebaseline. Open: pending G1.
- [x] DEC-006 Speculative decoding out of v0. Evidence: accepted in decision log.
- [ ] ADR-0001 MAX/Mojo substrate. Partial: draft renderer exists; accepted ADR open.
- [ ] ADR-0002 Pipeline-parallel default. Open/backfill.
- [ ] ADR-0003 Bounded remote-expert execution. Open/backfill.
- [ ] ADR-0004 Transport choice. Open/Phase 1.
- [ ] ADR-0005 Security posture. Partial: phase spec/contract exists; accepted ADR open.
- [ ] ADR-0006 Apple participation level. Open/G1.
- [ ] ADR-0007 Prefill/decode disaggregation rejection/deferral. Open/backlog.
- [ ] ADR-0008 Homogeneous intra-node tensor-parallel island policy. Open/backlog.

## RAID Issues and Assumptions

- [ ] I-1 `v0-target-contract.md` not written. Partial: draft/fixture/tooling exists; formal reviewed document/sign-off open.
- [ ] I-2 `runtime-format-and-invariants.md` not written. Partial: spec draft generator/golden vectors exist; reviewed document open.
- [ ] I-3 `networking-security-and-backpressure.md` not written. Partial: spec draft generator/contract exists; reviewed document open.
- [ ] I-4 `adr/0001-max-mojo-substrate.md` not written. Partial: ADR draft renderer exists; accepted ADR open.
- [ ] I-5 KER/Apple staffing gap unresolved. Open: no staffing closure evidence found.
- [x] I-6 Phase-0 preflight workflow not written. Evidence: `fornax preflight` and `fornax doctor` are implemented; workflow caveat remains that G1 artifacts need review/sign-off.
- [ ] A-1 Persona supplies pipeline-filling concurrency. Partial: simulation/sweep tooling exists; persona evidence open.
- [ ] A-2 MAX can run target expert-MLP on target Mac. Open: no measured Apple target probe found.
- [ ] A-3 Fabric tiers are procurable/available. Open: no procurement closure evidence found.
- [ ] A-4 Quant format byte-compatible across MAX backends. Partial: runtime-format spec/golden vectors exist; MAX backend compatibility proof open.
- [ ] A-5 Required skills are staffable. Open: no staffing closure evidence found.
- [ ] A-6 Partitioner cost model predicts within provisional bound. Partial: calibration and simulation exist; G2 real calibration proof open.

## Procurement Actions

- [ ] Confirm `desktop-minimal` is on hand for Apple expert-MLP probe. Open: no confirmation artifact found.
- [ ] Spec exact SKUs + fabric in `v0-target-contract.md`; place `prosumer-rack` order by lead time. Partial: target-contract draft/fixture exists; procurement order evidence open.
- [ ] Record negative hardware list. Partial: plan has target profile/unsupported targets; no dedicated reviewed negative hardware artifact found.

## Immediate Next Open Items

1. Continue implementing the WBS simulation-first using two local GPUs as two logical machines; do not block on Mac/AMD availability.
2. Materialize and review/sign off `v0-target-contract.md` with memory budget, throughput bar, concurrency/persona evidence, and seed acceptance/replacement rationale.
3. Materialize/review `runtime-format-and-invariants.md`, `networking-security-and-backpressure.md`, and `adr/0001-max-mojo-substrate.md` from the existing generators.
4. Keep Apple/Mac evidence as a deferred validation lane: run or record the rank-1 Apple expert-MLP probe on pinned build later, or explicitly demote Apple role for G1 when the Sponsor chooses.
5. Close the KER/Apple staffing answer and record Sponsor scope acceptance if narrowed.
6. Use the generated G1 evidence packet and gate-review draft to attach TL/SP/spec/staffing sign-offs and prepare DEC-005 once missing real evidence is available.
7. Continue moving simulation-complete T1 items into local H100 smoke/T2-style validation where this machine can provide real evidence, while preserving the distinction from final T3/T4 heterogeneous lab closure.
