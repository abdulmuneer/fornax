# Fornax - Project Plan (v4)

> **Approved 2026-07-10 by the project owner / Sponsor.** Version 4 supersedes
> `project-plan-v3.md` for current execution. V3 remains the historical baseline.
>
> **Current gate posture:** G1 `PROCEED` for bounded Engine v0 implementation
> under registered assumptions. Physical heterogeneous validation remains open
> and gates G2/G3 plus every supported-platform claim.

Authoritative inputs:

- [Platform review disposition](platform-review-action-plan-2026-07-10.md)
- [Reconciled plan review](fornax_plan_review_reconciled.md)
- [Project plan v3](project-plan-v3.md)
- [Program-management layer](program_management/README.md)

## 0. V4 rebaseline

V4 makes one central correction: Fornax must now cross the boundary from
executable specification to real heterogeneous execution. Additional proxy gates
cannot resolve the remaining feasibility questions.

### What changed from v3

| Area | V3 posture | V4 decision |
|---|---|---|
| Current maturity | Phase-0 plan with substantial later proxy work | Executable specification and bring-up environment; no distributed engine claim |
| Immediate phase | Phase 0 evidence and broad simulated backlog | Phase 0.5 assumption-driven Engine v0 using the production Stage ABI |
| MAX boundary | MAX used where it fits; ownership partly overlapping | MAX owns per-node graph/runtime primitives; Fornax owns cross-node orchestration |
| G1 evidence | Contracts, target proof, staffing, Apple probe | Authorizes implementation from accepted contracts and a registered assumption envelope; physical proof moves to G2 |
| Planner | Planner-first feasibility authority | Planner remains advisory until repaired and calibrated; uncalibrated placement fails closed |
| Runtime format | Logical JSON golden vector | Logical contract plus versioned binary wire ABI |
| Apple proof | Target expert-MLP role gate | Adds operator/stage/logit numerical parity before role assignment |
| Proxy work | Later-phase proxy backlog may advance | Existing proxy fixtures retained; new Phase 3-5 proxy scope frozen |
| Mechanism vs product target | One seed target | Separate mechanism target and frontier-capacity target |
| MAX fork | Local source checkout and documented command | Root-pinned dependency lineage and reproducible rebuild required |

### Preserved decisions

- Fornax is an engine, not an agent harness.
- Pipeline parallelism by complete layer groups is the cross-vendor spine.
- Remote experts are measured optionality, not a v0 dependency.
- MAX/Mojo remains the preferred per-node substrate.
- Ignis remains outside the model-execution engine boundary.
- Single-stream latency caveats remain explicit.
- Measurements and simulation claims remain separately labeled.
- `PROCEED`, `ITERATE`, `NARROW`, and `KILL` remain valid gate outcomes.

## 1. Objective

Build a local heterogeneous inference cloud that uses MAX to execute model stages
on different accelerator families while Fornax plans, orchestrates, and transports
work between nodes. Engine construction begins against reference/simulated MAX
backends and the production wire protocol; hardware backends replace simulations
without changing the architecture.

The long-term product objective remains serving one sparse-MoE model that exceeds
the usable memory of every individual node. The immediate engineering objective is
to build a first executable engine with two independently scheduled stage workers,
real framing/backpressure/failure semantics, and pluggable stage backends. Physical
heterogeneous correctness is the next validation milestone, not a prerequisite for
writing that engine.

## 2. Product hypothesis and constraints

### Hypothesis

A provisioned local fleet can combine heterogeneous memory and compute into a
useful shared inference service when:

1. each node executes a sufficiently large stage locally;
2. cross-node communication occurs at bounded stage boundaries;
3. enough requests are in flight to fill the pipeline;
4. placement is calibrated from measurements rather than vendor peak figures;
5. numerical behavior is validated at every vendor boundary.

### Constraints

- A model spanning machines has an unavoidable synchronization and network floor.
- Aggregate throughput, not single-request latency parity, is the primary target.
- Cross-vendor tensor-parallel collectives are excluded from the spanning spine.
- No supported-platform or performance claim is valid without a physical run;
  implementation may advance under explicitly registered assumptions.
- No Apple performance claim is valid without numerical parity on the same build.
- No planner prediction authorizes deployment until its inputs are measured and
  its error is within the current gate bound.

### Target operator

The primary operator is a small team or firm running a shared private-AI service
with sustained concurrent work. A single bursty user is not the primary spanning
persona. The target contract must validate that the persona can supply the
concurrency required by the measured pipeline.

## 3. Two target contracts

### 3.1 Mechanism target

The Phase 0.5 mechanism target exists to build and prove the runtime architecture
under deterministic simulation. The same contract is later replayed on physical
hardware without changing request, stage, wire, ownership, or evidence schemas.

| Field | Binding v4 starting point |
|---|---|
| Model | `deepseek-ai/DeepSeek-V2-Lite-Chat`, unless the target contract records a replacement |
| Purpose | Cross-vendor MAX stage execution, transport, and parity |
| Fleet | One Linux/NVIDIA node plus one M3 Max-class Apple node |
| Encoding | BF16 first |
| Context | 4k validation point; smaller fixtures allowed for fault isolation |
| Concurrency | 1, 4, 8 |
| Transport | Length-prefixed TCP on isolated lab network |
| Parallelism | Complete contiguous layer groups; experts remain stage-local |

The binding assumption/scenario matrix is defined in
[simulation-and-assumption-contract.md](simulation-and-assumption-contract.md).

The mechanism target may fit a single node. That does not invalidate it; it is a
runtime correctness gate.

### 3.2 Frontier-capacity target

The frontier-capacity target is selected only after the mechanism target passes.
It must:

- exceed the usable memory of every individual node at the selected encoding;
- fit the complete fleet with explicit operational headroom;
- use a model/encoding supported by the measured MAX backends;
- state context, concurrency, traffic distribution, and quality requirements;
- beat or justify itself against naive pipeline and capacity-offload baselines;
- include a kill metric and an Apple role that has already been measured.

Qwen3-235B-A22B-class remains a candidate, not a commitment. DeepSeek-R1-class is
a stretch candidate only if the memory and throughput budgets close.

The binding details live in [v0-target-contract.md](v0-target-contract.md).

## 4. Architecture

### 4.1 Ownership boundary

| Component | MAX owns | Fornax owns |
|---|---|---|
| Model graph | Graph construction, compilation, kernel selection, device execution | Stage-range selection and stage manifest |
| Kernels | Built-in kernels and Mojo custom ops | Missing cross-vendor boundary logic only where public extension seams suffice |
| Device memory | Tensor allocation and backend-local execution buffers | Cross-node payload lifetime and admission budget |
| KV | Per-stage KV implementation where exposed by MAX | Cluster-wide request/stage ownership and recovery policy |
| Homogeneous parallelism | Native MAX tensor/data/expert parallel features where supported | Island selection and placement policy |
| Scheduling | Backend-local execution primitives | Global stage/microbatch orchestration required to keep stages consistent |
| Transport | Intra-runtime/device transfers | Cross-node control and tensor data planes |
| Serving | MAX may be used as a single-node baseline | Cluster endpoint, request lifecycle, and distributed result assembly |

The binding decision is ADR-0009.

### 4.2 Stage execution model

A `StageExecutable` is a compiled, versioned function over one contiguous model
layer range. It receives a stage request containing activation data plus request,
plan, microbatch, token-position, and KV ownership metadata. It returns the next
activation or final logits plus updated ownership metadata.

```text
client
  -> Fornax gateway and lockstep orchestrator
  -> Linux/NVIDIA StageExecutable [layers 0..k]
  -> versioned activation frame over TCP
  -> macOS/Apple StageExecutable [layers k+1..n]
  -> logits / sampler
  -> response
```

Networking is outside compiled graphs. A v0 custom op may pack or unpack tensors,
but it may not synchronously perform a network RPC from inside the graph.

### 4.3 Parallelism policy

| Strategy | V4 status |
|---|---|
| Pipeline parallel by complete layer group | Required spanning spine |
| Homogeneous tensor/data parallel inside one island | Reuse MAX where supported |
| Remote expert execution | Deferred until after the core Engine v0 path; simulated experiments allowed, production placement disabled until physical calibration |
| Expert migration | Deferred with remote experts |
| Cross-vendor tensor parallel | Rejected for v0 |
| Prefill/decode disaggregation | Deferred |

### 4.4 Stage and wire contracts

The logical tensor contract is defined in
[runtime-format-and-invariants.md](runtime-format-and-invariants.md). The executable
stage and byte protocol are defined in
[stage-runtime-and-wire-abi.md](stage-runtime-and-wire-abi.md). Network, security,
backpressure, timeout, and reconnect rules are defined in
[networking-security-and-backpressure.md](networking-security-and-backpressure.md).

## 5. MAX/Mojo substrate

MAX remains the preferred substrate because it supplies the graph compiler,
portable device abstraction, kernels, custom-op extension model, model runtime,
and homogeneous parallel features that Fornax should not recreate.

This is a managed fork dependency, not an assumption:

- root Fornax records the exact upstream base and patch commit;
- every accepted build records OS, toolchain, MAX/Mojo version, and command;
- Apple patches require deterministic numerical tests, not launch-only tests;
- changes use public graph/custom-op/model-extension APIs where practical;
- internal MAX changes remain isolated, reviewable, and candidates for upstream;
- a fresh-clone rebuild is required before a build becomes evidence of record.

### Apple role ladder

Apple is assigned the highest role that passes the same build-specific evidence:

1. excluded from the hot path;
2. capacity/store only;
3. expert worker;
4. complete pipeline stage;
5. arbitrary/homogeneous-island participant where MAX supports it.

The current DeepSeek short-generation result is positive bring-up evidence. It is
not numerical parity or a role decision. ADR-0001 and ADR-0006 bind the substrate
and Apple decisions.

## 6. Planner and calibration

The planner is advisory until Phase 0.5 closes the following correctness gaps:

- expert-host weights, buffers, dtype compatibility, and concurrent assignments
  are charged to the host;
- node search preserves high-memory, topology-critical, and role-specialized
  candidates on fleets larger than six nodes;
- `supports_kv`, runtime/build compatibility, quantization, and operation coverage
  affect feasibility;
- replication uses the actual replica route and adjacent-stage communication;
- measured kernel/graph profiles replace a single generic compute-class scalar;
- predictions carry calibration provenance, error, and confidence status;
- unmeasured placements fail closed unless explicitly requested as exploratory.

The model and acceptance method are defined in
[cost-model-and-calibration.md](cost-model-and-calibration.md).

## 7. Network, security, and failure policy

V0 uses two planes:

- control plane: worker admission, capability exchange, health, plan install,
  cancellation, and lifecycle commands;
- tensor data plane: length-prefixed binary frames over persistent TCP channels.

Phase 0.5 first uses multi-process loopback and injected failures. A later physical
lab may use a trusted-network exception only when isolated and recorded. Node
identities and plan hashes remain mandatory in both modes. Queues and byte budgets
are bounded. Retries may reconnect or replay only at explicitly replay-safe stage
boundaries; an ambiguous partial execution fails the request rather than silently
duplicating tokens.

Production encryption, certificate lifecycle, and external endpoint hardening are
later gates, not Phase 0.5 scope.

## 8. Roadmap and authorization

### Test tiers

| Tier | Scope |
|---|---|
| T0 | Pure unit/property tests and golden contracts |
| T1 | Executable multi-process reference/simulated workers, protocol state machines, scheduling, and fault/latency injection |
| T2 | One physical accelerator/backend with numerical reference tests |
| T3 | Two or three physical nodes; real network and distributed generation |
| T4 | Full target heterogeneous lab and frontier-capacity model |
| T5 | Installable product and fresh-operator validation |

### Phases

| Phase | Purpose | Authorization | Exit |
|---|---|---|---|
| 0 | Planner/contracts/proxy evidence already accumulated | Complete at contract/proxy scope | Inputs retained and reclassified honestly |
| **0.5** | Engine v0 with real Stage ABI and simulated/reference MAX backends | **Authorized by G1** | Executable two-worker engine, loopback TCP, backpressure/failure semantics, planner regressions |
| 1 | Physical backend integration and validation as hardware becomes available | Authorized within the G1 boundary; formal exit feeds G2 | Repeated correct generation and measured calibration on 2-3 physical nodes |
| 2 | Frontier-capacity model larger than one node | Blocked | Memory closes and aggregate throughput meets contract |
| 2.5 | Remote-expert decision and implementation, only if profitable | Deferred decision | Independent measured benefit and parity |
| 3 | Broaden hardware coverage, including AMD if justified | Blocked | Target heterogeneous serve at predicted throughput |
| 4 | Replication, elasticity, and node-loss recovery | Blocked | Real failure and added-node evidence |
| 5 | Installation, operations, onboarding, and GA | Blocked | Fresh operator can install and run the service |

### Gate sequence

- **G0:** v4 architecture baseline approved.
- **G1:** implementation authorization from accepted contracts, assumption
  register, and bounded Engine v0 scope. Status: `PROCEED`.
- **G2:** distributed correctness, numerical parity, calibration, and batching on
  2-3 physical nodes.
- **G3:** frontier-capacity heterogeneous target.
- **G4:** resilience and elasticity.
- **G5:** product GA.

## 9. Gate criteria

### G1 — Engine implementation authorization

G1 `PROCEED` requires:

1. Plan v4 and the review disposition are Sponsor-approved.
2. The mechanism target and explicit assumption/scenario matrix are reviewable.
3. The Stage ABI, runtime format, network/backpressure specification, and core
   ADRs define one implementation architecture.
4. Engine v0 scope excludes unsupported product claims and later proxy expansion.
5. Functional owners and an active sprint are assigned.

Open hardware assumptions do not block G1. They remain visible and block the
specific G2/G3 claims they support.

### G2 — Physical distributed correctness

G2 requires:

1. Both reproduced planner defects are fixed and covered by regression tests.
2. The MAX fork/build is reproducibly pinned from Fornax.
3. A real request executes across two physical heterogeneous MAX stages.
4. Boundary activations and final logits meet accepted per-dtype tolerances.
5. Stage, packing, transport, and exposed-wait measurements are recorded.
6. The planner is calibrated for the measured path within +/-20%.
7. Apple receives a measured role decision.

Failure produces `ITERATE`, `NARROW`, or `KILL`, never a fabricated hardware
claim.

## 10. Metrics

### Phase 0.5 Engine v0 binding metrics

- Correctness: boundary activation and final-logit divergence within the target
  contract's dtype-specific tolerance.
- Completeness: one request executes through two independent stage-worker
  processes for prefill and decode using the production ABI.
- Stability: bounded queue and memory growth during the specified sustained run.
- Attribution: compute, packing, transfer, queue, and exposed wait measured
  separately.
- Assumption traceability: every injected compute/link/failure parameter cites a
  scenario and assumption ID.
- Determinism: repeated reference/simulated runs produce contract-equivalent
  outputs and event ledgers.
- Evidence freshness: artifacts are indexed and clearly classified as simulation
  or physical measurement.

### Later provisional metrics

- Capacity: frontier target exceeds one node's usable memory and fits the fleet
  with headroom.
- Throughput efficiency: provisional >=60% of the target contract's defined
  sum-of-node ideal at saturation.
- Planner accuracy: +/-20% by G2 and +/-10% by G3.
- Concurrency: saturation at or below the persona's contracted supply.
- Remote experts: enabled only when they improve saturated throughput materially
  after all transport/synchronization costs; the contract binds the threshold.
- Resilience: zero dropped in-flight requests on the G4 failure scenarios.
- Honesty: every reported metric identifies simulation or physical measurement.

## 11. Ranked risks

| Rank | Risk | Immediate mitigation |
|---|---|---|
| 1 | Engine implementation waits on unavailable hardware | Build through pluggable reference/simulated backends using the production ABI |
| 2 | Apple numerical path is incorrect or too slow | Deterministic parity before optimization; staged role |
| 3 | Planner admits impossible or misses feasible placements | Fix reproduced defects; fail closed; property tests |
| 4 | Wire/runtime contract is incomplete | Versioned Stage ABI and malformed-frame corpus |
| 5 | MAX internal churn makes the fork unmaintainable | Root pin, narrow patch set, rebase CI, upstreaming |
| 6 | Network/synchronization floor defeats target concurrency | Attribute exposed wait; compare baselines; narrow if needed |
| 7 | Target persona cannot fill the pipeline | Traffic evidence and concurrency sweep |
| 8 | Frontier model/fleet memory budget does not close | Exact capacity target after mechanism proof |
| 9 | Required MAX/Apple skills are not staffable | Named assignments before G1 `PROCEED` |
| 10 | Proxy evidence is mistaken for product capability | Freeze proxy expansion and use evidence classifications |

## 12. Required artifacts and decisions

### Required for G1 record and Engine v0 start

- `v0-target-contract.md`
- `stage-runtime-and-wire-abi.md`
- `runtime-format-and-invariants.md`
- `networking-security-and-backpressure.md`
- `cost-model-and-calibration.md`
- `two-node-max-validation-plan.md`
- `evidence-register.md`
- `simulation-and-assumption-contract.md`
- ADR-0001 through ADR-0006, ADR-0008, and ADR-0009 as applicable
- Root MAX dependency pin/rebuild instructions for the later physical lane
- Phase 0.5 active sprint

### Current decision

- Plan v4: approved.
- G0 v4 architecture baseline: passed.
- G1: `PROCEED` for the bounded assumption-driven Engine v0 scope as of 2026-07-10.
- Phase 0.5 Engine v0: authorized.
- G2 physical heterogeneous correctness: open.
- New Phase 3-5 proxy expansion: frozen.
- Remote experts, AMD, resilience, and GA work: deferred pending their gates.
