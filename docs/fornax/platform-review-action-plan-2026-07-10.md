# Fornax Platform Review Disposition and Action Plan

Date: 2026-07-10  
Decision authority: Project owner / Sponsor  
Review lens: compiler-first, explicit contracts, progressive lowering,
performance portability, and measured hardware evidence  
Status: Accepted as input to `project-plan-v4.md`

## Decision

Continue Fornax, but rebaseline the program around one real heterogeneous MAX
vertical slice. The current repository is accepted as an executable specification,
planner prototype, contract suite, and kernel bring-up environment. It is not yet
accepted as a distributed inference engine or heterogeneous local cloud.

G1 authorizes Engine v0 implementation under a registered assumption envelope.
Phase 0.5 builds the real Stage ABI, orchestrator, worker state machines, and wire
protocol against reference/simulated MAX backends without waiting for every
physical device. Physical heterogeneous validation runs as a parallel evidence
lane and remains mandatory for G2/G3 and any supported-platform claim.

## Review disposition

| ID | Finding | Disposition | Required action | Owner | Acceptance evidence |
|---|---|---|---|---|---|
| F-01 | No real Fornax stage execution path | Accept | Build the production Stage ABI and Engine v0 against reference/simulated MAX backends first; replace them with physical MAX backends as hardware becomes available. | RT + DIST | T1 Engine v0 bundle, followed by T3 generation/parity |
| F-02 | Planner can overcommit remote expert memory | Accept | Charge expert weights, buffers, dtype compatibility, and concurrent assignments to expert hosts; add regression tests. | DIST | T0 false-feasibility regression passes |
| F-03 | Planner can prune a feasible node on fleets larger than six nodes | Accept | Replace the single-order cutoff with bounded search that preserves capacity/topology candidates; add regression tests. | DIST | T0 false-infeasibility regression passes |
| F-04 | Runtime-format fixture is not a wire ABI | Accept | Define a versioned logical contract and binary frame protocol with identity, ordering, ownership, integrity, and compatibility rules. | RT + NET | ABI conformance and malformed-frame tests |
| F-05 | MAX fork is not reproducibly pinned by Fornax | Accept | Record the upstream base, patch commit, build inputs, and root dependency pin; require rebase/build/parity checks. | RT + KER | Fresh-clone reproducibility report |
| F-06 | Apple smoke proves launch, not numerical correctness | Accept | Add deterministic prefill, decode, MoE, gather, stage, and logit parity tests before throughput claims. | KER + LLM | Per-dtype parity report on pinned M3 Max build |
| F-07 | Proxy-gate breadth obscures the engine critical path | Accept | Freeze new Phase 3-5 proxy features; preserve existing fixtures as contract evidence; focus Phase 0.5 on an executable Engine v0 using the same ABI intended for hardware. | PM + TL | Roadmap, WBS, sprint, and gate posture refreshed |
| F-08 | MAX and Fornax responsibilities overlap | Accept | MAX owns per-node graph execution, kernels, device memory, and local runtime primitives; Fornax owns planning, node orchestration, and cross-node transport. | TL + RT | ADR-0009 accepted and Stage ABI implemented |
| F-09 | Planning and MAX dependency state are not durable | Accept | Version the technical source-of-truth or place it in a private tracked program repository; do not rely on loose local files. | SP + PM | Recorded repository policy and dependency manifest |

## Scope of Phase 0.5

### In scope

- A two-worker heterogeneous simulation profile representing Linux/NVIDIA and
  macOS/Apple Silicon, with parameterized memory, compute, link, and failure
  assumptions.
- `ReferenceStageBackend` and `SimulatedMaxStageBackend` implementations of the
  production `StageExecutable` contract.
- Multi-process loopback execution using the production v1 TCP frames.
- Physical Linux/NVIDIA and macOS/Apple validation whenever each resource is
  available; absence of the complete fleet does not stop Engine v0 construction.
- One pinned MAX source/build lineage.
- DeepSeek-V2-Lite-Chat as the mechanism target unless the target contract
  records a more bring-up-feasible replacement.
- BF16 first; FP16 only if both selected backends pass the same contract.
- Complete layer-group pipeline stages with stage-local expert execution.
- Length-prefixed TCP tensor transport on a trusted isolated lab network.
- Lockstep prefill/decode orchestration sufficient to prove correctness.
- Batch/concurrency points 1, 4, and 8 for mechanism evidence.
- Boundary activation parity, final-logit parity, and deterministic generation.
- Injected stage, pack, transfer, queue, and exposed-wait timing tied to named
  assumption IDs; physical results replace rather than silently tune assumptions.
- Planner repair and calibration against simulation scenarios first, then measured
  paths when available.

### Out of scope

- Remote expert execution or expert migration.
- AMD enablement.
- RDMA, UCX, NIXL, or production zero-copy transport.
- Elasticity, replay, stage replication, or node-loss guarantees.
- Kubernetes or a general-purpose cluster manager.
- Product authentication, public-network deployment, or GA packaging.
- Claims that the mechanism target proves the final frontier-capacity thesis.

## Required artifacts

| Artifact | Purpose | Review authority |
|---|---|---|
| `project-plan-v4.md` | Rebaseline architecture, phases, gates, and current posture | SP + TL |
| `v0-target-contract.md` | Bind mechanism and capacity targets, fleet, memory, concurrency, and thresholds | SP + TL |
| `stage-runtime-and-wire-abi.md` | Define the executable stage and cross-node byte contract | TL + RT + NET |
| `runtime-format-and-invariants.md` | Define logical activation, KV, expert, and tolerance invariants | TL + LLM |
| `networking-security-and-backpressure.md` | Define v0 transport, flow control, identity, and failure semantics | TL + NET |
| ADR set under `docs/fornax/adr/` | Record substrate, ownership, transport, security, Apple role, and deferred alternatives | SP/TL by ADR |
| `cost-model-and-calibration.md` | Make planner admission measurable and fail-closed | TL + DIST |
| `two-node-max-validation-plan.md` | Bind T2/T3 correctness and performance evidence | TL + LLM + KER |
| `evidence-register.md` | Index durable evidence without overstating proxy scope | PM + SRE |

## Exit criteria for the authorized evidence sprint

Phase 0.5 Engine v0 exits only when all of the following are true:

1. Two stage workers execute explicit layer ranges through the production
   `StageExecutable` contract using reference/simulated MAX backends.
2. The same request crosses a multi-process loopback boundary using the production
   versioned wire protocol.
3. Boundary activations and final logits pass the target contract against the
   deterministic reference backend.
4. Prefill and decode complete for batch/concurrency 1, 4, and 8, or the evidence
   records the first unsupported point and the resulting scope decision.
5. A sustained run demonstrates bounded memory and queue growth.
6. Stage compute, serialization, transport, queue, and exposed wait are attributed
   separately; simulated values cite assumption/scenario IDs.
7. The planner passes the reproduced feasibility regressions and predicts the
   deterministic scenarios consistently.
8. Every unvalidated hardware capability remains an open assumption mapped to a
   future T2/T3 test.

Physical heterogeneous execution, numerical parity, planner calibration, and the
Apple role decision are G2 evidence. They are pursued immediately when hardware
exists but do not block creation of Engine v0.

## Stop and narrow triggers

- Numerical parity cannot be reached on the pinned Apple path without a large,
  unmaintainable MAX fork.
- Simulation across the full declared network/compute envelope shows no plausible
  operating region at the target persona's attainable concurrency.
- The target model/fleet memory budget does not close with operational headroom.
- The team cannot staff MAX graph partitioning plus Apple numerical correctness.

Any trigger produces a Sponsor decision: `ITERATE`, `NARROW`, or `KILL`. It does
not silently expand scope.

## Authorization

The Sponsor's 2026-07-10 instructions accepted this disposition, approved project
plan v4, and explicitly authorized simulation- and assumption-driven Engine v0
implementation so hardware scarcity does not stall the program. G1 therefore
authorizes the bounded Engine v0 scope. It does not close G2/G3 physical evidence
or authorize product capability claims.
