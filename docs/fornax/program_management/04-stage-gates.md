# Stage-Gate Governance

Plan: `../project-plan-v4.md`  
Current posture: G0 passed; G1 `PROCEED`; Phase 0.5/M1 complete at T0/T1; G2 next

Every gate outcome is one of `PROCEED`, `ITERATE`, `NARROW`, or `KILL` and is
recorded in [08-decision-log.md](08-decision-log.md).

## Evidence classifications

| Class | Meaning | Physical-gate authority |
|---|---|---|
| T0 contract | Unit/property/golden | None |
| T1 simulation | Executable reference/simulated workers, including loopback sockets | None |
| Same-host proxy | Real devices represented as logical hosts on one machine | None for physical multinode |
| T2 physical | One physical backend/node | Backend-specific criteria only |
| T3 physical multinode | Real network across physical nodes | G2 criteria |
| T4 target lab | Exact capacity-target heterogeneous fleet | G3/G4 criteria |
| T5 operator | Fresh install/operator evidence | G5 |

## G0 — V4 architecture baseline — PASSED

- **Exit:** plan v4 and the 2026-07-10 review disposition approved; preserved
  decisions retained; assumption-driven execution and physical evidence boundary
  explicit.
- **Authority:** Sponsor.

## G1 — Engine implementation authorization — PROCEED

G1 deliberately does not require unavailable hardware. It authorizes building the
first engine under bounded assumptions.

### Exit criteria

- Plan v4 and review disposition approved.
- Mechanism target plus `simulation-and-assumption-contract.md` reviewable.
- Stage ABI, runtime format, networking/backpressure spec, and core ADRs define one
  coherent implementation.
- Engine v0 scope and out-of-scope product claims are explicit.
- Active Phase 0.5 sprint and functional owners exist.
- Evidence classification prevents simulation/proxy overclaim.

### Decision

`PROCEED` to Phase 0.5 Engine v0. Open SA-* assumptions remain active and block
their corresponding G2/G3 claims. See DEC-005 and the 2026-07-10 gate review.

## G2 — Physical distributed correctness — NEXT

### Entry

- [x] Engine v0 exits on two independent worker processes using the production ABI.
- [x] Planner regressions pass.
- [ ] At least the required physical backend resources become available.

The first two entry conditions are closed by
[DEC-008](08-decision-log.md) and the
[Phase 0.5 exit review](gate-reviews/phase-0-5-exit-2026-07-10.md). Hardware and
the physical `MaxStageBackend` path remain the admission constraint; G2 has not
been convened or passed.

### Exit

- Root-pinned reproducible MAX build lineage.
- Numerical operator/stage parity on each participating backend.
- Correct prefill/decode generation across 2–3 physical nodes.
- Boundary activations and logits within accepted dtype tolerances.
- Real cancellation, timeout, backpressure, stale-plan, and partition behavior.
- Planner predictions within +/-20% over the contracted measured range.
- Apple role decision recorded from physical evidence.
- Aggregate throughput scales with concurrency for the mechanism target.

### Outcomes

- `PROCEED`: select/bind frontier-capacity target.
- `ITERATE`: repair correctness, runtime, network, or calibration.
- `NARROW`: homogeneous islands, capacity-first, or reduced Apple role.
- `KILL`: stop the cross-vendor spanning thesis and retain reusable components.

## G3 — Frontier-capacity heterogeneous target

- Exact model exceeds one node's usable memory and fits the fleet with headroom.
- Real target serves on the accepted heterogeneous roles at contracted
  concurrency and throughput.
- Security/backpressure posture for the deployment is active.
- Backend coverage and planner accuracy meet v4/target-contract bounds.

## G4 — Real resilience and elasticity

- Added physical capacity improves throughput.
- The accepted single-node-loss scenarios meet the target contract's request-loss
  rule.
- Drain, restart, replay, and replacement are proven on the target lab.

## G5 — Product GA

- Fresh operator installs, configures, diagnoses, upgrades, and rolls back the
  service without oral context.
- Benchmark of record and supported/unsupported matrix are published.
- Sponsor accepts the product evidence.

## Gate operation

Use [templates/gate-review.md](templates/gate-review.md). Every review must show
assumptions validated/invalidated, evidence class, unresolved physical gaps, and
the exact scope authorized by `PROCEED`.
