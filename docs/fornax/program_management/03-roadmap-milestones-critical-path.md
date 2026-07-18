# Roadmap, Milestones, and Critical Path

Derived from plan v4 §8. Week offsets are notional and reset at the 2026-07-10
rebaseline; they are sequencing aids, not delivery promises.

## Authorization posture

G1 `PROCEED` authorized Engine v0 under the approved assumption envelope. Phase
0.5 / M1 is now complete at T0/T1. Physical hardware availability constrains the
active Phase 1 lane and blocks G2/G3 validation and supported-platform claims.

## Milestones

| Milestone | Phase | Notional window | Gate/status |
|---|---|---|---|
| M0 v4 architecture/contracts baseline | Rebaseline | W0 | G0 passed |
| M1 Engine v0: two workers, experimental FNX1 v1, loopback transport | 0.5 | Closed 2026-07-10 | **Complete in recorded T0/T1 scope; DEC-008** |
| M2 Physical MAX backend and two-node correctness | 1 | Starts when each backend is available; target W4–W12 in parallel | G2 |
| M3 Physical continuous batching and planner calibration | 1 | After first T3 correctness | G2 |
| M4 Frontier-capacity target selection and fit | 2 | After G2 | G3 input |
| M5 Remote-expert decision, if evidence warrants | 2.5 | After core pipeline measurements | Separate decision |
| M6 Broader heterogeneous target, including AMD if justified | 3 | After capacity target | G3 |
| M7 Real resilience and elasticity | 4 | After G3 | G4 |
| M8 Installable/operator-ready service | 5 | After G4 | G5 |

## Four-lane execution

```text
Engine lane:
plan/contracts -> Stage ABI -> reference/sim backends -> loopback engine -> batching
                                                               |
Validation lane:                                               v
MAX pin/build -> T2 backend parity -> physical T3 replay -> calibrated planner -> G2

Discovery lane: qualified interviews -> workload traces -> design partners -> paid pilot decision

External-readiness lane: team/IP provenance -> MAX rights -> financing model -> approved disclosure package
```

The lanes share manifests, frame protocol, traces, metrics, and the correctness
corpus. Candidate FNX2 and I-22 now cover integrated ragged execution, release,
idle expiry, internal leases, same-worker tombstones, and bounded state at
T0/T1. Restart durability, reviewed long-duration evidence, and physical-backend
validation remain incomplete. Physical adapters must conform to the versioned
contract rather than silently forking the engine design.

## Critical path

1. **Engine path:** B6 Stage ABI -> B7 backends -> E5 transport -> J3 loopback ->
   F1/F2 scheduling/batching -> M1.
2. **Validation path:** D1/MAX pin -> D2 Apple and NVIDIA parity -> B8 physical
   backend -> J5 T3 replay -> A9 calibration -> G2.
3. **Product path:** discovery runs now without capability claims; deployment
   engineering remains G2 -> capacity target budget -> G3 -> resilience -> GA.
4. **External-release path:** XG-1/XG-2 team and rights evidence -> XG-3
   bottoms-up financing -> XG-4…XG-7 bounded claims -> XG-8 approved package.
   Technical progress cannot silently close this path.

## Work allowed now

- Preserve the Stage ABI, planner regressions, and Engine v0 loopback suite as
  mandatory regression evidence.
- Run and review the implemented unique-request pressure harness at sustained
  duration; add restart-durable fencing and a physical native-KV/buffer soak
  (I-22).
- Re-verify the implemented MAX root pin/reconstruction on every clean physical
  evidence host (I-11/R-13).
- Implement `MaxStageBackend` without bypassing `StageExecutable`.
- Run T2 Apple/NVIDIA parity and T3 two-node replay when qualifying resources
  are available.
- Calibrate the planner against measured stage/packing/transport/wait data and
  prepare the G2 review packet.
- Run buyer/workload discovery, traffic-trace collection, design-partner
  qualification, pricing discovery, and pilot definition without claiming a
  supported product.
- Build the private principal/IP facts, component/license map, financing model,
  and disclosure index required by the external-readiness gates.

## Frozen/deferred work

- New Phase 3-5 proxy features.
- Remote expert runtime beyond contract-preserving experiments.
- AMD-specific execution until the core Stage ABI and G2 path are stable.
- Elasticity/product deployment packaging except small implementation
  prerequisites. Customer discovery and offer testing are not frozen.

## Rebaseline rule

This document records the M1 rebaseline. Rebaseline again at G2. Do not insert fictitious hardware dates.
When a physical resource becomes available, add its scheduled validation window
and owner to the active status report and evidence register.
