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
| M1 Engine v0: two workers, production ABI, loopback transport | 0.5 | Closed 2026-07-10 | **Complete at T0/T1; DEC-008** |
| M2 Physical MAX backend and two-node correctness | 1 | Starts when each backend is available; target W4–W12 in parallel | G2 |
| M3 Physical continuous batching and planner calibration | 1 | After first T3 correctness | G2 |
| M4 Frontier-capacity target selection and fit | 2 | After G2 | G3 input |
| M5 Remote-expert decision, if evidence warrants | 2.5 | After core pipeline measurements | Separate decision |
| M6 Broader heterogeneous target, including AMD if justified | 3 | After capacity target | G3 |
| M7 Real resilience and elasticity | 4 | After G3 | G4 |
| M8 Installable/operator-ready service | 5 | After G4 | G5 |

## Two-lane execution

```text
Engine lane:
plan/contracts -> Stage ABI -> reference/sim backends -> loopback engine -> batching
                                                               |
Validation lane:                                               v
MAX pin/build -> T2 backend parity -> physical T3 replay -> calibrated planner -> G2
```

The lanes share the same manifests, frame protocol, request lifecycle, traces,
metrics, and correctness corpus. Physical work replaces assumptions; it does not
fork the engine design.

## Critical path

1. **Engine path:** B6 Stage ABI -> B7 backends -> E5 transport -> J3 loopback ->
   F1/F2 scheduling/batching -> M1.
2. **Validation path:** D1/MAX pin -> D2 Apple and NVIDIA parity -> B8 physical
   backend -> J5 T3 replay -> A9 calibration -> G2.
3. **Product path:** G2 -> capacity target budget -> G3 -> resilience -> GA.

## Work allowed now

- Preserve the Stage ABI, planner regressions, and Engine v0 loopback suite as
  mandatory regression evidence.
- Root-pin the MAX fork/build lineage (I-11).
- Implement `MaxStageBackend` without bypassing `StageExecutable`.
- Run T2 Apple/NVIDIA parity and T3 two-node replay when qualifying resources
  are available.
- Calibrate the planner against measured stage/packing/transport/wait data and
  prepare the G2 review packet.

## Frozen/deferred work

- New Phase 3-5 proxy features.
- Remote expert runtime beyond contract-preserving experiments.
- AMD-specific execution until the core Stage ABI and G2 path are stable.
- Elasticity/product packaging except small implementation prerequisites.

## Rebaseline rule

This document records the M1 rebaseline. Rebaseline again at G2. Do not insert fictitious hardware dates.
When a physical resource becomes available, add its scheduled validation window
and owner to the active status report and evidence register.
