# Fornax Simulation and Assumption Contract

Version: 1.0  
Plan: `project-plan-v4.md` Phase 0.5  
Status: Phase 0.5 closure baseline; physical assumptions remain binding until replaced by evidence

## 1. Purpose

Hardware scarcity must not stall the first engine. Simulation is therefore a
first-class backend for the production Stage ABI, not a separate architecture.
Every simulated capability and timing value is named, bounded, and linked to the
physical test that will replace it.

Simulation can validate engine behavior, protocol correctness, resource
accounting, scheduling policy, and sensitivity to hardware envelopes. It cannot
validate MAX kernel correctness, real throughput, thermal stability, or a
supported hardware claim.

## 2. Backend interface

All engine paths use the `StageExecutable` contract from
`stage-runtime-and-wire-abi.md`.

| Backend | Purpose | Output rule |
|---|---|---|
| `ReferenceStageBackend` | Slow deterministic correctness oracle | Computes deterministic stage transform and KV epochs |
| `SimulatedMaxStageBackend` | Injected MAX-like service, memory, queue, and failure behavior | Returns reference-equivalent logical output unless fault scenario says otherwise |
| `MaxStageBackend` | Real compiled MAX layer-group execution | Must pass the same backend conformance suite before use |

The orchestrator, transport, framing, admission, ownership, cancellation, tracing,
and cleanup code may not branch on backend type except through declared capability
and measurement interfaces.

## 3. Assumption register

| ID | Assumption | Engine use | Validation/replacement | Gate blocked if open |
|---|---|---|---|---|
| SA-001 | Apple MAX can execute the selected BF16 stage correctly | Enables simulated Apple stage role | T2 operator/stage parity on pinned M3 Max build | G2/G3 |
| SA-002 | NVIDIA MAX can execute the selected BF16 stage correctly | Enables simulated NVIDIA stage role | T2 stage parity on pinned Linux build | G2 |
| SA-003 | MAX permits practical contiguous layer-range construction/loading | Shapes stage manifest and backend interface | First real `MaxStageBackend` compile/load | G2 |
| SA-004 | Logical contiguous activation conversion is available on both backends | Enables ABI payload handoff | T2 pack/unpack conformance | G2 |
| SA-005 | A provisioned local link falls inside the declared latency/bandwidth envelope | Drives planner and scheduler sensitivity | Fabric probe on actual route | G2/G3 |
| SA-006 | Stage-local KV can be owned independently per layer range | Enables distributed decode state model | Physical prefill/decode epoch parity | G2 |
| SA-007 | Target persona can supply concurrency 8 or more | Evaluates pipeline utilization | Operator/traffic study | G3/product |
| SA-008 | 80 GiB NVIDIA and 128 GiB Apple profiles are representative of intended capacity classes | Drives simulated memory feasibility | Exact inventory and measured usable memory | G2/G3 |
| SA-009 | BF16 tolerances in the target contract are meaningful | Enables simulated/reference acceptance schema | Cross-backend reference-error study | G2 |
| SA-010 | A root-pinned MAX fork can be rebuilt on both platforms | Shapes build identity and manifest | Fresh-clone platform builds | G2 |

Assumptions may be refined only by adding a versioned row or evidence reference.
They are never silently rewritten to fit a result.

## 4. Scenario dimensions

Every simulation run declares all dimensions below.

| Dimension | Required points |
|---|---|
| Link nominal rate | 1, 10, 25, 100 Gbit/s |
| RTT | 0.1, 0.5, 1, 5 ms |
| Effective payload factor | 0.5, 0.7, 0.9 of nominal |
| Stage service ratio | Apple stage at 0.25x, 0.5x, 1x NVIDIA stage, plus configurable measured replacement |
| Jitter | 0%, 5%, 20% deterministic seeded distributions |
| Concurrency | 1, 4, 8, 16, 32 |
| Context | 16, 128, 512, 4096 |
| Queue limits | At least tight, nominal, and oversized diagnostic profiles |
| Failure | none, slow stage, disconnect, corruption, timeout, cancel, stale plan |

The stage-service ratios are sensitivity parameters, not claims about either
device. Reports label them `assumed`.

## 5. Named scenarios

| Scenario | Link | Stage ratio | Jitter | Purpose |
|---|---|---:|---:|---|
| `S-WORST-DESKTOP` | 1 Gbit/s, 5 ms RTT, 50% payload factor | 0.25x | 20% | Reject designs that require ideal networking |
| `S-DESKTOP` | 10 Gbit/s, 1 ms RTT, 70% payload factor | 0.5x | 5% | Plausible readily available local setup |
| `S-PROSUMER-25` | 25 Gbit/s, 0.5 ms RTT, 70% payload factor | 0.5x | 5% | Candidate entry fabric |
| `S-PROSUMER-100` | 100 Gbit/s, 0.1 ms RTT, 90% payload factor | 1x | 5% | Intended high-end local fabric |
| `S-COMPUTE-SKEW` | 100 Gbit/s, 0.1 ms RTT, 90% payload factor | 0.25x | 5% | Stage-balance/replication pressure |

Custom scenarios are allowed, but gate reports include all named scenarios.

## 6. Deterministic simulation mechanics

- Use a virtual monotonic clock for T1 scheduling and resource events.
- Seed all jitter/fault decisions and record the seed.
- Compute payload bytes from the real runtime-format descriptor.
- Apply pack, queue, wire, unpack, and execute service separately.
- Enforce actual configured message/byte credits and memory reservations.
- Produce the same trace/metrics schema intended for physical backends.
- Support real multi-process loopback mode in addition to virtual-clock mode.
- Validate logical outputs against `ReferenceStageBackend` regardless of injected
  timing.

## 7. Assumption replacement

When physical evidence arrives:

1. add it to `evidence-register.md`;
2. create a measured calibration profile with environment/build identity;
3. link the profile to the affected SA-* row;
4. retain the old scenario for sensitivity/regression;
5. mark the assumption `validated`, `invalidated`, or `partially validated` in the
   program RAID log;
6. rerun the full scenario matrix with the measured point highlighted.

Measured evidence does not erase uncertainty outside its hardware, shape,
context, concurrency, or thermal range.

## 8. Engine v0 exit criteria

- Both backend implementations pass the same Stage ABI conformance suite.
- Multi-process loopback executes prefill/decode through two workers.
- Credits, deadlines, cancellation, stale plans, corruption, disconnect, and
  cleanup behave as specified.
- Planner regression cases pass and all selected plans cite scenario/assumption
  provenance.
- Named scenarios complete for the contracted workload points.
- No queue or modeled allocation exceeds its configured bound.
- Reports distinguish reference truth, simulated behavior, assumptions, and
  physical evidence.

Meeting these criteria authorizes continued engine development. It does not close
SA-001 through SA-010 or G2 physical distributed correctness.
