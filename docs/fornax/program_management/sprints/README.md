# Fornax Sprint Backlog

This directory is the execution backlog for the roadmap in
[`03-roadmap-milestones-critical-path.md`](../03-roadmap-milestones-critical-path.md)
and the WBS in [`02-work-breakdown-structure.md`](../02-work-breakdown-structure.md).

Plan v4 rebaselines the backlog around the first executable engine. Phase 0.5
closed the reference/simulated lockstep loopback and experimental FNX1 v1 at
T0/T1. The active lane now replaces assumptions with physical evidence as
hardware becomes available.

## Sprint Index

| Sprint | Window | Milestone / gate | Primary WBS | Current posture |
|---|---:|---|---|---|
| [Phase 0 evidence](phase-0-evidence-sprint.md) | Historical | v3 G1 input | A1-A6, contracts/proxies | Retained; superseded as active work. |
| [Phase 0.5 Engine v0](phase-0-5-engine-v0-sprint.md) | Closed 2026-07-10 | M1 / G1-authorized | A7-A8, B6-B7, E5, J1-J4/J6; A9 carried | **Historical closure complete at T0/T1.** Later A9/FNX2/lifecycle hardening does not rewrite DEC-008. |
| [Phase 1 physical validation](phase-1-worker-contract-transport-sprint.md) | Active when resources exist | M2 / G2 | A9/I-16, B8, D2-D3, J5 | **Active lane.** Authority, working-tree lineage, and FNX2 reference mechanisms exist; commit, authenticated calibration, physical MAX backend, and multinode evidence remain open. |
| [Phase 2 continuous batching](phase-2-continuous-batching-sprint.md) | W9-W13 | M3 / G2 input | F1-F2, G1-G2 | T1 simulations can close implementation risk; real scale evidence remains open. |
| [Phase 2.5 MoE runtime](phase-2-5-moe-runtime-sprint.md) | W11-W17 | M4 / G2 input | C1-C4, H2 | T1/CPU parity can advance; real runtime parity remains G2 evidence. |
| [Phase 3 heterogeneous frontier](phase-3-heterogeneous-frontier-sprint.md) | After G2 | M6 / G3 | capacity target | Frozen except prerequisites. |
| [Phase 4 resilience and elasticity](phase-4-resilience-elasticity-sprint.md) | After G3 | M7 / G4 | real resilience | Existing proxy retained; new scope frozen. |
| [Phase 5 productization and GA](phase-5-productization-ga-sprint.md) | After G4 | M8 / G5 | I1-I3 | Existing proxy retained; new scope frozen. |

## Execution Rule

- Preserve frozen FNX1 evidence and evolve candidate FNX2 across reference,
  simulated, and physical backends; physical validation binds the compatibility
  and rollout decision rather than silently forking the contract.
- Every simulation run names assumptions/scenarios; every physical run names
  hardware/build evidence.
- Keep new Phase 3-5 proxy expansion frozen until a reviewed v4/G2
  reprioritization explicitly releases it.
- Do not mark G2-G5 complete without the evidence in
  [`04-stage-gates.md`](../04-stage-gates.md).
- Keep the detailed completion ledger in
  [`../internal/archive/fornax_program_management_todo_status.md`](../internal/archive/fornax_program_management_todo_status.md).
