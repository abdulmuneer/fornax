# Fornax Sprint Backlog

This directory is the execution backlog for the roadmap in
[`03-roadmap-milestones-critical-path.md`](../03-roadmap-milestones-critical-path.md)
and the WBS in [`02-work-breakdown-structure.md`](../02-work-breakdown-structure.md).

The original folder held only the active Phase-0 evidence sprint because the
program was governed in a strict G1 authorization mode. The backlog below now
materializes every planned phase so implementation can continue with simulated
and local evidence while real heterogeneous gate evidence is collected later.

## Sprint Index

| Sprint | Window | Milestone / gate | Primary WBS | Current posture |
|---|---:|---|---|---|
| [Phase 0 evidence](phase-0-evidence-sprint.md) | W1-W4 | M1 / G1 | A1-A6, B1, B5, D1-D2/D4, E1, G1, X1-X3 | Repo has substantial T0/T1 evidence; G1 sign-off remains open. |
| [Phase 1 worker contract and transport](phase-1-worker-contract-transport-sprint.md) | W5-W10 | M2 | B2-B4, E2, H3 skeleton, F1 T1 | Sim/local work may proceed; real 2-3 node evidence remains gate work. |
| [Phase 2 continuous batching](phase-2-continuous-batching-sprint.md) | W9-W13 | M3 / G2 input | F1-F2, G1-G2 | T1 simulations can close implementation risk; real scale evidence remains open. |
| [Phase 2.5 MoE runtime](phase-2-5-moe-runtime-sprint.md) | W11-W17 | M4 / G2 input | C1-C4, H2 | T1/CPU parity can advance; real runtime parity remains G2 evidence. |
| [Phase 3 heterogeneous frontier](phase-3-heterogeneous-frontier-sprint.md) | W16-W24 | M5 / G3 | D2-D3, E2-E4, F3, G3, H1 | Two local GPUs can simulate hosts; NVIDIA/AMD/Mac proof remains later. |
| [Phase 4 resilience and elasticity](phase-4-resilience-elasticity-sprint.md) | W23-W27 | M6 / G4 | E4, F3, replay | Passed at two-H100 proxy scope; formal T4 node-loss and added-node proof remain later. |
| [Phase 5 productization and GA](phase-5-productization-ga-sprint.md) | W27+ | M7 / G5 | I1-I3 | Passed at two-H100 proxy scope; formal product GA and Sponsor acceptance remain later. |

## Execution Rule

- Use simulation and local two-GPU logical-host validation to avoid blocking
  engineering milestones.
- Proxy gates may pass using the accepted two-H100 logical-host setup when the
  packet explicitly keeps the formal heterogeneous/lab gate deferred. Do not mark
  formal G2-G5 complete until the gate-specific real hardware evidence in
  [`04-stage-gates.md`](../04-stage-gates.md) exists.
- Keep the detailed completion ledger in
  [`../../../../fornax_program_management_todo_status.md`](../../../../fornax_program_management_todo_status.md).
