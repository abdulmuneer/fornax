# ADR 0002 — Pipeline-Parallel Spanning Spine

Date: 2026-07-10  
Status: Accepted  
Authority: TL  
Plan: `../project-plan-v4.md` §4.3

## Decision

Cross-vendor execution uses contiguous complete layer-group stages. Communication
occurs at stage-boundary activations. Each stage owns the KV for its layers and,
for Engine v0, all experts required by those layers.

Homogeneous devices inside one node/island may use native MAX parallelism, but the
island presents one Stage ABI endpoint to the spanning pipeline.

## Rationale

This bounds communication and synchronization compared with cross-vendor tensor
parallelism or per-layer expert all-to-all. It also creates an explicit compiler
and transport boundary that can be simulated before every backend is available.

## Consequences

- The planner assigns contiguous layer ranges and balances stage service time.
- Single-request latency includes every stage/network boundary.
- Continuous batching is required for aggregate utilization.
- Non-contiguous stages require a future ADR.

## Reversal trigger

Reopen only if measured pipeline utilization cannot reach the target envelope and
a competing strategy demonstrates better correctness-preserving throughput on the
same fleet.
