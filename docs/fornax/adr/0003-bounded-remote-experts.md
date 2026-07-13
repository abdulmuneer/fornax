# ADR 0003 — Remote Experts Are Deferred Measured Optionality

Date: 2026-07-10  
Status: Accepted  
Authority: TL  
Plan: `../project-plan-v4.md` §4.3

## Decision

Engine v0 and the baseline physical pipeline keep routed experts with their owning
stage. Remote expert execution and migration may be simulated after the core
engine works, but planner placement remains disabled until physical evidence and a
separate enablement decision exist.

## Enablement criteria

- Expert host memory, compute, queues, and network contention are modeled.
- Routing/gather parity passes the reference corpus.
- Remote execution improves saturated aggregate throughput by a target-contract
  threshold after all transfer and synchronization costs.
- The path fails closed when assumptions are unmeasured.

## Consequences

Existing MoE simulation fixtures are retained as contract tests. They are not the
current implementation critical path.

## Reversal trigger

None required to keep remote experts off. Enabling them requires a new accepted
decision record with measured evidence.
