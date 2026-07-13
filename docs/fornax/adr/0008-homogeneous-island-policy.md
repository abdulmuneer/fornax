# ADR 0008 — Homogeneous Islands Use Native MAX Parallelism

Date: 2026-07-10  
Status: Accepted policy; implementation follows backend evidence  
Authority: TL  
Plan: `../project-plan-v4.md` §4

## Decision

Multiple compatible devices within one node or tightly coupled homogeneous island
should use native MAX tensor, data, or expert parallel features where supported.
Fornax treats the island as one stage endpoint and does not recreate internal
collectives.

## Consequences

- Inventory exposes island topology and one external Stage ABI capability.
- Planner calibration uses the island's measured aggregate stage profile.
- Cross-vendor spanning remains pipeline parallel between island endpoints.

## Reversal trigger

Use a Fornax-managed internal strategy only if MAX cannot express the required
model and a measured alternative justifies the maintenance cost.
