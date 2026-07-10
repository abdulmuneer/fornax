# ADR 0007 — Prefill/Decode Disaggregation Deferred

Date: 2026-07-10  
Status: Accepted deferral  
Authority: TL  
Plan: `../project-plan-v4.md` §4.3

## Decision

Engine v0 uses the same stage placement for prefill and decode. Separate
prefill/decode fleets, KV transfer, and phase-specific placement are out of scope.

## Rationale

Disaggregation adds KV movement and lifecycle complexity before the basic Stage
ABI, scheduling, and physical cross-vendor path are proven.

## Reversal trigger

Reconsider after G2 if measured prefill/decode imbalance materially limits the
frontier-capacity target and a KV-transfer design passes correctness and cost
analysis.
