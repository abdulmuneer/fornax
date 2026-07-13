# ADR 0006 — Apple Participation Is Assumed for Simulation and Measured for Use

Date: 2026-07-10  
Status: Accepted, physical role open  
Authority: Sponsor + TL  
Plan: `../project-plan-v4.md` §5

## Decision

Engine v0 includes an Apple-shaped simulated worker so Apple requirements affect
the protocol, planner, memory model, and scheduling now. This does not assign a
physical Apple production role.

The physical role ladder is:

1. excluded from hot path;
2. capacity/store only;
3. expert worker;
4. complete pipeline stage;
5. broader native MAX participant.

The highest role passing numerical correctness, stability, memory, and throughput
on the pinned target build is selected at G2/G3.

## Current evidence

Short DeepSeek-V2-Lite generation on patched M3 Max is positive bring-up evidence.
The repeated punctuation output and lack of numerical parity prevent role closure.

## Reversal trigger

If Apple cannot pass correctness or requires an unmaintainable MAX fork, narrow to
capacity-only/excluded without blocking the rest of Engine v0.
