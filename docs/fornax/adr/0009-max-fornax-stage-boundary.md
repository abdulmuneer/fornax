# ADR 0009 — MAX/Fornax Stage Boundary

Date: 2026-07-10  
Status: Accepted  
Authority: Sponsor + TL  
Plan: `../project-plan-v4.md` §4.1

## Decision

MAX owns per-node graph construction/compilation, kernel dispatch, device-local
execution buffers, and backend-local runtime primitives. Fornax owns target and
placement planning, global request/microbatch orchestration, worker lifecycle,
cross-node transport, plan integrity, distributed state ownership, and evidence.

The stable seam is `StageExecutable`. Engine v0 supplies reference and simulated
implementations; physical workers supply `MaxStageBackend` later. The engine code
uses declared capabilities and does not contain backend-specific scheduling forks.

## Consequences

- Fornax does not build a second per-device compiler or kernel runtime.
- MAX serving may remain a baseline, but the distributed gateway is a Fornax
  surface because it coordinates multiple stages.
- Stage-local KV implementation should reuse MAX where accessible; Fornax tracks
  ownership/epoch, not backend page internals.
- Network I/O remains outside compiled graphs.

## Reversal trigger

Reopen if public MAX interfaces cannot support layer-range execution and KV state
without pervasive unstable internal hooks. Any broader fork requires Sponsor/TL
review under ADR-0001.
