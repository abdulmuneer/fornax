# ADR 0004 — V0 Control and Tensor Transport

Date: 2026-07-10  
Status: Accepted for Engine v0  
Authority: TL + NET  
Plan: `../project-plan-v4.md` §7

## Decision

- Control plane: HTTP/1.1 JSON on configured endpoints.
- Tensor data plane: persistent TCP using Stage ABI v1 frames.
- First implementation: independent worker processes over loopback with injected
  latency, bandwidth, corruption, and disconnects.
- Physical implementation: the identical protocol over an isolated local link.

Networking remains outside compiled MAX graphs. V1 permits backend-buffer copies;
zero-copy is not a prerequisite.

## Rejected alternatives for v0

- Blocking network RPC inside a MAX custom op.
- RDMA/UCX/NIXL before measured TCP attribution.
- Kubernetes as a prerequisite for two-node execution.
- Unbounded pickle/JSON tensor payloads.

## Consequences

The Stage ABI defines framing, ordering, checksum, acknowledgement, credit, and
compatibility. Faster transports must preserve its logical semantics.

## Reversal trigger

Adopt another data plane only after physical measurements show TCP prevents the
target and the replacement is available across the required operating systems.
