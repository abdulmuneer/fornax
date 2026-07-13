# ADR 0005 — Security Posture by Evidence Tier

Date: 2026-07-10  
Status: Accepted for Engine v0; product posture deferred  
Authority: Sponsor + NET  
Plan: `../project-plan-v4.md` §7

## Decision

Engine v0 uses fixture identities over loopback. Physical Phase 1 validation may
use an unencrypted isolated trusted-lab exception when no production user data is
present. Node identity, plan/manifest integrity, bounded queues, deadlines,
cancellation, and audit metadata remain mandatory in every mode.

Production deployments require authenticated encrypted control/data channels,
certificate lifecycle, node admission, endpoint authentication, and a threat
model reviewed before G3/product exposure.

## Consequences

- Simulation includes identity, stale-plan, corruption, partition, and replay
  failures.
- Lab-unencrypted evidence is explicitly labeled.
- Activations, KV, logits, prompts, and weights are not logged by default.

## Reversal trigger

If the network is routed/shared, physical access is untrusted, or sensitive data
is used, the lab exception is invalid and mTLS is required immediately.
