# Fornax operator runbook

## Current support boundary

The repository is pre-alpha. Planner, contract, and T0/T1 Engine v0 workflows
are runnable. Physical heterogeneous serving, GA lifecycle operations, and
production security are not supported yet.

## Deploy and verify

Build a preflight bundle from an explicit target and inventory. Run `fornax
doctor --bundle <dir>` and resolve every error. Warnings that classify evidence
as simulation or proxy must remain visible.

## Drain and mutate

For future physical deployments, stop admission, drain in-flight work, record
the active plan/build identities, then upgrade, restart, rollback, or replace a
node. Do not mutate a live stage in place.

## Recover and roll back

Reinstall the last accepted immutable plan and backend build. A worker must fail
plan installation if its discovered capabilities or build identity do not match
the manifest.

## Escalate

Escalate any numerical divergence, dropped request, unbounded queue, identity
mismatch, silent fallback, or claim whose evidence tier is weaker than its gate.
