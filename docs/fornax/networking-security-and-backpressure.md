# Fornax Networking, Security, and Backpressure

Version: 1.1-draft

Plan: `project-plan-v4.md` §7

Status: Partial T1 implementation. Framed lockstep loopback, explicit
negotiation, one-frame credit, cancellation, typed errors, and control endpoints
exist. Explicit final release, bounded reference retention, opportunistic idle
expiry, internal execution leases, and same-worker reconnect tombstones are also
present at T0/T1. Periodic liveness, integrated scheduler/client backpressure,
durable replay epochs/restart fencing, TLS, and product security remain
requirements.

## 1. Scope and posture

Phase 0.5 first runs as two independent worker processes over loopback with
latency/bandwidth/fault injection. The same protocol later runs on an isolated
trusted lab network with explicitly configured physical workers. Simulation does
not remove node identity, plan integrity, bounds, cancellation, timeout, or audit
requirements.

Public-network exposure, automatic discovery, production certificate lifecycle,
and multi-tenant isolation are later gates.

## 2. Two-plane architecture

| Plane | V0 transport | Responsibilities |
|---|---|---|
| Control | Current T1: plain HTTP/1.1 JSON on loopback | health, backend-originated capabilities, plan identity, release, drain, cancel, status |
| Tensor data | Current T1: persistent loopback TCP using experimental FNX1 v1 | mechanism activations/logits, one-frame credit, ACK, explicit health/shutdown/release heartbeat, error |

The planes are logically separate. A future serving endpoint must not carry
inter-stage tensors; no bundled OpenAI-compatible Fornax endpoint exists today.

## 3. Node admission and identity

The T1 worker reports backend-originated backend/build/device/dtype/ABI/frame
capabilities and attests them against the manifest before load. The following is
the fuller physical/product requirement:

- hostname and physical device identity;
- OS, driver/runtime, MAX/Mojo/Fornax build IDs;
- supported dtypes and maximum frame size;
- available memory and stage/expert/KV capabilities;
- control and data endpoints;
- certificate identity when TLS is enabled.

Current explicit worker construction binds each stage to one configured backend
factory and fails closed on known capability mismatches. Full physical inventory,
certificate identity, and plan-wide node admission remain G2/security work. A
node may not self-assign a role in the target design.

## 4. Plan integrity

- Plans and stage manifests use canonical SHA-256 hashes.
- Every control request and data frame identifies plan and manifest hashes.
- A worker holds at most one active manifest per stage endpoint in v0.
- Stale plan messages return `STALE_PLAN` without execution.
- Plan replacement requires drain/unload/load; in-place mutation is forbidden.
- Evidence records the exact plan bytes and hash.

## 5. Connection lifecycle

```text
DISCONNECTED -> CONNECTING -> NEGOTIATING -> READY -> DRAINING -> CLOSED
                                  |            |
                                  +-> REJECTED +-> FAILED
```

Current negotiation checks exact ABI 1.0 plus plan, manifest, route, and stage
identity, then grants one fixed frame credit. Backend build/device capability
attestation occurs before manifest load, outside the socket handshake. Dynamic
frame-limit and identity negotiation remain open. Application frames are invalid
before `READY`.

Current code supports explicit health/shutdown/release heartbeat control frames;
it does not configure TCP keepalive or run a periodic heartbeat/failure detector.
Those are required before physical resilience claims. A heartbeat is not proof
of stage health; control-plane health separately reports manifest/backend state.

## 6. Bounded flow control

Current lockstep channels enforce one message credit and a byte credit for the
next frame. The table below is the broader product requirement.

| Limit | Configuration/source |
|---|---|
| Maximum metadata | 64 KiB |
| Maximum tensor frame | Manifest/config bound; never inferred from peer length alone |
| Maximum queued messages | Per channel and per destination stage |
| Maximum queued bytes | Per channel, per stage, and process total |
| Maximum in-flight requests | Orchestrator admission limit |
| Maximum unacknowledged bytes | Receiver-advertised credit |
| Reference retained request/result/transform state | Configured live-request, per-request result, global result-byte, transform-entry/byte, and event-history limits |

The T1 receiver returns `CREDIT` after a terminal result and the sender rejects a
send beyond current credit. This is not connected to the separate admission or
continuous-batching simulations, and there is no client endpoint. Integrated
propagation and process-wide/native-KV budgets remain open. Reference/backend
retention health reports current values, configured limits, and byte high-water
marks; that is contract evidence, not a physical allocator measurement.

## 7. Target backpressure propagation (not integrated)

```text
receiver memory/queue
  -> channel credit
  -> sending stage output queue
  -> global microbatch scheduler
  -> request admission / retry-after
  -> client
```

The current mechanism loopback records channel credit events only. The diagram is
the required product path. HTTP `429`, retry metadata, scheduler propagation, and
the admitted-request waiting policy are not implemented in Engine v0.

## 8. Deadlines, timeout, cancellation

- Requests carry one absolute deadline; stages may derive smaller local budgets.
- Expired work is rejected before execution.
- The separate scheduler simulation releases queued reservations on timeout;
  Engine v0 does not integrate that queue with stage execution.
- Cancellation exists on both control and data paths; an ordered end-to-end
  propagation policy remains product work.
- After execution begins, the stage reports whether KV state changed.
- Late-result discard after racing cancellation is a requirement, not yet an
  integrated multiworker test.

## 9. Retry and replay

Current FNX1 retains and exactly replays only the newest completed data-frame
response on an immediate identical retry; a conflicting or older retry fails
closed. Tensor execution is not automatically retried after an ambiguous
disconnect. Durable replay epochs, client-token fences, and recovery from an
acknowledged stage boundary are unimplemented future requirements. Backend
result state survives a new FNX1 connection to the same loaded worker, and final
release installs a count/time-bounded backend tombstone that fences request-ID
reuse there. The fence does not survive worker restart or configured expiry.
The implementation does not evict a non-expired tombstone: capacity exhaustion
fails release/expiry closed and preserves both old fences and live state. A
fence ends only at its configured expiry or worker restart.
Within one negotiated channel, a backend-level logical `SEQUENCE`
rejection restores the consumed credit so final release can still complete; a
wire-integrity conflict remains terminal.

## 10. Failure matrix

| Failure | Required behavior |
|---|---|
| Connection refused | Mark route unavailable; do not admit new work |
| Handshake/build mismatch | Reject node/route; report exact incompatibility |
| CRC or metadata failure | Reject frame; close channel on conflicting duplicate/integrity failure |
| Slow receiver/no credit | Backpressure upstream; bounded wait to deadline |
| Worker process loss | Fail in-flight Phase 0.5 requests unless replay-safe evidence exists |
| Network partition | Fence route; prevent split plan execution; recover through fresh negotiation |
| Stale plan | Reject without graph execution |
| Stage execution error | Return stable error; release buffers; mark worker degraded if repeated |
| Orchestrator loss | **Required, not implemented:** workers stop admitting new executions after lease expiry and drain/fence |

## 11. Loopback and lab security exception

Loopback Engine v0 runs are classified `T1-simulation` even though they use real
sockets and processes. They use fixture identities and contain no production user
data.

The unencrypted Phase 0.5 mode is permitted only when all are recorded:

- isolated non-routed network or direct link;
- trusted physical access and host administrators;
- no production/user-sensitive prompts;
- static allowlisted endpoints;
- plan/node identity checks enabled;
- evidence labeled `lab-unencrypted`.

If these conditions do not hold, control and data channels require TLS with mutual
node authentication. Product deployments always require a separately accepted
ADR-0005 posture.

## 12. Audit and telemetry

Record without tensor contents:

- connection/negotiation and identity outcome;
- plan/manifest install and rejection;
- request, microbatch, sequence, source/destination stage;
- bytes, queue/credit depth, wait and transfer times;
- ACK, timeout, cancellation, reconnect, and error codes;
- memory reservations and releases;
- evidence/build correlation.

Prompts, activations, KV, logits, and model weights are never logged by default.

## 13. Test status

Current T0/T1 regression coverage includes exact-1.0/plan/manifest negotiation,
one-frame credit, deadline and cancellation paths, CRC/truncation/metadata
negatives, exact newest-response replay, conflicting/old duplicate rejection,
malformed and non-finite tensor recovery, final release propagation, in-flight
release rejection, bounded request/result/transform/event retention, and
backend-originated capability attestation. Focused failure regressions also cover
upstream replay after downstream failure, atomic simulated-result finalization,
same-channel release after logical sequence rejection, pre-admission validation,
execution fencing across a partial release retry, idle expiry, late execution-
lease discard, bounded tombstones across a reconnect, and copy-explicit native
staging. A deterministic many-unique-request pressure regression and runnable
wall-duration evidence mode are available in `fornax.lifecycle_pressure`.

The immutable 2026-07-10 EV-009 artifact records a thirty-minute loopback and
fault-injection run against its historical contract. It now reports
`current_contract_authority=false`; it does not prove the new lifecycle bounds.

EV-016 now records 1,800.004556833 seconds of monotonic active churn and 113,718
unique requests within configured bounds. It remains non-authoritative because
the source was uncommitted, and its civil timestamps include a long suspension;
the current runner therefore also records and gates the maximum progress gap.
Still required are a committed-source uninterrupted rerun, restart-durable
replay/fencing, integrated queue-to-client backpressure,
worker-loss and partition recovery on the real engine path, physical native-KV/
RSS lifecycle evidence, and the corresponding physical two-node tests. Passing
reference/simulation fixtures cannot close G2.
