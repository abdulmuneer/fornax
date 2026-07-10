# Fornax Networking, Security, and Backpressure

Version: 1.0-draft  
Plan: `project-plan-v4.md` §7  
Status: Loopback implementation complete at T1; physical T3 and product-security evidence pending

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
| Control | HTTP/1.1 JSON on a dedicated configured port; TLS/mTLS optional in isolated lab | health, capabilities, plan install, drain, cancel, status |
| Tensor data | Persistent TCP connections using Stage ABI v1 frames | activations, logits, credit, ACK, heartbeat, error |

The planes are logically separate even when a future implementation multiplexes
them. Tensor payloads never travel through the OpenAI client endpoint.

## 3. Node admission and identity

Each worker has a configured `node_id` and reports:

- hostname and physical device identity;
- OS, driver/runtime, MAX/Mojo/Fornax build IDs;
- supported dtypes and maximum frame size;
- available memory and stage/expert/KV capabilities;
- control and data endpoints;
- certificate identity when TLS is enabled.

The orchestrator admits only nodes listed by the installed plan. Node identity,
device identity, and build compatibility are checked before a stage manifest is
installed. A node may not self-assign a role.

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

Negotiation exchanges ABI versions, node/build identities, frame limits, initial
credits, and plan/manifest hashes. Application frames are invalid before `READY`.

Connections use TCP keepalive plus application heartbeat. A heartbeat is not
proof of stage health; control-plane health separately reports graph/manifest
state.

## 6. Bounded flow control

Every channel enforces both message and byte credits.

| Limit | Configuration/source |
|---|---|
| Maximum metadata | 64 KiB |
| Maximum tensor frame | Manifest/config bound; never inferred from peer length alone |
| Maximum queued messages | Per channel and per destination stage |
| Maximum queued bytes | Per channel, per stage, and process total |
| Maximum in-flight requests | Orchestrator admission limit |
| Maximum unacknowledged bytes | Receiver-advertised credit |

The receiver sends `CREDIT` only after it has capacity. The sender must not enqueue
beyond credit. Credit exhaustion propagates to the global scheduler and client
admission; buffering is not allowed to grow without bound.

## 7. Backpressure propagation

```text
receiver memory/queue
  -> channel credit
  -> sending stage output queue
  -> global microbatch scheduler
  -> request admission / retry-after
  -> client
```

The orchestrator records where pressure originated. HTTP `429` may be used at the
client boundary with bounded retry metadata. An already admitted request is not
silently rejected because a downstream queue filled; it waits within its deadline
or terminates with a stable error.

## 8. Deadlines, timeout, cancellation

- Requests carry one absolute deadline; stages may derive smaller local budgets.
- Expired work is rejected before execution.
- A timeout while queued releases reservations immediately.
- Cancellation is propagated control-plane first and data-plane as `CANCEL` for
  correlation/recovery.
- After execution begins, the stage reports whether KV state changed.
- A late successful result for a cancelled request is acknowledged for buffer
  release but discarded by the orchestrator.

## 9. Retry and replay

V0 retries connection establishment and idempotent control reads. Tensor execution
is not automatically retried after ambiguous failure.

Replay is allowed only when:

1. the orchestrator has the last acknowledged stage boundary;
2. all participating stage KV epochs are known;
3. the new replay epoch is installed;
4. the request has not emitted an irreversible client token beyond that boundary.

Otherwise the request fails with an explicit retryability classification.

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
| Orchestrator loss | Workers stop admitting new executions after lease expiry and drain/fence |

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

## 13. Phase 0.5 tests

- Version/build/plan negotiation success and rejection.
- Credit exhaustion, recovery, bounded queues, and upstream admission response.
- Deadline before enqueue, while queued, and during stage execution.
- Cancellation before and after KV mutation.
- CRC failure, truncation, conflicting duplicate, and stale plan.
- Worker loss and network partition fencing.
- Reconnect with fresh negotiation and no duplicate execution.
- Thirty-minute sustained run with queue and memory bounds.

Passing simulation fixtures is necessary but insufficient; the same failure
classes must be exercised on the physical two-node path where safe.
