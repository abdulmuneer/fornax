---
title: "Part 3 - Fornax Architecture"
header:
  overlay_image: /assets/images/diagram-system-model.svg
  overlay_filter: 0.5
  teaser: /assets/images/diagram-system-model.svg
sidebar:
  nav: "fornax"
---

*Part 3. The engine boundary, stage pipeline, runtime contract, and evidence model.*

## An explicit facade and a target engine boundary

The public Python `Engine` is a small string-in/string-out facade over an
explicitly supplied generator. It does not load a model, start workers, tokenize,
sample, or select a physical/simulated backend. No bundled Fornax text generator,
serving endpoint, or Ignis integration exists yet.

That boundary keeps the public contract small:

```text
exact input string
  -> caller-supplied generator
  -> string result
```

Backend exceptions pass through; a non-string result is a contract error. The
target product would later connect the same small public boundary to an
installed, versioned multiworker plan, but Engine v0 currently begins at
activation tensors. See the
[adapter guide](../stage-backend-adapters.md#text-engine-boundary).

## MAX and Fornax ownership

The intended ownership is: MAX executes a physical model stage on one qualified
backend; Fornax makes several stages behave as one model execution. Only the
reference/simulated Fornax side and single-node MAX bring-up are evidenced today.

| Area | MAX | Fornax |
|---|---|---|
| Model graph | Build, compile, select kernels, and execute on a device or supported homogeneous group | Select the contiguous layer range and create its manifest |
| Kernels | Built-in kernels and Mojo custom operations | Cross-node boundary logic that is not supplied by MAX |
| Memory | Backend tensors and local execution buffers | Payload ownership, admission budgets, and release across nodes |
| KV cache | Stage-local implementation | Request ownership, stage epochs, cleanup, and recovery policy |
| Scheduling | Backend-local execution primitives | Global stage and microbatch coordination |
| Transport | Local runtime and device transfers | Cross-node control messages and tensor frames |
| Serving | Useful single-node baseline | Distributed request lifecycle and result assembly |

The architecture preserves MAX homogeneous multi-GPU features inside a stage
after the relevant backend is physically qualified. Fornax does not replace
those collectives with a cross-vendor implementation.

## Contiguous stages form the cross-node pipeline

The cross-vendor design uses pipeline parallelism by complete contiguous
layer groups. Each worker loads one `StageExecutable` for its assigned range.
Experts belonging to those layers remain with the stage.

```text
client
  -> Fornax gateway and orchestrator
  -> stage 0: MAX layers 0..k on worker A
  -> versioned activation frame
  -> stage 1: MAX layers k+1..n on worker B
  -> logits and sampler
  -> response
```

The same route handles prefill and decode. Each stage receives the request,
plan, microbatch, token-position, deadline, trace, and KV-epoch metadata required
to interpret an activation. It returns either the next activation or final
logits, together with the resulting KV epoch and local timing fields.

Network operations occur outside compiled MAX graphs. A graph may pack or unpack
a boundary tensor, but it does not perform a blocking network call. This keeps
the MAX execution unit testable and keeps transport failures in the Fornax
runtime where they can be bounded and reported.

## The Stage ABI

Every worker first loads a `StageManifest`. The manifest binds an immutable model
snapshot, tokenizer and template hashes, MAX build, plan hash, layer range,
input/output tensor contract, device requirements, weight artifacts, and
stage-local KV policy.

The versioned experimental `StageExecutable` contract covers capabilities, load,
health, execute, cancel, release, drain, and unload. An execution request carries the identity and state
needed to reject stale or malformed work before graph execution. A result uses
stable statuses and error codes so the orchestrator does not have to parse
backend-specific exception text.

The wire protocol wraps metadata and tensor bytes in a versioned frame with
bounded lengths, a monotonic sequence number, and a frame checksum over metadata
and payload. Metadata binds the frame to a plan, manifest, request, microbatch,
source and destination stage, token interval, KV epoch, tensor descriptor, and
deadline. The receiver validates those fields before giving data to the backend.

Acknowledgements, credits, cancellation, errors, and heartbeats use the same
framing rules. Each ABI version accepts only the payloads it defines and rejects
unsupported kinds before execution. The byte layout and supported message kinds
remain in the [Stage ABI specification](../stage-runtime-and-wire-abi.md).

## Independent workers and bounded transport

Stages run in independent worker processes. Control messages and
tensor data use separate logical planes. Tensor frames travel over persistent
TCP connections rather than opening a connection for every activation.

Each lockstep T1 channel has one message credit and a byte credit for the next
frame. The separate admission and continuous-batching simulations are not yet
integrated with channel credit or a client endpoint. The target design propagates
pressure upstream; current evidence establishes local channel enforcement and
bounded histories.

Plans and manifests are immutable during execution. Sequence checks, hashes,
shape checks, byte counts, and CRCs fail closed before a stage runs. Deadlines
and cancellation have T1 mechanism tests. An immediate exact duplicate can
replay the newest cached response without re-execution; older/conflicting
retries fail closed. Cross-connection replay epochs and client-token fences are
not implemented, so ambiguous partial execution is not recoverable today.

Reference and simulated MAX backends exercise these behaviors over loopback TCP.
Physical distributed-path qualification must repeat the same failure and
ownership checks over the deployed channel.

## KV remains stage-local

Each pipeline stage owns the KV pages for its own layer range. A request carries
the expected stage KV epoch, and a result reports the epoch before and after
execution. Cancellation and failure handling therefore have enough information
to determine whether state changed.

The baseline pipeline does not move KV pages between stages or nodes. Keeping KV
local avoids a second distributed state-transfer problem in the cross-node path.
Migration can be added only with a separate ownership, format, replay, and
measurement contract.

## Planner and calibration

The planner receives model dimensions, stage/expert memory, dtype, context,
concurrency, usable memory, modeled compute/link/packing costs, and selected node
capabilities. It searches for a feasible sequence of stages and explains modeled
rejections.

Feasibility and performance are separate decisions. Current placement enforces
memory/resource accounting and the KV-capable-node regression. Runtime/build,
operation, quantization, and calibration admission remain open under I-16; they
must not be described as enforced. For a modeled route, the planner estimates
service, packing, wire, unpacking, queueing, and exposed wait.

A prediction based on defaults or vendor peak figures is uncalibrated. It can be
used to explore a design, but it cannot authorize deployment or close a gate.
Physical profiles must name the exact hardware, build, model snapshot, layer
range, dtype, shape, context, concurrency, and measurement command. The current
planner does not encode provenance/confidence or automatically fail an
unmeasured operation or route; program governance therefore treats every
current plan as exploratory until I-16 is implemented.

## Evidence sources

Fornax separates evidence by where it comes from:

| Class | What it establishes | What it does not establish |
|---|---|---|
| Contract tests and golden vectors | Deterministic schemas, framing rules, planner regressions, and failure validation | Process behavior or hardware correctness |
| Reference and simulated execution | Independent lockstep workers, persistent sockets, one-frame credit, cancellation/timeouts/protocol negatives, explicit final release, plus separate queue/batching simulations | Integrated scheduling, reconnect leases/tombstones, MAX kernel correctness, physical network speed, or product throughput |
| Physical distributed-path validation | Cross-backend activation and logit parity, measured stage and transport costs, and planner calibration on named hardware | Frontier-capacity or production readiness unless separately validated |

Reference and simulated observations may include real process, socket, queue,
and wall-clock data, but they remain evidence about those backends. Cost-model
throughput is a prediction. Hardware throughput becomes a measurement only when
the artifact names the physical model, fleet, build, and command.

The product target is aggregate throughput at the concurrency needed to fill the
pipeline. A model that spans machines has an unavoidable network and
synchronization floor, so Fornax does not promise single-stream latency parity
with a model that fits on one node.

## Physical qualification and deferred extensions

A physical `MaxStageBackend` is qualified with the backend and wire conformance
corpus plus operator and stage-output tests on its named hardware and build. The
distributed route is qualified separately through repeated generation, boundary
and logit parity, stage and transport measurements, planner calibration, and a
measured role for each backend.

Remote expert execution is deferred until the contiguous-stage path is measured.
It will be enabled only if batching and locality overcome its routing, transport,
and synchronization cost. Expert migration, quantized activation transport,
distributed KV movement, RDMA-class transports, replication, and elasticity also
require their own correctness and measurement gates.

## Sources

- Fornax architecture plan:
  [project-plan-v4.md](../project-plan-v4.md)
- Stage runtime and wire protocol:
  [stage-runtime-and-wire-abi.md](../stage-runtime-and-wire-abi.md)
- Logical tensor and ownership rules:
  [runtime-format-and-invariants.md](../runtime-format-and-invariants.md)
- Network, failure, and backpressure rules:
  [networking-security-and-backpressure.md](../networking-security-and-backpressure.md)
- Planner calibration contract:
  [cost-model-and-calibration.md](../cost-model-and-calibration.md)
- Simulation boundary:
  [simulation-and-assumption-contract.md](../simulation-and-assumption-contract.md)
- Repository status:
  [README.md](../../../README.md)

---

*Previous: [MAX Platform Assessment](./02-max-platform-assessment.md). Next: [Model Bring-Up](./04-model-bring-up.md). [Series index](./fornax.md).*
