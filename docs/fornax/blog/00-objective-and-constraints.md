---
title: "Part 0 - Objective and Constraints"
header:
  overlay_image: /assets/images/hero-max.svg
  overlay_filter: 0.5
  teaser: /assets/images/hero-max.svg
sidebar:
  nav: "fornax"
---

*Part 0. Project objective, constraints, and the evidence standard.*

## Objective

Fornax is being built toward serving one frontier-scale sparse-MoE model that
does not fit on any single node in the target fleet. The target product would
combine several machines into one local inference service while preserving the
differences between their devices, memory systems, and runtimes. The repository
does not yet provide that physical service.

The candidate fleet includes:

- consumer and prosumer NVIDIA GPUs
- Apple Silicon Macs with large unified memory
- AMD GPUs after the required MAX and ROCm paths pass physical validation
- CPUs for control, validation, and low-throughput fallback

The primary metric is aggregate throughput at a stated concurrency.

## Why MoE

Sparse MoE separates total model capacity from the compute used by each token. A
token passes through dense layers, router logic, shared experts where the
architecture defines them, and a small selection of routed experts. The full
model still needs enough aggregate memory,
but only part of the expert capacity is active for each token.

The target design first partitions the model into complete contiguous layer
ranges. Each physical stage would keep the attention, KV state, and experts for
its assigned layers. Between compute stages, the tensor handoff would be an
activation frame and the true final stage would return vocabulary logits, while
control frames carry flow-control and failure state. Today's FNX1 loopback moves
mechanism tensors with activation/logit labels; it is not real model execution
or a cross-vendor path.

Remote expert execution is deferred. It will be considered only after the stage
pipeline works on physical hardware and measurements show that remote dispatch
improves aggregate throughput after packing, transport, and synchronization costs.

## Hard constraint

If the model spans nodes, every generated token crosses at least one stage
boundary. Network transfer and synchronization are structural costs. The target
product therefore aims to optimize:

- pipeline balance
- continuous batching
- communication and compute overlap
- bounded queues and backpressure
- placement accuracy

It does not promise single-stream latency parity for a spanned model.

## MAX and Fornax responsibilities

MAX provides the per-node execution substrate:

- graph construction and compilation
- kernels and custom ops
- device memory and local execution
- backend-local KV primitives
- supported model architecture implementations

The target Fornax ownership is the cross-node engine:

- hardware inventory and measured links
- planner and placement search
- global stage orchestration and request ownership
- versioned activation framing and cross-node transport
- scheduling, admission control, cancellation, and backpressure
- distributed result assembly and evidence collection

The public facade keeps the intended engine interface independent of the serving
layer through an explicit `str -> str` callable. No bundled physical generator
or OpenAI-compatible Fornax server exists yet; Ignis or another control plane can
use the same seam only after a real generator is supplied.

## Evidence boundary

Fornax develops physical MAX backends against an executable contract layer.
Reference and simulated backends exercise the planner, Stage ABI, transport
framing, orchestration, and failure rules without making hardware claims.
Physical MAX backends must satisfy the same contracts.

Cost-model throughput is a prediction. Reference and simulated backend runs can
produce real observations about process, socket, queue, and failure behavior, but
they are not hardware measurements. A platform support or performance claim
requires a physical run that records the model, hardware, MAX build, dtype, shape,
command, and numerical validation. The [README](../../../README.md) records the
current implementation and milestone status.

---

*Next: [What Accelerator Support Requires](./01-pytorch-parity-for-new-accelerators.md). [Series index](./fornax.md).*
