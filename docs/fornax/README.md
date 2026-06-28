# Fornax — heterogeneous frontier-model serving

**Fornax** is a **Mojo/MAX-native distributed inference engine** — a *custom
surgery of MAX* — that makes a fleet of heterogeneous, commodity machines
(consumer NVIDIA GPUs + Apple Silicon Macs + whatever else is on the LAN) serve a
**single frontier-scale MoE model that no individual node can hold**, at high
*aggregate* throughput, on-prem, for firms that want frontier capability in-house
without shipping data to a provider.

Fornax is an **engine, not a harness.** It is assembled from MAX's own
components — graph compiler, kernel library, KV-cache primitives, the custom-op
API — extended with the pieces MAX does not provide: heterogeneous pipeline
execution, cross-vendor activation/KV transport, and a model-specific
heterogeneous MoE expert runtime for Apple/NVIDIA/AMD workers (the critical path
— see [project-plan.md §5.5](project-plan.md#apple-runtime-readiness)). It plugs
into the Ignis **harness** via the `Engine` trait as a `FornaxBackend`: Ignis
owns the timeline/policy/replay/telemetry; **Fornax owns model execution.**

## The one-line thesis

> Frontier open models are large **sparse MoE** (capacity-bound to store,
> compute-light per token). Heterogeneous commodity hardware is **cheap capacity
> (Mac unified memory) + cheap compute (consumer GPUs)**. The shapes match — the
> bottleneck is the **interconnect**, and the engineering is an engine that hides
> it: MAX's portable kernels under a heterogeneous pipeline of our own.

## What it is not

Fornax is **not** a wrapper that load-balances `max serve`, and it is not Ignis
with a cluster backend. It cuts into the model execution path. The surgical seam
is the MoE block:

```text
hidden states -> router -> local expert batches + remote expert batches
              -> weighted gather -> next layer
```

The dense path (attention, KV, routers, shared/hot experts, sampler) stays on
the fastest local accelerator group whenever possible. Heterogeneous workers
extend model capacity and throughput by running pipeline stages and/or bounded
routed-expert batches, not by pretending every device can participate in every
operation equally.

## The honest constraint (read first)

When the model exceeds the biggest node, **every token crosses the network**.
Even when the network is provisioned for this workload, a spanned model has a
pipeline and synchronization floor that a single-node model does not. Fornax
preserves **aggregate throughput and utilization** (high total tok/s, high
$/token efficiency) via continuous batching, overlap, expert locality, and
balanced stages — *not* single-stream latency parity. That latency cost is the
irreducible price of spanning. See [project-plan.md](project-plan.md#the-constraint).

## Documents

| Doc | What it covers |
|---|---|
| [project-plan.md](project-plan.md) | Vision, scope, architecture, the six throughput levers, phased roadmap, risks, success metrics |
| [partitioner-spec.md](partitioner-spec.md) | The first artifact: the throughput-optimizing heterogeneous MoE partitioner (stage/expert placement, cost model, search) — pure logic, testable with no hardware |
| [apple-silicon-max-skills.md](apple-silicon-max-skills.md) | Skill map for MAX/Mojo work on Apple Silicon: platform setup, kernels, custom ops, MoE expert runtime, validation, profiling, transport |

## Status

Planning. No code yet. These docs are **untracked working notes** (per repo
convention — only `docs/extensions.md` is tracked); do not `git add` them.
