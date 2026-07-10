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
— see [project-plan-v4.md](project-plan-v4.md)). It plugs
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
irreducible price of spanning. See
[project-plan-v4.md](project-plan-v4.md#2-product-hypothesis-and-constraints).

## Documents

| Doc | What it covers |
|---|---|
| [project-plan-v4.md](project-plan-v4.md) | Current plan: assumption-driven Engine v0, production Stage ABI, physical G2 validation, and frontier-capacity roadmap |
| [simulation-and-assumption-contract.md](simulation-and-assumption-contract.md) | Named hardware assumptions, scenario matrix, simulator/backend contract, and physical replacement rules |
| [stage-runtime-and-wire-abi.md](stage-runtime-and-wire-abi.md) | Stable StageExecutable interface and framed TCP tensor protocol |
| [partitioner-spec.md](partitioner-spec.md) | Throughput-optimizing heterogeneous MoE partitioner: stage/expert placement, cost model, search, and contract/golden-vector expectations |
| [deepseek-v2-lite-max-check.md](deepseek-v2-lite-max-check.md) | 2026-07-01 source-built MAX evidence for `deepseek-ai/DeepSeek-V2-Lite-Chat` on Apple Silicon M3 Max, including MLA/MoE/gather blockers cleared and remaining caveats |
| [max-operator-platform-support.md](max-operator-platform-support.md) | Dated MAX operator/platform support map, now annotated with the local Apple DeepSeek MLA smoke result |
| [apple-flare-mla-prefill-port.md](apple-flare-mla-prefill-port.md) | Apple `flare_mla_prefill` porting note and status update: planned design became a local source-built MAX patch with smoke evidence |
| [apple-silicon-max-skills.md](apple-silicon-max-skills.md) | Skill map for MAX/Mojo work on Apple Silicon: platform setup, kernels, custom ops, MoE expert runtime, validation, profiling, transport |
| [program_management/](program_management/) | Charter, roadmap, gates, external watch, RAID log, decision log, sprint plans, and evidence governance |

## Status

The root `fornax/` package is the Python planner plus the completed Phase 0.5
Engine v0 simulation/contract/smoke layer. The production Stage ABI,
reference/simulated MAX backends, two-process worker path, loopback TCP, and
bounded scheduler are closed at T0/T1 so hardware scarcity did not stall M1.
The local `external/modular` checkout carries MAX backend work
for Apple Silicon. As of 2026-07-01, the local source-built MAX tree can run a
short `DeepSeek-V2-Lite-Chat` smoke on an M3 Max after adding Apple MLA prefill,
Apple MLA decode fallback, Apple MoE index fallback, and Apple rank-2 gather
support. That evidence is recorded in
[deepseek-v2-lite-max-check.md](deepseek-v2-lite-max-check.md).

This does **not** close G2 or imply production Apple serving. DEC-008 closes only
the bounded Engine v0 T0/T1 scope. The Apple run is rank-1 local probe
evidence for a single Mac, source-built MAX commit, model,
dtype, prompt size, and short generation count. Remaining work includes
numerical parity, optimized decode/MoE paths, serving-mode validation, longer
contexts, batching, memory-pressure checks, and the target expert-MLP probe that
decides Apple's v0 Fornax role.
