---
title: "Part 2 - MAX Platform Assessment"
header:
  overlay_image: /assets/images/hero-max.svg
  overlay_filter: 0.5
  teaser: /assets/images/hero-max.svg
sidebar:
  nav: "fornax"
---

*Part 2. The role MAX plays on each Fornax node, and how that role is qualified.*

## MAX is the per-node execution substrate

Fornax does not replace MAX inside a machine. MAX supplies the graph compiler,
runtime, kernels, model pipelines, KV-cache primitives, continuous batching, and
serving APIs that already exist for supported models and devices. Mojo provides
the extension path for kernels and custom operations when a model needs work that
the packaged runtime does not yet contain.

MAX also supports homogeneous multi-GPU execution, including tensor, data, and
expert parallel features where the selected model and backend expose them. Fornax
is designed to preserve that local parallelism: after the backend is physically
qualified, a stage may use a homogeneous accelerator island managed by MAX.

The Fornax problem begins at the boundary between unlike nodes. It must decide
which model layers run on each node, move activations between stages, keep request
and KV ownership consistent, and apply backpressure across the whole pipeline.
Those responsibilities sit outside a single MAX runtime.

## Support is a claim about an exact case

A device name in a compatibility table is useful for planning, but it does not
prove that a particular model stage works. Operator coverage changes with the
model, dtype, shape, rank, layout, runtime build, and hardware generation.

Fornax records a backend result with enough information to reproduce it:

```text
model snapshot, layer range, dtype, shape, hardware, OS and driver,
MAX build, command, output, numerical check
```

Sources are considered in this order:

1. A reproducible local run on the pinned build and target hardware.
2. Package requirements and the supported-model catalog for that build.
3. Release notes and changelogs for build-specific behavior.
4. Blog posts and launch announcements as leads for further testing.

When these sources disagree, the target case remains unproven until the local run
passes. This rule keeps changing platform documentation from turning into a
Fornax support claim.

## Qualifying a backend

Backend qualification starts with focused operator tests, then moves to the
actual stage boundary. The focused tests cover the operations that commonly
decide whether a sparse-MoE model can run:

- dense matrix operations, normalization, RoPE, and logits processing;
- router and top-k selection, expert MLP execution, and gather/scatter;
- MHA or MLA prefill and decode;
- KV-cache reads, writes, and remapping;
- quantized weight decoding and backend-native layout conversion.

Passing isolated operators is necessary but not sufficient. A candidate
`MaxStageBackend` must also load the assigned contiguous layer range, match a
known-good stage at its output boundary, and preserve stage-local KV state across
prefill and decode. A non-final stage is checked against reference activations; a
final stage is checked against reference logits. The complete distributed route
must separately pass end-to-end logit validation. The same build must pass
sustained execution and memory checks before it can support a performance claim.

## Recorded backend case study

On 2026-07-01, a source-built MAX runtime completed a short
`deepseek-ai/DeepSeek-V2-Lite-Chat` generation on an M3 Max. The model exposed
missing Apple paths in MLA prefill and decode, MoE bucketing, rank-2 gather, and
host-side dispatch. Narrow correctness-first implementations allowed the model
run to continue and made each backend gap testable.

This was a bounded source-build smoke, not packaged support, numerical parity,
efficient serving, or distributed Fornax execution. Part 4 presents the
[backend case study](./04-model-bring-up.md); the
[Apple DeepSeek runbook](../deepseek-v2-lite-max-check.md) holds the build
identity, captured outputs, and focused tests.

## Assessment outcome

MAX supplies model execution and supported homogeneous parallelism within a
node or accelerator island. Fornax supplies stage selection, cross-node
orchestration, activation transport, request ownership, backpressure, and result
assembly. The baseline pipeline keeps each stage's experts and KV state local.
Remote execution requires a separate contract and evidence that its additional
traffic improves the target workload.

Reference and simulated backends can test the distributed contracts, but they do
not establish physical backend correctness. Platform and performance claims
require numerical validation and measurements on the named hardware and build.

## Sources

- MAX model development overview:
  <https://docs.modular.com/max/develop/>
- MAX graph model:
  <https://docs.modular.com/max/develop/graph/>
- MAX package and platform requirements:
  <https://docs.modular.com/max/packages/>
- MAX supported models:
  <https://docs.modular.com/max/models/>
- MAX release notes:
  <https://docs.modular.com/max/changelog/>
- Fornax project plan:
  [project-plan-v4.md](../project-plan-v4.md)
- Fornax operator evidence map:
  [max-operator-platform-support.md](../max-operator-platform-support.md)
- Apple DeepSeek runbook:
  [deepseek-v2-lite-max-check.md](../deepseek-v2-lite-max-check.md)

---

*Previous: [What Accelerator Support Requires](./01-pytorch-parity-for-new-accelerators.md). Next: [Fornax Architecture](./03-fornax-architecture.md). [Series index](./fornax.md).*
