---
title: "Part 4 - Model Bring-Up"
header:
  overlay_image: /assets/images/hero-max.svg
  overlay_filter: 0.5
  teaser: /assets/images/hero-max.svg
sidebar:
  nav: "fornax"
---

*Part 4. What model support means, using a DeepSeek-V2-Lite run on Apple
Silicon as a case study.*

## A model-support record

Model support depends on the model, runtime, device, and workload. A useful
support record contains:

```text
model id
architecture
weights source
dtype / quantization
hardware
OS / driver / runtime
MAX build
command
input shape, context, batch, and output-token limit
result
validation method
claim boundary
```

The command and its full output belong in a runbook or evidence bundle. The
article can then discuss the engineering result without turning into an
installation log.

## Model-support validation levels

| Level | What it establishes |
|---|---|
| Model catalog entry | Documentation advertises the architecture or model family; installed runtime behavior remains unverified. |
| Graph compile | The recorded graph and shapes lower and compile for the selected target. |
| Focused kernel test | One operator path runs for a stated dtype, shape, and device. |
| Short generation smoke | The model loads, compiles, and emits tokens for one bounded configuration. |
| Numerical validation | Operator outputs, stage activations, or logits agree with a reference within a stated tolerance. |
| Serving validation | The server handles the recorded request, batching, streaming, cancellation, and recovery cases. |
| Target-system run | The defined model, fleet, and workload run on physical hardware with measured results. |

Each level answers a different question. A catalog entry cannot establish device
coverage, and a short generation smoke cannot establish numerical accuracy or
serving performance.

## Case study: DeepSeek-V2-Lite on M3 Max

DeepSeek-V2-Lite exercises Multi-head Latent Attention, routed experts, and the
indexing operations that connect the router to the expert MLPs. That combination
made it a useful test of MAX's Apple backend. The model was present in MAX's
architecture registry, but a complete run also required compatible graph
lowering, kernels, launch planning, and host-side dispatch.

### Tested configuration

| Field | Value |
|---|---|
| Date | 2026-07-01 |
| Model | `deepseek-ai/DeepSeek-V2-Lite-Chat` |
| Architecture | `DeepseekV2ForCausalLM` |
| Weights | Local four-shard BF16 snapshot |
| Hardware | Apple M3 Max, 40-core GPU, 128 GB unified memory |
| Runtime | Source-built `MAX 26.5.0.dev2026063006` |
| Workload | Prompt size 14; separate runs with 1 and 8 output tokens |
| Result | `max generate` completed and emitted tokens in both runs |

The [Apple DeepSeek runbook](../deepseek-v2-lite-max-check.md) records the build
environment, captured output, and focused kernel checks. The
[getting-started guide](../../getting-started.md) contains the successful Apple
invocation. The smoke artifact did not capture the exact macOS and Metal
toolchain versions, so it is not a complete support record.

## Backend limitations and resolutions

The test exercised several independent parts of the backend. Device support
required all of them to agree.

| Subsystem | Framework limitation | Resolution used for the test | Production implication |
|---|---|---|---|
| MLA prefill | The local source tree lacked an Apple route for the DeepSeek MLA shape. | An Apple-compatible MLA prefill and dispatch route covered the BF16/FP16 shape used by DeepSeek-V2-Lite. | Other chips, dtypes, and shapes need their own validation and dispatch policy. |
| MLA decode | The decode gate accepted NVIDIA and AMD targets, but not Metal. | A correctness-first Apple attention path read the MLA cache row and copied the required latent and value data. | The generic path provides bring-up coverage; an optimized Apple decode kernel is still required for performance claims. |
| MoE bucketing | Expert index construction used the `vote` operation, which the Metal target did not support. | A serial Apple fallback constructed the expert buckets without `vote`. | The fallback tests semantics but is unsuitable as the final high-throughput implementation. |
| MoE permutation gather | Generic rank-2 gather lowering emitted double-precision and floor operations that Metal could not compile. | A direct flat GPU path handled rank-2 axis-0 gathers with `int32` and `uint32` indices. | Gather support must be qualified by rank, axis, index dtype, and layout. |
| Decode launch planning | Host launch planning did not recognize Metal even though the graph and device paths did. | Metal received the single-partition scalar configuration expected by the Apple decode path. | Device capability checks must agree across graph selection, kernel dispatch, and host launch code. |

Model registration covers architecture discovery. Device support also depends
on operator implementations, graph lowering, kernel dispatch, and host launch
configuration.

## What the short smoke establishes

For the tested configuration, the source-built runtime could:

- recognize the DeepSeek V2 architecture and load the local weights;
- launch focused Apple paths for MLA prefill, MLA decode, MoE bucketing, and
  rank-2 gather;
- compile and initialize the full model graph;
- complete bounded generation and emit tokens.

The validation level for this result is a local source-built generation smoke.
Packaged MAX, numerical parity, production serving, and distributed Fornax
require separate evidence.

## Qualification before a serving claim

An Apple serving claim needs evidence in four areas:

- correctness: reference comparisons for MLA prefill and decode, MoE routing and
  gather, logits, and deterministic token output;
- performance: optimized decode and MoE implementations, sustained throughput,
  thermal behavior, and memory use;
- workload coverage: longer contexts, larger generations, concurrent batches,
  and memory pressure;
- serving behavior: `max serve`, streaming, cancellation, timeout, and recovery
  tests.

Assigning Apple a compute-stage role in distributed Fornax requires separate
evidence: physical stage execution through the Fornax Stage ABI and transport,
followed by end-to-end numerical validation of the complete route. Other Apple
roles require evidence specific to the work assigned to them.

Packaged support also requires reproducing the result with an official package
that contains equivalent backend support. A locally patched source build remains
a separate support class.

## Sources

- MAX supported models:
  <https://docs.modular.com/max/models/>
- Apple DeepSeek runbook:
  [deepseek-v2-lite-max-check.md](../deepseek-v2-lite-max-check.md)
- Project plan v4:
  [project-plan-v4.md](../project-plan-v4.md)
- Repository status:
  [README.md](../../../README.md)

---

*Previous: [Fornax Architecture](./03-fornax-architecture.md). Back to the start: [Objective and Constraints](./00-objective-and-constraints.md). [Series index](./fornax.md).*
