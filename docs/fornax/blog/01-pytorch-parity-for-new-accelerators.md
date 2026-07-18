---
title: "Part 1 - What Accelerator Support Requires"
header:
  overlay_image: /assets/images/hero-mojo.svg
  overlay_filter: 0.5
  teaser: /assets/images/hero-mojo.svg
sidebar:
  nav: "fornax"
---

*Part 1. What sits between a device name and reliable model execution.*

## The device API is the first layer

A user may see one line of code:

```python
model.to("device_name")
```

That line does not establish model support. The backend still needs several
working layers:

- runtime integration for device discovery, memory, streams, events, guards,
  random-number generation, and serialization
- operator implementations with the required shapes, ranks, dtypes, and layouts
- Python bindings and device-independent frontend behavior
- higher-level integration for features such as automatic mixed precision,
  compilation, profiling, export, and distributed execution when the workload
  needs them

PyTorch provides `PrivateUse1` for backends maintained outside the main PyTorch
repository. It supplies extension points for operator registration, generators,
device guards, serialization metadata, and a user-facing device name. It does not
supply the backend implementation or prove that a model works.

## Kernel coverage

Large-model inference commonly depends on these operator families:

- matrix multiplication and matrix-vector multiplication
- elementwise math and reductions
- normalization, softmax, and logits processing
- top-k, sorting, gather, scatter, and indexing
- attention and KV-cache update and read operations
- quantized matrix multiplication
- MoE routing and expert index construction

A backend can pass matrix-multiplication tests and still fail a model at a
rank-specific gather, masked scatter, dynamic-shape lowering, dtype and layout
combination, or synchronization operation. Operator names alone are not a useful
support boundary. The exact rank, shape, dtype, layout, and numerical tolerance
must be tested.

## Lowering and compilation

Eager kernels establish basic functional coverage. High-throughput workloads
usually also need a compilation or graph path that can reduce dispatch overhead
and manage memory across operations. Depending on the stack, that path may use:

- `torch.compile` and Inductor integration
- FX, ONNX, or export-based lowering
- a vendor intermediate representation and graph optimizer
- fusion, memory planning, layout selection, and tuning

The requirement is semantic compatibility for the supported workload. Ordinary
model code should not need a device-specific fork, and compiled execution should
stay within the accepted numerical tolerance of the reference path.

## Training versus inference

Training expands the backend surface with:

- backward kernels and optimizer coverage
- distributed training and high-scale collectives
- deterministic random-number generation and checkpointing
- mixed precision, loss scaling, and long-run numerical stability

Inference has a different set of runtime requirements:

- prefill and decode execution
- KV paging and eviction
- continuous batching and prefix caching
- quantization and low dispatch overhead
- model loading and compile-cache behavior
- tokenizer and chat-template correctness

Fornax targets inference rather than training, so training parity is not a project
requirement. The inference path can be narrower, but every supported model shape
and dtype must still pass its own tests.

## Validation must follow the workload

Backend validation should proceed from the smallest contract to the complete
workload:

1. Test individual operators across the required shapes, dtypes, and layouts.
2. Test model subgraphs such as attention, KV updates, and the expert MLP.
3. Compare full-model outputs and logits with a trusted reference.
4. Exercise prefill, decode, batching, cache behavior, and sustained serving.
5. For distributed execution, validate values and ownership metadata at every
   stage boundary.

Fallbacks must be visible in the results. A CPU fallback may preserve correctness
while hiding missing device coverage and invalidating the performance result.

## Relevance to MAX

PyTorch's accelerator pathway connects one framework to a device runtime. MAX
aims to provide graph compilation, kernels, custom ops, and model execution across
several accelerator families. Fornax then adds the cross-node layer.

A platform name is not enough to assign a Fornax stage. The exact model, shape,
dtype, hardware, and MAX build must pass numerical and operational validation.
Until a physical backend passes that test, it remains a candidate rather than a
supported Fornax platform.

## Sources

- PyTorch Accelerator Integration:
  <https://docs.pytorch.org/docs/stable/accelerator/index.html>
- PyTorch `PrivateUse1` backend integration:
  <https://docs.pytorch.org/tutorials/advanced/privateuseone.html>

---

*Previous: [Objective and Constraints](./00-objective-and-constraints.md). Next: [MAX Platform Assessment](./02-max-platform-assessment.md). [Series index](./fornax.md).*
