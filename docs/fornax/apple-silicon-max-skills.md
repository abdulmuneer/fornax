# Fornax - Apple Silicon MAX Skill Map

This is the technical skill map for working on Fornax's Apple Silicon path: a
MAX-derived heterogeneous MoE inference engine where Macs can become expert
workers, capacity nodes, and eventually full pipeline-stage workers when the
MAX/Mojo Apple stack is ready for the target model.

## Upstream status to hold in your head

Apple support is moving quickly and must be capability-probed per build:

- Modular 25.6 introduced Mojo GPU programming support for Apple Silicon GPUs and
  said end-to-end GenAI execution would arrive through nightlies.
- Modular 26.4 release notes/blog say MAX now serves many common models on M3+
  Apple Silicon GPUs, including Llama/Qwen-family models that fit memory, and
  that nightlies are expanding M1/M2 support.
- The MAX packages page still carries a broader caveat that large GenAI model
  inference via MAX is not generally available on Apple Silicon.

For Fornax, treat Apple as a measured capability surface, not a promise. Every
nightly must answer: which model architectures, dtypes, kernels, memory layouts,
and serving paths actually work on this Mac?

Primary upstream references:

- Modular 25.6 Apple GPU direction:
  https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple
- Modular 26.4 MoE + Apple notes:
  https://www.modular.com/blog/modular-26-4-sota-moe-serving-model-bringup-via-agent-skills-mojo-beta-2-and-more
- MAX changelog:
  https://docs.modular.com/max/changelog/
- MAX package/system caveats:
  https://docs.modular.com/max/packages/
- Mojo system requirements:
  https://mojolang.org/docs/requirements/

## 1. Apple platform bring-up

You need enough macOS/Metal operational skill to make failures boring:

- macOS Sequoia 15+ and Apple Silicon M1-M5 setup.
- Xcode or Command Line Tools 16+; know when to run
  `xcodebuild -downloadComponent MetalToolchain`.
- uv/pixi environments for nightly `modular`/`mojo` builds.
- Verification with a minimal Mojo `DeviceContext()` program and `has_accelerator()`.
- Debugging Apple GPU detection: Metal framework availability, Xcode CLT install,
  macOS version, and nightly/stable package mismatches.
- Reproducible environment capture: macOS build, Xcode build, `modular` version,
  `mojo --version`, model ID, dtype, and exact command line.

Exit skill: given a fresh Mac, the engineer can prove whether Mojo sees the GPU
and whether a target MAX model path actually runs on that exact build.

## 2. Mojo GPU programming on Apple

This is the kernel-authoring foundation:

- GPU execution model: grids, thread blocks, warps/simdgroups, occupancy, launch
  geometry, and asynchronous execution.
- `DeviceContext`, host/device buffers, copies, synchronization, and stream order.
- Mojo `comptime` specialization for `target == "gpu"` vs CPU fallback.
- SIMD programming, `LayoutTensor`, `TileTensor`, tiling, vectorized loads/stores,
  memory coalescing, and alignment.
- Reduction patterns needed by RMSNorm, softmax, routing top-k, and expert gather.
- Apple-specific debugging limitations and differences from CUDA/HIP, especially
  where docs or examples are still NVIDIA-centric.
- Microbenchmarking kernels in isolation before inserting them into MAX graphs.

Exit skill: the engineer can write and benchmark a small Apple GPU Mojo kernel,
then port the same logical op across Apple/NVIDIA/AMD target specializations.

## 3. MAX graph and model architecture internals

Fornax is surgery inside model execution, so MAX architecture literacy matters:

- How MAX registers a model architecture: `arch.py`, `model_config.py`,
  `model.py`, `weight_adapters.py`, and `ARCHITECTURES`.
- How Hugging Face config fields map to typed MAX config fields.
- Weight adapter implementation for safetensors/GGUF and strict state-dict
  validation.
- `max.nn` building blocks: attention, MLP, norms, embeddings, layer lists, and
  model graph composition.
- KV cache configuration: `num_key_value_heads`, `head_dim`, `num_layers`,
  `model_max_seq_len`, and the consequences of getting any of them wrong.
- Quantization concepts used by MAX (`QuantFormat`, `QuantConfig`, FP8, FP4,
  block scales) and how they interact with kernel availability.
- MAX pipeline/runtime concepts: prefill vs decode, batching, overlap scheduler,
  speculative decoding, structured output, and supported model surface.

Exit skill: the engineer can take a reference MAX architecture, identify exactly
where the target MoE differs, and modify only the minimal graph/components.

## 4. MAX custom ops and Mojo extension packaging

Fornax needs custom seams, not a fork of everything:

- `@compiler.register` custom-op structs and `execute[target: StaticString](...)`.
- `InputTensor`, `OutputTensor`, rank/dtype constraints, and shape functions.
- `ops.custom(...)`, `TensorType`, `DeviceRef`, `InferenceSession`, and
  `custom_extensions` package directories with `__init__.mojo`.
- When to use `foreach` vs explicit device kernels.
- CPU fallback paths for bring-up and correctness checks.
- Extension packaging/versioning so a Fornax custom-op library can be injected
  into MAX graphs consistently.

Exit skill: the engineer can add one custom op to a MAX graph, run it on Apple if
supported, and compare its output against NumPy/PyTorch.

## 5. MoE architecture and expert-runtime surgery

This is the core Fornax specialization:

- MoE routing math: top-k routing, sigmoid/softmax routing, correction bias,
  load-balancing variants, shared experts, and expert weighting.
- Expert MLP shapes and fused kernels: gate/up/down projections, activation, and
  quantized/dequantized execution.
- Expert bucketing: grouping tokens by expert, preserving original token order,
  and weighted gather.
- Expert placement: hot resident, warm remote, cold store, and migration rules.
- Remote expert ABI design: layer id, expert id, token indices, dtype, hidden
  shape, routing weights, checksum/version, and output ordering.
- Trace collection: prefill/decode expert hit rates, coactivation, remote wait,
  and migration decisions.

Exit skill: the engineer can replace one MoE layer's expert executor with an
explicit local/remote dispatch path while matching reference logits within dtype
tolerance.

## 6. Apple performance engineering

Apple unified memory is useful, but it has to be measured like a real accelerator:

- Unified memory bandwidth behavior and CPU/GPU contention.
- Metal/GPU occupancy and launch overhead as exposed through Mojo/MAX behavior.
- Matrix multiply shapes common in expert MLPs and which are fast/slow on the
  current Apple kernels.
- Dtype reality: fp16/bfloat16/fp8/fp4 availability, accumulation strategy, and
  numerical drift versus NVIDIA/AMD.
- Memory residency policy: pinned hot experts, cold expert store, temporary
  activation buffers, and avoiding page-pressure collapse.
- Kernel fusion opportunities for expert MLPs and routing/gather operations.

Exit skill: the engineer can say, with measurements, whether a Mac should run a
full stage, only expert MLPs, only store weights, or stay out of the hot path.

## 7. Cross-device numerics and validation

Heterogeneous execution fails quietly unless validation is ruthless:

- Layer-by-layer comparison against PyTorch or the unmodified MAX path.
- Logit-divergence hunts: isolate the first layer/op where Apple diverges.
- Tolerance policies by dtype and op type.
- Deterministic test prompts and fixed sampling for correctness.
- Weight checksum and activation checksum at every pipeline/expert seam.
- Golden traces for routing decisions and expert placement.

Exit skill: the engineer can prove a mixed Apple/NVIDIA execution path is correct
before looking at speed.

## 8. Profiling and benchmarking

You need both MAX-level and Apple-level profiling instincts:

- MAX benchmark/generate workflows, warmup separation, compile time vs steady
  inference time.
- MAX profiling markers and how they differ across NVIDIA, AMD, and Apple paths.
- `nsys`/NVTX on NVIDIA; Apple Instruments/Metal profiling on Mac; `kbench` or
  custom microbenchmarks for Mojo kernels.
- Metrics Fornax must report: TTFT, decode tok/s, prefill tok/s, remote expert
  hit rate, exposed remote expert wait, queue time, per-link bandwidth, memory
  pressure, and stage bubble fraction.
- Separating prefill, decode, routing, expert MLP, activation transport, and
  weighted gather in traces.

Exit skill: the engineer can explain every millisecond in a mixed Mac/GPU decode
step.

## 9. Distributed runtime and transport

The Apple worker is only useful if it fits the fabric:

- RPC/data-plane design for activation tensors: framing, backpressure, retries,
  cancellation, deadlines, and batching.
- Zero-copy or low-copy buffer handling where APIs allow it; explicit copies when
  they do not.
- Link benchmarking: Thunderbolt, 25/100GbE, direct peer links, host memory paths.
- Scheduler integration: stage placement, expert placement, continuous batching,
  migration, node health, and admission control.
- Failure behavior: remote expert timeout, fallback expert placement, replay, and
  correctness-preserving request cancellation.

Exit skill: the engineer can keep Apple remote expert execution from becoming an
unbounded per-token latency cliff.

## 10. Source-level MAX maintenance discipline

Because this is MAX surgery, the maintenance skill is part of the job:

- Read MAX changelogs every nightly/stable bump.
- Pin known-good builds and record exact model/runtime compatibility.
- Keep a thin Fornax internal interface around MAX internals.
- Delete custom kernels as upstream MAX support lands.
- File minimal repros upstream when Apple-specific kernels or model paths break.
- Maintain a matrix: model x dtype x chip x runtime path x pass/fail x measured
  throughput.

Exit skill: the engineer can upgrade MAX without guessing what changed.

## Staffing shape

A tiny team can cover this if the roles are explicit:

- **MAX model engineer:** model architecture bring-up, weight adapters, KV, logits
  comparison, quantization.
- **Mojo GPU kernel engineer:** Apple/NVIDIA/AMD kernels, custom ops, tiling,
  benchmarking.
- **Distributed runtime engineer:** transport, scheduler, batching, expert
  migration, failure handling.
- **Performance/numerics engineer:** profiling, cross-device validation,
  throughput model, regression gates.

One very strong systems engineer can prototype across all four, but the project
becomes much safer when at least two people cover kernel/runtime and model/numerics
separately.
