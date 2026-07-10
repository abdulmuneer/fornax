# Skill-Based Review Lenses for Code, Project, Architecture, and Plan Reviews

This document turns the combined skill groups from the earlier Chris Lattner / Mojo / MAX, Antirez / ds4, vLLM, and MLX analysis into review lenses.

The operating idea is simple: **each reviewer picks one headline skill and reviews the work as the owner of that concern**. The reviewer is not expected to solve every issue personally. Their responsibility is to represent that lens strongly, identify what is missing, ask sharper questions, and produce concrete review output.

This works for:

- code reviews,
- architecture reviews,
- project plans,
- technical design documents,
- product/roadmap reviews,
- ML systems reviews,
- inference/runtime/platform reviews,
- prototype-to-production readiness reviews.

## How to assign reviewers

1. Assign one reviewer to each headline lens.
2. Give every reviewer the same underlying artifact: code, design doc, plan, PR, roadmap, or prototype.
3. Each reviewer reads the artifact only through their assigned lens.
4. Each reviewer writes a short review with:
   - what looks strong from this lens,
   - what is risky or missing,
   - what questions must be answered,
   - what changes are required before approval,
   - whether issues are blockers, high-priority concerns, or normal follow-ups.
5. Combine the reviews. Do not average them. A serious blocker in one lens can be enough to reshape the whole plan.

## Standard reviewer output format

Use this format for every lens:

```markdown
## <Lens name> Review

### Summary judgment
Approve / Approve with comments / Needs revision / Blocked

### What looks strong
- ...

### Risks or missing details
- ...

### Questions for the author/team
- ...

### Required changes
- ...

### Non-blocking suggestions
- ...

### Final note from this lens
...
```

## Severity guide

- **Blocker**: The project may fail, produce wrong results, create unsafe behavior, or become unmaintainable unless this is addressed.
- **High priority**: The work can proceed, but the risk should be addressed before release or major investment.
- **Medium priority**: Important improvement that should be scheduled.
- **Low priority**: Nice-to-have improvement or polish.

---
# Hardware Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who will keep the discussion grounded in real machines rather than abstract compute.

## What this reviewer should treat themselves as

The Hardware reviewer should treat themselves as the physical-reality owner. They are the person in the room asking: what exact machine, accelerator, memory pool, interconnect, storage layer, and deployment environment is this meant to survive? Their job is not merely to ask whether the code can run, but whether it runs on the intended hardware class with believable memory, bandwidth, latency, power, cost, and portability assumptions.

## Core mission

This reviewer protects the project from hidden hardware assumptions. They should make vague phrases like 'GPU support', 'runs locally', 'multi-device', 'production scale', or 'accelerated' concrete enough to review. They should force the plan to name the target devices, expected limits, memory headroom, device placement strategy, and what changes when the same idea moves from a laptop to a workstation to a cloud GPU cluster.

## Skill items represented by this lens

- CPU/GPU/accelerator architecture literacy: understand how CPUs, GPUs, NPUs, and AI accelerators differ in execution model, scheduling, SIMD/SIMT behavior, memory bandwidth, synchronization cost, and throughput constraints.
- Memory hierarchy and capacity planning: reason about registers, cache, RAM, VRAM, unified memory, SSD streaming, file-backed caches, temporary buffers, and how model size changes feasibility.
- Apple silicon unified-memory awareness: understand the benefits and constraints of CPU/GPU shared memory, including reduced copy friction but non-infinite bandwidth and capacity.
- Accelerator and chip-design context: understand how compiler/runtime decisions connect to hardware capabilities, instruction sets, tensor cores, matrix engines, RISC-V or accelerator co-design concerns.
- Multi-accelerator topology awareness: map workers, processes, model shards, cache shards, and communication paths to devices and interconnects.
- Constraint-driven hardware selection: choose deliberately between laptop inference, edge devices, Apple hardware, NVIDIA/AMD GPUs, CPU fallback, and data-center clusters.
- Hardware trade-off judgment: balance latency, throughput, cost, memory, heat, power, availability, portability, and developer usability instead of optimizing only one metric.

## What to inspect

### Code review

- Check whether device assumptions are hard-coded or properly discovered/configured.
- Look for hidden CPU-GPU copies, temporary allocations, implicit dtype conversions, and oversized buffers.
- Verify that benchmark code records hardware model, memory size, driver/runtime versions, batch size, sequence length, and concurrency.
- Check whether fallback paths exist for smaller hardware and whether they fail clearly when unsupported.

### Architecture review

- Ask for a hardware matrix: minimum supported machine, recommended machine, production target, and unsupported targets.
- Check memory estimates for model weights, KV cache, activations, temporary buffers, request queues, logs, and fragmentation overhead.
- Review multi-device topology: how devices are selected, how work is placed, and what happens when devices are heterogeneous.
- Check whether the architecture confuses local inference constraints with server-scale inference constraints.

### Project or plan review

- Require explicit hardware acceptance criteria rather than generic performance goals.
- Ask whether the team has access to the real target hardware during development and testing.
- Check whether portability is a product requirement, a future aspiration, or explicitly out of scope.
- Make sure deployment cost and operational capacity are considered early, not after the design is fixed.

## Questions this reviewer should ask

- What exact hardware is in scope for v1, and what exact hardware is out of scope?
- How much memory is required for the largest planned model, context length, batch size, and concurrency level?
- What is the expected bottleneck: compute, memory bandwidth, interconnect, storage, CPU overhead, or scheduling?
- What breaks first when we double context length, concurrency, model size, or number of devices?
- Are benchmark claims tied to named hardware and reproducible settings?
- Can this run acceptably on a smaller machine, or does it fail with a clear explanation?
- Does the plan require hardware features that are not available across the intended deployment fleet?

## Typical red flags

- The proposal says 'GPU' without specifying GPU class, memory, backend, or interconnect.
- Model size and context length are discussed without KV-cache or buffer memory estimates.
- Benchmarks omit hardware details or use toy input sizes.
- The design assumes infinite memory bandwidth or ignores host-device transfer cost.
- Local, workstation, and production-server requirements are mixed together.
- The plan relies on hardware the team cannot access for testing.
- A portability claim is made without a test matrix.

## Expected output from this reviewer

- Hardware target matrix with minimum/recommended/production targets.
- Memory budget table, including model weights, KV cache, temporary buffers, and overhead.
- List of unsupported or risky hardware assumptions.
- Benchmark reproducibility requirements.
- Blocker list for hardware issues that could invalidate the plan.

## Acceptance bar

A plan passes this lens when a reviewer can state exactly where it runs, why the memory fits, what the expected hardware bottleneck is, how performance will be measured, and how the system behaves on weaker or different hardware.

---

# Low-level Software Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who is comfortable near runtimes, compilers, memory, kernels, build systems, and operating-system boundaries.

## What this reviewer should treat themselves as

The Low-level Software reviewer should treat themselves as the owner of the machinery below the public API. They are responsible for invariants, memory ownership, data layout, binary boundaries, build correctness, backend equivalence, and failure modes that only appear under pressure. Their role is to ask whether the implementation remains safe, deterministic, debuggable, and maintainable when the abstraction reaches the runtime, compiler, kernel, or OS boundary.

## Core mission

This reviewer prevents the project from hiding dangerous complexity behind a friendly interface. They should inspect whether low-level choices are explicit enough: what owns memory, how buffers are reused, what data layouts are assumed, how compiled kernels are selected, how extension modules are built, how optimized and reference paths are compared, and how platform-specific failures are diagnosed.

## Skill items represented by this lens

- Systems programming: work fluently in C, C++, Rust, Swift, Objective-C, Mojo, CUDA, HIP, Metal, Triton, and language-extension boundaries where necessary.
- Compiler infrastructure: understand IRs, frontends, optimizers, code generation, static analysis, MLIR/LLVM-style pipelines, and how high-level code becomes executable kernels.
- Memory-management implementation: implement and debug paged memory, block tables, KV allocation, cache eviction, file-backed cache behavior, pointer-level layout, and lifecycle boundaries.
- Kernel/runtime boundary design: expose custom kernels and custom ops while keeping clean boundaries between frontend APIs, graph runtime, and hardware-specific code.
- OS-inspired systems techniques: apply paging, virtual memory, scheduling, device/process ownership, file IO, and resource isolation to inference and ML systems.
- Deterministic low-level correctness: use golden vectors, deterministic reference paths, strict backend modes, fuzzing where useful, and regression tests.
- Build-system and toolchain fluency: handle compiler flags, ABI compatibility, packaging, source builds, language bindings, and cross-platform build differences.

## What to inspect

### Code review

- Inspect ownership and lifetime of every important buffer, tensor, array, request object, cache entry, and file descriptor.
- Check data layout assumptions: row-major/column-major, contiguous/strided, alignment, dtype, quantization format, endian assumptions, and padding.
- Compare optimized paths against reference paths using golden data, not just performance tests.
- Review thread-safety, async boundaries, locks, atomics, queue behavior, and shutdown/cleanup paths.
- Check that errors crossing C/Python/Swift/Mojo/CUDA/Metal boundaries are surfaced with useful context.

### Architecture review

- Draw the boundary between high-level API, runtime, compiler, kernels, backend adapter, and device-specific implementation.
- Identify invariants that every backend must preserve.
- Check whether low-level modules can be tested independently from the rest of the product.
- Review the build and packaging story for each supported platform.

### Project or plan review

- Ask whether the team has the skills to maintain low-level code after the initial prototype.
- Require a correctness strategy before accepting performance-oriented rewrites.
- Check whether the project has enough diagnostic tools for crashes, memory leaks, numerical mismatches, and backend-specific bugs.
- Make sure low-level complexity is justified by a real product or performance need.

## Questions this reviewer should ask

- What are the non-negotiable invariants of this runtime or kernel path?
- Who owns each memory region and when can it be reused or freed?
- Is there a slow but obviously correct reference path?
- How do we prove that an optimized backend gives the same result as the reference backend?
- How does this build on a fresh machine and in CI?
- What happens if a backend is unavailable, miscompiled, or returns a different numerical result?
- Where will a developer look first when this fails only on one platform?

## Typical red flags

- Hidden global state controls critical runtime behavior.
- Memory ownership is implied rather than explicit.
- Fast paths have no golden-vector comparison.
- The build works only on one developer's machine.
- A high-level API hides low-level failures so thoroughly that debugging becomes guesswork.
- Platform-specific code is scattered throughout the codebase instead of isolated.
- Performance changes are merged without correctness proof.

## Expected output from this reviewer

- List of low-level invariants and who enforces them.
- Reference-vs-optimized correctness checklist.
- Memory ownership and lifecycle notes.
- Build/toolchain risks and required CI coverage.
- Backend-specific risks and debugging recommendations.

## Acceptance bar

A plan passes this lens when low-level ownership, layout, boundaries, builds, tests, fallback behavior, and debugging paths are explicit enough that a new systems engineer could maintain the component without relying on oral history.

---

# High-level Software Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who thinks deeply about developer experience, APIs, workflows, product abstractions, and how users approach the system.

## What this reviewer should treat themselves as

The High-level Software reviewer should treat themselves as the advocate for the developer and product-facing interface. They should not get distracted only by internals. Their job is to check whether the system exposes the right concepts, names, workflows, defaults, errors, and extension points so that capable users can succeed without needing to understand every internal detail.

## Core mission

This reviewer protects the project from becoming powerful but unusable. They should evaluate whether the public surface reflects user intent rather than implementation accidents. They should make sure the happy path is simple, expert paths are possible, errors are actionable, concepts are named consistently, and the system fits real workflows such as local inference, serving, model conversion, quantization, custom ops, fine-tuning, or deployment.

## Skill items represented by this lens

- Developer-facing API design: design Python-native, NumPy-like, PyTorch-like, Swift-friendly, CLI-based, or OpenAI-compatible interfaces over complicated internals.
- Programming-language and framework design: create language features, type systems, ownership models, array semantics, graph APIs, and abstractions that preserve performance without overwhelming users.
- Model and graph abstractions: represent models as modules, tensors/arrays, graphs, operators, schedulers, sessions, caches, runtimes, and extensible execution units.
- Serving and product APIs: expose inference through OpenAI-compatible servers, Anthropic-style messages, gRPC, local chat CLIs, batch APIs, streaming APIs, and agent-oriented interfaces.
- Multi-language bindings: support Python, Swift, C, C++, Mojo, and other host languages while preserving semantics.
- Usability over complexity: hide hard systems details behind simple composable APIs while preserving escape hatches.
- End-user workflow design: support install, configure, load model, run, inspect, debug, benchmark, extend, deploy, and upgrade workflows.

## What to inspect

### Code review

- Check whether names match user mental models, not internal implementation names.
- Inspect public method signatures, defaults, error messages, configuration shape, and examples.
- Check whether simple tasks require unnecessary boilerplate or internal knowledge.
- Look for inconsistent terminology across modules, docs, CLI flags, and config files.
- Review whether advanced escape hatches are available without polluting the normal path.

### Architecture review

- Map internal subsystems to public abstractions and check if the mapping is understandable.
- Decide what should be stable public API versus private implementation detail.
- Review extension points: custom kernels, model plugins, backends, server middleware, configuration hooks, and debugging hooks.
- Check whether the architecture allows future features without breaking the current API.

### Project or plan review

- Define the target user: researcher, infra engineer, app developer, ML engineer, systems engineer, or end user.
- Ask for complete user journeys rather than isolated demos.
- Check whether the product can be explained in one or two coherent mental models.
- Review compatibility expectations with PyTorch, NumPy, Hugging Face, Swift, OpenAI APIs, or existing tools.

## Questions this reviewer should ask

- Who is the primary user, and what should they be able to do in the first 10 minutes?
- Which concepts are public, and which are internal?
- Are defaults safe and useful, or merely convenient for the implementation?
- Can a user recover from common mistakes using the error messages alone?
- Does the API make simple things simple and advanced things possible?
- How will this API evolve without breaking users?
- Is the demo a real workflow or only a proof that internals can run?

## Typical red flags

- Users must understand internal memory management to perform basic tasks.
- The API mirrors implementation details rather than user intent.
- Similar concepts have multiple names across modules.
- The demo works but the install-configure-run-debug loop is incomplete.
- Power-user features are exposed as unstable hacks rather than designed extension points.
- Error messages describe what failed but not what the user can do next.
- API stability is postponed until after adoption.

## Expected output from this reviewer

- User journey review with happy path and expert path.
- API naming and abstraction notes.
- Confusing concepts or terminology to simplify.
- Missing workflow steps.
- Public/private API boundary recommendations.

## Acceptance bar

A plan passes this lens when a strong engineer can understand and use the system without knowing the internals, while an expert can still reach controlled extension points for performance, debugging, or customization.

---

# LLM Expertise Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who understands language-model behavior, inference semantics, tokenization, serving, KV cache, model quality, and model-specific execution details.

## What this reviewer should treat themselves as

The LLM Expertise reviewer should treat themselves as the owner of model behavior and inference correctness. They should read every design choice through the question: does this respect how autoregressive transformer inference actually works, and will the user get correct, stable, high-quality language-model behavior under real workloads?

## Core mission

This reviewer prevents the project from treating an LLM like a stateless function. They should inspect prefill/decode separation, tokenization, chat templates, stopping behavior, KV-cache lifecycle, batching, scheduling, sampling, streaming, model architecture differences, quantization, quality regression tests, and application-level semantics such as tool calling or structured outputs.

## Skill items represented by this lens

- Transformer inference mechanics: understand autoregressive decoding, prefill vs decode, attention, tokenization, prompt rendering, generation loops, and stopping criteria.
- KV cache design and reuse: design cache allocation, paging, prefix reuse, snapshots, replay, RAM/disk handling, and distributed KV movement.
- Batching and scheduling: implement or evaluate continuous batching, request scheduling, chunked prefill, dual-batch overlap, fairness, and latency/throughput tuning.
- Quantization literacy: work with GGUF, low-bit quantization, activation statistics, imatrix-style calibration, MLX quantization, and quality-speed-memory tradeoffs.
- Model-specific optimization: account for MoE models, DeepSeek-family architectures, Llama-style inference, rope scaling, attention variants, speculative decoding, and backend-specific kernels.
- Serving semantics for LLM apps: support tool calling, structured outputs, chat templates, streaming responses, OpenAI-compatible endpoints, and prompt-role semantics.
- LLM quality and regression evaluation: compare outputs against official vectors, reference backends, deterministic sampling configurations, and task-level evaluations.
- Fine-tuning and local model workflows: understand model loading, conversion, quantization, local generation, fine-tuning, and distributed inference flows.

## What to inspect

### Code review

- Check tokenization, detokenization, chat-template handling, BOS/EOS behavior, stop tokens, and special tokens.
- Inspect generation loop correctness: prefill, decode, logits processing, sampling, streaming, stopping, cancellation, and cleanup.
- Review KV-cache allocation, reuse, eviction, prefix caching, fragmentation behavior, and cache invalidation.
- Check model-family conditionals and ensure architecture-specific behavior is tested.
- Review quantization code for quality checks, dtype correctness, and fallback behavior.

### Architecture review

- Map the end-to-end request lifecycle from prompt to tokens to response stream.
- Check scheduling design for mixed prompt lengths, long contexts, interactive latency, and throughput-heavy workloads.
- Review support boundaries for model architectures and features such as MoE, rope scaling, multimodal input, function calling, and structured output.
- Check whether model quality evaluation is a first-class part of release and performance work.

### Project or plan review

- Clarify whether the target is local single-user chat, batch generation, API serving, coding agents, tool-use agents, or research experimentation.
- Require model-specific acceptance tests, not just generic 'loads a model' tests.
- Ensure performance targets do not silently break output quality or streaming behavior.
- Check whether new optimizations are evaluated against both tokens/sec and output correctness.

## Questions this reviewer should ask

- Is prefill/decode behavior explicitly modeled, or hidden inside a generic forward call?
- How is KV cache sized, reused, evicted, serialized, or transferred?
- Which chat templates and tokenizers are supported, and how are they tested?
- What are the quality tests before and after quantization or kernel optimization?
- How does batching affect latency for short interactive requests?
- What model architectures are supported, and which are intentionally unsupported?
- Do streaming, cancellation, stop sequences, and structured outputs behave correctly under load?

## Typical red flags

- The plan discusses serving but ignores tokenizer and chat-template correctness.
- KV cache is mentioned but not sized or lifecycle-managed.
- Quantization is selected only by memory savings without quality checks.
- Throughput optimizations destroy streaming latency or fairness.
- All transformer models are assumed to behave the same.
- Tool calling or structured output support is assumed rather than designed.
- Tests only check that text is produced, not whether the model behavior is correct.

## Expected output from this reviewer

- LLM request lifecycle notes.
- Tokenizer/chat-template/support matrix.
- KV-cache design risks and memory estimates.
- Quality regression checklist.
- Model architecture support and exclusions.

## Acceptance bar

A plan passes this lens when model behavior is treated as a system contract: tokenization, prompt formatting, generation, KV cache, scheduling, quantization, streaming, and quality are all explicitly designed and tested.

---

# Hardware Acceleration Expertise Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who knows how performance is actually obtained from GPUs, accelerators, kernels, graph execution, profiling, and data movement.

## What this reviewer should treat themselves as

The Hardware Acceleration reviewer should treat themselves as the performance-path owner. They should not accept 'uses GPU' as evidence of acceleration. Their job is to ask where the time goes, whether the accelerator is fed efficiently, whether kernels are appropriate, whether transfers dominate, whether benchmarks are realistic, and whether fast paths preserve correctness.

## Core mission

This reviewer protects the project from fake acceleration. They should inspect kernel choices, graph compilation, lazy evaluation, fusion, memory movement, operator coverage, attention/GEMM/MoE paths, profiling methodology, fallback paths, and backend portability. They should force performance claims to be tied to measurements and force measurements to match the real workload.

## Skill items represented by this lens

- GPU kernel programming: write or integrate kernels in CUDA, HIP/ROCm, Metal, Mojo, C++ extensions, Triton, or similar backend layers.
- Optimized attention and GEMM backends: use or implement high-performance attention, GEMM, MoE, FlashAttention, FlashInfer, CUTLASS, Triton-backed kernels, and vendor libraries.
- Compiler-driven acceleration: use MLIR-style IRs, graph compilers, lowering passes, and runtime specialization to transform high-level operations into efficient execution.
- Lazy evaluation, compilation, and fusion: reduce overhead by fusing operations, compiling repeated computation, and avoiding unnecessary materialization.
- Cross-vendor accelerator portability: target NVIDIA, AMD, Apple GPUs, CPUs, and other accelerators while managing vendor-specific tradeoffs.
- Profiling and benchmarking: measure tokens/sec, time-to-first-token, inter-token latency, memory bandwidth, occupancy, throughput, utilization, and regression deltas.
- Data-movement minimization: reduce host-device transfers, use unified memory appropriately, keep hot data local, and design around bandwidth limits.
- Hardware-aware API design: expose accelerated execution without forcing ordinary users to manually control every device detail.

## What to inspect

### Code review

- Identify hot paths and check whether they use optimized kernels or accidentally fall back to slow generic operations.
- Inspect host-device transfers, synchronization points, implicit materialization, Python loops, and small-kernel launch overhead.
- Review backend selection logic, capability detection, dtype support, and fallback behavior.
- Check that accelerated kernels are compared against reference outputs.
- Look for unnecessary format conversions, tensor copies, or memory layout changes.

### Architecture review

- Review the acceleration strategy per operation class: attention, GEMM, normalization, sampling, quantization, MoE routing, communication, and preprocessing.
- Check whether graph compilation, lazy evaluation, or fusion is actually used where it matters.
- Evaluate whether portability is handled through clean backend interfaces or scattered conditionals.
- Review profiling methodology and make sure it captures realistic concurrency and sequence lengths.

### Project or plan review

- Require performance budgets for the real workload, not just synthetic microbenchmarks.
- Ask whether the team will maintain custom kernels or rely on upstream libraries.
- Check whether acceleration work is prioritized by measured bottlenecks.
- Ensure that correctness, portability, and maintenance cost are considered alongside speed.

## Questions this reviewer should ask

- What is the measured hot path today?
- Which operations are compute-bound, memory-bound, launch-bound, or communication-bound?
- What optimized kernels are used, and what are the fallback paths?
- Are there unnecessary host-device transfers or synchronization barriers?
- Do performance numbers include realistic context length, batch size, concurrency, and output length?
- How are accelerated outputs checked against reference outputs?
- What is the maintenance cost of this acceleration path?

## Typical red flags

- The plan claims acceleration simply because it uses a GPU.
- Most time is spent in data transfer, synchronization, Python overhead, or tiny kernel launches.
- Benchmarks use toy sequence lengths or batch sizes that hide bottlenecks.
- Optimized kernels are added without correctness comparisons.
- Portability is promised but backends are entangled.
- Profiling tools are not part of the development loop.
- The design cannot explain compute-bound versus memory-bound behavior.

## Expected output from this reviewer

- Hot-path and bottleneck summary.
- Acceleration coverage table by operation.
- Benchmark requirements and missing measurements.
- Correctness checks for optimized paths.
- Backend portability and maintenance risks.

## Acceptance bar

A plan passes this lens when acceleration is tied to measured hot paths, realistic benchmarks, correct optimized kernels, minimized data movement, and a clear backend strategy.

---

# Networking Expertise Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who understands serving interfaces, distributed execution, request flow, collective communication, security boundaries, and production traffic behavior.

## What this reviewer should treat themselves as

The Networking reviewer should treat themselves as the owner of communication and request flow. They should inspect every place where information crosses a process, machine, device, API, or trust boundary. Their job is to ask whether the system behaves correctly under real request traffic, distributed execution, partial failure, backpressure, and security constraints.

## Core mission

This reviewer prevents the project from assuming everything is a local function call. They should inspect API protocols, server behavior, queueing, backpressure, multi-node orchestration, collective communication, KV or state transfer, endpoint security, streaming behavior, cancellation, retries, and operational failure modes.

## Skill items represented by this lens

- Inference API serving: design HTTP/REST, OpenAI-compatible, gRPC, local server, CLI-server, and streaming endpoints for model serving.
- Distributed inference orchestration: coordinate multi-node and multi-device execution, worker roles, process placement, and Ray-style orchestration patterns.
- Collective communication: understand MPI, NCCL, TCP-ring communication, RDMA-oriented backends, all-reduce, all-gather, reduce-scatter, and when communication dominates compute.
- KV transfer and remote state: move, rebuild, replay, or invalidate inference state across workers and serving components.
- Security boundaries: treat network endpoints, internal RPCs, inter-node communication, plugins, and local servers as explicit trust-boundary problems.
- Networked data-structure thinking: understand a server as a protocol around data structures and operations, as seen in Redis-like systems and inference servers.
- Operational traffic awareness: reason about batching, queueing, load, backpressure, concurrency, tail latency, and request-level service behavior.

## What to inspect

### Code review

- Inspect API endpoint behavior, request validation, streaming response handling, cancellation, timeouts, retries, and error status mapping.
- Check queueing and batching code for fairness, starvation, backpressure, and graceful shutdown.
- Review distributed communication assumptions, worker discovery, rank mapping, and failure handling.
- Inspect any remote-state movement such as KV cache transfer, prefix caching, session state, or temporary artifacts.
- Check that security-sensitive options are not exposed accidentally.

### Architecture review

- Draw request flow from client to router to scheduler to worker to device to response stream.
- Review which communications are local, inter-process, inter-device, inter-node, or public network.
- Check whether communication cost is included in performance modeling.
- Review trust boundaries, authentication assumptions, exposed ports, plugin boundaries, and safe defaults.

### Project or plan review

- Clarify whether the system is single-user local, team-local, internet-facing, internal service, or production multi-tenant.
- Require load-test plans for realistic concurrency.
- Check whether operational behavior under overload is specified.
- Ensure that networking and distributed-system work is not postponed until after local architecture hardens incorrectly.

## Questions this reviewer should ask

- What protocols and endpoints are exposed, and who can access them?
- How does a request move through the system from client to generated tokens?
- Where can backpressure occur, and how is it signaled?
- What happens on worker failure, network partition, cancellation, timeout, or client disconnect?
- Is communication cost significant relative to compute?
- How is distributed state, especially KV cache or session state, moved or invalidated?
- Are security boundaries explicit and documented?

## Typical red flags

- The design treats a distributed system as a local function call.
- Streaming, cancellation, and timeout behavior are unspecified.
- Endpoints are exposed without authentication or trust-boundary discussion.
- Queueing and batching hide unbounded memory growth.
- Communication overhead is ignored in multi-node performance claims.
- Worker failure or partial failure is not considered.
- Remote cache/state movement is ad hoc.

## Expected output from this reviewer

- Request-flow diagram or narrative.
- Endpoint and trust-boundary review.
- Backpressure, timeout, and failure-mode notes.
- Distributed communication risks.
- Load-test and production-readiness recommendations.

## Acceptance bar

A plan passes this lens when communication paths, endpoints, state movement, backpressure, failure behavior, and security boundaries are explicit enough for realistic serving or distributed execution.

---

# Software Engineering Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who will evaluate maintainability, testing, modularity, release quality, contributor experience, and long-term code health.

## What this reviewer should treat themselves as

The Software Engineering reviewer should treat themselves as the maintainer of the codebase over time. Their job is to ask whether today’s implementation can survive tomorrow’s contributors, features, bugs, releases, regressions, packaging issues, and production incidents. They should protect the project from clever prototypes that cannot become reliable systems.

## Core mission

This reviewer ensures that good ideas are turned into maintainable engineering. They should evaluate scope, module boundaries, tests, CI, release process, build reproducibility, benchmark discipline, readable code, dependency management, API stability, contribution workflow, and the cost of future change.

## Skill items represented by this lens

- Scope control: choose a sharp product boundary and avoid turning every project into a universal framework.
- Modular architecture: design clean extension points such as plugins, custom ops, kernels, model runners, backend interfaces, and adapters.
- Testing discipline: use unit tests, integration tests, golden vectors, regression tests, CI, backend checks, and performance gates.
- Release and build engineering: maintain packages, source builds, changelogs, dependency constraints, reproducible releases, and versioning.
- Benchmark design: create fair, reproducible benchmarks and interpret results by hardware, model, workload, context length, and concurrency.
- Readable code and comments: write code future maintainers can reason about, with comments that explain invariants and mental models.
- Open-source contribution workflow: support issues, PRs, reviews, contributor guides, community feedback, and extension requests.
- Backward compatibility and API stability: balance innovation with migration paths and user trust.

## What to inspect

### Code review

- Review module boundaries, naming, cohesion, duplication, dependency direction, and hidden coupling.
- Check whether tests cover correctness, regressions, edge cases, error paths, and performance-sensitive changes.
- Inspect comments for useful invariant explanation rather than noise.
- Review configuration handling, logging, observability hooks, and failure messages.
- Check whether new complexity is isolated, documented, and justified.

### Architecture review

- Evaluate whether architecture is modular enough to add models, backends, kernels, APIs, or workflows without rewrites.
- Review public/private boundaries and dependency flow.
- Check whether build, packaging, CI, benchmark, and release pipelines are part of the architecture.
- Evaluate migration strategy for breaking changes.

### Project or plan review

- Ask whether the scope is small enough to ship and broad enough to matter.
- Require a testing and release plan appropriate to the risk level.
- Check whether maintainers and contributors can understand how to extend the system.
- Review whether technical debt is being tracked deliberately.

## Questions this reviewer should ask

- Is the scope crisp, or is the project trying to become everything?
- What are the stable modules and extension points?
- What tests would catch the most dangerous regressions?
- Can a new contributor build, run, test, and modify this?
- What is the release process and compatibility promise?
- Are benchmarks reproducible and tied to relevant workloads?
- Which parts are intentionally temporary and when will they be replaced?

## Typical red flags

- The prototype works but has no test strategy.
- Everything imports everything else.
- Benchmarks are informal screenshots or one-off runs.
- Release and packaging are postponed until late.
- Public APIs are changed casually without migration notes.
- Code comments restate syntax but do not explain invariants.
- Contribution workflow is unclear.

## Expected output from this reviewer

- Maintainability review summary.
- Module-boundary and dependency concerns.
- Testing and CI gaps.
- Release/build/packaging risks.
- Scope and technical-debt recommendations.

## Acceptance bar

A plan passes this lens when it is not merely functional, but maintainable: scoped, modular, tested, buildable, releasable, benchmarked, and understandable by future maintainers.

---

# Organizational Skill Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who will evaluate whether the work can actually be executed by the team or community, not just whether it is technically attractive.

## What this reviewer should treat themselves as

The Organizational Skill reviewer should treat themselves as the execution-clarity owner. Their job is to ask whether the people, roadmap, milestones, priorities, dependencies, contributor model, ownership boundaries, and feedback loops are strong enough for the technical plan to become reality.

## Core mission

This reviewer protects the project from ambition without execution structure. They should inspect whether the work is sequenced, who owns what, what must be decided now, what can be deferred, which teams or contributors are needed, how cross-layer work is coordinated, and how community/user feedback shapes the roadmap without causing chaos.

## Skill items represented by this lens

- Team building and scaling: recruit, structure, and scale teams around difficult compiler, runtime, ML, serving, and infrastructure work.
- Platform strategy: coordinate language, compiler, kernels, runtime, model APIs, docs, and deployment into one coherent platform.
- Open-source coordination: manage contributors, issues, releases, community expectations, governance, and contribution boundaries.
- Roadmap discipline: sequence prototypes, research ideas, public releases, documentation, production hardening, and ecosystem integration.
- Technical prioritization: decide when to stay narrow and self-contained versus building a broad platform.
- Mentoring and talent leverage: enable engineers and contributors to work across compilers, runtimes, kernels, serving, docs, and examples.
- Community feedback loops: turn benchmarks, bug reports, user requests, and contributor experience into product direction.

## What to inspect

### Code review

- Check ownership of subsystems introduced by the change.
- Ask whether the change creates maintenance burden for a team that has not agreed to own it.
- Review whether TODOs represent deliberate milestones or hidden debt.
- Check whether documentation and tests enable handoff to other engineers.

### Architecture review

- Map the architecture to team ownership: compiler, runtime, kernels, API, serving, docs, release, support.
- Identify cross-team dependencies and sequencing risks.
- Review whether the platform strategy is coherent or assembled from disconnected components.
- Check whether decisions are reversible, staged, or one-way doors.

### Project or plan review

- Review milestones, staffing, risk burn-down, decision points, and success criteria.
- Check whether the roadmap separates prototype, internal alpha, public beta, production-ready, and ecosystem maturity.
- Ask how feedback from users, contributors, benchmarks, and support load will be incorporated.
- Make sure the plan identifies owners for hard cross-layer problems.

## Questions this reviewer should ask

- Who owns each major subsystem after launch?
- What is the smallest useful milestone?
- What must be solved now, and what can be intentionally deferred?
- Which decisions are irreversible or expensive to reverse?
- How will contributors or users know where to participate?
- What feedback loops will shape the roadmap?
- Is the team structure aligned with the architecture?

## Typical red flags

- The roadmap is a list of features without sequencing or owners.
- Cross-layer work depends on informal coordination.
- The plan assumes a future team that does not exist.
- Every feature is considered v1-critical.
- Community feedback is either ignored or allowed to constantly change direction.
- Ownership after merge is unclear.
- Documentation, release, support, and maintenance are treated as someone else's problem.

## Expected output from this reviewer

- Ownership and milestone review.
- Roadmap sequencing concerns.
- Cross-team dependency list.
- Scope/prioritization recommendations.
- Contributor/community coordination risks.

## Acceptance bar

A plan passes this lens when there is a credible execution path: clear owners, phased milestones, aligned team structure, realistic priorities, feedback loops, and enough organizational capacity to maintain what is built.

---

# Analytical Skills Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who will be the skeptical diagnostician: the person who tests assumptions, models tradeoffs, and asks whether the reasoning is sound.

## What this reviewer should treat themselves as

The Analytical Skills reviewer should treat themselves as the reasoning and diagnosis owner. They should not represent a single subsystem. Instead, they should examine whether the plan correctly identifies the real problem, the real bottleneck, the real constraints, and the real tradeoffs. Their job is to prevent the team from solving the wrong problem elegantly.

## Core mission

This reviewer protects against shallow reasoning. They should inspect bottleneck analysis, tradeoff framing, algorithmic choices, performance models, architecture placement, cross-layer debugging strategy, and whether constraints are being used deliberately. They should be comfortable saying: the data does not support this conclusion, the benchmark does not test the claim, or the chosen abstraction belongs at the wrong layer.

## Skill items represented by this lens

- Bottleneck identification: find the real limiting factor, such as KV-cache fragmentation, memory transfer, kernel inefficiency, framework overhead, communication cost, or bad workload assumptions.
- Trade-off analysis: balance simplicity vs generality, latency vs throughput, local vs distributed, portability vs peak performance, and quality vs quantization.
- Algorithmic reasoning: apply paging, scheduling, data structures, quantization, graph transformations, caching, routing, and runtime algorithms to practical systems.
- Performance modeling: reason from memory bandwidth, FLOPs, device placement, communication cost, context length, batch size, tokens/sec, and tail latency.
- Architecture evaluation: decide whether a problem belongs in a language, compiler IR, graph runtime, kernel backend, plugin, serving layer, or product API.
- Cross-layer debugging: debug failures spanning tokenizer, model math, memory layout, kernel execution, scheduling, networking, and API behavior.
- Constraint-based design: turn deliberate limits into advantages, such as Apple-silicon focus, DeepSeek-specific local inference, or unified platform scope.

## What to inspect

### Code review

- Check whether the implementation matches the stated problem and constraints.
- Look for premature generality, premature optimization, or local changes that worsen the global system.
- Review benchmark interpretation and whether conclusions follow from data.
- Inspect algorithmic complexity and worst-case behavior.
- Check whether diagnostics exist to distinguish competing failure hypotheses.

### Architecture review

- Test whether the selected abstraction layer is correct.
- Review the tradeoff matrix behind major design choices.
- Ask what alternatives were rejected and why.
- Check whether the architecture optimizes the actual dominant workload.

### Project or plan review

- Clarify the core hypothesis of the plan.
- Define what evidence would prove the plan wrong.
- Check whether risks are ranked by likelihood and impact.
- Require experiments that isolate variables rather than producing ambiguous results.

## Questions this reviewer should ask

- What is the real bottleneck, and what evidence supports that?
- What is the strongest alternative design, and why was it rejected?
- What tradeoff are we making deliberately?
- What assumption would invalidate the plan if false?
- Does this problem belong at this layer of the stack?
- Are we optimizing for average case, worst case, p95/p99, developer time, or hardware cost?
- What small experiment would most reduce uncertainty?

## Typical red flags

- A conclusion is stronger than the evidence.
- The benchmark does not measure the claimed improvement.
- The design solves an imagined bottleneck rather than a measured one.
- Every tradeoff is described as all upside.
- Complexity is added before the simple baseline is understood.
- A problem is solved at the wrong layer because that is where the team is comfortable.
- Failure analysis depends on guessing rather than instrumentation.

## Expected output from this reviewer

- Assumption and evidence review.
- Bottleneck and tradeoff analysis.
- Alternative-design notes.
- Experiments needed to reduce uncertainty.
- Layering or abstraction concerns.

## Acceptance bar

A plan passes this lens when its reasoning is explicit, evidence-backed, tradeoffs are honest, bottlenecks are measured, alternatives are considered, and uncertainty is reduced by targeted experiments.

---

# System Engineering Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who can see the full stack: language/runtime, graph, kernel, model, scheduler, server, deployment, observability, and ecosystem integration.

## What this reviewer should treat themselves as

The System Engineering reviewer should treat themselves as the end-to-end integrator. They should ask whether all parts of the system actually compose into a coherent working whole. Their job is to catch gaps between layers: good kernels with bad scheduling, good APIs with missing runtime behavior, good local demos with no deployment path, or good model support with broken observability.

## Core mission

This reviewer protects against sub-system excellence without system coherence. They should inspect runtime architecture, graph/runtime integration, parallel execution, local inference product flow, framework design, operational readiness, ecosystem integration, and end-to-end failure behavior. They should ensure that the system can be explained as a full lifecycle, not a pile of impressive components.

## Skill items represented by this lens

- End-to-end stack thinking: connect language design, IR, kernels, graph compiler, runtime, model API, serving API, deployment, and operations into one mental model.
- Runtime architecture: design schedulers, workers, model runners, memory managers, cache managers, request coordinators, and lifecycle managers.
- Graph compiler/runtime integration: represent computation as graphs and optimize execution through compiler/runtime cooperation.
- Parallel and distributed execution: understand tensor, pipeline, data, expert, and context parallelism when model scale exceeds a single device.
- Local inference system design: integrate CLI, server, agent loop, model loader, tokenizer, KV cache, quantizer, benchmarks, and hardware backend into one product.
- Framework system design: build array semantics, dynamic graphs, function transforms, custom ops, distributed communication, and examples into a coherent framework.
- Operational readiness: include security, deployment, observability, benchmarking, releases, and documentation as first-class system concerns.
- Ecosystem integration: connect with PyTorch, ONNX, Hugging Face, OpenAI-compatible APIs, Apple tooling, container workflows, and existing developer expectations.

## What to inspect

### Code review

- Check whether a change fits into the runtime lifecycle and does not create cross-layer inconsistencies.
- Inspect integration points: model loading, tokenizer, scheduler, memory manager, backend, server, logging, and cleanup.
- Look for missing observability at boundaries where debugging will be hard.
- Check whether new features work in realistic end-to-end paths, not only isolated unit tests.
- Review lifecycle behavior: initialization, warmup, steady state, overload, cancellation, shutdown, and recovery.

### Architecture review

- Draw the complete system from user input to hardware execution to output and operational monitoring.
- Identify which layer owns each responsibility and where data changes representation.
- Check whether control flow and data flow are both understood.
- Review end-to-end compatibility with deployment environments and ecosystem tools.

### Project or plan review

- Require an end-to-end milestone before over-investing in isolated subsystem perfection.
- Check whether operational concerns are included early: config, logs, metrics, tracing, deployment, security, upgrades.
- Review whether integration risks are ranked alongside algorithmic and performance risks.
- Make sure the plan includes system-level validation, not only component-level validation.

## Questions this reviewer should ask

- Can we explain the entire lifecycle from input to output to cleanup?
- Which layer owns scheduling, memory, compilation, serving, observability, and errors?
- Where are the boundaries between model logic, runtime logic, backend logic, and product API?
- How do components behave together under real concurrency and failure?
- What is the first end-to-end milestone?
- Does the design integrate with the ecosystem users already depend on?
- What operational signals will tell us the system is healthy or broken?

## Typical red flags

- Each subsystem works independently, but no one owns the end-to-end path.
- The design has no lifecycle model.
- Operational readiness is treated as a post-launch task.
- Data representation changes are undocumented across layers.
- The project has impressive demos but no deployment story.
- Observability is missing at the hardest debugging boundaries.
- Ecosystem integration is assumed rather than tested.

## Expected output from this reviewer

- End-to-end lifecycle review.
- Layer ownership and boundary map.
- Integration risks.
- Operational readiness gaps.
- System-level validation recommendations.

## Acceptance bar

A plan passes this lens when the system is coherent end to end: layers have clear ownership, data/control flow is understood, integration paths are tested, and operational behavior is part of the design.

---

# People Skills Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who will evaluate collaboration, leadership, adoption, teaching, community, developer empathy, and human usability.

## What this reviewer should treat themselves as

The People Skills reviewer should treat themselves as the human-adoption owner. They should ask whether the project can attract trust, teach users, support contributors, align stakeholders, and create an engineering culture that sustains quality. Their job is to represent the humans around the system: maintainers, users, reviewers, contributors, operators, leadership, and downstream teams.

## Core mission

This reviewer protects the project from being technically correct but socially hard to adopt. They should inspect communication clarity, onboarding, contribution experience, review culture, issue handling, user empathy, leadership alignment, teaching material, and how the project absorbs feedback without losing coherence.

## Skill items represented by this lens

- Technical leadership: set a compelling direction for complex infrastructure and align teams or communities around it.
- Developer empathy: design APIs, docs, examples, errors, and comments that reduce friction for users and contributors.
- Community collaboration: work constructively with contributors, issue reporters, downstream users, ecosystem partners, and internal stakeholders.
- Teaching and explanation: explain hard ideas through talks, blog posts, examples, tutorials, manifestos, design docs, and review comments.
- Engineering culture: create norms around high standards, clarity, performance, testing, correctness, and practical craftsmanship.
- Review and feedback handling: absorb criticism, bug reports, benchmarks, feature requests, and user confusion without losing architectural coherence.
- User-centered constraints: choose constraints that match real users, such as local Mac inference, Apple silicon ML, Python-native serving, or unified AI infrastructure.

## What to inspect

### Code review

- Review whether code structure and comments are kind to future maintainers.
- Check whether error messages and logs help users rather than blame them.
- Look at whether contribution expectations are clear in tests, style, and review comments.
- Check whether the code creates hidden tribal knowledge that only original authors understand.

### Architecture review

- Ask whether the architecture can be taught to new engineers and users.
- Review whether complexity is divided in a way that supports team collaboration.
- Check whether different stakeholder needs are represented: researcher, infra engineer, app developer, operator, contributor.
- Review whether feedback channels can influence the architecture without causing random drift.

### Project or plan review

- Check whether the plan includes onboarding, examples, contributor support, stakeholder communication, and review rituals.
- Ask who needs to believe in this project for it to succeed.
- Review how disagreements will be resolved.
- Make sure success is not defined only by technical metrics but also by adoption and maintainability.

## Questions this reviewer should ask

- Who are the humans this system must serve?
- Can a new engineer or user build a correct mental model quickly?
- What parts of the project require teaching, not just documentation?
- How will the team handle external feedback and criticism?
- Does the review culture encourage quality without discouraging contributors?
- What stakeholder alignment is needed before execution?
- Are constraints chosen from user reality or from builder preference?

## Typical red flags

- The system is technically elegant but hard to explain.
- Only original authors can safely modify important parts.
- Users are expected to infer critical behavior from source code.
- Feedback is treated as noise rather than signal.
- Review comments are correct but demotivating or unclear.
- The project optimizes for builders rather than users.
- Adoption is assumed to follow automatically from technical quality.

## Expected output from this reviewer

- Adoption and stakeholder review.
- Onboarding and teaching gaps.
- Contributor-experience recommendations.
- Human-risk notes: alignment, trust, support, review culture.
- User-empathy improvements.

## Acceptance bar

A plan passes this lens when the technical system is also teachable, adoptable, reviewable, contributable, and aligned with real user and team needs.

---

# Documentation Review Lens

**Reviewer assignment:** Assign this lens to the reviewer who will evaluate whether the work can be understood, reproduced, taught, operated, and maintained through written artifacts.

## What this reviewer should treat themselves as

The Documentation reviewer should treat themselves as the owner of clarity and institutional memory. Their job is not to polish prose at the end; it is to ask whether the project has the written explanations needed for users, contributors, operators, reviewers, and future maintainers to understand the system without relying on hallway explanations.

## Core mission

This reviewer protects the project from undocumented brilliance. They should inspect architecture docs, vision docs, tutorials, API references, examples, benchmark notes, limitation statements, security notes, code comments, RFCs, migration guides, and issue writeups. They should ensure documentation is treated as part of the product and review process, not as a late accessory.

## Skill items represented by this lens

- Architecture documentation: write clear design docs for runtimes, graph compilers, model runners, plugins, distributed execution, memory management, and security boundaries.
- Vision documents and manifestos: explain why the system should exist, whom it serves, and what tradeoffs it deliberately makes.
- Tutorials and examples: provide runnable learning paths such as custom ops, MLX examples, LLM inference examples, serving guides, and extension tutorials.
- API references and developer guides: maintain reference docs, build instructions, contribution guides, extension guides, configuration docs, and usage examples.
- Benchmark and reproducibility notes: document hardware, model, workload, settings, versions, and methods so results can be interpreted.
- Code comments for systems software: explain invariants, edge cases, lifecycle behavior, and mental models where source alone is insufficient.
- RFCs, security notes, and issue writeups: make decisions reviewable, traceable, and understandable after the fact.
- Honest limitation writing: document supported paths, unsupported paths, known risks, deliberate exclusions, and tradeoffs.

## What to inspect

### Code review

- Check whether comments explain non-obvious invariants, not obvious syntax.
- Review public API docstrings, config docs, error explanations, and examples near changed code.
- Ensure benchmark changes include reproducibility context.
- Check whether new limitations or behavior changes are documented.
- Look for docs that will become stale because they repeat implementation details unnecessarily.

### Architecture review

- Require a design doc for major architecture decisions.
- Check whether diagrams or narratives explain control flow, data flow, lifecycle, and boundaries.
- Review whether alternatives and rejected designs are recorded.
- Check whether security, performance, compatibility, and migration implications are written down.

### Project or plan review

- Define documentation deliverables for each milestone.
- Ensure docs cover install, quickstart, concepts, examples, API reference, operations, troubleshooting, and limitations.
- Check whether documentation ownership is assigned.
- Make sure public claims are supported by reproducible notes and clear scope.

## Questions this reviewer should ask

- Can someone understand why this exists and when to use it?
- Can someone install, run, test, debug, and extend it from docs alone?
- Are architectural decisions and tradeoffs recorded?
- Do benchmarks include enough context to reproduce and interpret them?
- Are unsupported cases and limitations stated honestly?
- Do comments preserve invariants and mental models for maintainers?
- Who owns keeping docs updated after code changes?

## Typical red flags

- Docs explain how to run a demo but not the concepts.
- Architecture exists only in the heads of original authors.
- Benchmark claims lack hardware, model, or workload details.
- Limitations are hidden because they sound bad.
- Comments are absent in complex systems code or only restate syntax.
- There is no migration guide for breaking changes.
- Docs are not part of the review checklist.

## Expected output from this reviewer

- Documentation gap review.
- Required docs before merge/release.
- Concept/tutorial/API/operations coverage notes.
- Benchmark reproducibility requirements.
- Limitation and decision-record recommendations.

## Acceptance bar

A plan passes this lens when documentation is sufficient for learning, correct use, contribution, operation, review, and future maintenance—and when important limitations and decisions are written honestly.

---


---

# Source basis from the previous research

This review-lens document is synthesized from the earlier grouped-skill analysis of:

- **Chris Lattner / Mojo / MAX**: language design, compiler infrastructure, MLIR/LLVM-style systems, Mojo, MAX, custom ops, graph/runtime/platform strategy, and organizational leadership around Modular.
- **Antirez / ds4**: small self-contained systems, Redis-style systems craftsmanship, local inference constraints, DeepSeek-specific design, low-level C-style reasoning, comments, benchmarking, and deliberate scope control.
- **vLLM**: high-throughput LLM serving, PagedAttention, KV-cache management, scheduling, batching, plugins, distributed inference, serving APIs, security boundaries, and production-oriented inference architecture.
- **MLX**: Apple silicon ML framework design, unified memory, array semantics, lazy evaluation, graph/function transforms, local LLM workflows, Swift/Python usability, Metal acceleration, distributed communication, and examples/docs.

The categories are intentionally broad. They are not job titles; they are review responsibilities. In a small team, one person may own multiple lenses. In a large review, assigning lenses explicitly prevents the review from collapsing into only API taste, only benchmark numbers, or only implementation details.
