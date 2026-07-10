# Fornax Plan Review (Codex)

Reviewed artifacts:

- `docs/fornax/README.md`
- `docs/fornax/project-plan.md`
- `docs/fornax/partitioner-spec.md`
- `docs/fornax/apple-silicon-max-skills.md`
- `docs/fornax/review_lenses_by_skill_for_fornax.md`

Review mode: read-only review of the plan. I did not change the plan documents. This file is the review output requested by the user.

## Overall Review

### Summary judgment
Needs revision

### What looks strong
- The plan is directionally strong: it understands that Fornax is an inference engine built with parts of MAX, not a scheduling harness around existing servers.
- The core product thesis is sharp: make frontier-scale private MoE inference possible on heterogeneous consumer/prosumer hardware without accepting avoidable throughput loss.
- The plan correctly treats Apple Silicon support as a moving platform capability to be probed and captured early, not assumed away.
- The partitioner spec is a strong start. It names model structure, hardware inventory, links, placement, expected throughput, latency, and memory feasibility as first-class planning objects.
- The roadmap has the right broad sequence: planner, worker contract, batching, MoE expert surgery, heterogeneous frontier model, replication, then productization.

### Risks or missing details
- Blocker: The plan is not yet concrete enough about the first target model, exact hardware matrix, context length, concurrency target, memory budget, and fabric assumptions to prove that the architecture closes.
- High priority: The custom MAX surgery path needs a stricter reference implementation and correctness strategy before optimized distributed execution becomes the main line of work.
- High priority: The end-to-end request lifecycle is under-specified across tokenization, prefill, decode, KV ownership, remote expert execution, streaming, cancellation, failure, and cleanup.
- High priority: The plan needs sharper milestone gates that can kill or reshape the design early if network, Apple backend, or MAX internals fail to support the required path.
- Medium priority: The documents are thoughtful but still read more like an architectural thesis than an execution plan with owners, interfaces, tests, operational runbooks, and acceptance criteria.

### Questions for the author/team
- What is the first concrete frontier MoE target: model family, parameter count, active parameter count, expert count, quantization, context length, and expected tokens per second?
- What is the initial hardware acceptance matrix: exact Mac Mini/Mac Studio generation, Apple memory size, NVIDIA/AMD GPUs, CPU/RAM, NICs, switch, and OS versions?
- What is the smallest useful demonstration that proves the Fornax thesis without requiring the whole product to exist?
- Which MAX interfaces are stable enough to build against, and which areas are expected to require vendor-facing collaboration or rapid churn handling?
- What throughput loss is acceptable relative to an imaginary homogeneous cluster of equal aggregate compute, and what loss makes the project miss its purpose?

### Required changes
- Define a Phase 0/Phase 1 acceptance target with exact model, hardware, context, concurrency, and throughput numbers.
- Add a memory budget table covering weights, experts, KV cache, activations, routing metadata, temporary buffers, fragmentation headroom, and OS/runtime reserve.
- Add an end-to-end request lifecycle spec that names every state owner and every data movement path.
- Add a correctness-first reference path for distributed MoE inference before optimized MAX surgery is treated as viable.
- Add a milestone gate matrix that says what evidence advances, pauses, or kills each major architectural bet.

### Non-blocking suggestions
- Keep the current documents, but add a short `fornax-v0-target.md` that acts as the executable target contract for the first prototype.
- Use one concrete model and one concrete hardware bundle throughout examples, even if the architecture remains general.
- Track assumptions explicitly as assumption IDs so benchmark results can retire or revise them.

### Final note from this lens
The plan is promising and unusually well aimed at the actual hard problem. It needs to become more falsifiable before serious implementation or hardware investment.

## Hardware Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan correctly treats heterogeneous hardware as the product reality, not an inconvenience.
- It separates hot accelerators from capacity accelerators and recognizes that Apple Silicon memory can be valuable even when its compute profile differs from NVIDIA or AMD GPUs.
- It understands that the network fabric is part of the machine abstraction and cannot be left as a deployment detail.
- It calls out memory feasibility, link bandwidth, remote expert costs, and stage placement in the partitioner spec.

### Risks or missing details
- Blocker: The plan lacks a named target hardware matrix for v0. A Mac Mini M3 with 125 GB RAM and a two-GPU RTX or AMD box are useful examples, but the plan needs exact models, memory capacities, interconnects, NIC speeds, and OS/runtime versions.
- Blocker: There is no full memory budget for the first model. Without weights, active experts, KV cache, activations, routing buffers, runtime overhead, and safety margin, the plan cannot prove that the model exceeds one node yet fits the private cloud.
- High priority: The network assumption is stated as provisioned to match the workload, but the plan needs actual fabric classes and thresholds: 10/25/40/100/200 GbE, latency budget, RDMA availability, and switch oversubscription.
- High priority: Hardware acceptance criteria are not yet concrete enough to guide purchasing, topology design, or benchmark triage.
- Medium priority: The plan does not yet distinguish thermal, power, and sustained performance behavior for consumer GPUs and Mac Mini/Mac Studio hardware.

### Questions for the author/team
- Which exact Apple Silicon machines are in scope first: M3 Mac Mini, M4 Mac Mini, Mac Studio, or all of them behind capability probes?
- What is the first NVIDIA or AMD accelerator class: RTX 4090, RTX 5090, workstation RTX, Radeon Pro, MI-series, or a mixed rack?
- What network class is assumed for first success: commodity 10 GbE, 25 GbE, 100 GbE, or RDMA-capable fabric?
- What are the minimum sustained tokens/sec and maximum TTFT for the target personal-agent backend workload?
- Is the Mac tier intended for KV-heavy decode, expert hosting, cold expert capacity, or all of these depending on profiling?

### Required changes
- Add a v0 hardware table with exact device classes, memory, accelerator type, network adapter, expected bandwidth, expected latency, OS, and MAX backend readiness.
- Add a memory feasibility worksheet for the first target model and hardware bundle.
- Define hardware acceptance tests: local GEMM/attention throughput, inter-node transfer bandwidth, remote expert round-trip time, sustained thermal behavior, and failure/rejoin behavior.
- Define minimum fabric tiers and what workload each tier can support.

### Non-blocking suggestions
- Use three named bundles: `desktop-minimal`, `prosumer-rack`, and `lab-reference`.
- Record measured hardware data in machine-readable inventory files that feed the partitioner.
- Include a negative hardware list for configurations that are possible but not worth supporting initially.

### Final note from this lens
The hardware intuition is good. The plan now needs hard numbers, named machines, and measurement gates so the architecture does not drift into wishful capacity aggregation.

## Low-Level Software Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan correctly identifies that this is custom inference-engine work inside MAX/Mojo boundaries, not orchestration outside the runtime.
- The partitioner names stage workers, expert workers, tensor boundaries, and serialized execution plans.
- The Apple Silicon skills document calls out Mojo GPU programming, MAX graph internals, custom ops, ABI design, and buffer lifecycle as necessary skills.
- The plan has the right instinct to maintain reference-vs-optimized correctness checks.

### Risks or missing details
- Blocker: Buffer ownership and lifetime are not specified. KV pages, activation tensors, expert batches, routing metadata, and network receive buffers need explicit ownership rules.
- High priority: Tensor layout contracts are missing. The plan needs shapes, strides, dtype, quantization packing, alignment, padding, and device/host residency rules for every cross-worker payload.
- High priority: It is unclear where MAX graph execution ends and custom distributed runtime execution begins.
- High priority: There is no build/toolchain plan for multi-platform Mojo/MAX work across Linux GPU nodes and macOS Apple Silicon nodes.
- Medium priority: Error handling at low-level boundaries is not yet specified: malformed tensor payloads, dtype mismatch, stale plan IDs, missing expert shards, and device allocation failure.

### Questions for the author/team
- What is the initial internal ABI for `ActivationsOut`, `ExpertBatch`, `ExpertResult`, and KV page references?
- Are remote expert calls represented as graph custom ops, sidecar runtime calls, or explicit worker RPC outside the graph?
- Which objects are immutable per plan, per request, per token, and per microbatch?
- How will optimized kernels be compared against reference kernels for bit-level or tolerance-level correctness?
- How will macOS and Linux builds be kept reproducible while MAX Apple support is still maturing?

### Required changes
- Add a low-level runtime contract document for tensor layouts, buffer ownership, lifetimes, device residency, dtype rules, and error handling.
- Add a reference execution mode that uses the same placement plan but conservative CPU or local implementations for correctness comparison.
- Define the boundary between MAX graph components, custom Mojo kernels, network transport, and scheduler-owned state.
- Add a build matrix for Linux NVIDIA, Linux AMD if applicable, and macOS Apple Silicon.

### Non-blocking suggestions
- Start with a tiny MoE model fixture that exercises routing, remote experts, and KV cache without large-model complexity.
- Generate ABI schemas from one source of truth rather than duplicating shape/dtype assumptions in multiple modules.
- Maintain a debug mode that can dump small tensor payloads and routing decisions deterministically.

### Final note from this lens
The plan knows low-level work is central, but it needs crisp contracts before implementation. This is where ambiguity will become expensive fastest.

## High-Level Software Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The Fornax/Ignis separation is clear: Ignis can remain the application/operator layer while Fornax owns distributed inference execution.
- The plan orients toward a single endpoint abstraction, likely OpenAI-compatible, which is the right user-facing shape for agentic harnesses.
- It avoids pretending that `max serve` alone can solve distributed heterogeneous execution for a model bigger than one node.
- The roadmap separates planner, worker, batching, MoE runtime, and productization phases cleanly.

### Risks or missing details
- High priority: The user journey is not specified from install to first successful generation.
- High priority: Public APIs, private runtime APIs, config schemas, and CLI commands are not yet named.
- High priority: Debuggability for users is underspecified. A private-cloud owner needs to know why a placement failed, why throughput is low, or why a node was excluded.
- Medium priority: The plan does not yet define compatibility surface with existing agent harnesses beyond the general idea of an endpoint.
- Medium priority: There is no concrete multi-tenant or single-tenant stance for personal use, small firm use, and enterprise private AI use.

### Questions for the author/team
- What does the first user run: `fornax inventory`, `fornax plan`, `fornax serve`, or an Ignis command that wraps these?
- What is the minimum config file a user must write to describe the private cloud?
- What APIs are considered stable for users, and what APIs are explicitly internal?
- How does a user inspect placement, bottlenecks, memory use, and remote expert traffic?
- Is OpenAI-compatible chat completion the first serving target, or is a lower-level generate API first?

### Required changes
- Add a first-user workflow: install, discover hardware, validate fabric, download/load model, plan placement, serve, run a test prompt, inspect metrics.
- Define the initial CLI and config schema, even if provisional.
- Define public API boundaries and internal API boundaries.
- Add user-facing failure messages and debugging artifacts to the plan.

### Non-blocking suggestions
- Add one sample `cluster.yaml`, one sample `model.yaml`, and one sample generated `placement.json`.
- Keep early UX brutally practical: inventory, validation, planning explanation, launch, metrics.
- Provide an explicit statement that Fornax is not a general Kubernetes replacement or generic cluster scheduler.

### Final note from this lens
The high-level product shape is good. The plan needs to show what a real operator sees and touches during the first hour.

## LLM Expertise Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan correctly focuses on MoE because the user goal is specifically to run models that exceed the largest single node while preserving throughput.
- It recognizes prefill/decode asymmetry, KV cache pressure, expert placement, continuous batching, and expert locality.
- It treats tokenization and chat template handling as reusable from Ignis rather than reinventing them casually.
- It understands that bounded remote expert execution is plausible, while unbounded all-to-all is dangerous for v1.

### Risks or missing details
- Blocker: There is no model-specific correctness plan for tokenization, chat templates, logits, sampling, stop behavior, streaming, and cancellation.
- High priority: KV cache lifecycle is mentioned but not specified. Placement, migration, page ownership, eviction, and cleanup must be explicit.
- High priority: Quality impacts of quantization and heterogeneous numerical differences are not defined with acceptance tests.
- High priority: MoE routing semantics need stronger treatment: top-k routing, expert capacity, dropped tokens if any, router precision, expert locality, and deterministic replay.
- Medium priority: Tool calling, structured outputs, long-context behavior, and agentic workload patterns are not yet discussed.

### Questions for the author/team
- Which exact MoE architecture is the first target: Mixtral-like, DeepSeek-like, Qwen-MoE-like, or another family?
- What is the reference engine for correctness comparison: native MAX single-node, Hugging Face/PyTorch, llama.cpp-style output checks, or model-provider traces?
- What tolerance is acceptable for logits and generated outputs across heterogeneous accelerators?
- How are streaming and cancellation propagated through in-flight remote expert work?
- What agentic workload is the benchmark: short tool calls, long-context coding, RAG-heavy prompts, or multi-turn planning?

### Required changes
- Add a model support matrix with architecture, tokenizer, chat template, quantization formats, context length, and tested runtime features.
- Add LLM acceptance tests for prompt formatting, prefill, decode, KV reuse, streaming, stop sequences, cancellation, and deterministic replay under fixed seeds.
- Add a KV cache ownership and lifecycle spec.
- Add quality gates for quantized and heterogeneous execution against a reference output.

### Non-blocking suggestions
- Use a tiny MoE fixture and one real open-weight MoE as separate test tiers.
- Include a small golden-prompt corpus with expected formatting, routing traces, and sanity output checks.
- Treat agentic harness traffic as a benchmark class, not just generic chat throughput.

### Final note from this lens
The plan understands the LLM shape of the problem. It needs model-specific behavioral contracts before distributed performance work can be trusted.

## Hardware Acceleration Expertise Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan correctly focuses on continuous batching, overlap, stage sizing, replication, locality, and quantization as throughput levers.
- It treats Apple, NVIDIA, and AMD as capability-probed backends rather than pretending all accelerators behave the same.
- It understands that custom Mojo kernels and MAX graph-level integration may be necessary.
- It identifies profiling as a required skill and not an afterthought.

### Risks or missing details
- High priority: There is no operation coverage table mapping attention, MLP, router, expert GEMMs, collect/scatter, KV cache ops, sampling, and transfers to available optimized kernels per backend.
- High priority: The plan lacks per-kernel and per-stage performance budgets.
- High priority: Fallback behavior is unclear when Apple Silicon support is incomplete or when a kernel is fast on NVIDIA but weak on Apple.
- Medium priority: Cross-backend numerical drift and dtype differences need explicit validation.
- Medium priority: It is unclear whether AMD is a near-term target or a design constraint to preserve.

### Questions for the author/team
- Which hot operations are expected to run on Apple Silicon in v0, and which are explicitly out of scope until MAX support matures?
- What is the minimum viable accelerated path for a Mac node to be useful: expert GEMM, decode layer stage, KV storage, or CPU-assisted capacity?
- Which profiler stack will be used on macOS and Linux to produce comparable traces?
- What is the threshold for demoting a device from compute participant to memory/capacity participant?
- Which quantization formats must be optimized first?

### Required changes
- Add a backend operation coverage matrix for Apple, NVIDIA, and AMD.
- Add performance budgets for each critical op and stage class.
- Add profiler gates that determine whether Apple nodes can participate in hot decode, prefill, expert hosting, or only cold capacity.
- Add correctness tests for optimized kernels against reference implementations.

### Non-blocking suggestions
- Make kernel availability and measured throughput part of inventory discovery.
- Keep an explicit fallback mode that proves correctness even when acceleration is poor.
- Separate backend portability design from backend parity promises.

### Final note from this lens
The acceleration strategy is plausible, but it needs measurement-driven backend commitments. Apple support should be captured aggressively, but not assumed into the critical path without gates.

## Networking Expertise Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan correctly rejects treating the network as generic plumbing.
- The partitioner includes link bandwidth and latency and attempts to model remote expert costs.
- The architecture leans toward pipeline parallelism as the default distributed shape, which is more network-realistic than unrestricted per-token all-to-all.
- The plan names backpressure, timeouts, and partial failure as concerns.

### Risks or missing details
- Blocker: There is no end-to-end network protocol or request-flow spec.
- High priority: Control plane and data plane responsibilities are not separated clearly.
- High priority: Backpressure is conceptually mentioned but not specified across scheduler queues, stage workers, expert workers, network buffers, and client streaming.
- High priority: Timeout, retry, cancellation, and partial failure semantics are missing for in-flight token generation.
- High priority: Trust boundaries and local-network security are not defined.
- Medium priority: Serialization format, compression, zero-copy strategy, and tensor chunking are not specified.

### Questions for the author/team
- What transport is assumed first: gRPC, custom TCP, QUIC, UCX, RDMA-capable transport, or MAX-native mechanisms?
- Is the network payload tensor data, KV references, expert inputs, expert outputs, logits, or some combination by phase?
- What happens when an expert worker is late for one microbatch but healthy overall?
- How does cancellation from a client abort in-flight stage and expert work?
- How are authentication, node admission, and plan integrity handled on a private LAN?

### Required changes
- Add a networking spec with request lifecycle, control plane, data plane, message types, tensor payload format, timeout policy, retry policy, cancellation policy, and backpressure rules.
- Add fabric validation benchmarks and minimum pass/fail thresholds.
- Add failure-mode tests for node loss, slow node, packet loss, stale worker, and plan mismatch.
- Add a security stance for private-cloud operation, even if v0 is simple mutual trust plus network isolation.

### Non-blocking suggestions
- Keep v0 transport boring unless measurements prove it cannot meet the target.
- Emit per-link metrics that the partitioner can consume in future planning runs.
- Document when remote expert execution is disabled by the planner because the measured fabric cannot sustain it.

### Final note from this lens
Networking is one of the decisive risks for Fornax. The plan names it, but now needs protocol-level detail and failure semantics.

## Software Engineering Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The project scope is better defined after the Fornax docs: custom distributed MoE inference engine, not generic private-cloud orchestration.
- The phased roadmap gives a reasonable implementation sequence.
- The partitioner spec lends itself to model-free unit tests before large-model execution exists.
- The plan keeps productization later, which is sensible given the depth of runtime work.

### Risks or missing details
- High priority: Repository/module boundaries are not defined. It is unclear where planner, runtime, worker, MAX integration, transport, benchmarks, and docs will live.
- High priority: Testing strategy is underspecified across unit, integration, distributed, hardware-in-loop, correctness, regression, and benchmark tests.
- High priority: CI feasibility is unclear because the target hardware includes devices unlikely to be available in ordinary CI.
- Medium priority: Release and packaging strategy is absent for mixed Linux/macOS deployments.
- Medium priority: Dependency/version pinning for MAX, Mojo, drivers, macOS, CUDA/ROCm if any, and model artifacts is not described.

### Questions for the author/team
- What is the initial module map in the Ignis repository or future Fornax repository?
- What tests must pass without accelerators, with one accelerator, and with a distributed hardware lab?
- How will benchmark results be stored and compared over time?
- How will compatibility be maintained as MAX APIs evolve?
- What is the strategy for experimental code that must move fast but not destabilize Ignis?

### Required changes
- Add an implementation skeleton plan: modules, package boundaries, generated schemas, test fixtures, and benchmark locations.
- Add a CI/test matrix that distinguishes CPU-only simulation, single-node accelerator tests, and distributed lab tests.
- Add dependency/version policy for MAX, Mojo, model artifacts, OS, and accelerator drivers.
- Add benchmark-result persistence and regression thresholds.

### Non-blocking suggestions
- Keep the partitioner independently testable with JSON fixtures.
- Add deterministic simulated workers early to validate scheduling and backpressure without hardware.
- Treat benchmark harness code as production code, not disposable scripts.

### Final note from this lens
The plan is engineerable, but it needs a concrete implementation and test spine before the deeper runtime work starts.

## Organizational Skill Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan is candid about risk and does not hide the hard parts behind vague orchestration language.
- The roadmap phases are logically ordered.
- The Apple Silicon skills document gives a realistic staffing map across Mojo/MAX, GPU kernels, MoE inference, distributed systems, and benchmarking.
- The plan recognizes that upstream MAX capability changes must be tracked actively.

### Risks or missing details
- High priority: There are no named owners or role assignments for planner, MAX integration, Apple backend, networking, benchmarks, and LLM correctness.
- High priority: The smallest useful milestone is not crisp enough. Phase 0 is useful, but it needs a demo and acceptance contract.
- High priority: Decision gates are missing. The team needs explicit evidence thresholds for continuing, pivoting, or narrowing scope.
- Medium priority: The plan does not yet define communication cadence, review checkpoints, or external dependency tracking with Modular/MAX updates.
- Medium priority: Resourcing assumptions are missing. This is unlikely to be a one-person project beyond planning/prototype work.

### Questions for the author/team
- Who owns each major risk: Apple readiness, MAX internals, distributed runtime, MoE correctness, networking, and benchmarking?
- What milestone proves Fornax is worth continued investment?
- What evidence would cause the team to drop Apple from the hot path temporarily?
- What evidence would cause the team to abandon remote expert execution for v0?
- How will the team track Modular/MAX changes and decide when to rebase implementation assumptions?

### Required changes
- Add an ownership/RACI table for the next two phases.
- Add milestone gates with specific artifacts, metrics, and go/no-go criteria.
- Add an explicit risk burn-down plan ordered by uncertainty and cost of being wrong.
- Add a staffing plan that separates required skills from optional acceleration.

### Non-blocking suggestions
- Run architecture reviews at the end of each phase with the lens file as the rubric.
- Keep a decision log for major architecture commitments.
- Maintain a public-facing narrative separately from the internal risk ledger.

### Final note from this lens
The plan has strategic clarity. It needs execution ownership and decision discipline so the team can move fast without turning every uncertainty into permanent scope.

## Analytical Skills Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan is built around a real hypothesis: heterogeneous private-cloud hardware can run models that exceed any one node while preserving useful throughput.
- It identifies likely bottlenecks: network transfer, KV memory, stage imbalance, Apple backend readiness, remote experts, and heterogeneous numerics.
- The partitioner spec is an analytical asset because it can compare placement alternatives before implementation is complete.
- The roadmap includes simulation/planning before full runtime work.

### Risks or missing details
- High priority: The plan needs explicit falsification experiments for each major bet.
- High priority: Baselines are not defined. Fornax should be compared against single-node execution, homogeneous multi-GPU where available, naive pipeline, expert-only offload, and existing inference engines where applicable.
- High priority: Sensitivity analysis is missing for context length, concurrency, expert locality, network bandwidth, stage imbalance, and quantization.
- Medium priority: The plan does not yet specify how measured data updates the planner cost model.
- Medium priority: Tradeoff choices are explained qualitatively but not yet quantified.

### Questions for the author/team
- What experiment would disprove the claim that Mac memory usefully expands frontier MoE serving capacity?
- What experiment would disprove remote expert execution for the target workload?
- What is the minimum throughput efficiency target relative to a comparable homogeneous setup?
- Which bottleneck is expected first: network, Apple kernel throughput, KV capacity, scheduler overhead, or MAX integration overhead?
- How will benchmark noise and thermal variance be handled?

### Required changes
- Add a falsification matrix with hypothesis, experiment, hardware, workload, pass threshold, fail threshold, and decision.
- Add baseline systems and baseline placement strategies.
- Add sensitivity sweeps for context length, batch size, concurrency, expert hit rate, link bandwidth, and device speed ratios.
- Add a model for how benchmark results recalibrate planner estimates.

### Non-blocking suggestions
- Use synthetic traces before real model traces, but require real traces before architecture lock-in.
- Keep a `known-wrong` section in the planner docs for model assumptions that are intentionally approximate.
- Publish benchmark methodology with enough detail to reproduce failures, not just successes.

### Final note from this lens
The plan has good analytical instincts. It needs experiments designed to change decisions, not only confirm the preferred architecture.

## System Engineering Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The plan treats Fornax as a system with layers: model spec, hardware inventory, partitioner, runtime, worker contracts, batching, MoE execution, and serving.
- The Fornax/Ignis boundary is sensible and reduces conceptual sprawl.
- The architecture acknowledges integration risk across MAX, Mojo, Apple Silicon, network fabric, and application-serving semantics.
- The phased path allows incremental validation if the gates are strengthened.

### Risks or missing details
- Blocker: The end-to-end lifecycle is not yet specified from client request through tokenization, scheduling, prefill, decode, remote expert calls, streaming, cancellation, metrics, and cleanup.
- High priority: Layer ownership is not explicit enough. KV cache, plan state, worker state, routing state, and client stream state need named owners.
- High priority: Observability is under-specified. A distributed inference system needs metrics, traces, structured logs, plan snapshots, and per-request debug correlation from the beginning.
- High priority: Operational readiness is not covered: deploy, upgrade, drain, restart, recover, validate, and roll back.
- Medium priority: Security and trust assumptions need a v0 stance.

### Questions for the author/team
- What is the authoritative state machine for a request?
- Which component owns admission control and continuous batching decisions?
- Which component owns KV cache allocation and cleanup?
- How is a worker drained or restarted without corrupting in-flight requests?
- How does the system explain poor performance to an operator?

### Required changes
- Add an end-to-end sequence diagram and request state machine.
- Add a state ownership table for model state, KV cache, microbatches, expert routes, worker health, placement plans, and client streams.
- Add observability requirements for metrics, tracing, logs, debug dumps, and benchmark correlation.
- Add an operational lifecycle: bootstrap, validation, serving, drain, restart, failure recovery, and upgrade.

### Non-blocking suggestions
- Make the first implementation observable before it is fast.
- Keep plan IDs and request IDs visible in all logs and metrics.
- Add a `fornax doctor` concept for operator-facing system validation.

### Final note from this lens
Fornax is fundamentally a systems project. The plan needs lifecycle, state ownership, and observability to match the ambition of the architecture.

## People Skills Review Lens Review

### Summary judgment
Approve with comments

### What looks strong
- The documents explain the goal in plain terms and preserve the original product motivation: frontier-level private AI for individuals and firms using hardware they can actually own.
- The plan is honest about uncertainty, which will help align technical and non-technical stakeholders.
- The skills document is useful for recruiting, delegation, and identifying gaps.
- The architecture avoids dismissing consumer hardware; it treats the user's intended audience with respect.

### Risks or missing details
- Medium priority: Contributor onboarding is not yet planned. New engineers will need a guided path through Mojo, MAX, MoE inference, distributed runtime, and benchmark interpretation.
- Medium priority: Stakeholder messaging needs separation between aspirational vision and validated capability.
- Medium priority: The plan does not yet define how hardware owners report inventory, failures, benchmark results, or usability pain.
- Low priority: The documentation could do more to explain why certain tempting paths are out of scope for v0.

### Questions for the author/team
- Who is the first operator persona: solo expert user, small AI lab, enterprise IT owner, or internal developer?
- What should a contributor learn first to become useful in one week?
- How will the team communicate experimental status without overpromising?
- What feedback should early private-cloud users be asked to provide?
- Which parts of the system should be easy to contribute to before someone understands MAX internals?

### Required changes
- Add an onboarding path for new contributors and operators.
- Add a plain-language status model: experimental, validated on lab hardware, supported target, and unsupported.
- Add feedback artifacts for early users: hardware inventory report, benchmark bundle, failure report, and placement explanation.

### Non-blocking suggestions
- Create separate docs for `operator`, `runtime contributor`, and `kernel contributor` personas.
- Keep a short glossary for MAX, Mojo, MoE, KV cache, prefill, decode, expert routing, and placement.
- Use review gates to keep stakeholder expectations grounded.

### Final note from this lens
The plan communicates ambition well. It should now make participation easier and expectation-setting sharper.

## Documentation Review Lens Review

### Summary judgment
Needs revision

### What looks strong
- The Fornax docs have a coherent narrative and are much more specific than a generic distributed inference proposal.
- The partitioner spec is a strong technical artifact and gives the project a concrete planning core.
- The Apple Silicon skills document captures a useful map of the skills needed to work with MAX on Apple hardware.
- The review-lens rubric itself is excellent and should remain part of the project process.

### Risks or missing details
- High priority: The docs do not yet provide enough detail for an engineer to implement the first vertical slice without substantial oral context.
- High priority: There is no quickstart or operator workflow.
- High priority: Benchmark methodology and reproducibility are not yet documented.
- Medium priority: Diagrams are missing for request flow, placement, worker topology, and data movement.
- Medium priority: Limitations and unsupported configurations are discussed conceptually but not listed in an operator-facing way.
- Medium priority: Architecture decisions are not yet captured as decision records.

### Questions for the author/team
- Which document is the source of truth for the next implementation milestone?
- Where should an engineer find the current target hardware and model assumptions?
- Where should benchmark data and benchmark methodology live?
- How should docs distinguish stable commitments from active research assumptions?
- Should Fornax docs live inside Ignis long term, or move to a dedicated Fornax package/repository later?

### Required changes
- Add a v0 target contract document with exact model, hardware, workload, and acceptance metrics.
- Add diagrams for architecture, request lifecycle, and data movement.
- Add a benchmark methodology document with reproducible commands, input prompts/traces, metrics, and pass/fail thresholds.
- Add an ADR log for major decisions: pipeline default, bounded remote experts, Apple support strategy, MAX surgery boundary, and transport choice.
- Add a quickstart once the initial CLI/config shape is defined.

### Non-blocking suggestions
- Keep assumptions in tables rather than prose where possible.
- Add a glossary to reduce onboarding cost.
- Make every phase end with expected artifacts, not only engineering themes.

### Final note from this lens
The documentation is a strong conceptual base. It needs executable specificity: targets, diagrams, commands, benchmarks, and decisions.

## Consolidated Required Changes

- Define the first target model, hardware bundle, context length, concurrency target, and throughput/latency acceptance criteria.
- Add memory budgets for weights, experts, KV cache, activations, routing metadata, temporary buffers, runtime overhead, and safety margin.
- Add an end-to-end request lifecycle spec covering tokenization, planning, scheduling, prefill, decode, remote experts, KV lifecycle, streaming, cancellation, failure, metrics, and cleanup.
- Add low-level ABI contracts for tensor payloads, dtype/layout, buffer ownership, device residency, and worker message types.
- Add LLM correctness tests for tokenizer/chat template, logits or tolerance checks, KV reuse, streaming, cancellation, stop behavior, and deterministic replay.
- Add networking protocol details for control plane, data plane, flow control, timeouts, retries, cancellation, partial failure, and trust boundaries.
- Add backend operation coverage and profiler gates for Apple, NVIDIA, and AMD.
- Add an implementation skeleton, module map, test matrix, CI/lab strategy, and benchmark-result tracking plan.
- Add ownership, staffing, milestone gates, and falsification experiments for the major bets.
- Add diagrams, benchmark methodology, quickstart/config examples, glossary, and ADRs.

## Consolidated Go/No-Go Recommendation

Do not treat the current plan as implementation-ready for the full Fornax thesis yet. Treat it as a strong architecture draft that should advance into a Phase 0 evidence sprint.

Recommended next gate:

- Produce a `v0 target contract` for one model and one hardware bundle.
- Build or simulate the partitioner against that target.
- Measure the real hardware/fabric primitives needed by the cost model.
- Prove a correctness-first MoE execution slice with deterministic traces.
- Decide, with evidence, whether Apple participates in hot compute, expert hosting, KV-heavy capacity, or later-stage support for v0.

If that gate passes, the plan becomes much more credible as a custom MAX/Mojo inference-engine effort. If it fails, the project still gains valuable evidence about which narrower architecture can deliver private frontier-scale utility on consumer hardware.
