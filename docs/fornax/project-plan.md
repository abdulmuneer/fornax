# Fornax — Project Plan

> A **Mojo/MAX-native distributed MoE inference engine** — a custom surgery of
> MAX — serving a single frontier-scale MoE model across heterogeneous commodity
> hardware (consumer NVIDIA/AMD + Apple Silicon, provisioned LAN fabric) at high
> aggregate throughput, on-prem, for in-house private AI.
>
> **Fornax is an engine, not a harness.** It is built *from MAX's components*
> (graph compiler, kernel library, KV primitives, custom-op API, model bring-up
> workflow) plus the parts MAX does not provide as one product surface:
> heterogeneous pipeline execution, cross-vendor activation/KV transport,
> topology-aware continuous batching, and a model-specific distributed MoE
> expert runtime. It plugs into the Ignis harness as a `FornaxBackend` behind the
> `Engine` trait, but Fornax itself owns model execution.

---

## 1. Vision & problem

Firms and individuals increasingly want **frontier-level capability in-house** —
for privacy, control, and cost. The frontier of open weights in 2026 is **large
sparse MoE** (DeepSeek-R1 671B/37B-active, Kimi-class ~1T, Qwen3-235B-A22B,
Llama 4 MoE). These models **do not fit a single consumer node**, and renting
8×H100 (~$250–400K, supply-constrained) defeats the point.

Three facts make a heterogeneous commodity cluster the *right* shape for this:

1. **MoE splits cost into capacity vs compute.** Holding all experts is
   capacity-bound (671B @ 4-bit is hundreds of GB, rarely touched per token);
   per-token work is the sparse active set plus dense attention. Apple unified
   memory is unusually good expert capacity; consumer NVIDIA/AMD is unusually
   good compute. The shapes match.
2. **MoE gives a surgical seam.** Dense transformer execution should stay on the
   fastest accelerator group, but routed expert MLPs can be placed, migrated,
   cached, and remotely executed because the router exposes exactly which expert
   batches are needed.
3. **Spanning comms can be engineered, but only if measured.** Pipeline
   parallelism moves activation tensors at stage boundaries; bounded remote
   expert dispatch moves hidden-state batches for selected experts. Both can be
   overlapped and batched on a provisioned local fabric. Tensor-parallel
   all-reduce across mismatched consumer nodes remains the pattern to avoid.

**The gap:** the existing embodiments of this idea (Exo, Petals, llama.cpp-RPC,
GPUStack) either don't span one model across vendors, or do it at low throughput
because they lack continuous batching, speed-balanced stage sizing, and stage
replication. **That gap is the product.**

## 2. The constraint <a id="the-constraint"></a>

When the model exceeds the biggest node, **every token crosses the network** —
topology, not tuning. A network sized for this workload changes the feasible
throughput envelope, but it does not erase the pipeline/synchronization floor.
Therefore:

- ❌ **Single-request latency parity with a single big node — impossible** once a
  request must traverse multiple devices. Do not promise it.
- ✅ **High aggregate throughput + utilization** (total tok/s, $/token) —
  achievable by saturating a balanced pipeline, overlapping transfers, and
  keeping decode-time expert traffic local or heavily batched.

The per-request latency rise (proportional to pipeline depth and remote expert
waits) is the **irreducible price of spanning**. Every design decision below
optimizes aggregate throughput and accepts that latency floor. This must be
stated plainly; overselling latency is the one sure way to lose trust.

## 3. Goals / Non-goals

**Goals**
- Serve one frontier MoE model that exceeds the largest single node, across a
  heterogeneous LAN, at high aggregate throughput.
- Absorb whatever hardware is present (mixed NVIDIA/Apple, mixed memory, mixed
  links) — "maximal utility from heterogeneous commodity hardware."
- Single OpenAI-compatible endpoint; the cluster looks like one machine to
  clients.
- Honest, cluster-wide telemetry (real cache/util, no fabricated metrics — the
  Ignis discipline).
- Elasticity: tolerate flaky consumer nodes joining/leaving/dying mid-request.

**Non-goals (v1)**
- Competing with vLLM/SGLang on datacenter throughput or single-stream latency.
- Unbounded cross-vendor MoE **expert-parallel all-to-all** as the default data
  path. Bounded remote expert execution is in-scope only when the measured fabric
  and batching model keep it throughput-neutral.
- Training / fine-tuning.
- WAN / internet-scale federation (Petals-style). LAN only.
- Serving models that comfortably fit one node (use a single `max serve` /
  MLX / llama.cpp instance — no cluster needed; say so).

## 4. Target deployment profile

The throughput requirement raises the realistic hardware floor: **prosumer nodes
on a fast home/office fabric**, not phones on WiFi.

- **Hot accelerator tier:** consumer NVIDIA/AMD (RTX 40/50 series, Radeon/MI
  where MAX support exists) — attention, prefill/decode, routers, shared
  experts, hot routed experts, KV cache.
- **Capacity/expert tier:** Apple Silicon Macs (M3+ now in scope for MAX/Mojo
  experiments; high-unified-memory systems are the target) — cold/warm experts,
  expert weight reservoir, pipeline stages once measured performance justifies
  them.
- **Fabric (highest-leverage decision):** design the network as part of the
  engine: direct Thunderbolt where available, 25–100 GbE or better on hot stage
  boundaries, measured latency/bandwidth per link, and topology-aware placement.
  WiFi/1 GbE is not part of the hot path.
- **Concurrency:** designed for *many concurrent requests* (agents, batch,
  multi-user firm workloads) — this is what fills the pipeline and makes
  throughput preservation possible.

## 5. Architecture

### 5.1 Parallelism decision (settled, with a MoE escape hatch)

| Strategy | Per-token comms | Verdict |
|---|---|---|
| Tensor parallel across mismatched nodes | 2 all-reduces/layer | ❌ avoid outside a homogeneous high-bandwidth island |
| **Pipeline parallel (spine)** | one activation tensor/stage | ✅ default LAN-viable spanning pattern |
| Bounded remote expert execution | selected hidden-state batches | ✅ in-scope when profiler + fabric make it cheap |
| Unbounded expert-parallel all-to-all | data-dependent all-to-all/MoE layer | ⚠️ not v1 default; only for topology-proven fabrics |

**Spine = pipeline parallel by layer-group**, with speed-balanced stage sizing.
For the robust default, each stage holds its layers' full active expert set so
MoE routing stays node-local and comms collapses to activation tensors at stage
boundaries.

**MoE expert-runtime mode** is the custom surgery: the dense path (attention,
KV, routers, shared experts, hot routed experts, sampler) stays on the fastest
accelerator group, while cold/warm routed experts may live on Apple/AMD/peer-GPU
workers. The router produces expert batches; Fornax executes local batches
inline, sends remote batches over the expert fabric, gathers weighted outputs,
and continues the layer.

```text
hidden_states
  -> router_topk
  -> bucket tokens by expert
  -> local expert batches + remote expert batches
  -> weighted gather in original token order
  -> next transformer block
```

The planner chooses between resident, remote-expert, and replicated-stage modes
from measured workload traces. Decode throughput depends on keeping remote
expert hits rare, batched, overlapped, or quickly migrated local.

### 5.2 The six throughput levers

1. **Continuous batching to fill the pipeline.** Converts PP from a latency tax
   into a throughput-neutral layout. The single biggest win and exactly what
   Exo/llama.cpp-RPC lack.
2. **Comms/compute overlap (1F1B / interleaved).** Transfer microbatch *n*'s
   activation while computing *n+1*; hides the network.
3. **Speed-balanced heterogeneous stage sizing — the crux.** A pipeline runs at
   its slowest stage; nodes differ 5–10×. Partition layers to equalize per-stage
   wall-time, bounded by each node's memory. This is the core IP → see
   [partitioner-spec.md](partitioner-spec.md).
4. **Data-parallel replication of bottleneck stages.** Replicate the limiting
   stage onto spare nodes, round-robin microbatches → a 2D (depth × width) grid.
   How throughput scales and how odd hardware gets absorbed.
5. **Expert locality + migration.** Treat remote expert hits as a budgeted
   resource. Keep shared/hot experts local, execute cold/warm experts remotely
   only when batched/overlapped, and migrate experts when decode traces show
   sustained hotness.
6. **Quantize weights + activations (4-bit / fp8).** Fits the model, halves
   transfer. Mandatory (671B fp16 = 1.3 TB → 4-bit ≈ 400 GB).

### 5.3 Layered components (OS analogy)

| Layer | Role | Provided by | Fornax builds |
|---|---|---|---|
| Machine model | Live inventory + measured link-bandwidth topology graph | — | ✅ |
| Per-node stage runtime | Execute a contiguous layer group as a MAX graph or graphlet; activation/KV handoff at boundaries | **MAX graph compiler + kernel library** plus model-specific Mojo kernels where needed | stage host, boundary custom ops, heterogeneous dispatch |
| Control plane | Partition planner, continuous-batching scheduler, placement, admission, cache-affinity routing | — | ✅ the engine's scheduler |
| Data plane | Activation transport (pluggable TCP / RDMA / Thunderbolt-IP / shm), topology-aware boundary placement | MAX custom op + NIXL/UCX intra-NVIDIA | ✅ cross-vendor |
| State | Distributed KV/prefix registry; elasticity + replay on node loss | — | ✅ (Ignis single-node seed) |
| Client surface | One OpenAI-compatible endpoint | — | thin |

### 5.4 Substrate: a surgery of MAX (not a harness over llama.cpp)

Fornax is an engine **assembled from MAX's components**, because MAX's
portable-by-design kernels (one Mojo source → NVIDIA/AMD/Apple) are *with the
grain* of a heterogeneous engine — more so than llama.cpp's separate CUDA/Metal
backends. We own the KV/activation format end-to-end, so cross-stage and
cross-vendor handoff is consistent by construction.

**Reuse from MAX:**
- Graph compiler + kernel library (the vendor-portable compute substrate).
- Model architecture definitions / weight loading (`max.pipelines`) where reusable.
- KV-cache management primitives.
- The custom-op API (`@compiler.register` / `ops.custom`) — the pipeline-stage
  boundary, cross-device activation/KV handoff, and expert dispatch authored as
  first-class graph ops (the Mojo-native frontier the repo already targets).

**Surgery / build:**
- Heterogeneous distributed pipeline execution — stage partitioning, transport,
  1F1B scheduling, and topology-aware continuous batching as engine internals.
- A first-class MoE expert runtime: expert placement, activation packing,
  local/remote dispatch, weighted gather, migration, and profiling.
- One KV/activation/expert-batch format owned across vendors.
- **Apple/Mac runtime readiness and model-specific kernel gaps — the critical
  path (§5.5).**

### 5.5 The critical path: Apple/Mac runtime readiness <a id="apple-runtime-readiness"></a>

The Apple story is no longer "no LLM path exists." Modular 25.6 introduced
Apple Silicon GPU support for Mojo GPU programming, and the 26.4 announcement
says MAX supports many common model architectures, including Qwen 3.6 and Gemma
4, on M3 and newer Apple Silicon GPUs. The docs still caution that general large
GenAI model inference on Apple Silicon is not uniformly available from the
package surface, so Fornax must treat Apple support as **fast-moving,
capability-probed, and model-specific**, not as a blanket assumption.

For Fornax this changes the work from "write the whole Apple forward pass" to
"make the target MoE family work correctly and fast on the Apple roles we assign
it." The first Apple role should be the **expert worker**, because an expert MLP
has a much smaller op surface than a full autoregressive pipeline:

```text
input:  [tokens_for_expert, hidden_dim]
output: [tokens_for_expert, hidden_dim]
ops:    dequant/load weights, matmul, activation, matmul, optional normalization
```

If MAX supplies a fast kernel or model component, use it. If a required target
op is missing or too slow, write the narrow Mojo kernel and delete it when MAX
catches up. Do not fork MAX wholesale. Keep the surgery at explicit seams:
activation transport, expert dispatch, expert MLP kernels, KV/page handoff, and
stage scheduling.

**Current upstream anchors to track:**

- Modular 25.6 Apple GPU direction: https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple
- Modular 26.4 MoE + Apple note: https://www.modular.com/blog/modular-26-4-sota-moe-serving-model-bringup-via-agent-skills-mojo-beta-2-and-more
- MAX package/platform caveats: https://docs.modular.com/max/packages/
- MAX custom ops: https://docs.modular.com/max/develop/build-custom-ops/

## 6. Roadmap (phased)

Each phase has a concrete exit criterion. Phases 0–2 need **no GPU** (pure logic
+ fixtures, Ignis-style); hardware enters at Phase 3. **Apple/Mac runtime
readiness (§5.5) is a parallel critical-path workstream** — start capability
probes and target-model expert-kernel prototypes early, because Apple can become
a first-class expert worker before it is wise to place arbitrary full pipeline
stages there.

- **Phase 0 — Planner (model-free).** The throughput-optimizing partitioner +
  cost model. *Exit:* given a model spec + inventory, emit a placement plan and a
  predicted (throughput, TTFT, per-request latency) at target concurrency;
  validated against hand-worked cases. See [partitioner-spec.md](partitioner-spec.md).
- **Phase 1 — Worker contract + transport (single-vendor, small model).**
  `FornaxEngine` + `StageWorker` RPC + activation transport; run a small model
  pipeline-parallel across 2–3 like nodes. *Exit:* correct generation across a
  real pipeline; activation transfer measured and overlapped.
- **Phase 2 — Continuous batching over the pipeline.** Microbatch scheduler,
  1F1B overlap, bubble-fraction telemetry. *Exit:* aggregate tok/s scales with
  concurrency up to pipeline saturation; matches planner prediction within X%.
- **Phase 2.5 — MoE expert-runtime surgery.** Replace one target model's MoE
  executor with explicit expert bucketing, local/remote dispatch, weighted gather,
  and expert activation tracing. *Exit:* layer-by-layer logits match the MAX or
  reference implementation; remote expert batches are measured independently.
- **Phase 3 — Heterogeneous + frontier MoE.** Span a real frontier MoE across
  NVIDIA/AMD + Mac with speed-balanced sizing, resident/hot experts on the hot
  tier, and bounded remote expert execution where profitable. *Exit:* end-to-end
  serve at the planner's predicted throughput on real heterogeneous hardware.
- **Phase 4 — Stage replication + elasticity.** DP replicas of bottleneck
  stages; node-loss → reschedule + replay in-flight microbatches. *Exit:*
  throughput scales with added nodes; a killed node does not drop in-flight
  requests.
- **Phase 5 — Productization.** Single endpoint, model catalog, privacy/audit,
  honest cluster telemetry, install/onboarding. *Exit:* a firm can stand it up
  and serve internal users.

## 7. Risks & open questions

- **R1 — Throughput ceiling is real.** Commodity net + flaky nodes cap aggregate
  throughput well below datacenter. Mitigation: levers 1–6; honest positioning
  ("maximal utility," not "cloud parity").
- **R2 — Pipeline depth vs latency.** Deep pipelines (many small nodes) tank
  per-request latency. Mitigation: planner penalizes depth; prefer fewer
  big-memory nodes when latency-sensitive.
- **R3 — Heterogeneous numerics.** NVIDIA bf16/fp8 vs Apple fp16/fp19-truncated
  matmul → cross-vendor stage seams can diverge. Mitigation: pin dtypes across
  seams; numeric validation harness.
- **R4 — Apple/Mac runtime readiness (CRITICAL PATH).** MAX Apple support is
  moving quickly but is not a blanket guarantee for the target MoE, quantization,
  and kernels Fornax needs. Mitigation: capability-probe every nightly; scope to
  one model family; start with expert MLP workers; write only missing narrow
  Mojo kernels; delete custom kernels as MAX ships equivalents (§5.5).
- **R5 — Remote expert bandwidth and latency.** Remote expert execution only
  preserves throughput when hits are sparse, batched, overlapped, or migrated
  local. Open: crossover batch size vs link bandwidth and decode remote-hit rate
  (planner must model it).
- **R6 — Surgery against fast-moving MAX internals.** Deep dependence on MAX
  nightly internals (the build-drift gotcha) is a maintenance burden as APIs
  churn. Mitigation: pin a verified build; isolate the surgery behind a thin
  internal interface; track MAX's Apple progress to retire custom code.
- **R7 — Mojo toolchain maturity** for authoring a full engine + GPU kernels on
  a beta toolchain. Mitigation: lean on MAX kernels for the common path; keep
  custom kernels minimal and well-tested.

## 8. Success metrics

- **Capacity:** serve a model N× the largest single node (target N ≥ 2–3).
- **Throughput efficiency:** aggregate tok/s ≥ Y% of the sum-of-nodes ideal at
  saturation (the bubble/overhead tax).
- **Planner accuracy:** predicted vs measured throughput within ±X%.
- **Expert locality:** decode-time remote expert wait below the configured SLO;
  hot experts migrate local automatically.
- **Correctness:** layer/logit divergence against the MAX/reference path within
  dtype-specific tolerances at every pipeline and expert seam.
- **Cost:** $/token and $/capacity vs an 8×H100 baseline (the headline pitch).
- **Elasticity:** zero dropped in-flight requests on single-node loss.
- **Honesty:** every reported metric traceable to a real measurement (no
  fabricated numbers — Ignis invariant, cluster-wide).

## 9. Relationship to Ignis

**Ignis is the harness; Fornax is the engine.** They are complementary layers,
not the same thing:

- **Fornax** owns model execution — the distributed heterogeneous forward pass,
  kernels, KV. It is the MAX surgery.
- **Ignis** owns the agent loop — timeline, policy/exact-confirm gate, tools,
  replay, telemetry.

Fornax plugs into Ignis behind the **`Engine` trait** as a `FornaxBackend`
(joining/replacing `MaxBackend`) — `generate(messages_json, tools_json,
max_new_tokens) -> EngineResult` is the seam. Disciplines carried across: honest
KV/cache telemetry → cluster-wide utilization; **replay bytes** → reschedule
dropped microbatches on node loss; the **model-free fixture discipline** → the
planner/scheduler (Phase 0–2) are testable with no hardware. Fornax can also be
driven standalone (OpenAI-compatible endpoint) for non-Ignis harnesses.
