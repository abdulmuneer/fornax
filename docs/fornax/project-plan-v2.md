# Fornax — Project Plan (v2)

> **Version 2.** Supersedes [project-plan.md](project-plan.md) (v1), which is
> **preserved unchanged** so that the review chain
> ([claude](fornax_plan_review_claude.md) · [codex](fornax_plan_review_codex.md) ·
> [reconciled](fornax_plan_review_reconciled.md)) stays mapped to the artifact it
> graded. This v2 is the revision that *answers* that reconciled review. Every
> change below is tagged with the blocker / priority it resolves, e.g. `(resolves
> B3)`.
>
> Authoritative inputs: `fornax_plan_review_reconciled.md`. Companion artifacts it
> demands are referenced here and gated into Phase 0 (§10); they are **not yet
> written** — producing them is the next sprint.

## Changes in v2 — review → change map

| Review item | Severity (reconciled) | Resolved in v2 | Companion artifact (Phase 0) |
|---|---|---|---|
| **B1** No quantitative feasibility proof | Blocker | §3.1, §4, §8 (provisional targets + contract authority) | `v0-target-contract.md` |
| **B2** Concurrency–market fit unproven | Blocker (thesis) | §3.1, §3.2, §4, §7 R8 | concurrency section in `v0-target-contract.md` |
| **B3** Cross-vendor format undefined | Blocker | §5.6 | `runtime-format-and-invariants.md` |
| **B4** Security / trust / backpressure missing | Blocker (multi-node) | §5.8, §7 R9 | `networking-security-and-backpressure.md` |
| **B5** Apple/MAX bet lacks Plan B | Blocker (Apple-critical) | §5.5, §10 | `adr/0001-max-mojo-substrate.md` |
| **B6** Phase 0–2 hardware contradiction | High | §6 (roadmap rewrite) | — |
| **P1** Hardware target matrix | High | §4 (named bundles + matrix) | `v0-target-contract.md` |
| **P1** Backend op coverage + calibration | High | §5.10 | — |
| **P1** Model support matrix | High | §5.7, §6 | — |
| **P1** Observability design | High | §5.9 | — |
| **P1** CI / hardware-lab test matrix | High | §6 (test tiers) | — |
| **P2** Operator UX / ops / onboarding / ADRs / glossary | Medium | §10 (scheduled, listed) | later |

**Preserved unchanged from v1** (the reconciled review's "What Should Not
Change"): engine-not-harness (header, §9), pipeline-parallel spine (§5.1),
bounded remote experts as measured optionality (§5.1), planner-first sequencing
(§6), the honest single-stream latency caveat (§2), MAX/Mojo as the preferred bet
*wrapped in dated gates + Plan B* (§5.5, §10), Ignis as operator/product layer
(§9). See §11 for the verification checklist.

---

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

> **v2 addition — the throughput claim is now conditional and must be proven
> (resolves B1, B2).** "High aggregate throughput" holds *only at sufficient
> concurrency to fill the pipeline*. Below that concurrency the system degrades
> toward the single-stream latency floor above. The break-even concurrency for a
> concrete `(model, fleet)` is no longer asserted — it is a required output of
> `v0-target-contract.md` (§3.1) and gates Phase 1 (§6).

## 3. Scope, target operator, and the v0 contract

### 3.1 The v0 target contract gates everything <a id="v0-contract"></a>

`(resolves B1)` Before any Phase-1 engineering, the load-bearing claim must be
made falsifiable in **one** document, `v0-target-contract.md`, fixing:

- **One reference model** — family, total / active params, expert count, top-k,
  quantization, context length.
- **One exact fleet** — see §4 named bundles, down to SKUs, memory, NIC/fabric.
- **Memory budget per node** — weights, resident experts, KV (at target
  concurrency × context), activations, routing metadata, temporary buffers,
  runtime/OS reserve, fragmentation margin. Proves the model *exceeds one node
  yet fits the cluster*.
- **Throughput estimate** — per-stage times, bubble fraction, exposed vs
  overlapped transfer, remote-expert exposure, TTFT, decode tok/s — computed via
  the partitioner cost model ([partitioner-spec.md](partitioner-spec.md)).
- **Baselines to beat** — single-node (where it fits a quant), naive sequential
  pipeline, expert-only offload.
- **Pass/fail thresholds** — the numbers in §8, bound to this target.

### 3.2 Target operator & concurrency (resolves B2)

The product thesis is contingent on the buyer generating enough concurrency to
fill a heterogeneous pipeline. v2 makes that explicit rather than assumed:

- **Primary persona (v0):** a *small team / firm shared private-AI service* —
  several-to-many concurrent agentic requests — **not** a single bursty user. A
  single-user deployment is explicitly a poor fit for spanning (§2) and is
  redirected to single-node serving (§3.3 non-goals).
- **Required evidence:** a concurrency-sensitivity sweep in
  `v0-target-contract.md` stating the **minimum concurrency** that keeps the
  pipeline efficient on the target fleet, and confirming the persona can supply
  it. If it cannot, the thesis narrows (capacity-only, or a different fleet
  shape) before Phase 1 — a deliberate go/no-go, not a silent assumption.

### 3.3 Goals / Non-goals

**Goals**
- Serve one frontier MoE model that exceeds the largest single node, across a
  heterogeneous LAN, at high aggregate throughput **at the contracted
  concurrency (§3.2)**.
- Absorb whatever hardware is present (mixed NVIDIA/AMD/Apple, mixed memory,
  mixed links) — "maximal utility from heterogeneous commodity hardware."
- Single OpenAI-compatible endpoint; the cluster looks like one machine to
  clients.
- Honest, cluster-wide telemetry (real cache/util, no fabricated metrics — the
  Ignis discipline).
- Elasticity: tolerate flaky consumer nodes joining/leaving/dying mid-request.

**Non-goals (v1 of the product)**
- Competing with vLLM/SGLang on datacenter throughput or single-stream latency.
- Unbounded cross-vendor MoE **expert-parallel all-to-all** as the default data
  path. Bounded remote expert execution is in-scope only when the measured fabric
  and batching model keep it throughput-neutral.
- **Single-user / low-concurrency deployments** as a primary target — spanning
  cannot serve them well (§2); route to single-node `max serve` / MLX /
  llama.cpp.
- Training / fine-tuning.
- WAN / internet-scale federation (Petals-style). LAN only.
- Serving models that comfortably fit one node.

## 4. Target deployment profile

`(resolves B1 hardware vagueness, P1 hardware matrix)` v1 said "RTX 40/50 series,
Radeon/MI where MAX support exists" — too vague to review or purchase against. v2
replaces this with **three named bundles**; exact SKUs, memory, and measured link
data live in `v0-target-contract.md` and machine-readable inventory files that
feed the partitioner.

| Bundle | Shape | Role |
|---|---|---|
| `desktop-minimal` | 1 multi-GPU box + 1 Apple Silicon node | smoke tests, lifecycle, correctness |
| `prosumer-rack` | few consumer-GPU boxes + 1–2 high-unified-memory Macs, 25–100 GbE / TB | **the v0 target** |
| `lab-reference` | the team's controlled heterogeneous lab | calibration + benchmark of record |

- **Hot accelerator tier:** consumer NVIDIA/AMD — attention, prefill/decode,
  routers, shared experts, hot routed experts, KV cache. (Exact AMD support is a
  dated capability gate in the substrate ADR, §10 — not a blanket claim.)
- **Capacity/expert tier:** Apple Silicon Macs (M3+, high unified memory) —
  cold/warm experts, expert weight reservoir, pipeline stages *once measured
  performance justifies them* (the demotion gate, §5.10).
- **Fabric:** designed as part of the engine. Named tiers — 10 / 25 / 40 / 100
  GbE, direct Thunderbolt, RDMA-where-available — with measured latency/bandwidth
  per link and a stated minimum tier per workload. WiFi / 1 GbE are excluded from
  the hot path.
- **Concurrency:** sized to the contracted persona (§3.2), not assumed.
- **Negative hardware list:** configurations possible but explicitly unsupported
  in v0 (recorded in the contract) so triage stays bounded.

## 5. Architecture

### 5.1 Parallelism decision (settled, with a MoE escape hatch) — *preserved*

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

### 5.2 The six throughput levers — *preserved*

1. **Continuous batching to fill the pipeline.** Converts PP from a latency tax
   into a throughput-neutral layout. The single biggest win and exactly what
   Exo/llama.cpp-RPC lack.
2. **Comms/compute overlap (1F1B / interleaved).** Transfer microbatch *n*'s
   activation while computing *n+1*; hides the network.
3. **Speed-balanced heterogeneous stage sizing — the crux.** A pipeline runs at
   its slowest stage; nodes differ 5–10×. Partition layers to equalize per-stage
   wall-time, bounded by each node's memory. Core IP →
   [partitioner-spec.md](partitioner-spec.md).
4. **Data-parallel replication of bottleneck stages.** Replicate the limiting
   stage onto spare nodes, round-robin microbatches → a 2D (depth × width) grid.
5. **Expert locality + migration.** Treat remote expert hits as a budgeted
   resource. Keep shared/hot experts local, execute cold/warm experts remotely
   only when batched/overlapped, migrate experts when decode traces show
   sustained hotness.
6. **Quantize weights + activations (4-bit / fp8).** Fits the model, halves
   transfer. Mandatory (671B fp16 = 1.3 TB → 4-bit ≈ 400 GB).

### 5.3 Layered components (OS analogy) — *preserved*

| Layer | Role | Provided by | Fornax builds |
|---|---|---|---|
| Machine model | Live inventory + measured link-bandwidth topology graph | — | ✅ |
| Per-node stage runtime | Execute a contiguous layer group as a MAX graph or graphlet; activation/KV handoff at boundaries | **MAX graph compiler + kernel library** plus model-specific Mojo kernels where needed | stage host, boundary custom ops, heterogeneous dispatch |
| Control plane | Partition planner, continuous-batching scheduler, placement, admission, cache-affinity routing | — | ✅ the engine's scheduler |
| Data plane | Activation transport (pluggable TCP / RDMA / Thunderbolt-IP / shm), topology-aware boundary placement | MAX custom op + NIXL/UCX intra-NVIDIA | ✅ cross-vendor |
| State | Distributed KV/prefix registry; elasticity + replay on node loss | — | ✅ (Ignis single-node seed) |
| Client surface | One OpenAI-compatible endpoint | — | thin |

### 5.4 Substrate: a surgery of MAX (not a harness over llama.cpp) — *preserved*

Fornax is an engine **assembled from MAX's components**, because MAX's
portable-by-design kernels (one Mojo source → NVIDIA/AMD/Apple) are *with the
grain* of a heterogeneous engine — more so than llama.cpp's separate CUDA/Metal
backends. We own the KV/activation format end-to-end, so cross-stage and
cross-vendor handoff is consistent by construction.

**Reuse from MAX:** graph compiler + kernel library; model architecture
definitions / weight loading (`max.pipelines`) where reusable; KV-cache
primitives; the custom-op API (`@compiler.register` / `ops.custom`) for the
pipeline-stage boundary, cross-device handoff, and expert dispatch.

**Surgery / build:** heterogeneous distributed pipeline execution (partitioning,
transport, 1F1B, topology-aware continuous batching); a first-class MoE expert
runtime (placement, packing, local/remote dispatch, weighted gather, migration,
profiling); one KV/activation/expert-batch format owned across vendors (now
specified — §5.6); Apple/Mac runtime readiness (§5.5).

> **The substrate choice is a recorded decision, not an assumption (resolves
> B5).** Rationale, rejected alternatives (llama.cpp / MLX / vLLM / SGLang /
> hybrid), dated capability assumptions, and a reversal trigger live in
> `adr/0001-max-mojo-substrate.md` (§10).

### 5.5 Critical path: Apple/Mac runtime readiness + Plan B (resolves B5)

The Apple story is no longer "no LLM path exists." Modular 25.6 introduced Apple
Silicon GPU support for Mojo GPU programming; the 26.4 announcement says MAX
supports many common architectures on M3+ Apple GPUs. The docs still caution that
general large-model GenAI inference on Apple Silicon is not uniformly available
from the package surface — so Fornax treats Apple as **fast-moving,
capability-probed, model-specific**, not a blanket assumption.

The work is therefore "make the target MoE family work on the Apple *roles we
assign it*," starting with the **expert worker** (smallest op surface):

```text
input:  [tokens_for_expert, hidden_dim]
output: [tokens_for_expert, hidden_dim]
ops:    dequant/load weights, matmul, activation, matmul, optional normalization
```

Use MAX kernels/components where they exist; write a narrow Mojo kernel only when
a target op is missing or too slow, and delete it when MAX catches up. Do not
fork MAX wholesale. Keep surgery at explicit seams: activation transport, expert
dispatch, expert MLP kernels, KV/page handoff, stage scheduling.

**Plan B and reversal trigger (new in v2):**
- **Apple participation is staged and gated.** Roles in ascending risk: (1)
  cold-expert *capacity/store*, (2) *expert worker*, (3) KV-heavy *decode stage*,
  (4) arbitrary *pipeline stage*. v0 commits only to the lowest role that the
  measured fleet supports; higher roles unlock by passing the §5.10 gates.
- **Reversal trigger:** if, by the end of the Phase-0 sprint, MAX cannot run the
  target model's expert MLP on the target Mac within a stated tolerance/throughput
  bound, Fornax v0 **demotes Apple to capacity-only** (no hot compute) and the
  thesis is re-scoped — recorded against the contract, not discovered late.
- **Dated tracking:** Modular/MAX version + capability assumptions are pinned and
  dated in the ADR; re-checked each nightly (R4).

**Upstream anchors to track:**
- Modular 25.6 Apple GPU direction: https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple
- Modular 26.4 MoE + Apple note: https://www.modular.com/blog/modular-26-4-sota-moe-serving-model-bringup-via-agent-skills-mojo-beta-2-and-more
- MAX package/platform caveats: https://docs.modular.com/max/packages/
- MAX custom ops: https://docs.modular.com/max/develop/build-custom-ops/

### 5.6 Cross-vendor runtime format & invariants (resolves B3)

The shared format for **KV pages, activations, and expert batches** across Apple
/ NVIDIA / AMD / transport / MAX-graph boundaries is the engine's load-bearing
invariant. It is specified in `runtime-format-and-invariants.md` and summarized
here:

- **Activation tensor:** layout, dtype, shape, strides, alignment, padding,
  device/host residency, and **buffer ownership/lifetime** across send/receive.
- **KV page:** layout, page size, dtype, residency, ownership, eviction, and
  cross-node transfer rules.
- **Expert batch:** token indices, expert IDs, top-k weights, packed hidden
  states, output gather format, routing metadata.
- **Quantization:** the on-disk and in-flight format, and compatibility
  expectations across MAX backends (an expert must be byte-compatible wherever it
  executes).
- **Correctness:** a slow-but-obvious **reference path** using the same placement
  plan, plus golden vectors and a per-dtype tolerance methodology — the contract
  that proves an optimized cross-vendor seam matches (also §8 Correctness).

### 5.7 End-to-end request lifecycle & state ownership (resolves P0 lifecycle)

A single request, named owner at each hop:

```text
client → Ignis (timeline/policy) → FornaxBackend (Engine seam)
  → tokenize + chat template (owner: Fornax serving layer, reuses Ignis tokenizer)
  → admission + continuous-batch scheduler (owner: control plane)
  → prefill across stages (owner: stage workers; KV owner: per-stage KV manager)
  → decode loop: stage compute + local/remote expert dispatch + weighted gather
  → sampler (owner: last stage) → stream tokens back (owner: serving layer)
  → cancellation / failure / cleanup (owner: scheduler + state registry)
```

| State | Owner | Lifetime |
|---|---|---|
| Placement plan | control plane | per serving session |
| Microbatch / request slot | scheduler | per request |
| KV pages | per-stage KV manager | per request, evictable |
| Expert routes / batches | MoE runtime | per token |
| Worker health / membership | inventory service | continuous |
| Client stream | serving layer | per request |

The **tokenizer/chat-template/stop-token seam** (who owns BOS/EOS, stop strings,
streaming chunk boundaries now that Fornax owns execution) is specified here and
in the model support matrix (§6 exit of Phase 2.5) — reusing Ignis's tokenizer
but with the boundary written down, not implied.

### 5.8 Networking, security & backpressure (resolves B4)

Detailed in `networking-security-and-backpressure.md`; the plan-level stance:

- **Trust boundary (the privacy promise).** A multi-node fabric carries user
  activations and KV — i.e. user data. v0 minimum: **node identity + admission,
  endpoint authentication, plan-integrity tags** (a worker knows which plan it
  runs), and a stated decision on inter-node encryption per deployment class.
  "Trusted LAN" is acceptable **only** for the lab bundle and must be declared.
- **Backpressure** across admission → scheduler queues → stage workers → expert
  workers → network buffers → client streaming, with bounded memory (no unbounded
  queue growth).
- **Failure semantics:** timeout, retry, cancellation, slow-worker, lost-worker,
  network-partition — defined for in-flight token generation, backed by Phase-4
  replay.

### 5.9 Observability (resolves P1 observability)

Designed in from Phase 1, not deferred to productization: request IDs + plan IDs
threaded through every log/metric; per-stage timings; router + remote-expert
traces; queue depth + backpressure events; memory/KV metrics; and **placement
explanations** ("why this node was excluded / why throughput is low"). Honest
metrics only (Ignis invariant), per-node and aggregate.

### 5.10 Cost-model calibration & backend coverage (resolves P1)

The partitioner's predictions are only trustworthy if calibrated:

- **Backend operation coverage matrix** (Apple / NVIDIA / AMD): attention, dense
  MLP, router, expert GEMM, collect/scatter, KV ops, sampling, serialization,
  transfer — each marked MAX-kernel / custom / fallback / unsupported.
- **Calibration plan:** measure `compute_class`, memory bandwidth, pack/gather,
  serialization, and per-link bandwidth/latency on the `lab-reference` bundle;
  feed measured numbers back into the cost model (the partitioner consumes
  inventory probes, [partitioner-spec.md](partitioner-spec.md) §1.2).
- **Apple demotion gate:** a profiler threshold that decides whether a Mac
  participates as compute, expert host, KV capacity, or store-only — the
  mechanism behind §5.5's staged Apple roles.

## 6. Roadmap (phased)

`(resolves B6)` v1 wrongly claimed "Phases 0–2 need no GPU." Corrected: **only
Phase 0 is hardware-free.** Phases 1–2 are split into a model-free simulation
tier (CI) and a hardware-backed validation tier (lab). **Phase 1 is gated on the
Priority-0 artifacts (§10) passing review.**

**Test tiers (resolves P1 CI/lab):** (T0) planner/scheduler unit tests + golden
plans, CI, no hardware; (T1) simulated workers exercising scheduling, transport
contracts, backpressure, deterministically; (T2) single-node accelerator; (T3)
2–3 node hardware; (T4) full heterogeneous lab.

- **Phase 0 — Evidence sprint + planner (hardware-free, T0/T1).** The
  partitioner + cost model, *and* the five gating artifacts (§10). *Exit:* a
  reviewed `v0-target-contract.md` whose predicted budget+throughput close, plus
  golden-plan tests green.
- **Phase 1 — Worker contract + transport (T1 then T3).** `FornaxEngine` +
  `StageWorker` RPC + activation transport. **1a (T1):** contract tests against
  simulated workers — no GPU. **1b (T3):** small model pipeline-parallel across
  2–3 like nodes. *Exit:* correct generation across a real pipeline; activation
  transfer measured and overlapped; matches the format spec (§5.6).
- **Phase 2 — Continuous batching (T1 then T3).** **2a (T1):** scheduler + 1F1B
  in simulation. **2b (T3):** hardware continuous batching. *Exit:* aggregate
  tok/s scales with concurrency to saturation; matches planner prediction within
  the §8 provisional bound.
- **Phase 2.5 — MoE expert-runtime surgery (T2/T3).** Explicit expert bucketing,
  local/remote dispatch, weighted gather, expert tracing; **the model support
  matrix** (architecture, tokenizer, chat template, quant, MoE routing, stop,
  streaming, tool/structured-output). *Exit:* layer/logit match vs the reference
  path (§5.6); remote expert batches measured independently.
- **Phase 3 — Heterogeneous + frontier MoE (T4).** Span a real frontier MoE
  across NVIDIA/AMD + Mac with speed-balanced sizing, hot experts on the hot
  tier, bounded remote experts where profitable, Apple at its gated role (§5.5).
  *Exit:* end-to-end serve at predicted throughput on real heterogeneous
  hardware; security/backpressure (§5.8) active.
- **Phase 4 — Stage replication + elasticity (T4).** DP replicas; node-loss →
  reschedule + replay in-flight microbatches. *Exit:* throughput scales with
  added nodes; a killed node drops no in-flight requests.
- **Phase 5 — Productization.** Operator UX, ops lifecycle, onboarding (§10
  Priority 2). *Exit:* a firm can stand it up and serve internal users.

## 7. Risks & open questions

- **R1 — Throughput ceiling is real.** Commodity net + flaky nodes cap aggregate
  throughput well below datacenter. Mitigation: levers 1–6; honest positioning.
- **R2 — Pipeline depth vs latency.** Deep pipelines tank per-request latency.
  Mitigation: planner penalizes depth; prefer fewer big-memory nodes when
  latency-sensitive.
- **R3 — Heterogeneous numerics.** NVIDIA bf16/fp8 vs Apple fp16/fp19-truncated
  matmul → seams can diverge. Mitigation: pin dtypes across seams; reference path
  + golden vectors + per-dtype tolerances (§5.6).
- **R4 — Apple/Mac runtime readiness (CRITICAL PATH).** Mitigation: capability-
  probe every nightly; scope to one model family; expert-worker first; narrow
  custom kernels deleted as MAX ships equivalents; **staged roles + reversal
  trigger** (§5.5).
- **R5 — Remote expert bandwidth and latency.** Preserves throughput only when
  hits are sparse/batched/overlapped/migrated. Open: crossover batch vs link
  bandwidth and decode hit-rate (planner must model; calibrate §5.10).
- **R6 — Surgery against fast-moving MAX internals.** Mitigation: pin a verified
  build; isolate surgery behind a thin internal interface; track Apple progress
  to retire custom code; ADR dates the assumptions (§10).
- **R7 — Mojo toolchain maturity** for a full engine + GPU kernels on a beta
  toolchain. Mitigation: lean on MAX kernels; keep custom kernels minimal/tested.
- **R8 — Concurrency–market fit (new, resolves B2).** The persona may not supply
  the concurrency the pipeline needs, collapsing the thesis to the latency floor.
  Mitigation: §3.2 contract + sweep; narrow scope at the Phase-0 go/no-go if it
  fails.
- **R9 — Security/trust on a consumer fabric (new, resolves B4).** User
  data crosses nodes. Mitigation: §5.8 v0 stance; "trusted LAN" allowed only for
  the lab bundle and must be declared; blocker before any product deployment.

## 8. Success metrics

Targets are **provisional** here and become **binding** in
`v0-target-contract.md` (§3.1). Provisional values are the starting bar, tightened
by Phase 3.

- **Capacity:** serve a model **N× the largest single node**, target N ≥ 2–3.
- **Throughput efficiency:** aggregate tok/s ≥ **Y%** of the sum-of-nodes ideal at
  saturation; provisional **Y ≥ 60%** (bound in contract).
- **Planner accuracy:** predicted vs measured throughput within **±X%**;
  provisional **±20%**, tightening to **±10%** by Phase 3.
- **Concurrency efficiency (new):** pipeline saturates at ≤ the contracted
  minimum concurrency (§3.2).
- **Expert locality:** decode-time remote-expert wait below the configured SLO;
  hot experts migrate local automatically.
- **Correctness:** layer/logit divergence vs the reference path within per-dtype
  tolerances at every pipeline and expert seam (§5.6).
- **Cost:** $/token and $/capacity vs an 8×H100 baseline (the headline pitch).
- **Elasticity:** zero dropped in-flight requests on single-node loss.
- **Honesty:** every reported metric traceable to a real measurement (Ignis
  invariant, cluster-wide).

## 9. Relationship to Ignis — *preserved*

**Ignis is the harness; Fornax is the engine.** Complementary layers:

- **Fornax** owns model execution — the distributed heterogeneous forward pass,
  kernels, KV. It is the MAX surgery.
- **Ignis** owns the agent loop — timeline, policy/exact-confirm gate, tools,
  replay, telemetry.

Fornax plugs into Ignis behind the **`Engine` trait** as a `FornaxBackend`
(joining/replacing `MaxBackend`) — `generate(messages_json, tools_json,
max_new_tokens) -> EngineResult` is the seam. Disciplines carried across: honest
KV/cache telemetry → cluster-wide utilization; **replay bytes** → reschedule
dropped microbatches on node loss; the **model-free fixture discipline** → the
planner/scheduler (Phase 0, T0/T1) are testable with no hardware. Fornax can also
be driven standalone (OpenAI-compatible endpoint) for non-Ignis harnesses.

## 10. Decisions & companion artifacts

The reconciled review's go/no-go requires five Phase-0 outputs before Phase-1
engineering. They are referenced throughout this plan and **owned by the Phase-0
sprint** (not yet written):

| Artifact | Resolves | Status |
|---|---|---|
| `v0-target-contract.md` | B1, B2, P1 hardware | to write |
| `runtime-format-and-invariants.md` | B3 | to write |
| `networking-security-and-backpressure.md` | B4 | to write |
| `adr/0001-max-mojo-substrate.md` | B5 | to write |
| Roadmap correction (this doc §6) | B6 | **done in v2** |

**Priority-2 (before productization), scheduled not yet detailed:** operator
quickstart + `cluster.yaml` / `model.yaml` / generated `placement.json` +
`fornax doctor`; deploy / upgrade / drain / restart / rollback / node-replacement
procedures; contributor onboarding tracks (operator / runtime / kernel); further
ADRs (pipeline default, bounded remote experts, transport choice, security
posture, Apple participation, rejected alternatives); glossary; reproducible
benchmark methodology.

## 11. What v2 deliberately preserves (verification checklist)

Per the reconciled review's "What Should Not Change," reviewers can confirm each
survived intact:

- ✅ Engine-not-harness stance — header, §9.
- ✅ Pipeline-parallel spine — §5.1 (verbatim).
- ✅ Bounded remote experts as measured optionality, not default all-to-all —
  §5.1, Non-goals §3.3.
- ✅ Planner-first sequencing — §6 Phase 0.
- ✅ Honest single-stream latency caveat — §2 (verbatim), reinforced not weakened.
- ✅ MAX/Mojo as preferred bet, now wrapped in dated gates + Plan B — §5.5, §10.
- ✅ Ignis as operator/product layer — §9 (verbatim).

The v2 changes are **additive and corrective** (new specs, gates, and one
contradiction fix); none reverse a preserved decision.
