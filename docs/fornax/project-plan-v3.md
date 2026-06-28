# Fornax - Project Plan (v3)

> **Version 3.** Supersedes [project-plan-v2.md](project-plan-v2.md), which is
> preserved unchanged with its review output
> ([fornax_plan_v2_review_codex.md](fornax_plan_v2_review_codex.md)).
>
> V3 incorporates the v2 review. Its core correction is status honesty: V3 does
> **not** say the old blockers are solved merely because they are named. It says
> which blockers are **gated into Phase 0**, who owns them, what artifact must
> close them, and what evidence is required before Phase 1 can start.
>
> **Authorization status:** approved only as a Phase-0 evidence-sprint plan. It is
> **not** authorization for Phase-1 distributed runtime engineering. Phase 1 is
> blocked until G1 passes (§6, §10).

Authoritative inputs:

- [fornax_plan_review_reconciled.md](fornax_plan_review_reconciled.md)
- [fornax_plan_v2_review_codex.md](fornax_plan_v2_review_codex.md)
- Program-management layer:
  - [01-stakeholders-and-raci.md](program_management/01-stakeholders-and-raci.md)
  - [phase-0-evidence-sprint.md](program_management/sprints/phase-0-evidence-sprint.md)
  - [04-stage-gates.md](program_management/04-stage-gates.md)
  - [06-dependencies-and-external-watch.md](program_management/06-dependencies-and-external-watch.md)
  - [07-resourcing-and-skills.md](program_management/07-resourcing-and-skills.md)

## Changes in v3 - review to plan map

| Review item | Severity | Addressed / gated in v3 | Evidence status |
|---|---|---|---|
| B1: no quantitative feasibility proof | Blocker | §3.1, §3.2, §4, §8, §10 | open until `v0-target-contract.md` passes G1 review |
| B2: concurrency-market fit unproven | Blocker | §3.2, §3.3, §7 R8, §8 | open until concurrency sweep closes in target contract |
| B3: cross-vendor format undefined | Blocker | §5.6, §10 | open until `runtime-format-and-invariants.md` passes review |
| B4: security / trust / backpressure missing | Blocker for multi-node product deployment | §5.8, §6, §10 | open until `networking-security-and-backpressure.md` passes review; implementation gated by phase |
| B5: Apple/MAX bet lacks Plan B | Blocker if Apple is critical path | §5.4, §5.5, §5.10, §10 | open until substrate ADR and Apple expert-MLP probe pass G1 |
| B6: Phase 0-2 hardware contradiction | High | §6 | closed in v2/v3: only Phase 0 is hardware-free |
| V2 review: change map overstated resolution | High | this table + §10 | closed in v3 wording |
| V2 review: no owners/checklists | High | §0, §10 | gated by role ownership and G1 acceptance criteria |
| V2 review: no seed target | Medium | §3.2 | seed added; not binding until target contract |
| V2 review: operator UX deferred too late | Medium | §3.4, §6 | minimal preflight workflow moved to Phase 0/1 |
| V2 review: Apple source-of-truth fragile | High | §5.4, §5.5, §10 | source precedence defined; local probe wins gating |
| V2 review: module/test ownership thin | High | §5.3, §6, §10 | provisional module map + T0/T1 command contract added |

## Phase-0 gate status

The old blockers remain open until these artifacts are written and reviewed. Role
codes are defined in [01-stakeholders-and-raci.md](program_management/01-stakeholders-and-raci.md).
If a required role is unstaffed at G1, the Sponsor must choose ITERATE, NARROW,
or KILL rather than silently proceed.

| Artifact / evidence | Responsible | Accountable | Primary review lenses | G1 acceptance bar | Status |
|---|---|---|---|---|---|
| `v0-target-contract.md` | DIST + PM | TL, SP for gate | Hardware, Analytical, Hardware Accel, Software Eng | exact model/fleet/context/concurrency; memory closes with headroom; predicted throughput meets bar; baselines and kill metric stated | to write |
| concurrency sweep inside target contract | DIST | TL | Analytical, High-level, LLM | minimum saturation concurrency stated; target persona can supply it or scope narrows | to write |
| `runtime-format-and-invariants.md` | RT | TL | Low-level, LLM, System | activation/KV/expert-batch layouts; ownership; failure modes; golden vectors; reference path; build/toolchain notes | to write |
| `networking-security-and-backpressure.md` | NET | TL, SP for security posture | Networking, System, Software Eng | trust boundary; node identity; endpoint auth; plan-integrity tags; encryption decision; backpressure; timeout/retry/cancel/partition; lab exception rules | to write |
| `adr/0001-max-mojo-substrate.md` | TL | SP | Organizational, Hardware Accel, Low-level | MAX/Mojo rationale; rejected alternatives; source precedence; pinned build policy; Apple Plan B; reversal trigger | to write |
| Apple expert-MLP probe | KER | TL | Hardware Accel, Hardware, LLM | target model expert MLP on target Mac either passes tolerance/throughput or Apple demotes to capacity-only | to run |
| Phase-0 preflight workflow | DIST + SRE | PM/TL | High-level, System, Documentation | inventory, fabric probe, target validation, planning, simulation/benchmark, and diagnostics can be run without oral context | to write |
| Roadmap correction | TL | SP | Software Eng, Documentation | only Phase 0 is hardware-free; T0/T1/T2/T3/T4 are explicit | done in v2/v3 |

## 1. Vision and Problem

Firms and individuals increasingly want frontier-level capability in-house: for
privacy, control, cost, and local operational independence. The frontier of open
weights is increasingly large sparse MoE: DeepSeek-R1-class 671B/37B-active,
Kimi-class trillion-parameter MoE, Qwen3-235B-A22B-class models, and Llama 4 MoE
families. These models exceed a single consumer node, while full datacenter GPU
clusters defeat the cost and ownership goals.

Three facts make a heterogeneous commodity cluster the right shape to test:

1. **MoE splits capacity from compute.** Holding all experts is capacity-heavy;
   per-token work touches only a sparse active set plus dense attention. Apple
   unified memory is attractive for capacity and some expert roles; consumer
   NVIDIA/AMD accelerators are attractive for hot compute.
2. **MoE exposes a surgical seam.** Dense attention and hot paths can remain on
   fast accelerators while routed expert MLPs can be placed, cached, migrated, or
   remotely executed when measured economics allow it.
3. **Spanning communication can be engineered only when measured.** Pipeline
   boundaries and bounded remote expert calls can be overlapped and batched on a
   provisioned LAN. Cross-vendor tensor-parallel all-reduce remains the pattern
   to avoid outside homogeneous high-bandwidth islands.

The product gap remains: existing local/distributed experiments do not make one
heterogeneous private cluster behave like a high-throughput inference engine for
a model larger than any single node. Fornax is that engine, built from MAX/Mojo
parts and custom distributed runtime pieces.

## 2. The Constraint

When the model exceeds the biggest node, every token crosses a machine or device
boundary. A network sized for this workload changes the throughput envelope; it
does not erase the synchronization floor.

Therefore:

- **Do not promise single-request latency parity** with a single big node once a
  request spans multiple devices.
- **Do optimize aggregate throughput and utilization** at the contracted
  concurrency by balancing stages, overlapping communication, replicating
  bottlenecks, and keeping remote expert traffic rare, batched, overlapped, or
  migrated local.

The throughput claim is conditional. If the buyer cannot supply enough in-flight
requests to fill the pipeline, the system degrades toward the single-stream
latency floor. That condition is a G1 gate, not marketing copy.

## 3. Scope, Operator, and v0 Target

### 3.1 The v0 target contract gates everything

Before Phase 1, `v0-target-contract.md` must make the load-bearing claim
falsifiable for one concrete target:

- one reference model: family, total/active params, expert count, top-k,
  quantization, context length, tokenizer/template source;
- one exact fleet: SKUs, memory, OS/runtime, NIC/fabric, switch, topology;
- per-node memory budget: weights, resident experts, KV at target concurrency and
  context, activations, routing metadata, temporary buffers, runtime/OS reserve,
  fragmentation margin;
- throughput estimate: per-stage times, exposed vs overlapped transfer, bubble
  fraction, remote expert exposure, TTFT, decode tok/s;
- baselines: single-node where it fits a quant, naive sequential pipeline,
  expert-only offload, and existing engine where practical;
- kill metric: the single measurement or budget failure that invalidates the v0
  target;
- pass/fail thresholds bound to §8.

### 3.2 Seed target candidate, not yet the proof

The v2 review correctly noted that Phase 0 could become open-ended without a
seed. V3 adds this drafting seed so the target contract starts bounded. It is not
accepted until the contract closes.

| Field | Seed candidate |
|---|---|
| Model | Qwen3-235B-A22B-class open MoE, or nearest MAX-bringup-feasible MoE with published tokenizer, routing, and template metadata |
| Stretch candidate | DeepSeek-R1 671B/37B-active-class model only if memory and throughput close on the lab fleet |
| Quantization | 4-bit weights as first memory target; activation dtype and expert in-flight format decided in runtime-format spec |
| Context | 32k first target; 64k sensitivity sweep |
| Concurrency sweep | 4, 8, 16, 32, 64 in-flight requests; seed goal is saturation at or below 32 for the small-team persona |
| Fleet | `prosumer-rack`: one Linux high-VRAM two-GPU box plus one high-unified-memory Apple Silicon Mac, with optional second Mac; exact SKUs in target contract |
| Fabric | 100 GbE preferred for v0 proof; 25 GbE allowed only if the model/fleet budget still closes; Thunderbolt direct link may be used for `desktop-minimal` smoke tests |
| First failure to test | remote expert wait and pipeline bubble fraction under the seed concurrency sweep |

The contract may replace this seed, but only by recording why the replacement is
more useful or more feasible.

### 3.3 Target operator and concurrency

Primary v0 persona: a small team or firm running a shared private-AI service with
several-to-many concurrent agentic requests. A single bursty user is explicitly
not the primary spanning target; they should use single-node `max serve`, MLX, or
llama.cpp unless the model requires spanning and they accept the latency floor.

Required G1 evidence:

- estimate the persona's realistic concurrency range;
- measure or simulate saturation concurrency for the target fleet;
- prove the persona can supply enough load, or narrow the product thesis to
  capacity-only, homogeneous-island, or a different target fleet.

### 3.4 Minimal Phase-0 / Phase-1 operator workflow

Product UX remains Phase 5, but preflight UX moves earlier. Phase 0 must produce
a minimal operator workflow, even if the first implementation is scripts or
internal CLIs:

```text
fornax inventory collect --out inventory.json
fornax fabric probe --inventory inventory.json --out links.json
fornax target validate docs/fornax/v0-target-contract.md --inventory inventory.json --links links.json
fornax plan --target docs/fornax/v0-target-contract.md --inventory inventory.json --links links.json --out placement.json
fornax simulate --plan placement.json --trace synthetic-or-model-trace.json
fornax benchmark --plan placement.json --mode tiny-moe-or-expert-mlp
fornax doctor --bundle phase0-evidence-bundle/
```

Names may change. The workflow requirements may not: inventory, fabric probe,
target validation, placement, simulation, benchmark/probe, and diagnostics must
be runnable without oral context before Phase 1b hardware validation.

### 3.5 Goals and non-goals

Goals:

- Serve one frontier MoE model that exceeds the largest single node across a
  heterogeneous LAN at high aggregate throughput at the contracted concurrency.
- Extract maximal utility from mixed consumer/prosumer NVIDIA, AMD, and Apple
  hardware without pretending every device is equally useful.
- Present one OpenAI-compatible endpoint and one Ignis `Engine` backend.
- Preserve honest cluster-wide telemetry.
- Eventually support elasticity and node loss, but treat this as Phase 4 product
  capability rather than v0 proof.

Non-goals for v0 product:

- Datacenter parity with vLLM/SGLang on homogeneous H100/B200 fleets.
- Single-stream latency parity with a single big node.
- WAN federation.
- Training or fine-tuning.
- Serving models that comfortably fit one node.
- Speculative decoding unless the target contract explicitly opts it in; default
  v0 stance is out of scope.

## 4. Target Deployment Profile

The hardware plan is now a gate structure, not a feasibility proof. Exact SKUs
and measured links live in `v0-target-contract.md` and machine-readable inventory
files.

| Bundle | Shape | Role |
|---|---|---|
| `desktop-minimal` | one multi-GPU box plus one Apple Silicon node | smoke tests, lifecycle, correctness, Apple expert-MLP probe |
| `prosumer-rack` | few consumer/prosumer GPU boxes plus one or two high-unified-memory Macs, provisioned 25-100 GbE or direct links | v0 target shape |
| `lab-reference` | controlled heterogeneous lab with reproducible hardware and thermal conditions | calibration and benchmark of record |

Hardware acceptance additions:

- sustained thermal/performance run for each hot accelerator and Apple role;
- measured local attention/GEMM/expert-MLP throughput;
- measured host-device and inter-node transfer bandwidth/latency;
- measured pack/gather/serialization overhead;
- explicit unsupported list: WiFi, 1 GbE hot path, unmeasured nightlies, devices
  whose memory cannot meet the target with margin.

## 5. Architecture

### 5.1 Parallelism decision

| Strategy | Per-token communication | v3 stance |
|---|---|---|
| Tensor parallel across mismatched nodes | all-reduce per layer | avoid outside homogeneous high-bandwidth islands |
| Pipeline parallel by layer group | activation tensor per stage | default spanning spine |
| Bounded remote expert execution | selected hidden-state batches | optional when measured fabric and batching make it cheap |
| Unbounded expert all-to-all | data-dependent all-to-all per MoE layer | not v0 default |

The robust default is speed-balanced pipeline parallelism. Each stage holds its
layers' full active expert set when memory allows, collapsing communication to
stage-boundary activations. Remote expert execution is a measured optimization,
not a baseline assumption.

### 5.2 Throughput levers

1. Continuous batching to fill the pipeline.
2. 1F1B or interleaved communication/compute overlap.
3. Speed-balanced heterogeneous stage sizing.
4. Replication of bottleneck stages.
5. Expert locality, caching, and migration.
6. Quantized weights and, where safe, activation/expert-batch transfer formats.

### 5.3 Layered components and provisional module map

| Layer | Role | Provided by MAX / ecosystem | Fornax owns |
|---|---|---|---|
| Machine model | inventory, measured links, device capabilities | OS/runtime probes | inventory schema and planner input |
| Per-node runtime | execute layer group as graph/graphlet | MAX graph compiler, kernels, custom-op API | stage host, boundary ops, MAX integration |
| Control plane | planning, admission, batching, placement | - | planner and scheduler |
| Data plane | activation/expert/KV payload movement | possible UCX/NIXL/intra-vendor pieces | cross-vendor transport contract |
| State | KV/prefix registry, worker health, replay metadata | Ignis concepts as seed | distributed state manager |
| Client surface | OpenAI-compatible endpoint and Ignis backend | OpenAI API conventions | serving adapter and `FornaxBackend` |

Provisional implementation boundaries, subject to Phase-0 repository decision:

| Boundary | Responsibility |
|---|---|
| `fornax/planner` | model/hardware target, placement search, cost model |
| `fornax/inventory` | hardware and fabric probes, inventory JSON schemas |
| `fornax/runtime_format` | activation/KV/expert-batch schemas, golden-vector fixtures |
| `fornax/scheduler` | admission, continuous batching, 1F1B simulation |
| `fornax/workers` | stage/expert worker contracts and simulated workers |
| `fornax/transport` | transport interface, local/TCP first implementation, later RDMA/TB variants |
| `fornax/max_integration` | MAX graph/session/custom-op integration |
| `fornax/moe` | expert bucketing, local/remote dispatch, weighted gather |
| `fornax/serving` | OpenAI-compatible endpoint, Ignis `Engine` adapter |
| `fornax/benchmarks` | target-contract benchmarks, probes, result ledger |
| `docs/fornax` | architecture, ADRs, contracts, operator workflow |

### 5.4 MAX/Mojo substrate and source precedence

Fornax remains a surgery of MAX rather than a harness over llama.cpp because MAX
and Mojo are the intended portability layer across NVIDIA, AMD, and Apple. This
is a strategic bet, not a fact of nature. The substrate ADR must make it
auditable.

Source precedence for Apple/MAX capability claims:

1. **Local probe on the pinned build in the target environment** is the gate of
   record for Fornax role assignment.
2. **Package docs and changelog for the pinned build** define official support
   status.
3. **Supported-model catalog / model docs** define model-level availability when
   present.
4. **Blog posts and launch announcements** are directional signals, not release
   gates.
5. **Nightly behavior** can unblock only after the nightly is pinned, probed, and
   recorded; future promises never unblock a gate.

If these sources disagree, capability is treated as unproven until the local
probe passes. This matters because the Modular 26.4 blog announced expanded
Apple Silicon MAX model support, while current package docs still caution that
large GenAI model inference via MAX is not yet available on Apple Silicon. Fornax
must let measured behavior decide Apple's v0 role.

### 5.5 Apple/Mac critical path and Plan B

Apple participation is staged by measured capability:

1. capacity/store for cold experts or weight reservoir;
2. expert worker for target-model expert MLP;
3. KV-heavy decode stage;
4. arbitrary pipeline stage.

V0 commits only to the lowest role that passes the target contract and Apple
expert-MLP probe. If the target model's expert MLP cannot run on the target Mac
within the contract's correctness and throughput bound by G1, Apple is demoted to
capacity-only and the v0 thesis is narrowed. That is a valid gate outcome, not a
late failure.

### 5.6 Runtime format and invariants

`runtime-format-and-invariants.md` is the load-bearing low-level spec. Before
Phase 1, it must define:

- activation tensor layout, dtype, shape, strides, alignment, padding, residency,
  ownership, lifetime, send/receive reuse rules;
- KV page layout, page size, dtype, ownership, eviction, transfer rules, replay
  behavior;
- expert batch format: token indices, expert IDs, top-k weights, packed hidden
  states, output gather order, routing metadata;
- quantization format for at-rest expert weights and in-flight payloads;
- malformed/stale/mismatched payload failure behavior;
- reference execution path using the same placement plan;
- golden-vector corpus and per-dtype tolerance methodology;
- build/toolchain assumptions for Linux NVIDIA/AMD and macOS Apple Silicon.

### 5.7 End-to-end lifecycle and LLM semantic seam

Request lifecycle:

```text
client
  -> Ignis policy/timeline or standalone OpenAI endpoint
  -> Fornax serving layer
  -> canonical chat template + tokenizer version from model support matrix
  -> admission + continuous-batch scheduler
  -> prefill across stages, KV owned by per-stage KV manager
  -> decode loop: stage compute, MoE routing, local/remote experts, weighted gather
  -> sampler on final stage
  -> stream events to serving layer
  -> cancellation/failure/cleanup through scheduler + state registry
```

LLM semantic contract additions:

- canonical tokenizer and chat-template version live in the model support matrix
  and target contract; Ignis may provide shared tokenizer code, but Fornax records
  the template/tokenizer hash used for execution;
- `EngineRequest` acceptance tests must cover messages, tools, response format,
  stop sequences, sampling parameters, max tokens, stream on/off, cancellation,
  and template version;
- `EngineResult` / stream events must define token chunks, finish reasons,
  tool/structured-output behavior, errors, and cancellation result;
- speculative decoding is out of v0 unless the target contract opts in.

### 5.8 Networking, security, and backpressure

`networking-security-and-backpressure.md` is required before Phase 1a simulation
is considered complete. Implementation requirements phase in:

| Phase | Requirement |
|---|---|
| Phase 1a T1 simulation | spec exists; bounded queues, cancellation, timeout, and plan-integrity semantics simulated |
| Phase 1b T3 lab hardware | trusted-lab exception allowed only on isolated lab network; node identity and plan ID tags implemented; bounded memory/backpressure implemented |
| Phase 3 heterogeneous prototype | security/backpressure implementation active for all inter-node payloads; encryption decision implemented or explicitly waived for lab only |
| Product deployment | endpoint auth, node admission, plan integrity, encryption posture, audit logs, and failure behavior documented and enabled by default as appropriate |

Transport remains pluggable, but the first deterministic transport should be
boring: local process/shm for simulation and TCP for cross-node tests unless
measurements force a faster path. RDMA, UCX/NIXL, and Thunderbolt-IP are later
optimizations chosen by measured benefit and macOS/Linux support.

### 5.9 Observability

Observability is required from T1 simulation onward:

- request IDs and plan IDs threaded through logs and metrics;
- per-stage timings, bubble fraction, queue depth, backpressure events;
- router decisions, remote expert hits, expert wait time, migration events;
- KV page counts, memory pressure, allocation failures, eviction/replay events;
- placement explanations: why a node was selected, excluded, or demoted;
- T1 fixture logs that can reproduce a simulated bad plan without hardware.

### 5.10 Cost-model calibration and backend coverage

Phase 0/1 must produce a backend coverage matrix for Apple, NVIDIA, and AMD:

| Operation | Required status fields |
|---|---|
| attention | supported, fast enough, correct, used by target model |
| dense MLP | supported, fast enough, correct, used by target model |
| router/top-k | supported, fast enough, correct, used by target model |
| expert GEMM/MLP | supported, fast enough, correct, used by target model |
| collect/scatter/gather | supported, fast enough, correct, used by target model |
| KV operations | supported, fast enough, correct, used by target model |
| sampling/logits | supported, fast enough, correct, used by target model |
| serialization/pack/gather | supported, fast enough, correct, used by target model |
| transport | supported, fast enough, correct, used by target model |

Profiler/tooling requirements:

- Linux NVIDIA: kernel/runtime profiler named in target contract;
- Linux AMD: ROCm/MAX-compatible profiler named if AMD is in target;
- macOS Apple Silicon: Metal/Mojo/MAX-compatible profiler or timing harness named;
- benchmark ledger records hardware, OS, driver/runtime, MAX/Mojo version, model,
  context, concurrency, quantization, thermals, and command.

Quantization choice must be made before the Apple expert-worker gate can pass,
because kernel coverage and byte compatibility depend on it.

## 6. Roadmap and Test Tiers

Only Phase 0 is hardware-free. Phase 1 and Phase 2 both have simulation and
hardware-backed tracks.

Test tiers:

| Tier | Scope |
|---|---|
| T0 | planner/scheduler unit tests, golden plans, no hardware |
| T1 | simulated workers for scheduling, transport contracts, backpressure, observability |
| T2 | single-node accelerator tests |
| T3 | two- or three-node hardware validation |
| T4 | full heterogeneous lab |

Initial T0/T1 command contract:

```text
fornax test golden-plans
fornax simulate --plan placement.json --requests synthetic_trace.json
fornax test runtime-format --golden golden_vectors/
fornax test network-contract --mode simulated
```

Names may change. Equivalent commands must exist before Phase 1 begins.

Roadmap:

- **Phase 0 - Evidence sprint + planner (T0/T1, hardware-free except optional
  Apple probe).** Build planner/cost model, write five gate artifacts, run Apple
  expert-MLP probe if hardware exists, define preflight workflow. Exit: G1 entry
  criteria met.
- **Phase 1 - Worker contract + transport (T1 then T3).** `FornaxEngine`,
  `StageWorker`, simulated workers, activation transport, plan-integrity tags,
  bounded queues, T3 small-model pipeline. Exit: correct generation across real
  pipeline; measured activation transfer; format spec honored.
- **Phase 2 - Continuous batching (T1 then T3).** Scheduler, admission, 1F1B,
  queue fairness, observability. Exit: aggregate tok/s scales with concurrency
  and matches planner within provisional bound.
- **Phase 2.5 - MoE expert-runtime surgery (T2/T3).** Expert bucketing,
  local/remote dispatch, weighted gather, expert traces, model support matrix.
  Exit: layer/logit parity and remote expert batches measured independently.
- **Phase 3 - Heterogeneous frontier MoE (T4).** Real target MoE across
  NVIDIA/AMD + Mac at Apple's gated role. Exit: predicted throughput on real
  heterogeneous hardware; security/backpressure active.
- **Phase 4 - Replication + elasticity (T4).** Stage replicas, node loss,
  replay. Exit: added nodes improve throughput; single-node loss drops no
  in-flight requests.
- **Phase 5 - Productization.** Operator UX, install, upgrade, drain, rollback,
  onboarding, docs of record. Exit: a firm can stand it up and operate it.

## 7. Ranked Risks

| Risk | Likelihood | Impact | Rank | Mitigation |
|---|---|---|---|---|
| R4 Apple/MAX readiness misses target role | High | High | 1 | source precedence, pinned probes, staged roles, demotion trigger |
| R8 target persona cannot supply concurrency | Medium | High | 2 | concurrency sweep and persona evidence in target contract |
| R5 remote expert wait dominates decode | Medium | High | 3 | budget remote hits, migrate hot experts, disable remote mode if not profitable |
| R3 heterogeneous numerics diverge | Medium | High | 4 | format spec, reference path, golden vectors, dtype tolerances |
| R1 commodity network caps throughput | Medium | High | 5 | fabric tiers, measured links, pipeline sizing, replication |
| R6 MAX internals churn | Medium | Medium | 6 | pin builds, isolate surgery, ADR source watch |
| R7 Mojo/toolchain maturity slows runtime work | Medium | Medium | 7 | lean on MAX kernels, narrow custom kernels, keep fallback tests |
| R2 pipeline depth hurts latency | High | Medium | 8 | honest positioning, depth penalty in planner, fewer big-memory nodes when needed |
| R9 security/backpressure slips past prototype | Low if gated | High | 9 | spec before Phase 1a, implementation before Phase 3/product |
| R10 status drift makes planned artifacts look proven | Medium | Medium | 10 | gate status table, owner/checklist per artifact |

## 8. Success Metrics

Targets are provisional until bound in `v0-target-contract.md`.

- **Capacity:** model exceeds largest single node and fits the target fleet with
  memory margin; target N >= 2x largest single node for v0 proof.
- **Throughput efficiency:** aggregate tok/s >= 60% of sum-of-nodes ideal at
  saturation, unless target contract revises with rationale.
- **Planner accuracy:** predicted vs measured throughput within +/-20% through
  Phase 2, tightening to +/-10% by Phase 3.
- **Concurrency efficiency:** pipeline saturation at or below contracted minimum
  concurrency for the target persona.
- **Remote expert SLO:** decode-time remote expert wait below contract SLO;
  otherwise remote experts disabled or migrated local.
- **Correctness:** layer/logit divergence within per-dtype tolerances at every
  pipeline and expert seam.
- **Apple role decision:** Apple is assigned the highest role it proves by G1;
  capacity-only is a valid outcome.
- **Security/backpressure:** bounded queues and cancellation in simulation before
  Phase 1; product security active before product deployment.
- **Elasticity:** Phase 4 metric, not v0 metric: zero dropped in-flight requests
  on single-node loss.
- **Honesty:** every reported metric traceable to a measured run or explicit
  simulation fixture.

## 9. Relationship to Ignis

Ignis is the harness; Fornax is the engine.

- **Fornax** owns model execution: distributed heterogeneous forward pass,
  scheduler, runtime format, transport, kernels, KV, expert runtime.
- **Ignis** owns agent loop: timeline, policy/exact-confirm gate, tools, replay,
  user-facing orchestration.

Fornax plugs into Ignis as `FornaxBackend` behind the `Engine` trait and can also
serve a standalone OpenAI-compatible endpoint. The boundary must be tested with
messages, tools, response format, streaming, cancellation, stop sequences, and
canonical tokenizer/template versions.

## 10. Decisions, Artifacts, and Gate Closure

Phase 1 cannot start until G1 passes. G1 is defined in
[04-stage-gates.md](program_management/04-stage-gates.md) and fed by the Phase-0
sprint plan. V3 adds the following closure rules:

| Gate item | Closure rule |
|---|---|
| `v0-target-contract.md` | not closed until reviewed and signed off by TL/SP; includes seed acceptance or replacement rationale |
| `runtime-format-and-invariants.md` | not closed until golden-vector method and failure modes are reviewable |
| `networking-security-and-backpressure.md` | not closed until phase-specific spec vs implementation requirements are explicit |
| `adr/0001-max-mojo-substrate.md` | not closed until source precedence, pinned build policy, rejected alternatives, and reversal trigger are explicit |
| Apple expert-MLP probe | not closed until measured on pinned build or Apple is demoted by G1 |
| Phase-0 preflight workflow | not closed until commands or scripts exist for inventory, probe, plan, simulate, benchmark, and diagnostics |
| Owners/staffing | not closed until roles in §0 have named assignees or Sponsor accepts a narrowed scope |

Additional ADR backlog before productization:

- pipeline-parallel default and rejected alternatives;
- bounded remote experts vs all-to-all;
- transport choice;
- security posture;
- Apple participation level;
- prefill/decode disaggregation rejection or deferral;
- homogeneous intra-node tensor-parallel island policy.

## 11. What v3 Preserves

V3 deliberately preserves the decisions the reconciled review said not to change:

- engine, not harness;
- pipeline-parallel spine;
- bounded remote experts as measured optionality;
- planner-first sequencing;
- honest single-stream latency caveat;
- MAX/Mojo preferred bet, now with stronger source precedence and Plan B;
- Ignis as operator/product layer;
- Phase 0 as evidence sprint with real kill/narrow options.

## 12. Current Decision

Proceed with Phase 0 only.

Do not begin Phase 1 distributed runtime engineering until G1 passes with the
artifacts, owners, and evidence listed above. V3 is now a plan for getting to
evidence, not a claim that the evidence already exists.
