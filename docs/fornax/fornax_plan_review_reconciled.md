# Fornax Plan Review - Reconciled Codex + Claude

Inputs reconciled:

- `docs/fornax/fornax_plan_review_claude.md`
- `fornax_plan_review_codex.md`
- Review rubric: `docs/fornax/review_lenses_by_skill_for_fornax.md`

Mode: reconciliation only. The Fornax plan documents were not changed.

## Reconciled Judgment

Needs revision before Phase 1 investment.

There is no material disagreement between the Claude and Codex reviews. Both reviews say the Fornax plan is directionally strong, technically serious, and correctly framed as custom MAX/Mojo inference-engine work rather than a harness around existing servers. Both also say the plan is not yet implementation-ready because the load-bearing claims are not quantified and the core execution invariants are not specified.

The reconciliation upgrades several items from the Codex review into explicit blockers because Claude named the exact reason they can reshape the architecture:

- A quantitative feasibility proof is missing.
- The concurrency assumption may not match the intended private-cloud market.
- The cross-vendor KV/activation/expert-batch format is undefined.
- Security and trust boundaries are absent despite the privacy value proposition.
- The Apple/MAX substrate bet needs a Plan B and reversal trigger.
- The roadmap contains a concrete contradiction: Phase 1 and Phase 2 require real multi-node hardware measurement, so only Phase 0 is truly GPU-free.

## Consensus

### Shared strengths

- The "engine, not harness" framing should remain. It is the clearest description of the work.
- The plan is right to make pipeline parallelism the default distributed spine and bounded remote expert execution a measured optimization rather than the starting assumption.
- The planner-first sequencing is strong, but only if the planner is tied to a worked quantitative target and measured calibration.
- The separation between Ignis as product/operator layer and Fornax as distributed inference engine is sound.
- The plan is honest about single-stream latency floors and should keep that honesty.
- The Apple Silicon/MAX direction is strategically important, but it must be capability-probed and backed by a fallback path.

### Shared concerns

- The plan needs one concrete `(model, fleet, context, concurrency)` target before it can prove the thesis.
- The private-cloud hardware abstraction needs exact machine definitions, memory budgets, and link measurements.
- The low-level tensor/KV/expert-batch format is the central invariant and must be specified before serious runtime work.
- LLM-serving semantics at the Ignis/Fornax boundary are under-specified: tokenization, chat templates, stop tokens, streaming, cancellation, and tool/structured-output behavior.
- Networking needs protocol, backpressure, failure, cancellation, timeout, and security semantics.
- Observability needs to be designed early, not deferred to productization.
- Ownership, staffing, CI, benchmark, and milestone gates need sharper execution detail.

## Reconciled Blockers

### B1 - No quantitative feasibility proof

Reconciled severity: Blocker before Phase 1 investment.

Codex identified the missing v0 target contract, hardware matrix, memory budget, and acceptance metrics. Claude sharpens this into the central blocker: the plan asserts high aggregate throughput on heterogeneous commodity hardware without one worked example.

Required resolution:

- Choose one reference model.
- Choose one exact fleet.
- Choose target context length and concurrency.
- Estimate memory per node: weights, experts, KV, activations, routing metadata, temporary buffers, runtime overhead, and margin.
- Estimate throughput: stage times, bubble fraction, remote expert exposure, transfer overlap, exposed transfer, TTFT, and decode tokens/sec.
- State pass/fail thresholds.

Recommended artifact:

- `docs/fornax/v0-target-contract.md`

### B2 - Concurrency-market fit is unproven

Reconciled severity: Blocker for the business/product thesis.

Codex discussed concurrency and throughput as acceptance criteria. Claude adds the missing market-risk interpretation: if the target buyer is a person, small team, or small firm, they may not generate enough concurrent requests to fill a deep heterogeneous pipeline. In that case Fornax may deliver capacity but fail the user's stated requirement of not sacrificing throughput.

Required resolution:

- Define the first operator persona and expected traffic shape.
- Estimate realistic concurrent requests for personal agent use, small firm use, and shared private AI use.
- Run or simulate concurrency sensitivity for the target fleet.
- State the minimum concurrency needed to keep the pipeline efficient.
- Decide whether the product is for single-user bursty use, small-team shared use, or private-cloud shared service.

Recommended artifact:

- A concurrency sensitivity section inside `v0-target-contract.md`.

### B3 - Core cross-vendor format is undefined

Reconciled severity: Blocker before distributed runtime implementation.

Codex called for low-level ABI contracts. Claude correctly identifies this as the engine's load-bearing invariant: the shared format for KV, activations, and expert batches across Apple, NVIDIA, AMD, transport, and MAX graph boundaries.

Required resolution:

- Specify activation tensor layout, dtype, shape, strides, alignment, padding, and ownership.
- Specify KV page layout, page size, dtype, residency, ownership, eviction, and transfer rules.
- Specify expert batch format: token indices, expert IDs, top-k weights, packed hidden states, output gather format, and routing metadata.
- Specify quantization format and compatibility expectations across MAX backends.
- Define reference/golden-vector tests and tolerance methodology.

Recommended artifact:

- `docs/fornax/runtime-format-and-invariants.md`

### B4 - Security, trust boundary, and backpressure are missing

Reconciled severity: Blocker before any multi-node private-cloud deployment.

Codex flagged trust boundaries and security as high priority. Claude upgrades this because privacy is the product promise. A multi-node fabric carrying activations and KV cache is carrying user data. Treating a LAN as a trusted local bus is not acceptable without an explicit v0 threat model.

Required resolution:

- Define node admission and identity.
- Define endpoint authentication.
- Decide whether inter-node traffic is encrypted in v0 and under what deployment assumptions.
- Define plan integrity: workers must know which placement plan they are executing.
- Define backpressure across admission, scheduler queues, stage workers, expert workers, network buffers, and client streaming.
- Define timeout, retry, cancellation, slow-worker, lost-worker, and network-partition behavior.

Recommended artifact:

- `docs/fornax/networking-security-and-backpressure.md`

### B5 - Apple/MAX substrate bet lacks Plan B

Reconciled severity: Blocker for any roadmap that places Apple Silicon on the critical path; high priority for Phase 0.

Codex treated Apple readiness as a critical measured dependency. Claude adds the missing organizational/strategic piece: the llama.cpp-to-MAX substrate decision and dependence on Modular's Apple roadmap are one-way-door-like unless there is a written reversal trigger.

Required resolution:

- Record why MAX/Mojo is the chosen substrate over llama.cpp, MLX, vLLM, SGLang, or hybrid approaches.
- Define what Apple Silicon must prove to participate in v0: hot compute, expert hosting, KV-heavy capacity, or later-stage support.
- Define a Plan B if MAX Apple support stalls.
- Define a reversal trigger: what measured failure causes the team to narrow Apple scope or revisit substrate choice?
- Track Modular/MAX version and capability assumptions as dated dependencies.

Recommended artifact:

- `docs/fornax/adr/0001-max-mojo-substrate.md`
- Apple readiness gates inside `v0-target-contract.md`.

### B6 - Phase 0-2 hardware contradiction

Reconciled severity: High priority documentation correction.

Claude catches a concrete contradiction. The plan says Phases 0-2 need no GPU and hardware enters at Phase 3, but Phase 1 and Phase 2 require a small pipeline-parallel model across 2-3 nodes, activation transfer measurement, and aggregate throughput scaling. That requires real multi-node hardware, even if not the full heterogeneous frontier fleet.

Required resolution:

- State that only Phase 0 is truly hardware-free.
- Split Phase 1 into simulation-only contract tests and hardware-backed 2-3 node validation.
- Split Phase 2 into scheduler simulation and hardware-backed continuous batching validation.
- Define required hardware for each phase.

Recommended artifact:

- Update roadmap text in `docs/fornax/project-plan.md` after review acceptance.

## Lens-by-Lens Reconciliation

| Lens | Codex judgment | Claude judgment | Reconciled judgment |
| --- | --- | --- | --- |
| Hardware | Needs revision | Needs revision | Needs revision, with B1/B5 as hard gates |
| Low-level Software | Needs revision | Blocked on core format | Blocked on B3 |
| High-level Software | Needs revision | Needs revision | Needs revision |
| LLM Expertise | Needs revision | Needs revision | Needs revision, with tokenizer/streaming seam raised |
| Hardware Acceleration | Needs revision | Needs revision | Needs revision, with calibration required |
| Networking | Needs revision | Blocked on security | Blocked on B4 for multi-node deployment |
| Software Engineering | Needs revision | Needs revision | Needs revision, with phase contradiction fixed |
| Organizational | Needs revision | Needs revision | Needs revision, with substrate ADR and owners required |
| Analytical | Needs revision | Blocked on feasibility/concurrency | Blocked on B1/B2 |
| System Engineering | Needs revision | Needs revision | Needs revision |
| People Skills | Approve with comments | Approve with comments | Approve with comments |
| Documentation | Needs revision | Approve with comments | Approve as design narrative; needs revision as implementation gate |

## Reconciled Required Changes

Priority 0: before Phase 1 investment

- Write `v0-target-contract.md` with exact model, fleet, context, concurrency, memory budget, throughput estimate, fabric assumptions, and pass/fail thresholds.
- Add concurrency-market analysis showing the target buyer can generate enough load to fill the pipeline, or narrow the product thesis accordingly.
- Write `runtime-format-and-invariants.md` for KV, activations, expert batches, quantization, routing metadata, buffer ownership, and reference correctness.
- Write the end-to-end request lifecycle: client request -> Ignis/Fornax boundary -> tokenization/chat template -> admission -> prefill -> decode -> remote experts -> streaming -> cancellation/failure -> cleanup.
- Write the networking/security/backpressure spec, including node identity, endpoint auth, inter-node trust, flow control, timeout, retry, cancellation, and partial failure.
- Record the MAX/Mojo substrate ADR and Apple Plan B.
- Fix the Phase 0-2 roadmap contradiction.

Priority 1: before heterogeneous large-model prototype

- Add hardware target matrix: exact Apple, NVIDIA, AMD, NIC, switch, OS, driver/runtime, and MAX support status.
- Add backend operation coverage by vendor: attention, dense MLP, router, expert GEMM, collect/scatter, KV operations, sampling, serialization, transfer.
- Add cost-model calibration plan using measured kernel, memory bandwidth, pack/gather, serialization, and link data.
- Add model support matrix: architecture, tokenizer, chat template, quantization, context length, MoE routing, stop behavior, streaming, tool/structured-output support.
- Add observability design: request IDs, plan IDs, per-stage timings, router traces, remote expert traces, queue depth, backpressure events, memory/KV metrics, and placement explanations.
- Add CI and hardware-lab test matrix: planner-only CI, simulated workers, single-node accelerator tests, 2-3 node tests, and full heterogeneous lab tests.

Priority 2: before productization

- Add operator quickstart, `cluster.yaml`, `model.yaml`, generated `placement.json`, and `fornax doctor` concept.
- Add deployment, upgrade, drain, restart, rollback, and node replacement procedures.
- Add contributor onboarding tracks for operator, runtime contributor, and kernel contributor.
- Add ADRs for pipeline default, bounded remote experts, transport choice, security posture, Apple participation level, and rejected alternatives.
- Add glossary and reproducible benchmark methodology.

## Reconciled Treatment of Disagreements

### Documentation lens

Claude rates documentation as "Approve with comments" while Codex rates it "Needs revision." This is not a real disagreement. Reconciled view:

- As a pre-code architecture narrative, the documentation is strong and above average.
- As an implementation gate, the documentation needs revision because it lacks targets, formats, diagrams, benchmark methodology, and decision records.

### People lens

Both reviews approve with comments. Reconciled view:

- The plan communicates the vision well.
- It should add stakeholder handling for Modular/MAX roadmap dependency and early-user expectation setting.

### Security severity

Codex originally treated security as high priority. Claude treats it as a blocker. Reconciled view:

- Security is not a blocker for a local planner or simulation phase.
- Security is a blocker before any multi-node private-cloud deployment that carries user activations or KV across machines.

### Apple Plan B severity

Codex originally treated Apple readiness as a measured critical path. Claude treats missing Plan B as a blocker. Reconciled view:

- Apple Plan B is not required to run Phase 0 planner work.
- Apple Plan B is required before committing Fornax v0 to a design where Mac nodes are necessary for throughput or capacity.

## Reconciled Go/No-Go Recommendation

Go for a Phase 0 evidence sprint only.

No-go for Phase 1 multi-node engineering until the Priority 0 artifacts exist and have been reviewed.

A good next sprint would produce five concrete outputs:

1. `v0-target-contract.md`
2. `runtime-format-and-invariants.md`
3. `networking-security-and-backpressure.md`
4. `adr/0001-max-mojo-substrate.md`
5. A corrected roadmap statement that only Phase 0 is hardware-free

If those artifacts survive review, the project should proceed to a small hardware-backed Phase 1: 2-3 nodes, tiny MoE fixture, deterministic routing traces, measured activation transfer, and calibrated planner predictions.

## What Should Not Change

- Keep the engine-not-harness stance.
- Keep the pipeline-parallel spine.
- Keep bounded remote expert execution as measured optionality, not default all-to-all.
- Keep the planner-first sequencing.
- Keep the honest single-stream latency caveat.
- Keep the MAX/Mojo direction as the preferred bet, but wrap it in dated capability gates and a Plan B.
- Keep Ignis as the operator/product layer unless implementation evidence argues otherwise.
