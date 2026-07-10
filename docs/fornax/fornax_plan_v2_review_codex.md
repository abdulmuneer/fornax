# Fornax Project Plan v2 Review (Codex)

Reviewed artifact:

- `docs/fornax/project-plan-v2.md`

Review rubric:

- `docs/fornax/review_lenses_by_skill_for_fornax.md`

Context checked:

- `docs/fornax/fornax_plan_review_reconciled.md`
- Official Modular pages referenced by the plan:
  - https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple
  - https://www.modular.com/blog/modular-26-4-sota-moe-serving-model-bringup-via-agent-skills-mojo-beta-2-and-more
  - https://docs.modular.com/max/packages/
  - https://docs.modular.com/max/develop/build-custom-ops/

## Overall Review

### Summary judgment
Approve Phase 0 with comments; blocked for Phase 1 until the five Phase-0 companion artifacts are written and reviewed.

### What looks strong
- V2 directly answers the reconciled review instead of hand-waving around it.
- The plan now correctly treats throughput as conditional on sufficient concurrency, not an unconditional property of heterogeneous spanning.
- The Phase 0 evidence sprint is now the right first milestone: planner, cost model, v0 target contract, runtime-format spec, networking/security spec, substrate ADR, and roadmap correction.
- The Phase 0-2 hardware contradiction is fixed: only Phase 0 is hardware-free; Phases 1-2 now separate simulation from hardware validation.
- The Apple/MAX bet is no longer a blind commitment. It is staged by role and has a demotion trigger.
- The plan preserves the right architecture commitments: engine-not-harness, pipeline-parallel spine, bounded remote experts, planner-first sequencing, Ignis as operator/product layer, and honest latency positioning.

### Risks or missing details
- High priority: The change map says blockers are "Resolved in v2" even though four of the five blocker-closing artifacts are explicitly not yet written. See `project-plan-v2.md` lines 11-23 and 530-541. This should read "gated by v2" or "assigned to Phase 0," not "resolved."
- High priority: The Phase-0 companion artifacts are owned only by "the Phase-0 sprint." That is too soft for blocker-closing work. See lines 532-541. Each artifact needs an owner, review lens, acceptance checklist, and go/no-go decision.
- High priority: The Apple/MAX source-of-truth is still fragile. The plan cites the 26.4 blog for Apple Silicon MAX model support, but official package docs currently say Apple Silicon GPU support is functional for Mojo GPU programming and that large GenAI model inference via MAX is not yet available on Apple Silicon. The ADR must explicitly reconcile blog claims, package docs, nightlies, and local probes.
- Medium priority: `v0-target-contract.md` is the right place for the quantitative proof, but V2 still provides no seed candidate model/fleet. This keeps Phase 0 open-ended.
- Medium priority: Operator UX is deferred to productization, but Phase 0/1 still need a minimal preflight/debug workflow to run the target contract and hardware lab work.
- Medium priority: The Ignis/Fornax generation seam still needs a more exact streaming, cancellation, tools, structured-output, and tokenizer-template contract.
- Medium priority: Elasticity is listed as a product goal and success metric, but it only lands in Phase 4. That is fine if marked as a later product capability, but risky if read as a v0 property.

### Questions for the author/team
- What exact five people or roles own the five Phase-0 artifacts?
- What is the first candidate target for `v0-target-contract.md`: model, fleet, context, concurrency, and quantization?
- What source wins if Modular blog claims, package docs, changelog, nightly behavior, and local probes disagree?
- What minimum operator command set is needed before Phase 1: inventory, validate, plan, benchmark, or serve?
- Does Phase 1 require security/backpressure only in spec form, or does any implementation need to exist before T3 hardware work?

### Required changes
- Rename the change-map column from "Resolved in v2" to something like "Addressed/gated in v2" unless the companion artifact exists and passed review.
- Add owner, reviewer, acceptance criteria, and target date/status for each Phase-0 companion artifact.
- Add a small candidate target row to the plan so Phase 0 is bounded even before `v0-target-contract.md` is complete.
- In the MAX substrate ADR requirements, explicitly define source precedence: package docs, changelog, blog, nightly probes, and local benchmark results.
- Add a minimal Phase-0/Phase-1 operator workflow, even if product UX remains Phase 5.

### Non-blocking suggestions
- Keep a small table called "Phase 0 gate status" at the top of the plan so readers know what is plan-approved vs evidence-approved.
- Add a line saying V2 is approved only as an evidence-sprint plan, not as authorization for distributed runtime engineering.
- Add a checklist that maps each old blocker to the specific artifact section that closes it once written.

### Final note from this lens
V2 changes the project from "promising but unfalsifiable" to "credible if Phase 0 produces the promised evidence." That is a meaningful upgrade. The remaining danger is wording that makes planned artifacts sound already proven.

## Hardware Review

### Summary judgment
Approve with comments for Phase 0; needs the v0 target contract before Phase 1.

### What looks strong
- The plan now names deployment bundles: `desktop-minimal`, `prosumer-rack`, and `lab-reference`.
- It explicitly moves exact SKUs, memory, NIC/fabric, and measured links into `v0-target-contract.md` and inventory files.
- The fabric tiers are no longer vague; WiFi and 1 GbE are excluded from the hot path.
- Apple Silicon roles are staged from capacity/store through expert worker, decode stage, and arbitrary pipeline stage.

### Risks or missing details
- High priority: No exact machine is named yet, so hardware feasibility is still gated rather than proven.
- High priority: The memory budget is defined as a required artifact, but no provisional example is shown in V2.
- Medium priority: Thermal and sustained consumer-GPU/Mac performance are not called out as explicit acceptance tests.
- Medium priority: `prosumer-rack` is the v0 target, but it is still a shape rather than a bill of materials.

### Questions for the author/team
- Which exact machines are the likely first `prosumer-rack` candidate?
- Will Phase 0 use real measured hardware or manufacturer specs until hardware is available?
- What sustained-run duration proves thermal stability for consumer GPUs and Macs?

### Required changes
- Add a provisional candidate fleet to V2 or make it the first required section of `v0-target-contract.md`.
- Require sustained thermal/performance measurements in the hardware acceptance checklist.
- Add a pass/fail memory worksheet template to the target contract.

### Non-blocking suggestions
- Include hardware acquisition/access status in the target contract.
- Track unsupported hardware as a living negative list.

### Final note from this lens
The hardware plan is now reviewable as a gate structure. It does not yet pass as a hardware feasibility proof.

## Low-level Software Review

### Summary judgment
Approve with comments for Phase 0; blocked for runtime implementation until `runtime-format-and-invariants.md` exists.

### What looks strong
- V2 correctly identifies KV pages, activations, expert batches, quantization, ownership, and correctness as the central runtime invariant.
- It adds a slow reference path, golden vectors, and dtype tolerances as explicit requirements.
- The MAX surgery seams are still narrow: activation transport, expert dispatch, expert MLP kernels, KV/page handoff, and stage scheduling.

### Risks or missing details
- High priority: Section 5.6 summarizes the format but says it is specified in a file that does not yet exist.
- High priority: Build/toolchain reproducibility across macOS Apple Silicon and Linux GPU nodes is still not described in detail.
- Medium priority: Failure modes for malformed payloads, stale plan IDs, dtype mismatch, or unsupported backend are still deferred.

### Questions for the author/team
- Will the format spec be schema-first, code-first, or prose-first?
- What is the canonical test fixture for golden vectors?
- Which components are allowed to allocate/free KV pages?

### Required changes
- Add the format spec before Phase 1.
- Include build/toolchain requirements in the format or substrate ADR.
- Add low-level failure cases to the spec acceptance criteria.

### Non-blocking suggestions
- Generate debug dump formats alongside the ABI rather than after debugging becomes painful.
- Include a tiny MoE fixture that exercises every payload type.

### Final note from this lens
The low-level design now names the right invariant. The project should not implement distributed runtime paths until that invariant is written and tested.

## High-level Software Review

### Summary judgment
Needs revision before Phase 1.

### What looks strong
- The primary persona is clearer: a small team or firm shared private-AI service, not a single bursty user.
- The single endpoint and Ignis/Fornax split remain understandable.
- Placement explanations and honest metrics are now explicit operator-facing ideas.

### Risks or missing details
- High priority: Operator UX is deferred to Phase 5, but Phase 0/1 require some UX to inventory hardware, validate links, run the planner, inspect a plan, and benchmark.
- Medium priority: Public vs private API boundaries are still not named beyond the `Engine` trait and OpenAI-compatible endpoint.
- Medium priority: Error/status behavior for infeasible plans is implied by the planner but not surfaced as a user workflow.

### Questions for the author/team
- What does an operator run first: `fornax inventory`, `fornax validate`, `fornax plan`, or an Ignis command?
- What file shape describes the fleet before inventory is complete?
- What does a user see when the planner says the model cannot meet contracted concurrency?

### Required changes
- Move a minimal preflight/operator workflow from Priority 2 into Phase 0 or Phase 1.
- Define the first public config artifacts at least as stubs: cluster inventory, model target, placement result.
- Add failure UX for infeasible placement and failed hardware validation.

### Non-blocking suggestions
- Keep product polish in Phase 5, but do not defer operational introspection.
- Make `fornax doctor` an internal tool before it is a product feature.

### Final note from this lens
The product mental model is strong. The first-hour operator workflow still needs to exist earlier than productization.

## LLM Expertise Review

### Summary judgment
Approve with comments.

### What looks strong
- V2 adds an end-to-end request lifecycle with tokenization, scheduling, prefill, decode, expert dispatch, sampling, streaming, cancellation, failure, and cleanup.
- It explicitly names the tokenizer/chat-template/stop-token seam.
- Phase 2.5 now requires a model support matrix covering architecture, tokenizer, chat template, quantization, MoE routing, stop behavior, streaming, and tool/structured-output.
- Correctness now includes layer/logit matching against a reference path.

### Risks or missing details
- Medium priority: "Fornax serving layer, reuses Ignis tokenizer" still leaves room for drift in chat template ownership and versioning.
- Medium priority: Tool and structured-output behavior is named in Phase 2.5, but the `Engine` seam still lists only `generate(messages_json, tools_json, max_new_tokens)`.
- Medium priority: Speculative decoding is not addressed; this is acceptable if explicitly out of scope for v0.

### Questions for the author/team
- What object owns canonical chat template versions: Ignis, Fornax, model artifact metadata, or the target contract?
- Is tool calling passed through as OpenAI-compatible request semantics or normalized by Ignis before Fornax sees it?
- Is speculative decoding intentionally out of v0?

### Required changes
- Add tokenizer/template version ownership to the model support matrix requirements.
- Add streaming, cancellation, tools, structured-output, and stop behavior to the `Engine` seam acceptance tests.
- Explicitly mark speculative decoding in or out for v0.

### Non-blocking suggestions
- Create golden prompts that test BOS/EOS, chat roles, stop strings, tool calls, and streamed chunk boundaries.

### Final note from this lens
V2 materially improves the LLM-serving story. The remaining work is semantic exactness at the Ignis/Fornax boundary.

## Hardware Acceleration Expertise Review

### Summary judgment
Approve with comments.

### What looks strong
- V2 adds backend operation coverage and cost-model calibration as first-class Phase-0/Phase-1 concerns.
- The Apple demotion gate is the right pattern: measure and assign roles based on evidence.
- The plan keeps custom Mojo kernels narrow and disposable when MAX catches up.
- The official MAX custom-op path exists and matches the plan's assumption that custom graph ops can be written in Mojo and loaded into MAX graphs.

### Risks or missing details
- High priority: The Apple/MAX support wording needs source-of-truth discipline. The 26.4 blog says Apple Silicon MAX support has expanded for many common architectures, while the current package docs still say large GenAI model inference via MAX is not yet available on Apple Silicon.
- Medium priority: Operation coverage is a required matrix, but the plan does not say who measures it or which profiler stack is used per platform.
- Medium priority: Quantization choices are still deferred but heavily affect kernel coverage.

### Questions for the author/team
- Which official source has precedence for Apple support: package docs, changelog, blog, nightly behavior, or local probes?
- What profiler tools will be used on Linux NVIDIA/AMD and macOS Apple Silicon?
- Which quantization format is the first target for expert weights and in-flight activations?

### Required changes
- Add source precedence and local-probe requirements to the substrate ADR.
- Add profiler/tooling requirements to the backend coverage matrix.
- Require the quantization decision before any Apple expert-worker gate can pass.

### Non-blocking suggestions
- Track each operation as `supported`, `fast enough`, `correct`, and `used by target model`; support alone is not enough.

### Final note from this lens
The acceleration plan is now measurement-driven. The main remaining trap is treating vendor announcement language as equivalent to usable target-model throughput.

## Networking Expertise Review

### Summary judgment
Approve with comments for Phase 0; blocked for product deployment until the networking/security spec exists and passes review.

### What looks strong
- V2 now treats activations and KV as user data crossing a trust boundary.
- Node identity, admission, endpoint authentication, plan-integrity tags, encryption decisions, backpressure, and failure semantics are all named.
- Backpressure is scoped across the whole path rather than just the client endpoint.
- Phase 1 is gated on Priority-0 artifacts passing review.

### Risks or missing details
- High priority: The networking/security document is not written yet.
- Medium priority: It is unclear whether Phase 1b T3 hardware work can use a trusted-lab exception before security implementation exists.
- Medium priority: Transport choice remains open; that is fine, but the spec must define the evaluation criteria.

### Questions for the author/team
- Is `networking-security-and-backpressure.md` required before Phase 1a simulation, Phase 1b hardware, or only before Phase 3?
- What is the minimum implementation of node identity and plan integrity for lab hardware?
- Which transport will be evaluated first and why?

### Required changes
- State exactly which phases require the security/backpressure spec and which require implementation.
- Include trusted-lab exception rules in the spec.
- Define transport evaluation criteria: latency, throughput, zero-copy potential, operational simplicity, macOS/Linux support, and failure behavior.

### Non-blocking suggestions
- Keep transport pluggable, but pick one boring first transport for deterministic Phase 1 tests.

### Final note from this lens
Networking is no longer ignored. The next risk is ambiguity about when the spec becomes executable policy.

## Software Engineering Review

### Summary judgment
Needs revision before Phase 1.

### What looks strong
- The test tiers T0-T4 are a big improvement and separate CI-able simulation from hardware lab validation.
- Phase 1 and Phase 2 are now split into simulated and hardware-backed tracks.
- Phase 0 has concrete artifact outputs and golden-plan tests.

### Risks or missing details
- High priority: There is still no module map or repository/package boundary for planner, scheduler, workers, transport, MAX integration, benchmarks, and docs.
- High priority: The five Phase-0 artifacts have no individual owners or review acceptance checklists.
- Medium priority: CI is categorized by tier, but not mapped to actual commands, fixtures, or pass/fail gates.
- Medium priority: Packaging/build/version policy is not yet specified for mixed macOS/Linux work.

### Questions for the author/team
- Where will the planner, worker contracts, transport, runtime format schemas, and benchmark harness live?
- What command runs T0 and T1 in CI?
- How are MAX/Mojo versions pinned and upgraded?

### Required changes
- Add an implementation/module map before Phase 1.
- Add owners and acceptance checklists for Phase-0 artifacts.
- Add initial CI command expectations for T0/T1.
- Add dependency/version pinning expectations to the substrate ADR.

### Non-blocking suggestions
- Treat benchmark and inventory schemas as versioned APIs from the beginning.

### Final note from this lens
The roadmap is much more testable now. It still needs engineering ownership and module boundaries before code starts spreading.

## Organizational Skill Review

### Summary judgment
Needs revision.

### What looks strong
- V2 does a good job translating review feedback into roadmap gates.
- The Apple bet is now reversible by measured evidence.
- The project now has a smallest useful milestone: Phase 0 evidence sprint.

### Risks or missing details
- High priority: "Owned by the Phase-0 sprint" is not sufficient ownership for blocker-closing artifacts.
- High priority: No staffing assumptions are stated for MAX/Mojo, Apple profiling, distributed runtime, LLM correctness, and security/networking.
- Medium priority: The Modular dependency is tracked technically, but relationship/stakeholder handling is still not described.

### Questions for the author/team
- Who signs off each Phase-0 artifact?
- What skills must be staffed before Phase 1 begins?
- Who tracks Modular/MAX upstream changes and decides whether a nightly is safe to adopt?

### Required changes
- Add an ownership/RACI table for Phase 0 and Phase 1.
- Add a staffing/skill gate for Phase 1.
- Add an upstream-dependency owner for Modular/MAX.

### Non-blocking suggestions
- Schedule a short review per artifact using the same lens rubric instead of one giant review at the end of Phase 0.

### Final note from this lens
V2 is better sequenced. It still needs named accountability.

## Analytical Skills Review

### Summary judgment
Approve with comments for Phase 0.

### What looks strong
- The central hypothesis is now explicitly falsifiable through the v0 target contract.
- The concurrency-market risk is no longer hidden; it is a go/no-go gate.
- The plan preserves the right rejected default: no cross-vendor tensor-parallel all-reduce outside homogeneous islands.
- Planner calibration is now required before predictions are trusted.

### Risks or missing details
- Medium priority: The plan has no seed worked example, so Phase 0 could still spend too long choosing a target.
- Medium priority: Rejected alternatives are mentioned but the ADR backlog should make sure prefill/decode disaggregation and homogeneous intra-node tensor-parallel islands are recorded.
- Medium priority: Risks are listed but still not ranked by likelihood x impact.

### Questions for the author/team
- What one measurement would kill the v0 target?
- What is the expected minimum concurrency for the small-team persona before measurement?
- Which alternative is the strongest baseline: single dual-GPU node, naive pipeline, expert-only offload, or existing engine?

### Required changes
- Add a seed target candidate or candidate shortlist.
- Add risk ranking by likelihood and impact.
- Require rejected alternatives in the substrate/architecture ADRs.

### Non-blocking suggestions
- Include a `known-wrong assumptions` section in the v0 target contract for planner approximations.

### Final note from this lens
The reasoning is now disciplined enough for an evidence sprint. It should not be allowed to turn into an unbounded research phase.

## System Engineering Review

### Summary judgment
Approve with comments.

### What looks strong
- V2 adds a complete request lifecycle and state ownership table.
- It names control plane, data plane, state, client surface, and worker health responsibilities.
- Observability is designed from Phase 1 instead of Phase 5.
- The phased roadmap now composes simulation, hardware validation, MoE surgery, heterogeneous frontier serving, elasticity, and productization.

### Risks or missing details
- Medium priority: The lifecycle is good at plan level, but it does not yet show data representation transitions in detail.
- Medium priority: Operational lifecycle is still mostly Phase 5; some bootstrap/drain/restart mechanics will affect Phase 1-3 test design.
- Medium priority: State registry behavior under cancellation and replay is named but not specified.

### Questions for the author/team
- Where is the authoritative request state machine stored?
- Does the state registry own replay, or does the scheduler own replay using registry snapshots?
- What observability exists in T1 simulation before hardware exists?

### Required changes
- Add request-state-machine requirements to the networking or runtime-format artifact.
- Add minimal bootstrap/shutdown/drain semantics before Phase 1 hardware tests.
- Define T1 observability fixtures.

### Non-blocking suggestions
- Add sequence diagrams once the companion artifacts exist.

### Final note from this lens
V2 now reads like a system rather than a set of components. The companion specs need to preserve that end-to-end ownership.

## People Skills Review

### Summary judgment
Approve with comments.

### What looks strong
- The plan is candid about who the system is for and who it is not for.
- It keeps expectations honest for single-user and low-concurrency deployments.
- The review-to-change map is a good communication device.
- The plan is easier to teach: engine, target contract, runtime invariant, networking/security, substrate ADR.

### Risks or missing details
- Medium priority: Contributor onboarding is still deferred.
- Medium priority: Stakeholder alignment with Modular remains implicit.
- Low priority: The change-map wording may cause overconfidence by calling gated work resolved.

### Questions for the author/team
- Who needs to believe in the project after Phase 0: early customers, Modular, contributors, or internal leadership?
- What should a new contributor be able to do after one week?
- How will the team communicate failed gates without making the project look like it failed entirely?

### Required changes
- Fix the change-map wording so stakeholders understand evidence gates are still open.
- Add a stakeholder note for Modular/upstream dependency and early target operators.

### Non-blocking suggestions
- Create separate onboarding paths later for operator, runtime contributor, and kernel contributor as planned.

### Final note from this lens
The human-facing narrative is much healthier in V2. Keep the honesty; do not let roadmap optimism blur gate status.

## Documentation Review

### Summary judgment
Approve with comments as a Phase-0 plan; needs revision before Phase 1 authorization.

### What looks strong
- V2 is very readable and traceable to the reconciled review.
- It preserves v1 for review history and clearly identifies what changed.
- It adds the right companion artifacts and maps them to blockers.
- It fixes the Phase 0-2 contradiction in the roadmap.

### Risks or missing details
- High priority: The "Resolved in v2" column is inaccurate for blockers whose companion artifact is unwritten.
- Medium priority: The companion artifacts are listed but have no templates, owners, or acceptance checklists.
- Medium priority: The plan uses current Modular claims; because those sources are fast-moving and partially inconsistent, the ADR should include dated source snapshots and local probe results.
- Medium priority: Diagrams are still absent, though acceptable for Phase 0 if the companion artifacts add them.

### Questions for the author/team
- Will the companion artifacts each use the same review lens format or artifact-specific checklists?
- Where will dated external source snapshots live?
- Which doc is the source of truth for Phase 0 status once artifacts begin landing?

### Required changes
- Rename "Resolved in v2" to "Addressed/gated in v2" or equivalent.
- Add artifact templates or required section lists for the five Phase-0 outputs.
- Add dated external-source handling to the substrate ADR.

### Non-blocking suggestions
- Add diagrams after the runtime-format and networking specs land.
- Add a glossary once operator docs begin.

### Final note from this lens
V2 is a strong revision document. Its documentation risk is status precision: readers must know what is proven, what is planned, and what is still blocked.

## Consolidated Recommendation

Proceed with Phase 0 only.

Do not start Phase 1 distributed runtime engineering until:

- The change-map status language is corrected.
- `v0-target-contract.md` is written and passes review.
- `runtime-format-and-invariants.md` is written and passes review.
- `networking-security-and-backpressure.md` is written and passes review.
- `adr/0001-max-mojo-substrate.md` is written, including Apple source precedence and Plan B triggers.
- Each Phase-0 artifact has an owner, acceptance checklist, and go/no-go decision.

V2 is a good plan for getting to evidence. It is not yet the evidence.
