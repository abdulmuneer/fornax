# Fornax Project Plan — Multi-Lens Review

**Artifact reviewed:** `docs/fornax/project-plan.md` (the elaborated plan)
**Against:** `docs/fornax/review_lenses_by_skill_for_fornax.md`
**Mode:** Read-only. No changes made to the plan.
**Reviewer:** Claude (single reviewer representing all twelve lenses in turn)
**Date:** 2026-06-20

> Method note: per the lens doc, reviews are *combined, not averaged* — a single
> blocker can reshape the plan. The plan is unusually honest and well-reasoned for
> a pre-code design (the latency-floor honesty in §2 and the "engine not harness"
> framing are genuine strengths). The blockers below are therefore about
> **missing evidence and undefined core invariants**, not about direction.

---

## Combined judgment

**Overall: Needs revision before major engineering investment.**

The strategy is coherent and the scoping discipline (model-free Phases 0–2,
pipeline spine, bounded remote experts) is good. But the plan asserts its central
claim — *high aggregate throughput on commodity heterogeneous hardware* — **without
a single worked quantitative example**, leaves the **engine's core data invariant
(the cross-vendor KV/activation/expert-batch format) undefined**, and **omits
security and the Ignis↔Fornax tokenization/streaming seam entirely**. Any one of
these can reshape the plan.

**Blockers (must address before Phase 1 investment):**

- **B1 — No quantitative feasibility proof.** Not one (model, fleet, concurrency)
  → predicted tok/s worked example exists. The whole thesis is unvalidated on
  paper. *(Analytical, Hardware Acceleration, Hardware)*
- **B2 — Concurrency assumption may not match the market.** Throughput
  preservation requires concurrency that "a firm's in-house users" may not supply;
  if concurrency is low, the pipeline cannot be filled and the thesis weakens to
  the single-user latency floor the plan itself disowns. *(Analytical)*
- **B3 — Core format undefined.** "One KV/activation/expert-batch format owned
  across vendors" is named as the unlock but never specified (layout, dtype,
  paging, quant scheme). This is the engine's load-bearing invariant.
  *(Low-level Software, LLM)*
- **B4 — Security/trust boundary absent.** A "private AI" product moves user
  activations/KV across a multi-node consumer LAN with zero authN/authZ or
  threat model. *(Networking)*
- **B5 — Apple critical path has no Plan B.** If MAX-Apple stalls, the capacity
  tier is blocked; "CPU fallback" is not a throughput answer. A one-way-door bet
  with no recorded reversal cost. *(Hardware, Organizational, Analytical)*

**High-priority (before release / scale-up):** hardware matrix + memory budget
table; tokenizer/chat-template/streaming seam; observability design for
distributed debugging; backpressure/overload behavior; filled acceptance numbers
(X/Y/N); decision record for the llama.cpp→MAX pivot and the dropped
prefill/decode-disaggregation alternative.

**One concrete internal contradiction to fix regardless:** §6 says "Phases 0–2
need no GPU … hardware enters at Phase 3," but Phase 1 ("run a small model
pipeline-parallel across 2–3 like nodes; activation transfer measured") and Phase
2 ("aggregate tok/s scales with concurrency; matches planner prediction") both
require real multi-node hardware and measurement. Only Phase 0 is truly
GPU-free.

---

## Hardware Review

### Summary judgment
Needs revision

### What looks strong
- §4 correctly tiers hardware by role (hot accelerator vs capacity/expert) and
  ties the MoE capacity/compute split to unified memory vs consumer GPU — the
  right physical intuition.
- Fabric is treated as a first-class design input (§4, §5.3), not an afterthought.

### Risks or missing details
- **No hardware target matrix** (minimum / recommended / production / explicitly
  unsupported) with concrete SKUs, memory, and bandwidth. "RTX 40/50 series,
  Radeon/MI where MAX support exists" is exactly the vague phrasing this lens
  exists to catch — *which* AMD parts? MAX/ROCm coverage is narrow.
- **No memory budget table.** "671B 4-bit ≈ 400 GB" is a single weight figure.
  There is no per-node breakdown of weights + KV (at target concurrency ×
  context) + activations + expert reservoir + LRU cache + fragmentation. KV at
  high concurrency can dominate and is unbudgeted.
- **Apple bandwidth unquantified.** Expert-MLP and decode on Mac are
  bandwidth-bound; M3 (~400 GB/s) vs M3 Ultra (~800 GB/s) is a 2× swing that
  decides whether remote-expert economics close. No numbers.
- **Team hardware access not stated.** Plan assumes a NVIDIA+Mac fleet for
  Phases 1–3; is it in hand?

### Questions for the author/team
- What is the exact v1 fleet (SKUs, memory, links), and what is out of scope?
- For the reference model at target concurrency/context, what is the per-node
  memory budget — and where does KV live as concurrency grows?
- Which AMD GPUs are actually in scope given MAX's ROCm support today?

### Required changes
- Add a hardware target matrix and a worked memory-budget table for one concrete
  (model, fleet, concurrency, context).
- Replace "where MAX support exists" with a named, dated support list.

### Non-blocking suggestions
- State the assumed Apple chip/bandwidth class per tier so reviewers can sanity
  the expert-locality math.

### Final note from this lens
The plan reasons about hardware *shape* well but never *names the machine*. It
cannot pass this lens until memory fits are shown, not asserted.

---

## Low-level Software Review

### Summary judgment
Blocked on one item (B3); otherwise needs revision

### What looks strong
- §5.5 keeps the surgery at *explicit seams* (transport, dispatch, expert MLP,
  KV/page handoff, scheduling) and commits to deleting custom kernels as MAX
  ships equivalents — good boundary discipline.
- R3/R6/R7 already name heterogeneous numerics, MAX-internal churn, and beta
  toolchain risk.

### Risks or missing details
- **The cross-vendor KV/activation/expert-batch format is undefined (B3).**
  Layout (row/col, paging, alignment), dtype per field, and quantization scheme
  are the engine's central invariant and appear only as a phrase ("owned across
  vendors … consistent by construction"). Construction of *what* is unspecified.
- **No reference/golden-vector strategy.** R3's mitigation ("numeric validation
  harness") and the §8 metric ("layer/logit divergence within tolerance") need a
  named reference backend, a tolerance methodology per dtype, and golden vectors —
  none defined. Apple's fp19-truncated matmul means the *same* logical expert
  yields different bits; "within tolerance" is doing heavy undefined lifting.
- **Memory ownership across the transport boundary** (who owns an activation/KV
  buffer during send, when it frees, double-buffering for 1F1B) is unspecified.
- **Quant format compatibility across runtimes** (MAX quant vs MLX vs the stored
  expert weights) is not addressed; experts must be byte-compatible wherever they
  execute.

### Questions for the author/team
- What is the wire/at-rest format for KV, activations, and expert batches?
- What is the reference path, and what tolerance proves an Apple expert MLP
  "matches"?
- Which MAX APIs are treated as stable vs internal, and what is the version-pin +
  isolation strategy beyond "pin a verified build"?

### Required changes
- Write a short format-and-invariants spec (even one page) before Phase 1.
- Define the reference-vs-optimized correctness method and golden-vector source.

### Non-blocking suggestions
- Document buffer lifecycle in the `StageWorker` contract.

### Final note from this lens
The seams are well chosen; the *data flowing through them* is undefined. That gap
is where a heterogeneous engine silently corrupts results.

---

## High-level Software Review

### Summary judgment
Needs revision

### What looks strong
- The Ignis↔Fornax split (§9) and the `Engine`-trait seam give a clean public
  story; "single OpenAI-compatible endpoint" is the right client surface.
- "Engine, not a harness" is a crisp, teachable one-line mental model.

### Risks or missing details
- **Fleet/cluster configuration UX is undefined.** How does an operator declare
  nodes, links, model, quant, placement preferences, and remote-expert budget?
  Inventory is "discovered/measured" but the config surface and defaults are
  absent.
- **Distributed error model is invisible to the user.** What does a caller see
  when a node is too slow, a stage drops, or a partition is infeasible? §
  `infeasible_reason` exists in the partitioner but not in the serving API.
- **Escape hatches unspecified.** Expert placement / migration / remote-hit
  budget are core tuning knobs with no surfaced controls.

### Questions for the author/team
- What does a user do in the first 10 minutes — and what config file or CLI
  expresses "these 3 machines, this model"?
- Which concepts are public API vs internal (stages, experts, placement plan)?

### Required changes
- Specify the configuration surface and the serving-side error/status model.

### Non-blocking suggestions
- Promote the partitioner's "will it run, how fast" output to a first-class
  pre-flight CLI — it is the most adoptable artifact in the plan.

### Final note from this lens
Strong internal model; the operator-facing surface (configure → run → diagnose)
is not yet designed.

---

## LLM Expertise Review

### Summary judgment
Needs revision

### What looks strong
- Prefill/decode are modeled distinctly in the cost model; the MoE block
  (router → bucket → local/remote → weighted gather) is correctly described and
  Phase 2.5 gates on logit match — the right correctness contract.
- Decode-as-bandwidth-bound vs prefill-as-FLOP-bound is stated correctly.

### Risks or missing details
- **Tokenizer / chat template / stop tokens / special tokens are absent.** The
  seam `generate(messages_json, tools_json, …)` implies Fornax renders the prompt
  and tokenizes, but the plan never says where the HF chat template, BOS/EOS, and
  stop handling live — a real Ignis↔Fornax boundary question (Ignis currently owns
  prompt rendering per the repo).
- **Streaming / cancellation semantics** from the last pipeline stage back to the
  client are unspecified — non-trivial when generation is mid-pipeline under
  continuous batching.
- **Speculative decoding is unaddressed**, despite Ignis already having it. Draft
  placement and cross-stage verification interact strongly with a distributed
  pipeline; this is both a gap and a missed lever.
- **Prefill/decode disaggregation is dropped** with no rationale — notable since
  it was the originating idea (NVIDIA prefill / Mac decode). At least record why
  pipeline-parallel beat it.
- **Per-stage KV paging under continuous batching** is not designed; "distributed
  KV/prefix registry" is named but per-stage eviction/fragmentation at concurrency
  is the harder problem.
- **MoE router numerics** (top-k tie-breaking, renormalization) must match
  reference exactly across vendors — flagged implicitly by Phase 2.5 but not
  called out as a numerics risk.

### Questions for the author/team
- Where do tokenization and chat templating execute, and how are they tested?
- How do streaming and cancellation behave across stages under load?
- Is speculative decoding in or out, and how does it map onto the pipeline?

### Required changes
- Add a tokenizer/chat-template/stop-token ownership statement and a streaming
  contract.
- Record the prefill/decode-disaggregation rejection rationale.

### Non-blocking suggestions
- Add a small model-support matrix (architectures in/out) as in the LLM lens.

### Final note from this lens
Model *execution* is well thought through; model *semantics at the edges*
(templates, streaming, stops, spec-decode) are missing and these are where LLM
serving usually breaks.

---

## Hardware Acceleration Review

### Summary judgment
Needs revision

### What looks strong
- The six levers correctly target the real costs (bubbles, exposed transfer,
  imbalance, remote-expert exposure); the cost model separates overlapped vs
  exposed remote-expert wait (partitioner §3.2) — sophisticated and honest.
- "Network is free when t_xfer < t_compute" is the right framing.

### Risks or missing details
- **No calibration plan.** The cost model is analytical; there is no plan to
  calibrate `compute_class`, `mem_bandwidth`, kernel times, or pack/gather
  overhead against measured MAX kernels before trusting placement decisions.
- **The "network is free" claim is unbounded.** It fails precisely in the deep-
  pipeline / low-batch regime where per-stage compute drops below transfer+latency.
  The plan never states the batch/depth boundary where overlap stops working.
- **Remote-expert kernel regime unanalyzed.** Remote expert batches are small and
  data-dependent → likely launch/bandwidth-bound on Apple; the economics in §5.1
  ("keep remote hits rare/batched") hinge on numbers that aren't worked.
- **Quant scheme unspecified** (GGUF/AWQ/GPTQ/MLX/MAX-native) — decides both
  kernel availability per vendor and cross-vendor compatibility.

### Questions for the author/team
- How and when is the cost model calibrated against real kernels?
- At what (depth, batch) does boundary transfer stop being hideable?
- Which quantization format, and is it executable on all three vendors?

### Required changes
- Add a calibration phase (fold into Phase 1) and state the overlap-failure
  regime explicitly.

### Non-blocking suggestions
- Add an acceleration-coverage-by-op table (attention/GEMM/MoE/sampling) per
  vendor with "MAX kernel / custom / fallback" status.

### Final note from this lens
The model is unusually honest about exposed cost — but an uncalibrated cost model
driving placement is itself the hot path that must be validated first.

---

## Networking Review

### Summary judgment
Blocked on security (B4); otherwise needs revision

### What looks strong
- Transport is pluggable (TCP/RDMA/TB-IP/shm) and topology-aware boundary
  placement is in the design; communication cost is in the performance model
  (good — many plans omit it).

### Risks or missing details
- **No security/trust boundary at all (B4).** For a *private AI* product, inter-
  node RPCs carry user activations and KV — i.e. user data in the clear — across
  consumer machines. No authN, no encryption, no threat model, no endpoint auth.
  A single compromised node observes everything.
- **No backpressure/overload design.** Continuous batching + queues invites
  unbounded memory growth; the plan has no admission/backpressure/shedding story.
- **Failure semantics are thin.** Elasticity is a *goal* (§3) and Phase 4
  promises replay, but worker discovery/membership, rank mapping, partition-on-
  network-split, timeouts, and retries are unspecified.
- **Streaming/cancellation/timeout** at the endpoint under distributed execution
  are undefined (also LLM lens).

### Questions for the author/team
- What authenticates inter-node RPCs and the endpoint; is fabric traffic
  encrypted?
- Where does backpressure originate and how is it signaled to the client?
- What is the exact behavior on node loss, slow node, and network partition?

### Required changes
- Add a trust-boundary/threat model and a backpressure design before any
  multi-node deployment.

### Non-blocking suggestions
- Specify a membership/heartbeat protocol feeding the inventory probes.

### Final note from this lens
Treating the fabric as a trusted local bus is the classic distributed-systems
error — and here it also undercuts the product's core privacy promise.

---

## Software Engineering Review

### Summary judgment
Needs revision

### What looks strong
- Phased roadmap with concrete exit criteria; model-free Phase 0 planner is a
  genuinely testable, releasable unit (aligns with Ignis fixture discipline).
- Explicit seams (§5.5) give natural module boundaries.

### Risks or missing details
- **CI story for the distributed/heterogeneous path is absent.** You cannot
  easily CI a NVIDIA+Mac cluster; only the planner is CI-able. How are Phases 1–4
  regression-tested?
- **Acceptance numbers are placeholders** (N≥2–3 is set, but Y% and ±X% are not).
  Targets that aren't numbers can't gate.
- **Scope is very large for an unstated team** (engine + Metal kernels + scheduler
  + transport + planner). Scope-control risk; the smallest shippable slice (the
  planner as a pre-flight tool) isn't framed as a release.
- **The §6 "no GPU until Phase 3" claim contradicts Phase 1/2 content** (see
  combined judgment).

### Questions for the author/team
- What runs in CI vs on a hardware lab, and how is the hardware path gated?
- What are the real numbers behind X/Y, and who sets them?

### Required changes
- Add a test/CI strategy distinguishing model-free (CI) from hardware-in-the-loop
  stages; fix the Phase 0–2 hardware contradiction; fill or TBD-with-owner the
  metric placeholders.

### Non-blocking suggestions
- Carve the planner out as a standalone releasable artifact/milestone.

### Final note from this lens
Good bones and good phasing; the testability of everything past Phase 0 is the
unanswered maintainability question.

---

## Organizational Skill Review

### Summary judgment
Needs revision

### What looks strong
- Clear phase sequencing with exits; the Apple workstream is correctly called out
  as a *parallel* critical path rather than buried in a later phase.

### Risks or missing details
- **No owners, no team, no required-skills statement.** This needs Mojo/MAX
  kernel authors, Metal/Apple-GPU expertise, and distributed-systems engineering
  simultaneously — rare combination. Capacity is unaddressed.
- **One-way-door bet under-managed (B5).** The llama.cpp→MAX pivot and the
  dependency on Modular's Apple roadmap are high-cost-to-reverse and externally
  controlled, yet there's no decision record, no trigger condition for
  re-evaluating, and no Plan B if Apple inference stalls.
- **External-org dependency (Modular)** is treated as a technical risk (R4/R6)
  but not as a *relationship/roadmap-alignment* risk.

### Questions for the author/team
- Who owns the Apple kernel workstream in parallel with the engine?
- What observable condition would trigger reverting to a llama.cpp substrate?

### Required changes
- Add owners per workstream and a written decision record for the substrate bet
  with an explicit reversal trigger.

### Non-blocking suggestions
- Define the smallest useful milestone as a fundable unit (the planner).

### Final note from this lens
Ambition is well-sequenced but assumes a team and an external roadmap that the
plan never pins down.

---

## Analytical Skills Review

### Summary judgment
Blocked on B1/B2; needs revision

### What looks strong
- The §2 latency-floor honesty and the explicit refusal to promise single-stream
  parity is exactly the kind of deliberate-constraint reasoning this lens rewards.
- The parallelism table reasons from comms cost, correctly rejecting cross-node
  tensor-parallel.

### Risks or missing details
- **No falsifiable quantitative claim (B1).** "High aggregate throughput" has no
  worked example: pick DeepSeek-R1 4-bit on (2×RTX + 1 Mac), a concurrency, and
  show predicted tok/s, bubble fraction, and remote-expert exposure. Without it,
  the thesis is unfalsifiable and the cost model untested against intuition.
- **Central assumption (concurrency) may contradict the market (B2).** Throughput
  preservation needs enough in-flight requests to fill the pipeline. "A firm's
  in-house users" may produce tens, not hundreds, of concurrent requests — which
  may not fill a deep heterogeneous pipeline, collapsing toward the single-user
  latency floor §2 disowns. This tension is the plan's biggest unexamined risk.
- **Rejected alternatives not recorded.** Prefill/decode disaggregation and
  intra-node tensor-parallel "islands" (the 2-GPU box!) are mentioned then
  dropped without analysis; the latter is likely a real and important topology.
- **Risks aren't ranked by likelihood × impact.**

### Questions for the author/team
- What concurrency does the target buyer actually generate, and does the pipeline
  fill at that level? What experiment settles it cheapest?
- What single number, if measured below threshold, kills the project?

### Required changes
- Add at least one end-to-end worked throughput example and a
  concurrency-sensitivity analysis; record rejected alternatives.

### Non-blocking suggestions
- Rank R1–R7 by likelihood/impact.

### Final note from this lens
The reasoning is honest and mostly sound — but the load-bearing claim has no
arithmetic behind it, and the concurrency assumption may quietly invert the whole
business case.

---

## System Engineering Review

### Summary judgment
Needs revision

### What looks strong
- Layered component table (§5.3) with clear "provided vs build" columns; a real
  first end-to-end milestone exists early (Phase 1).
- Ignis integration gives a coherent product lifecycle at the edges.

### Risks or missing details
- **No single end-to-end request trace.** There is no narrative from client →
  endpoint → tokenize → prefill across stages → continuous-batched decode with
  remote experts → stream → cleanup. The MoE block diagram is a fragment, not the
  lifecycle.
- **Observability is a value, not a design.** "Honest telemetry" and
  "bubble-fraction telemetry" are named, but the hardest thing in this system —
  debugging a wrong/slow token across tokenizer, router, a remote expert, and a
  stage seam — has no tracing/metrics design.
- **Operational readiness is mostly Phase 5.** Config/deploy/upgrade deferred;
  node-restart/elasticity is Phase 4. Acceptable but the lens warns against late
  ops.
- **Data-representation changes across layers** (hidden states → packed expert
  batch → gathered output; KV across stages) are undocumented.

### Questions for the author/team
- Can you write the full lifecycle for one request, naming the owning layer at
  each hop?
- What traces/metrics exist at the tokenizer, router, expert, and stage
  boundaries?

### Required changes
- Add an end-to-end lifecycle narrative and a boundary observability plan.

### Non-blocking suggestions
- Add a deploy/upgrade/restart story sketch even if implementation is later.

### Final note from this lens
Impressive components; the connective tissue (one coherent request lifecycle +
the observability to debug it) is not yet drawn.

---

## People Skills Review

### Summary judgment
Approve with comments

### What looks strong
- The plan is genuinely teachable: "engine not harness," the MoE surgical seam,
  and the honest-constraint section give newcomers a fast, correct mental model.
- Honesty about limitations builds the trust this kind of project needs.

### Risks or missing details
- **Stakeholder alignment with Modular is unaddressed.** Success depends on an
  external org's Apple roadmap; no relationship/communication plan.
- **No onboarding/contributor surface** yet (acceptable at this stage).
- Some knowledge (numerics tolerances, format) risks living only in authors'
  heads until B3 is written down.

### Questions for the author/team
- Who outside the team must believe in this (Modular, early buyers), and how are
  they engaged?

### Required changes
- None blocking.

### Non-blocking suggestions
- Add a one-paragraph "who must believe this" stakeholder note.

### Final note from this lens
Communication quality is a real asset here; the main human risk is dependence on
an external roadmap the team doesn't control.

---

## Documentation Review

### Summary judgment
Approve with comments

### What looks strong
- The plan itself is a strong design doc: vision, deliberate tradeoffs, honest
  limitations, phased exits, and a companion partitioner spec + Apple skills doc.
- Upstream anchors (URLs) in §5.5 make the Apple claim traceable and dated.

### Risks or missing details
- **No decision record** for the llama.cpp→MAX substrate pivot (rationale exists
  in §5.4, but not the alternatives/reversal cost).
- **Placeholders (X/Y) and undefined tolerances** read as unfinished claims.
- **Missing pointers** to the (not-yet-written) format/invariants spec and a
  benchmark methodology doc.
- A worked example and a glossary would materially aid reproducibility.

### Questions for the author/team
- Where will the format spec, benchmark methodology, and decision records live?

### Required changes
- Add a short decision-record section (or link) and replace placeholder metrics
  with numbers or explicit TBD+owner.

### Non-blocking suggestions
- Add a glossary (stage, expert worker, remote-expert, hot/warm/cold) and one
  worked example.

### Final note from this lens
Documentation discipline is above average for pre-code; the gaps are decision
records and the missing format spec that other lenses also demand.

---

## Cross-lens synthesis — prioritized actions

1. **Write the quantitative feasibility example (B1)** — one (model, fleet,
   concurrency) → tok/s, bubbles, remote-expert exposure. Cheapest thing that most
   reduces uncertainty; exercises the cost model before any code. *(Analytical,
   HW-Accel, Hardware)*
2. **Resolve the concurrency↔market question (B2)** — state the buyer's real
   concurrency and show the pipeline fills at it; if it doesn't, the design or the
   target market must change. *(Analytical)*
3. **Specify the cross-vendor format + reference/golden-vector correctness (B3)** —
   the engine's load-bearing invariant; unblocks Low-level + LLM lenses. *(Low-level,
   LLM)*
4. **Add a security/trust model + backpressure (B4)** — required by the privacy
   value prop and basic serving safety. *(Networking)*
5. **Record the substrate decision + Apple Plan B (B5)** with a reversal trigger.
   *(Organizational, Analytical, Hardware)*
6. **High-priority fillers:** hardware matrix + memory budget; tokenizer/
   chat-template/streaming seam; end-to-end lifecycle + observability; fix the
   Phase 0–2 "no GPU" contradiction; fill X/Y/N.

**What not to change:** the honest latency-floor framing (§2), the model-free
planner-first sequencing, the pipeline-spine + bounded-remote-expert decision, and
the explicit-seams surgery discipline. These are the plan's strongest parts and
should survive the revision intact.
