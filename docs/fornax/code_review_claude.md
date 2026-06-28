# Fornax Code Review — Skill-Lens Pass

> Reviewer: Claude (Opus 4.8). Date: 2026-06-23. Branch: `fornax`.
> Method: [review_lenses_by_skill_for_fornax.md](review_lenses_by_skill_for_fornax.md).
> Scope: the `fornax/` Python package (~38 KLOC), planner core, gate validators,
> probes, and simulation modules. This reviews **code as written**, not the
> program-management gate posture (already tracked in
> [fornax_program_management_todo_status.md](../../fornax_program_management_todo_status.md)).

## Overall judgment

**Approve with comments.** This is an unusually disciplined simulation-first
codebase. Input validation is fail-closed everywhere, the accelerator probes do
real device-vs-reference correctness comparison, and the gate validators encode
the project's honesty invariants in code (a packet cannot claim a proxy pass once
its deferred assumptions are edited away — `phase4_resilience_gate.py`,
`phase5_ga_gate.py`). The honesty discipline that Ignis demands is present here.

The material risks are **not** dishonesty — they are (1) a cost model that omits
several memory terms its own target contract calls load-bearing while hardcoding
one of its named success metrics, (2) two monolithic source files that are
becoming unmaintainable, and (3) a proxy strategy that, by construction, flatters
the exact communication-bound regime the plan names as risk #1–#5. All are
fixable inside Phase 0/1.

The single highest-leverage code finding: **`remote_expert_hit_rate_decode` is a
constant `1.0`, not a prediction** (`planner/cost.py:99`), even though §8 of the
plan lists a remote-expert SLO as a success metric.

---

## Analytical Skills Review

### Summary judgment
Needs revision (cost-model fidelity); otherwise Approve with comments.

### What looks strong
- The planner is a real DP partitioner (`planner/search.py:64-93`), not a
  heuristic stub. Replication is a measured improvement loop with a positive-gain
  stopping rule (`search.py:127-170`).
- The proxy-vs-formal boundary is modeled as data, not prose, and validated.

### Risks or missing details
- **Hardcoded "prediction".** `_remote_expert_wait_s` returns hit-rate
  `1.0 if total_active else 0.0` (`planner/cost.py:99`). Any remote-expert plan
  therefore reports `remote_expert_hit_rate_decode = 1.0` regardless of expert
  locality, caching, or co-activation traces — which `ExpertTrace`
  (`planner/model.py:167`) already carries but the cost model never consumes. A
  named §8 success metric is thus not actually modeled. **High.**
- **DP objective ≠ scoring objective.** `_partition_for_order` minimizes
  `max(decode_compute + remote_wait)` and **ignores boundary transfer**
  (`search.py:86`), but the final stage time is
  `max(decode, transfer_in) + remote_wait` (`search.py:113`). The DP returns the
  cut that is optimal for a different cost than the one ultimately scored, so on
  transfer-bound fabrics the chosen partition can be provably suboptimal. **Medium.**
- **Roofline is serial, not overlapped.** `local_decode = weight/bw + kv/bw +
  flops/compute` (`cost.py:148-152`) sums memory time and compute time instead of
  `max()`. Conservative for memory-bound decode, but it means the model never
  credits compute/memory overlap and cannot express a compute-bound stage
  correctly. Name this assumption. **Medium.**
- `compute_class` is a single scalar FLOP/s per node (`model.py:192`) used for
  attention, dense MLP, and expert GEMM alike. The `backend_coverage` matrix that
  distinguishes op classes exists but is **not fed into the cost model**. **Medium.**

### Questions
- What evidence would falsify the planner's throughput prediction, given the proxy
  fabric cannot reproduce the 25–100 GbE regime (see Networking)?
- Should `ExpertTrace` hit-rates drive `remote_hit_rate_decode` before any G2
  calibration claim is made?

### Required changes
- Replace the constant remote hit-rate with a value derived from `ExpertTrace`, or
  rename the field to `remote_expert_active_fraction` so it stops reading as a
  prediction.
- Either fold `transfer_in` into the DP score or document that the partitioner is
  transfer-agnostic by design.

---

## Hardware Review Lens

### Summary judgment
Needs revision (memory budget completeness).

### Risks or missing details
- **Memory budget under-counts vs. the contract it serves.** `stage_memory_bytes`
  sums weights + KV + a fixed activation buffer (`cost.py:36-47`). The activation
  buffer is `2 * concurrency * hidden_dim * dtype` (`cost.py:23-26`) —
  independent of layer count and prompt length. The v0 target contract §3.1
  enumerates per-node budget terms the planner must close with headroom:
  **routing metadata, temporary buffers, runtime/OS reserve, and fragmentation
  margin** — none appear in `stage_memory_bytes`. The feasibility gate
  (`resident_mem <= node.mem_free_bytes`, `cost.py:118`) is therefore optimistic
  relative to the project's own definition of "fits with margin." **High.**
- KV bytes scale only with `concurrency`, not with context length
  (`cost.py:46`) — `prompt_len`/`gen_len` never enter resident KV sizing, though
  `kv_bytes_per_token` is per-token. A 32k vs 64k sensitivity sweep (plan §3.2)
  cannot be expressed by this term as written.

### Required changes
- Add explicit reserve/fragmentation/routing-metadata terms (even as configurable
  fractions) so the planner's "fits" decision matches the contract's budget table.
- Make resident KV a function of context length, not just concurrency.

### Final note
The hardware *probes* are honest about being same-host H100 stand-ins; the gap is
in the *analytical* memory model, not the measurements.

---

## Hardware Acceleration Review Lens

### Summary judgment
Approve with comments.

### What looks strong
- Probes are genuine: the torch path runs the kernel on the CUDA device and
  compares against a CPU float32/float64 reference with `max_abs_error <= tolerance`
  (`accelerator_probe.py:379-384`, `moe_parity.py:571-577`). `cpu-stdlib` backends
  are forbidden from claiming `accelerator_measured` (`moe_parity.py:1108`). This
  is the correct "fast path proven against reference" discipline.
- Timing uses CUDA events with warmup and synchronize (`accelerator_probe.py:388-396`).

### Risks or missing details
- **The proxy flatters the wrong regime.** Same-host CUDA P2P measures ~296 GiB/s
  (journal, 2026-06-23). The plan's v0 fabric is 25–100 GbE (~3–12 GiB/s) — a
  25–100× gap. Every transfer/bubble number measured on this proxy lives in a
  bandwidth regime the product will never see, and the communication-bound risks
  (R1, R3, R5) are precisely the ones the proxy cannot exercise. This is
  disclosed, but it should be elevated from a footnote to a stated limit on what
  the H100 proxy can ever prove. **Medium.**
- No profiler is wired into the dev loop yet (plan §5.10 names this as a
  requirement); timing is wall-clock/event-based only.

### Required changes
- Add a bandwidth-throttle or injected-latency mode to at least one transfer probe
  so the planner's transfer model can be exercised at fabric-realistic rates.

---

## Low-level Software Review Lens

### Summary judgment
Approve with comments.

### What looks strong
- Slow-correct reference paths exist and are compared per-dtype with tolerances
  across runtime-format, stage-host, pipeline, remote-expert, and MoE probes —
  exactly the invariant the lens demands.
- Dataclasses are `frozen=True` with `__post_init__` validation
  (`planner/model.py`), so the planner's data model has real invariants, not
  convention.
- Golden vectors and checksums are first-class (`reference_*_checksum` fields).

### Risks or missing details
- `_with_replicas` recomputes `estimate_stage_cost` for the chosen node a second
  time after selection (`search.py:159`) and re-derives `replica_time`; an
  `assert cost is not None` guards it (`search.py:160`). Asserts can be stripped
  under `-O`; prefer an explicit raise for a load-bearing invariant.
- The `cpu-stdlib` reference MoE/expert math is hand-rolled Python list math
  (`moe_parity.py:_run_reference_layer`). It is the correctness oracle, so it
  deserves its own direct unit tests independent of the probe wrapper (see
  Software Engineering).

### Final note
The boundary discipline (validators reject malformed/stale payloads, plan-hash
required on transport events — `transport.py:30-39`) is genuinely good.

---

## LLM Expertise Review Lens

### Summary judgment
Approve with comments.

### What looks strong
- Tokenizer/chat-template **hash** recording is threaded through the serving and
  target-fixture seams (model-support matrix + `target_fixture_probe`), matching
  plan §5.7. Stop-sequence and non-stream/SSE parity are checked in the local
  endpoint smoke.
- Router top-k → expert bucketing → weighted gather has a reference parity check
  including `next_tokens_match` (`moe_parity.py:366`), so MoE *behavior* is graded,
  not just tensor shapes.

### Risks or missing details
- The decode-time **remote expert hit rate is not modeled** (see Analytical) —
  for an MoE serving engine whose whole thesis is sparse expert placement, the
  hit-rate/SLO loop is the LLM-semantics crux and is currently a constant.
- KV lifecycle in the cost model is sizing-only; eviction/replay semantics live in
  the resilience simulation, not in the planner's KV term. Acceptable for Phase 0,
  but the two KV models should be reconciled before G2.

### Required changes
- Drive remote hit-rate from `ExpertTrace.hit_rate_decode` before any planner-
  accuracy claim (§8) is asserted against real traces.

---

## Networking Review Lens

### Summary judgment
Approve with comments (correctly phased; nothing real yet).

### What looks strong
- The transport contract enumerates a full event lifecycle including
  `backpressure`, `timeout`, `cancel`, `plan_integrity_reject`, `cleanup`
  (`transport.py:14-39`) and requires plan-hash tags on the events that carry
  trust weight.
- The local HTTP smoke exercises real failure mappings: 429 + `Retry-After`,
  409/`request_cancelled`, 504/`backend_timeout`, 503/`network_partition_fenced`,
  and recovery after each (journal, 2026-06-23). Backpressure is deterministic.

### Risks or missing details
- All transport is in-process simulation; mTLS is loopback-only. The plan accepts
  this until Phase 1b, but **no real partition, no real partial failure, and no
  cross-host backpressure** has been exercised — and the same-host proxy can't
  produce them. B4 remains spec+sim.
- Communication cost is in the planner's transfer term but, as above, never
  validated at fabric-realistic bandwidth.

### Final note
The phasing is honest and the local failure-semantics coverage is better than
most prototypes. The gap is inherent to the proxy, not to the code.

---

## Software Engineering Review Lens

### Summary judgment
Needs revision (module size + test topology).

### Risks or missing details
- **Monoliths.** `fornax/cli.py` is **3,681 lines / 163 KB with ~178 handlers**;
  `fornax/local_http_serving_smoke.py` is **~189 KB** mixing an HTTP server, TLS
  setup, four probe integrations, and validators. These are past the point where a
  new contributor can safely change them. The lens's "everything in one place" red
  flag applies. **High.**
- **Test topology.** 243 tests is healthy, but they live in a single
  `tests/test_fornax_planner.py` (4,188 lines) that imports nearly every module.
  The planner is genuinely unit-tested; most other modules (transport, moe,
  resilience, serving) are exercised mainly through their own validators + CLI
  smoke, not independent unit tests. The correctness oracles (`cpu-stdlib`
  references) especially need direct tests. **Medium.**
- No per-test runner / module test files means a change to `moe.py` reruns the
  whole planner suite to get signal.

### Required changes
- Split `cli.py` into per-command-group modules behind a thin dispatcher; extract
  the HTTP server and TLS plumbing out of `local_http_serving_smoke.py`.
- Add `tests/test_<module>.py` files for the non-planner modules, starting with the
  reference oracles.

### Non-blocking suggestions
- The repeated `_positive_int/_non_empty_string/_number_field` validator helpers
  are copy-pasted across ~15 modules; promote to one `fornax/_validate.py`.

---

## System Engineering Review Lens

### Summary judgment
Approve with comments.

### What looks strong
- The end-to-end lifecycle (client → serving → admission → stages → MoE → sampler
  → stream → cleanup) is represented and partially exercised in one local bundle
  (`local_http_serving_smoke.py`), and lifecycle resource alloc/release counts are
  asserted (journal). That is real integration, not isolated demos.
- Observability fields (request/plan IDs, per-stage timings, placement
  explanations) are threaded as data and validated.

### Risks or missing details
- Two parallel models of the same concept can drift: KV sizing (planner) vs KV
  eviction/replay (resilience); transport cost (planner) vs transport events
  (transport sim). No single test asserts they agree.
- The "logical host" abstraction is doing a lot of load-bearing work across
  modules; a one-line shared definition of what a logical host may and may not
  prove would prevent any module from over-claiming.

---

## Documentation Review Lens

### Summary judgment
Approve with comments.

### What looks strong
- `fornax_development_journal.md` (2,684 lines) is exceptional institutional
  memory, with explicit per-lens review notes and a repeated, honest "boundary
  remains explicit" disclaimer on every proxy artifact.
- Limitations are written honestly and prominently — the hardest thing to get
  right, and it is right here.

### Risks or missing details
- The five load-bearing G1 artifacts (`v0-target-contract.md`,
  `runtime-format-and-invariants.md`, `networking-security-and-backpressure.md`,
  `adr/0001-max-mojo-substrate.md`, Apple probe) still exist only as draft
  *generators* in code, not as reviewed docs (todo S0-2..S0-7). That is the
  correct status, but it means the code's invariants (e.g. memory terms, hit-rate
  semantics) are not yet pinned by a reviewed spec.
- No module-level README maps the ~50 `fornax/*.py` files to WBS work-streams; the
  mapping currently lives only in the todo file and journal.

### Required changes
- Add a short `fornax/README.md` mapping modules → WBS streams → owning lens.

---

## Cross-lens required-changes (ranked)

1. **High** — Drive `remote_expert_hit_rate_decode` from `ExpertTrace`, or rename
   it; today it is a constant `1.0` masquerading as a metric (`cost.py:99`).
2. **High** — Add routing-metadata / temp-buffer / OS-reserve / fragmentation
   terms to `stage_memory_bytes`; make resident KV context-length aware
   (`cost.py:36-47`).
3. **High** — Split `cli.py` and `local_http_serving_smoke.py`; they are over the
   maintainability cliff.
4. **Medium** — Reconcile the DP partition objective with the final stage-time
   score (transfer-awareness) (`search.py:86` vs `:113`).
5. **Medium** — Add a bandwidth-throttle mode to a transport probe so the planner's
   communication model can be tested outside the 296 GiB/s same-host regime.
6. **Medium** — Add per-module unit tests, beginning with the `cpu-stdlib`
   correctness oracles; replace the load-bearing `assert` in `search.py:160`.

## What does not need changing

- The honesty architecture (deferred-requirement validators, `formal_*_passed`
  invariants) is correct and should be preserved exactly.
- The probe-vs-reference correctness pattern is correct.
- The phased deferral of real transport/security is the right call, not a gap.
