# Fornax Development Journal

## 2026-06-20

- Created local branch `fornax` from `dev-agents`.
- Re-read the Phase-0 program-management docs, project plan v3, partitioner spec,
  review lenses, and the referenced venv/cache notes.
- Current governing constraint: Phase 0 only. G1 has not passed, so distributed
  worker/runtime/transport engineering is intentionally out of scope.
- Started S0-1 with a model-free Python package:
  - `fornax.planner` datamodels for `ModelSpec`, `Inventory`, `Target`, and
    `PlacementPlan`;
  - cost model for resident and remote-expert stage modes;
  - deterministic contiguous-stage search with basic topology and replica
    handling;
  - packaged golden-plan fixtures and `fornax test golden-plans`;
  - stdlib `unittest` coverage for golden fixtures, infeasibility, slow-node
    partitioning, and monotonicity checks;
  - preflight CLI skeleton for inventory, fabric probe, target validation,
    planning, simulation, dry benchmark, and doctor bundle checks.
- Implementation stance: Python/stdout JSON is deliberate for Phase 0 because
  the partitioner spec says it is pure logic and must run without model or GPU.
  Mojo/MAX runtime work remains gated by G1.

### Milestone review-lens pass: S0-1 initial planner

- Software Engineering: approve with comments. The slice is scoped to Phase-0
  pure logic, with separate modules for datamodels, cost model, search, fixtures,
  CLI, and tests. Test gaps to close next: more hand-worked cost cases, replica
  edge cases, and remote-expert trace behavior.
- Analytical: approve with comments. The planner exposes the right throughput,
  latency, bubble, and infeasibility outputs, but the current numbers are still
  synthetic predictions. G1 evidence will require measured probe inputs and a
  tighter calibration ledger before any throughput claim is treated as real.
- Hardware: approve with comments. Hardware is explicit in `Inventory` rather
  than hard-coded, and dry benchmarks mark `measured=false`. Gap: the local
  inventory command is a conservative CPU placeholder, not a real GPU/NIC probe.
- Low-level Software: approve with comments. This milestone deliberately avoids
  runtime memory ownership and kernel boundaries. The remaining B3 artifact,
  `runtime-format-and-invariants.md`, is still open and must precede Phase 1.
- High-level Software: approve with comments. The CLI names match the Phase-0
  preflight workflow. Gap: `target validate` currently consumes a JSON target
  bundle, while the G1 artifact is a markdown `v0-target-contract.md`; a contract
  parser/validator is needed before S0-2 can close.
- Important issue found and fixed: the slow-node golden fixture originally did
  not force a spanning plan, so the optimizer legally selected one stage. Added
  fixture `plan_options` support and pinned that case to exactly two stages.

### Contract parser follow-up

- Addressed the high-level review gap from the first milestone: `fornax target
  validate` now accepts a markdown target contract containing a fenced
  `json fornax-target` machine-readable block, in addition to bare JSON bundles.
- Added `fornax.contracts.TargetContractError` with actionable parse errors when
  the machine-readable block is missing or malformed.
- Added a packaged markdown fixture and tests so S0-2 can evolve toward the real
  `docs/fornax/v0-target-contract.md` artifact without making the CLI depend on
  oral context.
- Verification: `python3 -m unittest discover -s tests -p 'test_fornax*.py'`,
  `python3 -m fornax test golden-plans`, `python3 -m compileall -q fornax tests`,
  and direct markdown `fornax target validate` all passed.

### Inventory collection milestone

- Addressed the Hardware review gap from the initial planner milestone: `fornax
  inventory collect` now calls `nvidia-smi` when available and emits NVIDIA GPU
  nodes alongside the CPU node.
- Live smoke check on this machine wrote `/tmp/fornax_live_inventory.json` and
  discovered three nodes: the CPU host plus `gpu0` and `gpu1`, both `NVIDIA H100
  80GB HBM3`. Current free-memory-derived planner budgets were about 35.8 GB for
  `gpu0` and 27.7 GB for `gpu1` after the 10% reserve; no collection errors.
- Honesty invariant: GPU memory fields are marked measured from `nvidia-smi`, but
  `compute_class` and `mem_bandwidth_bytes_s` are still labeled static estimates
  until profiler/benchmark probes calibrate them. A review issue found during
  this milestone was fixed: `measured_fields` is now dynamic and only lists
  NVIDIA fields when NVIDIA rows were actually parsed.
- Added unit coverage for the `nvidia-smi` CSV parser, H100 node materialization,
  planner memory reserve, and malformed CSV rejection.
- Verification: `python3 -m unittest discover -s tests -p 'test_fornax*.py'`,
  `python3 -m fornax test golden-plans`, `python3 -m compileall -q fornax tests`,
  `make fornax-test`, `make fornax-golden`, and live `fornax inventory collect`
  all passed.

### Simulate command-contract alignment

- Aligned the implemented CLI with the Phase-0/T0 command contract by adding
  `fornax simulate --plan placement.json --requests synthetic_trace.json` while
  preserving `--trace` as a deprecated alias.
- Added `fornax.simulate` request-trace parsing for simple JSON traces shaped as
  either a list of request objects or `{ "requests": [...] }`. The summary now
  records request count, total prompt tokens, total generation tokens, and a
  planner-predicted decode wall time from the plan throughput.
- Added tests for request trace parsing, bad trace shape rejection, and simulation
  result calculation.
- Verification: `python3 -m unittest discover -s tests -p 'test_fornax*.py'`,
  `python3 -m fornax test golden-plans`, `python3 -m compileall -q fornax tests`,
  and a direct `python3 -m fornax simulate --plan /tmp/fornax_preflight/placement.json
  --requests /tmp/fornax_requests.json --out /tmp/fornax_preflight/simulate_requests.json`
  smoke check all passed.

### Target-contract validation milestone

- Added executable S0-2/S0-3 checks in `fornax.validation`: target contracts now
  validate required G1-facing metadata (`seed_target_rationale`, named baselines,
  kill metric, throughput threshold, memory-headroom threshold, concurrency sweep,
  and persona concurrency) in addition to planner feasibility.
- `fornax target validate` now runs the planner and emits named pass/fail checks,
  predicted throughput, and per-stage memory headroom instead of only reporting
  whether a placement exists.
- Updated the packaged markdown target-contract fixture with a `contract` block
  so it exercises the validation path expected for `docs/fornax/v0-target-contract.md`.
- Added tests for successful fixture validation, missing gate metadata, and unmet
  throughput threshold.
- Review-lens pass:
  - Analytical: approve with comments. This makes the contract falsifiable against
    the current planner, but the throughput threshold is still a planner prediction;
    real G1 evidence still needs measured probe inputs and benchmark baselines.
  - Hardware: approve with comments. Memory headroom is now computed from the
    inventory and placement, but measured links and calibrated device throughput
    remain open.
  - High-level Software: approve. CLI output names the failing checks, which is
    actionable for contract authors.
  - Software Engineering: approve after cleanup. Readability issues in the new
    validator were fixed before commit.
- Verification: `python3 -m unittest discover -s tests -p 'test_fornax*.py'`,
  `python3 -m fornax test golden-plans`, `python3 -m compileall -q fornax tests`,
  `make fornax-test`, `make fornax-golden`, and direct markdown `fornax target
  validate` all passed.

### Structured doctor milestone

- Replaced the placeholder `fornax doctor` check with structured Phase-0 evidence
  bundle diagnostics in `fornax.doctor`.
- Doctor now checks required JSON artifacts (`inventory.json`, `links.json`,
  `placement.json`), recommended artifacts (`target.json` or
  `v0-target-contract.md`, `validate.json`, `simulate.json`, `benchmark.json`),
  placement feasibility, validation status, simulation predicted block, and dry-run
  benchmark status.
- Review-lens pass:
  - High-level Software: approve. The command now reports actionable errors and
    warnings for the operator instead of only listing missing filenames.
  - Software Engineering: approve after fix. Initial implementation accepted an
    empty `inventory.json`; fixed before commit by requiring at least one node and
    validating that `links.json` contains a links list.
  - Analytical: approve with comments. Doctor distinguishes errors from warnings
    and explicitly warns when benchmark evidence is only a dry-run prediction;
    measured benchmark-of-record evidence remains open for G1.
- Verification: `python3 -m unittest discover -s tests -p 'test_fornax*.py'`,
  `python3 -m fornax test golden-plans`, `python3 -m compileall -q fornax tests`,
  `make fornax-test`, `make fornax-golden`, and structured `fornax doctor` smoke
  against `/tmp/fornax_preflight` all passed.

### Runtime-format command-contract milestone

- Implemented the missing T0/T1 command `fornax test runtime-format --golden
  golden_vectors/` using `fornax.runtime_format` and a packaged golden-vector
  manifest under `fornax/golden_vectors/runtime_format/manifest.json`.
- The validator now checks the Phase-0 runtime-format invariants that can be
  enforced before runtime implementation: activation dtype/shape/layout/payload
  length, KV page dtype/page size/token ownership bounds, expert-batch routing
  and gather permutation consistency, and presence of per-dtype tolerances.
- Added tests for the passing fixture plus negative controls for activation
  payload length, invalid expert gather order, weight-only activation dtype, and
  KV page shape/page-size mismatch.
- Review-lens pass:
  - Low-level Software: approve after fixes. Initial implementation accepted
    `q4/q8` as runtime payload dtypes and only required KV shape to cover tokens;
    fixed before commit by restricting payload dtypes to float runtime dtypes and
    requiring KV shape[0] to equal `page_size`.
  - LLM/correctness: approve with comments. This is a schema/golden-vector
    validator, not numerical logit parity or a slow reference path. The full
    `runtime-format-and-invariants.md` B3 artifact and optimized-vs-reference
    parity still remain open.
  - Software Engineering: approve. The command is pure model-free CI and keeps
    dispatch compatible with existing `fornax test golden-plans`.
- Verification: `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m unittest discover -s tests
  -p 'test_fornax*.py'`, `python3 -m fornax test golden-plans`, `python3 -m
  compileall -q fornax tests`, `make fornax-test`, and `make fornax-golden` all
  passed.

### Simulated network-contract command milestone

- Implemented the missing T1 command `fornax test network-contract --mode
  simulated` using `fornax.network_contract` and a packaged fixture under
  `fornax/golden_vectors/network_contract/simulated.json`.
- The validator checks the Phase-0 simulated transport contract: bounded queue
  depth, backpressure event at capacity, timeout threshold, cancellation event,
  plan-integrity rejection for mismatched plan IDs, and required enqueue/dequeue
  flow.
- Added tests for the passing fixture plus missing required events, queue
  overflow, and dequeueing an unknown request.
- Review-lens pass:
  - Networking/System: approve after fix. Initial implementation tracked only
    queue depth; fixed before commit by tracking request IDs through
    enqueue/dequeue/cancel and rejecting duplicate/unknown queue operations.
  - Software Engineering: approve. The command is model-free and does not create
    a Phase-1 transport implementation.
  - Analytical: approve with comments. This proves contract semantics in a
    fixture, not measured TCP/RDMA/Thunderbolt behavior; real transport evidence
    remains Phase 1+ and gated by G1.
- Verification: `python3 -m fornax test network-contract --mode simulated
  --fixture fornax/golden_vectors/network_contract`, `python3 -m unittest
  discover -s tests -p 'test_fornax*.py'`, `python3 -m compileall -q fornax
  tests`, `python3 -m fornax test golden-plans`, `python3 -m fornax test
  runtime-format --golden fornax/golden_vectors/runtime_format`, `make
  fornax-test`, and `make fornax-golden` all passed.

### Measured benchmark plumbing milestone

- Replaced the dry-run `fornax benchmark` artifact with a measured deterministic
  CPU tiny expert-MLP microbenchmark in `fornax.benchmark` while preserving the
  planner prediction under a separate `plan_predicted` field.
- The benchmark artifact now records `measured=true`, elapsed time, tokens/sec,
  expert-calls/sec, checksum, benchmark config, Python/platform environment, and
  a note that this is Phase-0 benchmark plumbing rather than target-model, MAX,
  or accelerator throughput evidence.
- `fornax doctor` now accepts the temporary evidence bundle without the previous
  dry-run benchmark warning when `benchmark.json` comes from the measured path.
- Review-lens pass:
  - Analytical: approve with comments. The number is traceable to an actual
    measurement and checksum, but it is not a G1 throughput claim for the target
    model or fleet.
  - Hardware: approve with comments. This is CPU stdlib measurement only; H100/MAX
    and fabric-calibrated probes remain open evidence work.
  - Software Engineering: approve after fix. Initial measured loop regenerated
    deterministic expert weights inside the timed section; fixed before commit by
    precomputing weights before timing and recording that in the config.
- Verification: `python3 -m fornax benchmark --plan /tmp/fornax_preflight/placement.json
  --mode tiny-moe-or-expert-mlp --iterations 2 --out /tmp/fornax_preflight/benchmark_measured.json`,
  `python3 -m fornax doctor --bundle /tmp/fornax_preflight`, `python3 -m unittest
  discover -s tests -p 'test_fornax*.py'`, `python3 -m compileall -q fornax tests`,
  `python3 -m fornax test golden-plans`, `python3 -m fornax test runtime-format
  --golden fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture fornax/golden_vectors/network_contract`,
  `make fornax-test`, and `make fornax-golden` all passed.

### Phase-0 preflight bundle milestone

- Added `fornax.preflight.run_phase0_preflight` and the `fornax preflight`
  command to produce the standard Phase-0 evidence bundle without oral context:
  target contract copy, `inventory.json`, `links.json`, `placement.json`,
  `validate.json`, `simulate.json`, `benchmark.json`, and `doctor.json`.
- Improved `fornax fabric probe` for local development inventories: declared
  links are preserved and same-host CPU/GPU/GPU links are synthesized as
  conservative topology-derived estimates so the two-H100 workstation produces a
  usable planner topology. The artifact and `fornax doctor` warnings still
  record that these are not active fabric measurements and must be replaced
  before G1 throughput sign-off.
- Review-lens pass:
  - High-level Software: approve. A single command now captures the documented
    Phase-0 workflow and writes predictable artifact names for review and
    handoff.
  - Hardware: approve with comments. The local link records are honest estimates,
    not fabric measurements. Active bandwidth/latency probing remains required
    for G1 target throughput claims.
  - SRE/Operations: approve with comments. The bundle is doctorable and
    reproducible, but richer operator documentation and remote-node probing are
    still open.
- Verification: `python3 -m fornax preflight --target fornax/golden_plans/v0_target_contract_fixture.md --out-dir /tmp/fornax_preflight_cmd --requests /tmp/fornax_preflight_cmd_requests.json --benchmark-iterations 1`,
  idempotent rerun against `/tmp/fornax_preflight_cmd/v0-target-contract.md`,
  `python3 -m fornax doctor --bundle /tmp/fornax_preflight_cmd` (ok with
  estimated-link warnings), `python3 -m unittest discover -s tests -p
  'test_fornax*.py'`, `python3 -m compileall -q fornax tests`, `python3 -m
  fornax test golden-plans`, `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture
  fornax/golden_vectors/network_contract`, `make fornax-test`, and `make
  fornax-golden` all passed.

### Target-contract draft milestone

- Added `fornax.target_contract.render_target_contract_draft` and the `fornax
  target draft` command to turn an executable seed/source contract plus inventory
  and links into a human-reviewable `v0-target-contract.md` draft.
- The draft includes explicit non-signoff status, seed/replacement rationale,
  target summary, fleet summary, fabric provenance, planner prediction, memory
  budget rows, thresholds, baselines, kill metric, validation checks, and the
  executable `json fornax-target` block with evidence metadata.
- Review-lens pass:
  - Analytical: approve with comments. The draft records planner predictions and
    validation checks, but it is still a draft; G1 closure requires TL/SP review,
    measured fabric/device evidence, and final threshold agreement.
  - Hardware: approve with comments. Fleet and fabric provenance are exposed in
    the draft; estimated links remain visible and cannot be mistaken for active
    fabric measurements.
  - High-level Software: approve. Contract authors now have a repeatable command
    for generating the review artifact from the same machine-readable source used
    by `target validate` and `preflight`.
- Verification: `python3 -m fornax inventory collect --out /tmp/fornax_target_draft_inventory.json`,
  `python3 -m fornax fabric probe --inventory /tmp/fornax_target_draft_inventory.json --out /tmp/fornax_target_draft_links.json`,
  `python3 -m fornax target draft fornax/golden_plans/v0_target_contract_fixture.md --inventory /tmp/fornax_target_draft_inventory.json --links /tmp/fornax_target_draft_links.json --out /tmp/fornax_v0_target_contract_draft.md`,
  `python3 -m fornax target validate /tmp/fornax_v0_target_contract_draft.md --inventory /tmp/fornax_target_draft_inventory.json --links /tmp/fornax_target_draft_links.json`,
  generated-contract `fornax preflight`, `python3 -m unittest discover -s tests
  -p 'test_fornax*.py'`, `python3 -m compileall -q fornax tests`,
  `python3 -m fornax test golden-plans`, `python3 -m fornax test
  runtime-format --golden fornax/golden_vectors/runtime_format`, `python3 -m
  fornax test network-contract --mode simulated --fixture
  fornax/golden_vectors/network_contract`, `make fornax-test`, and `make
  fornax-golden` all passed.

### Runtime-format spec draft milestone

- Added `fornax.runtime_format_spec.render_runtime_format_spec_draft` and the
  `fornax spec runtime-format` command to generate a reviewable
  `runtime-format-and-invariants.md` draft from the packaged golden-vector
  manifest.
- The draft covers activation tensor layout/ownership/lifetime, KV page
  ownership/eviction/transfer/replay rules, expert batch routing/gather
  semantics, in-flight dtype restrictions, quantization boundary, failure modes,
  reference path, golden-vector method, per-dtype tolerances, and build/toolchain
  assumptions.
- Review-lens pass:
  - Low-level Software: approve with comments. The spec makes ownership and
    failure semantics explicit, but optimized backend parity and binary layout
    checks remain future hardware/runtime work.
  - LLM/correctness: approve. The draft reinforces the slow reference path,
    golden vectors, and per-dtype tolerance discipline required before optimized
    cross-vendor trust.
  - High-level Software: approve. The review artifact can now be regenerated from
    the same golden fixture used by `fornax test runtime-format`.
- Verification: `python3 -m fornax spec runtime-format --golden
  fornax/golden_vectors/runtime_format --out
  /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m unittest
  discover -s tests -p 'test_fornax*.py'`, `python3 -m compileall -q
  fornax tests`, `python3 -m fornax test golden-plans`, `python3 -m
  fornax test runtime-format --golden fornax/golden_vectors/runtime_format`,
  `python3 -m fornax test network-contract --mode simulated --fixture
  fornax/golden_vectors/network_contract`, `make fornax-test`, and `make
  fornax-golden` all passed.

### Networking-security spec draft milestone

- Added `fornax.network_security_spec.render_network_security_spec_draft` and
  the `fornax spec network-security` command to generate a reviewable
  `networking-security-and-backpressure.md` draft from the simulated
  network-contract fixture.
- The draft covers the Phase-0/Phase-1a trust boundary, node identity, endpoint
  auth posture, plan-integrity rejection, bounded queue/backpressure behavior,
  timeout/retry/cancel/partition semantics, simulated event trace, phased
  implementation requirements, transport posture, and lab/product security
  separation.
- Review-lens pass:
  - Networking/System: approve with comments. The spec makes T1 semantics and
    later phase requirements explicit, but it is still simulated evidence rather
    than transport or product-security implementation.
  - Software Engineering: approve. The draft is generated from the same fixture
    used by `fornax test network-contract`, keeping spec text tied to executable
    checks.
  - Security/Product: approve with comments. Default-deny, endpoint auth,
    encryption posture, and audit logs are called out for product deployment;
    Phase 0 still uses lab/simulation boundaries only.
- Verification: `python3 -m fornax spec network-security --fixture
  fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `python3 -m
  unittest discover -s tests -p 'test_fornax*.py'`, `python3 -m compileall
  -q fornax tests`, `python3 -m fornax test golden-plans`, `python3 -m
  fornax test runtime-format --golden fornax/golden_vectors/runtime_format`,
  `python3 -m fornax test network-contract --mode simulated --fixture
  fornax/golden_vectors/network_contract`, `make fornax-test`, and `make
  fornax-golden` all passed.

### Substrate ADR draft milestone

- Added `fornax.substrate_adr.render_substrate_adr_draft` and the `fornax spec
  substrate-adr` command to generate a reviewable
  `adr/0001-max-mojo-substrate.md` draft.
- The draft records MAX/Mojo as the preferred substrate bet while keeping the
  decision auditable: source precedence ladder, pinned build policy, local
  rank-1 probe as gate of record, rejected alternatives, Apple Plan B, reversal
  trigger, watch-register fields, and review checklist.
- The command also records local `max`/`mojo`/`modular` tool discovery as
  provenance only. On this Linux H100 workstation, no local MAX/Mojo tools were
  discovered, so the generated draft correctly warns that Apple expert-MLP
  capability remains unproven.
- Review-lens pass:
  - Organizational/TL: approve with comments. The ADR shape now exists and names
    the gate rules, but final TL/SP review and dated decision recording remain
    required before G1 closure.
  - Hardware Acceleration: approve with comments. The draft does not infer Apple
    support from upstream claims; target-Mac expert-MLP measurement on a pinned
    build remains the rank-1 gate.
  - Low-level Software: approve. Rejected alternatives and the MAX/Mojo surgery
    boundary are explicit, while unsupported operations remain backend gaps
    rather than assumptions.
- Verification: `python3 -m fornax spec substrate-adr --out
  /tmp/fornax_adr_0001_default.md`, `python3 -m fornax spec substrate-adr
  --pinned-build max-26.4.0 --last-checked 2026-06-20 --status probing
  --apple-role capacity-only --out /tmp/fornax_adr_0001_max_mojo_substrate.md`,
  `python3 -m unittest discover -s tests -p 'test_fornax*.py'`, `python3 -m
  compileall -q fornax tests`, `python3 -m fornax test golden-plans`, `python3
  -m fornax test runtime-format --golden fornax/golden_vectors/runtime_format`,
  `python3 -m fornax test network-contract --mode simulated --fixture
  fornax/golden_vectors/network_contract`, `python3 -m fornax spec
  runtime-format --golden fornax/golden_vectors/runtime_format --out
  /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed.

### Apple expert-MLP probe tooling milestone

- Added `fornax.apple_probe` and the `fornax apple` CLI group with three Phase-0
  commands: `probe-template`, `validate-probe`, and `role-decision`.
- The probe artifact captures the S0-7 rank-1 evidence requirements: target
  model/expert shape, quantization, activation dtype, Apple hardware/OS,
  pinned MAX/Mojo builds, exact command/log path, local target-Mac measurement,
  correctness tolerance results, throughput threshold, iterations, thermals, and
  role-decision fields.
- Validator semantics now match the G1 gate: an unmeasured/incomplete template is
  invalid and non-closable; measured correctness+throughput pass recommends
  `expert-worker`; measured correctness or throughput miss remains valid evidence
  and recommends `capacity-only` demotion.
- Smoke validation used a synthetic measured demotion artifact under `/tmp` to
  exercise the CLI without claiming this Linux H100 workstation ran the target
  Mac probe. The real Apple expert-MLP measurement remains open until run on a
  target Apple Silicon Mac with a pinned build.
- Review-lens pass:
  - Hardware Acceleration: approve with comments. The tool enforces rank-1 local
    target-Mac evidence shape, but no Apple hardware measurement was performed
    in this environment.
  - Analytical: approve. Pass, demotion, and incomplete states are distinct, so
    a throughput miss cannot be misreported as success while still allowing G1
    to narrow scope.
  - Organizational/TL: approve with comments. The role-decision draft makes the
    Sponsor decision input reproducible, but final G1 closure still requires
    actual target-Mac evidence or explicit Sponsor narrowing.
- Verification: `python3 -m fornax apple probe-template --target-model
  qwen3-moe-target --pinned-build max-26.4.0 --threshold-tokens-s 10 --out
  /tmp/fornax_apple_probe_template.json`, `python3 -m fornax apple
  validate-probe /tmp/fornax_apple_probe_measured_demote.json --out
  /tmp/fornax_apple_probe_measured_demote_validation.json`, `python3 -m fornax
  apple role-decision --probe /tmp/fornax_apple_probe_measured_demote.json --out
  /tmp/fornax_apple_role_decision_demote.md`, expected-failing unmeasured
  template validation, `python3 -m unittest discover -s tests -p
  'test_fornax*.py'`, `python3 -m compileall -q fornax tests`, `python3 -m
  fornax test golden-plans`, `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture fornax/golden_vectors/network_contract`,
  `python3 -m fornax spec runtime-format --golden fornax/golden_vectors/runtime_format
  --out /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed, except the unmeasured template validation
  intentionally exited 2.

### Roadmap rebaseline and staffing-answer tooling milestone

- Added `fornax.program_rebaseline.render_program_rebaseline_draft` and the
  `fornax program rebaseline` command to generate the S0-8 roadmap/staffing
  review artifact.
- The draft records kickoff date, Phase-0 sprint length, KER staffing status,
  Sponsor scope status, milestone date ranges, Phase-0 role matrix, staffing
  answer, G1 decision-input table, critical-path interpretation, procurement
  actions, and review checklist.
- Generator semantics preserve the gate posture: Phase 1+ dates remain
  placeholders, `DEC-005` is not claimed, and KER-unavailable status warns that
  G1 must choose NARROW, ITERATE-to-staff, or KILL rather than silently proceed.
- Review-lens pass:
  - Program Management: approve with comments. The artifact makes S0-8
    reproducible, but named human assignees and Sponsor scope choice remain
    external decisions.
  - Organizational/TL: approve. The draft restates that unstaffed KER blocks
    silent PROCEED and forces an explicit gate outcome.
  - Budget/Procurement: approve with comments. It calls out `desktop-minimal`
    confirmation and keeps larger spend behind G1 unless Sponsor records a DEC.
- Verification: `python3 -m fornax program rebaseline --kickoff-date 2026-06-20
  --ker-status unavailable --scope pending --out
  /tmp/fornax_roadmap_staffing_rebaseline.md`, `python3 -m unittest discover -s
  tests -p 'test_fornax*.py'`, `python3 -m compileall -q fornax tests`,
  `python3 -m fornax test golden-plans`, `python3 -m fornax test runtime-format
  --golden fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture fornax/golden_vectors/network_contract`,
  `python3 -m fornax apple validate-probe
  /tmp/fornax_apple_probe_measured_demote.json --out
  /tmp/fornax_apple_probe_measured_demote_validation.json`, `python3 -m fornax
  spec runtime-format --golden fornax/golden_vectors/runtime_format --out
  /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed.

### G1 artifact-aware doctor milestone

- Extended `fornax.doctor.inspect_phase0_bundle` so Phase-0 bundles report the
  newer G1 gate artifacts, not only inventory/links/placement/validation/
  simulation/benchmark files.
- Doctor now tracks runtime-format spec, networking/security spec, substrate ADR,
  Apple probe artifact, Apple probe validation, Apple role decision, and roadmap
  staffing rebaseline artifacts. Missing items are warnings, not core preflight
  errors, so a minimal bundle remains usable while still exposing G1 gaps.
- `apple-probe-validation.json` is parsed when present; a closable validation
  records the recommended Apple role, while a non-closable validation warns.
- Review-lens pass:
  - SRE/Operations: approve. A single doctor report now distinguishes runnable
    preflight health from missing G1 review artifacts.
  - Program Management: approve. The report reduces status drift by making
    written-but-not-closed artifacts visible in the same bundle summary.
  - Software Engineering: approve with comments. The change is warning-only for
    new gate artifacts to avoid turning the minimal preflight command into a
    hard dependency on human-reviewed markdown.
- Verification: focused doctor/preflight tests, `python3 -m unittest discover -s
  tests -p 'test_fornax*.py'`, `python3 -m compileall -q fornax tests`, `python3
  -m fornax preflight --target fornax/golden_plans/v0_target_contract_fixture.md
  --out-dir /tmp/fornax_preflight_doctor_g1 --benchmark-iterations 1`, `python3
  -m fornax doctor --bundle /tmp/fornax_preflight_doctor_g1 --out
  /tmp/fornax_preflight_doctor_g1/doctor_rerun.json`, `python3 -m fornax test
  golden-plans`, `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture fornax/golden_vectors/network_contract`,
  `python3 -m fornax spec runtime-format --golden fornax/golden_vectors/runtime_format
  --out /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed.

### Preflight G1 draft bundle milestone

- Added optional `fornax preflight --include-g1-drafts` support so a single
  preflight run can materialize the generated G1 review drafts into the evidence
  bundle before `fornax doctor` runs.
- When enabled, preflight now writes `runtime-format-and-invariants.md`,
  `networking-security-and-backpressure.md`, `adr/0001-max-mojo-substrate.md`,
  `apple-expert-mlp-probe.json`, and `roadmap-staffing-rebaseline.md`, alongside
  the existing target/inventory/links/placement/validate/simulate/benchmark/
  doctor artifacts.
- Doctor warnings on the generated bundle shrink to the honest remaining gaps:
  active link estimates on this workstation plus missing measured
  `apple-probe-validation.json` and `apple-role-decision.md` artifacts.
- Review-lens pass:
  - SRE/Operations: approve. The richer bundle is reproducible with one command
    and no manual copying of draft artifacts.
  - Program Management: approve. The bundle keeps written draft artifacts next to
    executable evidence, reducing handoff ambiguity while preserving warnings for
    unmeasured Apple evidence.
  - Software Engineering: approve with comments. Draft generation is opt-in, so
    the minimal preflight path remains available for fast T0/T1 checks.
- Verification: focused preflight tests, `python3 -m unittest discover -s tests
  -p 'test_fornax*.py'`, `python3 -m compileall -q fornax tests`, `python3 -m
  fornax preflight --target fornax/golden_plans/v0_target_contract_fixture.md
  --out-dir /tmp/fornax_preflight_with_g1 --benchmark-iterations 1
  --include-g1-drafts --substrate-pinned-build max-26.4.0 --kickoff-date
  2026-06-20 --ker-status unavailable --scope pending`, `python3 -m fornax
  doctor --bundle /tmp/fornax_preflight_with_g1 --out
  /tmp/fornax_preflight_with_g1/doctor_rerun.json`, `python3 -m fornax test
  golden-plans`, `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture fornax/golden_vectors/network_contract`,
  `python3 -m fornax spec runtime-format --golden fornax/golden_vectors/runtime_format
  --out /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed.

### Local calibration artifact milestone

- Added `fornax.calibration.run_local_calibration` and the `fornax calibrate
  local` command to produce a Phase-0 calibration artifact with measured CPU
  memory-copy and scalar-compute probes plus local inventory/H100 provenance.
- The calibration artifact optionally attempts a torch CUDA microprobe when torch
  is already installed. On this workstation, PyTorch/NumPy/CuPy/Numba are not
  installed, so the artifact correctly records two discovered H100 GPUs but no
  measured CUDA microprobe rather than fabricating GPU throughput.
- `fornax preflight --include-calibration` now writes `calibration.json`, and
  `fornax doctor` parses it, records measured status/CUDA measured status, and
  surfaces calibration warnings without making CPU plumbing evidence look like a
  target-model throughput claim.
- Review-lens pass:
  - Analytical: approve with comments. CPU probes are measured and checksummed,
    but remain calibration plumbing; target-model and active fabric calibration
    remain G1 evidence gaps.
  - Hardware: approve with comments. H100 discovery is recorded from `nvidia-smi`,
    but no CUDA microprobe was measured because no CUDA Python backend is
    installed in the current environment.
  - Software Engineering: approve. The calibration path is dependency-light by
    default and optional-GPU by capability detection rather than a hard torch
    dependency.
- Verification: `python3 -m fornax calibrate local --out
  /tmp/fornax_local_calibration.json --cpu-memory-bytes 1048576
  --cpu-memory-iterations 2 --cpu-compute-iterations 10000`, `python3 -m fornax
  preflight --target fornax/golden_plans/v0_target_contract_fixture.md --out-dir
  /tmp/fornax_preflight_with_calibration --benchmark-iterations 1
  --include-g1-drafts --include-calibration --substrate-pinned-build max-26.4.0
  --kickoff-date 2026-06-20 --ker-status unavailable --scope pending`, `python3
  -m fornax doctor --bundle /tmp/fornax_preflight_with_calibration --out
  /tmp/fornax_preflight_with_calibration/doctor_rerun.json`, `python3 -m
  unittest discover -s tests -p 'test_fornax*.py'`, `python3 -m compileall -q
  fornax tests`, `python3 -m fornax test golden-plans`, `python3 -m fornax test
  runtime-format --golden fornax/golden_vectors/runtime_format`, `python3 -m
  fornax test network-contract --mode simulated --fixture
  fornax/golden_vectors/network_contract`, `python3 -m fornax spec
  runtime-format --golden fornax/golden_vectors/runtime_format --out
  /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed.

### External torch H100 calibration probe milestone

- Added external torch backend support to local calibration via `fornax calibrate
  local --torch-python` and `fornax preflight --calibration-torch-python`, so
  Phase-0 evidence can use an existing torch-capable venv without adding PyTorch
  as a dependency of the repo Python environment.
- Ran the H100 smoke through `/mnt/dataprocessing/venvs/asr-data-prep/bin/python`.
  The artifact records `backend_mode=external_python`, PyTorch `2.12.0+cu130`,
  and two measured `NVIDIA H100 80GB HBM3` CUDA devices. The only calibration
  warning left in that artifact is the intended CPU-plumbing caveat.
- Added deterministic unit coverage for external-Python JSON/fallback handling
  and wired preflight tests through the new `calibration_torch_python` path.
- Review-lens pass:
  - Hardware: approve with comments. H100 CUDA measurement is now real for the
    named ASR venv/build, but it remains a tiny matmul microprobe rather than
    target-model or interconnect throughput evidence.
  - Analytical: approve with comments. The artifact records provenance and
    inputs, but G1 claims still need target-model, MAX/Mojo, and active fabric
    calibration thresholds.
  - Software Engineering: approve. The implementation keeps the default path
    dependency-light while allowing explicit, auditable use of a known torch
    environment.
- Verification: focused calibration/preflight unit tests, `python3 -m unittest
  discover -s tests -p 'test_fornax*.py'`, `python3 -m compileall -q fornax
  tests`, `python3 -m fornax calibrate local --out
  /tmp/fornax_local_calibration_h100.json --cpu-memory-bytes 1048576
  --cpu-memory-iterations 2 --cpu-compute-iterations 10000 --torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --cuda-matrix-dim 512
  --cuda-iterations 3`, `python3 -m fornax preflight --target
  fornax/golden_plans/v0_target_contract_fixture.md --out-dir
  /tmp/fornax_preflight_with_external_calibration --benchmark-iterations 1
  --include-g1-drafts --include-calibration --calibration-torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --substrate-pinned-build
  max-26.4.0 --kickoff-date 2026-06-20 --ker-status unavailable --scope
  pending`, `python3 -m fornax doctor --bundle
  /tmp/fornax_preflight_with_external_calibration --out
  /tmp/fornax_preflight_with_external_calibration/doctor_rerun.json`, `python3
  -m fornax test golden-plans`, `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture fornax/golden_vectors/network_contract`,
  `python3 -m fornax spec runtime-format --golden fornax/golden_vectors/runtime_format
  --out /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed.


### G1 gate-review draft milestone

- Added `fornax program g1-review` support via `fornax.g1_review`, which renders
  the program-management gate-review template from a Phase-0 evidence bundle.
  The draft maps current artifacts to the G1 exit criteria and separates
  machine-checkable evidence from human closure requirements.
- Added optional `fornax test golden-plans --out <path>` JSON reporting so T0
  golden-plan evidence can be attached to a bundle and consumed by the G1 review
  draft instead of relying on console output.
- The G1 review draft intentionally does not claim gate closure. On the smoke
  bundle, it recommends `ITERATE` and surfaces four closure blockers after T0 is
  attached: missing TL/SP target-contract sign-off, missing Apple probe validation
  and role decision, missing review sign-off for generated specs, and missing
  owner/staffing sign-off.
- Review-lens pass:
  - Program Management: approve. The generated draft follows the gate template,
    keeps DEC-005 as a Sponsor decision, and makes G1 no-go/iterate evidence
    explicit rather than implied.
  - SRE/Operations: approve. A reproducible bundle-to-review flow now exists:
    preflight bundle, attach T0 JSON, render gate review, inspect blockers.
  - High-level Software: approve with comments. CLI names are direct and the
    draft is readable; a Markdown spacing issue found during review was fixed
    before commit.
- Verification: focused G1-review tests, `python3 -m unittest discover -s tests
  -p 'test_fornax*.py'`, `python3 -m compileall -q fornax tests`, `python3 -m
  fornax preflight --target fornax/golden_plans/v0_target_contract_fixture.md
  --out-dir /tmp/fornax_preflight_g1_review --benchmark-iterations 1
  --include-g1-drafts --include-calibration --calibration-torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --substrate-pinned-build
  max-26.4.0 --kickoff-date 2026-06-20 --ker-status unavailable --scope
  pending`, `python3 -m fornax test golden-plans --out
  /tmp/fornax_preflight_g1_review/golden-plans.json`, `python3 -m fornax
  program g1-review --bundle /tmp/fornax_preflight_g1_review --out
  /tmp/fornax_preflight_g1_review/g1-gate-review.md --date 2026-06-20
  --plan-version v3`, `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m fornax test
  network-contract --mode simulated --fixture fornax/golden_vectors/network_contract`,
  `make fornax-test`, and `make fornax-golden` all passed.


### Active local fabric measurement milestone

- Added optional active same-host fabric measurement to `fornax fabric probe` via
  `--active-local --torch-python`, using an explicit external torch Python so the
  repo Python remains dependency-light.
- Wired the same active local link probe into `fornax preflight` through
  `--active-local-links --fabric-torch-python`, allowing Phase-0 bundles to carry
  measured CPU/GPU and GPU/GPU same-host link provenance instead of only topology
  estimates.
- Ran the active probe through `/mnt/dataprocessing/venvs/asr-data-prep/bin/python`
  on this workstation. The H100 smoke recorded three active measurements
  (CPU-to-GPU0, CPU-to-GPU1, and GPU0-to-GPU1), zero estimated links, and no link
  warnings. A preflight bundle using active links removed the previous
  `no active fabric measurements recorded` warning; remaining warnings are the
  expected calibration caveat and open Apple probe artifacts.
- Review-lens pass:
  - Hardware: approve with comments. Same-host H100/CPU copy paths are now
    measured with device/runtime provenance, but this remains a local copy
    microprobe and not a distributed network benchmark.
  - SRE/Operations: approve. The active probe is opt-in, preserves the estimate
    fallback, and produces doctor-visible warnings when any link remains
    unmeasured.
  - Low-level Software: approve with comments after fix. Review found that
    unmeasured declared links could be hidden when other links were measured;
    warnings now surface unmeasured declarations explicitly.
- Verification: focused fabric probe tests, `python3 -m unittest discover -s
  tests -p 'test_fornax*.py'`, `python3 -m compileall -q fornax tests`, `python3
  -m fornax inventory collect --out /tmp/fornax_inventory_active_links.json`,
  `python3 -m fornax fabric probe --inventory /tmp/fornax_inventory_active_links.json
  --out /tmp/fornax_links_active_h100.json --active-local --torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --active-local-bytes
  1048576 --active-local-iterations 2`, `python3 -m fornax preflight --target
  fornax/golden_plans/v0_target_contract_fixture.md --out-dir
  /tmp/fornax_preflight_active_links --benchmark-iterations 1 --include-g1-drafts
  --include-calibration --calibration-torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --active-local-links
  --fabric-torch-python /mnt/dataprocessing/venvs/asr-data-prep/bin/python
  --active-local-link-bytes 1048576 --active-local-link-iterations 2
  --substrate-pinned-build max-26.4.0 --kickoff-date 2026-06-20 --ker-status
  unavailable --scope pending`, `python3 -m fornax doctor --bundle
  /tmp/fornax_preflight_active_links --out
  /tmp/fornax_preflight_active_links/doctor_rerun.json`, `python3 -m fornax
  test golden-plans --out /tmp/fornax_preflight_active_links/golden-plans.json`,
  `python3 -m fornax program g1-review --bundle /tmp/fornax_preflight_active_links
  --out /tmp/fornax_preflight_active_links/g1-gate-review.md --date 2026-06-20
  --plan-version v3`, `python3 -m fornax test runtime-format --golden
  fornax/golden_vectors/runtime_format`, `python3 -m fornax test network-contract
  --mode simulated --fixture fornax/golden_vectors/network_contract`, `python3 -m
  fornax spec runtime-format --golden fornax/golden_vectors/runtime_format --out
  /tmp/fornax_runtime_format_and_invariants.md`, `python3 -m fornax spec
  network-security --fixture fornax/golden_vectors/network_contract --out
  /tmp/fornax_networking_security_and_backpressure.md`, `make fornax-test`, and
  `make fornax-golden` all passed.


### Logical simulated cluster milestone

- Added `fornax inventory simulate-cluster` to split local NVIDIA GPUs into
  separate logical hosts. Each logical node keeps the source physical device and
  records the worker binding needed to launch one process per GPU through
  `CUDA_VISIBLE_DEVICES`.
- The default `two-gpu-heterogeneous` profile deliberately scales the second
  logical node's compute, memory bandwidth, and available memory so planner and
  scheduler work can exercise heterogeneous-placement paths before a real
  multi-machine lab is available.
- Wired simulated inventories into `fornax preflight --inventory` and added
  doctor-visible simulation provenance. Simulation bundles can validate normal
  inventory, link, placement, validation, benchmark, and G1-review plumbing, but
  now carry an explicit warning that they are not real multi-host hardware
  evidence.
- Ran the flow on this workstation: local inventory collection found two GPUs,
  produced `/tmp/fornax_sim_cluster_inventory.json` with `sim-host-0` and
  `sim-host-1`, and preflight wrote `/tmp/fornax_preflight_sim_cluster` with the
  expected simulation and unmeasured-link warnings.
- Review-lens pass:
  - Program Management: approve with comments. This unblocks milestone
    development and SIM-track validation, while preserving physical cluster
    closure as a later hardware gate.
  - SRE/Operations: approve. The simulation path is explicit, reproducible from
    a source inventory file, and doctor-visible so reports cannot confuse it with
    production cluster evidence.
  - Hardware: approve with comments. The profile is useful for local
    heterogeneity testing, but fabric bandwidth and latency are synthetic until
    replaced by real cross-machine measurements.
- Verification: `python3 -m py_compile fornax/inventory/simulated_cluster.py
  fornax/cli.py fornax/doctor.py tests/test_fornax_planner.py`, focused
  simulated-cluster unittest cases, `python3 -m unittest tests.test_fornax_planner`,
  `python3 -m fornax inventory collect --out /tmp/fornax_local_inventory.json`,
  `python3 -m fornax inventory simulate-cluster --source-inventory
  /tmp/fornax_local_inventory.json --out /tmp/fornax_sim_cluster_inventory.json
  --gpu-count 2 --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004
  --slow-node-factor 0.65`, `python3 -m fornax preflight --target
  fornax/golden_plans/v0_target_contract_fixture.md --inventory
  /tmp/fornax_sim_cluster_inventory.json --out-dir
  /tmp/fornax_preflight_sim_cluster --benchmark-iterations 1 --include-g1-drafts
  --substrate-pinned-build max-26.4.0 --kickoff-date 2026-06-21 --ker-status
  unavailable --scope pending`, `python3 -m fornax doctor --bundle
  /tmp/fornax_preflight_sim_cluster`, `python3 -m fornax program g1-review
  --bundle /tmp/fornax_preflight_sim_cluster --out
  /tmp/fornax_preflight_sim_cluster_g1_review.md --date 2026-06-21
  --plan-version v3`, `python3 -m fornax test golden-plans --out
  /tmp/fornax_golden_plans_sim_slice.json`, `make fornax-test`, and `make
  fornax-golden` all passed.


### Preflight T0 golden evidence milestone

- Added optional `fornax preflight --include-golden-plans` support so Phase-0
  evidence bundles can carry `golden-plans.json` without a separate manual
  attachment step.
- The generated JSON uses the same T0 report shape as `fornax test golden-plans`,
  allowing the existing G1 gate-review draft to consume preflight-produced T0
  evidence directly.
- Ran the new path on the two-GPU logical simulated cluster. The bundle at
  `/tmp/fornax_preflight_sim_golden` includes `golden-plans.json` with 3/3
  golden fixtures passing. The G1 review no longer reports `golden-plan tests T0
  green` as missing; the remaining machine gap is the Apple rank-1 probe/role
  decision, with human closure gaps for target/spec/staffing sign-off.
- Review-lens pass:
  - Program Management: approve. This moves S0-1/T0 evidence into the repeatable
    preflight bundle used for G1 review while preserving the existing G1 closure
    blockers.
  - SRE/Operations: approve. Operators can now run a single preflight command for
    inventory, fabric metadata, planning, simulation, benchmark, generated G1
    drafts, and T0 golden evidence.
  - Testing: approve. The default preflight path remains minimal; T0 attachment is
    opt-in, covered by focused tests, and uses the same report schema as the
    standalone golden command.
- Verification: `python3 -m py_compile fornax/preflight.py fornax/cli.py
  tests/test_fornax_planner.py`, focused G1/preflight tests, `python3 -m unittest
  tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make
  fornax-test`, `make fornax-golden`, `python3 -m fornax inventory collect --out
  /tmp/fornax_local_inventory_golden_preflight.json`, `python3 -m fornax inventory
  simulate-cluster --source-inventory /tmp/fornax_local_inventory_golden_preflight.json
  --out /tmp/fornax_sim_inventory_golden_preflight.json --gpu-count 2
  --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor
  0.65`, `python3 -m fornax preflight --target
  fornax/golden_plans/v0_target_contract_fixture.md --inventory
  /tmp/fornax_sim_inventory_golden_preflight.json --out-dir
  /tmp/fornax_preflight_sim_golden --benchmark-iterations 1 --include-g1-drafts
  --include-golden-plans --substrate-pinned-build max-26.4.0 --kickoff-date
  2026-06-21 --ker-status unavailable --scope pending`, and `python3 -m fornax
  program g1-review --bundle /tmp/fornax_preflight_sim_golden --out
  /tmp/fornax_preflight_sim_golden/g1-review.md --date 2026-06-21 --plan-version
  v3` all passed.


### Phase-0 milestone status report milestone

- Added `fornax program phase0-status` to generate machine-readable JSON and a
  Markdown weekly-status-style report from a Phase-0 evidence bundle.
- The report maps S0-1 through S0-9 to status values: `closed`,
  `machine_complete`, `simulation_complete`, and `incomplete`. This directly
  addresses R-10 status drift by making simulated evidence explicit instead of
  allowing it to look like real multi-host hardware closure.
- Reused the existing G1 evidence rows and doctor artifacts so the status report,
  G1 review draft, and preflight bundle share one interpretation of the evidence.
- Ran the report against a two-H100 logical simulated cluster bundle at
  `/tmp/fornax_preflight_phase0_status`. Current simulated-development posture:
  8/9 S0 deliverables are machine/simulation complete or closed. S0-7 remains
  incomplete because the Apple rank-1 probe validation and Apple role decision are
  absent. S0-2 and S0-9 are intentionally labeled `simulation_complete`, not
  physical hardware closure.
- Review-lens pass:
  - Program Management: approve. The report makes milestone posture readable by
    S0 item, keeps DEC-005 as pending, and surfaces the next gating decision.
  - Organizational/SRE: approve. It gives operators one command to produce the
    weekly-status artifact from a bundle without oral context.
  - Testing/Quality: approve with comments after fix. Review found that
    simulation-sensitive deliverables could show no gap when their machine checks
    passed; `simulation_complete` rows now carry an explicit simulation-only gap
    where needed.
- Verification: `python3 -m py_compile fornax/phase0_status.py fornax/g1_review.py
  fornax/cli.py tests/test_fornax_planner.py`, focused Phase-0 status tests,
  `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q
  fornax tests`, `make fornax-test`, `make fornax-golden`, `python3 -m fornax
  inventory collect --out /tmp/fornax_local_inventory_phase0_status.json`,
  `python3 -m fornax inventory simulate-cluster --source-inventory
  /tmp/fornax_local_inventory_phase0_status.json --out
  /tmp/fornax_sim_inventory_phase0_status.json --gpu-count 2
  --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor
  0.65`, `python3 -m fornax preflight --target
  fornax/golden_plans/v0_target_contract_fixture.md --inventory
  /tmp/fornax_sim_inventory_phase0_status.json --out-dir
  /tmp/fornax_preflight_phase0_status --benchmark-iterations 1 --include-g1-drafts
  --include-golden-plans --substrate-pinned-build max-26.4.0 --kickoff-date
  2026-06-21 --ker-status unavailable --scope pending`, and `python3 -m fornax
  program phase0-status --bundle /tmp/fornax_preflight_phase0_status --out
  /tmp/fornax_phase0_status.json --markdown-out /tmp/fornax_phase0_status.md
  --date 2026-06-21 --plan-version v3` all passed.


### Preflight program-report bundle milestone

- Added optional `fornax preflight --include-program-reports` support. A single
  preflight run can now materialize `g1-gate-review.md`, `phase0-status.json`,
  and `phase0-status.md` alongside inventory, links, placement, validation,
  simulation, benchmark, generated G1 drafts, and T0 golden evidence.
- Wired report metadata through `--program-report-date` and
  `--program-plan-version` so generated program artifacts use the same date and
  plan-version fields as the standalone `program g1-review` and
  `program phase0-status` commands.
- Ran the full simulated-cluster evidence path at
  `/tmp/fornax_preflight_program_reports`. The bundle includes the generated G1
  review and Phase-0 status report. Current report posture remains 8/9 S0 items
  machine/simulation complete or closed, with S0-7 incomplete until Apple probe
  validation and role decision are attached.
- Review-lens pass:
  - Program Management: approve. This makes the simulated milestone-validation
    flow reproducible from one command while keeping the generated artifacts in
    DRAFT/PENDING gate posture.
  - SRE/Operations: approve. The operator no longer needs to remember separate
    post-preflight report commands for routine evidence bundles.
  - Testing/Quality: approve after fix. Focused test initially checked generated
    files after the temporary bundle was deleted; the test now validates artifacts
    before cleanup.
- Verification: `python3 -m py_compile fornax/preflight.py fornax/cli.py
  fornax/phase0_status.py tests/test_fornax_planner.py`, focused preflight report
  tests, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall
  -q fornax tests`, `make fornax-test`, `make fornax-golden`, `python3 -m fornax
  inventory collect --out /tmp/fornax_local_inventory_program_reports.json`,
  `python3 -m fornax inventory simulate-cluster --source-inventory
  /tmp/fornax_local_inventory_program_reports.json --out
  /tmp/fornax_sim_inventory_program_reports.json --gpu-count 2
  --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor
  0.65`, and `python3 -m fornax preflight --target
  fornax/golden_plans/v0_target_contract_fixture.md --inventory
  /tmp/fornax_sim_inventory_program_reports.json --out-dir
  /tmp/fornax_preflight_program_reports --benchmark-iterations 1 --include-g1-drafts
  --include-golden-plans --include-program-reports --program-report-date
  2026-06-21 --substrate-pinned-build max-26.4.0 --kickoff-date 2026-06-21
  --ker-status unavailable --scope pending` all passed.


### Simulated Apple S0-7 evidence milestone

- Added development-only Apple simulation artifacts for local milestone validation:
  `apple-probe-simulation.json` and `apple-role-decision-simulated.md`.
- Added `fornax apple simulate-probe` for standalone generation, and
  `fornax preflight --include-simulated-apple-evidence` so the full simulated
  evidence bundle can exercise S0-7 status plumbing without a target Mac.
- The simulated artifact is deliberately not accepted by the real
  `validate_apple_probe_artifact` gate path and does not write
  `apple-probe-validation.json` or `apple-role-decision.md`. G1 review therefore
  still reports the Apple rank-1 probe/role-decision criterion as missing.
- Ran the full two-H100 logical-cluster preflight with simulated Apple evidence at
  `/tmp/fornax_preflight_apple_sim`. Phase-0 status now shows 9/9 S0 deliverables
  as closed, machine-complete, or simulation-complete. G1 still recommends
  `ITERATE`, with Apple rank-1 local probe evidence missing.
- Review-lens pass:
  - Program Management: approve with comments. This unblocks simulated milestone
    validation across all S0 rows while preserving the real G1 decision boundary.
  - Hardware/Acceleration: approve with comments. The artifact is clearly marked
    development-only and cannot satisfy the rank-1 local Apple probe requirement.
  - Testing/Quality: approve. Tests verify that the simulated Apple artifact is
    not gate-closable and that status distinguishes simulated S0-7 from real G1
    closure.
- Verification: `python3 -m py_compile fornax/apple_probe.py fornax/preflight.py
  fornax/phase0_status.py fornax/cli.py tests/test_fornax_planner.py`, focused
  simulated-Apple tests, `python3 -m unittest tests.test_fornax_planner`,
  `python3 -m compileall -q fornax tests`, `make fornax-test`, `make
  fornax-golden`, `python3 -m fornax apple simulate-probe --out
  /tmp/fornax_apple_probe_simulation.json --decision-out
  /tmp/fornax_apple_role_decision_simulated.md --target-model qwen3-moe-target
  --pinned-build max-26.4.0 --role capacity-only`, `python3 -m fornax inventory
  collect --out /tmp/fornax_local_inventory_apple_sim.json`, `python3 -m fornax
  inventory simulate-cluster --source-inventory
  /tmp/fornax_local_inventory_apple_sim.json --out
  /tmp/fornax_sim_inventory_apple_sim.json --gpu-count 2
  --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor
  0.65`, and `python3 -m fornax preflight --target
  fornax/golden_plans/v0_target_contract_fixture.md --inventory
  /tmp/fornax_sim_inventory_apple_sim.json --out-dir /tmp/fornax_preflight_apple_sim
  --benchmark-iterations 1 --include-g1-drafts --include-golden-plans
  --include-program-reports --include-simulated-apple-evidence
  --simulated-apple-role capacity-only --program-report-date 2026-06-21
  --substrate-pinned-build max-26.4.0 --kickoff-date 2026-06-21 --ker-status
  unavailable --scope pending` all passed.

### One-command simulated Phase-0 validation milestone

- Added `fornax program simulate-phase0`, which packages the local logical
  cluster simulation method into the program-management workflow. The command
  collects or accepts a source inventory, writes `source-inventory.json`, builds
  `simulated-cluster-inventory.json`, and runs the full preflight with G1 drafts,
  T0 golden evidence, program reports, and simulated Apple S0-7 evidence.
- This is now the default development milestone path for two local GPUs treated
  as two logical hosts. It validates planner, runtime-contract, networking,
  evidence-bundle, status-report, and Apple-simulation plumbing without waiting
  for a physical heterogeneous cluster.
- Ran the one-command path at `/tmp/fornax_phase0_simulate_onecmd`. Phase-0
  status shows 9/9 S0 deliverables closed, machine-complete, or
  simulation-complete: 2 closed, 4 machine-complete, 3 simulation-complete, and
  0 incomplete. G1 still recommends `ITERATE` because real Apple rank-1 probe
  validation, sign-offs, and staffing closure remain absent.
- Review-lens pass:
  - Program Management: approve. The command directly supports the simulated
    milestone plan and produces the status artifact needed for milestone review.
  - SRE/Operations: approve. Operators no longer have to stitch inventory
    collection, cluster simulation, preflight, and reports by hand.
  - Hardware/Acceleration: approve with comments. The generated inventory and
    Apple artifacts are explicitly simulation evidence and cannot close real
    multi-host or rank-1 Apple gates.
  - Testing/Quality: approve. Regression coverage asserts the 9/9 simulated S0
    posture while preserving `G1=ITERATE`.
- Verification: `python3 -m py_compile fornax/phase0_simulated_validation.py
  fornax/cli.py tests/test_fornax_planner.py`, focused
  `test_program_simulate_phase0_builds_full_simulated_bundle`, `python3 -m
  fornax program simulate-phase0 --target
  fornax/golden_plans/v0_target_contract_fixture.md --out-dir
  /tmp/fornax_phase0_simulate_onecmd --gpu-count 2
  --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004
  --slow-node-factor 0.65 --program-report-date 2026-06-21
  --substrate-pinned-build max-26.4.0 --kickoff-date 2026-06-21 --ker-status
  unavailable --scope pending --benchmark-iterations 1`, status-artifact
  inspection, `python3 -m unittest tests.test_fornax_planner`, `python3 -m
  compileall -q fornax tests`, `make fornax-test`, and `make fornax-golden` all
  passed.

### Engine seam contract milestone

- Added `fornax.engine_seam` and the golden fixture
  `fornax/golden_vectors/engine_seam/fixture.json` for the plan v3 §5.7
  Ignis↔Fornax LLM semantic seam. This is a Phase-0/T1 contract validator, not a
  Phase-1 `FornaxBackend` implementation.
- The fixture and validator cover `EngineRequest` fields for messages, tools,
  response format, stop sequences, sampling params, max tokens, stream mode,
  cancellation propagation, template version/hash, and tokenizer version/hash.
  `EngineResult` and stream checks cover token chunks, finish reasons, tool calls,
  structured output, usage accounting, error results, cancellation cleanup, and
  hash propagation.
- Added `fornax test engine-seam`; expanded `make fornax-golden` so the local
  golden target now runs planner golden plans plus runtime-format, network
  contract, and engine-seam contract checks.
- Review-lens pass:
  - Software Engineering: approve. The seam validator is isolated as a small
    model-free module, has regression tests for success and failure paths, and
    avoids importing runtime/backend concerns into Phase 0.
  - System Engineering: approve with comments. The contract makes the
    request/result/error/cancellation lifecycle explicit before backend work, but
    real observability threading and backend integration remain future milestones.
  - LLM Expertise: approve with comments. The milestone prevents tokenizer, chat
    template, streaming, tool-call, structured-output, and cancellation semantics
    from being assumed; model-family acceptance tests still need real backend
    coverage after G1.
  - Testing/Quality: approve. `make fornax-golden` now exercises all current
    Phase-0/T1 golden contracts from one command.
- Verification: `python3 -m py_compile fornax/engine_seam.py fornax/cli.py
  tests/test_fornax_planner.py`, focused engine-seam tests, `python3 -m fornax
  test network-contract`, `python3 -m fornax test engine-seam`, `python3 -m
  unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`,
  `make fornax-golden`, and `make fornax-test` all passed.

### T1 observability contract milestone

- Added `fornax.observability` and the golden fixture
  `fornax/golden_vectors/observability/fixture.json` for the plan v3 §5.9
  observability requirement pulled into Phase 0/T1 simulation.
- The validator requires request/plan ID propagation and checks event coverage for
  per-stage timings, bubble fraction, queue depth, backpressure, router decisions,
  remote expert hits, expert wait, migration, KV page counts, memory pressure,
  allocation failures, eviction/replay, placement explanations, and reproducible
  bad-plan fixture logs.
- Added `fornax test observability` and expanded `make fornax-golden` so the
  current model-free golden-contract command covers planner, runtime format,
  network contract, engine seam, and observability contracts.
- Review-lens pass:
  - System Engineering: approve. The milestone makes the T1 request lifecycle
    observable across scheduler, routing, KV, placement, and failure reproduction
    before real workers exist.
  - SRE/Operations: approve with comments. The fixture provides a concrete
    diagnostics contract for future logs/metrics; live log emission and dashboard
    wiring remain later milestones.
  - Low-level Software: approve with comments. KV pressure, allocation failure,
    eviction, and replay are explicit in the contract, but real memory-manager
    invariants still need implementation after G1.
  - Testing/Quality: approve. Regression tests catch missing required telemetry
    and broken plan-ID propagation, and `make fornax-golden` now exercises the
    observability contract.
- Verification: `python3 -m py_compile fornax/observability.py fornax/cli.py
  tests/test_fornax_planner.py`, focused observability tests, `python3 -m fornax
  test observability`, `python3 -m unittest tests.test_fornax_planner`,
  `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make
  fornax-test` all passed.

### Backend operation coverage matrix milestone

- Added `fornax.backend_coverage` and the golden fixture
  `fornax/golden_vectors/backend_coverage/fixture.json` for the plan v3 §5.10 /
  WBS D3 backend operation coverage matrix.
- The matrix covers Apple, NVIDIA, and AMD across attention, dense MLP,
  router/top-k, expert GEMM/MLP, collect/scatter/gather, KV operations,
  sampling/logits, serialization/pack/gather, and transport. Each backend cell
  carries `supported`, `fast_enough`, `correct`, `used_by_target_model`, and
  traceable evidence fields. Current op-level statuses are deliberately
  `unknown` where measured ledgers are not attached.
- Added `fornax test backend-coverage` and `fornax spec backend-coverage` to
  validate the machine-readable matrix and render
  `/tmp/fornax_backend_coverage_matrix.md`. Expanded `make fornax-golden` so the
  local golden-contract sweep includes backend coverage.
- Review-lens pass:
  - Hardware Acceleration: approve with comments. The artifact names the
    operation classes, backend targets, profiler/harness expectations, and
    missing measurements instead of accepting generic GPU claims. Real hot-path
    evidence is still pending measured ledgers.
  - Low-level Software: approve. The matrix forces backend-equivalence and
    correctness status per operation/backend and keeps platform-specific evidence
    isolated from runtime code.
  - Program Management: approve with comments. D3 now has a concrete artifact for
    G1 discussion, but unknown cells remain decision inputs, not closure claims.
  - Testing/Quality: approve. Regression tests catch missing required operations
    and missing benchmark-ledger fields; the renderer makes the matrix reviewable.
- Verification: `python3 -m py_compile fornax/backend_coverage.py fornax/cli.py
  tests/test_fornax_planner.py`, focused backend-coverage tests, `python3 -m
  fornax test backend-coverage`, `python3 -m fornax spec backend-coverage --out
  /tmp/fornax_backend_coverage_matrix.md`, rendered matrix inspection,
  `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q
  fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.

### Benchmark ledger and simulated validation evidence milestone

- Added `fornax.benchmark_ledger` and the golden fixture
  `fornax/golden_vectors/benchmark_ledger/ledger.jsonl` for plan v3 §5.10
  benchmark provenance. Ledger records now validate hardware, OS,
  driver/runtime, MAX/Mojo version, model, context, concurrency, quantization,
  thermals, command, and measured benchmark payloads.
- Extended `fornax benchmark --ledger-out` for standalone measured tiny expert-MLP
  ledger records, and added `fornax test benchmark-ledger` to the golden contract
  sweep.
- Wired `benchmark-ledger.jsonl` into Phase-0 preflight and
  `fornax program simulate-phase0`, so the two-GPU logical-cluster method now
  produces auditable benchmark provenance by default. The simulated record
  explicitly names `logical simulated cluster profile=two-gpu-heterogeneous`,
  `physical_gpus=2`, the `nvidia/max` runtime simulation, the target context,
  quantization, and the exact `program simulate-phase0` command.
- Doctor now validates a present benchmark ledger and warns when older bundles do
  not include one, without making legacy minimal bundles fail. This preserves the
  distinction between simulated milestone evidence and real benchmark-of-record
  hardware closure.
- Ran the requested simulation method through `/tmp/fornax_phase0_sim_ledger_cli_20260621`.
  The generated Phase-0 status remained 9/9 S0 deliverables machine/simulation
  complete or closed: 2 closed, 4 machine-complete, 3 simulation-complete, 0
  incomplete. G1 still recommends `ITERATE` because real Apple rank-1 probe
  evidence and human sign-offs remain absent.
- Review-lens pass:
  - Program Management: approve. The default development milestone path now uses
    the two-GPU logical cluster and produces benchmark provenance in the same
    evidence bundle, so milestone validation no longer blocks on a physical
    heterogeneous cluster.
  - Hardware Acceleration: approve with comments. The ledger is explicit that
    the current cluster is simulated; real thermals, cross-host fabric, and
    backend hot-path measurements still need replacement on the physical lab.
  - SRE/Operations: approve. Doctor-visible ledger validation makes provenance
    reviewable from one bundle and keeps the command needed to reproduce the run.
  - Testing/Quality: approve. Regression tests cover fixture validation,
    unmeasured-record rejection, preflight ledger generation, and the
    `simulate-phase0` ledger path.
- Verification: `python3 -m py_compile fornax/benchmark_ledger.py fornax/preflight.py
  fornax/doctor.py fornax/phase0_simulated_validation.py fornax/cli.py
  tests/test_fornax_planner.py`, focused benchmark-ledger and simulated-preflight
  tests, `python3 -m fornax test benchmark-ledger`, `python3 -m fornax program
  simulate-phase0 --target fornax/golden_plans/v0_target_contract_fixture.md
  --out-dir /tmp/fornax_phase0_sim_ledger_cli_20260621 --source-inventory
  /tmp/fornax_two_gpu_source_inventory_for_ledger.json --gpu-count 2
  --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004
  --slow-node-factor 0.65 --program-report-date 2026-06-21
  --substrate-pinned-build max-26.4.0 --kickoff-date 2026-06-21 --ker-status
  unavailable --scope pending --benchmark-iterations 1`, `python3 -m fornax test
  benchmark-ledger --fixture
  /tmp/fornax_phase0_sim_ledger_cli_20260621/benchmark-ledger.jsonl`, direct
  `python3 -m fornax benchmark --ledger-out ...` smoke, `python3 -m unittest
  tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make
  fornax-golden`, `make fornax-test`, and `git diff --check` all passed.

### T1 scheduler simulation contract milestone

- Added `fornax.scheduler` for model-free T1 scheduler simulation: bounded
  admission queues, deterministic microbatch formation, per-stage start/end
  timing events, completion/cancellation terminal events, and summary counts for
  queue depth, inflight requests, microbatches, backpressure, and makespan.
- Added `fornax scheduler simulate --plan placement.json --requests trace.json`
  to produce a scheduler contract artifact without real workers or transport.
  This stays on the simulation track and does not implement Phase-1 distributed
  runtime execution.
- Added `fornax test scheduler-contract` and the golden fixture
  `fornax/golden_vectors/scheduler_contract/fixture.json`; expanded
  `make fornax-golden` so T0/T1 local checks now include scheduler contract
  validation alongside planner, runtime format, network, engine seam,
  observability, backend coverage, and benchmark ledger checks.
- The validator enforces root/event plan-ID consistency, required event coverage,
  queue and inflight bounds, microbatch size limits, stage start/end pairing,
  terminal events for enqueued requests, cancellation cleanup, and summary/event
  consistency.
- Review-lens pass:
  - Distributed Systems/Scheduler: approve with comments. The contract exercises
    admission, bounded queues, microbatching, and per-stage events, but real 1F1B
    timing, fairness under arrivals, and worker transport remain later simulated
    milestones.
  - SRE/Operations: approve. The artifact is deterministic, CLI-reproducible,
    and summarizes the operational signals needed for backpressure review.
  - Low-level Software: approve with comments. State cleanup is explicit for
    cancellation and invariants are validator-enforced; future worker contracts
    still need memory/KV ownership checks.
  - Testing/Quality: approve. Focused tests cover fixture validation,
    bounded-queue backpressure, trace loading, and validator rejection of queue
    overflow.
- Verification: `python3 -m py_compile fornax/scheduler.py fornax/cli.py
  tests/test_fornax_planner.py`, `python3 -m fornax test scheduler-contract`,
  focused scheduler-contract unittests, `python3 -m fornax scheduler simulate
  --plan /tmp/fornax_scheduler_plan_20260621.json --requests
  /tmp/fornax_scheduler_requests_20260621.json --plan-id cli-scheduler-smoke
  --max-queue-depth 2 --max-inflight 2 --microbatch-size 2 --out
  /tmp/fornax_scheduler_contract_cli_20260621.json`, `python3 -m fornax test
  scheduler-contract --fixture /tmp/fornax_scheduler_contract_cli_20260621.json`,
  `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q
  fornax tests`, `make fornax-golden`, `make fornax-test`, and `git diff --check`
  all passed.

### T1 simulated worker contract milestone

- Added `fornax.workers` for the model-free T1 worker contract: simulated stage
  and expert worker registration, plan loading with plan hashes, activation
  receive/send, KV write, expert-batch receive/execute/result handoff,
  stale-plan rejection, and per-worker cleanup.
- Added `fornax workers simulate --out ...` to emit a deterministic worker
  contract artifact without real `StageWorker` execution, MAX graphs, or network
  transport. This keeps the milestone on the pre-G1 simulation track.
- Added `fornax test worker-contract` and the golden fixture
  `fornax/golden_vectors/worker_contract/fixture.json`; expanded
  `make fornax-golden` so the local T0/T1 sweep validates worker, scheduler,
  network, runtime-format, engine-seam, observability, backend-coverage, and
  benchmark-ledger contracts.
- The validator enforces worker identity/roles, runtime-format payload validity,
  root/event plan and request ID consistency, plan-hash tags on payload/execution
  events, stale-plan rejection on mismatched hashes, bounded queue depth,
  role-appropriate stage/expert events, start/end pairing, per-worker cleanup,
  and summary/event consistency.
- Review-lens pass:
  - Distributed Systems/Scheduler: approve with comments. The worker contract now
    connects scheduler microbatches to stage/expert worker lifecycle events, but
    real worker orchestration and 1F1B overlap remain later simulation/runtime
    milestones.
  - Networking/System: approve with comments. Plan-integrity tags and stale-plan
    rejection are explicit at the worker boundary; transport serialization and
    TCP/shm implementation are still out of scope until G1 authorizes Phase 1.
  - Low-level Software: approve. Runtime-format payload validation is reused
    instead of duplicating tensor-shape checks, and cleanup requirements make
    ownership failure visible in fixture tests.
  - Testing/Quality: approve. Focused tests cover fixture validity, generated
    artifacts, plan-hash mismatch rejection, missing cleanup, role/event mismatch,
    and unsupported payload rejection.
- Verification: `python3 -m py_compile fornax/workers.py fornax/cli.py
  tests/test_fornax_planner.py`, `python3 -m fornax test worker-contract`,
  focused worker-contract unittests, `python3 -m fornax workers simulate --out
  /tmp/fornax_worker_contract_cli_20260621.json --plan-id cli-worker-plan
  --request-id cli-request --plan-hash sha256:cli-worker-plan --max-queue-depth
  2`, `python3 -m fornax test worker-contract --fixture
  /tmp/fornax_worker_contract_cli_20260621.json`, `python3 -m unittest
  tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make
  fornax-golden`, `make fornax-test`, and `git diff --check` all passed.

### T1 simulated transport contract milestone

- Adopted the two-local-GPU simulation method for T1 transport development:
  `sim-gpu0` and `sim-gpu1` are modeled as separate logical hosts with explicit
  `CUDA_VISIBLE_DEVICES` worker bindings. This unblocks milestone validation on
  the local development machine while preserving the later requirement to rerun
  on a real heterogeneous cluster.
- Added `fornax.transport` for a model-free transport contract over those
  logical hosts: endpoint registration, channel open, activation/KV/expert
  payload enqueue/send/receive/ack lifecycle, timeout, cancel, backpressure,
  plan-integrity reject, and per-endpoint cleanup.
- Added `fornax transport simulate --out ...` and `fornax test
  transport-contract`; added the golden fixture
  `fornax/golden_vectors/transport_contract/fixture.json`; expanded
  `make fornax-golden` to validate the transport contract with the rest of the
  T0/T1 contract sweep.
- The validator enforces runtime-format payload validity, two logical GPU hosts,
  endpoint GPU bindings, channel endpoint consistency, plan/request/hash
  propagation, bounded queue depth, timeout thresholds, terminal payload states,
  cleanup coverage, and summary/event consistency.
- Review-lens pass:
  - Networking/System: approve with comments. The transport contract now
    validates the simulated distributed data-plane semantics, but real TCP/shm,
    RDMA, and cross-host fabric are still future hardware-tier work.
  - Distributed Systems/Scheduler: approve. Payload lifecycle and terminal
    states are explicit enough for scheduler/worker milestones to rely on this
    simulated cluster without blocking on physical heterogeneity.
  - SRE/Operations: approve with comments. The artifact is deterministic and
    CLI-reproducible; it clearly labels simulation-only evidence and keeps the
    real cluster validation gap visible.
  - Testing/Quality: approve. Focused tests cover fixture validity, generated
    logical-host bindings, plan-hash mismatch, queue overflow, missing terminal
    ack, short timeout, and missing GPU binding rejection.
- Verification: `python3 -m py_compile fornax/transport.py fornax/cli.py
  tests/test_fornax_planner.py`, `python3 -m fornax test transport-contract`,
  focused transport-contract unittests, `python3 -m fornax transport simulate
  --out /tmp/fornax_transport_contract_cli_20260621.json --plan-id
  cli-transport-plan --request-id cli-request --plan-hash
  sha256:cli-transport-plan --max-queue-depth 2 --timeout-ms 50`, `python3 -m
  fornax test transport-contract --fixture
  /tmp/fornax_transport_contract_cli_20260621.json`, `python3 -m unittest
  tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make
  fornax-golden`, `make fornax-test`, and `git diff --check` passed.

### One-command T1 simulated validation milestone

- Added `fornax.t1_simulated_validation` and `fornax program simulate-t1` to
  validate the T1 development track over the two-local-GPU logical-host method.
  The command writes `source-inventory.json`, `simulated-cluster-inventory.json`,
  generated scheduler/worker/transport contract artifacts, and
  `t1-simulated-validation.json` into one bundle.
- The bundle validator checks 11 local T0/T1 items together: logical cluster,
  golden plans, runtime format, network contract, engine seam, observability,
  scheduler contract, worker contract, transport contract, backend coverage, and
  benchmark ledger. This gives the program a single simulated-cluster milestone
  command without claiming real T3/T4 distributed hardware closure.
- The generated scheduler/worker/transport artifacts share the same plan ID,
  request ID, plan hash, queue depth, and timeout settings, so the simulated
  milestone validates cross-contract consistency rather than independent golden
  fixtures only.
- Review-lens pass:
  - Program Management: approve. The command directly implements the milestone
    strategy of using the two-H100 local box as a simulated heterogeneous
    cluster, while preserving the G2/G3 requirement for real 2-3 node and
    heterogeneous validation.
  - System Engineering: approve with comments. The bundle now composes planner,
    scheduler, worker, transport, observability, and seam contracts in one
    reproducible path; real `FornaxEngine` execution remains a later phase.
  - SRE/Operations: approve. The output is doctorable in shape, records all
    artifacts under one bundle, and reports failed checks explicitly.
  - Testing/Quality: approve. Regression coverage uses a synthetic two-GPU
    source inventory, validates the full 11-check bundle, and rejects a
    single-GPU source inventory.
- Verification: `python3 -m py_compile fornax/t1_simulated_validation.py
  fornax/cli.py tests/test_fornax_planner.py`, focused T1 simulated validation
  unittests, `python3 -m fornax program simulate-t1 --out-dir
  /tmp/fornax_t1_simulated_validation_cli_20260621 --source-inventory
  /tmp/fornax_t1_source_inventory_20260621.json --gpu-count 2 --profile
  two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s
  0.0004 --slow-node-factor 0.65 --plan-id cli-t1-plan --request-id
  cli-t1-request --plan-hash sha256:cli-t1-plan --max-queue-depth 2
  --max-inflight 2 --microbatch-size 2 --timeout-ms 50`, artifact summary
  inspection showing 11/11 passed, `python3 -m unittest tests.test_fornax_planner`,
  `python3 -m compileall -q fornax tests`, `make fornax-golden`, `make
  fornax-test`, and `git diff --check` all passed.

### T1 simulated FornaxEngine lifecycle milestone

- Added `fornax.engine_simulation` for a model-free `FornaxEngine` orchestration
  contract. The artifact composes generated scheduler, worker, and transport
  contracts and validates the internal request lifecycle across scheduler
  dispatch, stage worker calls, activation handoff, expert dispatch, streaming
  token emission, cancellation propagation, health probes, request finish, and
  cleanup.
- Added `fornax engine simulate --out ...` and `fornax test engine-simulation`;
  added the golden fixture `fornax/golden_vectors/engine_simulation/fixture.json`;
  expanded `make fornax-golden` and `fornax program simulate-t1` so the engine
  lifecycle is now part of the regular T1 simulated-cluster evidence path.
- Updated the one-command T1 bundle to write `engine-simulation.json` and validate
  12 checks: logical cluster, golden plans, runtime format, network contract,
  engine seam, observability, engine simulation, scheduler contract, worker
  contract, transport contract, backend coverage, and benchmark ledger.
- The validator enforces embedded scheduler/worker/transport contract validity,
  root plan/request/hash consistency, known worker and transport payload refs,
  terminal request events, emitted-token accounting, cancellation cleanup targets,
  health-probe readiness, and final cleanup components.
- Review-lens pass:
  - System Engineering: approve with comments. This is the first integrated
    `FornaxEngine` lifecycle artifact across scheduler, workers, and transport;
    real execution, MAX graph calls, and process orchestration remain future
    implementation tiers.
  - Distributed Runtime/Scheduler: approve. The simulated lifecycle now proves
    scheduler dispatch, stage calls, transport handoff, remote expert dispatch,
    and cancellation propagation are contractually connected.
  - SRE/Operations: approve. Health probes and cleanup are visible in the same
    request trace, making simulated bad-plan reproduction and lifecycle review
    more operationally useful.
  - Testing/Quality: approve. Regression tests cover fixture validity,
    generated contract composition, embedded transport hash mismatch, missing
    request finish, token summary mismatch, and missing cleanup.
- Verification: `python3 -m py_compile fornax/engine_simulation.py
  fornax/t1_simulated_validation.py fornax/cli.py tests/test_fornax_planner.py`,
  focused engine-simulation tests, `python3 -m fornax engine simulate --out
  /tmp/fornax_engine_simulation_cli_20260621.json --plan-id cli-engine-plan
  --request-id cli-engine-request --plan-hash sha256:cli-engine-plan
  --max-queue-depth 2 --max-inflight 2 --microbatch-size 2 --timeout-ms 50`,
  `python3 -m fornax test engine-simulation --fixture
  /tmp/fornax_engine_simulation_cli_20260621.json`, `python3 -m fornax program
  simulate-t1 --out-dir /tmp/fornax_t1_engine_simulated_validation_cli_20260621
  --source-inventory /tmp/fornax_t1_source_inventory_20260621.json --gpu-count 2
  --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000
  --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id cli-t1-engine-plan
  --request-id cli-t1-engine-request --plan-hash sha256:cli-t1-engine-plan
  --max-queue-depth 2 --max-inflight 2 --microbatch-size 2 --timeout-ms 50`,
  artifact summary inspection showing 12/12 passed, `python3 -m unittest
  tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make
  fornax-golden`, `make fornax-test`, and `git diff --check` all passed.

### T1 continuous batching and 1F1B simulation milestone

- Added `fornax.continuous_batching` for a model-free T1 continuous-batching
  contract. The artifact validates admission, FIFO fairness, bounded queue and
  inflight counts, microbatch formation, 1F1B-style stage overlap, activation
  transfer start/end pairing, token emission, request completion, and bubble
  telemetry.
- Added `fornax batching simulate --out ...` and `fornax test
  continuous-batching`; added the golden fixture
  `fornax/golden_vectors/continuous_batching/fixture.json`; expanded
  `make fornax-golden` and `fornax program simulate-t1` so continuous batching
  is included in the regular simulated-cluster evidence path.
- Updated the one-command T1 bundle to write `continuous-batching.json` and
  validate 13 checks: logical cluster, golden plans, runtime format, network
  contract, engine seam, observability, engine simulation, continuous batching,
  scheduler contract, worker contract, transport contract, backend coverage, and
  benchmark ledger.
- The validator enforces FIFO admission/formation order, fairness-window waits,
  required bubble samples, stage compute start/end pairing, activation transfer
  pairing, request terminal events, overlap observed across different pipeline
  stages, and summary/event consistency.
- Review-lens pass:
  - Distributed Runtime/Scheduler: approve with comments. The T1 simulation now
    covers continuous batching and 1F1B-style overlap, but real latency/throughput
    scaling and fairness under live arrivals remain T3 hardware work.
  - Analytical Performance: approve with comments. Bubble fraction and wait time
    are now emitted as contract telemetry; they are simulated signals, not
    measured planner-accuracy evidence.
  - SRE/Operations: approve. Backpressure, queue depth, inflight count, and
    fairness-yield events make operational saturation behavior visible in the
    simulated bundle.
  - Testing/Quality: approve. Regression tests cover fixture validity, generated
    fairness/overlap, FIFO order mismatch, missing overlap, fairness-window
    violation, and bubble-summary mismatch.
- Verification: `python3 -m py_compile fornax/continuous_batching.py
  fornax/t1_simulated_validation.py fornax/cli.py tests/test_fornax_planner.py`,
  focused continuous-batching tests, `python3 -m fornax batching simulate --out
  /tmp/fornax_continuous_batching_cli_20260621.json --plan-id cli-batching-plan
  --max-queue-depth 4 --max-inflight 4 --microbatch-size 2 --fairness-window-s
  0.05 --transfer-s 0.002`, `python3 -m fornax test continuous-batching --fixture
  /tmp/fornax_continuous_batching_cli_20260621.json`, `python3 -m fornax program
  simulate-t1 --out-dir /tmp/fornax_t1_continuous_batching_validation_cli_20260621
  --source-inventory /tmp/fornax_t1_source_inventory_20260621.json --gpu-count 2
  --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000
  --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id cli-t1-batching-plan
  --request-id cli-t1-batching-request --plan-hash sha256:cli-t1-batching-plan
  --max-queue-depth 2 --max-inflight 2 --microbatch-size 2 --timeout-ms 50`,
  artifact summary inspection showing 13/13 passed, `python3 -m unittest
  tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make
  fornax-golden`, `make fornax-test`, and `git diff --check` all passed.

### T1 simulated MoE expert-runtime milestone

- Added `fornax.moe` for a model-free MoE expert-runtime contract. The artifact validates router top-k, expert bucketing, local and remote expert dispatch, expert execute/result handoff, migration recommendation, weighted gather, expert trace recording, and cleanup over the two-GPU logical-host simulation method.
- Added `fornax moe simulate --out ...` and `fornax test moe-runtime`; added the golden fixture `fornax/golden_vectors/moe_runtime/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so MoE runtime is included in the regular simulated-cluster evidence path.
- Updated the one-command T1 bundle to write `moe-runtime.json` and validate 14 checks: logical cluster, golden plans, runtime format, network contract, engine seam, observability, engine simulation, continuous batching, MoE runtime, scheduler contract, worker contract, transport contract, backend coverage, and benchmark ledger.
- The validator enforces runtime-format payload validity, top-k weight sums, bucket contents matching routed tokens, placement-role consistency for local versus remote dispatch, remote wait budget, execute start/end pairing, result receipt coverage, weighted gather coverage, migration hotness threshold, trace remote-hit rate, and summary/event consistency.
- Review-lens pass:
  - LLM/Model Architecture: approve with comments. Routing and weighted gather semantics are represented, but layer/logit parity against a real model remains future T2/T3 work.
  - Distributed Runtime/Scheduler: approve. Local and remote expert dispatch, transport-facing payload IDs, result receipt, and migration recommendation are now contractually connected to the simulated runtime path.
  - Analytical Performance: approve with comments. Remote hit rate and wait budget are present as simulated telemetry, not measured performance evidence.
  - Testing/Quality: approve. Regression tests cover fixture validity, generated routing/dispatch/gather, plan-hash mismatch, remote wait over budget, missing gather, invalid top-k weights, and migration below threshold.
- Verification: `python3 -m py_compile fornax/moe.py fornax/t1_simulated_validation.py fornax/cli.py tests/test_fornax_planner.py`, focused MoE runtime tests, `python3 -m fornax moe simulate --out fornax/golden_vectors/moe_runtime/fixture.json`, `python3 -m fornax test moe-runtime`, `python3 -m fornax test moe-runtime --fixture /tmp/fornax_moe_runtime_cli_20260621.json`, focused T1 simulated validation tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_moe_runtime_validation_cli_20260621 --source-inventory /tmp/fornax_t1_source_inventory_20260621.json --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id cli-t1-moe-plan --request-id cli-t1-moe-request --plan-hash sha256:cli-t1-moe-plan --max-queue-depth 2 --max-inflight 2 --microbatch-size 2 --timeout-ms 50`, artifact summary showing 14/14 passed, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make fornax-golden`, `make fornax-test`, and `git diff --check` all passed.

### T1 model-support matrix milestone

- Added `fornax.model_support` for a model-support matrix contract covering the
  Phase 2.5 / H2 seam: architecture, tokenizer, chat template, quantization,
  context length, MoE routing, stop behavior, streaming, tool calling, and
  structured output. The matrix separates a supported model-free fixture row from
  the planned target-candidate row so T1 can validate semantics without claiming
  real target-model parity.
- Added `fornax model-support simulate --out ...`, `fornax test model-support`,
  and `fornax spec model-support`; added the golden fixture
  `fornax/golden_vectors/model_support/fixture.json`; expanded `make
  fornax-golden` and `fornax program simulate-t1` so model support is included
  in the regular simulated-cluster evidence path.
- Updated the one-command T1 bundle to write `model-support-matrix.json` and
  validate 15 checks: logical cluster, golden plans, runtime format, network
  contract, engine seam, observability, engine simulation, continuous batching,
  MoE runtime, model support, scheduler contract, worker contract, transport
  contract, backend coverage, and benchmark ledger.
- The validator enforces required capability coverage, one supported reference
  fixture row, one target-candidate row, resolved tokenizer/template hashes for
  supported rows, explicit required-before-T2 hash gaps for target rows, serving
  semantic coverage for stop/streaming/tools/structured output, and no false
  parity-pass claim without measured evidence.
- Review-lens pass:
  - LLM/Model Architecture: approve with comments. Tokenizer, chat-template,
    stop, streaming, tool-call, structured-output, and MoE routing ownership are
    now explicit; real target tokenizer/template hashes and layer/logit parity
    remain T2/T3 requirements.
  - High-level Software/API: approve. The CLI exposes both machine validation and
    a markdown matrix report, making the support boundary reviewable without
    digging through runtime internals.
  - System Engineering: approve. The matrix links H2 serving semantics back to
    the engine seam, MoE runtime, runtime-format, and target-contract artifacts.
  - Testing/Quality: approve. Regression tests cover fixture validity, generated
    target-gap semantics, report rendering, missing required capabilities,
    unresolved supported-tokenizer hashes, missing tool support, and false parity
    claims.
- Verification: `python3 -m py_compile fornax/model_support.py fornax/cli.py
  fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, focused
  model-support tests, focused T1 simulated validation tests, `python3 -m fornax
  model-support simulate --out fornax/golden_vectors/model_support/fixture.json`,
  `python3 -m fornax test model-support`, `python3 -m fornax test model-support
  --fixture /tmp/fornax_model_support_cli_20260621.json`, `python3 -m fornax spec
  model-support --fixture fornax/golden_vectors/model_support --out
  /tmp/fornax_model_support_matrix_20260621.md`, `python3 -m fornax program
  simulate-t1 --out-dir /tmp/fornax_t1_model_support_validation_cli_20260621
  --source-inventory /tmp/fornax_t1_source_inventory_20260621.json --gpu-count 2
  --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000
  --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id
  cli-t1-model-support-plan --request-id cli-t1-model-support-request
  --plan-hash sha256:cli-t1-model-support-plan --max-queue-depth 2
  --max-inflight 2 --microbatch-size 2 --timeout-ms 50`, artifact summary
  showing 15/15 passed, `python3 -m unittest tests.test_fornax_planner`,
  `python3 -m compileall -q fornax tests`, `make fornax-golden`, `make
  fornax-test`, and `git diff --check` all passed.

### T2 single-node expert-MLP accelerator probe milestone

- Added `fornax.accelerator_probe` and the CLI command `fornax accelerator
  expert-mlp-probe` for a measured single-node expert-MLP microprobe. The probe
  supports a dependency-free CPU reference backend and a torch backend that can
  run in an external CUDA-capable Python, matching the `/mnt/dataprocessing/venvs`
  setup from the referenced venv notes.
- Added `fornax test expert-mlp-probe --fixture ...` so generated probe artifacts
  are machine-validated before they are cited as evidence. The validator enforces
  measured status, correctness pass against the CPU reference, token/expert-call
  accounting, hardware/runtime metadata, and rejects false T2 accelerator claims
  when the device is not CUDA-backed.
- Ran the lab probe on this machine's H100 using
  `/mnt/dataprocessing/venvs/asr-data-prep/bin/python`: artifact
  `/tmp/fornax_expert_mlp_h100_probe_20260621.json` records
  `tier=T2-single-node-accelerator`, `accelerator_measured=true`, device
  `cuda:0`, hardware `NVIDIA H100 80GB HBM3`, torch `2.12.0+cu130`, CUDA `13.0`,
  `tokens_s=590.6496632577714`, `expert_calls_s=1181.2993265155428`, and
  `max_abs_error=0.012801170349121094` with `correctness_passed=true`.
- This is deliberately not part of `make fornax-golden`: it is lab hardware
  evidence and should be run explicitly. It reduces the T2 hardware gap for the
  expert-MLP operation, but it does not claim MAX graph integration, Apple
  expert-worker readiness, target-model layer/logit parity, or T3 multi-node
  correctness.
- Review-lens pass:
  - Hardware Acceleration: approve with comments. The command now measures a real
    H100 expert-MLP shaped path and records device/runtime metadata; benchmark
    shape is still tiny and Python-loop orchestration leaves plenty of overhead.
  - LLM/Model Architecture: approve with comments. Correctness is checked against
    a deterministic CPU expert-MLP reference, but this is operation parity rather
    than target MoE layer/logit parity.
  - Low-level Software: approve. The validator prevents false accelerator
    evidence, records dtype/tolerance, and keeps CPU fallback separate from CUDA
    evidence.
  - Testing/Quality: approve. Regression tests cover CPU reference validity,
    false accelerator claims, and failed correctness rejection; the real H100 run
    is validated through the CLI artifact rather than required in CI.
- Verification: `python3 -m py_compile fornax/accelerator_probe.py fornax/cli.py
  tests/test_fornax_planner.py`, focused expert-MLP probe tests, `python3 -m
  fornax accelerator expert-mlp-probe --backend cpu-stdlib --out
  /tmp/fornax_expert_mlp_cpu_probe_20260621.json --iterations 1 --batch-tokens 2
  --hidden-dim 4 --intermediate-dim 6 --experts 3 --top-k 2`, `python3 -m fornax
  test expert-mlp-probe --fixture /tmp/fornax_expert_mlp_cpu_probe_20260621.json`,
  `python3 -m fornax accelerator expert-mlp-probe --backend torch --torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --device cuda:0 --dtype
  float16 --out /tmp/fornax_expert_mlp_h100_probe_20260621.json --iterations 25
  --warmup 3 --batch-tokens 8 --hidden-dim 64 --intermediate-dim 128 --experts 4
  --top-k 2 --tolerance 0.25 --timeout-s 180`, `python3 -m fornax test
  expert-mlp-probe --fixture /tmp/fornax_expert_mlp_h100_probe_20260621.json`,
  `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q
  fornax tests`, `make fornax-golden`, `make fornax-test`, and `git diff --check`
  all passed.


### T3 same-host activation-transfer simulation probe milestone

- Added an activation-transfer probe to `fornax.accelerator_probe` and the CLI
  command `fornax accelerator activation-transfer-probe`. This uses the approved
  two-local-GPU simulation method: GPU0 and GPU1 are treated as separate logical
  hosts for development validation, while the artifact clearly records
  `tier=T3-same-host-two-gpu-simulation` rather than claiming a real multi-host
  cluster.
- Added `fornax test activation-transfer-probe --fixture ...` so generated
  transfer artifacts are machine-validated before being cited as milestone
  evidence. The validator enforces measured status, transfer accounting,
  bandwidth/latency presence, correctness pass, distinct logical hosts, and a
  distinct CUDA source/destination pair for T3 same-host simulation evidence; CPU
  copies remain valid only as reference plumbing.
- Confirmed local topology with `nvidia-smi topo -m`: GPU0 and GPU1 are connected
  by `NV18`. Ran the lab probe with
  `/mnt/dataprocessing/venvs/asr-data-prep/bin/python`: artifact
  `/tmp/fornax_activation_transfer_h100_pair_20260621.json` records
  `source_device=cuda:0`, `destination_device=cuda:1`, dtype `float16`,
  `effective_payload_bytes=16777216`, `bytes_transferred=335544320`,
  `bandwidth_gib_s=41.76813073154531`,
  `latency_s_per_transfer=0.00037408904172480105`, `max_abs_error=0.0`,
  `correctness_passed=true`, H100 source/destination names, torch
  `2.12.0+cu130`, CUDA `13.0`, and peer access true in both directions.
- This is deliberately not part of `make fornax-golden`: it is local lab
  hardware evidence. It supports the simulation path for Phase 1 transport
  development and activation movement, but it does not close real multi-host T3
  pipeline correctness, pipeline overlap, target-model generation correctness,
  or heterogeneous-cluster validation.
- Review-lens pass:
  - Hardware/Networking: approve with comments. The probe measures actual H100 to
    H100 activation movement over the local NVLink-connected pair; real networked
    host transport remains future cluster evidence.
  - Distributed Runtime/Scheduler: approve with comments. The source/destination
    device pair and logical-host labels match the simulated cluster method, but
    full pipeline scheduling and overlap are still separate milestones.
  - Low-level Software: approve. The validator rejects false T3 simulation claims
    without a distinct CUDA pair, rejects same-device claims, and records runtime
    metadata needed for repeatability.
  - Testing/Quality: approve. Regression tests cover CPU reference validity,
    false T3 claims without CUDA, same CUDA source/destination rejection, and
    failed correctness rejection; the real H100 run is validated through the CLI
    artifact rather than required in CI.
- Verification: `python3 -m py_compile fornax/accelerator_probe.py fornax/cli.py
  tests/test_fornax_planner.py`, focused activation-transfer probe tests,
  `python3 -m fornax accelerator activation-transfer-probe --out
  /tmp/fornax_activation_transfer_h100_pair_20260621.json --torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --source-device cuda:0
  --destination-device cuda:1 --dtype float16 --payload-mib 16 --iterations 20
  --warmup 3 --timeout-s 180`, `python3 -m fornax test
  activation-transfer-probe --fixture
  /tmp/fornax_activation_transfer_h100_pair_20260621.json`, `python3 -m
  unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`,
  `make fornax-golden`, `make fornax-test`, and `git diff --check` all passed.


### T3 same-host pipeline-correctness simulation probe milestone

- Added `fornax.pipeline_probe` for deterministic split-pipeline correctness. The
  probe runs a tiny two-stage language-model-shaped path, transfers the stage-0
  activation to stage 1, generates tokens, and validates generated-sequence plus
  final-logit parity against a monolithic reference path.
- Added `fornax pipeline correctness-probe --out ...` and `fornax test
  pipeline-correctness-probe`; added the CPU reference golden fixture
  `fornax/golden_vectors/pipeline_correctness/fixture.json`; expanded `make
  fornax-golden` and `fornax program simulate-t1` so pipeline correctness is now
  part of the regular simulated-cluster evidence path. The one-command T1 bundle
  now validates 16 checks including `pipeline-correctness`.
- The validator enforces measured status, generated-token parity, final-logit
  tolerance, token/activation accounting, distinct logical hosts, source and
  destination hardware metadata, and rejects false
  `T3-same-host-two-gpu-simulation` claims unless the artifact is a distinct
  CUDA source/destination pair. CPU artifacts remain valid only as reference
  plumbing and CI-safe golden evidence.
- Ran the lab probe on the same-host H100 logical pair with
  `/mnt/dataprocessing/venvs/asr-data-prep/bin/python`: artifact
  `/tmp/fornax_pipeline_correctness_h100_pair_20260621.json` records
  `source_device=cuda:0`, `destination_device=cuda:1`, dtype `float32`,
  `tokens_generated=160`, `tokens_s=1584.8816475543983`,
  `activation_bytes_transferred=20480`,
  `max_abs_error=2.384185791015625e-07`, `sequences_match=true`,
  `correctness_passed=true`, generated sequences `[[1, 2, 3, 4, 6, 9, 13], [4,
  5, 6, 7, 9, 12, 16]]`, H100 source/destination names, torch
  `2.12.0+cu130`, CUDA `13.0`, and peer access true in both directions.
- This narrows the Phase 1/G2 correctness gap for the approved simulation method:
  we now have both measured same-host activation transfer and measured same-host
  split-pipeline generation parity. It still does not close real multi-host T3
  pipeline correctness, real model/tokenizer parity, aggregate concurrency
  scaling, or MoE layer/logit parity.
- Review-lens pass:
  - LLM/Model Architecture: approve with comments. The probe checks generated-token
    and logit parity for a deterministic small model, but real tokenizer/template
    and target-model layer parity remain future work.
  - Distributed Runtime/Scheduler: approve with comments. The artifact exercises
    a stage boundary and activation handoff aligned with the logical-host model;
    it is not yet a live worker process pipeline or full 1F1B scheduler path.
  - Hardware/Networking: approve with comments. The H100 run records real
    same-host CUDA placement and peer access; real inter-host network transport
    remains separate T3 evidence.
  - Low-level Software: approve. The validator prevents false accelerator claims,
    same-device claims, sequence mismatches, and activation-accounting drift.
  - Testing/Quality: approve. Regression tests cover CPU reference validity, false
    T3 claims without CUDA, same CUDA source/destination rejection, generation
    mismatch rejection, and T1 bundle integration.
- Verification: `python3 -m py_compile fornax/pipeline_probe.py fornax/cli.py
  fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, `python3 -m
  fornax pipeline correctness-probe --backend cpu-stdlib --out
  fornax/golden_vectors/pipeline_correctness/fixture.json --iterations 2
  --warmup 1 --vocab-size 17 --hidden-dim 16 --new-tokens 3 --tolerance 0.0`,
  `python3 -m fornax test pipeline-correctness-probe`, focused pipeline
  correctness and T1 integration tests, `python3 -m fornax pipeline
  correctness-probe --backend torch --torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --out
  /tmp/fornax_pipeline_correctness_h100_pair_20260621.json --source-device
  cuda:0 --destination-device cuda:1 --dtype float32 --iterations 20 --warmup 3
  --vocab-size 17 --hidden-dim 32 --new-tokens 4 --tolerance 0.0001
  --timeout-s 180`, `python3 -m fornax test pipeline-correctness-probe
  --fixture /tmp/fornax_pipeline_correctness_h100_pair_20260621.json`,
  `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q
  fornax tests`, `make fornax-golden`, `make fornax-test`, `python3 -m fornax
  program simulate-t1 --out-dir
  /tmp/fornax_t1_pipeline_correctness_validation_cli_20260621 --gpu-count 2
  --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000
  --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id
  cli-t1-pipeline-plan --request-id cli-t1-pipeline-request --plan-hash
  sha256:cli-t1-pipeline-plan --max-queue-depth 2 --max-inflight 2
  --microbatch-size 2 --timeout-ms 50` showing 16/16 checks passed, and
  `git diff --check` all passed.


### T1 throughput-scaling metric contract milestone

- Added `fornax.throughput_scaling` for a deterministic concurrency-sweep
  contract covering the Phase 2/G2 metric language: aggregate throughput should
  scale with concurrency, saturate no later than the contracted minimum
  concurrency, satisfy the provisional throughput-efficiency floor, and match the
  planner within the provisional bound.
- Added `fornax throughput scaling-simulate --out ...` and `fornax test
  throughput-scaling`; added the golden fixture
  `fornax/golden_vectors/throughput_scaling/fixture.json`; expanded `make
  fornax-golden` and `fornax program simulate-t1` so throughput scaling is part
  of the regular simulated-cluster evidence path. The one-command T1 bundle now
  validates 17 checks including `throughput-scaling`.
- The fixture sweeps concurrency `[1, 2, 4, 8, 16, 32]`, uses contracted minimum
  concurrency `16`, observes saturation at `8`, records max planner error
  `0.08` against the `0.20` provisional bound, and records throughput efficiency
  at contract `0.666666667` against the `0.60` provisional floor. The validator
  recomputes monotonicity, saturation, planner error, throughput efficiency, and
  target-met status from rows so summary fields cannot drift.
- This closes another simulation-method gap for development planning, but it is
  deliberately `measurement_kind=deterministic-simulation`; it does not claim
  real T3/T4 hardware throughput, live arrivals, real model scheduling, or
  product workload/persona validation.
- Review-lens pass:
  - Analytical Performance: approve with comments. The artifact now expresses the
    provisional efficiency, planner-accuracy, and saturation gates as
    machine-checkable math; real measured throughput remains future T3/T4 work.
  - Program Management: approve. The milestone directly reduces R-8/R-10 status
    ambiguity by separating simulated metric closure from real persona/hardware
    evidence.
  - Distributed Runtime/Scheduler: approve with comments. The sweep aligns with
    the continuous-batching and pipeline-overlap model, but live admission and
    worker scheduling remain separate runtime evidence.
  - Testing/Quality: approve. Regression tests cover valid metric contracts,
    planner-error failures, non-monotonic sweeps, late saturation, and T1 bundle
    integration.
- Verification: `python3 -m py_compile fornax/throughput_scaling.py fornax/cli.py
  fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, `python3 -m
  fornax throughput scaling-simulate --out
  fornax/golden_vectors/throughput_scaling/fixture.json --plan-id
  golden-throughput-scaling --concurrency-levels 1,2,4,8,16,32
  --contracted-min-concurrency 16 --saturation-concurrency 8
  --planner-bound-fraction 0.20 --throughput-efficiency-floor 0.60
  --sum-node-ideal-tokens-s 45 --saturated-pipeline-tokens-s 30
  --planner-bias-fraction 0.08 --jitter-fraction 0.015`, `python3 -m fornax
  test throughput-scaling`, focused throughput-scaling and T1 integration tests,
  `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q
  fornax tests`, `make fornax-golden`, `make fornax-test`, `python3 -m fornax
  program simulate-t1 --out-dir
  /tmp/fornax_t1_throughput_scaling_validation_cli_20260621 --gpu-count 2
  --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000
  --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id
  cli-t1-throughput-plan --request-id cli-t1-throughput-request --plan-hash
  sha256:cli-t1-throughput-plan --max-queue-depth 2 --max-inflight 2
  --microbatch-size 2 --timeout-ms 50` showing 17/17 checks passed, and
  `git diff --check` all passed.


### T3 same-host remote expert batch probe milestone

- Added `fornax.remote_expert_probe` for an independent remote expert batch
  probe. The probe dispatches a deterministic hidden-state batch from a source
  host to an expert host, executes a small expert MLP, returns weighted results,
  and validates parity against a local/reference expert path.
- Added `fornax moe remote-expert-probe --out ...` and `fornax test
  remote-expert-probe`; added the CPU reference golden fixture
  `fornax/golden_vectors/remote_expert_batch/fixture.json`; expanded `make
  fornax-golden` and `fornax program simulate-t1` so remote expert batches are
  part of the regular simulated-cluster evidence path. The one-command T1 bundle
  now validates 18 checks including `remote-expert-batch`.
- The validator enforces measured status, batch/expert-call accounting, transfer
  byte accounting, checksum/reference checksum presence, correctness pass,
  distinct logical hosts, source/expert hardware metadata, and rejects false
  `T3-same-host-remote-expert-simulation` claims unless the artifact uses a
  distinct CUDA source/expert device pair. CPU artifacts remain CI-safe reference
  evidence only.
- Ran the lab probe on the same-host H100 logical pair with
  `/mnt/dataprocessing/venvs/asr-data-prep/bin/python`: artifact
  `/tmp/fornax_remote_expert_batch_h100_pair_20260621.json` records
  `source_device=cuda:0`, `expert_device=cuda:1`, dtype `float32`,
  `remote_batches=20`, `expert_calls=160`,
  `expert_calls_s=27694.131916956303`, `transfer_payload_bytes=81920`,
  `max_abs_error=0.0`, `correctness_passed=true`, H100 source/expert names,
  torch `2.12.0+cu130`, CUDA `13.0`, and peer access true in both directions.
- This narrows the Phase 2.5/G2 MoE gap for the approved simulation method: remote
  expert batches are now independently measured on the two-H100 logical-host
  setup. It still does not close real multi-host T3, target-model MoE
  layer/logit parity, all-to-all routing, or product workload locality decisions.
- Review-lens pass:
  - LLM/Model Architecture: approve with comments. The probe checks an
    expert-MLP-shaped batch and weighted gather against a reference path; real
    target MoE router traces and layer/logit parity remain future work.
  - Distributed Runtime/Scheduler: approve with comments. The artifact validates
    remote expert dispatch/execution/return as an independent batch primitive,
    but not yet live worker scheduling or migration decisions under load.
  - Hardware/Networking: approve with comments. The H100 run records real
    same-host CUDA placement and peer access; real inter-host network transport
    remains separate T3 evidence.
  - Low-level Software: approve. The validator prevents false accelerator claims,
    same-device claims, accounting drift, and failed-correctness artifacts from
    being treated as milestone evidence.
  - Testing/Quality: approve. Regression tests cover CPU reference validity,
    false T3 claims without CUDA, same CUDA source/expert rejection, failed
    correctness rejection, and T1 bundle integration.
- Verification: `python3 -m py_compile fornax/remote_expert_probe.py
  fornax/cli.py fornax/t1_simulated_validation.py tests/test_fornax_planner.py`,
  `python3 -m fornax moe remote-expert-probe --backend cpu-stdlib --out
  fornax/golden_vectors/remote_expert_batch/fixture.json --iterations 2
  --warmup 1 --token-count 4 --hidden-dim 16 --intermediate-dim 32
  --expert-id 5 --tolerance 0.0`, `python3 -m fornax test
  remote-expert-probe`, focused remote expert and T1 integration tests,
  `python3 -m fornax moe remote-expert-probe --backend torch --torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --out
  /tmp/fornax_remote_expert_batch_h100_pair_20260621.json --source-device
  cuda:0 --expert-device cuda:1 --dtype float32 --iterations 20 --warmup 3
  --token-count 8 --hidden-dim 64 --intermediate-dim 128 --expert-id 5
  --tolerance 0.0001 --timeout-s 180`, `python3 -m fornax test
  remote-expert-probe --fixture
  /tmp/fornax_remote_expert_batch_h100_pair_20260621.json`, `python3 -m
  unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`,
  `make fornax-golden`, `make fornax-test`, `python3 -m fornax program
  simulate-t1 --out-dir /tmp/fornax_t1_remote_expert_validation_cli_20260621
  --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s
  12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id
  cli-t1-remote-expert-plan --request-id cli-t1-remote-expert-request
  --plan-hash sha256:cli-t1-remote-expert-plan --max-queue-depth 2
  --max-inflight 2 --microbatch-size 2 --timeout-ms 50` showing 18/18 checks
  passed, and `git diff --check` all passed.


### T3 same-host MoE layer/logit parity simulation milestone

- Adopted the two-local-GPU simulation method for the next G2/M4 gap: MoE
  layer/logit parity. Added `fornax.moe_parity`, which runs deterministic
  router top-k, expert bucketing, local plus remote expert execution, weighted
  gather, residual layer output, and logit projection, then compares the split
  path against a monolithic reference path.
- Added `fornax moe parity-probe --out ...` and `fornax test
  moe-parity-probe`; added the CPU reference golden fixture
  `fornax/golden_vectors/moe_layer_parity/fixture.json`; expanded `make
  fornax-golden` and `fornax program simulate-t1` so MoE layer parity is part of
  the regular simulated-cluster evidence path. The one-command T1 bundle now
  validates 19 checks including `moe-layer-parity`.
- The validator enforces measured status, router trace shape, top-k weight sums,
  local/remote expert coverage, expert-call accounting, remote-batch and transfer
  byte accounting, next-token parity, layer and logit tolerance, distinct logical
  hosts, source/expert hardware metadata, and rejects false
  `T3-same-host-moe-parity-simulation` claims unless the artifact uses a distinct
  CUDA source/expert device pair. CPU artifacts remain CI-safe reference evidence
  only.
- Ran the lab probe on the same-host H100 logical pair with
  `/mnt/dataprocessing/venvs/asr-data-prep/bin/python`: artifact
  `/tmp/fornax_moe_layer_parity_h100_pair_20260622.json` records
  `source_device=cuda:0`, `expert_device=cuda:1`, dtype `float32`,
  `tokens_processed=160`, `expert_calls=320`, `remote_expert_calls=160`,
  `remote_batches=40`, `transfer_payload_bytes=81920`,
  `tokens_s=1946.8277830395537`, `expert_calls_s=3893.6555660791073`,
  `max_layer_abs_error=0.0`, `max_logit_abs_error=0.0`,
  `next_tokens_match=true`, `correctness_passed=true`, H100 source/expert names,
  torch `2.12.0+cu130`, CUDA `13.0`, and peer access true in both directions.
- This narrows the Phase 2.5/G2 MoE correctness gap for the approved simulation
  method: development can now validate router-to-logit parity over logical
  source/expert hosts without waiting for a physical heterogeneous cluster. It
  still does not close real multi-host T3, target-model MoE layer parity,
  tokenizer/chat-template parity, live worker scheduling, migration under load,
  or G3 real heterogeneous frontier serving.
- Review-lens pass:
  - Program Management: approve with comments. This directly uses the agreed
    simulation method to unblock M4 development while keeping real T3/T4 gate
    evidence separate.
  - LLM/Model Architecture: approve with comments. The artifact covers
    layer-output and logit parity for a deterministic MoE-shaped layer; target
    frontier model parity remains future work.
  - Distributed Runtime/Scheduler: approve with comments. The split path makes
    local and remote expert ownership explicit, but it is not yet live worker
    scheduling or 1F1B execution under load.
  - Hardware/Networking: approve with comments. The H100 run records distinct
    same-host CUDA devices and peer access; real inter-host network and mixed
    vendor effects remain separate T3/T4 evidence.
  - Low-level Software: approve. The validator prevents false accelerator claims,
    same-device claims, routing/accounting drift, and failed parity artifacts
    from being treated as milestone evidence.
  - Testing/Quality: approve. Regression tests cover CPU reference validity,
    false T3 claims without CUDA, same CUDA source/expert rejection, failed
    correctness rejection, and T1 bundle integration.
- Verification: `python3 -m py_compile fornax/moe_parity.py fornax/cli.py
  fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, `python3 -m
  fornax moe parity-probe --backend cpu-stdlib --out
  fornax/golden_vectors/moe_layer_parity/fixture.json --iterations 2 --warmup 1
  --token-count 4 --hidden-dim 16 --intermediate-dim 32 --vocab-size 17
  --expert-count 4 --top-k 2 --tolerance 0.0`, `python3 -m fornax test
  moe-parity-probe`, focused MoE parity and T1 integration tests, `python3 -m
  fornax moe parity-probe --backend torch --torch-python
  /mnt/dataprocessing/venvs/asr-data-prep/bin/python --out
  /tmp/fornax_moe_layer_parity_h100_pair_20260622.json --source-device cuda:0
  --expert-device cuda:1 --dtype float32 --iterations 20 --warmup 3
  --token-count 8 --hidden-dim 64 --intermediate-dim 128 --vocab-size 37
  --expert-count 4 --top-k 2 --tolerance 0.0001 --timeout-s 180`, `python3 -m
  fornax test moe-parity-probe --fixture
  /tmp/fornax_moe_layer_parity_h100_pair_20260622.json`, `python3 -m fornax
  program simulate-t1 --out-dir
  /tmp/fornax_t1_moe_layer_parity_validation_cli_20260622 --gpu-count 2
  --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000
  --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id
  cli-t1-moe-parity-plan --request-id cli-t1-moe-parity-request --plan-hash
  sha256:cli-t1-moe-parity-plan --max-queue-depth 2 --max-inflight 2
  --microbatch-size 2 --timeout-ms 50` showing 19/19 checks passed, `python3 -m
  unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`,
  `make fornax-golden`, and `make fornax-test` all passed.


### T1 hot-expert migration simulation milestone

- Added `fornax.moe_migration` for the WS-C C3 expert placement / migration
  policy gap. The artifact simulates a hot remote expert on one logical host,
  recommends migration, drains/copies state, commits the placement update,
  replays routing, and verifies parity against the same deterministic MoE
  reference path used by the layer/logit parity probe.
- Added `fornax moe migration-simulate --out ...` and `fornax test
  moe-migration`; added the golden fixture
  `fornax/golden_vectors/moe_migration/fixture.json`; expanded `make
  fornax-golden` and `fornax program simulate-t1` so hot-expert migration is
  part of the regular two-logical-host simulated-cluster evidence path. The
  one-command T1 bundle now validates 20 checks including `moe-migration`.
- The validator enforces the ordered migration sequence
  `placement_snapshot_before -> hot_expert_detected -> migration_recommendation
  -> migration_plan -> drain_started -> expert_state_copied -> placement_committed
  -> routing_replayed -> parity_verified -> cleanup`, plan/request/hash
  propagation, hotness threshold, hot expert remote-before/local-after placement,
  positive remote-token-copy reduction, zero dropped tokens, next-token parity,
  layer/logit parity, and summary/result consistency.
- The golden fixture records `tier=T1-simulation`,
  `simulation_method=two-logical-host-hot-expert-migration`, `hot_expert_id=1`,
  `hotness=0.5` against threshold `0.45`, pre/post remote token copies `7 -> 1`,
  remote-token-copy reduction `6`, pre/post remote batches `2 -> 1`,
  `max_post_layer_abs_error=0.0`, `max_post_logit_abs_error=0.0`,
  `next_tokens_match=true`, `dropped_tokens=0`, `correctness_passed=true`, and
  10 ordered events.
- This narrows the Phase 2.5/M4 MoE runtime gap for the approved simulation
  method: development can now validate that a hot remote expert is migrated into
  the local logical host without changing outputs. It remains T1 simulation only;
  live worker migration, accelerator state copy, real inter-host transport,
  target-model hot expert telemetry, and T3/T4 real-cluster migration evidence
  remain open.
- Review-lens pass:
  - Program Management: approve. This converts C3 from a recommendation-only
    milestone into a machine-checkable simulated migration artifact that can be
    tracked independently from real cluster gate closure.
  - Distributed Runtime/Scheduler: approve with comments. The state transition
    and placement ownership are explicit, but a live scheduler/worker handoff
    under concurrent requests is still future work.
  - LLM/Model Architecture: approve with comments. The migration replay preserves
    deterministic MoE layer/logit parity; target-model router telemetry and
    real expert weights remain future T3/T4 work.
  - Hardware/Networking: approve with comments. The artifact uses the logical
    two-host simulation method and does not claim accelerator or real network
    migration evidence.
  - Testing/Quality: approve. Regression tests cover fixture validity, remote
    reduction, parity failure rejection, missing placement commit rejection, and
    T1 bundle integration.
- Verification: `python3 -m py_compile fornax/moe_migration.py fornax/cli.py
  fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, `python3 -m
  fornax moe migration-simulate --out
  fornax/golden_vectors/moe_migration/fixture.json --plan-id golden-moe-migration
  --request-id golden-moe-migration-request --plan-hash
  sha256:golden-moe-migration --token-count 6 --hidden-dim 16
  --intermediate-dim 32 --vocab-size 17 --expert-count 4 --top-k 2
  --hot-expert-id 1 --migration-hotness-threshold 0.45 --tolerance 0.0`,
  `python3 -m fornax test moe-migration`, focused MoE migration and T1 bundle
  tests, `python3 -m fornax program simulate-t1 --out-dir
  /tmp/fornax_t1_moe_migration_validation_cli_20260622 --gpu-count 2 --profile
  two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s
  0.0004 --slow-node-factor 0.65 --plan-id cli-t1-moe-migration-plan
  --request-id cli-t1-moe-migration-request --plan-hash
  sha256:cli-t1-moe-migration-plan --max-queue-depth 2 --max-inflight 2
  --microbatch-size 2 --timeout-ms 50` showing 20/20 checks passed, `python3 -m
  unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`,
  `make fornax-golden`, and `make fornax-test` all passed.


### T1 stage-replication simulation milestone

- Added `fornax.stage_replication` for the WS-F F3 data-parallel stage
  replication gap. The artifact simulates adding a second replica to a bottleneck
  pipeline stage, assigns microbatches across replicas, compares each replica
  output against the single-replica reference checksum, and checks simulated
  throughput gain against a floor.
- Added `fornax replication simulate --out ...` and `fornax test
  stage-replication`; added the golden fixture
  `fornax/golden_vectors/stage_replication/fixture.json`; expanded `make
  fornax-golden` and `fornax program simulate-t1` so stage replication is part of
  the regular two-logical-host simulated-cluster evidence path. The one-command
  T1 bundle now validates 21 checks including `stage-replication`.
- The validator enforces baseline vs replicated replica sets, use of every
  replicated replica, per-microbatch assignment shape, stage start/end events,
  output comparison events, speedup math, replicated makespan improvement,
  output-reference tolerance, summary/result consistency, and a simulation-only
  warning so this cannot be mistaken for real added-node scaling evidence.
- The golden fixture records `record_kind=stage-replication-simulation-contract`,
  `mode=t1-simulation`, `simulation_method=deterministic-stage-data-parallel-replication`,
  `replica_count=2`, `microbatch_count=6`, baseline/replicated makespan
  `0.258s -> 0.129s`, `speedup=2.0` against floor `1.25`, baseline/replicated
  throughput `69.76744186046511 -> 139.53488372093022` tokens/s, both replica
  IDs used, `total_tokens=18`, `max_abs_error=0.0`, `outputs_match_reference=true`,
  `correctness_passed=true`, and 28 events.
- This narrows the Phase 3/F3 and later G4 scaling path for the approved
  simulation method: development can validate replica assignment, deterministic
  output parity, and expected scaling math before real added-node lab evidence.
  It remains T1 simulation only; live replicated workers, real memory pressure,
  cross-node scheduling, and T4 elasticity/zero-drop evidence remain open.
- Review-lens pass:
  - Distributed Runtime/Scheduler: approve with comments. The assignment and
    stage-replica lifecycle are explicit and machine-checked; live worker routing
    and backpressure under replicated load remain future work.
  - Analytical Performance: approve with comments. The artifact proves the
    speedup math and throughput comparison in simulation; real added-node
    scaling still needs benchmark-of-record evidence.
  - Program Management: approve. This gives F3 a concrete T1 milestone and keeps
    it separate from G4/T4 real elasticity closure.
  - Testing/Quality: approve. Regression tests cover fixture validity, all
    replicas used, speedup below floor, output mismatch, and T1 bundle
    integration.
- Verification: `python3 -m py_compile fornax/stage_replication.py fornax/cli.py
  fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, `python3 -m
  fornax replication simulate --out
  fornax/golden_vectors/stage_replication/fixture.json --plan-id
  golden-stage-replication --bottleneck-stage-index 1 --microbatch-token-counts
  4,4,3,3,2,2 --baseline-replica-id stage-1-replica-0 --added-replica-id
  stage-1-replica-1 --baseline-stage-time-s-per-token 0.014
  --replicated-stage-time-s-per-token 0.014 --transfer-overhead-s 0.001
  --speedup-floor 1.25 --tolerance 0.0`, `python3 -m fornax test
  stage-replication`, focused stage-replication and T1 bundle tests, `python3 -m
  fornax program simulate-t1 --out-dir
  /tmp/fornax_t1_stage_replication_validation_cli_20260622 --gpu-count 2
  --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000
  --link-latency-s 0.0004 --slow-node-factor 0.65 --plan-id
  cli-t1-stage-replication-plan --request-id cli-t1-stage-replication-request
  --plan-hash sha256:cli-t1-stage-replication-plan --max-queue-depth 2
  --max-inflight 2 --microbatch-size 2 --timeout-ms 50` showing 21/21 checks
  passed, `python3 -m unittest tests.test_fornax_planner`, `python3 -m
  compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all
  passed.
