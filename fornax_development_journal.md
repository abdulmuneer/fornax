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



### T1 resilience-replay simulation milestone

- Added `fornax.resilience` for the WS-E E4 / G4 single-node-loss replay gap. The artifact simulates the two-GPU development method as two logical hosts: one logical host fails while requests are in flight, checkpoints are used to schedule replay on the surviving logical host, and completed token streams are compared against deterministic references.
- Added `fornax resilience replay-simulate --out ...` and `fornax test resilience-replay`; added the golden fixture `fornax/golden_vectors/resilience_replay/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so replay is part of the regular two-logical-host simulated-cluster evidence path. The one-command T1 bundle now validates 22 checks including `resilience-replay`.
- The validator enforces the replay lifecycle (`request_started`, `checkpoint_recorded`, `node_loss_detected`, `replay_scheduled`, `replay_started`, `replay_completed`, `request_completed`, `cleanup`), one node-loss event, exactly one replay schedule per request, replay start/completion for every request, reference-token recovery, zero dropped tokens, zero duplicate tokens, replay-delay budget, event-count consistency, and a simulation-only warning so this cannot be mistaken for real T4 fault-tolerance evidence.
- The golden fixture records `record_kind=resilience-replay-simulation-contract`, `mode=t1-simulation`, `simulation_method=single-node-loss-replay`, `request_count=3`, `in_flight_request_count=3`, `replayed_request_count=3`, `dropped_request_count=0`, `dropped_token_count=0`, `duplicate_token_count=0`, `max_abs_error=0.0`, `max_replay_delay_s=0.01` against budget `0.025`, `replay_delay_within_budget=true`, `zero_dropped_in_flight=true`, `correctness_passed=true`, and 20 events.
- This lets development keep moving with the approved simulation method: we can validate replay bookkeeping, deterministic recovery, and zero-drop semantics before real heterogeneous-cluster testing. It remains T1 simulation only; live process death, real network partitions, persisted replay logs, scheduler failover under load, and T4 node-loss evidence remain open.
- Review-lens pass:
  - Distributed Runtime/Scheduler: approve with comments. Request lifecycle, replay scheduling, and survivor-host recovery are explicit and machine-checked; real worker ownership transfer remains future work.
  - SRE/Operations: approve with comments. The artifact verifies zero-drop accounting and cleanup events; real failure injection and observability under process loss remain required.
  - Program Management: approve. This unblocks a concrete E4/G4 development milestone without treating it as real T4 closure.
  - Testing/Quality: approve. Regression tests cover fixture validity, zero dropped replay, dropped-token rejection, duplicate-schedule rejection, late-replay rejection, and T1 bundle integration.
  - Hardware/Networking: approve with comments. Two local GPUs can be treated as two logical hosts for simulation; real heterogeneous fabric and host-loss behavior still need lab validation.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/resilience.py`, `python3 -m fornax resilience replay-simulate --out fornax/golden_vectors/resilience_replay/fixture.json --plan-id golden-resilience-replay --failed-node-id logical-host-1 --replay-node-id logical-host-0 --checkpoint-token-index 2 --node-loss-time-s 0.050 --replay-delay-s 0.010 --token-time-s 0.006 --max-replay-delay-s 0.025 --vocab-size 97`, `python3 -m fornax test resilience-replay`, focused resilience replay tests, focused T1 bundle test, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_resilience_replay_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 22/22 checks passed, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.



### T1 serving-adapter simulation milestone

- Added `fornax.serving` for the WS-H H1/H3 serving-surface gap and the LLM semantic seam in plan v3 §5.7. The artifact simulates both surfaces the plan requires: OpenAI-compatible chat completions and Ignis `Engine` via `FornaxBackend`, then proves both normalize into the same Engine seam contract.
- Added `fornax serving adapter-simulate --out ...` and `fornax test serving-adapter`; added the golden fixture `fornax/golden_vectors/serving_adapter/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so serving-surface normalization is part of the regular two-logical-host simulated-cluster evidence path. The one-command T1 bundle now validates 23 checks including `serving-adapter`.
- The validator reuses the existing `engine_seam` projection and additionally enforces required serving surfaces, OpenAI request shape, endpoint path, request-to-engine field normalization, sampling mapping, template/tokenizer hash preservation, OpenAI final response mapping, OpenAI stream chunk alignment with Engine stream events, cancellation mapping, error-to-OpenAI status mapping, lifecycle events, summary consistency, and a simulation-only warning so this cannot be mistaken for a live HTTP endpoint.
- The golden fixture records `record_kind=serving-adapter-simulation-contract`, `mode=t1-simulation`, `simulation_method=openai-and-ignis-to-engine-seam-roundtrip`, `surface_count=2`, `openai_chunk_count=5`, `engine_stream_event_count=5`, `tool_call_count=1`, `structured_output=true`, `template_hash_recorded=true`, `tokenizer_hash_recorded=true`, `cancellation_mapped=true`, `error_mapped=true`, `correctness_passed=true`, and 10 lifecycle events.
- This narrows the Phase 1/H3 and Phase 3/H1 serving path for the approved simulation method: development can validate API normalization, stable Engine handoff, stream mapping, cancellation, and error semantics before a real server exists. It remains T1 simulation only; live HTTP routing, auth, SSE framing, client disconnect handling, real Ignis integration, and production endpoint behavior remain open.
- Review-lens pass:
  - High-level Software/API: approve with comments. The public surfaces are named as user-facing concepts (`/v1/chat/completions`, `FornaxBackend`) rather than internal scheduler details; live endpoint UX and docs remain future work.
  - LLM Expertise: approve with comments. Messages, tools, response format, stop sequences, sampling, template/tokenizer hashes, structured output, streaming, cancellation, and error results are machine-checked; real tokenizer/rendering behavior remains future work.
  - Networking/Serving: approve with comments. Request flow, trust boundaries, stream chunking, cancellation, and error mapping are explicit; real HTTP/SSE, auth, timeout, and disconnect behavior remain required.
  - System Engineering: approve. The artifact composes serving semantics with the existing Engine seam instead of creating a parallel contract.
  - Program Management: approve. This creates a concrete H1/H3 development milestone while keeping it separate from G2/G3 live serving gates.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/serving.py tests/test_fornax_planner.py`, `python3 -m fornax serving adapter-simulate --out fornax/golden_vectors/serving_adapter/fixture.json --plan-id golden-serving-adapter --request-id golden-serving-request --model qwen3-moe-class-target --max-tokens 64`, `python3 -m fornax test serving-adapter`, focused serving-adapter tests, focused T1 bundle test, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_serving_adapter_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 23/23 checks passed, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.



### Planner placement-explanations milestone

- Added planner-native placement explanations for the WS-G G3 requirement: placement output now records why nodes were selected, excluded, or demoted/slow. This is attached directly to `PlacementPlan.to_dict()`, so `fornax plan`, preflight `placement.json`, validation consumers, and target-contract drafts share one source of truth.
- Added `PlacementExplanation` to the planner model and exported it from `fornax.planner`. The search path now emits primary-stage selections, data-parallel replica selections, excluded-node reasons such as unsupported activation dtype or non-stage-capable nodes, and demotion reasons for slower stage-capable nodes that receive reduced primary ownership or replica roles.
- Extended `fornax target draft` evidence with `placement_explanations` and a markdown `## Placement Explanations` table. The smoke draft at `/tmp/fornax_plan_explanations_cli/v0-target-contract.md` shows the slow node demoted because it is slower than the fastest stage-capable node, and the machine-readable block carries the same explanation row.
- This narrows the Phase 3/G3 operator-facing diagnosability gap before real heterogeneous hardware: infeasible and feasible plans are now explainable by the planner itself rather than only by observability fixtures. It does not claim real G3 serving; hardware calibration, measured links, and operator UX around the explanations remain future work.
- Review-lens pass:
  - High-level Software/API: approve. The plan output names user-facing decisions (`selected`, `excluded`, `demoted`) and gives actionable reasons instead of exposing only internal scoring.
  - System Engineering: approve with comments. Explanations are generated in the planner and flow through CLI/preflight/target draft; future work should connect them to live observability events.
  - Analytical Performance: approve with comments. Slow-node demotion includes compute ratios and stage metrics; real calibration evidence remains needed before using these as final performance claims.
  - Documentation: approve. Generated target contracts now expose a human-readable explanation table and the same data in the machine block.
  - Program Management: approve. This creates concrete G3 progress while preserving the distinction between explainable simulation/planning evidence and real heterogeneous frontier serving closure.
- Verification: `python3 -m py_compile fornax/planner/model.py fornax/planner/search.py fornax/target_contract.py tests/test_fornax_planner.py`, focused planner explanation and target-contract tests, `python3 -m fornax plan --target fornax/golden_plans/v0_target_contract_fixture.md --inventory /tmp/fornax_plan_explanations_cli/inventory.json --links /tmp/fornax_plan_explanations_cli/links.json --out /tmp/fornax_plan_explanations_cli/placement.json`, `python3 -m fornax target draft fornax/golden_plans/v0_target_contract_fixture.md --inventory /tmp/fornax_plan_explanations_cli/inventory.json --links /tmp/fornax_plan_explanations_cli/links.json --out /tmp/fornax_plan_explanations_cli/v0-target-contract.md`, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.



### T1 ops-lifecycle simulation milestone

- Added `fornax.ops_lifecycle` for the WS-I I1/I2 operator lifecycle path toward G5 productization: simulated `cluster.yaml`, `model.yaml`, and `placement.json` inputs now drive deploy, drain, upgrade, restart, rollback, and node replacement over two logical GPU hosts. This uses the approved development method of treating the two local GPUs as two logical hosts, while preserving real heterogeneous-cluster validation as a later gate.
- Added `fornax ops lifecycle-simulate --out ...` and `fornax test ops-lifecycle`; added the golden fixture `fornax/golden_vectors/ops_lifecycle/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so ops lifecycle is part of the regular simulated-cluster evidence path. The one-command T1 bundle now validates 24 checks including `ops-lifecycle`.
- The validator enforces required operator configs, node admission, health checks, completed lifecycle actions, drain-before-mutation ordering, zero dropped in-flight requests, rollback verification, node replacement, traffic restoration, summary consistency, and a simulation-only warning so this cannot be mistaken for G5 product-ops closure.
- The golden fixture records `record_kind=ops-lifecycle-simulation-contract`, `mode=t1-simulation`, `simulation_method=operator-lifecycle-two-logical-hosts`, `action_count=6`, `completed_action_count=6`, `event_count=28`, `config_artifacts_present=true`, `drain_before_mutation=true`, `dropped_in_flight_count=0`, `rollback_verified=true`, `node_replace_verified=true`, `active_node_count=2`, and `correctness_passed=true`.
- This unblocks development milestones around operator install/upgrade/drain/rollback semantics without waiting for a real heterogeneous cluster. It remains T1 simulation only; live deployment, real endpoint auth, encrypted transport, node admission against actual hosts, process supervision, audit logs, real rollback packages, and firm operator handoff remain open before G5.
- Review-lens pass:
  - SRE/Operations: approve with comments. The artifact verifies deploy/upgrade/restart/rollback/node-replace ordering and zero-drop drains; real deployment automation and failure injection remain required.
  - Networking/Serving: approve with comments. Simulated node identity, plan integrity, endpoint naming, and traffic restoration are explicit; real auth, TLS/encryption posture, timeout behavior, and audit trails remain future work.
  - High-level Software/API: approve. Operator-facing concepts are represented as `cluster.yaml`, `model.yaml`, `placement.json`, and lifecycle actions rather than hidden scheduler internals.
  - Program Management: approve. This creates concrete I1/I2 progress toward G5 while keeping the distinction between simulation evidence and real product-ops closure.
  - Testing/Quality: approve. Regression tests cover fixture validity, required actions/configs, missing drain rejection, dropped in-flight rejection, missing operator config rejection, and T1 bundle integration.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/ops_lifecycle.py tests/test_fornax_planner.py`, `python3 -m fornax ops lifecycle-simulate --out fornax/golden_vectors/ops_lifecycle/fixture.json --plan-id golden-ops-lifecycle --cluster-id golden-sim-cluster --model-id qwen3-moe-class-target --initial-version v0.1.0 --target-version v0.2.0 --node-ids logical-host-0,logical-host-1 --replacement-node-id logical-host-2 --in-flight-requests 4`, `python3 -m fornax test ops-lifecycle`, focused ops lifecycle tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_ops_lifecycle_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 24/24 checks passed, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.



### T1 onboarding-methodology simulation milestone

- Added `fornax.onboarding` for the WS-I I3 onboarding/glossary/benchmark-methodology path toward G5 productization. The artifact defines operator, developer, benchmark-owner, and reviewer tracks; required onboarding documents; required glossary coverage; and the benchmark-of-record methodology boundary from the quality-governance document.
- Added `fornax ops onboarding-simulate --out ...` and `fornax test onboarding-methodology`; added the golden fixture `fornax/golden_vectors/onboarding_methodology/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so onboarding methodology is part of the regular simulated-cluster evidence path. The one-command T1 bundle now validates 25 checks including `onboarding-methodology`.
- The validator enforces required tracks (`operator`, `developer`, `benchmark-owner`, `reviewer`), required docs (`quickstart`, `operator-runbook`, `developer-workflow`, `benchmark-methodology`, `glossary`), glossary terms, lab-reference benchmark-of-record boundary, correctness-before-throughput, reproducibility inputs, gate mapping, operator handoff checklist, summary consistency, and a simulation-only warning so this cannot be mistaken for G5 product-GA closure.
- The golden fixture records `record_kind=onboarding-methodology-contract`, `mode=t1-simulation`, `simulation_method=operator-onboarding-and-benchmark-methodology`, `track_count=4`, `document_count=5`, `glossary_term_count=10`, `lab_reference_required=true`, `correctness_first=true`, `onboarding_complete_for_simulation=true`, and `product_ga_complete=false`.
- This unblocks development milestones around operator onboarding and benchmark methodology without waiting for a real heterogeneous cluster or firm handoff. It remains T1 simulation only; rendered product docs, real install validation, design-partner onboarding, lab-reference benchmark runs, and Sponsor G5 closure remain open.
- Review-lens pass:
  - Program Management: approve. I3 now has a machine-checkable artifact and bundle integration while preserving G5 as open.
  - SRE/Operations: approve with comments. Operator quickstart/runbook prerequisites, first-run commands, success evidence, and escalation triggers are explicit; real operational handoff remains future work.
  - Documentation: approve with comments. Required document names and glossary coverage are validated; polished human-facing docs still need product editing before GA.
  - Benchmark/Analytical Performance: approve with comments. The methodology requires lab-reference, raw logs, ledger records, version manifests, and correctness artifacts before performance claims; actual benchmark-of-record execution remains future T2-T4/G5 work.
  - Testing/Quality: approve. Regression tests cover fixture validity, required materials, missing glossary rejection, missing lab-reference boundary rejection, empty first-run command rejection, missing required document rejection, and T1 bundle integration.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/onboarding.py tests/test_fornax_planner.py`, `python3 -m fornax ops onboarding-simulate --out fornax/golden_vectors/onboarding_methodology/fixture.json --plan-id golden-onboarding-methodology --package-id golden-operator-onboarding --benchmark-id golden-benchmark-methodology`, `python3 -m fornax test onboarding-methodology`, focused onboarding methodology tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_onboarding_methodology_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 25/25 checks passed, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.



### T1 program-governance simulation milestone

- Added `fornax.program_governance` for the WS-X continuous governance path: X1 gate operation and decision log, X2 RAID upkeep and external watch, and X3 cadence/status reporting. The artifact captures decision-log discipline, DEC-005 pending status, allowed gate outcomes, stable RAID IDs, D-1 source precedence, weekly/gate cadence artifacts, and R-10 status-drift controls.
- Added `fornax program governance-simulate --out ...` and `fornax test program-governance`; added the golden fixture `fornax/golden_vectors/program_governance/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so governance controls are part of the regular simulated-cluster evidence path. The one-command T1 bundle now validates 26 checks including `program-governance`.
- The validator enforces required DEC IDs, `DEC-005` remaining pending, Sponsor as decision authority, allowed outcomes `{PROCEED, ITERATE, NARROW, KILL}`, silent-PROCEED forbidden, required RAID IDs (`R-4`, `R-8`, `R-10`, `A-1`, `A-2`, `A-5`, `I-1`..`I-6`, `D-1`..`D-4`), source-precedence rank 1 as local-probe gate of record, required cadence artifacts, active X1/X2/X3 controls, summary consistency, and a simulation-only warning so this cannot be mistaken for G1/G5 closure.
- The golden fixture records `record_kind=program-governance-contract`, `mode=t1-simulation`, `simulation_method=program-governance-x1-x3-controls`, `decision_count=6`, `control_count=6`, `cadence_artifact_count=6`, `dec005_pending=true`, `g1_gate_ready=false`, `silent_proceed_forbidden=true`, `status_drift_controlled=true`, `external_watch_rank1_required=true`, and `simulation_only=true`.
- This narrows the program-management implementation gap by making governance rules machine-checkable without changing their authority. It remains T1 simulation only; Sponsor gate decisions, live RAID updates, real external-watch probe outcomes, staffing closure, and DEC-005 remain open.
- Review-lens pass:
  - Program Management: approve. WS-X now has one validator for gate discipline, decision logging, RAID/watch upkeep, cadence artifacts, and status-drift control while preserving Sponsor authority.
  - Organizational/TL: approve with comments. The source-precedence ladder and DEC flow are explicit; real decision recording still happens in the program docs at gate time.
  - SRE/Operations: approve. Cadence artifacts and phase0-status linkage keep status production runnable without oral context.
  - Security/Product: approve with comments. The governance control prevents simulated/planned evidence from being treated as product or gate proof; real security closure still follows E3/G3/G5 gates.
  - Testing/Quality: approve. Regression tests cover fixture validity, control coverage, DEC-005 proceed-claim rejection, R-10 removal rejection, blog-as-gate rejection, missing cadence artifact rejection, and T1 bundle integration.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/program_governance.py tests/test_fornax_planner.py`, `python3 -m fornax program governance-simulate --out fornax/golden_vectors/program_governance/fixture.json --plan-id golden-program-governance --report-date 2026-06-22 --plan-version v3 --current-gate G1`, `python3 -m fornax test program-governance`, focused program governance tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_program_governance_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 26/26 checks passed, `python3 -m unittest tests.test_fornax_planner`, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.




### T1 stage-host simulation milestone

- Added `fornax.stage_host` for the WS-B B2/B3/B4 runtime/MAX surgery path: a deterministic layer-group stage host, explicit activation/KV boundary operations, stage-local KV ownership, lifecycle events, and slow-correct reference parity. This follows the approved two-local-GPU-as-two-logical-host simulation method while keeping real heterogeneous-cluster validation as a later gate.
- Added `fornax runtime stage-host-simulate --out ...` and `fornax test stage-host`; added the golden fixture `fornax/golden_vectors/stage_host/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so stage-host execution is part of the regular simulated-cluster evidence path. The one-command T1 bundle now validates 27 checks including `stage-host`.
- The validator enforces required boundary ops (`activation_in`, `activation_out`, `kv_read`, `kv_write`), tensor shapes and payload hashes, KV owner preservation, required lifecycle events, exact output/reference equality, summary consistency, and simulation-only graphlet claims. It rejects missing boundary ops, output mismatch, KV owner drift, and any claim that the artifact is measured or hardware-accelerated MAX execution.
- The golden fixture records `record_kind=stage-host-simulation-contract`, `mode=t1-simulation`, `simulation_method=deterministic-layer-group-stage-host`, `backend=simulated-max-graphlet`, `max_graphlet_status=planned`, `reference_path=cpu-stdlib-slow-correct`, `boundary_op_count=4`, `event_count=8`, `max_abs_error=0.0`, `kv_ownership_preserved=true`, `graphlet_claim_is_simulated=true`, and `correctness_passed=true`.
- This narrows the Phase 1 runtime milestone gap without blocking on a real heterogeneous cluster: development can validate stage-host IO, KV handoff, and reference correctness before real MAX graphlets or live network transport exist. It remains T1 simulation only; real MAX graph execution, custom boundary ops in the runtime, measured accelerator behavior, real distributed correctness, and G2/G3 closure remain open.
- Review-lens pass:
  - Runtime/MAX: approve with comments. The layer-group and graphlet boundary are explicit and machine-checked, but the backend is intentionally `simulated-max-graphlet`; real MAX graphlet execution remains future work.
  - Distributed Runtime/Scheduler: approve. The artifact models predecessor/successor stage boundaries and can sit inside the two-logical-host T1 bundle without changing scheduler semantics.
  - Low-level Software: approve with comments. Payload hashes and shapes protect ABI-style handoffs; real custom ops, memory ownership, device synchronization, and zero-copy behavior remain open.
  - LLM/Correctness: approve. Slow-correct reference output is compared exactly with `max_abs_error=0.0` for the deterministic layer group.
  - Testing/Quality: approve. Regression tests cover fixture validity, simulation validity, missing boundary op rejection, output mismatch rejection, measured-MAX claim rejection, KV owner mismatch rejection, and T1 bundle integration.
  - Program Management: approve. This creates concrete B2-B4 progress under the simulation method while preserving the distinction between development evidence and G2/G3 gate evidence.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/stage_host.py tests/test_fornax_planner.py`, `python3 -m fornax runtime stage-host-simulate --out fornax/golden_vectors/stage_host/fixture.json --plan-id golden-stage-host --request-id golden-stage-host-request --stage-id stage-1 --logical-host-id logical-host-1 --predecessor-stage-id stage-0 --successor-stage-id stage-2 --layer-start 12 --layer-count 2 --token-count 3 --hidden-dim 4 --dtype fp16 --tolerance 0.0`, `python3 -m fornax test stage-host`, focused stage-host and T1 bundle tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_stage_host_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 27/27 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 179 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.




### T1 trust-boundary simulation milestone

- Added `fornax.trust_boundary` for the WS-E E3 trust-boundary path: simulated node identity manifests, endpoint auth challenges, signed capability-token claims, plan-integrity tags, anonymous rejection, stale-plan rejection, duplicate replay-nonce rejection, and unknown-identity rejection over the two-logical-host development cluster.
- Added `fornax transport trust-boundary-simulate --out ...` and `fornax test trust-boundary`; added the golden fixture `fornax/golden_vectors/trust_boundary/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so trust-boundary evidence is part of the regular simulated-cluster path. The one-command T1 bundle now validates 28 checks including `trust-boundary`.
- The validator enforces `allow_anonymous=false`, endpoint-auth/plan-hash/replay-nonce requirements, required token claims, deterministic simulated signatures, admitted identities, accepted-auth references for messages, plan-tag consistency, stale-plan rejection, duplicate-nonce rejection, unknown-identity rejection, summary consistency, and a simulation-only warning so this cannot be mistaken for real TLS/mTLS or product auth evidence.
- The golden fixture records `record_kind=trust-boundary-simulation-contract`, `mode=t1-simulation`, `simulation_method=node-identity-endpoint-auth-plan-integrity`, `auth_scheme=hmac-sha256-simulated-capability-token`, `identity_count=3`, `accepted_auth_count=2`, `rejected_auth_count=4`, `authenticated_message_count=1`, `anonymous_rejected=true`, `stale_plan_rejected=true`, `duplicate_nonce_rejected=true`, `unknown_identity_rejected=true`, and `correctness_passed=true`.
- This narrows E3 without blocking on real network security infrastructure: development can validate identity/auth/plan-tag invariants before TCP/RDMA/TB-IP/shm implementations or production auth exist. It remains T1 simulation only; real TLS/mTLS, key distribution, endpoint admission service, audit logs, cross-host network transport, and G3 security closure remain open.
- Review-lens pass:
  - Networking/Security: approve with comments. The trust boundary now has explicit identity, endpoint auth, plan-tag, stale-plan, anonymous, and replay controls; real cryptographic deployment and auditability remain future work.
  - Distributed Runtime/Scheduler: approve. The artifact composes with existing transport and worker plan hashes without changing scheduling behavior, so transport mechanics and auth semantics stay separately testable.
  - SRE/Operations: approve with comments. Rejection reasons and cleanup events are explicit enough for runbook and doctor integration later; real admission service operations remain open.
  - Testing/Quality: approve. Regression tests cover fixture validity, simulated happy path, signature tamper rejection, missing stale-plan rejection, anonymous policy weakening, unknown identity acceptance, duplicate nonce omission, and T1 bundle integration.
  - Program Management: approve. This gives WS-E3 a concrete T1 milestone while preserving the distinction between simulation evidence and G3 product-security evidence.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/trust_boundary.py tests/test_fornax_planner.py`, `python3 -m fornax transport trust-boundary-simulate --out fornax/golden_vectors/trust_boundary/fixture.json --plan-id golden-trust-boundary --request-id golden-trust-boundary-request --plan-hash sha256:golden-trust-boundary --cluster-id golden-sim-cluster --token-ttl-s 30.0`, `python3 -m fornax test trust-boundary`, focused trust-boundary and T1 bundle tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_trust_boundary_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 28/28 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 186 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.




### T1 metrics-ledger simulation milestone

- Added `fornax.metrics_ledger` for the WS-G G2 queue-depth, backpressure, memory, and KV telemetry path. The artifact simulates a deterministic run over the two-logical-host development cluster and records metric samples, runtime events, derived counters, gauges, stage-latency histograms, and alerts.
- Added `fornax observability metrics-simulate --out ...` and `fornax test metrics-ledger`; added the golden fixture `fornax/golden_vectors/metrics_ledger/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so metrics-ledger consistency is part of the regular simulated-cluster evidence path. The one-command T1 bundle now validates 29 checks including `metrics-ledger`.
- The validator recomputes counters and gauges from events/samples, enforces bounded queue depth, inflight limits, KV page limits, memory pressure fractions, required backpressure/memory/KV alerts, histogram bucket consistency, summary consistency, and a simulation-only warning so this cannot be mistaken for live runtime telemetry or dashboard evidence.
- The golden fixture records `record_kind=metrics-ledger-simulation-contract`, `mode=t1-simulation`, `simulation_method=queue-backpressure-kv-memory-metrics`, `sample_count=5`, `event_count=15`, `alert_count=3`, `max_queue_depth_observed=4`, `max_memory_pressure_fraction=0.9`, `kv_pages_evicted_total=2`, `stage_latency_sample_count=10`, and `correctness_passed=true`.
- This narrows G2 without waiting for live workers: development can validate metric semantics, alert thresholds, and aggregate consistency before a real runtime emits Prometheus/OpenTelemetry/dashboard data. It remains T1 simulation only; live metric exporters, production dashboards, retention, alert routing, and measured SLOs remain open.
- Review-lens pass:
  - Observability/SRE: approve with comments. Queue depth, backpressure, KV, memory pressure, eviction, alerts, and cleanup are machine-checkable; live exporter and dashboard wiring remain future work.
  - Distributed Runtime/Scheduler: approve. Metrics compose with existing scheduler/transport semantics and keep bounded admission visible as an auditable ledger.
  - Low-level Software: approve with comments. KV-page and memory-pressure aggregates are internally consistent, but real allocator ownership, fragmentation measurement, and eviction policy still need runtime integration.
  - Analytical Performance: approve. The validator rejects stale counters and histogram summaries, so metric claims must match raw samples.
  - Testing/Quality: approve. Regression tests cover fixture validity, derived metrics, counter mismatch, queue overflow, missing memory alert, pressure mismatch, histogram mismatch, and T1 bundle integration.
  - Program Management: approve. This gives WS-G2 a concrete T1 milestone while preserving the distinction between simulated metrics and G2/G3 hardware evidence.
- Verification: `python3 -m py_compile fornax/cli.py fornax/t1_simulated_validation.py fornax/metrics_ledger.py tests/test_fornax_planner.py`, `python3 -m fornax observability metrics-simulate --out fornax/golden_vectors/metrics_ledger/fixture.json --plan-id golden-metrics-ledger --request-id golden-metrics-ledger-request --max-queue-depth 4 --max-inflight 3 --kv-page-limit 16 --memory-limit-bytes 85899345920 --memory-warning-fraction 0.85 --memory-critical-fraction 0.95 --sample-period-ms 10.0`, `python3 -m fornax test metrics-ledger`, focused metrics-ledger and T1 bundle tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_metrics_ledger_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 29/29 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 193 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.



### T1 state-ownership simulation milestone

- Added `fornax.state_ownership` for the WS-H H1 end-to-end lifecycle and state-ownership gap. The artifact simulates the approved development method: two local GPUs treated as two logical hosts, with request, engine, scheduler, microbatch, activation, transport, KV, stream, cancellation, and cleanup ownership recorded as an auditable lifecycle ledger.
- Added `fornax serving state-ownership-simulate --out ...` and `fornax test state-ownership`; added the golden fixture `fornax/golden_vectors/state_ownership/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so state ownership is part of the regular simulated-cluster evidence path. The one-command T1 bundle now validates 30 checks including `state-ownership`.
- The validator enforces resource uniqueness, required resource kinds, known owners, monotonic ownership transitions, `from_owner` matching current owner, no transitions after release, required lifecycle events, no multiple active owners in snapshots, all required resources released, summary consistency, and a simulation-only warning so this cannot be mistaken for live serving runtime evidence.
- The golden fixture records `record_kind=state-ownership-simulation-contract`, `mode=t1-simulation`, `simulation_method=end-to-end-serving-state-ownership`, `resource_count=11`, `transition_count=34`, `snapshot_count=4`, `terminal_released_count=11`, `stale_resource_count=0`, `dual_owner_detected=false`, `normal_request_terminal_owner=released`, `cancel_request_terminal_owner=released`, and `correctness_passed=true`.
- This unblocks lifecycle/state-ownership milestones without waiting for a real heterogeneous cluster. It remains T1 simulation only; live HTTP request objects, real engine contexts, actual device memory ownership, cross-process cancellation, transport backpressure under load, and real heterogeneous-cluster cleanup evidence remain open.
- Review-lens pass:
  - High-level Software/API: approve with comments. The ledger follows user-facing request and stream concepts through the Engine seam; live API behavior remains future validation.
  - Distributed Runtime/Scheduler: approve. Scheduler admission, microbatch ownership, cancellation return, and cleanup are explicit and machine-checked across logical hosts.
  - Low-level Software: approve with comments. Ownership and release invariants are now explicit, but real tensor/KV/device-memory lifetime and zero-copy handoff still need runtime integration.
  - SRE/Operations: approve with comments. The artifact gives operators an auditable ownership trail and stale-resource count; live telemetry and incident-time diagnostics remain future work.
  - Testing/Quality: approve. Regression tests cover fixture validity, happy-path cleanup, wrong owner rejection, missing cleanup release, dual active owner rejection, stale summary rejection, and T1 bundle integration.
  - Program Management: approve. This creates concrete H1 progress under the two-GPU simulation method while preserving the distinction between development evidence and real heterogeneous-cluster gate evidence.
- Verification: `python3 -m py_compile fornax/state_ownership.py fornax/cli.py fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, `python3 -m fornax serving state-ownership-simulate --out fornax/golden_vectors/state_ownership/fixture.json --plan-id golden-state-ownership --request-id golden-state-ownership-request --cancel-request-id golden-state-ownership-cancel --model qwen3-moe-class-target`, `python3 -m fornax test state-ownership`, focused state-ownership and T1 bundle tests, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_state_ownership_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 30/30 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 199 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-golden`, `make fornax-test`, and `git diff --check` all passed.



### Local H100 accelerator smoke bundle milestone

- Added `fornax.local_accelerator_smoke` and `fornax program local-accelerator-smoke` so local accelerator evidence is produced as a repeatable program bundle instead of ad hoc probe files. The command runs the existing expert-MLP probe and optional same-host activation-transfer probe, writes their artifacts, validates them, and records a bundle-level policy check.
- This follows the current execution policy: use two local GPUs as two logical machines and keep implementation moving across the WBS. Mac/AMD and real heterogeneous-cluster troubleshooting are deferred validation tasks, not development blockers.
- The bundle is deliberately honest about scope. It can pass local H100/T2 smoke readiness and same-host two-GPU transfer simulation, but records `g2_g3_gate_evidence=false` and warns that it is not Apple rank-1 probe evidence, target-model parity, real 2-3 node pipeline evidence, or heterogeneous G2/G3 closure evidence.
- Added CI-safe unit coverage with `--allow-reference`/`require_accelerator=False` behavior so CPU reference paths can validate bundle shape without pretending to satisfy accelerator policy. A CPU reference bundle passes local smoke validation but remains `t2_smoke_passed=false`; requiring accelerator evidence rejects it.
- Ran the real command on this machine using `/mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python` with `cuda:0` and `cuda:1` as two logical hosts. The bundle at `/tmp/fornax_local_accelerator_smoke_h100_20260622` passed 3/3 checks with `expert_accelerator_measured=true`, `activation_transfer_accelerator_measured=true`, expert device `NVIDIA H100 80GB HBM3`, expert throughput about `593.69` tokens/s for the tiny probe, and activation-transfer bandwidth about `243.29` GiB/s.
- Review-lens pass:
  - Hardware: approve with comments. The command records concrete CUDA devices and H100 names; it still does not stand in for Mac/AMD or multi-host fabric.
  - Low-level Software: approve. Probe outputs are validated for measured correctness, device identity, transfer byte counts, and policy overclaim boundaries.
  - Distributed Runtime/Scheduler: approve with comments. Treating `cuda:0` and `cuda:1` as logical hosts aligns with the simulated-cluster method; real cross-host behavior remains later.
  - Testing/Quality: approve. Unit tests cover CI reference mode and rejection of CPU reference as required accelerator evidence; full regression stays green.
  - Program Management: approve. This reduces the T2/local-smoke gap while preserving G1/G2/G3 gate honesty and avoiding blockers on unavailable hardware.
- Verification: `python3 -m py_compile fornax/local_accelerator_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local accelerator smoke tests, `python3 -m fornax program local-accelerator-smoke --out-dir /tmp/fornax_local_accelerator_smoke_h100_20260622 --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --expert-device cuda:0 --expert-dtype float16 --expert-iterations 10 --expert-warmup 2 --expert-batch-tokens 8 --expert-hidden-dim 64 --expert-intermediate-dim 128 --expert-count 4 --expert-top-k 2 --transfer-source-device cuda:0 --transfer-destination-device cuda:1 --transfer-dtype float16 --transfer-iterations 8 --transfer-warmup 2 --transfer-payload-mib 16 --logical-source-host logical-host-0 --logical-destination-host logical-host-1 --timeout-s 180` showing 3/3 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 201 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-golden`, `make fornax-test`, and `git diff --check` all passed.


### Program management sprint backlog materialization

- Materialized the full Fornax sprint backlog under `docs/fornax/program_management/sprints/` instead of leaving only the Phase-0 evidence sprint. The new sprint index links Phase 0, Phase 1 worker/transport, Phase 2 continuous batching, Phase 2.5 MoE runtime, Phase 3 heterogeneous frontier, Phase 4 resilience/elasticity, and Phase 5 productization/GA.
- Updated the program-management README and Phase-0 sprint note to reflect the current execution rule: implementation can continue through the full backlog using simulation and local two-GPU logical-host validation, but formal G2-G5 gate closure still requires the real evidence named in `04-stage-gates.md`.
- Review-lens pass:
  - Organizational Skill: approve. The sprint folder now has execution structure for all roadmap phases, not only the originally active G1 evidence sprint.
  - Documentation: approve with comments. The backlog is navigable and states simulation-vs-gate limitations; detailed completion evidence still lives in `fornax_program_management_todo_status.md`.
  - Program Management: approve. This removes ambiguity that one sprint file limited development while preserving the stage-gate honesty invariant.
- Verification: local Markdown link check across `docs/fornax/program_management/sprints/*.md` passed.


### T1 trace-ledger simulation milestone

- Added `fornax.trace_ledger` for WS-G1 request/plan-ID propagation, per-stage timings, router/expert trace correlation, and lifecycle cleanup evidence. The artifact records a request flowing through serving gateway, engine, scheduler, stage 0, transport, stage 1, KV manager, and metrics ledger over two logical hosts.
- Added `fornax observability trace-simulate --out ...` and `fornax test trace-ledger`; added the golden fixture `fornax/golden_vectors/trace_ledger/fixture.json`; expanded `make fornax-golden` and `fornax program simulate-t1` so trace correlation is part of the regular simulated-cluster evidence path. The one-command T1 bundle now validates 31 checks including `trace-ledger`.
- The validator enforces top-level trace/request/plan IDs, span parentage, component edges, event logical-host consistency, required lifecycle events, stage IDs 0 and 1, router/expert dispatch fields, activation handoff fields, KV fields, metric fields, cleanup fields, causal order, summary count consistency, and a simulation-only warning so this cannot be mistaken for live runtime telemetry or G2/G3 gate evidence.
- The golden fixture records `record_kind=trace-correlation-ledger`, `mode=t1-simulation`, `simulation_method=request-plan-span-correlation`, `component_count=8`, `span_count=8`, `event_count=16`, `stage_count=2`, `remote_expert_event_count=1`, `metric_event_count=1`, `cleanup_event_count=1`, `correlation_complete=true`, and `g2_g3_gate_evidence=false`.
- This narrows G1/H1/C2/E2 observability and lifecycle risk without blocking on Mac/AMD or a real heterogeneous cluster. It remains T1 simulation only; live runtime OpenTelemetry/exporter integration, real cross-process spans, real fabric traces, and G2/G3 gate telemetry remain open.
- Review-lens pass:
  - Networking: approve with comments. Request flow and state movement across logical host, transport, KV, and expert dispatch boundaries are explicit; real endpoint/network failure behavior remains future validation.
  - System Engineering: approve. The ledger ties serving, engine, scheduler, stage, transport, KV, metrics, stream, and cleanup into one lifecycle rather than isolated component logs.
  - Software Engineering: approve. The module is isolated, has a CLI, golden fixture, T1 bundle integration, and regression tests for mismatched IDs, parent spans, host drift, causal order, missing edges, and stale summaries.
  - Hardware: approve with comments. The artifact uses two logical hosts and records simulation-only scope; it does not claim Mac/AMD, real multi-node fabric, or G2/G3 evidence.
  - Documentation: approve with comments. The journal, todo status, and sprint backlog now record the trace-ledger milestone and remaining live-telemetry gaps.
- Verification: `python3 -m py_compile fornax/trace_ledger.py fornax/cli.py fornax/t1_simulated_validation.py tests/test_fornax_planner.py`, `python3 -m fornax observability trace-simulate --out fornax/golden_vectors/trace_ledger/fixture.json --plan-id golden-trace-ledger --request-id golden-trace-request --trace-id golden-trace-id`, `python3 -m fornax test trace-ledger`, focused trace-ledger and T1 bundle tests showing 9/9 passed, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_trace_ledger_validation_cli_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 31/31 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 209 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-golden`, and `make fornax-test` all passed.


### G1 evidence packet milestone

- Added `fornax.g1_evidence_packet` to turn a Phase-0 preflight bundle into a machine-checkable G1 review packet. The packet ties together target-contract validation, generated runtime/network/substrate drafts, Apple probe/role evidence, program rebaseline, T0 golden plans, benchmark ledger status, preflight doctor status, required sign-off files, machine-missing criteria, and closure blockers.
- Added `fornax program g1-evidence-packet --bundle ... --out ... --markdown-out ...`; integrated packet generation into `run_phase0_preflight(..., include_program_reports=True)` so `fornax program simulate-phase0` now emits `g1-evidence-packet.json` and `g1-evidence-packet.md` alongside `g1-gate-review.md` and `phase0-status.*`.
- The validator enforces required evidence IDs, required sign-off requirements, summary count consistency, human sign-off consistency, and overclaim prevention: `g1_gate_ready` cannot be true while closure blockers remain or required human sign-offs are missing. It also records `g2_g3_gate_evidence=false` so a G1 packet cannot be used as later hardware-gate evidence.
- The public two-logical-host Phase-0 simulation at `/tmp/fornax_phase0_g1_packet_20260622` produced a valid packet with `machine_complete=false`, `g1_ready=false`, `closure_blockers=4`, and recommended G1 outcome `ITERATE`. This is correct: the packet makes review preparation reproducible but does not replace TL/SP/Sponsor sign-off, rank-1 Apple probe evidence, staffing closure, or DEC-005.
- Review-lens pass:
  - Program Management: approve. The G1 evidence packet converts scattered Phase-0 artifacts into one reviewable gate packet while preserving G1 as open.
  - Organizational Skill: approve with comments. Sign-off requirements are explicit file-level handoffs; actual Sponsor/TL/SP decisions still need people to attach evidence.
  - Documentation: approve. JSON and Markdown outputs document evidence, caveats, blockers, and the decision boundary without relying on oral context.
  - Testing/Quality: approve. Regression tests cover packet emission from preflight, required evidence/signoff validation, and rejection of gate-ready overclaims.
  - Hardware: approve with comments. The packet reports simulation/local evidence honestly and keeps real Apple/multi-node/heterogeneous gate evidence open.
- Verification: `python3 -m py_compile fornax/g1_evidence_packet.py fornax/cli.py fornax/preflight.py tests/test_fornax_planner.py`, focused G1 packet/preflight tests showing 4/4 passed, `python3 -m fornax program simulate-phase0 --target fornax/golden_plans/v0_target_contract_fixture.md --out-dir /tmp/fornax_phase0_g1_packet_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65 --benchmark-iterations 1 --program-report-date 2026-06-22 --substrate-pinned-build max-26.4.0 --kickoff-date 2026-06-22 --ker-status unavailable --scope pending` showing 9/9 machine/simulation complete with ITERATE recommendation, `python3 -m fornax program g1-evidence-packet --bundle /tmp/fornax_phase0_g1_packet_20260622 --out /tmp/fornax_phase0_g1_packet_20260622/g1-evidence-packet-cli.json --markdown-out /tmp/fornax_phase0_g1_packet_20260622/g1-evidence-packet-cli.md --date 2026-06-22 --plan-version v3`, `python3 -m unittest tests.test_fornax_planner` showing 211 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-golden`, `make fornax-test`, and `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_g1_packet_regression_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 31/31 checks passed.


### Extended local H100 accelerator smoke milestone

- Extended `fornax.local_accelerator_smoke` from expert-MLP plus optional activation-transfer evidence into a broader local accelerator bundle. It can now run and validate expert MLP, split-pipeline correctness, and MoE layer/logit parity probes, treating `cuda:0` and `cuda:1` as two logical hosts on this machine.
- Added CLI controls to `fornax program local-accelerator-smoke` for `--skip-pipeline-correctness`, `--pipeline-*`, `--skip-moe-parity`, and `--moe-*` options. The default CLI path now attempts the extended accelerator smoke; CI/reference tests can explicitly skip or run CPU reference variants without claiming T2 accelerator evidence.
- The bundle policy now counts required accelerator probes and rejects CPU/reference output when accelerator evidence is required. Summary fields include pipeline/MoE accelerator flags, devices, throughput, error metrics, required accelerator probe count, and `g2_g3_gate_evidence=false`.
- Real local H100 run: `/tmp/fornax_local_accelerator_smoke_extended_no_transfer_h100_20260622` passed 4/4 checks with expert MLP on `cuda:0`, split-pipeline correctness from `cuda:0` to `cuda:1`, and MoE parity from `cuda:0` to `cuda:1`. Summary: expert throughput about `590.11` tokens/s, pipeline throughput about `5428.84` tokens/s, pipeline max abs error `6.7055e-08`, MoE throughput about `1942.76` tokens/s, MoE expert calls about `3885.51` calls/s, MoE max logit abs error `0.0`, `t2_smoke_passed=true`, and `g2_g3_gate_evidence=false`.
- Note: the first extended run with activation transfer included was interrupted after the activation-transfer subprocess ran longer than expected. Earlier local H100 activation-transfer evidence remains in `/tmp/fornax_local_accelerator_smoke_h100_20260622`; the new extended run intentionally used `--skip-activation-transfer` to isolate the new pipeline/MoE accelerator evidence.
- Review-lens pass:
  - Hardware: approve with comments. The bundle records concrete H100 devices and CUDA paths; it remains same-host logical-host evidence, not real multi-host or Mac/AMD proof.
  - Low-level Software: approve. Probe validators enforce device identity, parity errors, generated sequence equality, routing parity, and accelerator-measured claims.
  - System Engineering: approve with comments. Local smoke now exercises expert, activation/transport-adjacent pipeline, and MoE parity pieces in one bundle; real serving/runtime integration remains open.
  - Testing/Quality: approve. CPU reference mode stays CI-safe, required-accelerator policy rejects reference probes, and full unit/golden/T1 regressions remain green.
  - Program Management: approve. This advances T2-style local accelerator readiness without mislabeling it as T3/T4 or G2/G3 gate evidence.
- Verification: `python3 -m py_compile fornax/local_accelerator_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local accelerator smoke tests showing 2/2 passed, `python3 -m fornax program local-accelerator-smoke --out-dir /tmp/fornax_local_accelerator_smoke_extended_no_transfer_h100_20260622 --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --expert-device cuda:0 --expert-dtype float16 --expert-iterations 6 --expert-warmup 2 --expert-batch-tokens 8 --expert-hidden-dim 64 --expert-intermediate-dim 128 --expert-count 4 --expert-top-k 2 --skip-activation-transfer --pipeline-source-device cuda:0 --pipeline-destination-device cuda:1 --pipeline-dtype float32 --pipeline-iterations 4 --pipeline-warmup 1 --pipeline-hidden-dim 16 --pipeline-new-tokens 4 --pipeline-tolerance 0.0001 --moe-source-device cuda:0 --moe-expert-device cuda:1 --moe-dtype float32 --moe-iterations 4 --moe-warmup 1 --moe-token-count 4 --moe-hidden-dim 16 --moe-intermediate-dim 32 --moe-vocab-size 17 --moe-expert-count 4 --moe-top-k 2 --moe-tolerance 0.0001 --logical-source-host logical-host-0 --logical-destination-host logical-host-1 --timeout-s 180` showing 4/4 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 211 tests passed, `python3 -m compileall -q fornax tests`, `python3 -m fornax program simulate-t1 --out-dir /tmp/fornax_t1_extended_smoke_regression_20260622 --gpu-count 2 --profile two-gpu-heterogeneous --link-bandwidth-bytes-s 12500000000 --link-latency-s 0.0004 --slow-node-factor 0.65` showing 31/31 checks passed, `make fornax-golden`, and `make fornax-test` all passed.


### Local H100 serving/runtime smoke milestone

- Added `fornax.local_serving_smoke` and `fornax program local-serving-smoke` for the WS-H H3 / Phase 3 serving-runtime lane. The bundle composes the existing OpenAI/Ignis serving-adapter simulation with same-host two-H100 pipeline correctness and MoE layer/logit parity probes, treating `cuda:0` and `cuda:1` as logical hosts for local T2-style development evidence.
- Added `fornax test local-serving-smoke` as a CI-safe reference-mode validator. The test command generates a CPU/reference bundle by default so normal regressions validate artifact shape, serving seam checks, and overclaim policy without requiring GPUs. The explicit program command remains the GPU-required path.
- The bundle enforces the evidence boundary directly: summary fields set `live_http_endpoint=false`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`. CPU/reference probes are accepted only with `require_accelerator=False`; required-accelerator mode rejects reference outputs via bundle policy.
- Real local H100 run: `/tmp/fornax_local_serving_smoke_h100_20260622` passed 4/4 checks with serving adapter validation, split-pipeline correctness from `cuda:0` to `cuda:1`, and MoE parity from `cuda:0` to `cuda:1`. Summary: pipeline throughput about `1742.32` tokens/s, pipeline max abs error `6.7055e-08`, MoE throughput about `651.35` tokens/s, MoE expert calls about `1302.70` calls/s, MoE max logit abs error `0.0`, `t2_smoke_passed=true`, `live_http_endpoint=false`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Review-lens pass:
  - High-level Software: approve. The command is surfaced as a program-level serving/runtime smoke with familiar serving terms (`plan-id`, `request-id`, `model`, streaming) and advanced probe knobs kept consistent with the existing local accelerator command.
  - Hardware Acceleration: approve with comments. The bundle records measured CUDA pipeline and MoE parity paths with reference comparisons; it remains a tiny synthetic workload and does not prove target-model throughput.
  - Software Engineering: approve. The module is isolated, has focused tests for reference-mode and required-accelerator policy, and exposes a repeatable CLI/test path without mutating the existing accelerator bundle boundary.
  - System Engineering: approve with comments. This composes serving-surface normalization with local runtime probes in one artifact, narrowing the end-to-end integration gap; real HTTP/SSE endpoint behavior, live scheduler ownership, and target model loading remain open.
  - Hardware: approve with comments. The run uses the two local H100s as logical hosts and records same-host scope honestly; it is not Mac/AMD, real 2-3 node, or heterogeneous lab evidence.
  - Documentation/Program Management: approve. The journal and todo status now record the sprint lane, measured H100 evidence, and remaining gate boundaries.
- Verification: `python3 -m py_compile fornax/local_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local serving smoke tests showing 2/2 passed, `python3 -m fornax test local-serving-smoke --out /tmp/fornax_local_serving_smoke_reference_test_cli_20260622` showing 4/4 reference checks passed, `python3 -m fornax program local-serving-smoke --out-dir /tmp/fornax_local_serving_smoke_h100_20260622 --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --pipeline-source-device cuda:0 --pipeline-destination-device cuda:1 --pipeline-dtype float32 --pipeline-iterations 4 --pipeline-warmup 1 --pipeline-hidden-dim 16 --pipeline-new-tokens 4 --pipeline-tolerance 0.0001 --moe-source-device cuda:0 --moe-expert-device cuda:1 --moe-dtype float32 --moe-iterations 4 --moe-warmup 1 --moe-token-count 4 --moe-hidden-dim 16 --moe-intermediate-dim 32 --moe-vocab-size 17 --moe-expert-count 4 --moe-top-k 2 --moe-tolerance 0.0001 --logical-source-host logical-host-0 --logical-destination-host logical-host-1 --timeout-s 180` showing 4/4 H100 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 213 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.


### Local HTTP/SSE serving endpoint smoke milestone

- Added `fornax.local_http_serving_smoke` and `fornax program local-http-serving-smoke` for the Phase 3 / WS-H endpoint lane. The smoke starts a localhost stdlib HTTP server on an ephemeral port, serves `/v1/chat/completions`, validates non-stream JSON completion behavior, validates streaming SSE chunks plus `[DONE]`, and checks deterministic rejection for plan-integrity mismatch and bad paths.
- Added `fornax test local-http-serving-smoke` as a repeatable local endpoint validation command. The artifact embeds the existing serving-adapter simulation and runs `validate_serving_adapter_fixture` so endpoint behavior stays tied to the OpenAI/Ignis semantic seam.
- The bundle records the boundary explicitly: `live_http_endpoint=true`, `localhost_only=true`, `tls_enabled=false`, `production_auth_enabled=false`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`. This advances live endpoint behavior but does not claim product auth/TLS, target-model execution, real multi-host serving, or G3 closure.
- Explicit run: `/tmp/fornax_local_http_serving_smoke_20260622.json` passed 5/5 checks with non-stream status 200, stream status 200, 5 SSE chunks, `[DONE]` observed, plan-integrity rejection true, bad-path rejection true, and gate evidence false.
- Review-lens pass:
  - High-level Software: approve. The smoke exercises the OpenAI-compatible endpoint shape users expect rather than only internal adapter objects.
  - System Engineering: approve with comments. This connects serving adapter semantics to a real local HTTP/SSE lifecycle; target model loading, live scheduler/runtime ownership, TLS/auth, and multi-host runtime remain open.
  - Networking/Security: approve with comments. The plan-ID/hash rejection path is deterministic and machine-checked; production auth, encryption, and node admission are intentionally not claimed.
  - Software Engineering: approve. The module is stdlib-only, isolated, has focused unit tests, a CLI smoke path, and overclaim validation.
  - Program Management: approve. S3-7/H3 now has local live-endpoint smoke evidence while G3 remains open until real frontier MoE serving across the required heterogeneous fleet.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 2/2 passed, `python3 -m fornax test local-http-serving-smoke --out /tmp/fornax_local_http_serving_smoke_test_cli_20260622` showing 5/5 checks passed, and `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_smoke_20260622.json --plan-id phase3-local-http-serving-plan --plan-hash sha256:phase3-local-http-serving-plan --request-id phase3-local-http-serving-request --model qwen3-moe-class-target --max-tokens 64` showing 5/5 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 215 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.


### Local FornaxBackend HTTP integration smoke milestone

- Extended `fornax.local_http_serving_smoke` so accepted `/v1/chat/completions` requests now route through a named `LocalFornaxBackend` wrapper instead of calling the serving adapter directly from the HTTP handler. The backend exposes a local `FornaxBackend` summary with `engine_trait_compatible=true`, records accepted request count, and returns the existing EngineRequest/EngineResult/stream-event seam for OpenAI JSON/SSE translation.
- The smoke now validates six checks: serving-adapter validity, `FornaxBackend` integration, non-stream HTTP response, SSE stream response, plan-integrity rejection, and bad-path rejection. The plan-integrity and bad-path rejection paths do not increment backend request count, so the backend call count remains exactly two for the accepted non-stream and stream requests.
- Boundary remains explicit: this is localhost endpoint plus backend-seam integration evidence only. The artifact keeps `target_model_loaded=false`, `target_model_parity=false`, `tls_enabled=false`, `production_auth_enabled=false`, and `g2_g3_gate_evidence=false`.
- Explicit run: `/tmp/fornax_local_http_serving_backend_smoke_20260622.json` passed 6/6 checks with SSE chunks `5`, plan rejection true, `FornaxBackend` request count `2`, target-model parity false, and gate evidence false.
- Review-lens pass:
  - High-level Software: approve. The endpoint now goes through an explicitly named backend seam, aligning the smoke with the user-facing `FornaxBackend` concept instead of a bare adapter fixture.
  - System Engineering: approve with comments. This closes more of H3 at local-smoke scope by connecting HTTP/SSE, Engine seam, and backend naming; real target model loading and live scheduler/runtime remain open.
  - Networking/Security: approve. Accepted requests are separated from plan-integrity rejects, and failed plan checks do not reach backend execution.
  - Software Engineering: approve. The backend wrapper is isolated, stdlib-only, and covered by unit/CLI validation while preserving overclaim checks.
  - Program Management: approve with comments. This advances Phase 3 S3-7/H3 local integration evidence; it is still not G3 closure because target-model parity and real heterogeneous fleet evidence are absent.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 2/2 passed, `python3 -m fornax test local-http-serving-smoke --out /tmp/fornax_local_http_serving_smoke_backend_test_20260622.json` showing 6/6 checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_backend_smoke_20260622.json --plan-id phase3-local-http-backend-plan --plan-hash sha256:phase3-local-http-backend-plan --request-id phase3-local-http-backend-request --model qwen3-moe-class-target --max-tokens 64` showing 6/6 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 215 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.



### Local HTTP auth-boundary smoke milestone

- Extended `fornax.local_http_serving_smoke` so `/v1/chat/completions` now requires a local bearer token before plan-hash validation or backend execution. Accepted non-stream and stream requests still route through `LocalFornaxBackend`, while unauthorized requests return `401` with `endpoint_auth_required` and do not increment backend request count.
- The smoke now validates seven checks: serving-adapter validity, `FornaxBackend` integration, endpoint-auth rejection, non-stream HTTP response, SSE stream response, plan-integrity rejection, and bad-path rejection. The artifact redacts the configured token from `config`, records `local_auth_enabled=true`, and keeps `production_auth_enabled=false` and `tls_enabled=false` so the scope is explicit.
- Boundary remains explicit: this is localhost auth-boundary evidence only. It is not product auth, TLS/mTLS, node identity, target-model loading/parity, real multi-host serving, or G3 closure evidence.
- Explicit run: `/tmp/fornax_local_http_serving_auth_smoke_20260622.json` passed 7/7 checks with SSE chunks `5`, auth rejection true, plan rejection true, target-model parity false, and gate evidence false.
- Review-lens pass:
  - Networking/Security: approve with comments. The endpoint now rejects missing bearer credentials before plan integrity or backend execution; production auth, TLS/mTLS, and key distribution remain open.
  - Software Engineering: approve. The check is stdlib-only, artifact validation rejects token leakage, and unit/CLI coverage exercises the new failure boundary.
  - System Engineering: approve with comments. This advances S3-3 at local-smoke scope by combining auth rejection, plan-integrity tags, and backend execution ordering; real heterogeneous node admission remains future work.
  - Program Management: approve with comments. Phase 3 local endpoint evidence is stronger, but G3 remains open until frontier MoE runs across the target heterogeneous fleet with production security active.
  - Hardware: approve with comments. This is endpoint behavior on localhost and does not claim additional accelerator or Mac/AMD evidence.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 2/2 passed, `python3 -m fornax test local-http-serving-smoke --out /tmp/fornax_local_http_serving_auth_test_20260622.json` showing 7/7 checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_auth_smoke_20260622.json --plan-id phase3-local-http-auth-plan --plan-hash sha256:phase3-local-http-auth-plan --request-id phase3-local-http-auth-request --model qwen3-moe-class-target --max-tokens 64 --auth-token phase3-local-auth-token` showing 7/7 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 215 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.



### Local HTTP backpressure/failure semantics smoke milestone

- Extended `fornax.local_http_serving_smoke` with a real threaded local admission-control path. The smoke fills the configured local inflight slots with authenticated, plan-valid holder requests, then verifies that the next authenticated, plan-valid request is rejected with HTTP `429` and `backpressure_queue_full` before backend execution.
- The smoke now validates eight checks: serving-adapter validity, `FornaxBackend` integration, endpoint-auth rejection, backpressure rejection, non-stream HTTP response, SSE stream response, plan-integrity rejection, and bad-path rejection. Backend request count is now expected to be accepted requests only: non-stream, stream, plus one holder per local inflight slot; auth, plan, bad-path, and backpressure rejects do not increment it.
- Boundary remains explicit: this is localhost deterministic backpressure/failure-semantics evidence only. It is not product autoscaling, production queueing, distributed partition handling, target-model loading/parity, real multi-host serving, or G3 closure evidence.
- Explicit run: `/tmp/fornax_local_http_serving_backpressure_smoke_20260622.json` passed 8/8 checks with `max_inflight=2`, `max_observed_inflight=2`, holder statuses `[200, 200]`, one 429 backpressure rejection, backend request count `4`, cleanup count `4`, target-model parity false, and gate evidence false.
- Review-lens pass:
  - Networking/Security: approve with comments. Auth and plan checks still gate execution before backpressure admission; production TLS/mTLS and key distribution remain open.
  - System Engineering: approve with comments. S3-4 now has live local endpoint evidence for deterministic overload behavior; distributed partitions, retries, and cancellation remain future work.
  - Software Engineering: approve. Admission state is isolated behind server methods, guarded by a lock, validated in artifact shape, and covered by focused unit/CLI tests.
  - Program Management: approve with comments. Phase 3 local endpoint evidence now includes S3-3 and S3-4 smoke scope, while G3 remains open pending real heterogeneous target-model serving.
  - Hardware: approve with comments. The run is endpoint concurrency behavior on localhost, not additional H100/Mac/AMD performance evidence.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 2/2 passed, `python3 -m fornax test local-http-serving-smoke --out /tmp/fornax_local_http_serving_backpressure_test_20260622.json` showing 8/8 checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_backpressure_smoke_20260622.json --plan-id phase3-local-http-backpressure-plan --plan-hash sha256:phase3-local-http-backpressure-plan --request-id phase3-local-http-backpressure-request --model qwen3-moe-class-target --max-tokens 64 --auth-token phase3-local-backpressure-token --max-inflight 2 --backpressure-delay-ms 250` showing 8/8 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 215 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.


### Local HTTP lifecycle/state ownership smoke milestone

- Extended `fornax.local_http_serving_smoke` with a local lifecycle ledger for accepted and rejected localhost `/v1/chat/completions` requests. Accepted requests now allocate a request envelope, engine context, scheduler slot, response stream, and KV-cache placeholder, then release each resource during request cleanup.
- Rejected auth, backpressure, plan-integrity, and bad-path requests are recorded as rejected lifecycle events and do not allocate backend resources. The smoke now validates nine checks: serving-adapter validity, local `FornaxBackend` integration, endpoint-auth rejection, deterministic 429 backpressure rejection, lifecycle cleanup, non-stream JSON, SSE stream chunks, plan-integrity rejection, and bad-path rejection.
- Explicit run: `/tmp/fornax_local_http_serving_lifecycle_smoke_20260622.json` passed 9/9 checks with `max_inflight=2`, backend/lifecycle accepted count `4`, rejected count `4`, cleanup count `4`, allocated resources `20`, released resources `20`, active resources `0`, lifecycle event count `44`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is localhost lifecycle and cleanup evidence only. It is not live distributed state ownership, target-model loading/parity, production auth/TLS, real NVIDIA/AMD/Mac heterogeneous serving, or G3 closure evidence.
- Review-lens pass:
  - System Engineering: approve with comments. The smoke ties endpoint admission, backend execution, scheduler slot ownership, response state, and KV placeholder cleanup into a single local lifecycle artifact; real distributed ownership remains open.
  - Software Engineering: approve. Lifecycle state is isolated behind server methods, guarded by a lock, validated in artifact shape, and covered by focused unit/CLI regression tests.
  - Networking/Security: approve with comments. Auth, plan-integrity, and backpressure rejects are tracked without reaching backend execution; production TLS/mTLS and key distribution remain open.
  - Program Management: approve with comments. This advances Phase 3 S3-7/H1 at local-smoke scope while preserving the G3 boundary.
  - Hardware: approve with comments. The run is localhost endpoint lifecycle evidence and does not claim additional H100, Mac, AMD, multi-node, or frontier-model proof.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 2/2 passed, `python3 -m fornax test local-http-serving-smoke --out /tmp/fornax_local_http_serving_lifecycle_test_20260622.json` showing 9/9 checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_lifecycle_smoke_20260622.json --plan-id phase3-local-http-lifecycle-plan --plan-hash sha256:phase3-local-http-lifecycle-plan --request-id phase3-local-http-lifecycle-request --model qwen3-moe-class-target --max-tokens 64 --auth-token phase3-local-lifecycle-token --max-inflight 2 --backpressure-delay-ms 250` showing 9/9 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 215 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.


### Local HTTP target-fixture backend smoke milestone

- Extended `fornax.local_http_serving_smoke` with an opt-in local target-fixture backend mode selected by `fornax program local-http-serving-smoke --backend-mode target-fixture`. The default adapter mode remains unchanged and still validates the conservative 9/9 localhost endpoint smoke without claiming fixture loading.
- The target-fixture path loads a deterministic local fixture behind `LocalFornaxBackend`, records tokenizer and chat-template hashes, tokenizes the OpenAI chat messages, honors the `</final>` stop sequence, emits the generated text `fixture target parity`, and verifies non-stream and SSE stream parity across accepted requests.
- Explicit run: `/tmp/fornax_local_http_serving_target_fixture_smoke_20260622.json` passed 10/10 checks with `max_inflight=2`, backend request count `4`, target fixture run count `4`, `target_fixture_loaded=true`, `target_fixture_parity=true`, `target_fixture_non_stream_matches_stream=true`, generated tokens `["fixture", "target", "parity"]`, template hash `sha256:cccc...`, tokenizer hash `sha256:dddd...`, `target_model_parity=false`, `real_frontier_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is local fixture loading/parity evidence only. It is not the real frontier MoE target model, product auth/TLS, a real NVIDIA/AMD/Mac heterogeneous serve, or G3 closure evidence.
- Review-lens pass:
  - High-level Software: approve. The smoke now exercises a backend-selected local model fixture through the same OpenAI-compatible HTTP/SSE endpoint users will call.
  - System Engineering: approve with comments. Endpoint admission, lifecycle cleanup, backend execution, tokenizer/template metadata, stop behavior, and streaming parity are now captured in one artifact; real distributed serving remains open.
  - Software Engineering: approve. The mode is opt-in, keeps the default contract stable, has artifact validation for hashes/parity/run counts, and adds focused regression coverage.
  - Testing/Quality: approve. Focused unit tests cover default adapter mode, target-fixture mode, and gate-overclaim rejection; full unit, compile, Makefile test, and golden validation passed.
  - Program Management: approve with comments. This advances H2/H3 local target-model-readiness evidence while preserving the formal G3 gap for the real frontier heterogeneous fleet.
  - Hardware: approve with comments. The run is localhost CPU/stdlib fixture evidence and does not claim additional H100, Mac, AMD, multi-node, or frontier-model proof.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 3/3 passed, `python3 -m fornax test local-http-serving-smoke --out /tmp/fornax_local_http_serving_target_fixture_default_test_20260622.json` showing 9/9 default checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_target_fixture_smoke_20260622.json --plan-id phase3-local-http-target-fixture-plan --plan-hash sha256:phase3-local-http-target-fixture-plan --request-id phase3-local-http-target-fixture-request --model qwen3-moe-class-target --max-tokens 64 --auth-token phase3-local-target-fixture-token --max-inflight 2 --backpressure-delay-ms 250 --backend-mode target-fixture` showing 10/10 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 216 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.


### Local H100 target-fixture execution probe milestone

- Added `fornax.target_fixture_probe` and `fornax accelerator target-fixture-probe` for measured local target-fixture execution evidence. The probe has a CPU reference path for CI and a Torch/CUDA subprocess path for H100 evidence using the `/mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python` training venv.
- The probe validates deterministic fixture decode, tokenizer/template hash metadata, stop-token handling, generated-text parity, generated-token parity, final-logit parity against the reference path, hardware/environment metadata, and overclaim rejection if CPU evidence is mislabeled as accelerator evidence.
- Explicit H100 run: `/tmp/fornax_target_fixture_probe_h100_20260622.json` passed validation on `cuda:0` with backend `torch`, hardware `NVIDIA H100 80GB HBM3`, Torch `2.12.0+cu130`, CUDA `13.0`, generated text `fixture h100`, `tokens_generated=40`, throughput about `4110.02` tokens/s, max abs error about `2.37e-06`, `accelerator_measured=true`, and `real_frontier_model=false`.
- Boundary remains explicit: this is a measured single-H100 local fixture execution probe. It is not the real frontier MoE target model, real heterogeneous NVIDIA/AMD/Mac serving, target throughput proof, product auth/TLS, or G3 closure evidence.
- Review-lens pass:
  - Hardware Acceleration: approve with comments. The artifact records actual H100/CUDA/Torch execution and throughput, while clearly limiting the claim to a small local fixture.
  - Low-level Software: approve. The probe compares CUDA execution against a deterministic reference path and validates final logits, generated token IDs, stop-token behavior, and metadata invariants.
  - Software Engineering: approve. The probe is isolated, follows existing subprocess probe patterns, has CI-safe CPU coverage, and rejects accelerator overclaims.
  - Testing/Quality: approve. Focused tests cover CPU reference validity and false T2 accelerator claims; full unit, compile, Makefile test, golden, and diff-hygiene validation passed.
  - Program Management: approve with comments. This advances the Phase 3 path from local fixture semantics toward measured accelerator execution, but leaves real target-model parity and G3 lab evidence open.
  - Hardware: approve with comments. The run uses the available local H100 as a development stand-in and does not claim Mac/AMD or multi-node evidence.
- Verification: `python3 -m py_compile fornax/target_fixture_probe.py fornax/cli.py tests/test_fornax_planner.py`, focused target-fixture probe tests showing 2/2 passed, `python3 -m fornax accelerator target-fixture-probe --out /tmp/fornax_target_fixture_probe_cpu_test_20260622.json --backend cpu-stdlib --iterations 2 --warmup 0 --new-tokens 4`, `python3 -m fornax accelerator target-fixture-probe --out /tmp/fornax_target_fixture_probe_h100_20260622.json --backend torch --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --device cuda:0 --dtype float32 --iterations 20 --warmup 3 --new-tokens 4 --tolerance 0.0001 --timeout-s 180`, `python3 -m unittest tests.test_fornax_planner` showing 218 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local serving bundle target-fixture execution milestone

- Integrated `fornax.target_fixture_probe` into `fornax.local_serving_smoke`, so `fornax program local-serving-smoke` now emits `target-fixture-execution-probe.json` alongside the serving adapter, split-pipeline correctness probe, MoE parity probe, and bundle validation artifact.
- The bundle policy now requires target-fixture accelerator evidence when `require_accelerator=true`, counts pipeline/MoE/target-fixture probes as 3 required accelerator probes, and keeps the boundary fields explicit: `live_http_endpoint=false`, `target_model_parity=false`, `target_fixture_real_frontier_model=false`, and `g2_g3_gate_evidence=false`.
- Added matching CLI options for the program command and kept `fornax test local-serving-smoke` CI-safe by running the CPU/reference target fixture path with `require_accelerator=false`.
- Explicit H100 run: `/tmp/fornax_local_serving_smoke_target_fixture_h100_20260622` passed 5/5 checks with serving adapter validation, split-pipeline correctness from `cuda:0` to `cuda:1`, MoE parity from `cuda:0` to `cuda:1`, and target-fixture execution on `cuda:0`. Summary: pipeline throughput about `3600.68` tokens/s, pipeline max abs error `6.7055e-08`, MoE throughput about `1925.69` tokens/s, MoE expert calls about `3851.38` calls/s, MoE max logit abs error `0.0`, target-fixture throughput about `5366.57` tokens/s, target-fixture generated text `fixture h100`, target-fixture max abs error about `2.37e-06`, `accelerator_probe_count=3`, `required_accelerator_probe_count=3`, `t2_smoke_passed=true`, `live_http_endpoint=false`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Review-lens pass:
  - High-level Software: approve. The program-level serving smoke now reports endpoint-adjacent serving semantics and the target-fixture execution result in one artifact instead of forcing readers to correlate a separate accelerator probe manually.
  - Hardware Acceleration: approve with comments. The artifact records measured H100/CUDA execution for pipeline, MoE parity, and fixture decode; it remains a tiny deterministic fixture and does not prove real target-model throughput.
  - Low-level Software: approve. The target-fixture path compares CUDA execution against the deterministic reference path and carries generated-token/logit parity into the serving-runtime bundle.
  - Software Engineering: approve. The integration is isolated, has CPU/reference coverage, required-accelerator rejection coverage, CLI coverage, and does not widen the local accelerator smoke contract.
  - System Engineering: approve with comments. This narrows the gap between serving surface validation and runtime probe evidence; live distributed lifecycle, target model loading, product auth/TLS, and multi-host ownership remain open.
  - Program Management: approve with comments. Phase 3 T2/H2/H3 local evidence is stronger, while G3 remains open until real frontier MoE serving across the required heterogeneous fleet.
- Verification: `python3 -m py_compile fornax/local_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local serving/target-fixture tests showing 5/5 passed, `python3 -m fornax test local-serving-smoke --out /tmp/fornax_local_serving_target_fixture_reference_test_20260622` showing 5/5 reference checks passed, `python3 -m fornax program local-serving-smoke --out-dir /tmp/fornax_local_serving_smoke_target_fixture_h100_20260622 --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --pipeline-source-device cuda:0 --pipeline-destination-device cuda:1 --pipeline-dtype float32 --pipeline-iterations 4 --pipeline-warmup 1 --pipeline-hidden-dim 16 --pipeline-new-tokens 4 --pipeline-tolerance 0.0001 --moe-source-device cuda:0 --moe-expert-device cuda:1 --moe-dtype float32 --moe-iterations 4 --moe-warmup 1 --moe-token-count 4 --moe-hidden-dim 16 --moe-intermediate-dim 32 --moe-vocab-size 17 --moe-expert-count 4 --moe-top-k 2 --moe-tolerance 0.0001 --target-fixture-device cuda:0 --target-fixture-dtype float32 --target-fixture-iterations 20 --target-fixture-warmup 3 --target-fixture-new-tokens 4 --target-fixture-tolerance 0.0001 --logical-source-host logical-host-0 --logical-destination-host logical-host-1 --timeout-s 180` showing 5/5 H100 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 219 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local HTTPS target-fixture endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with opt-in local HTTPS mode via `fornax program local-http-serving-smoke --enable-tls`. The smoke uses a local self-signed fixture certificate with SANs for `localhost` and `127.0.0.1`, writes the private key only to a temporary directory, redacts it from artifacts, and uses a verifying client SSL context for all accepted and rejected endpoint requests.
- The target-fixture HTTPS smoke now validates eleven checks: serving-adapter validity, local `FornaxBackend` integration, endpoint-auth rejection, local TLS handshake/client verification, deterministic 429 backpressure rejection, lifecycle cleanup, target-fixture parity, non-stream JSON, SSE stream chunks, plan-integrity rejection, and bad-path rejection.
- Explicit run: `/tmp/fornax_local_http_serving_tls_target_fixture_smoke_20260622.json` passed 11/11 checks with endpoint `https://127.0.0.1:45243/v1/chat/completions`, `tls_enabled=true`, `local_tls_enabled=true`, `tls_client_verified=true`, TLS mode `local-self-signed`, certificate hash `sha256:17818238ab58cdf7a0d9ec3df8feabe55c5af492904ccc078cc6841fd58111c0`, SANs `DNS:localhost` and `IP:127.0.0.1`, `production_tls_enabled=false`, local bearer-token auth rejection true, backpressure rejection true, lifecycle cleanup true, target-fixture run count `4`, target-fixture non-stream/SSE parity true, `target_model_parity=false`, `real_frontier_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is local TLS/client-verification evidence for the localhost endpoint path. It is not product TLS/mTLS, production key management, real node identity, real frontier target-model parity, real multi-host serving, or G3 closure evidence.
- Review-lens pass:
  - Networking/Security: approve with comments. Local HTTPS is now exercised with certificate verification and bearer-token auth before plan/backend execution; product TLS/mTLS, key rotation, node identity, and production auth remain open.
  - System Engineering: approve with comments. The TLS path is integrated through the same endpoint, backend, lifecycle, backpressure, and target-fixture checks rather than a separate synthetic probe.
  - Software Engineering: approve. TLS is opt-in, stdlib-only, has focused regression coverage, redacts private key material, and preserves existing HTTP default behavior.
  - Testing/Quality: approve. Focused local HTTP smoke tests cover default HTTP, target-fixture HTTP, target-fixture HTTPS, and gate-overclaim rejection; full unit, compile, Makefile test, golden, and diff-hygiene validation passed.
  - Program Management: approve with comments. This advances Phase 3 S3-3/H3 local evidence while keeping production auth/TLS and G3 explicitly open.
  - Hardware: approve with comments. This is endpoint security behavior on localhost and does not claim new H100, Mac, AMD, multi-node, or frontier-model proof.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 4/4 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_tls_target_fixture_smoke_20260622.json --plan-id phase3-local-http-tls-target-fixture-plan --plan-hash sha256:phase3-local-http-tls-target-fixture-plan --request-id phase3-local-http-tls-target-fixture-request --model qwen3-moe-class-target --max-tokens 64 --auth-token phase3-local-tls-target-fixture-token --max-inflight 2 --backpressure-delay-ms 250 --backend-mode target-fixture --enable-tls --timeout-s 5` showing 11/11 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 220 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local mTLS target-fixture endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with opt-in local mTLS via `fornax program local-http-serving-smoke --enable-mtls`. The smoke uses local CA/server/client fixture certificates, requires a verified client certificate for every endpoint request, records the peer subject `fornax-local-client`, verifies that a client without a certificate fails at transport level, and redacts private keys from artifacts.
- The target-fixture mTLS smoke now validates twelve checks: serving-adapter validity, local `FornaxBackend` integration, endpoint-auth rejection, local TLS handshake/client verification, local mTLS node identity, deterministic 429 backpressure rejection, lifecycle cleanup, target-fixture parity, non-stream JSON, SSE stream chunks, plan-integrity rejection, and bad-path rejection.
- Explicit run: `/tmp/fornax_local_http_serving_mtls_target_fixture_smoke_20260622.json` passed 12/12 checks with endpoint `https://127.0.0.1:42835/v1/chat/completions`, `tls_enabled=true`, `mtls_enabled=true`, TLS mode `local-mutual-tls`, CA hash `sha256:e90deba81223b984e19b6920a58768cd90d79219c19b7079e0308926aae09a29`, client cert hash `sha256:3372a09773e63bca17ea20a31fd4e078ea83313b33bc88686022cdcbcdab3d75`, client subject `fornax-local-client`, `mtls_missing_client_cert_rejected=true`, `mtls_verified_peer_count=8`, `mtls_expected_peer_count=8`, `mtls_all_peers_expected=true`, local bearer-token auth rejection true, backpressure rejection true, lifecycle cleanup true, target-fixture run count `4`, target-fixture non-stream/SSE parity true, `target_model_parity=false`, `real_frontier_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is local mTLS/node-identity evidence for the localhost endpoint path. It is not product mTLS, production key management, real node admission, real frontier target-model parity, real multi-host serving, or G3 closure evidence.
- Review-lens pass:
  - Networking/Security: approve with comments. Local mTLS now verifies a client certificate, rejects missing client credentials before backend execution, and keeps production keying and product mTLS open.
  - System Engineering: approve with comments. The mTLS path is integrated through the same endpoint, backend, lifecycle, backpressure, and target-fixture checks rather than a detached probe.
  - Software Engineering: approve. The mode is opt-in, stdlib-only, validates artifact shape, and preserves the existing HTTP and local TLS defaults.
  - Testing/Quality: approve. Focused regression covers local TLS target-fixture and local mTLS target-fixture behavior; full unit, compile, Makefile test, golden, and diff-hygiene validation passed.
  - Program Management: approve with comments. This advances Phase 3 S3-3/H3 local evidence while keeping product mTLS, real heterogeneous target-model serving, and G3 explicitly open.
  - Hardware: approve with comments. This is endpoint security behavior on localhost and does not claim new H100, Mac, AMD, multi-node, or frontier-model proof.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local TLS/mTLS HTTP serving smoke tests showing 2/2 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_mtls_target_fixture_smoke_20260622.json --plan-id phase3-local-http-mtls-target-fixture-plan --plan-hash sha256:phase3-local-http-mtls-target-fixture-plan --request-id phase3-local-http-mtls-target-fixture-request --model qwen3-moe-class-target --max-tokens 64 --auth-token phase3-local-mtls-target-fixture-token --max-inflight 2 --backpressure-delay-ms 250 --backend-mode target-fixture --enable-mtls --timeout-s 5` showing 12/12 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 221 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local mTLS admitted-cancel endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with a deterministic admitted-cancel request. The request passes mTLS, bearer-token auth, plan-hash validation, and admission, allocates the normal lifecycle resources, records cancellation, returns HTTP `409` with `request_cancelled`, releases all lifecycle resources and inflight state, and does not call `LocalFornaxBackend`.
- The target-fixture mTLS smoke now validates thirteen checks: serving-adapter validity, local `FornaxBackend` integration, endpoint-auth rejection, local TLS handshake/client verification, local mTLS node identity, deterministic 429 backpressure rejection, admitted cancellation cleanup, lifecycle cleanup, target-fixture parity, non-stream JSON, SSE stream chunks, plan-integrity rejection, and bad-path rejection.
- Explicit run: `/tmp/fornax_local_http_serving_cancel_mtls_target_fixture_smoke_20260622.json` passed 13/13 checks with endpoint `https://127.0.0.1:38119/v1/chat/completions`, backend request count `4`, target-fixture run count `4`, admitted cancel status `409`, `request_cancelled_after_admit=true`, `request_cancelled_before_backend=true`, lifecycle accepted/cleanup count `5`, lifecycle cancelled count `1`, lifecycle resources allocated/released `25`, inflight cleanup count `5`, `mtls_verified_peer_count=9`, `mtls_expected_peer_count=9`, client subject `fornax-local-client`, missing-client-cert rejection true, `target_model_parity=false`, `real_frontier_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is local endpoint failure-semantics evidence for admitted cancellation cleanup before backend execution. It is not distributed cancellation, client disconnect recovery, production partition handling, product mTLS/keying, real frontier target-model parity, real multi-host serving, or G3 closure evidence.
- Review-lens pass:
  - System Engineering: approve with comments. The local endpoint now exercises an admitted cancellation path through the same admission and lifecycle accounting as successful requests; distributed cancellation and partition behavior remain open.
  - Networking/Security: approve with comments. Cancellation still occurs after mTLS, bearer auth, and plan-hash verification, so it does not bypass the trust boundary.
  - Software Engineering: approve. Backend execution and lifecycle counts are separated, artifact validation rejects stale accounting, and the default/TLS/mTLS smoke tests cover the new invariant.
  - Testing/Quality: approve. Focused local HTTP tests cover default, target-fixture, TLS, mTLS, and gate-overclaim behavior after the count split.
  - Program Management: approve with comments. This advances S3-4/S3-7 local evidence but does not close G3 or the real distributed failure-semantics requirement.
  - Hardware: approve with comments. This is localhost endpoint behavior and does not claim additional H100, Mac, AMD, multi-node, or frontier-model evidence.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py tests/test_fornax_planner.py`, `git diff --check`, focused local HTTP serving smoke tests showing 5/5 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_cancel_mtls_target_fixture_smoke_20260622.json --plan-id phase3-local-http-cancel-mtls-target-fixture-plan --plan-hash sha256:phase3-local-http-cancel-mtls-target-fixture-plan --request-id phase3-local-http-cancel-mtls-target-fixture-request --model qwen3-moe-class-target --max-tokens 64 --auth-token phase3-local-cancel-mtls-target-fixture-token --max-inflight 2 --backpressure-delay-ms 250 --backend-mode target-fixture --enable-mtls --timeout-s 5` showing 13/13 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 221 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, and `make fornax-golden` all passed.

### Local mTLS endpoint plus H100 target-fixture execution milestone

- Extended `fornax.local_http_serving_smoke` with an opt-in target-fixture execution probe. When enabled with `--include-target-fixture-execution-probe`, the local HTTP artifact now combines the live localhost mTLS/SSE endpoint checks with the existing `fornax.target_fixture_probe` CPU or Torch/CUDA execution validator.
- The default local HTTP smoke remains lightweight and unchanged. The probe requires `backend-mode=target-fixture`, records raw probe and validation objects in the endpoint artifact, and keeps `target_model_parity=false`, `real_frontier_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Explicit H100 run: `/tmp/fornax_local_http_serving_runtime_mtls_target_fixture_smoke_20260622.json` passed 14/14 checks with endpoint `https://127.0.0.1:36369/v1/chat/completions`, backend request count `4`, lifecycle accepted/cleanup count `5`, lifecycle cancelled count `1`, `mtls_verified_peer_count=9`, target-fixture endpoint run count `4`, integrated target-fixture execution probe backend `torch`, device `cuda:0`, logical host `logical-host-0`, generated text `fixture h100`, throughput about `5421.38` tokens/s, max abs error about `2.37e-06`, and accelerator evidence true for the fixture probe only.
- Boundary remains explicit: this is still a tiny deterministic local fixture. It links live endpoint behavior to measured local H100 fixture execution in one artifact, but it is not real frontier target-model loading/parity, production auth/mTLS/keying, distributed cancellation/partition proof, real NVIDIA/AMD/Mac heterogeneous serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - System Engineering: approve with comments. The endpoint artifact now correlates serving-surface behavior with runtime probe evidence instead of leaving those as separate artifacts; true distributed serving remains open.
  - Hardware Acceleration: approve with comments. The integrated probe records actual H100/CUDA execution and throughput, but only for the deterministic local fixture.
  - Low-level Software: approve. The probe reuses the existing reference-vs-CUDA validator and keeps backend execution counts distinct from endpoint request counts.
  - Software Engineering: approve. The feature is opt-in, has CLI and unit coverage, validates raw probe shape, and preserves default smoke behavior.
  - Testing/Quality: approve. Focused tests cover the CPU-reference integrated path, and the explicit H100 CLI run covers the accelerator path.
  - Program Management: approve with comments. This advances Phase 3 T2/H2/H3 local evidence but keeps G3 and real target-model parity open.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, `git diff --check`, focused local HTTP tests showing 2/2 passed, the explicit H100 CLI smoke above showing 14/14 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 222 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, and `make fornax-golden` all passed.


### Local mTLS endpoint runtime-probe bundle milestone

- Extended `fornax.local_http_serving_smoke` with opt-in endpoint runtime probes via `--include-runtime-probes`. The local HTTP artifact now carries raw split-pipeline correctness and MoE layer parity probe payloads, their validation objects, top-level summary metrics, and explicit check names alongside the existing local mTLS/SSE endpoint, target-fixture backend parity, admitted-cancel, lifecycle, and target-fixture execution probe evidence.
- Added compact CLI wiring for the endpoint bundle: shared runtime-probe backend/device/dtype/iteration/tolerance/logical-host flags plus small-model shape flags for pipeline and MoE. Defaults remain CI-safe with `cpu-stdlib`; the H100 run switches the runtime probes and target-fixture execution probe to `torch` under `/mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python`.
- Added focused unit coverage for the CPU/reference endpoint runtime-probe bundle. The default endpoint path stays unchanged; runtime probes are only emitted and required when explicitly requested, and validation rejects missing probe payloads, failed probe validations, mismatched top-level summary fields, or missing check names.
- Explicit H100 run: `/tmp/fornax_local_http_serving_runtime_bundle_mtls_target_fixture_smoke_20260622.json` passed 16/16 checks with endpoint `https://127.0.0.1:33769/v1/chat/completions`, backend request count `4`, lifecycle accepted/cleanup count `5`, lifecycle cancelled count `1`, `mtls_verified_peer_count=9`, target-fixture endpoint run count `4`, runtime probe backend `torch`, runtime accelerator probes `2/2`, split-pipeline `cuda:0` to `cuda:1` throughput about `3802.85` tokens/s with max abs error about `6.71e-08`, MoE parity `cuda:0` to `cuda:1` throughput about `1859.88` tokens/s and `3719.75` expert calls/s with max logit abs error `0.0`, target-fixture execution on `cuda:0` generated `fixture h100` at about `5250.46` tokens/s with max abs error about `2.37e-06`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is a localhost endpoint plus same-host H100 logical-host development artifact. It strengthens Phase 3 T2/H2/H3 local evidence, but it is not real frontier target-model loading/parity, production auth/mTLS/keying, distributed cancellation/partition proof, real NVIDIA/AMD/Mac heterogeneous serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - System Engineering: approve with comments. The endpoint artifact now correlates serving surface, local trust boundary, failure semantics, lifecycle cleanup, target-fixture backend parity, and measured H100 runtime probes in one artifact; true distributed serving and ownership remain open.
  - Hardware Acceleration: approve with comments. The artifact records measured H100/CUDA execution for split pipeline, MoE parity, and fixture decode, while preserving the logical-host/same-host and tiny-fixture limitations.
  - Low-level Software: approve. The runtime probes reuse existing deterministic reference-vs-CUDA validators and keep endpoint backend execution counts separate from post-endpoint probe execution.
  - Software Engineering: approve. The feature is opt-in, validates raw probe payloads and summaries, has CI-safe CPU coverage, and preserves existing local HTTP defaults.
  - Networking/Security: approve with comments. The H100 bundle still exercises local mTLS, missing-client-certificate rejection, bearer auth, and plan-hash rejection; product keying, product auth, and real node identity remain open.
  - Testing/Quality: approve. Focused local HTTP tests cover default, target-fixture, target-fixture execution, runtime-probe bundle, TLS, mTLS, and overclaim behavior; full unit, compile, Makefile test, golden, and diff-hygiene validation passed.
  - Program Management: approve with comments. This updates the active Phase 3 lane from separate endpoint and serving-runtime evidence toward a single endpoint runtime bundle, while keeping G3 open until real frontier MoE serves across the required heterogeneous fleet.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 7/7 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_runtime_probe_cpu_smoke_20260622.json --backend-mode target-fixture --include-runtime-probes --runtime-probe-backend cpu-stdlib --runtime-probe-iterations 2 --runtime-probe-warmup 0 --pipeline-probe-hidden-dim 4 --pipeline-probe-new-tokens 2 --moe-probe-token-count 2 --moe-probe-hidden-dim 4 --moe-probe-intermediate-dim 6 --moe-probe-vocab-size 11 --moe-probe-expert-count 2 --moe-probe-top-k 1` showing 13/13 checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_runtime_bundle_mtls_target_fixture_smoke_20260622.json --backend-mode target-fixture --enable-mtls --include-runtime-probes --runtime-probe-backend torch --runtime-probe-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --runtime-probe-source-device cuda:0 --runtime-probe-destination-device cuda:1 --include-target-fixture-execution-probe --target-fixture-execution-backend torch --target-fixture-execution-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --target-fixture-execution-device cuda:0 --timeout-s 5` showing 16/16 H100 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 223 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.


### Local mTLS endpoint transport-runtime bundle milestone

- Extended `fornax.local_http_serving_smoke` with an opt-in activation-transfer probe via `--include-activation-transfer-probe`. The endpoint artifact now can carry same-host two-H100 transport/topology timing evidence alongside the local mTLS/SSE endpoint, target-fixture backend parity, admitted-cancel cleanup, lifecycle cleanup, split-pipeline correctness, MoE parity, and target-fixture execution probes.
- Added CLI wiring for activation-transfer backend, torch Python, source/destination devices, dtype, iterations, warmup, payload size, tolerance, logical hosts, and timeout. The CLI stores payload bytes in the artifact while exposing `--activation-transfer-payload-mib` for operator ergonomics.
- Added CI-safe unit coverage for the endpoint activation-transfer artifact path using the CPU/reference backend. Validation now rejects missing raw transfer payloads, failed transfer validations, summary/probe mismatches, missing `activation-transfer-probe` check names, and stale probe payloads when the feature is omitted.
- Explicit H100 run: `/tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json` passed 17/17 checks with endpoint `https://127.0.0.1:43955/v1/chat/completions`, backend request count `4`, lifecycle accepted/cleanup count `5`, lifecycle cancelled count `1`, `mtls_verified_peer_count=9`, target-fixture endpoint run count `4`, activation transfer `cuda:0` to `cuda:1` at about `242.37` GiB/s with about `0.0000645` s per transfer, `335544320` bytes transferred, and max abs error `0.0`, split-pipeline `cuda:0` to `cuda:1` throughput about `5466.50` tokens/s with max abs error about `6.71e-08`, MoE parity `cuda:0` to `cuda:1` throughput about `1830.03` tokens/s and `3660.07` expert calls/s with max logit abs error `0.0`, target-fixture execution on `cuda:0` generated `fixture h100` at about `5563.52` tokens/s with max abs error about `2.37e-06`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: the transfer probe is same-host H100 logical-host evidence. It strengthens Phase 3 S3-2/T2 topology and endpoint-runtime local evidence, but it is not real heterogeneous fabric transport, product TLS/mTLS/keying, real frontier target-model parity, distributed partition proof, NVIDIA/AMD/Mac frontier serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - Networking/Security: approve with comments. The mTLS endpoint artifact now also records local transfer timing while preserving bearer auth, plan hash, missing-client-certificate rejection, and non-product keying boundaries.
  - System Engineering: approve with comments. Endpoint evidence now correlates trust boundary, lifecycle, failure semantics, local target fixture, topology transfer, pipeline, MoE, and fixture execution in one artifact; real multi-host transport remains open.
  - Hardware Acceleration: approve with comments. The transfer probe records measured H100/CUDA pair bandwidth and latency, clearly scoped to same-host logical hosts.
  - Low-level Software: approve. The transfer probe reuses the existing reference/CUDA validator and keeps transport timing evidence separate from backend request execution counts.
  - Software Engineering: approve. The feature is opt-in, has CLI and validator coverage, and preserves existing default endpoint behavior.
  - Testing/Quality: approve. Focused local HTTP tests cover default, target-fixture, activation transfer, runtime probes, TLS, mTLS, target-fixture execution, and overclaim behavior; full validation is recorded below.
  - Program Management: approve with comments. This advances Phase 3 S3-2/H3 local evidence while keeping G3 open until real frontier MoE serves across the required heterogeneous fleet.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py fornax/cli.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 8/8 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_activation_transfer_cpu_smoke_20260622.json --backend-mode target-fixture --include-activation-transfer-probe --activation-transfer-backend cpu-stdlib --activation-transfer-iterations 2 --activation-transfer-warmup 0 --activation-transfer-payload-mib 1` showing 12/12 checks passed, and `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json --backend-mode target-fixture --enable-mtls --include-activation-transfer-probe --activation-transfer-backend torch --activation-transfer-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --activation-transfer-source-device cuda:0 --activation-transfer-destination-device cuda:1 --include-runtime-probes --runtime-probe-backend torch --runtime-probe-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --runtime-probe-source-device cuda:0 --runtime-probe-destination-device cuda:1 --include-target-fixture-execution-probe --target-fixture-execution-backend torch --target-fixture-execution-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --target-fixture-execution-device cuda:0 --timeout-s 5` showing 17/17 H100 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 224 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local mTLS admitted-timeout endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with a deterministic admitted-timeout request. The request passes local mTLS, bearer-token auth, plan-hash validation, and admission, allocates normal lifecycle resources, records timeout ownership through the serving gateway and scheduler, returns HTTP `504` with `backend_timeout`, releases all lifecycle resources and inflight state, and does not call `LocalFornaxBackend`.
- The endpoint artifact now separates backend execution count from admitted-but-terminal request count: the H100 bundle still has backend execution count `4`, while lifecycle accepted/cleanup count is `6` because it includes one admitted cancellation and one admitted timeout before backend execution. Validation now rejects stale timeout accounting, missing `admitted-timeout-cleanup` checks, missing timeout responses, and lifecycle resource leaks.
- Explicit H100 run: `/tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json` passed 18/18 checks with endpoint `https://127.0.0.1:37095/v1/chat/completions`, `mtls_verified_peer_count=10`, admitted timeout status `504`, `request_timed_out_after_admit=true`, `request_timed_out_before_backend=true`, lifecycle accepted/cleanup count `6`, lifecycle timed-out count `1`, lifecycle resources allocated/released `30`, activation transfer `cuda:0` to `cuda:1` at about `244.98` GiB/s with about `0.0000638` s per transfer, split-pipeline throughput about `5185.64` tokens/s with max abs error about `6.71e-08`, MoE parity throughput about `1647.61` tokens/s and `3295.22` expert calls/s with max logit abs error `0.0`, target-fixture execution on `cuda:0` generated `fixture h100` at about `5026.16` tokens/s with max abs error about `2.37e-06`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this strengthens local endpoint failure-semantics and lifecycle cleanup evidence. It is not distributed partition handling, client-disconnect recovery, production mTLS/keying, real frontier target-model parity, real multi-host NVIDIA/AMD/Mac serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - System Engineering: approve with comments. The endpoint now covers success, rejection, cancellation, timeout, and cleanup states in one artifact; distributed partition behavior remains open.
  - Networking/Security: approve with comments. Timeout occurs only after local mTLS, bearer auth, and plan-integrity checks, preserving trust-boundary ordering.
  - Low-level Software: approve. Lifecycle ownership is explicit, terminal states are machine-checked, and backend request count is not inflated by timeout paths.
  - Software Engineering: approve. The timeout path reuses existing handler and artifact structure, has focused tests, and preserves default API semantics for normal requests.
  - Testing/Quality: approve. Focused local HTTP tests, CPU CLI smoke, H100 endpoint bundle, full unit suite, compile, Makefile test, golden, and diff-hygiene checks passed after stabilizing the backpressure holder window under full-suite load.
  - Program Management: approve with comments. This advances Phase 3 S3-4/S3-7 local evidence while keeping real distributed failure semantics and G3 open.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 8/8 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_timeout_cpu_smoke_20260622.json --backend-mode target-fixture --include-activation-transfer-probe --activation-transfer-backend cpu-stdlib --activation-transfer-iterations 2 --activation-transfer-warmup 0 --activation-transfer-payload-mib 1` showing 13/13 CPU checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json --backend-mode target-fixture --enable-mtls --include-activation-transfer-probe --activation-transfer-backend torch --activation-transfer-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --activation-transfer-source-device cuda:0 --activation-transfer-destination-device cuda:1 --include-runtime-probes --runtime-probe-backend torch --runtime-probe-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --runtime-probe-source-device cuda:0 --runtime-probe-destination-device cuda:1 --include-target-fixture-execution-probe --target-fixture-execution-backend torch --target-fixture-execution-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --target-fixture-execution-device cuda:0 --timeout-s 5` showing 18/18 H100 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 224 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local mTLS retry-after endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with a deterministic retry-after recovery path. The smoke now fills the local inflight slots, verifies an authenticated/plan-valid request receives HTTP `429` with `backpressure_queue_full` and `retry_after_ms=25`, releases the holder requests, then sends a fresh authenticated/plan-valid retry that is admitted and returns HTTP `200`.
- The endpoint artifact now separates four local failure/recovery cases: rejected backpressure before backend execution, successful retry after capacity clears, admitted cancellation before backend execution, and admitted timeout before backend execution. Backend request count rises only for successful executions, while lifecycle cleanup covers successful, cancelled, and timed-out admitted requests.
- Explicit H100 run: `/tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json` passed 19/19 checks with endpoint `https://127.0.0.1:40623/v1/chat/completions`, `mtls_verified_peer_count=11`, backend execution count `5`, retry-after recovery status `200`, `retry_after_ms=25`, lifecycle accepted/cleanup count `7`, lifecycle resources allocated/released `35`, activation transfer `cuda:0` to `cuda:1` at about `249.15` GiB/s with about `0.0000627` s per transfer, split-pipeline throughput about `5129.13` tokens/s with max abs error about `6.71e-08`, MoE parity throughput about `1458.73` tokens/s and `2917.46` expert calls/s with max logit abs error `0.0`, target-fixture execution on `cuda:0` generated `fixture h100` at about `5556.55` tokens/s with max abs error about `2.37e-06`, `target_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is local retry-after recovery evidence for localhost endpoint backpressure. It is not distributed retry, partition recovery, client disconnect recovery, production autoscaling, product mTLS/keying, real frontier target-model parity, real multi-host NVIDIA/AMD/Mac serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - Networking/Security: approve with comments. Retry recovery is exercised only after local mTLS, bearer auth, and plan-integrity checks; production retry policy and key distribution remain open.
  - System Engineering: approve with comments. The artifact now covers reject, retry, cancel, timeout, and cleanup in one endpoint lifecycle bundle; distributed partition semantics remain open.
  - Software Engineering: approve. Retry recovery is represented as an ordinary successful backend execution after capacity clears, so backend counts and lifecycle counts remain mechanically auditable.
  - Testing/Quality: approve. Focused local HTTP tests, CPU CLI smoke, H100 endpoint bundle, full unit suite, compile, Makefile test, golden, and diff-hygiene checks passed.
  - Program Management: approve with comments. This advances Phase 3 S3-4 retry semantics under local evidence scope while keeping real G3 heterogeneous serving open.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 8/8 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_retry_cpu_smoke_20260622.json --backend-mode target-fixture --include-activation-transfer-probe --activation-transfer-backend cpu-stdlib --activation-transfer-iterations 2 --activation-transfer-warmup 0 --activation-transfer-payload-mib 1` showing 14/14 CPU checks passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json --backend-mode target-fixture --enable-mtls --include-activation-transfer-probe --activation-transfer-backend torch --activation-transfer-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --activation-transfer-source-device cuda:0 --activation-transfer-destination-device cuda:1 --include-runtime-probes --runtime-probe-backend torch --runtime-probe-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --runtime-probe-source-device cuda:0 --runtime-probe-destination-device cuda:1 --include-target-fixture-execution-probe --target-fixture-execution-backend torch --target-fixture-execution-torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --target-fixture-execution-device cuda:0 --timeout-s 5` showing 19/19 H100 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 224 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local mTLS partition-fence endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with a deterministic local partition-fence path. The request now passes local mTLS, bearer-token auth, plan-hash validation, and admission, allocates normal lifecycle resources, records a partition-fenced terminal state through the serving gateway and scheduler, returns HTTP `503` with `network_partition_fenced`, releases all lifecycle and inflight state, and does not call `LocalFornaxBackend`.
- The endpoint artifact now separates five local failure/recovery cases: rejected backpressure before admission, successful retry after capacity clears, admitted cancellation before backend execution, admitted timeout before backend execution, and local partition fencing before backend execution. Backend execution count remains tied only to successful requests.
- Explicit H100 run: `/tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json` passed 20/20 checks with endpoint `https://127.0.0.1:33217/v1/chat/completions`, `mtls_verified_peer_count=12`, backend execution count `5`, partition fence status `503`/`network_partition_fenced`, lifecycle accepted/cleanup count `8`, lifecycle partitioned count `1`, lifecycle resources allocated/released `40`, activation transfer `cuda:0` to `cuda:1` at about `257.25` GiB/s with about `0.0000607` s per transfer, split-pipeline throughput about `4617.57` tokens/s with max abs error about `6.71e-08`, MoE parity throughput about `1772.99` tokens/s and `3545.97` expert calls/s with max logit abs error `0.0`, target-fixture execution on `cuda:0` generated `fixture h100` at about `4800.63` tokens/s with max abs error about `2.37e-06`, `target_model_parity=false`, `production_partition_evidence=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is a localhost partition-fence simulation inside a measured local H100 endpoint bundle. It strengthens Phase 3 S3-4/S3-7 failure-semantics and lifecycle coverage, but it is not real distributed partition recovery, product mTLS/keying, real frontier target-model parity, real NVIDIA/AMD/Mac heterogeneous serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - Networking/Security: approve with comments. The partition fence occurs only after local mTLS, bearer auth, and plan-integrity checks; production keying and real partition handling remain open.
  - System Engineering: approve with comments. The endpoint artifact now covers reject, retry, cancel, timeout, partition fence, and cleanup in one lifecycle bundle; real multi-host failure semantics remain open.
  - Low-level Software: approve. Terminal state ownership is explicit, resources are released by the common cleanup path, and backend execution count is not inflated by partition-fenced requests.
  - Software Engineering: approve. The behavior is deterministic, locally testable, and encoded in the artifact validator rather than only in prose.
  - Testing/Quality: approve. Focused local HTTP tests, CPU CLI smoke, and H100 endpoint smoke passed before broader validation.
  - Program Management: approve with comments. This advances Phase 3 S3-4/S3-7 local evidence while keeping real distributed partition proof and G3 open.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 8/8 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_partition_cpu_smoke_20260622.json --backend-mode target-fixture --include-activation-transfer-probe --activation-transfer-backend cpu-stdlib --activation-transfer-iterations 2 --activation-transfer-warmup 0 --activation-transfer-payload-mib 1` showing 15/15 CPU checks passed, the H100 command above showing 20/20 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 224 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` all passed.

### Local mTLS partition-recovery endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with a deterministic local partition-recovery request after the partition fence. The smoke now proves that a plan-valid, authenticated request can be fenced with HTTP `503`/`network_partition_fenced`, then a fresh plan-valid request is admitted and returns HTTP `200` after the simulated partition clears.
- The endpoint artifact now separates six local failure/recovery cases: rejected backpressure before admission, successful retry after capacity clears, admitted cancellation before backend execution, admitted timeout before backend execution, local partition fencing before backend execution, and local recovery after the partition fence. Backend execution count is still tied only to successful requests.
- Explicit H100 run: `/tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json` passed 21/21 checks with endpoint `https://127.0.0.1:37199/v1/chat/completions`, `mtls_verified_peer_count=13`, backend execution count `6`, partition fence status `503`/`network_partition_fenced`, partition recovery status `200`, lifecycle accepted/cleanup count `9`, lifecycle partitioned count `1`, lifecycle resources allocated/released `45`, activation transfer `cuda:0` to `cuda:1` at about `231.33` GiB/s with about `0.0000675` s per transfer, split-pipeline throughput about `2931.70` tokens/s with max abs error about `6.71e-08`, MoE parity throughput about `1871.53` tokens/s and `3743.07` expert calls/s with max logit abs error `0.0`, target-fixture execution on `cuda:0` generated `fixture h100` at about `5096.16` tokens/s with max abs error about `2.37e-06`, `target_model_parity=false`, `production_partition_evidence=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is localhost partition recovery inside a measured local H100 endpoint bundle. It is not real distributed partition recovery, client reconnect semantics, product mTLS/keying, real frontier target-model parity, real NVIDIA/AMD/Mac heterogeneous serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - Networking/Security: approve with comments. Recovery is allowed only after local mTLS, bearer auth, and plan-integrity checks; production key distribution and real partition handling remain open.
  - System Engineering: approve with comments. The endpoint artifact now covers reject, retry, cancel, timeout, partition fence, partition recovery, and cleanup in one lifecycle bundle; real multi-host failure semantics remain open.
  - Low-level Software: approve. The fenced request remains terminal before backend execution, while the recovery request is a normal successful backend execution with independent lifecycle cleanup.
  - Software Engineering: approve. The validator now rejects artifacts that claim partition behavior without the recovery response, summary fields, and check name.
  - Testing/Quality: approve. Focused local HTTP tests, CPU CLI smoke, and H100 endpoint smoke passed before broader validation.
  - Program Management: approve with comments. This advances Phase 3 S3-4/S3-7 local failure-semantics evidence while keeping real distributed partition proof and G3 open.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py tests/test_fornax_planner.py` passed; focused local HTTP serving smoke tests showed 8/8 passed; `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_partition_recovery_cpu_smoke_20260622.json --backend-mode target-fixture --include-activation-transfer-probe --activation-transfer-backend cpu-stdlib --activation-transfer-iterations 2 --activation-transfer-warmup 0 --activation-transfer-payload-mib 1` showed 16/16 CPU checks passed; the H100 command above showed 21/21 checks passed; `python3 -m unittest tests.test_fornax_planner` passed 224 tests; `python3 -m compileall -q fornax tests` passed; `make fornax-test` passed 224 tests; `make fornax-golden` passed; and `git diff --check` passed.


### Local mTLS Retry-After header endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` so deterministic local backpressure returns real HTTP retry metadata, not only JSON body metadata. The local 429 path now emits standard `Retry-After: 1` plus exact `X-Fornax-Retry-After-Ms: 25`, while the body still records `retry_after_ms=25`.
- The smoke client now captures response headers for JSON responses and HTTP errors. The fixture validator recomputes expected retry metadata from `config.retry_after_ms` and rejects artifacts missing the `responses.backpressure_reject.headers`, `summary.backpressure_retry_after_header_*`, `failure_semantics.retry_after_header_*`, or `http-retry-after-header` check.
- Explicit H100 run: `/tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_retry_header_smoke_20260623.json` passed 22/22 checks with endpoint `https://127.0.0.1:44039/v1/chat/completions`, `mtls_verified_peer_count=13`, backend execution count `6`, backpressure status `429`, `Retry-After: 1`, `x-fornax-retry-after-ms=25`, retry recovery status `200`, partition fence status `503`/`network_partition_fenced`, partition recovery status `200`, lifecycle accepted/cleanup count `9`, lifecycle timed-out count `1`, lifecycle partitioned count `1`, lifecycle resources allocated/released `45`, activation transfer `cuda:0` to `cuda:1` at about `199.67` GiB/s with about `0.0000783` s per transfer, split-pipeline throughput about `3429.29` tokens/s with max abs error about `6.71e-08`, MoE parity throughput about `1452.15` tokens/s and `2904.31` expert calls/s with max logit abs error `0.0`, target-fixture execution on `cuda:0` generated `fixture h100` at about `5803.20` tokens/s with max abs error about `2.37e-06`, `target_model_parity=false`, `production_partition_evidence=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is localhost HTTP retry metadata and local recovery evidence inside a measured H100 endpoint bundle. It is not distributed retry behavior, production autoscaling, client reconnect semantics, product mTLS/keying, real frontier target-model parity, real NVIDIA/AMD/Mac heterogeneous serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - Networking/Security: approve with comments. The endpoint now exposes machine-readable retry timing in HTTP metadata after local mTLS, bearer auth, and plan-integrity checks; production retry policy and key distribution remain open.
  - System Engineering: approve with comments. The artifact now covers reject, HTTP retry metadata, retry recovery, cancel, timeout, partition fence, partition recovery, and cleanup in one lifecycle bundle; real multi-host failure semantics remain open.
  - Low-level Software: approve. Header emission and capture are local to the HTTP boundary, and validator checks prevent stale artifacts from silently passing.
  - Software Engineering: approve. The behavior is deterministic, keeps the exact millisecond value separate from the standards-compliant seconds header, and preserves the existing body contract.
  - Testing/Quality: approve. Focused local HTTP tests, CPU CLI smoke, H100 endpoint bundle, full unit suite, compile, Makefile test, golden, and diff-hygiene checks passed.
  - Program Management: approve with comments. This advances Phase 3 S3-4 local endpoint backpressure semantics while keeping distributed retry/failure proof and G3 open.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py tests/test_fornax_planner.py` passed; focused local HTTP serving smoke tests showed 8/8 passed; `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_retry_header_cpu_smoke_20260623.json --backend-mode target-fixture --include-activation-transfer-probe --activation-transfer-backend cpu-stdlib --activation-transfer-iterations 2 --activation-transfer-warmup 0 --activation-transfer-payload-mib 1` showed 17/17 CPU checks passed; the H100 command above showed 22/22 checks passed; `python3 -m unittest tests.test_fornax_planner` passed 224 tests; `python3 -m compileall -q fornax tests` passed; `make fornax-test` passed 224 tests; `make fornax-golden` passed; and `git diff --check` passed before commit hygiene.

### Local mTLS topology-route endpoint smoke milestone

- Extended `fornax.local_http_serving_smoke` with a measured-probe-triggered `local_topology_route` artifact. The route correlates the localhost serving gateway, scheduler admission, activation-transfer transport, split-pipeline runtime path, remote-MoE expert route, and target-fixture execution into one local logical-host record.
- The topology route is deliberately scoped as `local-logical-host-only`: it records `production_topology_evidence=false`, `distributed_topology_evidence=false`, and `g3_gate_evidence=false`, and includes explicit deferred hardware explanations for the missing AMD GPU node and Apple Silicon Mac lanes.
- Added validator and unit coverage so artifacts with measured probes must include a valid `local-topology-route` check, matching summary fields, at least two logical hosts, required component names for the enabled probes, and the deferred AMD/Apple hardware entries. Baseline local HTTP smoke continues to omit the route.
- Explicit CPU run: `/tmp/fornax_local_http_serving_topology_route_cpu_smoke_20260623.json` passed 18/18 checks with target-fixture backend mode and CPU activation-transfer probe coverage.
- Explicit H100 run: `/tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_topology_route_smoke_20260623.json` passed 23/23 checks with endpoint `https://127.0.0.1:34133/v1/chat/completions`, `mtls_verified_peer_count=12`, backend request count `5`, HTTP `Retry-After: 1` and `x-fornax-retry-after-ms=25`, lifecycle accepted/cleanup count `8`, lifecycle resource allocation/release count `40`, activation transfer `cuda:0` to `cuda:1` at about `296.44` GiB/s, split-pipeline throughput about `4102.43` tokens/s, MoE parity throughput about `1207.39` tokens/s and `2414.78` expert calls/s, target-fixture execution on `cuda:0` at about `2943.24` tokens/s, local topology route over `localhost`, `logical-host-0`, and `logical-host-1` with 6 components, deferred AMD/Apple hardware explanations, `target_model_parity=false`, and `g2_g3_gate_evidence=false`.
- Boundary remains explicit: this is localhost endpoint plus same-host H100 logical-host evidence. It strengthens Phase 3 S3-2/S3-6/S3-7 local evidence, but it is not real heterogeneous fabric transport, product auth/mTLS/keying, distributed partition proof, real frontier target-model loading/parity, NVIDIA/AMD/Mac frontier serving, predicted-throughput proof, or G3 closure evidence.
- Review-lens pass:
  - System Engineering: approve with comments. The endpoint artifact now ties serving, admission, local transport, runtime probes, and fixture execution into one route record, while real distributed serving remains open.
  - Networking/Security: approve with comments. The route complements local mTLS, bearer auth, and plan-hash checks without claiming product keying or real node identity.
  - Hardware Acceleration: approve with comments. H100 probes remain measured CUDA execution, but logical hosts are same-machine stand-ins and not heterogeneous lab proof.
  - Low-level Software: approve. The route is derived from validated probe summaries and does not conflate endpoint backend request counts with post-endpoint probe execution.
  - Software Engineering: approve. The feature is opt-in through measured probes, keeps default smoke output unchanged, and is machine-checked by the artifact validator.
  - Testing/Quality: approve. Focused local HTTP tests, CPU CLI smoke, H100 endpoint smoke, full unit, compile, Makefile test, golden, and diff-hygiene validation passed.
  - Program Management: approve with comments. This updates the active Phase 3 lane from endpoint/runtime evidence to endpoint/runtime/topology-route evidence while keeping G3 open.
- Verification: `python3 -m py_compile fornax/local_http_serving_smoke.py tests/test_fornax_planner.py`, focused local HTTP serving smoke tests showing 8/8 passed, `python3 -m fornax program local-http-serving-smoke --out /tmp/fornax_local_http_serving_topology_route_cpu_smoke_20260623.json --backend-mode target-fixture --include-activation-transfer-probe --activation-transfer-backend cpu-stdlib --activation-transfer-iterations 2 --activation-transfer-warmup 0 --activation-transfer-payload-mib 1` showing 18/18 CPU checks passed, the H100 command above showing 23/23 checks passed, `python3 -m unittest tests.test_fornax_planner` showing 224 tests passed, `python3 -m compileall -q fornax tests`, `make fornax-test`, `make fornax-golden`, and `git diff --check` passed.
