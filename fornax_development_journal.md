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
