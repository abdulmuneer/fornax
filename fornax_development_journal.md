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

