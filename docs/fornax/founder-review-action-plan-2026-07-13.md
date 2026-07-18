# Founder review and action plan

Date: 2026-07-13  
Plan of record: `project-plan-v4.md`  
Decision posture: `ITERATE` on product readiness; continue Phase 1 toward G2

> This is a simulated review using public compiler/platform and
> developer-experience principles associated with Chris Lattner and Soumith
> Chintala. It is not their actual opinion, participation, endorsement, or a
> statement that they reviewed this repository.

## Joint verdict

Fornax is a credible pre-product systems program with an unusually strong
executable specification, evidence taxonomy, and MAX/Fornax ownership boundary.
It is not yet a launch-ready heterogeneous inference product. The honest current
claim is:

> Fornax is building a MAX-native heterogeneous inference engine. This
> repository currently contains its planner, executable contracts, two-worker
> reference/simulated engine, framed transport prototype, and single-node MAX
> bring-up evidence. Physical cross-vendor correctness and frontier-capacity
> economics remain open G2/G3 proofs.

The next fundable outcome is one product-shaped physical vertical slice plus
qualified design-partner evidence—not more proxy breadth.

## What must not change

- MAX owns per-node graph compilation, kernels, device execution, and local KV;
  Fornax owns cross-node planning, orchestration, transport, and evidence.
- Complete contiguous layer groups remain the cross-vendor spine.
- Network I/O remains outside compiled graphs.
- Remote experts remain measured optionality rather than a baseline dependency.
- Simulation, proxy, and physical measurements remain visibly distinct.
- Single-stream latency caveats and `PROCEED/ITERATE/NARROW/KILL` remain explicit.
- The reference implementation remains simple and inspectable even after a
  native hot path exists.

## Reconciled review disposition

| Priority | Comment | Evidence | Action taken now | Satisfaction condition |
|---|---|---|---|---|
| P0 | The physical MAX seam is not implemented end to end. | `MaxStageBackend` delegates to an optional adapter; Engine v0 workers instantiate the simulator. | Kept G2 open; added the vertical slice to the pitch and program critical path. | Tracked backend factory + concrete MAX adapter; clean-build contiguous stages; one real NVIDIA→Apple request; parity, timing, memory, stability, and planner calibration evidence. |
| P0 | ABI v1 cannot express real ragged continuous batching. | `TensorDescriptor` has no row offsets; `StageRequest` has one request ID/KV epoch; scheduler and stage execution remain separate. | Registered as an active technical issue; no false closure claim. | ABI v2 supports token/activation/logit roles, per-sequence row ranges/positions/KV epochs/errors, unequal prefill, independent decode and cancellation; same case passes two workers and physical MAX. |
| P0 | There is no humane adoption path. | First run previously required hand-authored inputs and exposed 29 peer CLI groups. | Added install metadata, `fornax quickstart`, task-focused top-level help, concise input errors, and real onboarding docs. | Correct single-node/reference generation and later distributed generation use the same request/response semantics; public CLI centers on serve/plan/inspect/benchmark/doctor. |
| P0 | Customer and concurrency fit are assumptions. | A-1/R-8 remain open; no interviews, traffic traces, commitments, pricing, or revenue artifacts exist. | Added a discovery/pilot lane and evidence requirements in `product-and-commercialization-strategy.md`. | Qualified interviews, real anonymized traces, committed design partners, at least one paid pilot, and concurrency fit—or an explicit `NARROW`. |
| P1 | Planner capability enforcement was overstated. | `supports_kv` existed but was not consumed; runtime/build/operation/calibration capabilities remain absent. | KV-bearing attention placement now rejects `supports_kv=false` with a regression test and explanation. | Backend/build/operation/quantization/calibration identities are discovered, modeled, and fail closed with regression coverage. |
| P1 | Worker capabilities echo requested manifest values. | `/capabilities` reports manifest build/dtypes rather than backend-discovered facts. | Registered as a G2 implementation requirement. | Immutable backend-generated capabilities are attested before manifest load; requested and observed identities are both recorded. |
| P1 | Python tensor materialization cannot be the production hot path. | Receive/CRC/tensor validation allocate and decode Python values. | Preserved as the conformance reference path; native buffer work stays open. | Pooled buffer views, incremental/native CRC, native finite checks, explicit MAX import/staging, and component benchmarks at contracted sizes. |
| P1 | Governance golden certified stale v3/G1 state. | Passing fixture said DEC-005 pending, G1 not ready, Phase 1 unauthorized. | Golden and validator now require plan v4, accepted G1, authorized Phase 1, and open G2; Stage ABI gate text corrected. | CI rejects plan-version or gate-posture drift. |
| P1 | Onboarding golden described nonexistent docs and Make targets. | Embedded paths and `make fornax-golden`/`make fornax-test` did not exist. | Materialized onboarding docs, corrected commands/config names, and made validation require repository files. | Docs smoke runs every first-run command and a fresh operator reproduces the supported path. |
| P1 | Product packaging and service economics were absent. | No install manifest, offer definition, pricing discovery, services boundary, customer metrics, or investor claim ledger. | Added `pyproject.toml` and the product/commercialization strategy. | Repeatable install, supported serving endpoint, priced pilot, support model, and measured unit economics. |

## Changes made in this review

### Developer experience

- `python3 -m fornax quickstart --out-dir <dir>` now produces an explicit target,
  synthetic heterogeneous inventory, two-stage placement, validation,
  simulation, and summary in one command.
- Every quickstart artifact says `simulation_fixture` and
  `physical_measurement=false`.
- Top-level CLI help explains workflows and the prediction/measurement boundary.
- Missing input paths return exit code 2 with a concise message instead of a
  traceback.
- `pyproject.toml` provides a pre-alpha `fornax` console entry point.
- `make doctor` now requires and forwards `BUNDLE=<path>`.
- Actual operator, developer, benchmark, and glossary documents replace
  imaginary onboarding paths and commands.

### Correctness and governance

- The planner now rejects KV-bearing attention stages on KV-incapable nodes.
- A regression test pins the rejection and its explanation.
- The program-governance golden now records DEC-005/DEC-008, plan v4, G1 passed,
  Phase 1 authorized, and G2 open.
- The governance validator rejects stale plan versions/gates.
- The wire-ABI spec now correctly requires physical conformance before G2, not
  before the already-passed G1.

### Product and venture readiness

- `product-and-commercialization-strategy.md` separates product, services,
  customer discovery, pilot evidence, business-model hypotheses, and claims.
- The pitch is re-centered on financing G2/G3 proof and design-partner
  conversion, not claiming GA.

## Golden-vector change record

Two golden contracts changed deliberately:

1. `program_governance/fixture.json` moved from historical v3/G1-pending state to
   current v4/G2-open state so a passing test detects rather than blesses status
   drift.
2. `onboarding_methodology/fixture.json` now names real repository documents and
   runnable Make targets. The validator requires those files to exist.

Neither change modifies engine, wire, numerical, or planner output semantics.

## Required next sequence

1. Design ABI v2 for ragged multi-sequence batching before optimizing the engine.
2. Add discovered backend capabilities and a serializable worker backend factory.
3. Pin the MAX fork from the root and implement the concrete local physical stage
   adapter.
4. Prove two real local contiguous stages, then cross the physical NVIDIA→Apple
   boundary and close G2 evidence.
5. In parallel, run qualified customer discovery and acquire traffic traces and
   design partners.
6. Bind the frontier target, measured baselines, customer-representative BOM, and
   TCO before G3 or an economics claim.

Until those close, the correct external label is **deep-tech pre-product with a
working executable specification**, not a launched frontier inference product.
