# Quality and Test Governance

Plan: `../project-plan-v4.md` §8–§10

## Core rule

Engine code may advance under assumptions; capability claims may not. Reference,
simulation, same-host proxy, physical backend, physical multinode, target-lab, and
operator evidence remain separately classified.

## Test tiers

| Tier | Scope | Gate authority |
|---|---|---|
| T0 | Unit/property/golden contracts | Required everywhere; no hardware claim |
| T1 | Executable reference/simulated workers, virtual clock, multi-process loopback, fault injection | Engine v0/G1 scope |
| T2 | One physical backend with numerical reference | Backend assumption replacement |
| T3 | 2–3 physical nodes and real network | G2 |
| T4 | Exact frontier-capacity heterogeneous lab | G3/G4 |
| T5 | Fresh install/operator validation | G5 |

Same-host multi-device proxy evidence is recorded separately and never promoted to
T3.

## Phase 0.5 mandatory suites

### Planner regressions

- Remote expert host with insufficient memory must be infeasible.
- A feasible high-memory node remains searchable on fleets larger than six nodes.
- Memory/resource and KV-capable-node exclusions are enforced. Runtime/build,
  operation, dtype/quantization, structured provenance/confidence/error, and
  fail-closed deployment admission are implemented at T0 under I-16.
- Deployment authority still requires authentic physical source IDs and a
  measured calibration result within the current gate bound. Repository
  fixtures remain exploratory; copying `measured` into JSON is not evidence.

### Stage ABI conformance

- Valid BF16 activation/logit frames.
- Version, shape, byte-count, checksum, plan, manifest, sequence, deadline, and
  credit negatives.
- Duplicate/idempotency and cleanup semantics.
- Shared suite against reference, simulated, and eventually MAX backends.
- Candidate FNX2 T0/T1 conformance includes unequal prefill, row offsets,
  per-sequence position/KV/error state, independent decode, strict result
  correlation, generation-bound leases, bounded replay bytes, cancellation,
  and cleanup. The same corpus is mandatory on physical adapters before a
  throughput claim.

### Governance and onboarding freshness

- The current governance golden must match the plan-of-record version, accepted
  decisions, authorized phase, and next open gate; historical fixtures are
  labeled and cannot satisfy this lane.
- Onboarding fixtures must reference repository files and runnable Make/CLI
  commands. A missing document or stale command fails validation.
- Golden updates record the contract reason in the relevant review/spec.

### Engine lifecycle

- Admission, batching, queue/byte bounds, cancellation, timeout, disconnect,
  stale plan, and drain.
- Implemented T0/T1 mechanism: explicit final request/KV release, in-flight
  release rejection, idempotent repeat release, idle expiry, internal execution
  leases, same-worker tombstones, bounded idempotency/transform count/time/bytes,
  bounded event history, and health high-water counters.
- Before production memory/stability claims: restart-durable fencing, a reviewed
  long-duration many-unique-request artifact, physical/native-KV validation, and
  current evidence.
- Deterministic trace/metric/resource ledger.
- Thirty-minute simulation/loopback sustained run.
- Full named scenario matrix with seeds and SA-* provenance.

## Correctness-first rule

The slow reference path is the oracle. A simulated backend returns
reference-equivalent logical output while injecting service/failure behavior. A
physical MAX backend must pass numerical operator, stage activation, routing, and
final-logit comparisons before throughput evidence is accepted.

Finite output or non-empty generated text is a smoke condition, not correctness.

## Definition of done

| Deliverable | Done when |
|---|---|
| Engine code | Appropriate T0/T1 suites pass; scheduler/wire bounds and traces observable; final release and reference-retention bounds pass, while the remaining I-22 production criteria stay explicit |
| Assumption | Named, bounded, owned, and mapped to a physical replacement test |
| Simulation result | Scenario/seed/assumption IDs recorded; never labeled measured |
| Physical result | Exact hardware/build/model/command and correctness evidence recorded |
| Spec/ADR | Review authority, decision, consequences, and reversal trigger present |
| Gate packet | Evidence classes and unresolved assumptions explicit |

## Gate mapping

| Gate | Required quality evidence |
|---|---|
| G1 | Accepted v4 contracts/ADRs, active Engine v0 sprint, assumption register |
| G2 | Planner regressions, reproducible MAX builds, T2 parity, T3 physical generation/calibration |
| G3 | T4 capacity-target correctness/throughput/security |
| G4 | T4 real fault/recovery/scaling |
| G5 | T5 fresh operator and reproducible benchmark |

## CI lanes

- `cpu-contract`: T0 on every change.
- `engine-simulation`: T1 deterministic/virtual-clock on every change.
- `engine-loopback`: multi-process socket suite on every supported CI host.
- `max-nvidia`, `max-apple`, `max-amd`: opportunistic/pinned T2 hardware lanes.
- `physical-multinode`: scheduled T3 evidence lane.
- `docs-smoke`: quickstart, referenced onboarding files, CLI help/error exits,
  and documented no-hardware commands.

Hardware lane absence is reported as `not run`, never `pass` or a reason to stop
the CPU/loopback lanes.

## Phase 0.5 closure baseline

DEC-008 fixed the historical T0/T1 closure floor at:

- exact DeepSeek-shaped Stage ABI manifests and 24 conformance checks;
- two independent workers completing prefill/decode over loopback TCP;
- all 60 named scenario rows, seven fault outcomes, and scheduler concurrency
  1/4/8;
- at least 1,800 wall-clock seconds at observed concurrency 8 with credit/RSS
  bounds and clean teardown;
- the I-7/I-8 planner regressions; and
- `make test` passing all golden/contract suites plus 275 unit tests.

The immutable closure artifact is EV-009. The current Stage ABI v1 corpus has 31
checks, adding exact minor-version rejection, backend attestation, and bounded
wire replay. EV-009 remains valid
for its recorded scope but reports `current_contract_authority=false`; a fresh
sustained artifact is required for later gates. Regressing either the historical
floor or the current 31-check contract reopens the corresponding claim.
