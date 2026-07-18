# Fornax Cost Model and Calibration Contract

Version: 1.0-draft  
Plan: `project-plan-v4.md` §6  
Status: hardware-independent provenance/capability authority and the separate
SHA-bound evidence resolver are implemented; physical G2 profiles and
calibration remain pending

Plan v4's authority wording is interpreted by the dated
[`planner-status-erratum-2026-07-17.md`](planner-status-erratum-2026-07-17.md).

## 1. Policy

The planner separates modeled feasibility from deployment authority. Its
backward-compatible default mode labels every output `exploratory`. Deployment
mode fails closed on missing or incompatible capabilities, unmeasured critical
inputs, unattributed calibration, an error bound above the target threshold, or
an unresolved evidence reference. Model, inventory, link, and target files only
declare source IDs. A separate evidence registry must resolve every declaration
used during deployment search to a content-verified artifact record. No complete
physical registry or calibration is bundled, so current repository fixtures
remain exploratory and cannot close G2.

## 2. Planner repair ledger

| Repair | Current status | Acceptance test |
|---|---|---|
| Charge remote expert weights/buffers to each expert host | Implemented at T0 | A one-byte expert host cannot receive a larger expert placement |
| Preserve capacity/topology candidates when node count >6 | Implemented at T0 | A slower high-memory node remains discoverable and makes the reproduced fleet feasible |
| Use `supports_kv` for attention stages | Implemented at T0 | KV-incapable node cannot host an attention layer requiring KV |
| Enforce backend/build/dtype/operation/quantization compatibility | Implemented for declared complete capabilities and fail-closed deployment admission | Unsupported runtime/build/operation/dtype/quantization is infeasible with explanation |
| Model expert-host contention | Required; not accepted as physically calibrated | Two stages sharing one host consume shared compute/memory/network capacity |
| Model replica routes independently | Required; physical calibration open | Replica cost uses its actual predecessor/successor links |
| Use trace coactivation/locality | Schema/target behavior only; remote-expert placement is not product-enabled | Expert placement accounts for observed joint load or labels it uncalibrated |
| Bound search approximation | Feasibility-preserving regression exists; approximation reporting remains subject to G2 review | Search reports approximation mode, candidates pruned, and a feasibility-preserving fallback |

## 3. Calibration records

Each measured record identifies:

- hardware SKU/device and usable memory;
- OS, driver, MAX/Mojo/Fornax build;
- model snapshot, layer range, dtype/encoding;
- prompt/context, microbatch/concurrency, phase;
- warmup, iterations, thermals, and exact command;
- median, p95, and dispersion;
- correctness result tied to the same case.

## 4. Primitive profiles

| Profile | Dimensions |
|---|---|
| Stage graph | layer range, prefill/decode, token rows, context, dtype |
| Weight load | shard bytes, source storage, compile/cache state |
| KV | layers, context, concurrency, page size, dtype |
| Pack/unpack | dtype, shape, source/destination residency |
| Link | payload bytes, direction, concurrency, warm/cold connection |
| End-to-end boundary | pack + send + receive + unpack + queue/exposed wait |
| Expert MLP | expert shape, route count, dtype, resident/cold state |

Peak FLOP/s and memory-bandwidth specifications may be retained as priors, never
as evidence of measured stage time.

## 5. Prediction model

The target calibrated model for each stage/microbatch is:

```text
service = measured_stage_graph(shape, phase, context)
boundary = pack + wire + unpack
exposed_boundary = max(0, boundary - legal_overlap)
stage_interval = queue + service + exposed_boundary + remote_expert_exposed
pipeline_throughput = completed_tokens / max_stage_interval
```

Interpolation is allowed only inside a profile's declared shape range. Outside
that range, the estimate is extrapolated and cannot be gate evidence.

## 6. Memory feasibility

Memory is charged per node for:

- resident weights and quantization metadata;
- stage-local KV at target context/concurrency;
- activation input/output and overlap buffers;
- router/expert metadata and expert weights;
- serialization and transport staging;
- compiler/runtime/allocator reserve;
- OS/process reserve and fragmentation margin.

Shared resources are charged once plus per-request/per-channel allocations as
applicable. The planner rejects a plan below the target contract's headroom.

## 7. Confidence and admission

| Status | Meaning | Deployment authority |
|---|---|---|
| `measured` | Exact or accepted interpolated physical profile | Eligible |
| `estimated` | Calibrated surrogate with error history | Exploratory unless Sponsor waives |
| `uncalibrated` | Peak/spec/default assumption | Never |
| `unsupported` | Missing capability or failed correctness | Never |

Target requirement: every deployment-authoritative prediction must carry source
record IDs and an expected error interval, and every source ID must resolve
through a separate evidence registry.

The implemented input schema is:

```json
{
  "measurement_provenance": {
    "compute_class": {
      "status": "measured",
      "source_id": "EV-G2-stage-profile-001",
      "confidence": "high",
      "expected_relative_error": 0.08
    }
  }
}
```

Nodes carry this mapping for `mem_free_bytes`, `compute_class`, and
`mem_bandwidth_bytes_s`; links carry it for `bandwidth_bytes_s` and `latency_s`.
Targets carry the same shape as `prediction_calibration`, plus
`max_expected_relative_error`. Models identify their source snapshot and
quantization evidence. Complete node capabilities identify `build_id`,
`supported_operations`, `supported_quantizations`, and `capability_source_id`.

### 7.1 Separate evidence registry

Source strings inside planner inputs are references, not proof. Deployment mode
requires a second file with schema
`fornax.planner-evidence-registry.v1`. Each record binds one source ID to an
evidence type and an artifact's exact SHA-256:

```json
{
  "schema_version": "fornax.planner-evidence-registry.v1",
  "records": [
    {
      "source_id": "EV-G2-stage-profile-001",
      "evidence_type": "measurement",
      "artifact_path": "artifacts/EV-G2-stage-profile-001.json",
      "artifact_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "status": "active",
      "not_before": "2026-07-01T00:00:00Z",
      "expires_at": "2026-10-01T00:00:00Z"
    }
  ]
}
```

`artifact_path` is resolved relative to the registry file unless absolute. The
resolver reads the artifact and compares its bytes with `artifact_sha256`.
Missing files, hash mismatches, absent IDs, wrong evidence types, revoked
records, records that are not active yet, and expired records all reject
deployment authority. `not_before` and `expires_at` are optional, timezone-aware
ISO-8601 timestamps; use `expires_at` for hardware or route evidence whose
validity is time bounded.

Evidence types are `model`, `quantization`, `expert_trace`, `capability`,
`measurement`, `calibration`, and `route`. Node numeric profiles resolve as
`measurement`; link bandwidth/latency profiles resolve as `route`. Deployment
search can rank or exclude every candidate node and link, so resolution covers
the complete input inventory, not only the winning stage list.

`PlacementPlan.authority` emits requested mode, status, source IDs, the registry
manifest SHA-256, aggregate confidence, prediction error, maximum declared input
error, and rejection reasons. `Predicted.prediction_intervals` emits a symmetric
throughput interval only when a calibration record declares an expected relative
error; no interval is invented for uncalibrated throughput, and the throughput
calibration is not misapplied to TTFT or latency.

Authority statuses are:

| Plan status | Meaning |
|---|---|
| `exploratory` | The default; useful for simulation/search, never deployment authority |
| `rejected` | Deployment was requested but at least one authority check failed |
| `deployment_authoritative` | Exact capability checks passed and every source reference resolved to an active SHA-verified registry record; the evidence must still be governed and physically authentic |

Use the following form for fail-closed CLI admission (`authoritative` remains an
alias for `deployment`):

```bash
python3 -m fornax plan \
  --target target.json \
  --inventory inventory.json \
  --authority-mode deployment \
  --evidence-registry planner-evidence-registry.json \
  --out plan.json
```

Omitting `--evidence-registry` in deployment mode writes a rejected plan. Exact
declarations are still checked in exploratory mode when supplied: an explicitly
incompatible runtime, build, operation, or quantization is not silently treated
as unknown. Supplying a registry never turns an exploratory request into an
authorized plan.

## 8. Phase 0.5 disposition and open G2 acceptance

Recorded Phase 0.5 T0 evidence establishes only that both reproduced planner
defects are fixed and covered by tests. Its selected plans use explicit fixture
or assumption inputs; they are not evidence of measured physical memory,
runtime/build compatibility, route timing, or prediction accuracy.

G2 acceptance remains unmet until:

- the selected two-node plan is feasible under measured usable memory;
- stage and route predictions use physical records from the accepted builds;
- runtime/build/dtype/operation/quantization compatibility is populated from
  physical backend capability reports; and
- maximum absolute predicted/measured throughput error is <=20% over the
  contracted operating range.

A fixture cannot manufacture calibration accuracy or deployment authority. The
resolver proves that a cited local artifact existed with the recorded bytes at
planning time; it is not a signature, remote attestation, or proof that those
bytes came from the claimed physical run. Evidence acquisition, review, and
registry governance remain external gate duties. Remote-expert plans remain
non-authoritative until their contention/route model is calibrated; deployment
mode also omits exploratory replica allocation until replica routes are modeled
independently.

## 9. Later tightening

- G3: <=10% maximum error for the benchmark-of-record fleet/model.
- Remote experts: separate service-demand, batching, queueing, and coactivation
  model required before placement is enabled.
