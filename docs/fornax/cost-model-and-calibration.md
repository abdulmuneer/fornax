# Fornax Cost Model and Calibration Contract

Version: 1.0-draft  
Plan: `project-plan-v4.md` §6  
Status: Phase 0.5 planner repairs complete; physical G2 calibration pending

## 1. Policy

The planner separates feasibility from performance and measured inputs from
estimates. A plan containing an unmeasured critical backend operation or route is
`exploratory`; it cannot authorize model deployment or close a gate.

## 2. Required planner repairs

| Repair | Acceptance test |
|---|---|
| Charge remote expert weights/buffers to each expert host | A one-byte expert host cannot receive a larger expert placement |
| Preserve capacity/topology candidates when node count >6 | A slower high-memory node remains discoverable and makes the reproduced fleet feasible |
| Enforce backend/build/dtype/operation compatibility | Unsupported stage/expert/KV role is infeasible with explanation |
| Use `supports_kv` for attention stages | KV-incapable node cannot host an attention layer requiring KV |
| Model expert-host contention | Two stages sharing one host consume shared compute/memory/network capacity |
| Model replica routes independently | Replica cost uses its actual predecessor/successor links |
| Use trace coactivation/locality | Expert placement accounts for observed joint load or labels it uncalibrated |
| Bound search approximation | Search reports approximation mode, candidates pruned, and a feasibility-preserving fallback |

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

For each stage/microbatch:

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

Every prediction carries source record IDs and an expected error interval.

## 8. Phase 0.5 acceptance

- Both reproduced planner defects are fixed and covered by tests.
- The selected two-node plan is feasible under measured peak memory.
- Stage and route predictions use physical records.
- Maximum absolute predicted/measured throughput error is <=25% over the accepted
  mechanism cases.
- A prediction outside the bound produces G1 `ITERATE`; fixtures cannot be used to
  manufacture accuracy.

## 9. Later tightening

- G2: <=20% maximum error over the contracted operating range.
- G3: <=10% maximum error for the benchmark-of-record fleet/model.
- Remote experts: separate service-demand, batching, queueing, and coactivation
  model required before placement is enabled.
