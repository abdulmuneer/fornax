# Fornax V0 Target Contract

Version: 0.4-draft  
Plan: `project-plan-v4.md`  
Status: Approved for assumption-driven Engine v0; physical fields remain G2-open  
Last updated: 2026-07-10

## Contract purpose

This contract separates Engine v0 construction, physical runtime validation, and
the eventual frontier-capacity proof. Phase 0.5 binds the mechanism target and its
simulation envelope below. Physical evidence replaces assumptions when hardware
is available and gates G2/G3, not the start of implementation.

No number in this document is a measured result unless its evidence field names a
physical artifact in `evidence-register.md`. Simulated numbers cite scenario and
assumption IDs from `simulation-and-assumption-contract.md`.

## A. Mechanism target — binding for Phase 0.5

### Model

| Field | Value | Source/status |
|---|---|---|
| Repository/model | `deepseek-ai/DeepSeek-V2-Lite-Chat` | Local snapshot present |
| Snapshot | `85864749cd611b4353ce1decdb286193298f64c7` | Local HF cache |
| Architecture | `DeepseekV2ForCausalLM` | `config.json` |
| Model type | `deepseek_v2` | `config.json` |
| Storage dtype | BF16 | `config.json`; Phase 0.5 binding |
| Hidden size | 2048 | `config.json` |
| Layers | 27 | `config.json` |
| Attention heads | 16 | `config.json` |
| Routed experts | 64 | `config.json` |
| Shared experts | 2 | `config.json` |
| Experts selected/token | 6 | `config.json` |
| Weight count | 15,706,484,224 parameters | Existing local model check |
| Stored weights | 31,413,626,576 bytes | Existing local model check |
| Canonical tokenizer/template hash | Open — record before first T3 run | G2 blocker; simulator uses a versioned fixture hash |

The mechanism target is allowed to fit a single node. Its purpose is to validate
the cross-vendor Stage ABI, not capacity scaling.

### Engine v0 simulation fleet

| Worker | Assumed capability envelope | Rule |
|---|---|---|
| `sim-nvidia-stage-0` | 80 GiB usable-device profile, BF16 stage capable | Memory is a feasibility scenario, not an H100 measurement |
| `sim-apple-stage-1` | 128 GiB unified-memory profile, BF16 stage capable | Compute/kernel correctness remains assumption SA-001 |
| `loopback-fabric` | Parameterized latency/bandwidth/fault profiles | Every run names a scenario; no value is labeled measured |

The executable simulator must support at least 1, 10, 25, and 100 Gbit/s nominal
link profiles; RTT points 0.1, 0.5, 1, and 5 ms; configurable stage service times;
queue/byte credits; disconnect, timeout, corruption, and slow-worker injection.
The scenario matrix, rather than one favorable default, is the planning input.

### Future physical fleet

| Role | Binding target | Known evidence | Open fields before T3 |
|---|---|---|---|
| Stage 0 | Existing Linux NVIDIA host, one NVIDIA H100 80 GB device | Same-host H100 smokes exist | Host SKU, CPU/RAM, OS, NVIDIA driver, NIC, link route |
| Stage 1 | Apple M3 Max, 40-core GPU, 128 GB unified memory | Short source-built MAX generation exists | Exact macOS build, free memory at run, NIC/link route, thermal state |
| Fabric | Physical Ethernet or Thunderbolt IP between the two hosts | Not yet measured for this contract | MTU, route, measured one-way/round-trip latency, payload bandwidth |

The first evidence packet must fill every open field. Substitution of either node
requires a target-contract revision, not an undocumented run.

### MAX and toolchain pin

| Field | Value/status |
|---|---|
| MAX upstream base | `0735fa29762a5c53d65a0456d0b53eac1472180f` |
| Current local Apple patch commit | `957aeded5296d6638386409849b60f82c36146dd` |
| Reported MAX CLI | `MAX 26.5.0.dev2026063006` |
| Root Fornax dependency pin | Open — required before T3 evidence is accepted |
| Fresh-clone rebuild | Open — required before G2 `PROCEED` |
| Linux build equality/compatibility | Open — both workers must report accepted build identities |

See `max-fork-build-reproducibility.md` and ADR-0001.

### Workload

| Field | Binding value |
|---|---|
| Prompt/context validation point | 4096 tokens |
| Short diagnostic contexts | 16, 128, and 512 tokens |
| Generated tokens | 128 for correctness; sustained run separately |
| Concurrency points | 1, 4, 8 |
| Sampling | Greedy/temperature 0 for parity runs |
| Transport dtype | BF16 unless a recorded compatibility failure forces FP16 |
| Stage mode | Complete contiguous layer groups; all experts stage-local |
| Initial cut candidate | Stage 0 layers 0–13; Stage 1 layers 14–26 |
| Cut selection | May change only from measured memory/stage profiles and must be recorded in the plan artifact |
| KV ownership | Stage-local; no KV transfer in Phase 0.5 baseline |
| Transport | Stage-boundary activations over the v1 TCP frame protocol |

### Memory budget method

Each worker evidence record must report, before and at peak:

- assigned weight bytes and loaded parameter count;
- KV allocation for context and concurrency;
- activation input/output and double-buffer bytes;
- graph compiler/runtime reserve;
- serialization/staging buffers;
- OS/process baseline;
- free/allocated/reserved device or unified memory;
- fragmentation/headroom estimate;
- measured peak and post-request retained allocation.

Mechanism acceptance requires at least 10% free operational headroom after the
measured peak on each worker. This is a contract threshold, not a claim that the
current path meets it.

### Correctness thresholds

| Check | Provisional Phase 0.5 threshold |
|---|---|
| BF16 operator/stage activation comparison | `atol <= 0.02`, `rtol <= 0.02`, plus finite-value check |
| Final logits | Same provisional tolerance and identical top-1 token for the deterministic corpus |
| Token sequence | Exact match under greedy decoding for the accepted prompt corpus |
| Routing | Same selected expert IDs; top-k weights within dtype tolerance |
| Malformed/stale frame | Rejected before stage execution |
| Repeated run | No unexplained nondeterministic divergence |

Before G2, the LLM/correctness owner must either accept these thresholds from a
reference-error study or revise them with the study attached. A loose threshold
cannot be introduced merely to make a failing backend pass.

### Performance and stability thresholds

| Metric | Phase 0.5 Engine v0 rule |
|---|---|
| Attribution | Stage compute, pack, network, queue, unpack, and exposed wait reported separately with scenario IDs |
| Determinism | Repeated runs with the same seed/scenario produce identical results and event ordering where the contract requires it |
| Sustained run | 30 minutes at the highest simulated contracted concurrency with bounded queues and memory accounting |
| Scenario coverage | Best/base/worst compute-link combinations plus fault cases; no cherry-picked single scenario |
| Planner consistency | Planner and simulator agree on feasibility and resource accounting for deterministic inputs |
| Physical comparison | Deferred to G2; same schemas and workload replayed without contract changes |

### Assumption-driven narrow metric

If no plausible point in the approved compute/link assumption envelope can reach
50% pipeline utilization at concurrency 8, the current frontier-throughput thesis
does not proceed unchanged even before hardware arrives. The Sponsor chooses one
of:

- `ITERATE` on stage balance/transport;
- `NARROW` to capacity-first or homogeneous-island serving;
- `KILL` the cross-vendor spanning target.

## B. Frontier-capacity target — selection gate

### Candidate set

| Candidate | Current status |
|---|---|
| Qwen3-235B-A22B-class MoE | Preferred candidate; encoding/backend/fleet budget unclosed |
| DeepSeek-R1 671B/37B-active-class | Stretch only; memory and throughput unclosed |
| Alternative MAX-supported MoE | Allowed if it better isolates the capacity thesis; replacement rationale required |

### Required binding fields before Phase 2

- Exact model snapshot, tokenizer/template hashes, quantization, and quality
  acceptance.
- Exact fleet SKUs, usable memory, OS, drivers, MAX builds, NICs, switch, and
  topology.
- Per-node weights, KV, activations, buffers, reserves, and headroom.
- Stage cut, native homogeneous islands, and Apple role.
- Context and concurrency distribution from the target persona.
- Single-node quantized, naive pipeline, capacity-offload, and existing-engine
  baselines where applicable.
- Aggregate throughput threshold, planner-error threshold, and kill metric.
- Procurement and power/thermal envelope.

### Provisional product thresholds

- Model exceeds the largest node's usable memory at the selected encoding.
- Fleet memory closes with at least 10% operational headroom per node.
- Aggregate saturated throughput reaches at least 60% of the contract-defined
  sum-of-node ideal, unless a Sponsor-approved revision records a more meaningful
  target.
- Planner error is within +/-20% at G2 and +/-10% at G3.
- Saturation concurrency does not exceed the persona's validated supply.

These are thresholds, not current results.

## Evidence and approval status

| Contract section | Status | Closure needed |
|---|---|---|
| Mechanism model | Bound | Tokenizer/template hash |
| Simulation fleet | Bound | Engine v0 scenario execution |
| Physical fleet | Partial | G2: exact Linux host, OS/driver/NIC, physical-link measurements |
| MAX build | Partial | Root pin, Linux compatibility, fresh-clone rebuild |
| Workload | Bound | Engine v0 execution, then identical physical replay |
| Memory | Method bound | Measured budget and headroom |
| Correctness | Provisional | Engine reference parity; G2 physical parity |
| Performance | Thresholds bound | Physical results and planner calibration |
| Apple role | Open | Parity plus throughput evidence |
| Frontier target | Open | Post-mechanism selection and full budget |

Sponsor approval authorizes Engine v0 implementation under the named assumptions.
TL acceptance of the schemas is part of the G1 documentation record. Open
physical fields remain explicit G2 blockers and cannot be treated as supported
hardware evidence.
