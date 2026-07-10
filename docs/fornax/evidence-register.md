# Fornax Evidence Register

Plan: `project-plan-v4.md`  
Status: Living index  
Last updated: 2026-07-10

## Classification

| Class | Meaning | Can close physical gate? |
|---|---|---|
| `T0-contract` | Unit/property/golden contract | No |
| `T1-simulation` | Deterministic simulated workers/protocol | No |
| `T2-physical-single-node` | Measured one physical backend/node | Only T2 criteria |
| `same-host-proxy` | Multiple devices/logical hosts on one physical machine | No for T3/T4 |
| `T3-physical-multinode` | Real network across physical nodes | Yes for G2 criteria named by plan |
| `T4-target-lab` | Exact heterogeneous capacity-target fleet | Yes for G3/G4 |
| `T5-operator` | Fresh installation/operator evidence | Yes for G5 |

## Current evidence

| ID | Date | Class | Evidence | Build/hardware | Gate effect |
|---|---|---|---|---|---|
| EV-001 | 2026-07-10 | T0-contract | `make test`: all golden/contracts plus 275 unit tests pass when localhost socket binding is permitted | Local Python environment | Supports contract and Engine v0 regression baseline only |
| EV-002 | 2026-07-01 | T2-physical-single-node | DeepSeek-V2-Lite short `max generate`, 1 and 8 tokens | M3 Max; source-built MAX `26.5.0.dev2026063006`; patch `957aede...` | Positive bring-up; no numerical parity/G2 closure |
| EV-003 | 2026-06-29 | same-host-proxy | Four-H100 real Qwen3-Omni text generation | Same Linux host, four H100 GPUs, Transformers/PyTorch path | Real model load/generation; not Fornax distributed/MAX T3 |
| EV-004 | 2026-06-23 | same-host-proxy | Local HTTP/TLS/runtime/topology bundle and Phase 3 proxy packet | Two H100 devices represented as logical hosts | Contract/proxy only; formal G3 false |
| EV-005 | 2026-06-23 | T1-simulation | Phase 4 resilience proxy | Simulation artifacts | Formal G4 false |
| EV-006 | 2026-06-23 | T1-simulation | Phase 5 GA proxy | Lifecycle/onboarding fixtures | Formal G5 false |

## Engine v0 closure and open G2 evidence slots

| ID | Class | Required artifact | Gate/sprint effect | Status |
|---|---|---|---|---|
| EV-007 | T0-contract | Planner defect regression report | Phase 0.5 exit | **Closed** — I-7/I-8 regressions pass in `tests/test_fornax_planner.py`; see [development journal](program_management/internal/journal/fornax_development_journal.md) |
| EV-008 | T0/T1 | Stage ABI and backend conformance corpus | Phase 0.5 exit | **Closed** — `fornax/golden_vectors/stage_abi_v1`, 24/24 checks |
| EV-009 | T1-simulation | Deterministic two-process pipeline/fault/sustained-run bundle | Phase 0.5 exit | **Closed** — [`evidence/phase05-engine-v0-2026-07-10.json`](evidence/phase05-engine-v0-2026-07-10.json), SHA-256 `d9f57d940306568959fd87139c0e95b8dcdd770166eabc2c31e9d425d40d1e37` |
| EV-010 | T2-physical-single-node | Fresh Apple numerical operator/stage parity | G2 / Phase 1 | Open |
| EV-011 | T2-physical-single-node | NVIDIA stage parity on accepted build | G2 / Phase 1 | Open |
| EV-012 | T3-physical-multinode | Physical Linux/NVIDIA -> macOS/Apple generation | G2 | Open |
| EV-013 | T3-physical-multinode | Concurrency and performance-attribution sweep | G2/G3 | Open |
| EV-014 | T3-physical-multinode | Stability and failure bundle | G2/G4 | Open |
| EV-015 | T3-physical-multinode | Planner calibration and Apple role recommendation | G2/G3 | Open |

EV-007 through EV-009 close Phase 0.5 at T0/T1. Physical slots EV-010 through
EV-015 remain open and block the corresponding G2/G3 physical or product claim.

## Record requirements

Every new row links or identifies a durable artifact containing:

- immutable model/tokenizer/template/build/plan hashes;
- exact hardware, OS, runtime, driver, network, and command;
- evidence class and whether the result is measured;
- correctness result and tolerance source;
- metric units and raw sample location;
- warnings/limitations and formal gate booleans;
- artifact SHA-256 and retention location.

If raw artifacts cannot be committed, store them in an approved durable location
and commit a manifest with hashes. An ephemeral local path is never the sole
evidence reference.
