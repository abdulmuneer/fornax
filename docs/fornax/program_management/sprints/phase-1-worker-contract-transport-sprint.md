# Phase-1 Worker Contract and Transport Sprint

**Goal:** turn the Phase-0 contracts into a runnable worker boundary and
simulated distributed runtime that can later be proven on a real 2-3 node fleet.

**Duration:** notional W5-W10.
**Milestone:** M2 Pipeline correctness (simulation, then 2-3 nodes).
**Gate contribution:** feeds G2, but does not close G2 without real 2-3 node
pipeline evidence.

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S1-1 | Stage host executes a layer-group boundary | RT | B2 | stage-host contract passes reference and boundary checks |
| S1-2 | Boundary activation/KV handoff semantics | RT + NET | B3 | golden vectors cover activation, KV pages, ownership, dtype, and shape |
| S1-3 | Slow-correct reference path | RT | B4 | optimized/simulated path has a reference comparator |
| S1-4 | Activation/KV transport contract | NET | E2 | simulated logical-host transport covers latency, bandwidth, ordering, and failure metadata |
| S1-5 | Engine backend skeleton | API + DIST | H3 | `FornaxBackend` seam records request, plan, tokenizer/template, and result contract fields |
| S1-6 | T1 worker-contract bundle | SRE + QA | B2-B4, E2, H3 | `fornax program simulate-t1` includes worker, transport, and serving checks |

## Sprint Board

| Deliverable | Status |
|---|---|
| S1-1 | Partial: simulated stage-host evidence exists; live MAX graphlet execution remains open. |
| S1-2 | Partial: runtime-format and state-ownership simulations exist; real custom ops remain open. |
| S1-3 | Done at T1 scope: slow-correct/reference fixtures exist. |
| S1-4 | Partial: logical-host transport simulation exists; real TCP/RDMA/TB-IP/shm transport remains open. |
| S1-5 | Partial: serving adapter skeleton exists; live backend endpoint remains open. |
| S1-6 | Partial: T1 bundle exists; G2 still needs real 2-3 node proof. |

## Validation

- `python3 -m fornax test runtime-format --golden fornax/golden_vectors/runtime_format`
- `python3 -m fornax test stage-host --golden fornax/golden_vectors/stage_host`
- `python3 -m fornax test network-contract --mode simulated`
- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`

## Exit Criteria

- Worker boundary, transport, and serving seam are deterministic under T1.
- Two local GPUs can be treated as two logical hosts for smoke evidence when
  accelerator validation is available.
- Remaining G2 gap is explicit: real 2-3 node pipeline correctness and planner
  accuracy evidence.
