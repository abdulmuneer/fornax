# Phase-3 Heterogeneous Frontier Sprint

**Goal:** assemble the heterogeneous serving path using simulation and local
logical-host validation first, then replace simulated hosts with the real
NVIDIA/AMD/Mac fleet when available.

**Duration:** notional W16-W24.
**Milestone:** M5 Heterogeneous frontier serve.
**Gate:** G3 Heterogeneous frontier.

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S3-1 | Apple expert-MLP role proof or demotion | KER + PM | D2-D3 | Apple role is compute, expert-host, or capacity-only based on measured/pinned evidence |
| S3-2 | Topology-aware activation/KV transport | NET + DIST | E2 | placement accounts for bandwidth, latency, and heterogeneity |
| S3-3 | Trust boundary and plan-integrity tags | SEC + NET | E3 | node identity, endpoint auth, and plan hash validation are enforced |
| S3-4 | Backpressure and failure semantics | NET + SRE | E4 | timeout, retry, cancel, and partition behavior are deterministic |
| S3-5 | Data-parallel stage replication | DIST | F3 | replicated stages preserve correctness and improve modeled throughput |
| S3-6 | Placement explanations | DIST + SRE | G3 | excluded/slow placement reasons are visible in artifacts |
| S3-7 | End-to-end lifecycle and state ownership | API + RT | H1 | request lifecycle, KV ownership, plan ownership, and cleanup are explicit |

## Sprint Board

| Deliverable | Status |
|---|---|
| S3-1 | Open for real Apple evidence; simulation must not pretend to close this gate item. |
| S3-2 | Partial: simulated transport and topology placement exist; real heterogeneous fabric remains open. |
| S3-3 | Partial: trust-boundary simulation exists; production auth/keying remains open. |
| S3-4 | Partial: backpressure and replay simulations exist; real partition/failure proof remains open. |
| S3-5 | Partial: stage-replication simulation exists; real replicated runtime remains open. |
| S3-6 | Done at planner artifact scope; live runtime linkage remains future work. |
| S3-7 | Partial: state-ownership simulation exists; localhost HTTP lifecycle cleanup and local target-fixture parity evidence exist; live distributed lifecycle proof remains open. |

## Validation

- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`
- `python3 -m fornax program local-accelerator-smoke --expert-device cuda:0 --transfer-source-device cuda:0 --transfer-destination-device cuda:1`
- `python3 -m fornax test trust-boundary`
- `python3 -m fornax test state-ownership`
- `python3 -m fornax test stage-replication`
- `python3 -m fornax program local-http-serving-smoke --backend-mode target-fixture --out /tmp/fornax_local_http_serving_target_fixture_smoke_20260622.json`

## Exit Criteria

- Local two-GPU logical-host validation no longer blocks implementation.
- Security, topology, replication, lifecycle, local target-fixture parity, and observability artifacts are
  coherent under simulation/local smoke scope.
- G3 remains open until a real frontier MoE is served across the required
  heterogeneous fleet at predicted throughput.
