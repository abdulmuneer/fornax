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
| S3-2 | Partial: simulated transport/topology placement and measured same-host two-H100 activation-transfer evidence inside the endpoint artifact exist; real heterogeneous fabric remains open. |
| S3-3 | Partial: trust-boundary simulation exists and localhost mTLS/HTTPS bearer-token/plan-hash smoke now passes with local client-certificate node identity; production auth/keying and product mTLS remain open. |
| S3-4 | Partial: backpressure and replay simulations exist, and localhost mTLS endpoint smoke now validates deterministic 429 backpressure, retry-after recovery after capacity clears, and admitted cancellation/admitted timeout cleanup plus a local partition fence before backend execution; real distributed partition/failure proof remains open. |
| S3-5 | Partial: stage-replication simulation exists; real replicated runtime remains open. |
| S3-6 | Done at planner artifact scope; live runtime linkage remains future work. |
| S3-7 | Partial: state-ownership simulation exists; localhost mTLS/HTTPS lifecycle cleanup, retry-after recovery, admitted cancellation cleanup, admitted timeout cleanup, local partition fencing, local target-fixture parity, measured two-H100 activation transfer, split-pipeline, and MoE parity inside the endpoint artifact, measured H100 target-fixture execution inside the endpoint artifact, and a 5/5 H100 local serving-runtime bundle with integrated target-fixture execution evidence exist; live distributed lifecycle proof remains open. |

## Validation

- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`
- `python3 -m fornax program local-accelerator-smoke --expert-device cuda:0 --transfer-source-device cuda:0 --transfer-destination-device cuda:1`
- `python3 -m fornax test trust-boundary`
- `python3 -m fornax test state-ownership`
- `python3 -m fornax test stage-replication`
- `python3 -m fornax program local-http-serving-smoke --backend-mode target-fixture --enable-mtls --include-activation-transfer-probe --activation-transfer-backend torch --include-runtime-probes --runtime-probe-backend torch --include-target-fixture-execution-probe --target-fixture-execution-backend torch --out /tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_smoke_20260622.json`
- `python3 -m fornax accelerator target-fixture-probe --backend torch --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --device cuda:0 --out /tmp/fornax_target_fixture_probe_h100_20260622.json`
- `python3 -m fornax program local-serving-smoke --out-dir /tmp/fornax_local_serving_smoke_target_fixture_h100_20260622 --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --pipeline-source-device cuda:0 --pipeline-destination-device cuda:1 --moe-source-device cuda:0 --moe-expert-device cuda:1 --target-fixture-device cuda:0`

## Exit Criteria

- Local two-GPU logical-host validation no longer blocks implementation.
- Security, topology, replication, local failure semantics including retry-after recovery plus admitted cancel, timeout cleanup, and local partition fencing, lifecycle, local mTLS/HTTPS target-fixture parity, measured local H100 fixture execution inside both endpoint and serving-runtime artifacts, measured endpoint-level two-H100 activation-transfer/split-pipeline/MoE runtime probes, and the integrated local serving-runtime target-fixture bundle are
  coherent under simulation/local smoke scope.
- G3 remains open until a real frontier MoE is served across the required
  heterogeneous fleet at predicted throughput.
