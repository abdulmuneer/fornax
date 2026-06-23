# Phase-3 Heterogeneous Frontier Sprint

**Goal:** assemble the heterogeneous serving path using simulation and local
logical-host validation first, then replace simulated hosts with the real
NVIDIA/AMD/Mac fleet when available.

**Duration:** notional W16-W24.
**Milestone:** M5 Heterogeneous frontier serve.
**Gate:** G3 Heterogeneous frontier proxy passed on 2026-06-23 using two local H100 logical hosts; formal NVIDIA/AMD/Mac validation deferred.

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
| S3-1 | Closed for proxy gate: Apple/Mac participation is explicitly deferred by the operator decision; the proxy gate uses two local H100 logical hosts and records Apple validation as follow-up. |
| S3-2 | Closed for proxy gate: simulated transport/topology placement, measured same-host two-H100 activation-transfer evidence, and verified local logical-host topology route exist; real heterogeneous fabric validation is deferred. |
| S3-3 | Closed for proxy gate: localhost mTLS/HTTPS bearer-token/plan-hash smoke passes with local client-certificate node identity; production auth/keying and product mTLS are deferred. |
| S3-4 | Closed for proxy gate: localhost mTLS endpoint smoke validates deterministic 429 backpressure with HTTP `Retry-After`, retry-after recovery, admitted cancellation, admitted timeout, local partition fence, and recovery after the fence; real distributed partition/failure proof is deferred. |
| S3-5 | Closed for proxy gate: stage-replication simulation exists and local proxy evidence preserves correctness; real replicated runtime scaling remains deferred. |
| S3-6 | Closed for proxy gate: planner artifacts expose placement explanations, and the local endpoint artifact records selected route components plus deferred AMD GPU/Apple Silicon hardware explanations. |
| S3-7 | Closed for proxy gate: localhost mTLS/HTTPS lifecycle cleanup, HTTP retry metadata, retry-after recovery, admitted cancellation/timeout cleanup, local partition fencing/recovery, target-fixture parity, measured two-H100 activation transfer, split-pipeline, MoE parity, H100 target-fixture execution, local topology-route linkage, and the 5/5 H100 local serving-runtime bundle exist. |

## Validation

- `python3 -m fornax program simulate-t1 --gpu-count 2 --profile two-gpu-heterogeneous`
- `python3 -m fornax program local-accelerator-smoke --expert-device cuda:0 --transfer-source-device cuda:0 --transfer-destination-device cuda:1`
- `python3 -m fornax test trust-boundary`
- `python3 -m fornax test state-ownership`
- `python3 -m fornax test stage-replication`
- `/mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python -m fornax program local-http-serving-smoke --backend-mode target-fixture --enable-mtls --include-activation-transfer-probe --activation-transfer-backend torch --include-runtime-probes --runtime-probe-backend torch --runtime-probe-tolerance 0.05 --include-target-fixture-execution-probe --target-fixture-execution-backend torch --target-fixture-execution-tolerance 0.05 --out /tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_topology_route_smoke_20260623.json`
- `python3 -m fornax program phase3-proxy-gate --endpoint-artifact /tmp/fornax_local_http_serving_transport_runtime_bundle_mtls_target_fixture_topology_route_smoke_20260623.json --out /tmp/fornax_phase3_g3_two_h100_proxy_gate_20260623.json --date 2026-06-23 --outcome PROCEED --accepted-by operator`
- `python3 -m fornax accelerator target-fixture-probe --backend torch --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --device cuda:0 --out /tmp/fornax_target_fixture_probe_h100_20260622.json`
- `python3 -m fornax program local-serving-smoke --out-dir /tmp/fornax_local_serving_smoke_target_fixture_h100_20260622 --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python --pipeline-source-device cuda:0 --pipeline-destination-device cuda:1 --moe-source-device cuda:0 --moe-expert-device cuda:1 --target-fixture-device cuda:0`

## Exit Criteria

- Local two-GPU logical-host validation is accepted as the current Phase 3/G3 proxy gate of record.
- Security, topology, replication, local failure semantics including HTTP Retry-After metadata, retry-after recovery plus admitted cancel, timeout cleanup, and local partition fencing/recovery, lifecycle, local mTLS/HTTPS target-fixture parity, measured local H100 fixture execution inside both endpoint and serving-runtime artifacts, measured endpoint-level two-H100 activation-transfer/split-pipeline/MoE runtime probes, local logical-host topology-route and deferred hardware explanations, and the integrated local serving-runtime target-fixture bundle are coherent under the accepted two-H100 proxy scope.
- `/tmp/fornax_phase3_g3_two_h100_proxy_gate_20260623.json` records `phase3_proxy_passed=true`, `outcome=PROCEED`, 8/8 proxy checks, and `formal_g3_validation_deferred=true` for real frontier target-model, AMD GPU, Apple Silicon Mac, product auth/mTLS keying, and distributed partition proof follow-up.
