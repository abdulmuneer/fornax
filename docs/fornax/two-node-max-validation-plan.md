# Two-Node MAX Validation Plan

Version: 1.0  
Plan: `project-plan-v4.md` Phase 0.5  
Status: Engine v0/T1 complete; physical `MaxStageBackend` and two-node execution pending

## Objective

First prove that one deterministic request executes through two independent
worker processes using the production Fornax Stage ABI and reference/simulated MAX
backends. Then replay the same plan and corpus across Linux/NVIDIA and macOS/Apple
MAX backends as hardware becomes available.

## Test environment

Engine v0 runs name an approved simulation scenario and assumption set from
`simulation-and-assumption-contract.md`. Physical runs must close all physical
fleet/build fields in `v0-target-contract.md`. The test owner stops if model,
plan, scenario/build, or tensor-contract identity differs from the approved run.

## Evidence ladder

| Step | Tier | Test | Pass condition |
|---|---|---|---|
| V1 | T0 | Planner defect regressions | Remote expert memory and >6-node cases pass |
| V2 | T0/T1 | ABI and malformed-frame corpus | All valid/negative cases produce specified outcomes |
| V3 | T1 | Reference stage backend | Exact deterministic activation/logit and ownership ledger |
| V4 | T1 | Simulated MAX stage backend with service/failure injection | Same logical outputs; scenario timings/faults attributed |
| V5 | T1 | Two worker processes over TCP loopback | Production Stage ABI, credits, cancellation, and cleanup pass |
| V6 | T2 | NVIDIA and Apple single-node tests as available | Operator/stage activations and logits within accepted tolerance |
| V7 | T3 physical | Linux stage -> Apple stage | Prefill/decode and final greedy output pass |
| V8 | T3 load | Concurrency 1, 4, 8 | Correctness retained; timings and utilization recorded |
| V9 | T3 stability | Thirty-minute highest-supported load | No unbounded queue/memory growth or silent divergence |
| V10 | T3 failure | Cancel, timeout, stale plan, CRC, link loss | Specified bounded outcomes and cleanup |

## Correctness corpus

- At least 20 deterministic prompts spanning short prose, code, numbers, Arabic,
  long repeated context, and boundary tokenization cases.
- Diagnostic contexts 16, 128, and 512 tokens.
- Contract context point 4096 tokens.
- 128 generated tokens for accepted correctness cases.
- Router capture for selected MoE layers and token positions.

Prompt contents may remain outside the public repository, but the corpus hash,
token IDs, tokenizer/template hashes, and licensing/privacy classification are
recorded.

## Apple kernel matrix

| Operation | Minimum cases |
|---|---|
| MLA prefill | Short/ragged prompts, cache page boundary, all contract dtypes |
| MLA decode | Context 16/512/4096, repeated decode, numerical reference |
| MoE indices | 64 experts, top-k 6, empty/skewed buckets, deterministic expert IDs |
| Gather | Rank-2 axis-0, int32/uint32 indices, duplicate/boundary indices |
| Expert MLP | Target dimensions, route counts 1/4/8+, parity and timing |
| Complete stage | Candidate layer range, prefill/decode, KV epoch transitions |

Finite-output-only checks do not pass this matrix.

## Performance collection

For every concurrency point capture median and p95:

- gateway/orchestrator queue;
- Stage 0 MAX execution;
- pack/staging;
- TCP payload transfer;
- receive/unpack;
- Stage 1 MAX execution;
- final logits/sample;
- end-to-end TTFT and inter-token latency;
- queue depth, credit, bytes, and memory high-water marks.

Collect a single-node reference and naive two-stage baseline using the same model,
prompt corpus, build lineage, and stage cut.

## Fault cases

- Wrong ABI major/minor.
- Wrong build/plan/manifest/stage.
- Truncated and CRC-invalid activation.
- Duplicate same payload and conflicting duplicate.
- No credit / full receiver queue.
- Cancellation while queued and during execution.
- Deadline expiration.
- Kill/restart worker and disconnect cable/interface where safe.
- Reconnect with fresh handshake; no duplicate stage execution.

## Engine v0 acceptance report

The Phase 0.5 packet contains:

1. code/environment and simulation-scenario manifest;
2. exact commands, plan/stage manifests, assumption IDs, and seeds;
3. correctness summary plus raw comparison artifacts;
4. simulated performance attribution and planner/scenario comparison;
5. failure/cleanup results;
6. unresolved failures and scope limitations;
7. mapping from every open hardware assumption to V6/V7 validation;
8. Engine v0 completion recommendation.

The later physical packet adds measured performance, numerical parity, Apple role,
and the recommended G2 outcome.

Evidence is indexed in `evidence-register.md`; `/tmp` paths alone are not durable
evidence.
