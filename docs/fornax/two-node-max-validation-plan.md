# Two-Node MAX Validation Plan

Version: 1.0  
Plan: `project-plan-v4.md` Phase 0.5  
Status: Engine v0/T1 complete; physical `MaxStageBackend` and two-node execution pending

## Objective

The recorded T1 proof sends one deterministic request through two independent
loopback workers using experimental FNX1 v1 and reference/simulated backends
under a lockstep orchestrator. The next objective is to replay a corrected
physical/ragged contract across Linux/NVIDIA and macOS/Apple MAX backends as
hardware becomes available. See the
[`ABI terminology erratum`](abi-terminology-erratum-2026-07-17.md).

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
| V5 | T1 | Two worker processes over TCP loopback | Experimental FNX1 v1, credits, cancellation, and recorded cleanup pass |
| V6 | T2 | NVIDIA and Apple single-node tests as available | Preapproved reference and dtype `atol`/`rtol` pass; no non-finite value; exact top-1/routing; Apple role derived from raw criteria |
| V7 | T3 physical | Linux stage -> Apple stage | Full correctness corpus, prefill/decode, boundary/final-logit parity, and greedy output pass |
| V8 | T3 load | Concurrency 1, 4, 8 | Correctness corpus passes at every point; predictions match frozen plan input; timings and utilization recorded |
| V9 | T3 stability | Thirty-minute highest-supported load | Integer cadence/load/request/lifecycle evidence, configured memory/queue bounds, post-drain zero state, and no divergence |
| V10 | T3 failure | Cancel, timeout, stale plan, CRC, link loss | Canonical typed outcome plus exact replay, mutation, and cleanup semantics |

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

Before execution, the reviewed manifest freezes the reference implementation
and artifact hash, tolerance approval ID, per-dtype `atol`/`rtol`, rejection of
non-finite values, and exact top-1/routing policy. Each accepted prompt is
compared against that identity; a physical command cannot self-approve a new
reference or tolerance in its result file.

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

The 1/4/8 planner predictions are inputs in the hash-verified plan artifact, not
values first reported by V8. Raw observations must repeat them exactly before
relative error is calculated. V9 uses a predeclared integer sampling interval,
target inflight, minimum completed-request count, and drain timeout. Sample
sequence/time, request started/completed/failed/live counts, explicit/expiry
releases, and all resource counts are integer evidence; the final drain record
must reconcile totals and show zero live request, queue, credit, retained state,
native buffer, KV, and inflight work.

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

V10 records canonical outcome codes (`CANCELLED`, `DEADLINE_EXCEEDED`,
`STALE_PLAN`, `CRC_MISMATCH`, and `LINK_LOSS_RECOVERED`). Each case also records
before/after state hashes, mutation count, replay attempts, replay executions,
replay disposition, and zero-valued cleanup counters. Non-link terminal failures
must not mutate state; link-loss recovery permits exactly one mutation and a
deduplicated replay with no second execution.

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

## One-command runner

The executable form of V1-V10 is documented in
[`g2-in-a-box.md`](g2-in-a-box.md):

```bash
# Readiness only: T0/T1 runs; physical V6-V10 are explicitly BLOCKED.
python3 -m fornax program g2-validate \
  --out-dir evidence/g2-readiness-YYYYMMDD-HHMMSS

# Physical execution requires both reviewed identities/commands and explicit authorization.
python3 -m fornax program g2-validate \
  --out-dir /durable/fornax-evidence/g2-YYYYMMDD-HHMMSS \
  --run-manifest /approved/fornax-g2-run.json \
  --run-physical
```

The runner refuses to overwrite an evidence directory, verifies the current
working tree's uncommitted MAX root-pin/reconstruction mechanism, copies and
recomputes the actual plan/stage-manifest hashes, checks the
model identity, contiguous stage cut, physical node bindings, and
deployment-authoritative plan status, resolves the authority's source IDs
through a copied evidence registry, and rehashes every registry artifact before
admission. It blocks load/stability work behind parity,
validates the step-specific machine result contracts, derives summaries from
nonce-correlated raw samples, requires runner-observed wall time for V9, and
hashes every retained artifact.
This is consistency validation rather than cryptographic remote attestation;
passing the technical packet still requires Sponsor/TL review of the lab wrapper
and artifact custody.
