# Fornax — heterogeneous frontier-model serving

**Fornax** is building toward a **Mojo/MAX-native distributed inference engine**
— a proposed *custom surgery of MAX* — intended to make a qualified fleet of
heterogeneous commodity machines serve one sparse-MoE model that no individual
node can hold. The current repository is a planner plus reference/simulated
mechanism layer; physical cross-vendor serving is an open G2 proof.

The product boundary is **engine, not harness**. The target design composes MAX
graph compilation, kernels, KV primitives, and custom ops with Fornax-owned
cross-node planning, orchestration, and transport. The public Python `Engine`
currently wraps only an explicitly supplied `str -> str` generator; no bundled
physical generator or Ignis integration is implemented. See
[project-plan-v4.md](project-plan-v4.md) for the target boundary.

## The one-line thesis

> Frontier open models are large **sparse MoE** (capacity-bound to store,
> compute-light per token). Heterogeneous commodity hardware is **cheap capacity
> (Mac unified memory) + cheap compute (consumer GPUs)**. The shapes match — the
> bottleneck is the **interconnect**. The product hypothesis is that MAX kernels
> plus a Fornax-owned heterogeneous pipeline can manage that constraint.

## What it is not

Fornax is **not** a wrapper that load-balances `max serve`, and it is not Ignis
with a cluster backend. It cuts into the model execution path. The surgical seam
is the MoE block:

```text
hidden states -> router -> local expert batches + remote expert batches
              -> weighted gather -> next layer
```

In the target design, the dense path stays on the fastest qualified accelerator
group whenever possible. Pipeline stages are the baseline cross-node spine;
routed-expert batches are deferred and unimplemented.

## The honest constraint (read first)

When the model exceeds the biggest node, **every token crosses the network**.
Even when the network is provisioned for this workload, a spanned model has a
pipeline and synchronization floor that a single-node model does not. Fornax
targets aggregate throughput and utilization via future integrated batching,
overlap, expert locality, and balanced stages; those physical benefits are not
yet proven. It does *not* target single-stream latency parity. That latency cost is the
irreducible price of spanning. See
[project-plan-v4.md](project-plan-v4.md#2-product-hypothesis-and-constraints).

## Documents

| Doc | What it covers |
|---|---|
| [project-plan-v4.md](project-plan-v4.md) | Current plan of record; its “production ABI” target is implemented today only as experimental FNX1 v1 at T0/T1 |
| [simulation-and-assumption-contract.md](simulation-and-assumption-contract.md) | Named hardware assumptions, scenario matrix, simulator/backend contract, and physical replacement rules |
| [stage-runtime-and-wire-abi.md](stage-runtime-and-wire-abi.md) | Versioned experimental StageExecutable interface and framed TCP tensor protocol |
| [stage-abi-v2-ragged-design.md](stage-abi-v2-ragged-design.md) | Implemented candidate FNX2 T0/T1 contract, exact-wire golden, slow oracle, integrated scheduler, and two-worker ragged loopback; physical MAX conformance remains open |
| [abi-terminology-erratum-2026-07-17.md](abi-terminology-erratum-2026-07-17.md) | Clarifies that plan v4's “production ABI” phrase names a target role; current FNX1 is experimental T0/T1 |
| [planner-status-erratum-2026-07-17.md](planner-status-erratum-2026-07-17.md) | Records the implemented exploratory/deployment authority split, provenance/confidence/error fields, exact capability admission, and the remaining physical-calibration boundary |
| [founder-proxy-review-follow-up-2026-07-17.md](founder-proxy-review-follow-up-2026-07-17.md) | Dated proxy-lens disposition of the 2026-07-13 platform/DX review; records actions closed, partial, and still blocking without implying named-person participation or endorsement |
| [g2-in-a-box.md](g2-in-a-box.md) | One-command, fail-closed G2 readiness/physical-validation runner; exact lineage/model/device/command capture and durable bundle contract |
| [consumer-hardware-recipes.md](consumer-hardware-recipes.md) | Dated three-model by six-platform C1 qualification cohort, content-addressed operator packets, exact identity/artifact checks, and the C2–C5 promotion path |
| [max-fork-build-reproducibility.md](max-fork-build-reproducibility.md) | Tracked MAX lineage pin, clean-build procedure, and cross-node compatibility requirements |
| [partitioner-spec.md](partitioner-spec.md) | Throughput-optimizing heterogeneous MoE partitioner: stage/expert placement, cost model, search, and contract/golden-vector expectations |
| [deepseek-v2-lite-max-check.md](deepseek-v2-lite-max-check.md) | 2026-07-01 source-built MAX evidence for `deepseek-ai/DeepSeek-V2-Lite-Chat` on Apple Silicon M3 Max, including MLA/MoE/gather blockers cleared and remaining caveats |
| [max-operator-platform-support.md](max-operator-platform-support.md) | Dated MAX operator/platform support map, now annotated with the local Apple DeepSeek MLA smoke result |
| [apple-flare-mla-prefill-port.md](apple-flare-mla-prefill-port.md) | Apple `flare_mla_prefill` porting note and status update: planned design became a local source-built MAX patch with smoke evidence |
| [apple-silicon-max-skills.md](apple-silicon-max-skills.md) | Skill map for MAX/Mojo work on Apple Silicon: platform setup, kernels, custom ops, MoE expert runtime, validation, profiling, transport |
| [program_management/](program_management/) | Charter, roadmap, gates, external watch, RAID log, decision log, sprint plans, and evidence governance |

## Status

The root `fornax/` package is the Python planner plus the completed Phase 0.5
Engine v0 simulation/contract/smoke layer. Experimental FNX1 v1 and its
two-process lockstep loopback remain the historical T0/T1 baseline. Candidate
FNX2 now adds an integrated bounded ragged scheduler, unequal prefill,
changing-subset decode, per-sequence results, lease/replay semantics, and two
independent loopback workers at T0/T1. The reference/simulated lifecycle also
has opportunistic expiry, internal leases, same-worker reconnect tombstones,
and a copy-explicit bounded buffer seam. Restart-durable fencing, a current
1,800-second evidence artifact, and physical native-KV/buffer validation remain
open, so the engine is not evidenced as memory-bounded for indefinite service.
The local `external/modular` checkout carries MAX backend work
for Apple Silicon. As of 2026-07-01, the local source-built MAX tree can run a
short `DeepSeek-V2-Lite-Chat` smoke on an M3 Max after adding Apple MLA prefill,
Apple MLA decode fallback, Apple MoE index fallback, and Apple rank-2 gather
support. That evidence is recorded in
[deepseek-v2-lite-max-check.md](deepseek-v2-lite-max-check.md).

This does **not** close G2 or imply production Apple serving. DEC-008 closes only
the recorded Engine v0 T0/T1 scope. The Apple run is rank-1 local probe
evidence for a single Mac, source-built MAX commit, model,
dtype, prompt size, and short generation count. Remaining work includes
numerical parity, optimized decode/MoE paths, serving-mode validation, longer
contexts, batching, memory-pressure checks, and the target expert-MLP probe that
decides Apple's v0 Fornax role.

The current working tree contains an uncommitted MAX root-pin/reconstruction
mechanism in
[`dependencies/max-lineage.json`](../../dependencies/max-lineage.json). Run
`python3 -m fornax program g2-validate --out-dir <new-directory>` to verify the
pin and current T0/T1 prerequisites; without a physical run manifest it reports
V6-V10 as blocked and does not close G2. The mechanism becomes durable
repository lineage only after commit.
