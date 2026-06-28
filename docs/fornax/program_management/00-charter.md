# Program Charter — Fornax

## Purpose

Deliver **Fornax**: a Mojo/MAX-native distributed MoE inference engine that serves
a single frontier-scale MoE model across heterogeneous commodity hardware
(consumer NVIDIA/AMD + Apple Silicon, provisioned LAN) at high *aggregate*
throughput, on-prem, for in-house private AI. Engineering definition:
[`../project-plan-v3.md`](../project-plan-v3.md).

> **Authorization status (plan v3 §0, §12):** v3 is approved as a **Phase-0
> evidence-sprint plan only** — *not* authorization for Phase-1 engineering. The
> program's job right now is to reach G1 with evidence, not to build the runtime.

## Why now / why this program

- Frontier open weights are large sparse MoE that exceed a single consumer node;
  the 8×H100 alternative (~$250–400K) defeats the privacy/cost motive.
- MoE's capacity-vs-compute split matches heterogeneous commodity hardware; no
  existing system spans one model across vendors at good throughput.
- A credible path needs **evidence before investment** — hence a gated program,
  not an open-ended build.

## Objectives (program-level)

| # | Objective | Measured by |
|---|---|---|
| O1 | Prove the thesis is feasible before Phase-1 spend | G1 evidence gate clears (contract closes) |
| O2 | Serve a model **2–3× the largest single node** | Capacity metric (§8 plan) |
| O3 | Preserve **aggregate throughput** at the contracted concurrency | Throughput-efficiency ≥ provisional 60% |
| O4 | Keep the engine **honest & correct** across vendors | Reference-path logit match; no fabricated metrics |
| O5 | Manage the **Apple/MAX external dependency** without it sinking the program | Reversal trigger respected; Plan B ready |

## In scope

The Phase 0–5 roadmap in plan v3 §6, organized into the workstreams in
[02-work-breakdown-structure.md](02-work-breakdown-structure.md).

## Out of scope (program guardrails)

- Datacenter-throughput / single-stream-latency competition with vLLM/SGLang.
- Single-user / low-concurrency as a primary target (plan §3.3).
- Training/fine-tuning; WAN federation; models that fit one node.
- **Forking MAX wholesale** — surgery stays at named seams (plan §5.5).

## Success / failure definition

- **Success:** G1 clears with a closing contract, then the program reaches G3
  (serve a real frontier MoE on heterogeneous hardware at predicted throughput).
- **Acceptable failure:** G1 produces evidence that the full thesis does not
  close, and the program **narrows deliberately** (e.g. capacity-only, or
  homogeneous-island) — capturing the learning. A no-go at G1 is a *valid program
  outcome*, not a failure of execution.
- **Unacceptable:** drifting into Phase-1+ engineering spend without G1 evidence.

## Sponsor & authority

- **Sponsor / decision authority:** project owner (Abdul Muneer) — holds go/no-go
  at all gates.
- **Program management:** owns gates, RAID, cadence, decision log.
- **Technical authority:** lead engineer (architecture conformance to plan v3).

> Roles are by **function**, not headcount — at current size one person may hold
> several. See [07-resourcing-and-skills.md](07-resourcing-and-skills.md).

## Guardrails the PM enforces

1. No Phase-1 work before **G1**.
2. Apple Silicon stays at its **gated role** (plan §5.5) until its profiler gate
   passes; the **reversal trigger** is honored.
3. Every irreversible/expensive decision gets a **DEC-\*** record
   ([08-decision-log.md](08-decision-log.md)).
4. The plan changes only by **version bump**, never silent edit.
