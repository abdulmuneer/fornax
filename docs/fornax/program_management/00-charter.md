# Program Charter — Fornax

## Purpose

Deliver **Fornax**: a Mojo/MAX-native distributed MoE inference engine that serves
a single frontier-scale MoE model across heterogeneous commodity hardware
(consumer NVIDIA/AMD + Apple Silicon, provisioned LAN) at high *aggregate*
throughput, on-prem, for in-house private AI. Engineering definition:
[`../project-plan-v4.md`](../project-plan-v4.md).

> **Authorization status (plan v4 §8–§9):** G1 `PROCEED` authorizes Engine v0
> implementation under explicit assumptions. Physical heterogeneous validation
> remains open and gates G2/G3 claims.

## Why now / why this program

- Frontier open weights include sparse MoE models that can exceed one available
  node; the customer cost/privacy advantage over cloud or homogeneous systems is
  a hypothesis to prove with dated baselines and TCO, not an assumed fact.
- MoE's capacity-vs-compute split may match heterogeneous commodity hardware;
  physical G2/G3 evidence and a source-backed competitive study decide whether
  Fornax's cross-vendor approach is useful and differentiated.
- A credible path needs **evidence before investment** — hence a gated program,
  not an open-ended build.

## Objectives (program-level)

| # | Objective | Measured by |
|---|---|---|
| O1 | Build the first executable engine without waiting for the complete hardware fleet | Phase 0.5 Engine v0 exits at T0/T1 on versioned experimental FNX1 v1 |
| O2 | Serve a model that exceeds the usable memory of the largest single target node | Capacity metric (§10 plan) |
| O3 | Preserve **aggregate throughput** at the contracted concurrency | Throughput-efficiency ≥ provisional 60% |
| O4 | Keep the engine **honest & correct** across vendors | Reference-path logit match; no fabricated metrics |
| O5 | Manage the **Apple/MAX external dependency** without it sinking the program | Simulated Apple-shaped worker now; physical role measured later |

## In scope

The Phase 0–5 roadmap in plan v4 §8, organized into the workstreams in
[02-work-breakdown-structure.md](02-work-breakdown-structure.md).

## Out of scope (program guardrails)

- Datacenter-throughput / single-stream-latency competition with vLLM/SGLang.
- Single-user / low-concurrency as a primary target (plan §3.3).
- Training/fine-tuning; WAN federation; models that fit one node.
- **Forking MAX wholesale** — surgery stays at named seams (plan §5.5).

## Success / failure definition

- **Near-term success:** Engine v0 runs two independent loopback workers through
  experimental FNX1 v1 under a lockstep orchestrator; admission/batching bounds
  are separately simulated with transport, failure, and evidence behavior.
- **Program success:** G2 replaces critical assumptions with physical evidence,
  then G3 serves a real frontier-capacity MoE at contracted throughput.
- **Acceptable failure:** G2/G3 evidence shows that a physical role or the full
  thesis does not close, and the program **narrows deliberately** (for example,
  capacity-only Apple participation or homogeneous islands) while preserving the
  reusable engine contracts. A physical no-go is a valid program outcome, not a
  failure of execution.
- **Unacceptable:** reporting simulation as measured hardware, silently changing
  assumptions, or letting unavailable hardware stall contract-valid engine work.

## Sponsor & authority

- **Sponsor / decision authority:** project owner (Abdul Muneer) — holds go/no-go
  at all gates.
- **Program management:** owns gates, RAID, cadence, decision log.
- **Technical authority:** lead engineer (architecture conformance to plan v4).

> Roles are by **function**, not headcount — at current size one person may hold
> several. See [07-resourcing-and-skills.md](07-resourcing-and-skills.md).

## Guardrails the PM enforces

1. Work stays inside the G1-authorized Engine v0 scope until G2 physical evidence.
2. Apple Silicon stays at its **gated role** (plan §5.5) until its profiler gate
   passes; the **reversal trigger** is honored.
3. Every irreversible/expensive decision gets a **DEC-\*** record
   ([08-decision-log.md](08-decision-log.md)).
4. The plan changes only by **version bump**, never silent edit.
5. Every simulated hardware parameter cites an SA-* assumption and named scenario.
6. Code, tests, or user-supplied intent never establish named-person
   participation, IP ownership, commercial rights, financing approval, customer
   traction, or market size; each follows its own evidence gate.
