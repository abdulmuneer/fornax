# Concepts

The vocabulary and the mental model the rest of the docs assume. Read this once
and the CLI and file formats will make sense.

## What Fornax is

Fornax is a **distributed inference engine for frontier-scale sparse-MoE models**
that runs across a fleet of **heterogeneous, commodity machines** — consumer
NVIDIA GPUs, Apple-Silicon Macs, whatever else is on the LAN — so that a fleet
can serve **one model that no single node could hold**, on-prem, at high
*aggregate* throughput.

It is an **engine**, not a load balancer. It does not fan requests out across
copies of `max serve`; it cuts into the model-execution path itself and spreads
*one* model's layers and experts across machines.

## The thesis (why this can work)

> Frontier open models are large **sparse Mixture-of-Experts**: expensive to
> *store* (you must hold all the expert weights) but cheap to *compute* per token
> (only a few experts fire). Heterogeneous commodity hardware is exactly that
> mismatch in reverse — Macs bring **cheap capacity** (large unified memory),
> consumer GPUs bring **cheap compute**. The shapes match. The bottleneck is the
> **interconnect**, and Fornax's whole job is to hide it.

## The honest constraint

**Read this before you trust any number Fornax prints.**

When a model is bigger than your biggest node, **every token crosses the
network.** A spanned model has a pipeline-and-synchronization floor a single-node
model does not. Fornax optimizes **aggregate throughput and utilization** — high
total tok/s, high \$/token efficiency, via continuous batching, overlap, and
expert locality. It does **not** promise single-stream latency parity with a
model that fits on one box. That latency cost is the irreducible price of
spanning, and the docs and the tool both say so rather than hiding it.

A second honesty rule runs through the codebase: **every number is either a
measurement or a prediction, and is always labelled as which.** Today's
`simulate` output is a **cost-model prediction** for a placement — not a
benchmark of your hardware. The *gate validators* (see below) exist to keep those
predictions traceable to an explicit model or fixture and to stop fabricated
metrics from leaking in.

## The core objects

You describe three things; Fornax produces a fourth.

| Object | What it is | You provide it as |
|---|---|---|
| **Model** | The network to serve: hidden dim, layer list (attention / dense / MoE), per-layer weight bytes, FLOPs, and KV bytes. | the `model` block of a *target contract* |
| **Target** | The serving goal: concurrency, prompt/generation length, and the objective (`max_throughput`, `min_latency`, `balanced`). | the `target` block of a *target contract* |
| **Inventory** | The fleet: a list of *nodes* (machines) and *links* (the network between them). | an *inventory* JSON file |
| **Plan** | Fornax's output: how the model is placed across the fleet, plus a feasibility verdict and a predicted performance profile. | produced by `plan` |

See [Input file reference](input-formats.md) for every field.

## Stages, replicas, and experts

A **plan** divides the model into **stages** — contiguous groups of layers — laid
out as a pipeline. Each stage is hosted on one or more **replicas** (nodes), in a
**mode**:

- **`resident`** — the stage's weights live in the node's memory for the whole
  session (the fast path; used whenever the layers fit).
- other modes cover cases where weights must be streamed or shared rather than
  held resident.

For **MoE layers**, the dense path (attention, routers, hot/shared experts,
sampler) stays on the fastest local accelerator group, while **routed experts**
can be placed on other workers — the `expert_placement` part of a plan. This is
the surgical seam the engine is built around: the router splits a token's experts
into local batches and remote batches, the remote ones execute on capacity-rich
workers (e.g. Macs), and the results are gathered back.

## Feasibility is an outcome, not an error

`plan` answers "does this model fit this fleet?" as a first-class result.
A **feasible** plan has stages, an `expert_placement`, and a `predicted` profile.
An **infeasible** plan has `"feasible": false` and an `infeasible_reason` (e.g.
the model exceeds total memory). The CLI mirrors this in its exit code — `0` for
feasible, `2` for infeasible — so you can gate on it in scripts.

## The prediction: throughput, latency, bubble

`simulate` reads a feasible plan's `predicted` block and reports three numbers:

- **throughput (tok/s)** — predicted aggregate generation rate for the placement.
- **per-request latency (s)** — predicted time for one request, including the
  network-crossing floor described above.
- **bubble fraction** — the share of time pipeline stages sit idle waiting on
  others (pipeline imbalance / sync overhead); lower is better. A perfectly
  balanced single-stage plan has bubble `0`.

All three come from the planner's cost model. They tell you whether a placement
is *worth building and benchmarking*, not what a benchmark will say.

## Golden vectors and gates (why you can trust the logic)

Fornax's behavior is pinned by **golden vectors** — recorded expected outputs for
the planner, the runtime-format contracts, the engine seam, and the phase gates
— under `fornax/golden_vectors/**` and `fornax/golden_plans/**`. The `test`
command replays them; `make test` runs the lot. Changing planner or contract
behavior means regenerating the affected vector *and* recording why. This is the
mechanism that keeps the simulation honest as the code evolves, and it is why you
can run `make test` and trust a green result.

## Where the runtime is

The planner and the simulation/contract layer ship today and are self-tested. The
**heterogeneous Mojo/MAX expert runtime** — the part that takes a plan and serves
real tokens across real GPUs and Macs — is the active build, and it is being
built *against these contracts*. The engine exposes a stable, harness-agnostic
**string-in / string-out** generation seam so any control plane can drive it once
it lands. Until then, treat Fornax as the tool that tells you *what to build and
whether it will fit*.
