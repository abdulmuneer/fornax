# Concepts

This page defines the vocabulary used by the CLI and file-format docs.

## What Fornax is

Fornax is a distributed inference engine for frontier-scale sparse-MoE models. It
places one model across a heterogeneous commodity fleet so the fleet can serve a
model larger than any single node can hold.

The fleet can include consumer NVIDIA GPUs, Apple Silicon Macs, AMD devices, and
CPU workers. The engine runs inside the model execution path and places layers,
stages, and routed experts across machines.

## Why sparse MoE matters

Sparse Mixture-of-Experts models require memory for all expert weights, while
each token activates only a small subset of experts. This makes capacity the
first constraint and per-token compute the second constraint. A heterogeneous
fleet can match that shape when the planner keeps dense work close to the fast
accelerators and sends bounded expert work to capacity-rich machines.

The network remains the main constraint. Every plan should be read with that in
mind.

## Prediction and measurement scope

When a model is larger than the biggest node, every token crosses the network.
That adds pipeline latency and synchronization cost. Fornax optimizes aggregate
throughput and utilization through continuous batching, overlap, expert locality,
and balanced stages. Single-stream latency includes the cost of spanning the
model.

Fornax labels simulator output as predictions. `simulate` reports cost-model
estimates for a placement. Benchmarks and serving smokes produce measurements for
the named hardware, model, command, and artifact.

## Core objects

You provide three inputs. Fornax produces a plan.

| Object | Meaning | Provided as |
|---|---|---|
| Model | Network description: hidden dimension, layers, weight bytes, FLOPs, KV bytes, and MoE fields. | `model` block in a target contract |
| Target | Serving goal: concurrency, prompt length, generation length, and objective. | `target` block in a target contract |
| Inventory | Fleet description: nodes and network links. | inventory JSON file |
| Plan | Placement, feasibility verdict, expert placement, and predicted profile. | output from `plan` |

See [Input file reference](input-formats.md) for the full schema.

## Stages, replicas, and experts

A plan divides the model into stages. A stage is a contiguous group of layers
placed as part of a pipeline. A stage can have one or more replicas. Each replica
has a placement mode.

- `resident`: the stage weights live in node memory for the session. This is the
  preferred mode when the layers fit.
- Other modes cover future cases where weights are streamed or shared.

For MoE layers, the dense path includes attention, routers, shared or hot
experts, and the sampler. Routed experts can be placed on other workers through
the plan's `expert_placement` block. The router builds local and remote expert
batches, remote workers execute their assigned experts, and the results are
gathered before the next layer.

## Feasibility

`plan` answers whether the model fits the fleet.

A feasible plan has stages, expert placement, and a `predicted` profile. An
infeasible plan has `"feasible": false` and an `infeasible_reason`, such as total
model memory exceeding total available fleet memory.

The CLI uses exit `0` for feasible and exit `2` for infeasible so scripts can
gate on the result.

## Predicted throughput, latency, and bubble

`simulate` reads a feasible plan and reports three values:

- Throughput in tokens per second: predicted aggregate generation rate for the
  placement.
- Per-request latency in seconds: predicted time for one request, including the
  network-crossing cost.
- Bubble fraction: predicted share of time that pipeline stages wait on other
  stages. Lower is better. A balanced single-stage plan has bubble `0`.

Use these predictions to compare placements and choose what to benchmark.

## Golden vectors and gates

Fornax records expected outputs under `fornax/golden_vectors/**` and
`fornax/golden_plans/**`. The golden files cover planner behavior, runtime-format
contracts, engine-interface contracts, and phase gates.

`python3 -m fornax test ...`, `make golden`, and `make test` replay those
expectations. A planner or contract change that affects behavior should update
the relevant golden vector and document why the expected output changed.

## Runtime status

The planner and the simulation/contract layer ship today. The heterogeneous
Mojo/MAX expert runtime is the active build. That runtime will consume these
contracts to serve real tokens across real GPUs and Macs. Until that runtime is
available, use Fornax to decide whether a model fits a fleet, compare placements,
and prepare evidence for the build that follows.
