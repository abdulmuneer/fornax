# Concepts

This page defines the vocabulary used by the CLI and file-format docs.

## What Fornax is

Fornax is a pre-alpha program for a distributed sparse-MoE inference engine. The
target runtime would place one model across a qualified heterogeneous commodity
fleet so the fleet can serve a model larger than any single node can hold.

The target fleet may include consumer NVIDIA GPUs, Apple Silicon Macs, AMD
devices, and CPU workers. Today the planner models layers/stages/experts, while
physical cross-vendor stage execution and remote experts remain unimplemented.

## Why sparse MoE matters

Sparse Mixture-of-Experts models require memory for all expert weights, while
each token activates only a small subset of experts. This makes capacity the
first constraint and per-token compute the second constraint. A heterogeneous
fleet may match that shape if measurements show that the planner can keep dense
work close to fast accelerators. Remote expert work is a future measured option.

The network remains the main constraint. Every plan should be read with that in
mind.

## Prediction and measurement scope

When a model is larger than the biggest node, every token crosses the network.
That adds pipeline latency and synchronization cost. Fornax targets aggregate
throughput through future integrated batching/overlap and balanced stages; the
candidate FNX2 path now integrates ragged scheduling with two reference
loopback workers at T1, while overlap and physical MAX batching remain
unimplemented. Single-stream latency includes the cost of spanning the model.

Fornax labels simulator output as predictions. `simulate` reports cost-model
estimates for a placement. A benchmark or serving smoke is a measurement only
for the exact hardware, model, build, command, and artifact it records; it does
not make the planner input calibrated automatically.

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

For MoE layers, the planner can emit an `expert_placement` block. That is a
placement/simulation object today, not an integrated remote-expert runtime. The
future design would build local/remote expert batches and gather results before
the next layer only after physical correctness and performance justify it.

## Feasibility

`plan` answers whether the model is feasible under the supplied numeric model,
inventory, and the capability checks currently implemented. It does not verify
that those numbers were measured or that the runtime/build supports every
operation.

A modeled-feasible plan has stages, expert placement, and a `predicted` profile. An
infeasible plan has `"feasible": false` and an `infeasible_reason`, such as total
model memory exceeding total available fleet memory.

The CLI uses exit `0` for modeled feasibility and exit `2` for modeled
infeasibility so scripts can gate this calculation. Neither exit code grants
deployment authority; current plans remain exploratory under I-16.

## Predicted throughput, latency, and bubble

`simulate` reads a feasible plan and reports three values:

- Throughput in tokens per second: predicted aggregate generation rate for the
  placement.
- Per-request latency in seconds: predicted time for one request, including the
  network-crossing cost.
- Bubble fraction: predicted share of time that pipeline stages wait on other
  stages. Lower is better. A balanced single-stage plan has bubble `0`.

Use these predictions to compare hypotheses and choose what to benchmark, not to
claim hardware performance.

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
available, use Fornax to explore modeled fit, compare placement hypotheses, and
prepare evidence for the build that follows.
