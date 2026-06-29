# Fornax documentation

Start here for the runnable Fornax workflow.

## Current scope

The repository currently ships the planner and the simulation/contract layer.
This code is pure Python and needs no GPU, model download, or network access.
You can describe a model and a fleet, ask Fornax to place the model across the
fleet, and inspect the predicted throughput, latency, and feasibility result.

The heterogeneous Mojo/MAX runtime that turns a plan into served tokens across
real machines is still under development. Guide pages label simulator output as
predictions and hardware runs as measurements.

## Guides

| Guide | Use it for |
|---|---|
| [Getting started](getting-started.md) | Install from a clone, verify the repo, and run the first plan and simulation. |
| [Concepts](concepts.md) | Learn the model, fleet, stage, expert, prediction, and runtime vocabulary used by the other docs. |
| [Planning and simulation](planning-and-simulation.md) | Work through the full planning flow: target contract, inventory, plan, simulation, validation, and preflight bundle. |
| [Input file reference](input-formats.md) | Look up the JSON fields for target contracts, inventories, and links files. |
| [CLI reference](cli-reference.md) | Find the `python3 -m fornax` command surface grouped by task. |

## Quick check

```bash
make test
python3 -m fornax plan \
    --target my_target.md \
    --inventory my_fleet.json \
    --out plan.json
python3 -m fornax simulate --plan plan.json
```

`plan` exits `0` when the placement is feasible and writes `plan.json`. It exits
`2` when the model cannot be placed; the output file records the reason.

`simulate` reports cost-model predictions for the placement. Treat those numbers
as planning inputs until a benchmark or serving smoke produces measured runtime
data.
