# Fornax documentation

Start here for the runnable Fornax workflow.

## Current scope

The repository currently ships the planner plus the Phase 0.5 Engine v0
simulation/contract layer. This code is pure Python and needs no GPU or model
download; the Engine v0 tests use local loopback sockets and spawned processes.
You can describe a model and a fleet, ask Fornax to place the model across the
fleet, and inspect the predicted throughput, latency, and feasibility result.

The physical Mojo/MAX backend that turns the same Stage ABI into served tokens
across heterogeneous machines is still under development. Guide pages label
simulator output as predictions and hardware runs as measurements.

Plan v4 Phase 0.5 is complete at T0/T1 against reference/simulated MAX stage
backends using the production Stage ABI and multi-process loopback TCP. Physical
evidence remains required before any supported-platform, throughput, or G2 claim.

## Guides

| Guide | Use it for |
|---|---|
| [Getting started](getting-started.md) | Install from a clone, verify the repo, run the first plan and simulation, and optionally smoke DeepSeek-V2-Lite with source-built MAX on Apple Silicon. |
| [Concepts](concepts.md) | Learn the model, fleet, stage, expert, prediction, and runtime vocabulary used by the other docs. |
| [Planning and simulation](planning-and-simulation.md) | Work through the full planning flow: target contract, inventory, plan, simulation, validation, and preflight bundle. |
| [Input file reference](input-formats.md) | Look up the JSON fields for target contracts, inventories, and links files. |
| [CLI reference](cli-reference.md) | Find the `python3 -m fornax` command surface grouped by task. |
| [Project plan v4](fornax/project-plan-v4.md) | Current architecture, assumption-driven execution scope, and gates. |
| [Simulation contract](fornax/simulation-and-assumption-contract.md) | Named hardware assumptions, sensitivity scenarios, and replacement rules. |

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
