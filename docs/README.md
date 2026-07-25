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

Plan v4 Phase 0.5 is complete in its recorded T0/T1 scope against
reference/simulated stage backends: two independent loopback workers use
versioned experimental FNX1 v1 framing under a lockstep orchestrator. Candidate
FNX2 2.0 adds an integrated ragged scheduler, reference oracle, and two-worker
golden path at T0/T1. Physical evidence remains required before any
supported-platform, throughput, or G2 claim. Release, idle expiry, internal
leases, same-worker tombstones, and bounded reference retention are implemented;
restart durability, reviewed long-duration evidence, and physical native-memory
validation remain required before a production-memory claim.

## Guides

| Guide | Use it for |
|---|---|
| [Operator quickstart](onboarding/quickstart.md) | Get a complete two-stage simulated result and inspect its evidence boundary in one command. |
| [Getting started](getting-started.md) | Install from a clone, verify the repo, run the first plan and simulation, and optionally smoke DeepSeek-V2-Lite with source-built MAX on Apple Silicon. |
| [Concepts](concepts.md) | Learn the model, fleet, stage, expert, prediction, and runtime vocabulary used by the other docs. |
| [Planning and simulation](planning-and-simulation.md) | Work through the full planning flow: target contract, inventory, plan, simulation, validation, and preflight bundle. |
| [Input file reference](input-formats.md) | Look up the JSON fields for target contracts, inventories, and links files. |
| [CLI reference](cli-reference.md) | Find the `python3 -m fornax` command surface grouped by task. |
| [Consumer MoE qualification recipes](fornax/consumer-hardware-recipes.md) | Materialize the 18 C1 model/platform bring-up packets and understand exactly what remains unproven before hardware arrives. |
| [Stage Backend adapters](fornax/stage-backend-adapters.md) | Implement an explicit physical worker factory, use the public Python SDK, and run the bounded functional conformance smoke. |
| [Stage ABI v2 ragged contract](fornax/stage-abi-v2-ragged-design.md) | Review the implemented T0/T1 candidate for unequal prefill, independent decode, per-sequence KV/errors, leases, expiry, tombstones, and two-worker loopback execution; physical MAX conformance remains open. |
| [Project plan v4](fornax/project-plan-v4.md) | Current architecture, assumption-driven execution scope, and gates. |
| [Simulation contract](fornax/simulation-and-assumption-contract.md) | Named hardware assumptions, sensitivity scenarios, and replacement rules. |

## Quick check

```bash
python3 -m fornax quickstart
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
