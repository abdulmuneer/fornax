# CLI reference

Everything Fornax exposes through `python3 -m fornax <command>`, grouped by what
it's for. Every command supports `-h/--help`; run that for the exact, current
flags. Commands that produce a feasibility/validation verdict exit `0` on
pass/feasible and `2` on fail/infeasible.

```bash
python3 -m fornax --help            # top-level command list
python3 -m fornax <command> --help  # flags for one command
```

> Most commands below the core workflow are **simulation / contract**
> subcommands: they exercise and validate the contracts the eventual
> heterogeneous runtime must satisfy, against golden vectors — no GPU, model, or
> network required. They are how the project keeps the design honest before the
> runtime exists. The everyday end-user surface is the **core workflow** group.

## Core workflow

The commands you use to take a model + fleet to a placement and a prediction.
Walkthrough: [Planning and simulation](planning-and-simulation.md).

| Command | What it does |
|---|---|
| `plan --target T --inventory INV [--links L] --out P` | Place the model across the fleet; write the plan to `P`. Exit 2 if infeasible. |
| `simulate --plan P [--requests R] [--out O]` | Predict throughput / latency / bubble for a plan; optionally project a request trace's decode wall-time. |
| `target validate TARGET --inventory INV [--links L] [--out O]` | Plan, then check the placement against the target contract's acceptance thresholds. Exit 2 if any check fails. |
| `target draft --source S --inventory INV [--links L] --out O` | Render a target-contract draft from a model/inventory; reports whether it is already valid. |
| `preflight --target T --out-dir D [many opts]` | Run the planning/contract pipeline and write a directory of evidence artifacts. |
| `doctor --bundle DIR [--out O]` | Inspect a phase-0 evidence bundle (e.g. a `preflight` output). |

## Hardware & fleet inspection

Collect or simulate the inventory the planner consumes, and probe accelerators.

| Command | What it does |
|---|---|
| `inventory collect` | Collect the local machine's inventory. |
| `inventory simulate-cluster` | Generate a simulated multi-node cluster inventory. |
| `accelerator {expert-mlp-probe,activation-transfer-probe,target-fixture-probe}` | Accelerator micro-probes (expert MLP, activation transfer, target fixture). |
| `apple {probe-template,simulate-probe,validate-probe,role-decision}` | Apple-Silicon worker probing and role (capacity-only vs expert-worker) decisions. |
| `fabric probe` | Probe a network link (bandwidth/latency) for the inventory's `links`. |
| `calibrate local` | Calibrate the cost model against the local machine. |

## Benchmarking

| Command | What it does |
|---|---|
| `benchmark --plan P [opts]` | Run a benchmark for a plan and record a ledger entry (hardware, OS, driver, model, concurrency, quantization, …). |
| `throughput scaling-simulate` | Simulate throughput scaling behavior. |

## Runtime-component simulations & contracts

These model and validate the pieces of the distributed engine. Each has a matching
golden vector under `fornax/golden_vectors/**` and a `test` target (below).

| Command | Component it models |
|---|---|
| `engine simulate` | the engine's request/queue/microbatch contract. |
| `serving {adapter-simulate,state-ownership-simulate}` | the serving adapter and state ownership. |
| `runtime stage-host-simulate` | a pipeline stage host. |
| `workers simulate` | worker lifecycle/contract. |
| `transport {simulate,trust-boundary-simulate}` | cross-vendor activation/KV transport and its trust boundary. |
| `replication simulate` | stage replication. |
| `resilience replay-simulate` | failure/replay resilience. |
| `scheduler simulate` | request scheduling. |
| `moe {simulate,migration-simulate,remote-expert-probe,parity-probe}` | the MoE expert runtime: routing, migration, remote-expert batches, and numeric parity. |
| `model-support simulate` | model-support coverage. |
| `batching simulate` | continuous batching. |
| `pipeline correctness-probe` | pipeline correctness. |
| `observability {metrics-simulate,trace-simulate}` | metrics and trace ledgers. |
| `ops {lifecycle-simulate,onboarding-simulate}` | operational lifecycle and onboarding. |

## Specs & program management

| Command | What it does |
|---|---|
| `spec {runtime-format,network-security,model-support,backend-coverage,substrate-adr}` | Emit/validate the named specification. |
| `program {rebaseline,governance-simulate,phase3-proxy-gate,phase4-resilience-gate,phase5-ga-gate,g1-evidence-packet,…}` | Program-governance and phase-gate tooling (roadmap rebaseline, stage gates, G1 evidence). |

## The gate runner: `test`

`test` replays a named golden/contract suite — the mechanism behind `make
golden`. Exit non-zero on any failure.

```bash
python3 -m fornax test golden-plans        # the planner golden plans
python3 -m fornax test engine-seam         # the string-in/out engine seam contract
python3 -m fornax test --help              # the full list of suites
```

Available suites include `golden-plans`, `runtime-format`, `network-contract`,
`engine-seam`, `stage-host`, `serving-adapter`, `moe-runtime`, `moe-parity-probe`,
`continuous-batching`, `scheduler-contract`, `stage-replication`,
`resilience-replay`, `throughput-scaling`, `phase4-resilience-gate`,
`phase5-ga-gate`, and more (see `test --help`). `make golden` runs the core set;
`make test` runs the golden set plus the unit tests.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success / feasible / valid / all gates passed |
| `2` | infeasible plan, failed validation, or a failed gate/usage error |
