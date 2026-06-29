# CLI reference

Fornax exposes commands through `python3 -m fornax <command>`. Every command
supports `-h` and `--help`. Commands with feasibility or validation verdicts exit
`0` on pass/feasible and `2` on fail/infeasible.

```bash
python3 -m fornax --help            # top-level command list
python3 -m fornax <command> --help  # flags for one command
```

Most commands below the core workflow are simulation and contract commands. They
validate the contracts that the heterogeneous runtime will use and replay golden
vectors. The everyday user flow is in the core workflow group.

## Core workflow

Use these commands to turn a model and fleet into a placement and prediction. See
[Planning and simulation](planning-and-simulation.md) for the walkthrough.

| Command | Purpose |
|---|---|
| `plan --target T --inventory INV [--links L] --out P` | Place the model across the fleet and write plan `P`. Exit `2` if infeasible. |
| `simulate --plan P [--requests R] [--out O]` | Predict throughput, latency, and pipeline bubble for a plan. Optionally project request-trace decode wall time. |
| `target validate TARGET --inventory INV [--links L] [--out O]` | Plan, then check the placement against target-contract thresholds. Exit `2` if any check fails. |
| `target draft --source S --inventory INV [--links L] --out O` | Render a target-contract draft from a model and inventory, then report whether it is already valid. |
| `preflight --target T --out-dir D [opts]` | Run the planning/contract pipeline and write evidence artifacts. |
| `doctor --bundle DIR [--out O]` | Inspect a phase-0 evidence bundle, such as a `preflight` output. |

## Hardware and fleet inspection

| Command | Purpose |
|---|---|
| `inventory collect` | Collect local machine inventory. |
| `inventory simulate-cluster` | Generate a simulated multi-node cluster inventory. |
| `accelerator {expert-mlp-probe,activation-transfer-probe,target-fixture-probe}` | Run accelerator micro-probes for expert MLP, activation transfer, and target fixture behavior. |
| `program local-4gpu-moe-serving-smoke --out-dir D [--devices cuda:0,cuda:1,cuda:2,cuda:3]` | Same-host CUDA smoke for a tiny MoE serving fixture: one gateway GPU plus three expert GPUs, with split-vs-reference parity. Scope excludes live HTTP, frontier parity, production distributed transport, and formal gate closure. |
| `program local-real-moe-serving-smoke --out O [--model-path P] [--devices cuda:0,cuda:1,cuda:2,cuda:3]` | Same-host real Qwen3-Omni MoE text-generation smoke. The default model is `Qwen/Qwen3-Omni-30B-A3B-Instruct`. The artifact records model and device placement evidence. Scope excludes live HTTP, production distributed serving, target-model parity, and formal gate closure. |
| `apple {probe-template,simulate-probe,validate-probe,role-decision}` | Apple Silicon worker probing and role decisions. |
| `fabric probe` | Probe a network link for inventory `links` bandwidth and latency. |
| `calibrate local` | Calibrate the cost model against the local machine. |

## Benchmarking

| Command | Purpose |
|---|---|
| `benchmark --plan P [opts]` | Run a benchmark for a plan and record a ledger entry with hardware, OS, driver, model, concurrency, and quantization metadata. |
| `throughput scaling-simulate` | Simulate throughput scaling behavior. |

## Runtime-component simulations and contracts

These commands model and validate pieces of the distributed engine. Each has a
matching golden vector under `fornax/golden_vectors/**` and a `test` target.

| Command | Component |
|---|---|
| `engine simulate` | Engine request, queue, and microbatch contract. |
| `serving {adapter-simulate,state-ownership-simulate}` | Serving adapter and state ownership. |
| `runtime stage-host-simulate` | Pipeline stage host. |
| `workers simulate` | Worker lifecycle contract. |
| `transport {simulate,trust-boundary-simulate}` | Cross-vendor activation/KV transport and trust boundary. |
| `replication simulate` | Stage replication. |
| `resilience replay-simulate` | Failure and replay resilience. |
| `scheduler simulate` | Request scheduling. |
| `moe {simulate,migration-simulate,remote-expert-probe,parity-probe}` | MoE expert routing, migration, remote expert batches, and numeric parity. |
| `model-support simulate` | Model-support coverage. |
| `batching simulate` | Continuous batching. |
| `pipeline correctness-probe` | Pipeline correctness. |
| `observability {metrics-simulate,trace-simulate}` | Metrics and trace ledgers. |
| `ops {lifecycle-simulate,onboarding-simulate}` | Operational lifecycle and onboarding. |

## Specs and program management

| Command | Purpose |
|---|---|
| `spec {runtime-format,network-security,model-support,backend-coverage,substrate-adr}` | Emit or validate the named specification. |
| `program {rebaseline,governance-simulate,local-accelerator-smoke,local-serving-smoke,local-http-serving-smoke,local-4gpu-moe-serving-smoke,local-real-moe-serving-smoke,phase3-proxy-gate,phase4-resilience-gate,phase5-ga-gate,g1-evidence-packet,...}` | Program-governance, local hardware-smoke, and phase-gate tooling. The four-GPU MoE smokes are same-host proxy evidence. |

## Gate runner: `test`

`test` replays a named golden or contract suite. It is the command behind
`make golden`. The command exits non-zero on any failure.

```bash
python3 -m fornax test golden-plans        # planner golden plans
python3 -m fornax test engine-seam         # string-in/string-out engine interface
python3 -m fornax test --help              # full suite list
```

Available suites include `golden-plans`, `runtime-format`, `network-contract`,
`engine-seam`, `stage-host`, `serving-adapter`, `local-4gpu-moe-serving-smoke`,
`local-real-moe-serving-smoke`, `state-ownership`, `engine-simulation`,
`observability`, `metrics-ledger`, `trace-ledger`, `worker-contract`,
`transport-contract`, `trust-boundary`, `moe-runtime`, `moe-parity-probe`,
`model-support`, `continuous-batching`, `scheduler-contract`,
`stage-replication`, `resilience-replay`, `ops-lifecycle`,
`onboarding-methodology`, `program-governance`, `backend-coverage`,
`phase3-proxy-gate`, `phase4-resilience-gate`, `phase5-ga-gate`,
`benchmark-ledger`, `pipeline-correctness-probe`, `throughput-scaling`, and more.
Run `python3 -m fornax test --help` for the current list.

`make golden` runs the deterministic no-hardware contract and golden suites.
`make test` runs those suites plus the unit tests. `local-4gpu-moe-serving-smoke`
and `local-real-moe-serving-smoke` validate saved artifacts when passed
`--fixture` or `--out`; they are outside the no-hardware golden run.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success, feasible, valid, or all gates passed. |
| `2` | Infeasible plan, failed validation, failed gate, or usage error. |
