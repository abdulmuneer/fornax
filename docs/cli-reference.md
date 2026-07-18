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

Start with the deterministic no-hardware tour:

```bash
python3 -m fornax quickstart --out-dir fornax-quickstart
```

## Core workflow

Use these commands to turn a model and fleet into a placement and prediction. See
[Planning and simulation](planning-and-simulation.md) for the walkthrough.

| Command | Purpose |
|---|---|
| `quickstart [--out-dir D]` | Create a tiny synthetic NVIDIA/Apple target and fleet, force a two-stage plan, validate and simulate it, and write an honest summary. |
| `plan --target T --inventory INV [--links L] --out P [--authority-mode exploratory\|deployment] [--evidence-registry R]` | Place the model across the fleet and write plan `P`. The default is explicitly exploratory; `deployment` requires complete capability/measurement provenance plus a separate SHA-bound evidence registry and fails closed when `R` is absent or unresolved. Exit `2` if infeasible or authority is rejected. |
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
| `program apple-silicon-moe-serving-smoke --out O [--runtime-mode generate\|serve] [--max-command C] [--max-cwd D] [--max-extra-arg A] [--model-id M]` | Single-Mac Apple Silicon MAX-only smoke for a real Qwen, DeepSeek, Kimi, or GLM MoE. The default model is `Qwen/Qwen3-30B-A3B`, run through `max generate --devices gpu`; `--runtime-mode serve` starts `max serve` and requires one successful local `/v1/chat/completions` response. Pixi installs should use `--max-command "pixi run max" --max-cwd D`. Repeat `--max-extra-arg` for MAX bring-up flags. The artifact records MAX/Mojo version, Apple hardware, model MoE metadata, generated text or serve-start failure, HTTP status for serve mode, and explicit non-closure claims. |
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
| `runtime backend-conformance --factory module:create --manifest M [--options O] [--out R]` | Run the public Stage Backend lifecycle smoke. A pass is functional-contract evidence only and sets `closes_g2=false`. |
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
| `program phase05-engine-v0 --out O [--sustained-wall-seconds 1800] [--sustained-min-iterations 1800]` | Run the two-process Engine v0 closure workload, full scenario/fault/scheduler matrix, and real wall-clock sustained loopback; writes T1 evidence only. |
| `python3 -m fornax.lifecycle_pressure --out O [--wall-seconds 1800] [--min-iterations 1800]` | Exercise unique-request release, expiry, lease/tombstone fencing, and bounded state under pressure. Short runs are smoke evidence; a production claim still needs a reviewed sustained and physical/native-memory artifact. |
| `program g2-validate --out-dir D [--run-manifest M --run-physical]` | Verify the root MAX lineage, run and record V1-V5 prerequisites, and emit fail-closed V6-V10 physical statuses plus hashed machine/human evidence. Without authorized physical inputs it exits non-zero with `BLOCKED`; exit `0` means only that the technical packet passed, not that Sponsor/TL closed G2. |

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
python3 -m fornax test stage-abi-v1        # FNX1/backend conformance
python3 -m fornax test stage-abi-v2        # candidate FNX2 ragged reference + two workers
python3 -m fornax test phase05-engine-v0 \
  --fixture docs/fornax/evidence/phase05-engine-v0-2026-07-10.json
python3 -m fornax test --help              # full suite list
```

The Phase 0.5 fixture command validates immutable historical EV-009 and reports
`current_contract_authority=false`; it is not current physical or product
evidence. New sustained runs must use a new dated artifact and evidence ID.

Available suites include `golden-plans`, `runtime-format`, `network-contract`,
`engine-seam`, `stage-host`, `serving-adapter`, `local-4gpu-moe-serving-smoke`,
`local-real-moe-serving-smoke`, `apple-silicon-moe-serving-smoke`,
`state-ownership`, `engine-simulation`,
`observability`, `metrics-ledger`, `trace-ledger`, `worker-contract`,
`transport-contract`, `trust-boundary`, `moe-runtime`, `moe-parity-probe`,
`model-support`, `continuous-batching`, `scheduler-contract`,
`stage-replication`, `resilience-replay`, `ops-lifecycle`,
`onboarding-methodology`, `program-governance`, `backend-coverage`,
`phase3-proxy-gate`, `phase4-resilience-gate`, `phase5-ga-gate`,
`benchmark-ledger`, `pipeline-correctness-probe`, `throughput-scaling`,
`stage-abi-v1`, `stage-abi-v2`, `phase05-engine-v0`, and more.
Run `python3 -m fornax test --help` for the current list.

`make golden` runs the deterministic no-hardware contract and golden suites.
`make test` runs those suites plus the unit tests. `local-4gpu-moe-serving-smoke`,
`local-real-moe-serving-smoke`, and `apple-silicon-moe-serving-smoke` validate
saved artifacts when passed `--fixture` or `--out`; they are outside the
no-hardware golden run.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success, feasible, valid, or all gates passed. |
| `1` | Contract smoke failed, or `g2-validate` is blocked/failed and therefore not a passing technical packet. |
| `2` | Infeasible plan, failed validation, failed gate, or usage error. |
