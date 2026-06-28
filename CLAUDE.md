# CLAUDE.md

Guidance for Claude Code when working in the Fornax repository.

## What this is

Fornax is a **Mojo/MAX-native distributed inference engine** that makes a fleet of
heterogeneous commodity machines (consumer NVIDIA GPUs + Apple Silicon Macs)
serve a **single frontier-scale sparse-MoE model that no individual node can
hold**, at high *aggregate* throughput, on-prem. It is a *custom surgery of MAX*:
MAX's graph compiler, kernels, KV primitives, and custom-op API, extended with the
pieces MAX does not provide — heterogeneous pipeline execution, cross-vendor
activation/KV transport, and a heterogeneous MoE expert runtime.

Read [README.md](README.md) and [docs/fornax/project-plan-v3.md](docs/fornax/project-plan-v3.md)
before substantive work.

## The current codebase is Python (simulation + contracts)

Today's `fornax/` package is **pure Python**: a throughput-optimizing partitioner
(cost model + placement search), engine/serving/transport **simulations**, and
**gate validators** checked against **golden vectors**. This layer is deliberately
hardware-free — it specifies and tests the contracts the eventual Mojo/MAX runtime
must satisfy. The Mojo/MAX expert-runtime work (the critical path) builds against
these contracts; do not break a golden vector without updating its spec and saying
why.

## Commands

```bash
make test          # golden self-tests + unittest suite (no hardware/model/network)
make golden        # contract/golden-vector self-tests only
make unittest      # unittest suites only
python3 -m fornax --help          # full CLI (planner, simulate, gates, doctor, ...)
python3 -m fornax doctor          # inspect a phase-0 evidence bundle
```

The whole suite runs on CPU with no model and no network — that is the point of
the simulation/contract layer.

## Invariants to preserve

- **Golden vectors are the contract.** `fornax/golden_vectors/**` and
  `fornax/golden_plans/**` pin expected outputs. Changing engine/planner behavior
  means regenerating the affected vector *and* recording why in the corresponding
  doc/spec — never silently.
- **The engine seam is harness-agnostic.** Generation crosses a stable
  string-in / string-out boundary so any control plane can drive Fornax; keep that
  seam stable and do not bake in a specific consumer.
- **Metrics are honest.** Simulated throughput/latency/cache numbers must trace to
  an explicit model or fixture — never fabricate a measurement and present it as
  observed. Gate validators exist to enforce this.
- **No single-stream latency promises.** Fornax optimizes aggregate throughput and
  utilization for spanned models; the per-token network-crossing floor is real and
  documented. Keep docs/code honest about it.

## Docs

- `docs/fornax/` — project plan (v1–v3), partitioner spec, Apple-Silicon/MAX skill
  map, design reviews.
- `docs/fornax/program_management/` — charter, RACI, WBS, roadmap/critical-path,
  stage gates, RAID log, decision log, sprints, templates. The
  `fornax-program-manager` skill (`.claude/skills/`) (re)derives this tree from the
  current project plan.
- `docs/external_knowledge/` — background research PDFs.
