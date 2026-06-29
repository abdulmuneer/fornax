# Fornax — heterogeneous frontier-model serving

**Fornax** is a **Mojo/MAX-native distributed inference engine** — a *custom
surgery of MAX* — that makes a fleet of heterogeneous, commodity machines
(consumer NVIDIA GPUs + Apple Silicon Macs + whatever else is on the LAN) serve a
**single frontier-scale MoE model that no individual node can hold**, at high
*aggregate* throughput, on-prem, for firms that want frontier capability in-house
without shipping data to a provider.

Fornax is an **engine**. It is assembled from MAX's own components — graph
compiler, kernel library, KV-cache primitives, the custom-op API — extended with
the pieces MAX does not provide: heterogeneous pipeline execution, cross-vendor
activation/KV transport, and a model-specific heterogeneous MoE expert runtime for
Apple/NVIDIA/AMD workers. It exposes a stable, harness-agnostic engine seam
(string-in / string-out generation) so any control plane can drive it.

## The one-line thesis

> Frontier open models are large **sparse MoE** (capacity-bound to store,
> compute-light per token). Heterogeneous commodity hardware is **cheap capacity
> (Mac unified memory) + cheap compute (consumer GPUs)**. The shapes match — the
> bottleneck is the **interconnect**, and the engineering is an engine that hides
> it: MAX's portable kernels under a heterogeneous pipeline of our own.

## What it is not

Fornax is **not** a wrapper that load-balances `max serve`. It cuts into the
model execution path. The surgical seam is the MoE block:

```text
hidden states -> router -> local expert batches + remote expert batches
              -> weighted gather -> next layer
```

The dense path (attention, KV, routers, shared/hot experts, sampler) stays on
the fastest local accelerator group whenever possible. Heterogeneous workers
extend model capacity and throughput by running pipeline stages and/or bounded
routed-expert batches, not by pretending every device can participate in every
operation equally.

## The honest constraint (read first)

When the model exceeds the biggest node, **every token crosses the network**.
Even when the network is provisioned for this workload, a spanned model has a
pipeline and synchronization floor that a single-node model does not. Fornax
preserves **aggregate throughput and utilization** (high total tok/s, high
$/token efficiency) via continuous batching, overlap, expert locality, and
balanced stages — *not* single-stream latency parity. That latency cost is the
irreducible price of spanning.

## Repository layout

| Path | What it is |
|---|---|
| `fornax/` | The Python package: planner (cost model + placement search), engine/serving simulations, gate validators, golden plans/vectors, and the `python -m fornax` CLI. |
| `tests/` | `unittest` suites (planner and contracts). |
| `docs/` | End-user documentation — start at [docs/README.md](docs/README.md). |

**New here? Read [docs/getting-started.md](docs/getting-started.md)** — install,
verify, and run your first plan → simulate in a few minutes (no GPU, no model).
The full guide index is [docs/README.md](docs/README.md).

## Quickstart

```bash
make test          # golden self-tests + unittest suite (no hardware required)
make golden        # deterministic CLI contract/golden self-tests
python3 -m fornax --help          # the CLI surface
python3 -m fornax doctor          # inspect a phase-0 evidence bundle
python3 -m fornax test golden-plans
```

Everything above runs **without a GPU or a model** — the planner, contracts, and
gate validators are pure logic, validated against golden vectors. Machines with
four visible CUDA GPUs can additionally validate either the same-host tiny MoE
serving fixture or a cached real Qwen MoE text-generation smoke:

```bash
python3 -m fornax program local-4gpu-moe-serving-smoke \
    --out-dir /tmp/fornax_local_4gpu_moe_serving_smoke

python3 -m fornax program local-real-moe-serving-smoke \
    --out /tmp/fornax_qwen3_omni_real_moe_smoke.json \
    --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python \
    --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --model-path /mnt/dataprocessing/cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3
```

## Status

Active development. The planner, contract validators, local serving/proxy-gate
fixtures, same-host four-H100 tiny MoE serving smoke, same-host four-H100
Qwen3-Omni real MoE text-generation smoke, and Phase 3-5 two-H100 proxy packets
are implemented and self-tested at their stated scope. Formal G1-G5 closure still
requires the human sign-offs and real heterogeneous
lab evidence tracked in
[fornax_program_management_todo_status.md](fornax_program_management_todo_status.md).
See the [program-management tree](docs/fornax/program_management/) for the
roadmap, stage gates, and sprint backlog.
