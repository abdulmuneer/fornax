# Fornax - heterogeneous frontier-model serving

Fornax is a Mojo/MAX-native distributed inference engine for serving a single
frontier-scale sparse-MoE model across a fleet of heterogeneous commodity
machines. The target fleet can include consumer NVIDIA GPUs, Apple Silicon Macs,
AMD devices, and CPU workers on a local network.

The engine uses MAX components where they fit: graph compilation, kernels,
KV-cache primitives, and custom ops. Fornax adds the missing distributed pieces:
heterogeneous pipeline execution, activation and KV transport across vendors, and
model-specific MoE expert execution on Apple, NVIDIA, and AMD workers. The public
engine interface is string-in/string-out generation so a serving layer can drive
it without owning the execution internals.

## Thesis

Frontier open models increasingly use sparse Mixture-of-Experts. They need large
aggregate memory for all expert weights, while each token activates only a small
subset of experts. Commodity fleets have a similar shape: Macs provide large
unified memory, consumer GPUs provide inexpensive compute, and the network is the
constraint to manage.

Fornax plans and executes around that constraint. It keeps dense work on the
fastest local accelerator group when possible and sends bounded expert batches to
capacity-rich workers.

## Execution model

Fornax places one model across the fleet. The key split is inside the MoE block:

```text
hidden states -> router -> local expert batches + remote expert batches
              -> weighted gather -> next layer
```

The dense path includes attention, KV, routers, shared or hot experts, and the
sampler. Routed experts can live on other workers when that improves capacity or
throughput. The planner decides where stages and experts live, then reports both
feasibility and predicted performance.

## Throughput and latency scope

When the model is larger than the biggest node, every token crosses the network.
That adds a pipeline and synchronization floor. Fornax optimizes aggregate
throughput and utilization through continuous batching, overlap, expert locality,
and balanced stages. Single-stream latency includes the cost of spanning the
model.

Simulator output is a cost-model prediction. Benchmark and serving-smoke output
is measured data for the hardware and model named in the artifact.

## Repository layout

| Path | Contents |
|---|---|
| `fornax/` | Python package: planner, cost model, placement search, simulations, validators, golden plans and vectors, and the `python3 -m fornax` CLI. |
| `tests/` | `unittest` suites for the planner and contracts. |
| `docs/` | User documentation. Start at [docs/README.md](docs/README.md). |

New users should start with [docs/getting-started.md](docs/getting-started.md).
It verifies the repo and runs the first plan and simulation without a GPU or
model download. The full guide index is [docs/README.md](docs/README.md).

## Quickstart

```bash
make test                         # golden self-tests + unittest suite
make golden                       # deterministic CLI contract/golden self-tests
python3 -m fornax --help          # CLI surface
python3 -m fornax doctor          # inspect a phase-0 evidence bundle
python3 -m fornax test golden-plans
```

These commands run on CPU with no model. Machines with four visible CUDA GPUs can
also run same-host MoE serving smokes:

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

Active development. The planner, contract validators, local serving and proxy
fixtures, same-host four-H100 tiny MoE serving smoke, same-host four-H100
Qwen3-Omni real MoE text-generation smoke, and Phase 3-5 two-H100 proxy packets
are implemented and self-tested at their stated scope.

Formal G1-G5 closure still requires human sign-off and real heterogeneous lab
evidence tracked in [fornax_program_management_todo_status.md](fornax_program_management_todo_status.md).
See [docs/fornax/program_management/](docs/fornax/program_management/) for the
roadmap, gates, and sprint backlog.
