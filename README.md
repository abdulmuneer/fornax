# [WiP] Fornax - heterogeneous frontier-model serving

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

Fornax places one model across the fleet. The v0 spanning spine is pipeline
parallelism by complete contiguous layer groups:

```text
gateway -> stage 0 (MAX graph) -> activation frame -> stage 1 (MAX graph)
        -> ... -> final logits / sampler
```

Remote experts remain a deferred measured optimization. Engine v0 first builds
the production Stage ABI, orchestrator, and TCP framing against reference and
simulated MAX backends. Physical backends replace assumptions without changing
the engine contract.

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

Apple Silicon Macs with a current MAX runtime can run the MAX-only single-Mac
real-MoE smoke against a Qwen/DeepSeek/Kimi/GLM MoE:

```bash
python3 -m fornax program apple-silicon-moe-serving-smoke \
    --out /tmp/fornax_apple_qwen3_moe_smoke.json \
    --max-command "pixi run max" \
    --max-cwd /path/to/pixi-max-project \
    --model-id Qwen/Qwen3-30B-A3B \
    --devices gpu \
    --max-new-tokens 8
```

To exercise MAX's OpenAI-compatible server startup path instead of the default
single-shot `max generate` path, add `--runtime-mode serve`:

```bash
python3 -m fornax program apple-silicon-moe-serving-smoke \
    --out /tmp/fornax_apple_qwen3_moe_serve_smoke.json \
    --runtime-mode serve \
    --max-command "pixi run max" \
    --max-cwd /path/to/pixi-max-project \
    --model-id Qwen/Qwen3-30B-A3B \
    --devices gpu \
    --max-new-tokens 8
```

This is MAX/model bring-up evidence only; non-MAX Apple runtimes do not satisfy
this smoke. It is not distributed serving or formal G2/G3 gate closure.

## Status

Active development follows [project plan v4](docs/fornax/project-plan-v4.md).
Phase 0.5 / M1 is complete at T0/T1: the Python package now includes two
independent workers, the production Stage ABI and framed TCP transport,
reference/simulated MAX backends, bounded scheduling, fault injection, evidence
ledgers, and the planner regressions. The
[exit review](docs/fornax/program_management/gate-reviews/phase-0-5-exit-2026-07-10.md)
records the exact scope.

The active lane is Phase 1 physical `MaxStageBackend` integration and G2 evidence
acquisition as hardware becomes available. Phase 0.5 does not establish physical
heterogeneous correctness, supported-platform status, or production performance;
all remaining hardware assumptions stay explicit in
[the simulation contract](docs/fornax/simulation-and-assumption-contract.md).

```bash
python3 -m fornax test stage-abi-v1
python3 -m fornax test phase05-engine-v0 \
  --fixture docs/fornax/evidence/phase05-engine-v0-2026-07-10.json
```
