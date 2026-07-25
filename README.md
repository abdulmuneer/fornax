# [WiP] Fornax - heterogeneous frontier-model serving

Fornax is building a Mojo/MAX-native distributed inference engine for serving a single
frontier-scale sparse-MoE model across a fleet of heterogeneous commodity
machines. The target fleet can include consumer NVIDIA GPUs, Apple Silicon Macs,
AMD devices, and CPU workers on a local network.

Today this repository is a **pre-alpha executable specification and Engine v0
prototype**: planner, the frozen experimental FNX1 mechanism, an implemented
candidate FNX2 ragged reference engine, independent loopback workers, validators,
and hardware bring-up tools. A physical
cross-vendor `MaxStageBackend` and frontier-capacity proof remain open G2/G3
milestones.

The target runtime uses MAX components where they fit: graph compilation,
kernels, KV-cache primitives, and custom ops. Fornax is designed to add the
distributed pieces: heterogeneous pipeline execution, activation/KV transport
across vendors, and model-specific MoE expert execution on qualified workers.
Those physical paths remain G2/G3 work. The public
Python `Engine` contract is string-in/string-out over an explicitly supplied
generator, so a serving layer can drive it without owning execution internals.
No bundled physical text generator or `fornax serve` exists yet; Engine v0 starts
at activation tensors. See the
[Stage Backend adapter guide](docs/fornax/stage-backend-adapters.md).

## Thesis

Frontier open models increasingly use sparse Mixture-of-Experts. They need large
aggregate memory for all expert weights, while each token activates only a small
subset of experts. Commodity fleets have a similar shape: Macs provide large
unified memory, consumer GPUs provide inexpensive compute, and the network is the
constraint to manage.

The current planner models that constraint. The target runtime would keep dense
work on the fastest qualified accelerator group when possible; remote expert
batches remain a deferred, measured optimization rather than an implemented
path.

## Execution model

The target runtime places one model across the fleet. The v0 design spine is
pipeline parallelism by complete contiguous layer groups:

```text
gateway -> stage 0 (MAX graph) -> activation frame -> stage 1 (MAX graph)
        -> ... -> final logits / sampler
```

Remote experts remain a deferred measured optimization. Historical Engine v0
exercises experimental FNX1 with a lockstep orchestrator. Candidate FNX2 2.0
adds token/activation/logit stage roles, unequal prefill, changing-subset decode,
per-sequence KV/error state, leases, tombstones, multi-dimensional credits, and
an integrated scheduler over two independent reference workers. Physical MAX
adapters must conform to FNX2 before making a ragged-batching claim.

## Throughput and latency scope

When the model is larger than the biggest node, every token crosses the network.
That adds a pipeline and synchronization floor. Fornax targets aggregate
throughput and utilization through batching/overlap, expert locality, and
balanced stages. Ragged batching is now integrated only in the model-free T1
reference runtime; batching performance on physical MAX stages remains unproven.
Single-stream latency includes the cost of spanning.

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

Hardware bring-up work can start before devices arrive with the
[consumer MoE qualification recipes](docs/fornax/consumer-hardware-recipes.md).
The packaged three-model by six-platform matrix produces deterministic C1
operator packets; it makes no physical compatibility or product-support claim.

Backend authors should use the versioned public imports in `fornax.backends` and
run `fornax runtime backend-conformance`; passing that functional smoke is not
physical G2 evidence.
FNX2 logical types and the reference oracle are exported from `fornax.ragged`.

## Quickstart

```bash
python3 -m fornax quickstart          # one-command, no-hardware tour
make test                         # golden self-tests + unittest suite
make golden                       # deterministic CLI contract/golden self-tests
python3 -m fornax --help          # CLI surface
python3 -m fornax --version
python3 -m fornax doctor --bundle <preflight-dir>
python3 -m fornax test golden-plans
python3 -m fornax test stage-abi-v2
python3 -m fornax program g2-validate --out-dir /tmp/fornax-g2
```

The quickstart forces a tiny model across synthetic NVIDIA and Apple nodes and
writes inspectable target, inventory, placement, validation, and simulation
artifacts under `fornax-quickstart/`. Its numbers are predictions, not hardware
measurements.

To build the fail-closed G2 readiness packet before hardware is available:

```bash
python3 -m fornax program g2-validate \
  --out-dir evidence/g2-readiness-YYYYMMDD-HHMMSS
```

The expected exit is nonzero until V6-V10 physical evidence exists. The bundle
still verifies the root MAX lineage and current V1-V5/FNX2 prerequisites, then
records every physical step as blocked. See
[G2-in-a-box](docs/fornax/g2-in-a-box.md).

For an isolated editable install with a `fornax` console command:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/fornax quickstart
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

The MAX-only single-Mac command can attempt a bounded real-MoE smoke against a
Qwen/DeepSeek/Kimi/GLM/GPT-OSS model on Apple Silicon:

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
Phase 0.5 / M1 remains complete in its recorded FNX1 T0/T1 scope. A separate
candidate FNX2 T0/T1 path now executes unequal prefill and independent decode
through an integrated bounded scheduler and two independently spawned loopback
workers. The historical
[exit review](docs/fornax/program_management/gate-reviews/phase-0-5-exit-2026-07-10.md)
records the exact scope.

The reference/simulated runtime now has explicit final release, opportunistic
idle expiry, internal execution leases, same-worker reconnect tombstones, count
and byte caps, a copy-explicit buffer adapter seam, and a runnable unique-request
pressure evidence mode. These are T0/T1 mechanisms: tombstones do not survive a
worker restart, EV-016's 1,800-second active-churn candidate used uncommitted
source and crossed a long civil-clock suspension, and no physical native-KV/
buffer soak supports an indefinite-service claim. The current runner records a
maximum progress gap and fails interrupted sustained runs.

The active lane is Phase 1 physical `MaxStageBackend` integration and G2 evidence
acquisition as hardware becomes available. Phase 0.5 does not establish physical
heterogeneous correctness, supported-platform status, or production performance;
all remaining hardware assumptions stay explicit in
[the simulation contract](docs/fornax/simulation-and-assumption-contract.md).

```bash
python3 -m fornax test stage-abi-v1
python3 -m fornax test stage-abi-v2
python3 -m fornax program g2-validate --out-dir /tmp/fornax-g2
# Historical EV-009 validation; expect current_contract_authority=False.
python3 -m fornax test phase05-engine-v0 \
  --fixture docs/fornax/evidence/phase05-engine-v0-2026-07-10.json
```
