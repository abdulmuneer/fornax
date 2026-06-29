# Getting started

This guide takes a fresh clone through the first placement plan and simulation.
The default path runs on a laptop: no GPU, model download, or network access. The
runnable layer today is the planner and simulation/contract layer, implemented in
standard-library Python. An optional four-GPU smoke section is included for CUDA
machines with PyTorch.

Read [Concepts](concepts.md) first if you want the model and fleet vocabulary.

## 1. Prerequisites

- Python 3.10 or newer. The project is developed and tested on CPython 3.12.
- No third-party packages for the default planner and contract workflow.
- Optional hardware smokes need a separate Python environment with PyTorch and
  visible CUDA devices.

```bash
python3 --version
```

## 2. Get the code

```bash
git clone git@github.com:abdulmuneer/fornax.git
cd fornax
```

Run Fornax as a module from the repo root:

```bash
python3 -m fornax --help
```

There is no installed `fornax` shell command in the default workflow.

## 3. Verify the repo

Run the deterministic golden-vector checks and the unit tests:

```bash
make test
```

Useful subsets:

```bash
make golden                       # contract and golden-vector checks
make unittest                     # unit tests
python3 -m fornax --help          # command list
```

A healthy run reports all golden suites passed and ends the unit tests with
`OK`.

## 4. Run the first plan and simulation

The core loop has two commands:

1. `plan` reads a model target and an inventory, places the model across the
   fleet, and reports feasibility.
2. `simulate` reads a feasible plan and reports predicted throughput, latency,
   and pipeline bubble.

Create a one-node inventory:

```bash
cat > my_fleet.json <<'JSON'
{
  "nodes": [
    {
      "id": "gpu0",
      "vendor": "nvidia",
      "runtime": "max",
      "mem_free_bytes": 16777216,
      "compute_class": 1000000000000.0,
      "mem_bandwidth_bytes_s": 100000000000.0,
      "supports_stage": true,
      "supports_expert_worker": true,
      "supports_kv": true,
      "supported_dtypes": ["fp16", "fp8"]
    }
  ],
  "links": []
}
JSON
```

Place the bundled example target on that inventory, then simulate it:

```bash
python3 -m fornax plan \
    --target fornax/golden_plans/v0_target_contract_fixture.md \
    --inventory my_fleet.json \
    --out plan.json

python3 -m fornax simulate --plan plan.json
```

Expected output:

```text
wrote placement plan: plan.json
simulate: throughput=89214.343 tok/s latency=0.000359s bubble=0.000
```

The throughput is a cost-model prediction for this placement. It is useful for
checking the workflow and comparing placements. Use benchmarks for
machine-specific runtime measurements. See
[Prediction and measurement scope](concepts.md#prediction-and-measurement-scope).

Open `plan.json` to inspect the selected stages, per-stage memory and timing,
and the `explanations` entries that describe each placement decision.

## 5. Validate a plan against a target

A target contract can include acceptance thresholds such as a throughput floor,
memory headroom, and a concurrency sweep. `target validate` plans the model onto
the fleet and checks those thresholds:

```bash
python3 -m fornax target validate \
    fornax/golden_plans/v0_target_contract_fixture.md \
    --inventory my_fleet.json
```

It prints `valid` and exits `0` when every check passes. It prints
`invalid: <failed checks>` and exits `2` when any check fails.

## 6. Inspect an infeasible plan

Feasibility is part of the planner result. Shrink the available memory and rerun
`plan`:

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path('my_fleet.json').read_text())
data['nodes'][0]['mem_free_bytes'] = 1024
Path('tiny_fleet.json').write_text(json.dumps(data))
PY

python3 -m fornax plan \
    --target fornax/golden_plans/v0_target_contract_fixture.md \
    --inventory tiny_fleet.json \
    --out infeasible.json
echo "exit=$?"
```

`plan` exits `2`. `infeasible.json` records `"feasible": false` and an
`infeasible_reason`. The bundled golden plan `model_too_big.json` is a permanent
example of this output shape.

## 7. Optional: four-GPU MoE serving smokes

On a machine with four visible CUDA GPUs and a PyTorch environment, run the
same-host tiny MoE serving fixture:

```bash
python3 -m fornax program local-4gpu-moe-serving-smoke \
    --out-dir /tmp/fornax_local_4gpu_moe_serving_smoke \
    --torch-python /path/to/torch/python \
    --devices cuda:0,cuda:1,cuda:2,cuda:3
```

The first device is the gateway, router, and gather GPU. The other three devices
run the tiny fixture experts. A passing run writes
`local-4gpu-moe-serving-smoke.json` and child artifacts under the output
directory, then prints the check count, GPU count, gateway GPU, expert GPUs, and
generated fixture text.

Validate a saved artifact without rerunning CUDA work:

```bash
python3 -m fornax test local-4gpu-moe-serving-smoke \
    --fixture /tmp/fornax_local_4gpu_moe_serving_smoke
```

Scope covered: deterministic tiny MoE serving on one physical host, four visible
GPUs, routed expert work on all three expert GPUs, and split-path parity against
the reference path.

Scope excluded: live HTTP serving, frontier-model parity, production distributed
transport, and formal G2/G3 gate closure.

To exercise the cached Qwen3-Omni MoE checkpoint, run the real-model text smoke:

```bash
python3 -m fornax program local-real-moe-serving-smoke \
    --out /tmp/fornax_qwen3_omni_real_moe_smoke.json \
    --torch-python /mnt/dataprocessing/venvs/aiccu_falcon_tdt/bin/python \
    --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --model-path /mnt/dataprocessing/cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Validate the saved artifact without reloading the checkpoint:

```bash
python3 -m fornax test local-real-moe-serving-smoke \
    --fixture /tmp/fornax_qwen3_omni_real_moe_smoke.json
```

This smoke loads `Qwen/Qwen3-Omni-30B-A3B-Instruct` through Transformers with
BF16 and `device_map=auto`, renders a cached tokenizer/template chat prompt, and
generates text with `return_audio=False`. The artifact records architecture,
expert counts, device-map placement, per-device parameter counts, H100 device
names, generated text, token counts, and throughput.

Scope excluded: live HTTP serving, full multimodal serving, production
distributed transport, target-model parity reference, and formal G2/G3 gate
closure.

## 8. Next docs

- [Planning and simulation](planning-and-simulation.md): target contracts,
  inventories, request traces, plan output, predictions, and preflight bundles.
- [Input file reference](input-formats.md): JSON fields for targets,
  inventories, and links.
- [CLI reference](cli-reference.md): command groups and exit codes.
- [Concepts](concepts.md): model placement vocabulary and runtime scope.
