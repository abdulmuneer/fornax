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

## 8. Optional: Apple Silicon MAX MoE smoke

On an Apple Silicon Mac with a current MAX runtime, run the MAX-only single-Mac
real-MoE smoke:

```bash
python3 -m fornax program apple-silicon-moe-serving-smoke \
    --out /tmp/fornax_apple_qwen3_moe_smoke.json \
    --max-command "max" \
    --model-id Qwen/Qwen3-30B-A3B \
    --devices gpu \
    --max-new-tokens 8
```

If MAX is installed in a temporary `pixi` project, invoke it through Pixi and
set the project directory as the MAX working directory:

```bash
python3 -m fornax program apple-silicon-moe-serving-smoke \
    --out /tmp/fornax_apple_qwen3_moe_smoke.json \
    --max-command "pixi run max" \
    --max-cwd /tmp/fornax-max-smoke \
    --model-id Qwen/Qwen3-30B-A3B \
    --devices gpu \
    --max-new-tokens 8
```

Validate a saved artifact without reloading the model:

```bash
python3 -m fornax test apple-silicon-moe-serving-smoke \
    --fixture /tmp/fornax_apple_qwen3_moe_smoke.json
```

This smoke runs `max generate` through Fornax, reads the downloaded model config,
and requires real MoE metadata such as `num_experts` and `num_experts_per_tok`.
It records MAX/Mojo versions, Apple hardware metadata, generated text, and an
OpenAI-compatible response-shaped wrapper around the string result.

To test MAX's live OpenAI-compatible server startup path, switch modes:

```bash
python3 -m fornax program apple-silicon-moe-serving-smoke \
    --out /tmp/fornax_apple_qwen3_moe_serve_smoke.json \
    --runtime-mode serve \
    --max-command "pixi run max" \
    --max-cwd /tmp/fornax-max-smoke \
    --model-id Qwen/Qwen3-30B-A3B \
    --devices gpu \
    --max-new-tokens 8
```

In `serve` mode, Fornax starts `max serve`, sends one local
`/v1/chat/completions` request, records the HTTP status and response body, and
then stops the server. A passing `serve` artifact requires a live HTTP completion
with generated text; server startup alone is not enough.

Use a current `pixi`/MAX install. The legacy `modular` CLI can be too old to
fetch current MAX releases and may report expired manifests. On this Mac, a
temporary nightly MAX install used `MAX 26.5.0.dev2026062906` and `Mojo
1.0.0b3.dev2026062906`. Invoking the raw Pixi environment binary directly can
skip Pixi's runtime environment setup and fail while resolving MAX built-in
kernel packages; prefer `pixi run max` with `--max-cwd` for Pixi projects.
If a MAX bring-up needs raw runtime flags, repeat `--max-extra-arg`, for example
`--max-extra-arg=--no-enable-overlap-scheduler --max-extra-arg=--force`.
Some Hugging Face repositories require executing model repository code during
configuration. Only use `--max-extra-arg=--trust-remote-code` after explicitly
deciding to trust that repository.

Additional MAX-only probes on this Mac:

- `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` with
  `--quantization-encoding float8_e4m3fn` still failed in MAX's Apple Metal/KGEN
  graph compiler before generated text.
- Cached Qwen FP8 retries with `--sample-on-host` and `--prefer-module-v3`
  failed with the same Metal/KGEN signature, so sampling placement and the
  module-v3 preference did not bypass the graph compiler path on this nightly.
- Cached Qwen FP8 with `--runtime-mode serve` started MAX's server supervisor
  and model worker, but the worker crashed before the HTTP endpoint became ready.
  The structured artifact is `/tmp/fornax_apple_qwen3_moe_fp8_serve_smoke.json`;
  it records `mode=max-serve`, `live_http_endpoint=false`, and the same
  Metal/KGEN failure signature.
- The Apple CPU backend rejected cached Qwen FP8 and BF16 attempts because
  `float8_e4m3fn` and `bfloat16` are not compatible with `--devices cpu` on this
  MAX nightly. A float32 CPU retry was not run because the 30B MoE footprint
  would likely exceed the practical 128 GB unified-memory headroom once runtime
  overhead is included.
- `moonshotai/Kimi-VL-A3B-Instruct` records routed expert metadata
  (`n_routed_experts=64`, `num_experts_per_tok=6`). Without
  `--trust-remote-code`, Hugging Face rejects config loading. With
  `--trust-remote-code`, MAX reaches config parsing but rejects the model because
  only `yarn` rope scaling is currently supported.

Scope covered: one Apple Silicon Mac, real Qwen and Kimi MoE model bring-up
attempts through MAX, optional MAX live-server startup plus one local HTTP
completion in `serve` mode, Apple hardware/runtime/model metadata capture, and
Fornax artifact validation. A passing artifact requires generated text; a
passing `serve` artifact also requires HTTP status 200. The recorded Qwen and
Kimi artifacts on this nightly do not pass those gates.

Scope excluded: non-MAX Apple runtimes, heterogeneous distributed runtime,
remote expert transport, target-model parity reference, production serving, and
formal G2/G3 gate closure.

## 9. Optional: DeepSeek-V2-Lite on Apple Silicon with the patched MAX tree

This path is for reproducing the local DeepSeek-V2-Lite evidence on an Apple
Silicon Mac. It is not the packaged `max` path from `/tmp/fornax-max-smoke`.
It uses the source-built MAX CLI from `external/modular`, which contains the
local Apple MLA/MoE/gather backend work.

The root `fornax/` Python package supplies the planner, simulator, contracts,
and evidence wrappers. The DeepSeek execution path below also needs the
companion patched MAX source checkout at `external/modular`; a plain Fornax
clone without that tree cannot run this smoke.

Use this as a short model smoke, not as serving-grade validation. The verified
host was an M3 Max with 128 GB unified memory. DeepSeek-V2-Lite-Chat BF16
weights are large enough that smaller-memory Macs may fail before reaching the
MAX backend path.

Prerequisites:

- An Apple Silicon Mac, with M3+ and large unified memory recommended.
- Xcode/Metal command-line tools installed.
- The Fornax checkout plus `external/modular` present and containing the local
  Apple backend patch, at or after nested commit `957aede`. If your clone does
  not include `external/modular`, obtain the companion patched MAX source tree
  before continuing.
- A local Hugging Face snapshot of `deepseek-ai/DeepSeek-V2-Lite-Chat`.
  The verified revision was
  `85864749cd611b4353ce1decdb286193298f64c7`. Download it with your preferred
  Hugging Face tooling, for example:

```bash
hf download deepseek-ai/DeepSeek-V2-Lite-Chat \
  --revision 85864749cd611b4353ce1decdb286193298f64c7 \
  --local-dir /path/to/deepseek-v2-lite-chat
```

Build the source MAX CLI:

```bash
cd /path/to/fornax/external/modular
git rev-parse --short HEAD
./bazelw build //max/python/max/_entrypoints:pipelines
cd /path/to/fornax
```

Run a one-device GPU smoke:

```bash
export FORNAX_ROOT=/path/to/fornax
export MAX_CLI="${FORNAX_ROOT}/external/modular/bazel-bin/max/python/max/_entrypoints/pipelines"
export RUNFILES_DIR="${MAX_CLI}.runfiles"
export DEEPSEEK_SNAPSHOT=/path/to/deepseek-v2-lite-chat
export METAL_TOOLCHAIN="$(dirname "$(xcrun --find metal)")"

export PATH="${METAL_TOOLCHAIN}:${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODULAR_MOJO_MAX_PACKAGE_ROOT="${RUNFILES_DIR}/+rebuild_wheel+module_platlib_macos_arm64/modular"
export MODULAR_MOJO_MAX_IMPORT_PATH="${RUNFILES_DIR}/_main/max/kernels/src,${RUNFILES_DIR}/_main/max/kernels/src/graph_compiler,${RUNFILES_DIR}/_main/max/driver/src,${RUNFILES_DIR}/_main/max/compiler/src,${RUNFILES_DIR}/_main/mojo/stdlib,${RUNFILES_DIR}/+rebuild_wheel+module_platlib_macos_arm64/modular/lib/mojo"
export MODULAR_MOJO_MAX_DRIVER_PATH="${RUNFILES_DIR}/rules_mojo++mojo+mojo_toolchain_macos_arm64/bin/mojo"
export MODULAR_MOJO_MAX_LLD_PATH="${RUNFILES_DIR}/rules_mojo++mojo+mojo_toolchain_macos_arm64/bin/lld"
export MODULAR_MOJO_MAX_COMPILERRT_PATH="${RUNFILES_DIR}/rules_mojo++mojo+mojo_toolchain_macos_arm64/lib/libKGENCompilerRTShared.dylib"

"${MAX_CLI}" --version
"${MAX_CLI}" generate \
  --model "${DEEPSEEK_SNAPSHOT}" \
  --devices gpu:0 \
  --quantization-encoding bfloat16 \
  --max-length 128 \
  --max-batch-size 1 \
  --max-batch-total-tokens 128 \
  --max-new-tokens 8 \
  --top-k 1 \
  --temperature 0 \
  --prompt "User: Say hi. Assistant:"
```

Expected shape of a successful run:

```text
MAX 26.5.0.dev2026063006
architecture: DeepseekV2ForCausalLM
devices: gpu[0]
!!!!!!!!
Prompt size: 14
Output size: 8
```

The local M3 Max evidence run produced about 365 ms TTFT and about 4 tok/s
token-generation throughput for this 8-token smoke. Treat those numbers as
evidence for the patched backend path, not as a Fornax serving target.

Common failure modes:

- `flareMLA_prefill currently only supports Nvidia and AMD GPUs`: you are using
  packaged/public MAX instead of the patched source-built CLI.
- `MAXG_addKernelPackage: failed to import kernels from ''`: one of the
  `RUNFILES_DIR` or `MODULAR_MOJO_MAX_*` environment variables is missing or
  points at a stale build tree.
- Offline Hugging Face errors for `configuration_deepseek.py`: use the local
  snapshot path and omit `--trust-remote-code`; this run uses MAX's built-in
  `DeepseekV2ForCausalLM` path.
- `xcrun` or Metal compiler lookup failures: install Xcode/Metal command-line
  tools, confirm `xcrun --find metal` returns a path, and keep that directory
  first in `PATH`. On the verified host it resolved to the Apple MobileAsset
  Metal toolchain under `/var/run/com.apple.security.cryptexd/...`.

Scope of this smoke:

- Covered: one Apple Silicon GPU device, source-built MAX, short generate path.
- Not covered: `max serve`, distributed Fornax execution, long-context behavior,
  batching, numerical parity, or G1 throughput gates.

See [docs/fornax/deepseek-v2-lite-max-check.md](fornax/deepseek-v2-lite-max-check.md)
for the backend evidence and remaining point of failure.

## 10. Next docs

- [Planning and simulation](planning-and-simulation.md): target contracts,
  inventories, request traces, plan output, predictions, and preflight bundles.
- [Input file reference](input-formats.md): JSON fields for targets,
  inventories, and links.
- [CLI reference](cli-reference.md): command groups and exit codes.
- [Concepts](concepts.md): model placement vocabulary and runtime scope.
