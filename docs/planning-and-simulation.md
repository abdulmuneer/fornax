# Planning and simulation

This guide walks through the core Fornax workflow: describe a model and fleet,
place the model, simulate the placement, validate thresholds, and write a
preflight evidence bundle. If you have not run the small example yet, start with
[Getting started](getting-started.md).

```text
  target contract ----+
                       +--> plan --> plan.json --> simulate --> prediction
  inventory ----------+
                       +--> target validate --> valid / invalid
```

## Step 1: describe the model and serving target

The model and serving goal live in a target contract. The file can be JSON, or it
can be Markdown with a fenced `json fornax-target` block.

Minimal JSON form:

```json
{
  "model": {
    "hidden_dim": 1024,
    "num_layers": 2,
    "dtype_weight": "q4",
    "dtype_activation": "fp16",
    "layers": [
      { "kind": "attention", "weight_bytes": 1048576, "active_flops_per_token": 2000000, "kv_bytes_per_token": 8192 },
      { "kind": "dense", "weight_bytes": 1048576, "active_flops_per_token": 2000000 }
    ]
  },
  "target": {
    "concurrency": 4,
    "prompt_len": 16,
    "gen_len": 8,
    "objective": "max_throughput"
  }
}
```

- `layers` is the main model description. Each entry records memory
  (`weight_bytes`), compute (`active_flops_per_token`), and optional KV cost
  (`kv_bytes_per_token`). MoE layers also set `num_experts`, `experts_active`,
  `expert_bytes`, and related fields. See [Input file reference: Model](input-formats.md#model).
- `objective` is one of `max_throughput`, `min_latency`, or `balanced`.

See [`v0_target_contract_fixture.md`](../fornax/golden_plans/v0_target_contract_fixture.md)
for a runnable Markdown target. It also includes a `contract` block used by
`target validate`.

## Step 2: describe the fleet

An inventory lists machines in `nodes` and network edges in `links`:

```json
{
  "nodes": [
    {
      "id": "gpu0", "vendor": "nvidia", "runtime": "max",
      "mem_free_bytes": 16777216,
      "compute_class": 1000000000000.0,
      "mem_bandwidth_bytes_s": 100000000000.0,
      "supports_stage": true, "supports_expert_worker": true, "supports_kv": true,
      "supported_dtypes": ["fp16", "fp8"]
    }
  ],
  "links": []
}
```

Each node records memory capacity, compute throughput, memory bandwidth, role
flags, vendor, runtime, and supported dtypes. The role flags tell the planner
which machines may host pipeline stages, routed experts, and KV cache. See
[Input file reference: Inventory](input-formats.md#inventory).

Links record bandwidth and latency between nodes. Single-node inventories use an
empty `links` list. You can also pass a reusable links file with `--links`; see
[Input file reference: Links](input-formats.md#links).

## Step 3: place the model

```bash
python3 -m fornax plan \
    --target my_target.md \
    --inventory my_fleet.json \
    --out plan.json
```

Exit behavior:

| Exit | Meaning |
|---|---|
| `0` | The model fits. `plan.json` contains the placement. |
| `2` | The model is infeasible. `plan.json` contains `"feasible": false` and an `infeasible_reason`. |

Important plan fields:

| Field | Meaning |
|---|---|
| `feasible` / `infeasible_reason` | Placement verdict and failure reason. |
| `stages` | Pipeline stages, layer ranges, replicas, and placement mode. |
| `expert_placement` | Routed MoE expert placement. Empty for dense models. |
| `predicted` | Cost-model profile: `throughput_tok_s`, `per_request_latency_s`, `bubble_fraction`, and `bottleneck_stage`. |
| `explanations` | Per-stage placement reason and supporting metrics. |

Read `explanations` when a plan surprises you. It records the selected node,
mode, and effective time that drove each stage decision.

## Step 4: simulate performance

```bash
python3 -m fornax simulate --plan plan.json
# simulate: throughput=89214.343 tok/s latency=0.000359s bubble=0.000
```

The output is a cost-model prediction for the placement. Use it to compare
placements and select benchmark candidates. See
[Prediction and measurement scope](concepts.md#prediction-and-measurement-scope).

### Request traces

Pass a request trace to project decode wall time for a specific workload:

```bash
cat > requests.json <<'JSON'
{"requests": [
  {"prompt_len": 128, "gen_len": 256},
  {"prompt_len": 64,  "gen_len": 512}
]}
JSON

python3 -m fornax simulate --plan plan.json --requests requests.json --out sim.json
# simulate: throughput=... requests=2 gen_tokens=768
```

`sim.json` adds a `requests` block with `total_generation_tokens` and
`predicted_decode_wall_time_s`, computed as total generated tokens divided by
predicted throughput. The trace can be a bare JSON list or an object with a
`requests` list. Each request accepts `prompt_len` and `gen_len`, or the aliases
`prompt_tokens` and `max_new_tokens`.

## Step 5: validate acceptance thresholds

If the target contract has a `contract` block, `target validate` plans the model
onto the fleet and checks every threshold:

```bash
python3 -m fornax target validate my_target.md --inventory my_fleet.json
# valid
# invalid: <failed checks>
```

Use this in CI when a target has required thresholds. Exit `0` means every check
passed. Exit `2` means one or more checks failed. Add `--out result.json` to
write the per-check breakdown.

`python3 -m fornax target draft` can render a starting target-contract draft from
a model and inventory and report whether the draft is already valid.

## Step 6: write a preflight bundle

`preflight` runs the planning and contract pipeline and writes a reproducible
artifact directory. The bundle can include the plan, simulation, validation,
calibration or benchmark data, and program reports.

```bash
python3 -m fornax preflight \
    --target my_target.md \
    --out-dir preflight_out \
    --inventory my_fleet.json
```

Inspect a bundle with:

```bash
python3 -m fornax doctor --bundle preflight_out
```

Run `python3 -m fornax preflight --help` for optional calibration, golden-plan,
program-report, and simulated Apple-worker evidence switches.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `plan` exits 2, `infeasible_reason` mentions memory | Resident model weights exceed available fleet memory. Add capacity, add a node, or use a smaller `dtype_weight`. |
| `unsupported vendor` / `unsupported ... dtype` | A node `vendor` is outside `{nvidia, apple, amd, cpu}`, or the dtype is unknown to the cost model. Check spelling against [Input file reference](input-formats.md). |
| `num_layers must match layers length` | The model's `num_layers` disagrees with the length of `layers`. |
| `simulate` prints `infeasible plan: ...` | The plan is infeasible. Fix the placement before simulating. |
| `simulate: invalid request trace` | A request object is missing or malformed. Each entry must be an object with non-negative token counts. |

## See also

- [Input file reference](input-formats.md): fields for every input file.
- [CLI reference](cli-reference.md): commands for inventory, accelerators,
  Apple worker probes, benchmarks, and gate tests.
- [Concepts](concepts.md): stages, experts, predictions, and runtime scope.
