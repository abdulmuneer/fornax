# Planning and simulation

The core Fornax workflow, in depth. By the end you can describe your own model
and fleet, place the model, predict its performance, validate it against a
target, and bundle the evidence. If you have not run the toy example yet, do
[Getting started](getting-started.md) first.

```text
  target contract ──┐
                    ├─►  plan  ──►  plan.json  ──►  simulate  ──►  prediction
  inventory  ───────┘                    │
                                         └──►  target validate  ──►  valid / invalid
```

## Step 1 — describe the model and the serving target

The model and the serving goal travel together in a **target contract**. It can
be a JSON file, or a Markdown file with a fenced ` ```json fornax-target ` block
(handy because the same file can hold prose explaining the choices). Minimal
JSON form:

```json
{
  "model": {
    "hidden_dim": 1024,
    "num_layers": 2,
    "dtype_weight": "q4",
    "dtype_activation": "fp16",
    "layers": [
      { "kind": "attention", "weight_bytes": 1048576, "active_flops_per_token": 2000000, "kv_bytes_per_token": 8192 },
      { "kind": "dense",     "weight_bytes": 1048576, "active_flops_per_token": 2000000 }
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

- `layers` is the heart of the model description — one entry per layer, each with
  its memory (`weight_bytes`), compute (`active_flops_per_token`), and KV cost
  (`kv_bytes_per_token`). MoE layers add `num_experts`, `experts_active`,
  `expert_bytes`, and friends. Full field list: [Input file reference →
  Model](input-formats.md#model).
- `objective` is one of `max_throughput`, `min_latency`, `balanced`.

See the bundled [`v0_target_contract_fixture.md`](../fornax/golden_plans/v0_target_contract_fixture.md)
for a complete, runnable Markdown example (it also carries a `contract` block of
acceptance thresholds — used by `target validate`, step 4).

## Step 2 — describe the fleet (inventory)

An **inventory** lists the machines (`nodes`) and the network between them
(`links`):

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

Each node carries its **capacity** (`mem_free_bytes`), **compute**
(`compute_class`, FLOP/s), **bandwidth** (`mem_bandwidth_bytes_s`), and **role
flags** (`supports_stage`, `supports_expert_worker`, `supports_kv`) — these tell
the planner what each machine is allowed to host. `vendor` is one of `nvidia`,
`apple`, `amd`, `cpu`. Full field list: [Input file reference →
Inventory](input-formats.md#inventory).

**Links** describe bandwidth and latency between nodes; for a single node, leave
`links` empty. You can also keep links in a separate file and pass it with
`--links` (see [Input file reference → Links](input-formats.md#links)) — useful
when you reuse one fleet topology across several inventories.

## Step 3 — place the model (`plan`)

```bash
python3 -m fornax plan \
    --target my_target.md \
    --inventory my_fleet.json \
    --out plan.json
```

- Exit `0` and `wrote placement plan` → the model fits; `plan.json` holds the
  placement.
- Exit `2` → infeasible; `plan.json` has `"feasible": false` and an
  `infeasible_reason`.

What `plan.json` contains:

| Field | Meaning |
|---|---|
| `feasible` / `infeasible_reason` | the verdict, and why if it failed |
| `stages` | the pipeline: each stage's `layers`, hosting `replicas`, and `mode` |
| `expert_placement` | where routed MoE experts are placed (empty for dense models) |
| `predicted` | the cost-model profile: `throughput_tok_s`, `per_request_latency_s`, `bubble_fraction`, `bottleneck_stage` |
| `explanations` | per-stage human-readable `reason` plus the metrics behind the decision |

The `explanations` block is the one to read when a plan surprises you — it states,
per stage, which node was chosen, the mode, and the effective time that drove the
choice.

## Step 4 — predict performance (`simulate`)

```bash
python3 -m fornax simulate --plan plan.json
# simulate: throughput=89214.343 tok/s latency=0.000359s bubble=0.000
```

These are **cost-model predictions for the placement, not hardware benchmarks**
(see [the honest constraint](concepts.md#the-honest-constraint)). Use them to
compare placements and decide what is worth building.

### Against a real request mix

Pass a **request trace** — a list of `{prompt_len, gen_len}` objects — to project
a wall-clock decode time for a specific workload:

```bash
cat > requests.json <<'JSON'
{"requests": [
  {"prompt_len": 128, "gen_len": 256},
  {"prompt_len": 64,  "gen_len": 512}
]}
JSON

python3 -m fornax simulate --plan plan.json --requests requests.json --out sim.json
# simulate: throughput=… requests=2 gen_tokens=768
```

`sim.json` adds a `requests` block with `total_generation_tokens` and a
`predicted_decode_wall_time_s` (= total generated tokens ÷ predicted throughput).
The trace may be a bare JSON list or an object with a `requests` list; each
request accepts `prompt_len`/`gen_len` (or the aliases `prompt_tokens` /
`max_new_tokens`).

## Step 5 — validate against acceptance thresholds (`target validate`)

If your target contract carries a `contract` block (a throughput floor, memory
headroom, a concurrency sweep), `target validate` plans the model onto the fleet
and checks every threshold:

```bash
python3 -m fornax target validate my_target.md --inventory my_fleet.json
# valid        (exit 0)   — or — invalid: <failed checks>   (exit 2)
```

This is the CI gate for a target: it fails loudly if a planner change or a fleet
change drops a placement below its promised thresholds. Add `--out result.json`
to capture the per-check breakdown.

Need a starting point for the contract block? `python3 -m fornax target draft`
renders a target-contract draft from a model/inventory and reports whether it is
already valid.

## Step 6 — bundle the evidence (`preflight`)

`preflight` runs the planning/contract pipeline and writes a directory of
artifacts (the plan, simulation, validation, and optional calibration/benchmark
and program reports) — a single reproducible bundle you can hand to a reviewer or
archive with a decision:

```bash
python3 -m fornax preflight --target my_target.md --out-dir preflight_out \
    --inventory my_fleet.json
```

Inspect a produced bundle with `python3 -m fornax doctor --bundle preflight_out`.
`preflight` has many optional switches (calibration, golden plans, program
reports, simulated Apple-worker evidence); run `python3 -m fornax preflight
--help` for the full set.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `plan` exits 2, `infeasible_reason` mentions memory | the model's resident weight bytes exceed the fleet's `mem_free_bytes`; add capacity or a node, or use a smaller `dtype_weight`. |
| `unsupported vendor` / `unsupported … dtype` | a node `vendor` outside `{nvidia, apple, amd, cpu}`, or a `dtype` the cost model doesn't know — check spelling against [Input file reference](input-formats.md). |
| `num_layers must match layers length` | the model's `num_layers` disagrees with the length of its `layers` array. |
| `simulate` prints `infeasible plan: …` | you passed an infeasible plan; fix the placement (step 3) before simulating. |
| `simulate: invalid request trace` | a request object is missing/!malformed; each entry must be an object with non-negative token counts. |

## See also

- [Input file reference](input-formats.md) — every field of every input file.
- [CLI reference](cli-reference.md) — the rest of the command surface
  (`inventory`, `accelerator`, `apple`, `benchmark`, the `test` gates, …).
- [Concepts](concepts.md) — stages, experts, and what the predicted numbers mean.
