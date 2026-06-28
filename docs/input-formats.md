# Input file reference

The exact schema for every file you hand to Fornax: the **target contract**
(model + serving target, with optional acceptance thresholds), the **inventory**
(the fleet), and the **links** file (network topology). Field types and defaults
are taken from the planner data model (`fornax/planner/model.py`); the planner
validates these on load and raises on violations.

All files are JSON. A target contract may alternatively be Markdown with a fenced
` ```json fornax-target ` block.

---

## Target contract

A target contract bundles a `model` and a `target`, and may add a `contract`
block of acceptance thresholds. `plan` and `simulate` use `model` + `target`;
`target validate` additionally enforces `contract`.

### As JSON

```json
{
  "model":   { … },
  "target":  { … },
  "contract": { … }
}
```

### As Markdown

A `.md` file whose fenced block is tagged `json fornax-target` (the info string
may also be `fornax-target`, `fornax target`, or plain `json`):

````markdown
# My model — target contract

Prose explaining the choices, baselines, kill metric…

```json fornax-target
{ "model": { … }, "target": { … }, "contract": { … } }
```
````

The Markdown form is preferred for real contracts because the rationale lives
next to the numbers. See the runnable
[`v0_target_contract_fixture.md`](../fornax/golden_plans/v0_target_contract_fixture.md).

### Model

The network to serve.

| Field | Type | Required | Notes |
|---|---|---|---|
| `hidden_dim` | int > 0 | yes | model hidden dimension. |
| `num_layers` | int | no | defaults to `len(layers)`; if given, must equal it. |
| `dtype_weight` | str | yes | weight dtype (e.g. `q4`, `fp8`, `fp16`); must be a dtype the cost model knows. |
| `dtype_activation` | str | yes | activation dtype (e.g. `fp16`). |
| `layers` | list | yes | one entry per layer, in order — see below. |

Each entry in `layers` (a **LayerSpec**):

| Field | Type | Default | Notes |
|---|---|---|---|
| `kind` | str | — | layer type, e.g. `attention`, `dense`, `moe`. |
| `weight_bytes` | int | — | resident weight size of the layer. |
| `active_flops_per_token` | int | — | compute per token through this layer. |
| `kv_bytes_per_token` | int | `0` | KV-cache bytes per token (attention layers). |
| `num_experts` | int | `0` | MoE: total experts in the layer. |
| `experts_active` | int | `0` | MoE: experts that fire per token. |
| `expert_bytes` | int | `0` | MoE: bytes per expert. |
| `expert_flops_per_token` | int | `0` | MoE: compute per active expert per token. |
| `shared_expert_bytes` | int | `0` | MoE: bytes of always-on shared experts. |

A dense model leaves the MoE fields at their defaults. A MoE layer sets
`num_experts`/`experts_active`/`expert_bytes` so the planner can split routed
experts across workers (the `expert_placement` in the resulting plan).

### Target

The serving goal.

| Field | Type | Default | Notes |
|---|---|---|---|
| `concurrency` | int > 0 | — | concurrent requests to plan for. |
| `prompt_len` | int > 0 | — | prompt tokens per request. |
| `gen_len` | int > 0 | — | generated tokens per request. |
| `objective` | str | `max_throughput` | one of `max_throughput`, `min_latency`, `balanced`. |
| `remote_expert_wait_slo_s` | float \| null | `null` | optional SLO on remote-expert wait time. |
| `memory_reserve_fraction` | float ≥ 0 | `0.05` | fraction of node memory held back. |
| `fragmentation_margin_fraction` | float ≥ 0 | `0.05` | allocator fragmentation margin. |
| `routing_metadata_bytes_per_token` | float ≥ 0 | `16.0` | per-token routing-metadata overhead. |
| `temp_buffer_fraction` | float ≥ 0 | `0.05` | scratch-buffer reserve. |
| `runtime_reserve_bytes` | int ≥ 0 | `0` | flat per-node runtime reservation. |

The defaults are sensible for a first plan; tune the reserve/margin fractions
when you need the cost model to mirror a specific allocator's behavior.

### Contract (optional, for `target validate`)

Acceptance thresholds checked by `target validate` — e.g.
`throughput_threshold_tok_s`, `memory_headroom_fraction_min`,
`concurrency_sweep`, `persona_min_concurrency`, `baselines`, and a `kill_metric`.
See the `contract` block of
[`v0_target_contract_fixture.md`](../fornax/golden_plans/v0_target_contract_fixture.md)
for a complete, runnable example.

---

## Inventory

The fleet: a list of `nodes` and a list of `links`.

```json
{ "nodes": [ { … } ], "links": [ { … } ] }
```

### Node

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | str | — | unique, non-empty node id. |
| `vendor` | str | — | one of `nvidia`, `apple`, `amd`, `cpu`. |
| `runtime` | str | — | runtime label (e.g. `max`). |
| `mem_free_bytes` | int > 0 | — | memory available to host model state. |
| `compute_class` | float > 0 | — | compute throughput, FLOP/s. |
| `mem_bandwidth_bytes_s` | float > 0 | — | memory bandwidth, bytes/s. |
| `reliability` | float 0–1 | `1.0` | node reliability (used by resilience planning). |
| `supports_stage` | bool | `true` | may host a pipeline stage. |
| `supports_expert_worker` | bool | `false` | may host routed MoE experts. |
| `supports_kv` | bool | `true` | may hold KV cache. |
| `supported_dtypes` | list[str] | `["fp16"]` | dtypes the node can execute. |

The role flags are how you express a heterogeneous fleet: a capacity-rich Mac
might be `supports_expert_worker: true` but `supports_stage: false`, while a fast
GPU hosts the dense stages.

### Links

Network edges between nodes. May live inline under the inventory's `links`, or in
a separate file passed with `--links`.

| Field | Type | Notes |
|---|---|---|
| `a` | str | one endpoint node id. |
| `b` | str | the other endpoint (must differ from `a`). |
| `bandwidth_bytes_s` | float > 0 | link bandwidth, bytes/s. |
| `latency_s` | float ≥ 0 | link latency, seconds. |

A separate links file is either a bare list of link objects or an object with a
`links` key:

```json
{ "links": [
  { "a": "gpu0", "b": "mac0", "bandwidth_bytes_s": 1250000000.0, "latency_s": 0.00005 }
] }
```

```bash
python3 -m fornax plan --target my_target.md \
    --inventory my_fleet.json --links my_topology.json --out plan.json
```

When `--links` is given it supplies (or overrides) the inventory's `links`. For a
single-node fleet, use an empty list.

---

## Validation behavior

Loaders validate eagerly and raise a `ValueError` with a specific message on bad
input — e.g. `hidden_dim must be > 0`, `unsupported vendor: …`, `num_layers must
match layers length`, `link endpoints must be different`. Treat these as schema
errors and fix the file; see [Planning and simulation →
Troubleshooting](planning-and-simulation.md#troubleshooting).
