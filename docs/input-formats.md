# Input file reference

This page documents the files passed to Fornax: target contracts, inventories,
and links files. Field types and defaults come from the planner data model in
`fornax/planner/model.py`. Loaders validate input on read and raise `ValueError`
with a specific message for invalid files.

All files are JSON. A target contract can also be Markdown with a fenced
`json fornax-target` block.

## Target contract

A target contract bundles a `model` and a `target`. It can also include an
optional `contract` block with acceptance thresholds. `plan` and `simulate` use
`model` and `target`; `target validate` also checks `contract`.

### JSON form

```json
{
  "model": { "...": "..." },
  "target": { "...": "..." },
  "contract": { "...": "..." }
}
```

### Markdown form

A `.md` file can contain a fenced block tagged `json fornax-target`. Accepted info
strings are `json fornax-target`, `fornax-target`, `fornax target`, and `json`.

````markdown
# My target contract

Rationale, baselines, and acceptance thresholds.

```json fornax-target
{ "model": { "...": "..." }, "target": { "...": "..." }, "contract": { "...": "..." } }
```
````

The Markdown form keeps rationale next to the numeric target. See the runnable
[`v0_target_contract_fixture.md`](../fornax/golden_plans/v0_target_contract_fixture.md).

### Model

The network to serve.

| Field | Type | Required | Notes |
|---|---|---|---|
| `hidden_dim` | int > 0 | yes | Model hidden dimension. |
| `num_layers` | int | no | Defaults to `len(layers)`. If supplied, it must equal that length. |
| `dtype_weight` | str | yes | Weight dtype, such as `q4`, `fp8`, or `fp16`. Must be known to the cost model. |
| `dtype_activation` | str | yes | Activation dtype, such as `fp16`. |
| `layers` | list | yes | One entry per layer, in order. |

Layer entry fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `kind` | str | required | Layer type, such as `attention`, `dense`, or `moe`. |
| `weight_bytes` | int | required | Resident weight size of the layer. |
| `active_flops_per_token` | int | required | Compute per token through this layer. |
| `kv_bytes_per_token` | int | `0` | KV-cache bytes per token for attention layers. |
| `num_experts` | int | `0` | MoE total experts in the layer. |
| `experts_active` | int | `0` | MoE experts activated per token. |
| `expert_bytes` | int | `0` | Bytes per expert. |
| `expert_flops_per_token` | int | `0` | Compute per active expert per token. |
| `shared_expert_bytes` | int | `0` | Bytes of always-on shared experts. |

Dense models leave the MoE fields at their defaults. MoE layers set
`num_experts`, `experts_active`, and `expert_bytes` so the planner can place
routed experts in the resulting plan.

### Target

The serving goal.

| Field | Type | Default | Notes |
|---|---|---|---|
| `concurrency` | int > 0 | required | Concurrent requests to plan for. |
| `prompt_len` | int > 0 | required | Prompt tokens per request. |
| `gen_len` | int > 0 | required | Generated tokens per request. |
| `objective` | str | `max_throughput` | One of `max_throughput`, `min_latency`, or `balanced`. |
| `remote_expert_wait_slo_s` | float or null | `null` | Optional SLO on remote-expert wait time. |
| `memory_reserve_fraction` | float >= 0 | `0.05` | Fraction of node memory held back. |
| `fragmentation_margin_fraction` | float >= 0 | `0.05` | Allocator fragmentation margin. |
| `routing_metadata_bytes_per_token` | float >= 0 | `16.0` | Per-token routing metadata overhead. |
| `temp_buffer_fraction` | float >= 0 | `0.05` | Scratch-buffer reserve. |
| `runtime_reserve_bytes` | int >= 0 | `0` | Flat per-node runtime reservation. |

The defaults are suitable for a first plan. Tune the reserve and margin fractions
when you need the cost model to match a specific allocator.

### Contract

Optional acceptance thresholds checked by `target validate`. Examples include
`throughput_threshold_tok_s`, `memory_headroom_fraction_min`,
`concurrency_sweep`, `persona_min_concurrency`, `baselines`, and `kill_metric`.
See the `contract` block of
[`v0_target_contract_fixture.md`](../fornax/golden_plans/v0_target_contract_fixture.md)
for a complete runnable example.

## Inventory

An inventory describes the fleet with `nodes` and `links`.

```json
{ "nodes": [ { "...": "..." } ], "links": [ { "...": "..." } ] }
```

### Node

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | str | required | Unique non-empty node id. |
| `vendor` | str | required | One of `nvidia`, `apple`, `amd`, or `cpu`. |
| `runtime` | str | required | Runtime label, such as `max`. |
| `mem_free_bytes` | int > 0 | required | Memory available for model state. |
| `compute_class` | float > 0 | required | Compute throughput in FLOP/s. |
| `mem_bandwidth_bytes_s` | float > 0 | required | Memory bandwidth in bytes/s. |
| `reliability` | float 0..1 | `1.0` | Node reliability for resilience planning. |
| `supports_stage` | bool | `true` | Node may host a pipeline stage. |
| `supports_expert_worker` | bool | `false` | Node may host routed MoE experts. |
| `supports_kv` | bool | `true` | Node may hold KV cache. |
| `supported_dtypes` | list[str] | `["fp16"]` | Dtypes the node can execute. |

Use role flags to describe heterogeneous fleets. For example, a capacity-rich Mac
can be an expert worker while a fast GPU hosts dense stages.

### Links

Links describe network edges between nodes. They can live inline under the
inventory's `links` key or in a separate file passed with `--links`.

| Field | Type | Notes |
|---|---|---|
| `a` | str | One endpoint node id. |
| `b` | str | Other endpoint node id. It must differ from `a`. |
| `bandwidth_bytes_s` | float > 0 | Link bandwidth in bytes/s. |
| `latency_s` | float >= 0 | Link latency in seconds. |

A separate links file can be a bare list of link objects or an object with a
`links` key:

```json
{ "links": [
  { "a": "gpu0", "b": "mac0", "bandwidth_bytes_s": 1250000000.0, "latency_s": 0.00005 }
] }
```

```bash
python3 -m fornax plan \
    --target my_target.md \
    --inventory my_fleet.json \
    --links my_topology.json \
    --out plan.json
```

When `--links` is supplied, it supplies or overrides the inventory's inline
links. Use an empty list for a single-node fleet.

## Validation behavior

Loaders validate eagerly and raise `ValueError` with specific messages such as
`hidden_dim must be > 0`, `unsupported vendor: ...`, `num_layers must match
layers length`, and `link endpoints must be different`.

Treat these as schema errors and fix the file. See
[Planning and simulation: Troubleshooting](planning-and-simulation.md#troubleshooting).
