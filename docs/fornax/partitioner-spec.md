# Fornax — Partitioner Spec (Phase 0)

> The **throughput-optimizing heterogeneous MoE partitioner**: the brain of
> Fornax. Given a quantized MoE, expert-activation traces, and a live inventory
> of heterogeneous nodes + links, produce a layer/stage/expert placement that
> **maximizes steady-state aggregate throughput** under memory and network
> constraints, and predict (throughput, TTFT, per-request latency, remote-expert
> wait) at a target concurrency.

It is **pure logic** — no model, no GPU — so it is unit-testable in the Ignis
fixture style. It also answers the buyer's first question directly: *"will my
pile of hardware run this model, and how fast?"*

---

## 1. Inputs

### 1.1 Model spec (`ModelSpec`)

Per-layer, quantization-aware. For an MoE model:

```
ModelSpec:
  hidden_dim: int                     # activation vector width
  num_layers: int
  layers: [LayerSpec]                 # per-layer, in order
  dtype_weight: enum {q4, q8, fp8, fp16}
  dtype_activation: enum {fp8, fp16}

LayerSpec:
  kind: enum {dense, attention, moe}
  weight_bytes: int                   # dense/shared resident bytes
  active_flops_per_token: int         # compute touched per token (sparse for moe)
  kv_bytes_per_token: int             # 0 for non-attention layers
  num_experts: int                    # moe only
  experts_active: int                 # moe only (top-k)
  expert_bytes: int                   # per routed expert, post-quantization
  expert_flops_per_token: int         # per routed expert MLP
  shared_expert_bytes: int            # if architecture has shared experts

ExpertTrace:                          # optional at first; mandatory for real plans
  layer_id: int
  expert_id: int
  hit_rate_prefill: float             # observed from representative workload
  hit_rate_decode: float
  coactivation: [(expert_id, float)]   # experts commonly selected together
```

### 1.2 Inventory (`Inventory`)

```
Node:
  id: str
  vendor: enum {nvidia, apple, amd, cpu}
  runtime: enum {max, mojo, mlx, llama_cpp, custom}
  mem_free_bytes: int                 # usable for weights + KV + activations
  compute_class: float                # effective FLOP/s for this dtype/op (measured)
  mem_bandwidth_bytes_s: float        # decode/expert MLP can be bandwidth-bound
  reliability: float                  # 0..1, for replica/placement weighting
  supports_stage: bool                # can execute a full layer group
  supports_expert_worker: bool        # can execute routed expert MLP batches
  supports_kv: bool                   # can host KV for attention layers
  supported_dtypes: [dtype]

Link:
  a: node_id; b: node_id
  bandwidth_bytes_s: float            # measured, both directions
  latency_s: float                    # measured RTT/2
# Inventory = {nodes: [Node], links: [Link]}  -> a weighted topology graph
```

All compute/bandwidth/link numbers are **measured by probes**, not nameplate
specs — heterogeneous reality diverges from datasheets.

### 1.3 Workload target (`Target`)

```
Target:
  concurrency: int                    # expected in-flight requests (fills the pipeline)
  prompt_len: int                     # for TTFT / prefill cost
  gen_len: int                        # for per-request latency
  objective: enum {max_throughput, min_latency, balanced}
```

## 2. Output (`PlacementPlan`)

```
Stage:
  index: int
  layers: [layer_id]                  # contiguous group
  replicas: [node_id]                 # >=1; DP width for this stage
  mode: enum {resident, remote_experts, weight_lru}
  expert_hosts: [node_id]             # capacity/expert workers for this stage

PlacementPlan:
  stages: [Stage]                     # ordered pipeline
  boundary_links: [(stage_i, stage_j, link)]   # which physical link each cut uses
  expert_placement: [(layer_id, expert_id, node_id, role)]
                                      # role: hot_resident | warm_remote | cold_store
  predicted:
    throughput_tok_s: float           # steady-state aggregate
    ttft_s: float
    per_request_latency_s: float
    remote_expert_wait_s_per_token: float
    remote_expert_hit_rate_decode: float
    bottleneck_stage: int
    bubble_fraction: float
  feasible: bool
  infeasible_reason: str?             # e.g. "model 412GB > total usable 380GB"
```

## 3. Cost model

The objective is **steady-state aggregate throughput**, gated by the slowest
stage once the pipeline is saturated.

### 3.1 Per-stage compute time (per microbatch, decode)

Decode is bandwidth-bound; prefill is FLOP-bound. For a stage `s` on node `n`
holding layer set `L_s`, batch `b`:

```
t_decode(s)  = sum_{l in L_s} weight_bytes(l) / bw(n)          # weights streamed per token-step
             + b * sum_{l in L_s} kv_bytes_per_token(l) / bw(n) # KV read
t_prefill(s) = (b * prompt_len * sum active_flops(l)) / compute_class(n)
```

For MoE stages, split expert cost into resident local experts and remote expert
execution. The remote path sends hidden states, not whole model layers, and must
include queueing + return traffic:

```
t_remote_expert(s) = sum_remote_expert_batches(
    pack_time
  + (tokens_for_expert * hidden_dim * activation_bytes) / bw(link)
  + remote_queue_wait
  + expert_mlp_time(remote_node, tokens_for_expert)
  + (tokens_for_expert * hidden_dim * activation_bytes) / bw(link)
  + gather_time
)
```

For cold expert *weight fetch* into a GPU LRU, model the one-time load separately
from steady-state execution:

```
t_fetch_weight = E_migrated * expert_bytes / bw(weight_link)
```

The planner should prefer migration when a remote expert becomes hot during
decode; remote execution is for sparse/cold/warm experts, not the common path.

### 3.2 Transfer time per boundary (overlappable)

```
t_xfer(i,j) = (b * hidden_dim * activation_bytes) / bandwidth(link_ij) + latency(link_ij)
```

Effective stage time after 1F1B overlap (transfer hidden behind compute):

```
t_stage(s) = max( t_compute(s), t_xfer(boundary_into_s) ) + exposed_remote_expert_wait(s)
```

If `t_xfer < t_compute` the pipeline-boundary network is **free** (the common,
designed-for case). Remote expert wait is only free when it is overlapped behind
other useful work in the same microbatch window; the cost model must report the
exposed portion explicitly.

### 3.3 Steady-state throughput (with replication)

A stage with `r_s` replicas has effective time `t_stage(s) / r_s`. Pipeline
throughput is set by the slowest *effective* stage:

```
throughput_tok_s = b / max_s( t_stage(s) / r_s )
bottleneck_stage = argmax_s( t_stage(s) / r_s )
bubble_fraction  = 1 - (mean_s t_stage(s)) / (max_s t_stage(s))   # imbalance penalty
```

### 3.4 Latency & TTFT

```
per_request_latency = sum_s t_stage(s) * gen_len   # request traverses all stages each token
ttft = sum_s t_prefill(s) + sum_boundaries t_xfer   # first-token path
```

Per-request latency rises with **pipeline depth** — the irreducible spanning
cost. The `min_latency` / `balanced` objectives penalize depth.

### 3.5 Memory feasibility (hard constraint)

For every stage `s` on node `n` (each replica):

```
sum_{l in L_s} weight_bytes(l) (resident experts)
  + b * sum kv_bytes_per_token(l)
  + activation_buffers
  <= mem_free_bytes(n)
```

`remote_experts` and `weight_lru` relax the resident expert-weight term by
placing cold/warm experts on `expert_hosts` or fetching them into a local cache.
Infeasible if no partition fits → emit `infeasible_reason` with the shortfall.

## 4. Search

The space is: contiguous layer→stage cuts × node assignment × replica count ×
per-stage mode × boundary→link mapping × expert placement. NP-hard in general;
use a staged search:

1. **Feasible depth range.** Min stages = ceil(model_bytes / max usable node mem)
   given quantization; max stages = num_nodes. Iterate depth `P` in range.
2. **Balanced cut (DP).** For each `P`, dynamic-program the contiguous layer
   partition that **minimizes the max per-stage time** (classic
   min-max-load linear partition, weights = per-layer time on the *assigned*
   node) — this is lever 3 (speed-balanced sizing). Assign heavier layer groups
   to faster nodes; bind memory as a hard constraint.
3. **Topology-aware boundary placement.** Map each stage cut onto a physical link
   so the **highest-traffic boundaries land on the fastest links** (Thunderbolt /
   25–100 GbE); low-traffic cuts tolerate slow links. Greedy by descending
   boundary traffic.
4. **Replica allocation.** Spend spare nodes on the bottleneck stage first
   (greedy: replicate `argmax` effective-time stage until nodes exhausted or
   another stage becomes the bottleneck) — lever 4.
5. **Expert placement.** For each MoE stage, place shared/hot experts on the
   stage's hot accelerator first, then assign warm/cold experts to expert workers
   by measured `ExpertTrace`, memory, and link cost. Prefer local migration over
   remote decode hits when the trace shows sustained hotness.
6. **Mode selection.** Compare `resident`, `remote_experts`, and `weight_lru`
   using §3.1 + §3.5 at target batch/concurrency; pick the cheapest feasible mode
   that satisfies the remote-wait SLO.
7. **Score & keep best** by `Target.objective`. Return the winning `PlacementPlan`
   with its `predicted` block.

Start exact-DP for the partition (cheap: O(num_layers² · P)); greedy for replicas
and boundary mapping. Refine later (the structure leaves room for a proper
ILP/beam search without changing the interface).

## 5. Testing (model-free, Ignis-style)

- **Cost-model unit tests:** hand-worked cases where `t_stage`, throughput,
  bubble fraction, and feasibility are computable by hand; assert the planner
  matches.
- **Degenerate cases:** single node (no spanning), homogeneous nodes (even cut),
  one slow node (must get fewer layers / get replicated), model-too-big
  (infeasible with correct shortfall).
- **Monotonicity properties:** adding a node never lowers predicted throughput;
  faster link never raises predicted latency; deeper pipeline never lowers
  per-request latency.
- **Expert placement cases:** traces with one hot expert, uniform random experts,
  changing hotness, and impossible remote-wait SLO; assert migration/offload
  decisions are explainable.
- **Golden plans:** fixture `(ModelSpec, Inventory, Target) → PlacementPlan`
  snapshots, deterministic, checked in. (Mirrors Ignis's deterministic-eval
  discipline — the planner gates, like `make eval`.)

## 6. Interface it schedules onto (`FornaxEngine` / worker contract)

The planner emits plans; the **scheduler** (Phase 1–2) executes them against
workers implementing this contract. Sketch:

```
trait StageWorker:                       # one per node, a Fornax MAX-graph stage
  def load_stage(layers, weights_ref, mode, expert_hosts) raises
  def prefill(microbatch_id, activations_in) raises -> ActivationsOut
  def decode_step(microbatch_id, activations_in) raises -> ActivationsOut
  def kv_evict(prefix_id) raises
  def health() raises -> NodeStats      # live compute/bw/mem -> feeds Inventory probes

trait ExpertWorker:                      # Apple/AMD/NVIDIA expert executor
  def load_experts(layer_id, expert_ids, weights_ref, dtype) raises
  def run_expert_batch(batch_id, layer_id, expert_id, activations) raises -> ActivationsOut
  def migrate_expert(layer_id, expert_id, dst_node_id) raises
  def health() raises -> NodeStats

# FornaxEngine fans a request across the stage pipeline, drives 1F1B microbatch
# scheduling, activation handoff, and local/remote expert dispatch. Ignis can call
# it later through an Engine-trait shim, but this contract belongs to the engine.
```

`StageWorker.health()` closes the loop: live stats re-probe the `Inventory` so the
planner can re-plan under drift (a node heats up, a link degrades, a node leaves).

---

**Phase 0 deliverable:** this planner as a standalone, model-free module with the
golden-plan fixtures green — runnable and reviewable before any GPU is touched.
