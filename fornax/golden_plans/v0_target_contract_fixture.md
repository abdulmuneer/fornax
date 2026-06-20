# V0 Target Contract Fixture

This is a small executable markdown fixture for the Phase-0 contract parser. It is
not a real target contract.

```json fornax-target
{
  "model": {
    "hidden_dim": 1024,
    "num_layers": 2,
    "dtype_weight": "q4",
    "dtype_activation": "fp16",
    "layers": [
      {
        "kind": "attention",
        "weight_bytes": 1048576,
        "active_flops_per_token": 2000000,
        "kv_bytes_per_token": 8192
      },
      {
        "kind": "dense",
        "weight_bytes": 1048576,
        "active_flops_per_token": 2000000
      }
    ]
  },
  "target": {
    "concurrency": 4,
    "prompt_len": 16,
    "gen_len": 8,
    "objective": "max_throughput"
  },
  "contract": {
    "seed_target_rationale": "Accepts the project-plan seed for parser and validator testing only.",
    "throughput_threshold_tok_s": 1000.0,
    "memory_headroom_fraction_min": 0.05,
    "concurrency_sweep": [
      4,
      8,
      16,
      32
    ],
    "persona_min_concurrency": 4,
    "persona_can_supply_concurrency": true,
    "baselines": [
      {
        "name": "single-node-fit",
        "status": "fixture-only"
      },
      {
        "name": "naive-sequential-pipeline",
        "status": "fixture-only"
      }
    ],
    "kill_metric": "Fixture is invalid if the planner cannot produce a feasible plan above threshold."
  }
}
```
