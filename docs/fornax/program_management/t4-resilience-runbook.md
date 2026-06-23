# T4 Resilience and Elasticity Runbook

Plan ID: `phase4-resilience-elasticity`

G4 evidence collection for node loss, replay, added capacity, and operator lifecycle hooks

Proxy development scope: two local H100 GPUs may stand in for two logical hosts until AMD/Mac lab hardware is available

## Formal Gate Rule

G4 can pass only with real lab node-loss and added-node evidence; two-H100 local proxy evidence may pass only the current development proxy gate.

## Scenarios

### single-node-loss-zero-drop

prove in-flight requests survive loss of one serving/stage node

Setup:
- start from a validated placement over at least two logical hosts
- record request ids, plan id, KV/checkpoint owner, and active node ownership before fault injection
- disable new admission to the failed node before replay scheduling if the control plane is reachable

Evidence commands:
- `python3 -m fornax test resilience-replay`
- `python3 -m fornax resilience replay-simulate --out <artifact.json>`

Pass criteria:
- dropped_request_count == 0
- dropped_token_count == 0
- duplicate_token_count == 0
- completed_tokens match reference_tokens for every replayed request

### added-node-scaling

prove adding a replica increases modeled bottleneck-stage throughput without correctness regression

Setup:
- capture baseline bottleneck-stage makespan with one replica
- admit a second logical host as a stage replica
- assign microbatches across both replicas using deterministic scheduling

Evidence commands:
- `python3 -m fornax test stage-replication`
- `python3 -m fornax replication simulate --out <artifact.json>`

Pass criteria:
- replicated_makespan_s < baseline_makespan_s
- speedup >= speedup_floor
- every replica receives work
- max_abs_error <= tolerance

### drain-restart-rollback

prove operator mutations are auditable and drain in-flight work before node changes

Setup:
- load cluster, model, and placement operator configs
- run deploy, upgrade, restart, rollback, and node replacement actions
- record drain and health-check events around every mutation

Evidence commands:
- `python3 -m fornax test ops-lifecycle`
- `python3 -m fornax ops lifecycle-simulate --out <artifact.json>`

Pass criteria:
- drain_completed precedes each mutation
- dropped_in_flight_total == 0
- rollback_verified is true
- node_replace_verified is true

### heterogeneous-lab-followup

turn proxy evidence into formal T4/G4 evidence once AMD and Apple systems are available

Setup:
- replace same-host logical H100 stand-ins with real NVIDIA, AMD, and Apple nodes
- collect node-loss and added-node evidence with production transport and auth enabled
- attach raw artifacts, hardware inventory, operator acceptance, and gate-review decision record

Evidence commands:
- `python3 -m fornax program phase4-resilience-gate --resilience-artifact <real-node-loss.json> --replication-artifact <real-added-node.json> --ops-artifact <real-lifecycle.json> --out <packet.json>`

Pass criteria:
- formal_g4_passed remains false until real heterogeneous evidence is attached
- zero dropped in-flight requests under real single-node loss
- added-node throughput improves on real hardware
- operator signs the G4 gate packet

## Required Artifacts

- resilience replay artifact
- stage replication or added-node scaling artifact
- ops lifecycle artifact
- hardware inventory and topology record
- operator gate decision record
