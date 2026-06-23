# G5 Productization and GA Runbook

Plan ID: `phase5-productization-ga`

operator UX, lifecycle, onboarding, benchmark methodology, and Sponsor GA evidence packaging

Proxy development scope: two local H100 GPUs may stand in for two logical hosts until formal product, lab, and Sponsor evidence is available

## Formal Gate Rule

G5 can pass only with installable product evidence, operator acceptance, benchmark-of-record evidence, and Sponsor GA approval; two-H100 local proxy evidence may pass only the current development proxy gate.

## Scenarios

### operator-config-doctor

prove the operator can prepare and inspect cluster, model, and placement artifacts

Setup:
- prepare cluster.yaml, model.yaml, and placement.json from the operator package
- keep plan-integrity and auth requirements enabled
- write a preflight bundle before lab deployment or Sponsor review

Evidence commands:
- `python3 -m fornax doctor --bundle <preflight-bundle>`
- `python3 -m fornax test ops-lifecycle`

Pass criteria:
- cluster.yaml, model.yaml, and placement.json are documented
- doctor exits cleanly or reports only acknowledged evidence-tier warnings
- operator config validation reports at least two nodes and a serving endpoint

### deploy-upgrade-drain-restart-rollback

prove lifecycle actions are repeatable, audited, and drain before mutation

Setup:
- load the operator lifecycle artifact
- run deploy, upgrade, restart, and rollback actions with request accounting enabled
- preserve event order and active version history

Evidence commands:
- `python3 -m fornax test ops-lifecycle`
- `python3 -m fornax ops lifecycle-simulate --out <ops-lifecycle.json>`

Pass criteria:
- drain_completed precedes each mutation
- dropped_in_flight_total == 0
- rollback_verified is true

### node-replacement

prove replacement preserves identity, placement, and cleanup invariants

Setup:
- drain the old node before removal
- admit a replacement node with managed identity
- verify traffic restoration and final active-node state

Evidence commands:
- `python3 -m fornax test ops-lifecycle`

Pass criteria:
- node_replace_verified is true
- removed node is no longer active
- traffic_restored event is present

### onboarding-handoff

prove operator, developer, benchmark-owner, and reviewer tracks are complete enough for proxy handoff

Setup:
- publish quickstart, operator runbook, developer workflow, benchmark methodology, and glossary
- map each track to first-run commands, success evidence, and escalation paths
- preserve simulation warnings until formal GA evidence exists

Evidence commands:
- `python3 -m fornax test onboarding-methodology`

Pass criteria:
- required tracks are present
- required documents are present
- required glossary terms are present

### benchmark-of-record

prove benchmark methodology requires commands, traces, versions, correctness, environment, logs, and ledger records

Setup:
- capture benchmark commands and raw logs
- attach correctness artifacts before throughput claims
- append each measured run to ledger.jsonl

Evidence commands:
- `python3 -m fornax test benchmark-ledger`
- `python3 -m fornax benchmark --plan placement.json --mode tiny-moe-or-expert-mlp --out benchmark.json`

Pass criteria:
- ledger contains at least one measured record
- required benchmark methodology inputs are defined
- lab-reference boundary remains explicit for formal G5

### sponsor-ga-review

package install, operate, upgrade, serve, and benchmark evidence for Sponsor G5 decision

Setup:
- attach prior G4 proxy or formal packet
- attach productization runbook, onboarding package, lifecycle evidence, and benchmark ledger
- record which formal GA requirements remain deferred

Evidence commands:
- `python3 -m fornax program phase5-ga-gate --ops-artifact <ops> --onboarding-artifact <onboarding> --benchmark-ledger <ledger> --phase4-artifact <phase4> --out <packet.json>`

Pass criteria:
- phase5_proxy_passed may be true only for the local proxy gate
- formal_g5_passed remains false until Sponsor accepts real GA evidence
- all deferred requirements are visible in the packet

## Required Artifacts

- cluster.yaml, model.yaml, and placement.json operator configs
- fornax doctor preflight bundle or equivalent diagnostics
- ops lifecycle artifact
- onboarding package with glossary and benchmark methodology
- benchmark ledger with measured records
- prior G4 resilience gate packet
- Sponsor G5 decision record
