# Fornax Package Map

This package contains the machine-checkable Fornax development artifacts. Keep this map aligned with the program management work breakdown in `docs/fornax/program_management/02-work-breakdown-structure.md` and the sprint backlog in `docs/fornax/program_management/sprints/`.

| Area | Modules | WBS stream | Owning review lenses |
|---|---|---|---|
| Planner and cost model | `planner/`, `scheduler.py`, `throughput_scaling.py` | DIST, API, PERF | Analytical, Hardware, LLM Expertise |
| Target contracts and runtime format | `target_contract.py`, `contracts.py`, `runtime_format.py`, `runtime_format_spec.py`, `validation.py` | API, CORR | Low-level Software, Documentation |
| Inventory and calibration | `inventory/`, `calibration.py`, `preflight.py`, `doctor.py` | HW, OPS | Hardware, Hardware Acceleration, System Engineering |
| Transport, trust, and lifecycle contracts | `transport.py`, `network_contract.py`, `network_security_spec.py`, `trust_boundary.py`, `state_ownership.py`, `ops_lifecycle.py` | DIST, SEC, OPS | Networking, Security, System Engineering |
| Engine and serving surfaces | `engine_seam.py`, `engine_simulation.py`, `serving.py`, `local_serving_smoke.py`, `local_http_serving_smoke.py` | API, DIST, OPS | System Engineering, Networking, LLM Expertise |
| Accelerator and parity probes | `accelerator_probe.py`, `pipeline_probe.py`, `remote_expert_probe.py`, `moe_parity.py`, `target_fixture_probe.py`, `apple_probe.py` | HW, PERF, CORR | Hardware Acceleration, Low-level Software |
| MoE behavior and migration | `moe.py`, `moe_migration.py`, `stage_host.py`, `stage_replication.py`, `continuous_batching.py` | MOE, DIST, PERF | LLM Expertise, Analytical, System Engineering |
| Resilience and phase gates | `resilience.py`, `phase3_proxy_gate.py`, `phase4_resilience_gate.py`, `phase5_ga_gate.py`, `g1_evidence_packet.py`, `g1_review.py`, `phase0_status.py`, `phase0_simulated_validation.py`, `t1_simulated_validation.py` | PM, OPS, DIST | Program Management, System Engineering, Documentation |
| Governance, onboarding, and reporting | `program_governance.py`, `program_rebaseline.py`, `onboarding.py`, `benchmark.py`, `benchmark_ledger.py`, `metrics_ledger.py`, `observability.py`, `trace_ledger.py` | PM, OPS, PERF | Program Management, Documentation, System Engineering |
| CLI and golden runners | `cli.py`, `__main__.py`, `golden.py`, `simulate.py`, `golden_plans/`, `golden_vectors/` | QA, OPS | Software Engineering, Documentation |

## Boundaries

The package is still simulation-first. Local two-H100 logical-host evidence may close the current development proxy gates, but it must not be described as formal AMD/Mac, production transport, or real frontier-model evidence unless the corresponding artifact proves that scope.

`cli.py` and `local_http_serving_smoke.py` remain intentionally listed as CLI/integration surfaces here, but the current code review calls them out as refactor candidates. Prefer moving new code into smaller domain modules and keeping these files as thin dispatch/integration layers.
