from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .io import read_json
from .ops_lifecycle import validate_ops_lifecycle_fixture
from .resilience import validate_resilience_replay_fixture
from .stage_replication import validate_stage_replication_fixture

RECORD_KIND = "phase4-g4-resilience-proxy-gate"
GATE = "G4"
MODE = "two-h100-local-proxy"
VALID_OUTCOMES = {"PROCEED", "ITERATE", "NARROW", "KILL"}

REQUIRED_RUNBOOK_SCENARIOS = {
    "single-node-loss-zero-drop",
    "added-node-scaling",
    "drain-restart-rollback",
    "heterogeneous-lab-followup",
}

DEFERRED_REQUIREMENTS = [
    {
        "id": "real-single-node-loss",
        "status": "deferred",
        "reason": "current evidence replays deterministic in-flight requests in simulation; real node power/network loss remains T4 lab work",
    },
    {
        "id": "real-added-node-scaling",
        "status": "deferred",
        "reason": "current added-node evidence is simulated stage replication over two local logical hosts",
    },
    {
        "id": "real-amd-gpu-node",
        "status": "deferred",
        "reason": "AMD GPU node is not present on this development machine",
    },
    {
        "id": "real-apple-silicon-mac",
        "status": "deferred",
        "reason": "Apple Silicon Mac is not present on this development machine",
    },
    {
        "id": "production-drain-restart-hooks",
        "status": "deferred",
        "reason": "operator lifecycle hooks are simulated and auditable but not wired to a production deployment",
    },
]


def _check(name: str, ok: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "evidence": evidence}


def _load_fixture(path: str | Path) -> tuple[Path, dict[str, Any]]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    data = read_json(fixture_path)
    if not isinstance(data, dict):
        raise ValueError(f"{fixture_path} must contain a JSON object")
    return fixture_path, data



def _normalize_string_list(value: list[str] | None, *, default: list[str], field: str) -> list[str]:
    raw = default if value is None else value
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _validate_string_list(
    value: Any,
    field: str,
    errors: list[str],
    *,
    min_count: int = 2,
) -> list[str]:
    if not isinstance(value, list) or len(value) < min_count:
        errors.append(f"{field} must contain at least {min_count} entries")
        return []
    result: list[str] = []
    valid = True
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
            valid = False
        else:
            result.append(item)
    if not valid:
        return []
    if len(set(result)) != len(result):
        errors.append(f"{field} must contain unique non-empty strings")
        return []
    return result


def _validate_artifact_paths(
    packet: dict[str, Any],
    required_names: tuple[str, ...],
    errors: list[str],
) -> dict[str, str]:
    artifacts = packet.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
        return {}
    result: dict[str, str] = {}
    for name in required_names:
        value = artifacts.get(name)
        if not isinstance(value, str) or not value:
            errors.append(f"artifacts.{name} must be a non-empty string")
        else:
            result[name] = value
    return result


def _proxy_hardware_record(
    *,
    proxy_hardware_name: str,
    proxy_devices: list[str] | None,
    proxy_logical_hosts: list[str] | None,
) -> dict[str, Any]:
    if not isinstance(proxy_hardware_name, str) or not proxy_hardware_name:
        raise ValueError("proxy_hardware_name must be a non-empty string")
    selected_devices = _normalize_string_list(
        proxy_devices,
        default=["cuda:0", "cuda:1"],
        field="proxy_devices",
    )
    logical_hosts = _normalize_string_list(
        proxy_logical_hosts,
        default=["logical-host-0", "logical-host-1"],
        field="proxy_logical_hosts",
    )
    return {
        "hardware_name": proxy_hardware_name,
        "selected_devices": selected_devices,
        "logical_hosts": logical_hosts,
        "proxy_mode": MODE,
        "evidence_tier": "local-proxy",
        "formal_lab_evidence": False,
        "note": "Selected local H100 devices stand in for separate logical hosts during proxy development.",
    }

def build_phase4_t4_runbook(*, plan_id: str = "phase4-resilience-elasticity") -> dict[str, Any]:
    scenarios = [
        {
            "id": "single-node-loss-zero-drop",
            "goal": "prove in-flight requests survive loss of one serving/stage node",
            "setup": [
                "start from a validated placement over at least two logical hosts",
                "record request ids, plan id, KV/checkpoint owner, and active node ownership before fault injection",
                "disable new admission to the failed node before replay scheduling if the control plane is reachable",
            ],
            "evidence_commands": [
                "python3 -m fornax test resilience-replay",
                "python3 -m fornax resilience replay-simulate --out <artifact.json>",
            ],
            "pass_criteria": [
                "dropped_request_count == 0",
                "dropped_token_count == 0",
                "duplicate_token_count == 0",
                "completed_tokens match reference_tokens for every replayed request",
            ],
        },
        {
            "id": "added-node-scaling",
            "goal": "prove adding a replica increases modeled bottleneck-stage throughput without correctness regression",
            "setup": [
                "capture baseline bottleneck-stage makespan with one replica",
                "admit a second logical host as a stage replica",
                "assign microbatches across both replicas using deterministic scheduling",
            ],
            "evidence_commands": [
                "python3 -m fornax test stage-replication",
                "python3 -m fornax replication simulate --out <artifact.json>",
            ],
            "pass_criteria": [
                "replicated_makespan_s < baseline_makespan_s",
                "speedup >= speedup_floor",
                "every replica receives work",
                "max_abs_error <= tolerance",
            ],
        },
        {
            "id": "drain-restart-rollback",
            "goal": "prove operator mutations are auditable and drain in-flight work before node changes",
            "setup": [
                "load cluster, model, and placement operator configs",
                "run deploy, upgrade, restart, rollback, and node replacement actions",
                "record drain and health-check events around every mutation",
            ],
            "evidence_commands": [
                "python3 -m fornax test ops-lifecycle",
                "python3 -m fornax ops lifecycle-simulate --out <artifact.json>",
            ],
            "pass_criteria": [
                "drain_completed precedes each mutation",
                "dropped_in_flight_total == 0",
                "rollback_verified is true",
                "node_replace_verified is true",
            ],
        },
        {
            "id": "heterogeneous-lab-followup",
            "goal": "turn proxy evidence into formal T4/G4 evidence once AMD and Apple systems are available",
            "setup": [
                "replace same-host logical H100 stand-ins with real NVIDIA, AMD, and Apple nodes",
                "collect node-loss and added-node evidence with production transport and auth enabled",
                "attach raw artifacts, hardware inventory, operator acceptance, and gate-review decision record",
            ],
            "evidence_commands": [
                "python3 -m fornax program phase4-resilience-gate --resilience-artifact <real-node-loss.json> --replication-artifact <real-added-node.json> --ops-artifact <real-lifecycle.json> --out <packet.json>",
            ],
            "pass_criteria": [
                "formal_g4_passed remains false until real heterogeneous evidence is attached",
                "zero dropped in-flight requests under real single-node loss",
                "added-node throughput improves on real hardware",
                "operator signs the G4 gate packet",
            ],
        },
    ]
    return {
        "version": 1,
        "plan_id": plan_id,
        "title": "T4 Resilience and Elasticity Runbook",
        "scope": "G4 evidence collection for node loss, replay, added capacity, and operator lifecycle hooks",
        "proxy_development_scope": "two local H100 GPUs may stand in for two logical hosts until AMD/Mac lab hardware is available",
        "scenarios": scenarios,
        "required_artifacts": [
            "resilience replay artifact",
            "stage replication or added-node scaling artifact",
            "ops lifecycle artifact",
            "hardware inventory and topology record",
            "operator gate decision record",
        ],
        "formal_gate_rule": "G4 can pass only with real lab node-loss and added-node evidence; two-H100 local proxy evidence may pass only the current development proxy gate.",
    }


def render_phase4_t4_runbook_markdown(runbook: dict[str, Any]) -> str:
    lines = [
        "# T4 Resilience and Elasticity Runbook",
        "",
        f"Plan ID: `{runbook.get('plan_id')}`",
        "",
        str(runbook.get("scope", "")),
        "",
        f"Proxy development scope: {runbook.get('proxy_development_scope')}",
        "",
        "## Formal Gate Rule",
        "",
        str(runbook.get("formal_gate_rule", "")),
        "",
        "## Scenarios",
    ]
    for scenario in runbook.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        lines.extend([
            "",
            f"### {scenario.get('id')}",
            "",
            str(scenario.get("goal", "")),
            "",
            "Setup:",
        ])
        lines.extend(f"- {item}" for item in scenario.get("setup", []))
        lines.extend(["", "Evidence commands:"])
        lines.extend(f"- `{item}`" for item in scenario.get("evidence_commands", []))
        lines.extend(["", "Pass criteria:"])
        lines.extend(f"- {item}" for item in scenario.get("pass_criteria", []))
    lines.extend(["", "## Required Artifacts", ""])
    lines.extend(f"- {item}" for item in runbook.get("required_artifacts", []))
    lines.append("")
    return "\n".join(lines)


def build_phase4_resilience_gate_packet(
    resilience_artifact: str | Path,
    replication_artifact: str | Path,
    ops_artifact: str | Path,
    *,
    packet_date: str | None = None,
    outcome: str = "PROCEED",
    accepted_by: str = "operator",
    decision: str = "Use two local H100 GPUs as logical hosts for the current Phase 4 resilience/elasticity proxy gate; defer formal heterogeneous T4 validation.",
    proxy_hardware_name: str = "NVIDIA H100 80GB HBM3",
    proxy_devices: list[str] | None = None,
    proxy_logical_hosts: list[str] | None = None,
) -> dict[str, Any]:
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}")
    resilience_path, resilience = _load_fixture(resilience_artifact)
    replication_path, replication = _load_fixture(replication_artifact)
    ops_path, ops = _load_fixture(ops_artifact)

    resilience_validation = validate_resilience_replay_fixture(resilience)
    replication_validation = validate_stage_replication_fixture(replication)
    ops_validation = validate_ops_lifecycle_fixture(ops)

    resilience_summary = resilience.get("summary") if isinstance(resilience.get("summary"), dict) else {}
    replication_summary = replication.get("summary") if isinstance(replication.get("summary"), dict) else {}
    replication_result = replication.get("result") if isinstance(replication.get("result"), dict) else {}
    ops_summary = ops.get("summary") if isinstance(ops.get("summary"), dict) else {}
    ops_accounting = ops.get("request_accounting") if isinstance(ops.get("request_accounting"), dict) else {}
    runbook = build_phase4_t4_runbook()
    proxy_hardware = _proxy_hardware_record(
        proxy_hardware_name=proxy_hardware_name,
        proxy_devices=proxy_devices,
        proxy_logical_hosts=proxy_logical_hosts,
    )
    h100_proxy_scope = (
        "H100" in proxy_hardware["hardware_name"]
        and len(proxy_hardware["selected_devices"]) >= 2
        and len(proxy_hardware["logical_hosts"]) >= 2
        and proxy_hardware["formal_lab_evidence"] is False
    )
    runbook_scenarios = {
        scenario.get("id")
        for scenario in runbook.get("scenarios", [])
        if isinstance(scenario, dict)
    }

    evidence_checks = [
        _check(
            "resilience-artifact-valid",
            resilience_validation.get("ok") is True,
            f"validate_resilience_replay_fixture({resilience_path})",
        ),
        _check(
            "zero-dropped-in-flight",
            resilience_summary.get("zero_dropped_in_flight") is True
            and resilience_summary.get("dropped_request_count") == 0
            and resilience_summary.get("dropped_token_count") == 0
            and resilience_summary.get("duplicate_token_count") == 0
            and resilience_summary.get("correctness_passed") is True,
            "single-node-loss replay completes every in-flight request without dropped or duplicate tokens",
        ),
        _check(
            "added-node-scaling",
            replication_validation.get("ok") is True
            and replication_summary.get("replica_count", 0) >= 2
            and replication_summary.get("speedup_passed") is True
            and replication_summary.get("correctness_passed") is True
            and replication_result.get("replicated_makespan_s", 0) < replication_result.get("baseline_makespan_s", 0),
            "stage replication uses added capacity, improves modeled makespan, and preserves output parity",
        ),
        _check(
            "drain-restart-rollback-hooks",
            ops_validation.get("ok") is True
            and ops_summary.get("dropped_in_flight_count") == 0
            and ops_summary.get("rollback_verified") is True
            and ops_summary.get("node_replace_verified") is True
            and ops_accounting.get("dropped_in_flight_total") == 0,
            "operator deploy, drain, restart, rollback, and node replacement hooks are auditable",
        ),
        _check(
            "t4-runbook-defined",
            REQUIRED_RUNBOOK_SCENARIOS.issubset(runbook_scenarios)
            and len(runbook.get("required_artifacts", [])) >= 5,
            "runbook defines node-loss, added-node, lifecycle, and heterogeneous-lab follow-up evidence collection",
        ),
        _check(
            "two-h100-proxy-scope",
            h100_proxy_scope
            and resilience.get("mode") == "t1-simulation"
            and replication.get("mode") == "t1-simulation"
            and ops.get("mode") == "t1-simulation",
            "selected local H100 devices and logical hosts are recorded for the simulation/proxy scope",
        ),
        _check(
            "no-formal-g4-overclaim",
            True,
            "formal_g4_passed remains false until real T4 node-loss and added-node evidence is attached",
        ),
    ]
    proxy_passed = outcome == "PROCEED" and all(item["ok"] for item in evidence_checks)
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "gate": GATE,
        "mode": MODE,
        "date": packet_date or date.today().isoformat(),
        "outcome": outcome,
        "accepted_by": accepted_by,
        "decision": decision,
        "phase4_proxy_passed": proxy_passed,
        "formal_g4_passed": False,
        "formal_g4_validation_deferred": True,
        "artifacts": {
            "resilience_replay": str(resilience_path),
            "stage_replication": str(replication_path),
            "ops_lifecycle": str(ops_path),
        },
        "proxy_hardware": proxy_hardware,
        "artifact_validation": {
            "resilience_replay": {
                "artifact": str(resilience_path),
                "ok": resilience_validation.get("ok") is True,
                "errors": list(resilience_validation.get("errors", [])),
                "warnings": list(resilience_validation.get("warnings", [])),
            },
            "stage_replication": {
                "artifact": str(replication_path),
                "ok": replication_validation.get("ok") is True,
                "errors": list(replication_validation.get("errors", [])),
                "warnings": list(replication_validation.get("warnings", [])),
            },
            "ops_lifecycle": {
                "artifact": str(ops_path),
                "ok": ops_validation.get("ok") is True,
                "errors": list(ops_validation.get("errors", [])),
                "warnings": list(ops_validation.get("warnings", [])),
            },
        },
        "evidence_checks": evidence_checks,
        "runbook": runbook,
        "deferred_requirements": DEFERRED_REQUIREMENTS,
        "summary": {
            "check_count": len(evidence_checks),
            "passed_count": sum(1 for item in evidence_checks if item["ok"]),
            "in_flight_request_count": resilience_summary.get("in_flight_request_count"),
            "dropped_request_count": resilience_summary.get("dropped_request_count"),
            "dropped_token_count": resilience_summary.get("dropped_token_count"),
            "duplicate_token_count": resilience_summary.get("duplicate_token_count"),
            "zero_dropped_in_flight": resilience_summary.get("zero_dropped_in_flight") is True,
            "replica_count": replication_summary.get("replica_count"),
            "replication_speedup": replication_summary.get("speedup"),
            "replication_correctness_passed": replication_summary.get("correctness_passed") is True,
            "lifecycle_action_count": ops_summary.get("action_count"),
            "lifecycle_dropped_in_flight_count": ops_summary.get("dropped_in_flight_count"),
            "rollback_verified": ops_summary.get("rollback_verified") is True,
            "node_replace_verified": ops_summary.get("node_replace_verified") is True,
            "runbook_scenario_count": len(runbook_scenarios),
            "formal_g4_deferred_requirement_count": len(DEFERRED_REQUIREMENTS),
            "proxy_hardware_name": proxy_hardware["hardware_name"],
            "proxy_device_count": len(proxy_hardware["selected_devices"]),
            "proxy_logical_host_count": len(proxy_hardware["logical_hosts"]),
        },
    }


def validate_phase4_resilience_gate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if packet.get("version") != 1:
        errors.append("version must be 1")
    if packet.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if packet.get("gate") != GATE:
        errors.append("gate must be G4")
    if packet.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if packet.get("outcome") not in VALID_OUTCOMES:
        errors.append("outcome must be a valid gate outcome")
    if packet.get("formal_g4_passed") is not False:
        errors.append("formal_g4_passed must remain false for the two-H100 proxy gate")
    if packet.get("formal_g4_validation_deferred") is not True:
        errors.append("formal_g4_validation_deferred must be true")


    proxy_hardware = packet.get("proxy_hardware")
    if not isinstance(proxy_hardware, dict):
        errors.append("proxy_hardware must be an object")
        proxy_hardware = {}
    hardware_name = proxy_hardware.get("hardware_name")
    if not isinstance(hardware_name, str) or not hardware_name:
        errors.append("proxy_hardware.hardware_name must be a non-empty string")
    elif "H100" not in hardware_name:
        errors.append("proxy_hardware.hardware_name must identify local H100 proxy hardware")
    selected_devices = _validate_string_list(
        proxy_hardware.get("selected_devices"),
        "proxy_hardware.selected_devices",
        errors,
    )
    logical_hosts = _validate_string_list(
        proxy_hardware.get("logical_hosts"),
        "proxy_hardware.logical_hosts",
        errors,
    )
    if proxy_hardware.get("formal_lab_evidence") is not False:
        errors.append("proxy_hardware.formal_lab_evidence must be false for proxy gate packets")

    artifact_paths = _validate_artifact_paths(
        packet,
        ("resilience_replay", "stage_replication", "ops_lifecycle"),
        errors,
    )
    artifact_validation = packet.get("artifact_validation")
    if not isinstance(artifact_validation, dict):
        errors.append("artifact_validation must be an object")
        artifact_validation = {}
    for name in ("resilience_replay", "stage_replication", "ops_lifecycle"):
        entry = artifact_validation.get(name)
        if not isinstance(entry, dict):
            errors.append(f"artifact_validation.{name} must be an object")
        else:
            if entry.get("ok") is not True:
                errors.append(f"artifact_validation.{name}.ok must be true")
            if artifact_paths.get(name) is not None and entry.get("artifact") != artifact_paths[name]:
                errors.append(f"artifact_validation.{name}.artifact must match artifacts.{name}")

    checks = packet.get("evidence_checks")
    if not isinstance(checks, list) or not checks:
        errors.append("evidence_checks must be a non-empty list")
        checks = []
    required_checks = {
        "resilience-artifact-valid",
        "zero-dropped-in-flight",
        "added-node-scaling",
        "drain-restart-rollback-hooks",
        "t4-runbook-defined",
        "two-h100-proxy-scope",
        "no-formal-g4-overclaim",
    }
    names = {item.get("name") for item in checks if isinstance(item, dict)}
    missing = sorted(required_checks - names)
    if missing:
        errors.append(f"evidence_checks missing required checks: {missing}")
    failed: list[Any] = []
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            failed.append(f"index-{index}")
        elif item.get("ok") is not True:
            failed.append(item.get("name"))
    if failed:
        errors.append(f"evidence_checks failed: {failed}")

    runbook = packet.get("runbook")
    if not isinstance(runbook, dict):
        errors.append("runbook must be an object")
        runbook = {}
    scenarios = runbook.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("runbook.scenarios must be a non-empty list")
        scenarios = []
    scenario_ids = {scenario.get("id") for scenario in scenarios if isinstance(scenario, dict)}
    missing_scenarios = sorted(REQUIRED_RUNBOOK_SCENARIOS - scenario_ids)
    if missing_scenarios:
        errors.append(f"runbook.scenarios missing required scenarios: {missing_scenarios}")
    for index, scenario in enumerate(scenarios):
        field = f"runbook.scenarios[{index}]"
        if not isinstance(scenario, dict):
            errors.append(f"{field} must be an object")
            continue
        if not isinstance(scenario.get("evidence_commands"), list) or not scenario["evidence_commands"]:
            errors.append(f"{field}.evidence_commands must be a non-empty list")
        if not isinstance(scenario.get("pass_criteria"), list) or not scenario["pass_criteria"]:
            errors.append(f"{field}.pass_criteria must be a non-empty list")
    if not isinstance(runbook.get("required_artifacts"), list) or len(runbook.get("required_artifacts", [])) < 5:
        errors.append("runbook.required_artifacts must include at least five artifacts")
    if "G4 can pass only" not in str(runbook.get("formal_gate_rule", "")):
        errors.append("runbook.formal_gate_rule must preserve the formal G4 boundary")

    deferred = packet.get("deferred_requirements")
    if not isinstance(deferred, list) or len(deferred) < len(DEFERRED_REQUIREMENTS):
        errors.append("deferred_requirements must include formal G4 deferred items")
        deferred = []
    deferred_ids = {item.get("id") for item in deferred if isinstance(item, dict)}
    for required_id in {item["id"] for item in DEFERRED_REQUIREMENTS}:
        if required_id not in deferred_ids:
            errors.append(f"deferred_requirements missing {required_id}")
    proxy_should_pass = packet.get("outcome") == "PROCEED" and not failed and not missing
    if packet.get("phase4_proxy_passed") is not proxy_should_pass:
        errors.append("phase4_proxy_passed must match PROCEED outcome and passing evidence checks")

    summary = packet.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("check_count") != len(checks):
        errors.append("summary.check_count must match evidence check count")
    passed_count = sum(1 for item in checks if isinstance(item, dict) and item.get("ok") is True)
    if summary.get("passed_count") != passed_count:
        errors.append("summary.passed_count must match passing evidence check count")
    if summary.get("zero_dropped_in_flight") is not True:
        errors.append("summary.zero_dropped_in_flight must be true")
    if summary.get("replication_correctness_passed") is not True:
        errors.append("summary.replication_correctness_passed must be true")
    if summary.get("rollback_verified") is not True or summary.get("node_replace_verified") is not True:
        errors.append("summary rollback/node replacement flags must be true")
    if selected_devices and summary.get("proxy_device_count") != len(selected_devices):
        errors.append("summary.proxy_device_count must match proxy_hardware.selected_devices")
    if logical_hosts and summary.get("proxy_logical_host_count") != len(logical_hosts):
        errors.append("summary.proxy_logical_host_count must match proxy_hardware.logical_hosts")
    if packet.get("phase4_proxy_passed") is True:
        warnings.append("Phase 4 proxy gate passed using simulation artifacts over two local H100 logical hosts; formal G4 T4 validation remains deferred.")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "phase4_proxy_passed": packet.get("phase4_proxy_passed") is True,
            "formal_g4_passed": packet.get("formal_g4_passed") is True,
            "check_count": len(checks),
            "passed_count": passed_count,
            "runbook_scenario_count": len(scenario_ids),
            "deferred_requirement_count": len(deferred),
        },
    }


def validate_phase4_resilience_gate(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid Phase 4 resilience gate artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["Phase 4 resilience gate artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_phase4_resilience_gate_packet(data)
    result["fixture"] = str(fixture_path)
    return result
