from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .benchmark_ledger import validate_benchmark_ledger
from .io import read_json
from .onboarding import validate_onboarding_methodology_fixture
from .ops_lifecycle import validate_ops_lifecycle_fixture
from .phase4_resilience_gate import validate_phase4_resilience_gate_packet

RECORD_KIND = "phase5-g5-productization-proxy-gate"
GATE = "G5"
MODE = "two-h100-local-proxy"
VALID_OUTCOMES = {"PROCEED", "ITERATE", "NARROW", "KILL"}

REQUIRED_RUNBOOK_SCENARIOS = {
    "operator-config-doctor",
    "deploy-upgrade-drain-restart-rollback",
    "node-replacement",
    "onboarding-handoff",
    "benchmark-of-record",
    "sponsor-ga-review",
}

DEFERRED_REQUIREMENTS = [
    {
        "id": "installable-release-package",
        "status": "deferred",
        "reason": "current evidence validates productization contracts and docs, not a signed release artifact",
    },
    {
        "id": "real-design-partner-onboarding",
        "status": "deferred",
        "reason": "onboarding paths are machine-validated but have not been completed by a fresh operator",
    },
    {
        "id": "lab-reference-benchmark-of-record",
        "status": "deferred",
        "reason": "current benchmark ledger is a measured fixture record; gate-grade lab-reference benchmark remains future work",
    },
    {
        "id": "production-deploy-upgrade-rollback",
        "status": "deferred",
        "reason": "lifecycle workflow is simulated over logical hosts and not wired to a production deployment",
    },
    {
        "id": "formal-sponsor-ga-acceptance",
        "status": "deferred",
        "reason": "G5 requires Sponsor review after install, operate, upgrade, and serve evidence is attached",
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
        "note": "Selected local H100 devices stand in for separate logical hosts during productization proxy validation.",
    }


def _operator_config_summary(ops: dict[str, Any]) -> dict[str, Any]:
    configs = ops.get("operator_configs") if isinstance(ops.get("operator_configs"), dict) else {}
    cluster = configs.get("cluster.yaml") if isinstance(configs.get("cluster.yaml"), dict) else {}
    model = configs.get("model.yaml") if isinstance(configs.get("model.yaml"), dict) else {}
    placement = configs.get("placement.json") if isinstance(configs.get("placement.json"), dict) else {}
    nodes = cluster.get("nodes") if isinstance(cluster.get("nodes"), list) else []
    serving = model.get("serving") if isinstance(model.get("serving"), dict) else {}
    return {
        "config_artifacts": sorted(configs),
        "cluster_node_count": len(nodes),
        "auth_required": cluster.get("auth", {}).get("required") is True,
        "plan_integrity_required": cluster.get("plan_integrity", {}).get("required") is True,
        "serving_endpoint": serving.get("endpoint"),
        "serving_engine": serving.get("engine"),
        "placement_feasible": placement.get("feasible") is True,
    }


def build_phase5_g5_runbook(*, plan_id: str = "phase5-productization-ga") -> dict[str, Any]:
    scenarios = [
        {
            "id": "operator-config-doctor",
            "goal": "prove the operator can prepare and inspect cluster, model, and placement artifacts",
            "setup": [
                "prepare cluster.yaml, model.yaml, and placement.json from the operator package",
                "keep plan-integrity and auth requirements enabled",
                "write a preflight bundle before lab deployment or Sponsor review",
            ],
            "evidence_commands": [
                "python3 -m fornax doctor --bundle <preflight-bundle>",
                "python3 -m fornax test ops-lifecycle",
            ],
            "pass_criteria": [
                "cluster.yaml, model.yaml, and placement.json are documented",
                "doctor exits cleanly or reports only acknowledged evidence-tier warnings",
                "operator config validation reports at least two nodes and a serving endpoint",
            ],
        },
        {
            "id": "deploy-upgrade-drain-restart-rollback",
            "goal": "prove lifecycle actions are repeatable, audited, and drain before mutation",
            "setup": [
                "load the operator lifecycle artifact",
                "run deploy, upgrade, restart, and rollback actions with request accounting enabled",
                "preserve event order and active version history",
            ],
            "evidence_commands": [
                "python3 -m fornax test ops-lifecycle",
                "python3 -m fornax ops lifecycle-simulate --out <ops-lifecycle.json>",
            ],
            "pass_criteria": [
                "drain_completed precedes each mutation",
                "dropped_in_flight_total == 0",
                "rollback_verified is true",
            ],
        },
        {
            "id": "node-replacement",
            "goal": "prove replacement preserves identity, placement, and cleanup invariants",
            "setup": [
                "drain the old node before removal",
                "admit a replacement node with managed identity",
                "verify traffic restoration and final active-node state",
            ],
            "evidence_commands": [
                "python3 -m fornax test ops-lifecycle",
            ],
            "pass_criteria": [
                "node_replace_verified is true",
                "removed node is no longer active",
                "traffic_restored event is present",
            ],
        },
        {
            "id": "onboarding-handoff",
            "goal": "prove operator, developer, benchmark-owner, and reviewer tracks are complete enough for proxy handoff",
            "setup": [
                "publish quickstart, operator runbook, developer workflow, benchmark methodology, and glossary",
                "map each track to first-run commands, success evidence, and escalation paths",
                "preserve simulation warnings until formal GA evidence exists",
            ],
            "evidence_commands": [
                "python3 -m fornax test onboarding-methodology",
            ],
            "pass_criteria": [
                "required tracks are present",
                "required documents are present",
                "required glossary terms are present",
            ],
        },
        {
            "id": "benchmark-of-record",
            "goal": "prove benchmark methodology requires commands, traces, versions, correctness, environment, logs, and ledger records",
            "setup": [
                "capture benchmark commands and raw logs",
                "attach correctness artifacts before throughput claims",
                "append each measured run to ledger.jsonl",
            ],
            "evidence_commands": [
                "python3 -m fornax test benchmark-ledger",
                "python3 -m fornax benchmark --plan placement.json --mode tiny-moe-or-expert-mlp --out benchmark.json",
            ],
            "pass_criteria": [
                "ledger contains at least one measured record",
                "required benchmark methodology inputs are defined",
                "lab-reference boundary remains explicit for formal G5",
            ],
        },
        {
            "id": "sponsor-ga-review",
            "goal": "package install, operate, upgrade, serve, and benchmark evidence for Sponsor G5 decision",
            "setup": [
                "attach prior G4 proxy or formal packet",
                "attach productization runbook, onboarding package, lifecycle evidence, and benchmark ledger",
                "record which formal GA requirements remain deferred",
            ],
            "evidence_commands": [
                "python3 -m fornax program phase5-ga-gate --ops-artifact <ops> --onboarding-artifact <onboarding> --benchmark-ledger <ledger> --phase4-artifact <phase4> --out <packet.json>",
            ],
            "pass_criteria": [
                "phase5_proxy_passed may be true only for the local proxy gate",
                "formal_g5_passed remains false until Sponsor accepts real GA evidence",
                "all deferred requirements are visible in the packet",
            ],
        },
    ]
    return {
        "version": 1,
        "plan_id": plan_id,
        "title": "G5 Productization and GA Runbook",
        "scope": "operator UX, lifecycle, onboarding, benchmark methodology, and Sponsor GA evidence packaging",
        "proxy_development_scope": "two local H100 GPUs may stand in for two logical hosts until formal product, lab, and Sponsor evidence is available",
        "scenarios": scenarios,
        "required_artifacts": [
            "cluster.yaml, model.yaml, and placement.json operator configs",
            "fornax doctor preflight bundle or equivalent diagnostics",
            "ops lifecycle artifact",
            "onboarding package with glossary and benchmark methodology",
            "benchmark ledger with measured records",
            "prior G4 resilience gate packet",
            "Sponsor G5 decision record",
        ],
        "formal_gate_rule": "G5 can pass only with installable product evidence, operator acceptance, benchmark-of-record evidence, and Sponsor GA approval; two-H100 local proxy evidence may pass only the current development proxy gate.",
    }


def render_phase5_g5_runbook_markdown(runbook: dict[str, Any]) -> str:
    lines = [
        "# G5 Productization and GA Runbook",
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


def build_phase5_ga_gate_packet(
    ops_artifact: str | Path,
    onboarding_artifact: str | Path,
    benchmark_ledger: str | Path,
    phase4_artifact: str | Path,
    *,
    packet_date: str | None = None,
    outcome: str = "PROCEED",
    accepted_by: str = "operator",
    decision: str = "Use two local H100 GPUs as logical hosts for the current Phase 5 productization/GA proxy gate; defer formal Sponsor G5 validation.",
    proxy_hardware_name: str = "NVIDIA H100 80GB HBM3",
    proxy_devices: list[str] | None = None,
    proxy_logical_hosts: list[str] | None = None,
) -> dict[str, Any]:
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}")
    ops_path, ops = _load_fixture(ops_artifact)
    onboarding_path, onboarding = _load_fixture(onboarding_artifact)
    phase4_path, phase4 = _load_fixture(phase4_artifact)

    ops_validation = validate_ops_lifecycle_fixture(ops)
    onboarding_validation = validate_onboarding_methodology_fixture(onboarding)
    benchmark_validation = validate_benchmark_ledger(benchmark_ledger)
    phase4_validation = validate_phase4_resilience_gate_packet(phase4)

    ops_summary = ops.get("summary") if isinstance(ops.get("summary"), dict) else {}
    ops_accounting = ops.get("request_accounting") if isinstance(ops.get("request_accounting"), dict) else {}
    onboarding_summary = onboarding.get("summary") if isinstance(onboarding.get("summary"), dict) else {}
    benchmark_summary = benchmark_validation.get("summary") if isinstance(benchmark_validation.get("summary"), dict) else {}
    operator_configs = _operator_config_summary(ops)
    runbook = build_phase5_g5_runbook()
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
            "ops-lifecycle-valid",
            ops_validation.get("ok") is True,
            f"validate_ops_lifecycle_fixture({ops_path})",
        ),
        _check(
            "operator-ux-configs",
            operator_configs["cluster_node_count"] >= 2
            and {"cluster.yaml", "model.yaml", "placement.json"}.issubset(operator_configs["config_artifacts"])
            and operator_configs["auth_required"]
            and operator_configs["plan_integrity_required"]
            and operator_configs["serving_endpoint"] == "/v1/chat/completions"
            and operator_configs["placement_feasible"],
            "cluster.yaml, model.yaml, placement.json, auth, plan integrity, and serving endpoint are represented",
        ),
        _check(
            "upgrade-drain-rollback-node-replace",
            ops_validation.get("ok") is True
            and ops_summary.get("dropped_in_flight_count") == 0
            and ops_summary.get("rollback_verified") is True
            and ops_summary.get("node_replace_verified") is True
            and ops_accounting.get("dropped_in_flight_total") == 0,
            "deploy, drain, upgrade, restart, rollback, and node replacement are audited with zero dropped in-flight requests",
        ),
        _check(
            "onboarding-methodology-valid",
            onboarding_validation.get("ok") is True,
            f"validate_onboarding_methodology_fixture({onboarding_path})",
        ),
        _check(
            "onboarding-and-glossary",
            onboarding_summary.get("required_tracks_present") is True
            and onboarding_summary.get("required_documents_present") is True
            and onboarding_summary.get("required_glossary_terms_present") is True
            and onboarding_summary.get("onboarding_complete_for_simulation") is True
            and onboarding_summary.get("product_ga_complete") is False,
            "operator, developer, benchmark-owner, reviewer, required docs, and glossary are present for proxy handoff",
        ),
        _check(
            "benchmark-ledger-valid",
            benchmark_validation.get("ok") is True,
            f"validate_benchmark_ledger({benchmark_validation.get('ledger')})",
        ),
        _check(
            "benchmark-methodology-of-record",
            onboarding_summary.get("benchmark_methodology_present") is True
            and onboarding_summary.get("lab_reference_required") is True
            and onboarding_summary.get("correctness_first") is True
            and benchmark_summary.get("record_count", 0) >= 1
            and benchmark_summary.get("measured_record_count", 0) >= 1,
            "benchmark methodology defines gate mapping and required inputs; ledger contains measured records",
        ),
        _check(
            "prior-g4-proxy-attached",
            phase4_validation.get("ok") is True
            and phase4.get("phase4_proxy_passed") is True
            and phase4.get("formal_g4_passed") is False,
            f"validate_phase4_resilience_gate_packet({phase4_path})",
        ),
        _check(
            "g5-runbook-defined",
            REQUIRED_RUNBOOK_SCENARIOS.issubset(runbook_scenarios)
            and len(runbook.get("required_artifacts", [])) >= 7,
            "runbook defines operator UX, lifecycle, node replacement, onboarding, benchmark, and Sponsor review evidence collection",
        ),
        _check(
            "two-h100-proxy-scope",
            h100_proxy_scope
            and ops.get("mode") == "t1-simulation"
            and onboarding.get("mode") == "t1-simulation"
            and phase4.get("mode") == "two-h100-local-proxy",
            "selected local H100 devices and logical hosts are recorded for productization proxy scope",
        ),
        _check(
            "no-formal-g5-overclaim",
            True,
            "formal_g5_passed remains false until installable product, benchmark-of-record, and Sponsor GA evidence are attached",
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
        "phase5_proxy_passed": proxy_passed,
        "formal_g5_passed": False,
        "formal_g5_validation_deferred": True,
        "artifacts": {
            "ops_lifecycle": str(ops_path),
            "onboarding_methodology": str(onboarding_path),
            "benchmark_ledger": str(benchmark_validation.get("ledger", benchmark_ledger)),
            "phase4_resilience_gate": str(phase4_path),
        },
        "proxy_hardware": proxy_hardware,
        "operator_configs": operator_configs,
        "artifact_validation": {
            "ops_lifecycle": {
                "artifact": str(ops_path),
                "ok": ops_validation.get("ok") is True,
                "errors": list(ops_validation.get("errors", [])),
                "warnings": list(ops_validation.get("warnings", [])),
            },
            "onboarding_methodology": {
                "artifact": str(onboarding_path),
                "ok": onboarding_validation.get("ok") is True,
                "errors": list(onboarding_validation.get("errors", [])),
                "warnings": list(onboarding_validation.get("warnings", [])),
            },
            "benchmark_ledger": {
                "artifact": str(benchmark_validation.get("ledger", benchmark_ledger)),
                "ok": benchmark_validation.get("ok") is True,
                "errors": list(benchmark_validation.get("errors", [])),
                "warnings": list(benchmark_validation.get("warnings", [])),
            },
            "phase4_resilience_gate": {
                "artifact": str(phase4_path),
                "ok": phase4_validation.get("ok") is True,
                "errors": list(phase4_validation.get("errors", [])),
                "warnings": list(phase4_validation.get("warnings", [])),
            },
        },
        "evidence_checks": evidence_checks,
        "runbook": runbook,
        "deferred_requirements": DEFERRED_REQUIREMENTS,
        "summary": {
            "check_count": len(evidence_checks),
            "passed_count": sum(1 for item in evidence_checks if item["ok"]),
            "operator_config_count": len(operator_configs["config_artifacts"]),
            "cluster_node_count": operator_configs["cluster_node_count"],
            "serving_endpoint": operator_configs["serving_endpoint"],
            "lifecycle_action_count": ops_summary.get("action_count"),
            "lifecycle_dropped_in_flight_count": ops_summary.get("dropped_in_flight_count"),
            "rollback_verified": ops_summary.get("rollback_verified") is True,
            "node_replace_verified": ops_summary.get("node_replace_verified") is True,
            "onboarding_track_count": onboarding_summary.get("track_count"),
            "onboarding_document_count": onboarding_summary.get("document_count"),
            "onboarding_glossary_term_count": onboarding_summary.get("glossary_term_count"),
            "benchmark_ledger_record_count": benchmark_summary.get("record_count"),
            "benchmark_measured_record_count": benchmark_summary.get("measured_record_count"),
            "phase4_proxy_passed": phase4.get("phase4_proxy_passed") is True,
            "phase4_formal_g4_passed": phase4.get("formal_g4_passed") is True,
            "product_ga_complete": onboarding_summary.get("product_ga_complete") is True,
            "runbook_scenario_count": len(runbook_scenarios),
            "formal_g5_deferred_requirement_count": len(DEFERRED_REQUIREMENTS),
            "proxy_hardware_name": proxy_hardware["hardware_name"],
            "proxy_device_count": len(proxy_hardware["selected_devices"]),
            "proxy_logical_host_count": len(proxy_hardware["logical_hosts"]),
        },
    }


def _validate_proxy_hardware(packet: dict[str, Any], errors: list[str]) -> tuple[list[Any], list[Any]]:
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
    return selected_devices, logical_hosts


def validate_phase5_ga_gate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if packet.get("version") != 1:
        errors.append("version must be 1")
    if packet.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if packet.get("gate") != GATE:
        errors.append("gate must be G5")
    if packet.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if packet.get("outcome") not in VALID_OUTCOMES:
        errors.append("outcome must be a valid gate outcome")
    if packet.get("formal_g5_passed") is not False:
        errors.append("formal_g5_passed must remain false for the two-H100 proxy gate")
    if packet.get("formal_g5_validation_deferred") is not True:
        errors.append("formal_g5_validation_deferred must be true")

    selected_devices, logical_hosts = _validate_proxy_hardware(packet, errors)

    artifact_paths = _validate_artifact_paths(
        packet,
        ("ops_lifecycle", "onboarding_methodology", "benchmark_ledger", "phase4_resilience_gate"),
        errors,
    )
    artifact_validation = packet.get("artifact_validation")
    if not isinstance(artifact_validation, dict):
        errors.append("artifact_validation must be an object")
        artifact_validation = {}
    for name in ("ops_lifecycle", "onboarding_methodology", "benchmark_ledger", "phase4_resilience_gate"):
        entry = artifact_validation.get(name)
        if not isinstance(entry, dict):
            errors.append(f"artifact_validation.{name} must be an object")
        else:
            if entry.get("ok") is not True:
                errors.append(f"artifact_validation.{name}.ok must be true")
            if artifact_paths.get(name) is not None and entry.get("artifact") != artifact_paths[name]:
                errors.append(f"artifact_validation.{name}.artifact must match artifacts.{name}")

    operator_configs = packet.get("operator_configs")
    if not isinstance(operator_configs, dict):
        errors.append("operator_configs must be an object")
        operator_configs = {}
    if operator_configs.get("cluster_node_count", 0) < 2:
        errors.append("operator_configs.cluster_node_count must be at least 2")
    if operator_configs.get("serving_endpoint") != "/v1/chat/completions":
        errors.append("operator_configs.serving_endpoint must be /v1/chat/completions")
    if operator_configs.get("auth_required") is not True:
        errors.append("operator_configs.auth_required must be true")
    if operator_configs.get("plan_integrity_required") is not True:
        errors.append("operator_configs.plan_integrity_required must be true")
    if operator_configs.get("placement_feasible") is not True:
        errors.append("operator_configs.placement_feasible must be true")

    checks = packet.get("evidence_checks")
    if not isinstance(checks, list) or not checks:
        errors.append("evidence_checks must be a non-empty list")
        checks = []
    required_checks = {
        "ops-lifecycle-valid",
        "operator-ux-configs",
        "upgrade-drain-rollback-node-replace",
        "onboarding-methodology-valid",
        "onboarding-and-glossary",
        "benchmark-ledger-valid",
        "benchmark-methodology-of-record",
        "prior-g4-proxy-attached",
        "g5-runbook-defined",
        "two-h100-proxy-scope",
        "no-formal-g5-overclaim",
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
    if not isinstance(runbook.get("required_artifacts"), list) or len(runbook.get("required_artifacts", [])) < 7:
        errors.append("runbook.required_artifacts must include at least seven artifacts")
    if "G5 can pass only" not in str(runbook.get("formal_gate_rule", "")):
        errors.append("runbook.formal_gate_rule must preserve the formal G5 boundary")

    deferred = packet.get("deferred_requirements")
    if not isinstance(deferred, list) or len(deferred) < len(DEFERRED_REQUIREMENTS):
        errors.append("deferred_requirements must include formal G5 deferred items")
        deferred = []
    deferred_ids = {item.get("id") for item in deferred if isinstance(item, dict)}
    for required_id in {item["id"] for item in DEFERRED_REQUIREMENTS}:
        if required_id not in deferred_ids:
            errors.append(f"deferred_requirements missing {required_id}")
    proxy_should_pass = packet.get("outcome") == "PROCEED" and not failed and not missing
    if packet.get("phase5_proxy_passed") is not proxy_should_pass:
        errors.append("phase5_proxy_passed must match PROCEED outcome and passing evidence checks")

    summary = packet.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("check_count") != len(checks):
        errors.append("summary.check_count must match evidence check count")
    passed_count = sum(1 for item in checks if isinstance(item, dict) and item.get("ok") is True)
    if summary.get("passed_count") != passed_count:
        errors.append("summary.passed_count must match passing evidence check count")
    if summary.get("operator_config_count", 0) < 3:
        errors.append("summary.operator_config_count must include cluster/model/placement configs")
    if summary.get("lifecycle_dropped_in_flight_count") != 0:
        errors.append("summary.lifecycle_dropped_in_flight_count must be 0")
    if summary.get("rollback_verified") is not True or summary.get("node_replace_verified") is not True:
        errors.append("summary rollback/node replacement flags must be true")
    if summary.get("benchmark_ledger_record_count", 0) < 1 or summary.get("benchmark_measured_record_count", 0) < 1:
        errors.append("summary benchmark ledger counts must include measured records")
    if summary.get("phase4_proxy_passed") is not True or summary.get("phase4_formal_g4_passed") is not False:
        errors.append("summary must attach a passing proxy G4 packet without formal G4 overclaim")
    if summary.get("product_ga_complete") is not False:
        errors.append("summary.product_ga_complete must be false for the proxy gate")
    if selected_devices and summary.get("proxy_device_count") != len(selected_devices):
        errors.append("summary.proxy_device_count must match proxy_hardware.selected_devices")
    if logical_hosts and summary.get("proxy_logical_host_count") != len(logical_hosts):
        errors.append("summary.proxy_logical_host_count must match proxy_hardware.logical_hosts")
    if packet.get("phase5_proxy_passed") is True:
        warnings.append("Phase 5 proxy gate passed using lifecycle/onboarding/benchmark fixtures over two local H100 logical hosts; formal G5 Product GA remains deferred.")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "phase5_proxy_passed": packet.get("phase5_proxy_passed") is True,
            "formal_g5_passed": packet.get("formal_g5_passed") is True,
            "check_count": len(checks),
            "passed_count": passed_count,
            "runbook_scenario_count": len(scenario_ids),
            "deferred_requirement_count": len(deferred),
        },
    }


def validate_phase5_ga_gate(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid Phase 5 GA gate artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["Phase 5 GA gate artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_phase5_ga_gate_packet(data)
    result["fixture"] = str(fixture_path)
    return result
