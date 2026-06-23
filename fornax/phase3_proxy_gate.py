from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .io import read_json
from .local_http_serving_smoke import validate_local_http_serving_smoke_fixture

RECORD_KIND = "phase3-g3-proxy-gate"
GATE = "G3"
MODE = "two-h100-local-proxy"
VALID_OUTCOMES = {"PROCEED", "ITERATE", "NARROW", "KILL"}

DEFERRED_REQUIREMENTS = [
    {
        "id": "real-frontier-target-model",
        "status": "deferred",
        "reason": "current proxy uses deterministic target fixture; real frontier target-model loading/parity remains future validation",
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
        "id": "product-auth-mtls-keying",
        "status": "deferred",
        "reason": "local mTLS/auth is sufficient for the proxy gate but not product keying evidence",
    },
    {
        "id": "distributed-partition-proof",
        "status": "deferred",
        "reason": "local partition fencing/recovery is sufficient for the proxy gate but not real distributed partition evidence",
    },
]


def _check(name: str, ok: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "evidence": evidence}


def build_phase3_proxy_gate_packet(
    endpoint_artifact: str | Path,
    *,
    packet_date: str | None = None,
    outcome: str = "PROCEED",
    accepted_by: str = "operator",
    decision: str = "Use two local H100 GPUs as logical hosts for the current Phase 3 proxy gate; defer real NVIDIA/AMD/Mac validation.",
) -> dict[str, Any]:
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}")
    endpoint_path = Path(endpoint_artifact)
    endpoint_bundle = read_json(endpoint_path)
    endpoint_validation = validate_local_http_serving_smoke_fixture(endpoint_bundle)
    summary = endpoint_bundle.get("summary") if isinstance(endpoint_bundle, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    route = endpoint_bundle.get("local_topology_route") if isinstance(endpoint_bundle, dict) else {}
    if not isinstance(route, dict):
        route = {}
    component_names = {
        component.get("name")
        for component in route.get("components", [])
        if isinstance(component, dict)
    }
    deferred_hardware = {
        item.get("hardware_class")
        for item in route.get("deferred_hardware", [])
        if isinstance(item, dict)
    }
    evidence_checks = [
        _check(
            "endpoint-artifact-valid",
            endpoint_validation.get("ok") is True,
            f"validate_local_http_serving_smoke_fixture({endpoint_path})",
        ),
        _check(
            "h100-two-logical-hosts",
            summary.get("activation_transfer_source_device") == "cuda:0"
            and summary.get("activation_transfer_destination_device") == "cuda:1"
            and summary.get("pipeline_correctness_source_device") == "cuda:0"
            and summary.get("pipeline_correctness_destination_device") == "cuda:1"
            and summary.get("moe_layer_parity_source_device") == "cuda:0"
            and summary.get("moe_layer_parity_expert_device") == "cuda:1",
            "activation-transfer, split-pipeline, and MoE probes span cuda:0->cuda:1 logical hosts",
        ),
        _check(
            "endpoint-security",
            summary.get("mtls_enabled") is True
            and summary.get("mtls_missing_client_cert_rejected") is True
            and summary.get("endpoint_auth_rejected") is True
            and summary.get("plan_integrity_rejected") is True,
            "local mTLS, missing-client-cert rejection, bearer auth rejection, and plan-hash rejection",
        ),
        _check(
            "failure-semantics",
            summary.get("backpressure_retry_after_header_verified") is True
            and summary.get("backpressure_retry_after_capacity_clear") is True
            and summary.get("request_cancelled_before_backend") is True
            and summary.get("request_timed_out_before_backend") is True
            and summary.get("partitioned_before_backend") is True
            and summary.get("partition_recovery_after_fence") is True,
            "local retry-after, cancellation, timeout, partition fence, and partition recovery",
        ),
        _check(
            "lifecycle-state-ownership",
            summary.get("lifecycle_all_released") is True
            and summary.get("lifecycle_single_owner_preserved") is True
            and summary.get("lifecycle_active_resource_count") == 0,
            "lifecycle cleanup releases all local resources with single-owner accounting",
        ),
        _check(
            "runtime-probes",
            summary.get("activation_transfer_probe_ok") is True
            and summary.get("pipeline_correctness_probe_ok") is True
            and summary.get("moe_layer_parity_probe_ok") is True
            and summary.get("target_fixture_execution_probe_ok") is True,
            "activation-transfer, split-pipeline, MoE parity, and target-fixture execution probes pass",
        ),
        _check(
            "topology-route",
            summary.get("local_topology_route_verified") is True
            and {"activation-transfer", "split-pipeline", "remote-moe-expert", "target-fixture-execution"}.issubset(component_names)
            and {"AMD GPU node", "Apple Silicon Mac"}.issubset(deferred_hardware),
            "local topology route includes measured components and deferred AMD/Apple hardware explanations",
        ),
        _check(
            "no-formal-g3-overclaim",
            summary.get("target_model_parity") is False
            and summary.get("g2_g3_gate_evidence") is False
            and summary.get("local_topology_production_evidence") is False
            and summary.get("local_topology_distributed_evidence") is False,
            "source endpoint artifact remains scoped as local proxy evidence",
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
        "phase3_proxy_passed": proxy_passed,
        "formal_g3_passed": False,
        "formal_g3_validation_deferred": True,
        "endpoint_artifact": str(endpoint_path),
        "endpoint_validation": {
            "ok": endpoint_validation.get("ok") is True,
            "errors": list(endpoint_validation.get("errors", [])),
            "warnings": list(endpoint_validation.get("warnings", [])),
        },
        "evidence_checks": evidence_checks,
        "deferred_requirements": DEFERRED_REQUIREMENTS,
        "summary": {
            "check_count": len(evidence_checks),
            "passed_count": sum(1 for item in evidence_checks if item["ok"]),
            "endpoint_check_count": summary.get("check_count"),
            "endpoint_passed_count": summary.get("passed_count"),
            "endpoint": summary.get("endpoint"),
            "logical_hosts": summary.get("local_topology_logical_hosts"),
            "component_count": summary.get("local_topology_component_count"),
            "deferred_hardware_count": summary.get("local_topology_deferred_hardware_count"),
            "activation_transfer_bandwidth_gib_s": summary.get("activation_transfer_bandwidth_gib_s"),
            "pipeline_tokens_s": summary.get("pipeline_correctness_tokens_s"),
            "moe_tokens_s": summary.get("moe_layer_parity_tokens_s"),
            "target_fixture_execution_tokens_s": summary.get("target_fixture_execution_tokens_s"),
        },
    }


def validate_phase3_proxy_gate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if packet.get("version") != 1:
        errors.append("version must be 1")
    if packet.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if packet.get("gate") != GATE:
        errors.append("gate must be G3")
    if packet.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if packet.get("outcome") not in VALID_OUTCOMES:
        errors.append("outcome must be a valid gate outcome")
    if packet.get("formal_g3_passed") is not False:
        errors.append("formal_g3_passed must remain false for the two-H100 proxy gate")
    if packet.get("formal_g3_validation_deferred") is not True:
        errors.append("formal_g3_validation_deferred must be true")
    checks = packet.get("evidence_checks")
    if not isinstance(checks, list) or not checks:
        errors.append("evidence_checks must be a non-empty list")
        checks = []
    failed = [item.get("name") for item in checks if not isinstance(item, dict) or item.get("ok") is not True]
    if failed:
        errors.append(f"evidence_checks failed: {failed}")
    required = {
        "endpoint-artifact-valid",
        "h100-two-logical-hosts",
        "endpoint-security",
        "failure-semantics",
        "lifecycle-state-ownership",
        "runtime-probes",
        "topology-route",
        "no-formal-g3-overclaim",
    }
    names = {item.get("name") for item in checks if isinstance(item, dict)}
    missing = sorted(required - names)
    if missing:
        errors.append(f"evidence_checks missing required checks: {missing}")
    deferred = packet.get("deferred_requirements")
    if not isinstance(deferred, list) or len(deferred) < 5:
        errors.append("deferred_requirements must include formal G3 deferred items")
        deferred = []
    deferred_ids = {item.get("id") for item in deferred if isinstance(item, dict)}
    for required_id in {
        "real-frontier-target-model",
        "real-amd-gpu-node",
        "real-apple-silicon-mac",
        "product-auth-mtls-keying",
        "distributed-partition-proof",
    }:
        if required_id not in deferred_ids:
            errors.append(f"deferred_requirements missing {required_id}")
    proxy_should_pass = packet.get("outcome") == "PROCEED" and not failed and not missing
    if packet.get("phase3_proxy_passed") is not proxy_should_pass:
        errors.append("phase3_proxy_passed must match PROCEED outcome and passing evidence checks")
    summary = packet.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("check_count") != len(checks):
        errors.append("summary.check_count must match evidence check count")
    if summary.get("passed_count") != sum(1 for item in checks if isinstance(item, dict) and item.get("ok") is True):
        errors.append("summary.passed_count must match passing evidence check count")
    if packet.get("phase3_proxy_passed") is True:
        warnings.append("Phase 3 proxy gate passed using two local H100 logical hosts; formal NVIDIA/AMD/Mac G3 validation remains deferred.")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "phase3_proxy_passed": packet.get("phase3_proxy_passed") is True,
            "formal_g3_passed": packet.get("formal_g3_passed") is True,
            "check_count": len(checks),
            "passed_count": sum(1 for item in checks if isinstance(item, dict) and item.get("ok") is True),
            "deferred_requirement_count": len(deferred),
        },
    }
