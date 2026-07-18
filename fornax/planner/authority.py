from __future__ import annotations

import math
from collections.abc import Iterable

from .evidence import EvidenceRegistry
from .model import (
    BoundaryLink,
    Inventory,
    Link,
    MeasurementProvenance,
    ModelSpec,
    Node,
    PlanAuthority,
    Stage,
    Target,
)


CRITICAL_NODE_MEASUREMENTS = (
    "mem_free_bytes",
    "compute_class",
    "mem_bandwidth_bytes_s",
)
CRITICAL_LINK_MEASUREMENTS = ("bandwidth_bytes_s", "latency_s")
_CONFIDENCE_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _numeric_issue(
    label: str, value: object, *, integer: bool = False
) -> str | None:
    if isinstance(value, bool):
        return f"{label} is boolean, not numeric"
    if integer and not isinstance(value, int):
        return f"{label} is not an integer"
    if not isinstance(value, (int, float)):
        return f"{label} is not numeric"
    if isinstance(value, float) and not math.isfinite(value):
        return f"{label} is not finite"
    return None


def _model_target_numeric_issues(
    model: ModelSpec, target: Target
) -> tuple[str, ...]:
    issues: list[str] = []
    for label, value in (
        ("model hidden_dim", model.hidden_dim),
        ("model num_layers", model.num_layers),
        ("target concurrency", target.concurrency),
        ("target prompt_len", target.prompt_len),
        ("target gen_len", target.gen_len),
        ("target runtime_reserve_bytes", target.runtime_reserve_bytes),
    ):
        issue = _numeric_issue(label, value, integer=True)
        if issue:
            issues.append(issue)
    for layer_id, layer in enumerate(model.layers):
        for field_name in (
            "weight_bytes",
            "active_flops_per_token",
            "kv_bytes_per_token",
            "num_experts",
            "experts_active",
            "expert_bytes",
            "expert_flops_per_token",
            "shared_expert_bytes",
        ):
            issue = _numeric_issue(
                f"model layer {layer_id} {field_name}",
                getattr(layer, field_name),
                integer=True,
            )
            if issue:
                issues.append(issue)
    for trace_id, trace in enumerate(model.expert_traces):
        for field_name in ("layer_id", "expert_id"):
            issue = _numeric_issue(
                f"model expert trace {trace_id} {field_name}",
                getattr(trace, field_name),
                integer=True,
            )
            if issue:
                issues.append(issue)
        for field_name in ("hit_rate_prefill", "hit_rate_decode"):
            issue = _numeric_issue(
                f"model expert trace {trace_id} {field_name}",
                getattr(trace, field_name),
            )
            if issue:
                issues.append(issue)
        for coactivation_id, (expert_id, rate) in enumerate(trace.coactivation):
            for label, value, integer in (
                ("expert_id", expert_id, True),
                ("rate", rate, False),
            ):
                issue = _numeric_issue(
                    "model expert trace "
                    f"{trace_id} coactivation {coactivation_id} {label}",
                    value,
                    integer=integer,
                )
                if issue:
                    issues.append(issue)
    for field_name in (
        "memory_reserve_fraction",
        "fragmentation_margin_fraction",
        "routing_metadata_bytes_per_token",
        "temp_buffer_fraction",
        "max_expected_relative_error",
    ):
        issue = _numeric_issue(
            f"target {field_name}", getattr(target, field_name)
        )
        if issue:
            issues.append(issue)
    if target.remote_expert_wait_slo_s is not None:
        issue = _numeric_issue(
            "target remote_expert_wait_slo_s", target.remote_expert_wait_slo_s
        )
        if issue:
            issues.append(issue)
    return _unique(issues)


def _node_numeric_issues(node: Node) -> tuple[str, ...]:
    issues: list[str] = []
    issue = _numeric_issue(
        f"node {node.id} mem_free_bytes", node.mem_free_bytes, integer=True
    )
    if issue:
        issues.append(issue)
    for field_name in ("compute_class", "mem_bandwidth_bytes_s", "reliability"):
        issue = _numeric_issue(
            f"node {node.id} {field_name}", getattr(node, field_name)
        )
        if issue:
            issues.append(issue)
    return _unique(issues)


def _link_numeric_issues(link: Link) -> tuple[str, ...]:
    issues: list[str] = []
    for field_name in ("bandwidth_bytes_s", "latency_s"):
        issue = _numeric_issue(
            f"link {link.a}<->{link.b} {field_name}", getattr(link, field_name)
        )
        if issue:
            issues.append(issue)
    return _unique(issues)


def _measurement_deployment_issues(
    label: str,
    provenance: MeasurementProvenance,
    max_expected_relative_error: float,
) -> list[str]:
    issues: list[str] = []
    if provenance.status != "measured":
        issues.append(f"{label} status is {provenance.status}, not measured")
    if provenance.source_id is None:
        issues.append(f"{label} has no measurement source_id")
    if provenance.confidence not in {"high", "medium"}:
        issues.append(
            f"{label} confidence is {provenance.confidence}, not high or medium"
        )
    error = provenance.expected_relative_error
    if error is None:
        issues.append(f"{label} has no expected_relative_error")
    elif isinstance(error, bool) or not isinstance(error, (int, float)):
        issues.append(f"{label} expected_relative_error is not numeric")
    elif not math.isfinite(error):
        issues.append(f"{label} expected_relative_error is not finite")
    elif error > max_expected_relative_error:
        issues.append(
            f"{label} expected_relative_error={error:.6g} exceeds "
            f"limit={max_expected_relative_error:.6g}"
        )
    return issues


def global_deployment_issues(model: ModelSpec, target: Target) -> tuple[str, ...]:
    issues: list[str] = list(_model_target_numeric_issues(model, target))
    if model.source_id is None:
        issues.append("model has no source_id")
    if model.quantization_source_id is None:
        issues.append("model has no quantization_source_id")
    if target.required_runtime is None:
        issues.append("target has no required_runtime")
    if not target.accepted_build_ids:
        issues.append("target has no accepted_build_ids")
    if not target.required_operations:
        issues.append("target has no required_operations")
    issues.extend(
        _measurement_deployment_issues(
            "prediction calibration",
            target.prediction_calibration,
            target.max_expected_relative_error,
        )
    )
    return _unique(issues)


def evidence_registry_deployment_issues(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    evidence_registry: EvidenceRegistry | None,
) -> tuple[str, ...]:
    """Resolve every planner evidence reference through a separate registry.

    Deployment search consumes all candidate node and link values when ranking
    or excluding placements, so the fail-closed boundary covers the complete
    input inventory, not only the nodes selected by the eventual winning plan.
    """

    if evidence_registry is None:
        return ("deployment evidence registry is required",)

    references: list[tuple[str, str, str]] = []

    def add(source_id: str | None, evidence_type: str, label: str) -> None:
        if source_id is not None:
            references.append((source_id, evidence_type, label))

    add(model.source_id, "model", "model")
    add(
        model.quantization_source_id,
        "quantization",
        "model quantization",
    )
    add(model.expert_trace_source_id, "expert_trace", "model expert trace")
    add(
        target.prediction_calibration.source_id,
        "calibration",
        "prediction calibration",
    )
    for node in inventory.nodes:
        add(
            node.capability_source_id,
            "capability",
            f"node {node.id} capability",
        )
        for field_name, provenance in node.measurement_provenance:
            add(
                provenance.source_id,
                "measurement",
                f"node {node.id} {field_name}",
            )
    for link in inventory.links:
        for field_name, provenance in link.measurement_provenance:
            add(
                provenance.source_id,
                "route",
                f"link {link.a}<->{link.b} {field_name}",
            )

    issues: list[str] = []
    for source_id, evidence_type, label in references:
        issues.extend(
            evidence_registry.resolution_issues(
                source_id,
                evidence_type=evidence_type,
                label=label,
            )
        )
    return _unique(issues)


def node_admission_issues(
    node: Node,
    model: ModelSpec,
    target: Target,
    *,
    deployment: bool,
) -> tuple[str, ...]:
    """Return exact capability and, in deployment mode, evidence failures.

    Explicit incompatibilities fail in both modes.  An absent/incomplete
    capability declaration remains usable only for exploratory planning.
    """

    issues: list[str] = list(_node_numeric_issues(node))
    if target.required_runtime is not None and node.runtime != target.required_runtime:
        issues.append(
            f"node {node.id} runtime={node.runtime!r} does not match "
            f"required_runtime={target.required_runtime!r}"
        )
    if target.accepted_build_ids and node.build_id is not None:
        if node.build_id not in target.accepted_build_ids:
            issues.append(
                f"node {node.id} build_id={node.build_id!r} is not accepted"
            )
    elif deployment and node.build_id is None:
        issues.append(f"node {node.id} has no build_id")

    if node.capabilities_complete:
        missing_operations = sorted(
            set(target.required_operations) - set(node.supported_operations)
        )
        if missing_operations:
            issues.append(
                f"node {node.id} lacks required operations={missing_operations}"
            )
        if model.dtype_weight not in node.supported_quantizations:
            issues.append(
                f"node {node.id} does not support weight quantization "
                f"{model.dtype_weight}"
            )
    elif deployment:
        issues.append(f"node {node.id} capability declaration is incomplete")

    if deployment and node.capability_source_id is None:
        issues.append(f"node {node.id} has no capability_source_id")

    for field_name in CRITICAL_NODE_MEASUREMENTS:
        provenance = node.provenance_for(field_name)
        label = f"node {node.id} {field_name}"
        if provenance.status == "unsupported":
            issues.append(f"{label} is explicitly unsupported")
        elif deployment:
            issues.extend(
                _measurement_deployment_issues(
                    label, provenance, target.max_expected_relative_error
                )
            )
    return _unique(issues)


def link_admission_issues(
    link: Link, target: Target, *, deployment: bool
) -> tuple[str, ...]:
    issues: list[str] = list(_link_numeric_issues(link))
    for field_name in CRITICAL_LINK_MEASUREMENTS:
        provenance = link.provenance_for(field_name)
        label = f"link {link.a}<->{link.b} {field_name}"
        if provenance.status == "unsupported":
            issues.append(f"{label} is explicitly unsupported")
        elif deployment:
            issues.extend(
                _measurement_deployment_issues(
                    label, provenance, target.max_expected_relative_error
                )
            )
    return _unique(issues)


def _selected_nodes(
    inventory: Inventory, stages: tuple[Stage, ...]
) -> tuple[Node, ...]:
    node_ids = {
        node_id
        for stage in stages
        for node_id in stage.replicas + stage.expert_hosts
    }
    return tuple(inventory.node(node_id) for node_id in sorted(node_ids))


def _evidence_for_plan(
    target: Target,
    nodes: tuple[Node, ...],
    boundary_links: tuple[BoundaryLink, ...],
) -> tuple[MeasurementProvenance, ...]:
    evidence: list[MeasurementProvenance] = [target.prediction_calibration]
    for node in nodes:
        evidence.extend(
            node.provenance_for(field_name)
            for field_name in CRITICAL_NODE_MEASUREMENTS
        )
    for boundary in boundary_links:
        evidence.extend(
            boundary.link.provenance_for(field_name)
            for field_name in CRITICAL_LINK_MEASUREMENTS
        )
    return tuple(evidence)


def assess_plan_authority(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    stages: tuple[Stage, ...],
    boundary_links: tuple[BoundaryLink, ...],
    *,
    requested_mode: str,
    evidence_registry: EvidenceRegistry | None = None,
    additional_reasons: Iterable[str] = (),
) -> PlanAuthority:
    selected_nodes = _selected_nodes(inventory, stages)
    readiness_issues: list[str] = list(global_deployment_issues(model, target))
    readiness_issues.extend(
        evidence_registry_deployment_issues(
            model,
            inventory,
            target,
            evidence_registry,
        )
    )
    for node in selected_nodes:
        readiness_issues.extend(
            node_admission_issues(node, model, target, deployment=True)
        )
    for boundary in boundary_links:
        readiness_issues.extend(
            link_admission_issues(boundary.link, target, deployment=True)
        )
    if any(stage.mode == "remote_experts" for stage in stages):
        readiness_issues.append(
            "remote-expert placement is not deployment-authoritative until "
            "contention and route calibration close"
        )
        if model.expert_trace_source_id is None:
            readiness_issues.append(
                "remote-expert placement has no expert_trace_source_id"
            )
    readiness_issues.extend(additional_reasons)
    readiness_issues = list(_unique(readiness_issues))

    evidence = _evidence_for_plan(target, selected_nodes, boundary_links)
    confidences = [item.confidence for item in evidence]
    confidence = (
        min(confidences, key=lambda value: _CONFIDENCE_RANK[value])
        if confidences
        else "unknown"
    )
    input_errors = [
        item.expected_relative_error
        for item in evidence[1:]
        if item.expected_relative_error is not None
    ]
    source_ids: list[str] = [
        value
        for value in (
            model.source_id,
            model.quantization_source_id,
            model.expert_trace_source_id,
            target.prediction_calibration.source_id,
        )
        if value is not None
    ]
    # Candidate declarations participate in ranking/exclusion even when their
    # node is not in the winning stage list, so expose the full source set that
    # the deployment search resolved.
    for node in inventory.nodes:
        if node.capability_source_id is not None:
            source_ids.append(node.capability_source_id)
        source_ids.extend(
            provenance.source_id
            for _, provenance in node.measurement_provenance
            if provenance.source_id is not None
        )
    for link in inventory.links:
        source_ids.extend(
            provenance.source_id
            for _, provenance in link.measurement_provenance
            if provenance.source_id is not None
        )

    if requested_mode == "exploratory":
        status = "exploratory"
        deployment_authorized = False
        reasons = _unique(("exploratory mode requested", *readiness_issues))
    elif readiness_issues:
        status = "rejected"
        deployment_authorized = False
        reasons = tuple(readiness_issues)
    else:
        status = "deployment_authoritative"
        deployment_authorized = True
        reasons = ("all deployment authority checks passed",)

    return PlanAuthority(
        requested_mode=requested_mode,
        status=status,
        deployment_authorized=deployment_authorized,
        confidence=confidence,
        prediction_expected_relative_error=(
            target.prediction_calibration.expected_relative_error
        ),
        input_max_expected_relative_error=(
            max(input_errors) if input_errors else None
        ),
        source_ids=tuple(sorted(set(source_ids))),
        evidence_registry_sha256=(
            evidence_registry.manifest_sha256
            if evidence_registry is not None
            else None
        ),
        reasons=reasons,
    )
