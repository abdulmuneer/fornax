from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "ops-lifecycle-simulation-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "operator-lifecycle-two-logical-hosts"
REQUIRED_ACTIONS = {
    "deploy",
    "upgrade",
    "drain",
    "restart",
    "rollback",
    "node_replace",
}
REQUIRED_EVENT_KINDS = {
    "config_loaded",
    "deploy_started",
    "node_admitted",
    "health_check_passed",
    "drain_started",
    "drain_completed",
    "upgrade_started",
    "upgrade_completed",
    "restart_started",
    "restart_completed",
    "rollback_started",
    "rollback_completed",
    "node_replace_started",
    "node_removed",
    "node_replace_completed",
    "traffic_restored",
    "lifecycle_completed",
}


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _positive_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return None
    return value


def _non_negative_number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{field} must be a non-negative number")
        return None
    return float(value)


def _event(
    kind: str,
    *,
    timestamp_s: float,
    plan_id: str,
    action: str,
    **fields: Any,
) -> dict[str, Any]:
    event = {
        "kind": kind,
        "timestamp_s": round(timestamp_s, 9),
        "plan_id": plan_id,
        "action": action,
    }
    event.update(fields)
    return event


def _action(
    name: str,
    *,
    started_at_s: float,
    completed_at_s: float,
    affected_nodes: list[str],
    preconditions: list[str],
    postconditions: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "completed",
        "started_at_s": round(started_at_s, 9),
        "completed_at_s": round(completed_at_s, 9),
        "affected_nodes": affected_nodes,
        "preconditions": preconditions,
        "postconditions": postconditions,
    }


def simulate_ops_lifecycle(
    *,
    plan_id: str = "ops-lifecycle-plan",
    cluster_id: str = "fornax-sim-cluster",
    model_id: str = "qwen3-moe-class-target",
    initial_version: str = "v0.1.0",
    target_version: str = "v0.2.0",
    node_ids: list[str] | None = None,
    replacement_node_id: str = "logical-host-2",
    in_flight_requests: int = 4,
) -> dict[str, Any]:
    if node_ids is None:
        node_ids = ["logical-host-0", "logical-host-1"]
    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(cluster_id, "cluster_id", errors)
    _non_empty_string(model_id, "model_id", errors)
    _non_empty_string(initial_version, "initial_version", errors)
    _non_empty_string(target_version, "target_version", errors)
    _non_empty_string(replacement_node_id, "replacement_node_id", errors)
    _positive_int(in_flight_requests, "in_flight_requests", errors)
    if not isinstance(node_ids, list) or len(node_ids) < 2:
        errors.append("node_ids must contain at least two nodes")
        node_ids = []
    else:
        normalized_node_ids: list[str] = []
        for index, node_id in enumerate(node_ids):
            if not isinstance(node_id, str) or not node_id:
                errors.append(f"node_ids[{index}] must be a non-empty string")
            else:
                normalized_node_ids.append(node_id)
        if len(normalized_node_ids) == len(node_ids):
            if len(set(normalized_node_ids)) != len(normalized_node_ids):
                errors.append("node_ids must be unique")
            node_ids = normalized_node_ids
    if replacement_node_id in set(node_ids):
        errors.append("replacement_node_id must not already be in node_ids")
    if initial_version == target_version:
        errors.append("initial_version and target_version must differ")
    if errors:
        raise ValueError("; ".join(errors))

    primary, secondary = node_ids[0], node_ids[1]
    cluster_config = {
        "cluster_id": cluster_id,
        "nodes": [
            {
                "node_id": node_id,
                "logical_host_id": node_id,
                "role": "stage-worker" if index == 0 else "stage-replica",
                "admission": "managed",
            }
            for index, node_id in enumerate(node_ids)
        ],
        "auth": {"mode": "simulated-node-identity", "required": True},
        "plan_integrity": {"plan_id": plan_id, "required": True},
    }
    model_config = {
        "model_id": model_id,
        "initial_version": initial_version,
        "target_version": target_version,
        "serving": {"endpoint": "/v1/chat/completions", "engine": "FornaxBackend"},
    }
    placement = {
        "plan_id": plan_id,
        "feasible": True,
        "stages": [
            {"index": 0, "layers": [0], "replicas": [primary]},
            {"index": 1, "layers": [1], "replicas": [secondary]},
        ],
    }
    events: list[dict[str, Any]] = [
        _event(
            "config_loaded",
            timestamp_s=0.000,
            plan_id=plan_id,
            action="deploy",
            config_artifacts=["cluster.yaml", "model.yaml", "placement.json"],
        ),
        _event("deploy_started", timestamp_s=0.001, plan_id=plan_id, action="deploy"),
        *[
            _event(
                "node_admitted",
                timestamp_s=0.002 + index * 0.001,
                plan_id=plan_id,
                action="deploy",
                node_id=node_id,
                version=initial_version,
            )
            for index, node_id in enumerate(node_ids)
        ],
        _event(
            "health_check_passed",
            timestamp_s=0.005,
            plan_id=plan_id,
            action="deploy",
            checked_nodes=node_ids,
            active_version=initial_version,
        ),
        _event(
            "drain_started",
            timestamp_s=0.010,
            plan_id=plan_id,
            action="upgrade",
            node_id=primary,
            in_flight_before=in_flight_requests,
        ),
        _event(
            "drain_completed",
            timestamp_s=0.012,
            plan_id=plan_id,
            action="upgrade",
            node_id=primary,
            in_flight_before=in_flight_requests,
            drained_request_count=in_flight_requests,
            dropped_in_flight_count=0,
        ),
        _event(
            "upgrade_started",
            timestamp_s=0.013,
            plan_id=plan_id,
            action="upgrade",
            node_id=primary,
            from_version=initial_version,
            to_version=target_version,
        ),
        _event(
            "upgrade_completed",
            timestamp_s=0.016,
            plan_id=plan_id,
            action="upgrade",
            node_id=primary,
            active_version=target_version,
        ),
        _event(
            "health_check_passed",
            timestamp_s=0.017,
            plan_id=plan_id,
            action="upgrade",
            checked_nodes=[primary],
            active_version=target_version,
        ),
        _event(
            "drain_started",
            timestamp_s=0.020,
            plan_id=plan_id,
            action="restart",
            node_id=secondary,
            in_flight_before=in_flight_requests,
        ),
        _event(
            "drain_completed",
            timestamp_s=0.022,
            plan_id=plan_id,
            action="restart",
            node_id=secondary,
            in_flight_before=in_flight_requests,
            drained_request_count=in_flight_requests,
            dropped_in_flight_count=0,
        ),
        _event(
            "restart_started",
            timestamp_s=0.023,
            plan_id=plan_id,
            action="restart",
            node_id=secondary,
            active_version=initial_version,
        ),
        _event(
            "restart_completed",
            timestamp_s=0.026,
            plan_id=plan_id,
            action="restart",
            node_id=secondary,
            active_version=initial_version,
        ),
        _event(
            "health_check_passed",
            timestamp_s=0.027,
            plan_id=plan_id,
            action="restart",
            checked_nodes=[secondary],
            active_version=initial_version,
        ),
        _event(
            "drain_started",
            timestamp_s=0.030,
            plan_id=plan_id,
            action="rollback",
            node_id=primary,
            in_flight_before=in_flight_requests,
        ),
        _event(
            "drain_completed",
            timestamp_s=0.032,
            plan_id=plan_id,
            action="rollback",
            node_id=primary,
            in_flight_before=in_flight_requests,
            drained_request_count=in_flight_requests,
            dropped_in_flight_count=0,
        ),
        _event(
            "rollback_started",
            timestamp_s=0.033,
            plan_id=plan_id,
            action="rollback",
            node_id=primary,
            from_version=target_version,
            to_version=initial_version,
        ),
        _event(
            "rollback_completed",
            timestamp_s=0.036,
            plan_id=plan_id,
            action="rollback",
            node_id=primary,
            active_version=initial_version,
        ),
        _event(
            "health_check_passed",
            timestamp_s=0.037,
            plan_id=plan_id,
            action="rollback",
            checked_nodes=[primary],
            active_version=initial_version,
        ),
        _event(
            "drain_started",
            timestamp_s=0.040,
            plan_id=plan_id,
            action="node_replace",
            node_id=secondary,
            in_flight_before=in_flight_requests,
        ),
        _event(
            "drain_completed",
            timestamp_s=0.042,
            plan_id=plan_id,
            action="node_replace",
            node_id=secondary,
            in_flight_before=in_flight_requests,
            drained_request_count=in_flight_requests,
            dropped_in_flight_count=0,
        ),
        _event(
            "node_replace_started",
            timestamp_s=0.043,
            plan_id=plan_id,
            action="node_replace",
            old_node_id=secondary,
            replacement_node_id=replacement_node_id,
        ),
        _event(
            "node_removed",
            timestamp_s=0.044,
            plan_id=plan_id,
            action="node_replace",
            node_id=secondary,
        ),
        _event(
            "node_admitted",
            timestamp_s=0.045,
            plan_id=plan_id,
            action="node_replace",
            node_id=replacement_node_id,
            version=initial_version,
        ),
        _event(
            "node_replace_completed",
            timestamp_s=0.047,
            plan_id=plan_id,
            action="node_replace",
            old_node_id=secondary,
            replacement_node_id=replacement_node_id,
        ),
        _event(
            "traffic_restored",
            timestamp_s=0.048,
            plan_id=plan_id,
            action="node_replace",
            active_nodes=[primary, replacement_node_id],
        ),
        _event(
            "lifecycle_completed",
            timestamp_s=0.050,
            plan_id=plan_id,
            action="lifecycle",
            active_nodes=[primary, replacement_node_id],
        ),
    ]
    actions = [
        _action(
            "deploy",
            started_at_s=0.000,
            completed_at_s=0.005,
            affected_nodes=node_ids,
            preconditions=["cluster.yaml", "model.yaml", "placement.json"],
            postconditions=["nodes admitted", "health checks passed"],
        ),
        _action(
            "drain",
            started_at_s=0.010,
            completed_at_s=0.042,
            affected_nodes=[primary, secondary],
            preconditions=["bounded queue", "active placement"],
            postconditions=["zero dropped in-flight", "traffic quiesced before mutation"],
        ),
        _action(
            "upgrade",
            started_at_s=0.010,
            completed_at_s=0.017,
            affected_nodes=[primary],
            preconditions=["drain completed"],
            postconditions=["target version healthy"],
        ),
        _action(
            "restart",
            started_at_s=0.020,
            completed_at_s=0.027,
            affected_nodes=[secondary],
            preconditions=["drain completed"],
            postconditions=["node rejoined at same version"],
        ),
        _action(
            "rollback",
            started_at_s=0.030,
            completed_at_s=0.037,
            affected_nodes=[primary],
            preconditions=["drain completed", "previous version available"],
            postconditions=["initial version restored", "health checks passed"],
        ),
        _action(
            "node_replace",
            started_at_s=0.040,
            completed_at_s=0.048,
            affected_nodes=[secondary, replacement_node_id],
            preconditions=["drain completed", "replacement admitted"],
            postconditions=["old node removed", "traffic restored"],
        ),
    ]
    final_state = {
        "active_nodes": [
            {"node_id": primary, "version": initial_version, "state": "serving"},
            {"node_id": replacement_node_id, "version": initial_version, "state": "serving"},
        ],
        "removed_nodes": [{"node_id": secondary, "state": "removed"}],
        "active_version": initial_version,
    }
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "simulation_method": SIMULATION_METHOD,
        "operator_configs": {
            "cluster.yaml": cluster_config,
            "model.yaml": model_config,
            "placement.json": placement,
        },
        "actions": actions,
        "events": events,
        "request_accounting": {
            "in_flight_before_each_drain": in_flight_requests,
            "dropped_in_flight_total": 0,
            "drain_event_count": 4,
        },
        "final_state": final_state,
        "summary": {
            "action_count": len(actions),
            "completed_action_count": len(actions),
            "event_count": len(events),
            "config_artifacts_present": True,
            "drain_before_mutation": True,
            "dropped_in_flight_count": 0,
            "rollback_verified": True,
            "node_replace_verified": True,
            "active_node_count": len(final_state["active_nodes"]),
            "correctness_passed": True,
        },
        "note": (
            "T1 ops lifecycle simulation: validates operator deploy, upgrade, drain, "
            "restart, rollback, and node replacement semantics over logical hosts. "
            "Not a G5 product-ops closure claim."
        ),
    }


def _validate_configs(data: dict[str, Any], errors: list[str]) -> None:
    configs = data.get("operator_configs")
    if not isinstance(configs, dict):
        errors.append("operator_configs must be an object")
        return
    for name in ("cluster.yaml", "model.yaml", "placement.json"):
        if not isinstance(configs.get(name), dict):
            errors.append(f"operator_configs.{name} must be an object")
    cluster = configs.get("cluster.yaml") if isinstance(configs.get("cluster.yaml"), dict) else {}
    if not isinstance(cluster.get("nodes"), list) or len(cluster.get("nodes", [])) < 2:
        errors.append("operator_configs.cluster.yaml.nodes must contain at least two nodes")
    if cluster.get("auth", {}).get("required") is not True:
        errors.append("operator_configs.cluster.yaml.auth.required must be true")
    if cluster.get("plan_integrity", {}).get("required") is not True:
        errors.append("operator_configs.cluster.yaml.plan_integrity.required must be true")
    model = configs.get("model.yaml") if isinstance(configs.get("model.yaml"), dict) else {}
    _non_empty_string(model.get("model_id"), "operator_configs.model.yaml.model_id", errors)
    if model.get("initial_version") == model.get("target_version"):
        errors.append("operator_configs.model.yaml versions must differ")
    placement = configs.get("placement.json") if isinstance(configs.get("placement.json"), dict) else {}
    if placement.get("feasible") is not True:
        errors.append("operator_configs.placement.json.feasible must be true")


def _validate_actions(data: dict[str, Any], errors: list[str]) -> set[str]:
    actions = data.get("actions")
    if not isinstance(actions, list) or not actions:
        errors.append("actions must be a non-empty list")
        return set()
    names: set[str] = set()
    for index, action in enumerate(actions):
        field = f"actions[{index}]"
        if not isinstance(action, dict):
            errors.append(f"{field} must be an object")
            continue
        name = _non_empty_string(action.get("name"), f"{field}.name", errors)
        if name is not None:
            names.add(name)
        if action.get("status") != "completed":
            errors.append(f"{field}.status must be completed")
        started = _non_negative_number(action.get("started_at_s"), f"{field}.started_at_s", errors)
        completed = _non_negative_number(action.get("completed_at_s"), f"{field}.completed_at_s", errors)
        if started is not None and completed is not None and completed < started:
            errors.append(f"{field}.completed_at_s must be >= started_at_s")
        if not isinstance(action.get("affected_nodes"), list) or not action["affected_nodes"]:
            errors.append(f"{field}.affected_nodes must be a non-empty list")
        if not isinstance(action.get("preconditions"), list):
            errors.append(f"{field}.preconditions must be a list")
        if not isinstance(action.get("postconditions"), list):
            errors.append(f"{field}.postconditions must be a list")
    missing = REQUIRED_ACTIONS - names
    if missing:
        errors.append(f"actions missing required names: {sorted(missing)}")
    return names


def _validate_events(data: dict[str, Any], plan_id: str | None, errors: list[str]) -> None:
    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        return
    event_kinds = {event.get("kind") for event in events if isinstance(event, dict)}
    missing = REQUIRED_EVENT_KINDS - event_kinds
    if missing:
        errors.append(f"events missing required kinds: {sorted(missing)}")
    drained_by_node: dict[str, int] = {}
    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        kind = _non_empty_string(event.get("kind"), f"{field}.kind", errors)
        if event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match plan_id")
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        _non_empty_string(event.get("action"), f"{field}.action", errors)
        node_id = event.get("node_id")
        if kind == "drain_completed":
            if not isinstance(node_id, str):
                errors.append(f"{field}.node_id must be set for drain_completed")
            else:
                drained_by_node[node_id] = index
            if event.get("dropped_in_flight_count") != 0:
                errors.append(f"{field}.dropped_in_flight_count must be 0")
            if event.get("drained_request_count") != event.get("in_flight_before"):
                errors.append(f"{field}.drained_request_count must equal in_flight_before")
        if kind in {"upgrade_started", "restart_started", "rollback_started", "node_removed"}:
            if not isinstance(node_id, str):
                errors.append(f"{field}.node_id must be set for {kind}")
            elif node_id not in drained_by_node or drained_by_node[node_id] > index:
                errors.append(f"{field}.{kind} must occur after drain_completed for node")
        if kind == "node_replace_started":
            old_node_id = event.get("old_node_id")
            if not isinstance(old_node_id, str):
                errors.append(f"{field}.old_node_id must be set")
            elif old_node_id not in drained_by_node or drained_by_node[old_node_id] > index:
                errors.append(f"{field}.node_replace_started must occur after drain_completed for old node")
            _non_empty_string(event.get("replacement_node_id"), f"{field}.replacement_node_id", errors)
        if kind == "health_check_passed":
            if not isinstance(event.get("checked_nodes"), list) or not event["checked_nodes"]:
                errors.append(f"{field}.checked_nodes must be a non-empty list")


def validate_ops_lifecycle_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if data.get("simulation_method") != SIMULATION_METHOD:
        errors.append(f"simulation_method must be {SIMULATION_METHOD}")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    _validate_configs(data, errors)
    action_names = _validate_actions(data, errors)
    _validate_events(data, plan_id, errors)

    request_accounting = data.get("request_accounting")
    if not isinstance(request_accounting, dict):
        errors.append("request_accounting must be an object")
        request_accounting = {}
    if request_accounting.get("dropped_in_flight_total") != 0:
        errors.append("request_accounting.dropped_in_flight_total must be 0")
    _positive_int(request_accounting.get("in_flight_before_each_drain"), "request_accounting.in_flight_before_each_drain", errors)
    events = data.get("events") if isinstance(data.get("events"), list) else []
    drain_event_count = sum(
        1
        for event in events
        if isinstance(event, dict) and event.get("kind") == "drain_completed"
    )
    if request_accounting.get("drain_event_count") != drain_event_count:
        errors.append("request_accounting.drain_event_count must equal drain_completed count")

    final_state = data.get("final_state")
    if not isinstance(final_state, dict):
        errors.append("final_state must be an object")
        final_state = {}
    active_nodes = final_state.get("active_nodes")
    removed_nodes = final_state.get("removed_nodes")
    active_node_count = len(active_nodes) if isinstance(active_nodes, list) else None
    if not isinstance(active_nodes, list) or len(active_nodes) < 2:
        errors.append("final_state.active_nodes must contain at least two nodes")
    if not isinstance(removed_nodes, list) or not removed_nodes:
        errors.append("final_state.removed_nodes must be a non-empty list")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    actions = data.get("actions") if isinstance(data.get("actions"), list) else []
    events = data.get("events") if isinstance(data.get("events"), list) else []
    if summary.get("action_count") != len(actions):
        errors.append("summary.action_count must equal len(actions)")
    if summary.get("completed_action_count") != len(actions):
        errors.append("summary.completed_action_count must equal len(actions)")
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must equal len(events)")
    if summary.get("active_node_count") != active_node_count:
        errors.append("summary.active_node_count must equal len(final_state.active_nodes)")
    if summary.get("config_artifacts_present") is not True:
        errors.append("summary.config_artifacts_present must be true")
    if summary.get("drain_before_mutation") is not True:
        errors.append("summary.drain_before_mutation must be true")
    if summary.get("dropped_in_flight_count") != 0:
        errors.append("summary.dropped_in_flight_count must be 0")
    if summary.get("rollback_verified") is not True:
        errors.append("summary.rollback_verified must be true")
    if summary.get("node_replace_verified") is not True:
        errors.append("summary.node_replace_verified must be true")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")
    if not REQUIRED_ACTIONS.issubset(action_names):
        errors.append("summary cannot pass unless every lifecycle action is represented")
    warnings.append("ops lifecycle is simulation evidence, not G5 product-ops closure evidence")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "action_count": summary.get("action_count"),
            "completed_action_count": summary.get("completed_action_count"),
            "event_count": summary.get("event_count"),
            "dropped_in_flight_count": summary.get("dropped_in_flight_count"),
            "rollback_verified": summary.get("rollback_verified") is True,
            "node_replace_verified": summary.get("node_replace_verified") is True,
            "correctness_passed": summary.get("correctness_passed") is True,
        },
    }


def validate_ops_lifecycle(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid ops lifecycle artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["ops lifecycle artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_ops_lifecycle_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
