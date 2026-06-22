from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "state-ownership-simulation-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "end-to-end-serving-state-ownership"
REQUIRED_EVENT_KINDS = {
    "request_received",
    "engine_request_normalized",
    "scheduler_admitted",
    "microbatch_formed",
    "activation_handoff_sent",
    "activation_handoff_received",
    "kv_read_granted",
    "kv_write_returned",
    "stream_opened",
    "response_ready",
    "cancellation_propagated",
    "cleanup",
}
REQUIRED_RESOURCE_KINDS = {
    "request_envelope",
    "engine_context",
    "scheduler_slot",
    "microbatch",
    "activation_buffer",
    "kv_cache",
    "transport_payload",
    "response_stream",
}
KNOWN_OWNERS = {
    "unowned",
    "serving_gateway",
    "fornax_engine",
    "scheduler",
    "stage-0",
    "stage-1",
    "transport",
    "kv_manager",
    "released",
}


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _non_negative_number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{field} must be a non-negative number")
        return None
    return float(value)


def _transition(
    resource_id: str,
    *,
    event: str,
    timestamp_s: float,
    from_owner: str,
    to_owner: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "resource_id": resource_id,
        "event": event,
        "timestamp_s": round(timestamp_s, 6),
        "from_owner": from_owner,
        "to_owner": to_owner,
        "reason": reason,
    }


def _claim(resource_id: str, owner: str, *, state: str = "active") -> dict[str, Any]:
    return {"resource_id": resource_id, "owner": owner, "state": state}


def simulate_state_ownership(
    *,
    plan_id: str = "state-ownership-plan",
    request_id: str = "req-state-ownership",
    cancel_request_id: str = "req-state-ownership-cancel",
    model_id: str = "qwen3-moe-class-target",
) -> dict[str, Any]:
    """Simulate end-to-end request/resource ownership across serving components."""

    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(request_id, "request_id", errors)
    _non_empty_string(cancel_request_id, "cancel_request_id", errors)
    _non_empty_string(model_id, "model_id", errors)
    if request_id == cancel_request_id:
        errors.append("request_id and cancel_request_id must differ")
    if errors:
        raise ValueError("; ".join(errors))

    resources = [
        {"resource_id": "request:primary", "kind": "request_envelope", "initial_owner": "unowned", "required": True},
        {"resource_id": "engine-context:primary", "kind": "engine_context", "initial_owner": "unowned", "required": True},
        {"resource_id": "scheduler-slot:primary", "kind": "scheduler_slot", "initial_owner": "unowned", "required": True},
        {"resource_id": "microbatch:primary", "kind": "microbatch", "initial_owner": "unowned", "required": True},
        {"resource_id": "activation:primary", "kind": "activation_buffer", "initial_owner": "unowned", "required": True},
        {"resource_id": "transport-payload:primary", "kind": "transport_payload", "initial_owner": "unowned", "required": True},
        {"resource_id": "kv-cache:primary", "kind": "kv_cache", "initial_owner": "unowned", "required": True},
        {"resource_id": "response-stream:primary", "kind": "response_stream", "initial_owner": "unowned", "required": True},
        {"resource_id": "request:cancel", "kind": "request_envelope", "initial_owner": "unowned", "required": True},
        {"resource_id": "scheduler-slot:cancel", "kind": "scheduler_slot", "initial_owner": "unowned", "required": True},
        {"resource_id": "response-stream:cancel", "kind": "response_stream", "initial_owner": "unowned", "required": True},
    ]
    transitions = [
        _transition("request:primary", event="request_received", timestamp_s=0.001, from_owner="unowned", to_owner="serving_gateway", reason="OpenAI request accepted"),
        _transition("request:primary", event="engine_request_normalized", timestamp_s=0.002, from_owner="serving_gateway", to_owner="fornax_engine", reason="request normalized into EngineRequest"),
        _transition("engine-context:primary", event="engine_request_normalized", timestamp_s=0.002, from_owner="unowned", to_owner="fornax_engine", reason="engine context created"),
        _transition("request:primary", event="scheduler_admitted", timestamp_s=0.003, from_owner="fornax_engine", to_owner="scheduler", reason="scheduler owns admission and queue state"),
        _transition("scheduler-slot:primary", event="scheduler_admitted", timestamp_s=0.003, from_owner="unowned", to_owner="scheduler", reason="bounded queue slot allocated"),
        _transition("microbatch:primary", event="microbatch_formed", timestamp_s=0.004, from_owner="unowned", to_owner="scheduler", reason="scheduler batches request"),
        _transition("microbatch:primary", event="stage0_started", timestamp_s=0.005, from_owner="scheduler", to_owner="stage-0", reason="stage worker owns execution"),
        _transition("activation:primary", event="activation_produced", timestamp_s=0.006, from_owner="unowned", to_owner="stage-0", reason="stage-0 owns produced activation"),
        _transition("activation:primary", event="activation_handoff_sent", timestamp_s=0.007, from_owner="stage-0", to_owner="transport", reason="transport owns in-flight activation"),
        _transition("transport-payload:primary", event="activation_handoff_sent", timestamp_s=0.007, from_owner="unowned", to_owner="transport", reason="payload registered for handoff"),
        _transition("activation:primary", event="activation_handoff_received", timestamp_s=0.008, from_owner="transport", to_owner="stage-1", reason="destination stage owns activation"),
        _transition("transport-payload:primary", event="activation_handoff_received", timestamp_s=0.008, from_owner="transport", to_owner="released", reason="payload acked"),
        _transition("kv-cache:primary", event="kv_read_granted", timestamp_s=0.009, from_owner="unowned", to_owner="kv_manager", reason="KV cache opened by manager"),
        _transition("kv-cache:primary", event="kv_read_granted", timestamp_s=0.010, from_owner="kv_manager", to_owner="stage-1", reason="stage-1 receives read lease"),
        _transition("kv-cache:primary", event="kv_write_returned", timestamp_s=0.011, from_owner="stage-1", to_owner="kv_manager", reason="stage returns write ownership"),
        _transition("activation:primary", event="stage1_completed", timestamp_s=0.012, from_owner="stage-1", to_owner="released", reason="activation buffer released after stage completion"),
        _transition("microbatch:primary", event="stage1_completed", timestamp_s=0.012, from_owner="stage-0", to_owner="released", reason="microbatch state completed"),
        _transition("scheduler-slot:primary", event="response_ready", timestamp_s=0.013, from_owner="scheduler", to_owner="released", reason="queue slot released"),
        _transition("request:primary", event="response_ready", timestamp_s=0.014, from_owner="scheduler", to_owner="fornax_engine", reason="engine owns final result assembly"),
        _transition("request:primary", event="response_ready", timestamp_s=0.015, from_owner="fornax_engine", to_owner="serving_gateway", reason="serving gateway owns response emission"),
        _transition("response-stream:primary", event="stream_opened", timestamp_s=0.015, from_owner="unowned", to_owner="serving_gateway", reason="stream state opened"),
        _transition("response-stream:primary", event="cleanup", timestamp_s=0.016, from_owner="serving_gateway", to_owner="released", reason="stream finalized"),
        _transition("request:primary", event="cleanup", timestamp_s=0.017, from_owner="serving_gateway", to_owner="released", reason="primary request released"),
        _transition("engine-context:primary", event="cleanup", timestamp_s=0.017, from_owner="fornax_engine", to_owner="released", reason="engine context released"),
        _transition("kv-cache:primary", event="cleanup", timestamp_s=0.017, from_owner="kv_manager", to_owner="released", reason="KV lease closed"),
        _transition("request:cancel", event="request_received", timestamp_s=0.020, from_owner="unowned", to_owner="serving_gateway", reason="cancelable request accepted"),
        _transition("request:cancel", event="engine_request_normalized", timestamp_s=0.021, from_owner="serving_gateway", to_owner="fornax_engine", reason="cancel path normalized"),
        _transition("request:cancel", event="scheduler_admitted", timestamp_s=0.022, from_owner="fornax_engine", to_owner="scheduler", reason="cancel path admitted"),
        _transition("scheduler-slot:cancel", event="scheduler_admitted", timestamp_s=0.022, from_owner="unowned", to_owner="scheduler", reason="cancel path queue slot allocated"),
        _transition("response-stream:cancel", event="stream_opened", timestamp_s=0.023, from_owner="unowned", to_owner="serving_gateway", reason="cancel stream opened"),
        _transition("request:cancel", event="cancellation_propagated", timestamp_s=0.024, from_owner="scheduler", to_owner="serving_gateway", reason="scheduler returns canceled request to serving"),
        _transition("scheduler-slot:cancel", event="cancellation_propagated", timestamp_s=0.024, from_owner="scheduler", to_owner="released", reason="cancel releases scheduler slot"),
        _transition("response-stream:cancel", event="cleanup", timestamp_s=0.025, from_owner="serving_gateway", to_owner="released", reason="cancel stream released"),
        _transition("request:cancel", event="cleanup", timestamp_s=0.025, from_owner="serving_gateway", to_owner="released", reason="cancel request released"),
    ]
    snapshots = [
        {
            "snapshot_id": "after-scheduler-admit",
            "timestamp_s": 0.003,
            "claims": [
                _claim("request:primary", "scheduler"),
                _claim("scheduler-slot:primary", "scheduler"),
                _claim("engine-context:primary", "fornax_engine"),
            ],
        },
        {
            "snapshot_id": "activation-in-flight",
            "timestamp_s": 0.007,
            "claims": [
                _claim("request:primary", "scheduler"),
                _claim("activation:primary", "transport"),
                _claim("transport-payload:primary", "transport"),
            ],
        },
        {
            "snapshot_id": "response-owned-by-serving",
            "timestamp_s": 0.015,
            "claims": [
                _claim("request:primary", "serving_gateway"),
                _claim("response-stream:primary", "serving_gateway"),
                _claim("kv-cache:primary", "kv_manager"),
            ],
        },
        {
            "snapshot_id": "all-released",
            "timestamp_s": 0.030,
            "claims": [
                _claim(resource["resource_id"], "released", state="released")
                for resource in resources
            ],
        },
    ]
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "request_id": request_id,
        "cancel_request_id": cancel_request_id,
        "model_id": model_id,
        "simulation_method": SIMULATION_METHOD,
        "target_workstreams": ["WS-H1", "WS-G1", "WS-E4"],
        "ownership_policy": {
            "single_active_owner": True,
            "terminal_owner": "released",
            "owners": sorted(KNOWN_OWNERS),
            "cleanup_required": True,
        },
        "resources": resources,
        "ownership_transitions": transitions,
        "ownership_snapshots": snapshots,
        "result": {
            "single_owner_preserved": True,
            "all_required_resources_released": True,
            "normal_request_completed": True,
            "cancel_request_cleaned": True,
            "dual_owner_detected": False,
            "stale_resource_count": 0,
            "correctness_passed": True,
        },
        "summary": {
            "resource_count": len(resources),
            "transition_count": len(transitions),
            "snapshot_count": len(snapshots),
            "terminal_released_count": len(resources),
            "stale_resource_count": 0,
            "dual_owner_detected": False,
            "normal_request_terminal_owner": "released",
            "cancel_request_terminal_owner": "released",
            "correctness_passed": True,
        },
        "note": (
            "T1 state-ownership simulation: validates single-owner request, "
            "scheduler, transport, KV, activation, stream, and cancellation "
            "lifecycle semantics without claiming live serving runtime evidence."
        ),
    }


def _resource_map(resources: Any, errors: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(resources, list) or not resources:
        errors.append("resources must be a non-empty list")
        return {}
    result: dict[str, dict[str, Any]] = {}
    kinds: set[str] = set()
    for index, resource in enumerate(resources):
        field = f"resources[{index}]"
        if not isinstance(resource, dict):
            errors.append(f"{field} must be an object")
            continue
        resource_id = _non_empty_string(resource.get("resource_id"), f"{field}.resource_id", errors)
        kind = _non_empty_string(resource.get("kind"), f"{field}.kind", errors)
        owner = _non_empty_string(resource.get("initial_owner"), f"{field}.initial_owner", errors)
        if kind is not None:
            kinds.add(kind)
        if owner is not None and owner not in KNOWN_OWNERS:
            errors.append(f"{field}.initial_owner must be a known owner")
        if resource.get("required") is not True:
            errors.append(f"{field}.required must be true")
        if resource_id is not None:
            if resource_id in result:
                errors.append(f"duplicate resource_id: {resource_id}")
            result[resource_id] = resource
    missing_kinds = REQUIRED_RESOURCE_KINDS - kinds
    if missing_kinds:
        errors.append(f"resources missing required kinds: {sorted(missing_kinds)}")
    return result


def _validate_transitions(
    transitions: Any,
    resources: dict[str, dict[str, Any]],
    errors: list[str],
) -> tuple[dict[str, str], set[str]]:
    if not isinstance(transitions, list) or not transitions:
        errors.append("ownership_transitions must be a non-empty list")
        return {}, set()
    owners = {
        resource_id: str(resource.get("initial_owner"))
        for resource_id, resource in resources.items()
    }
    seen_events: set[str] = set()
    previous_ts: float | None = None
    for index, transition in enumerate(transitions):
        field = f"ownership_transitions[{index}]"
        if not isinstance(transition, dict):
            errors.append(f"{field} must be an object")
            continue
        resource_id = _non_empty_string(transition.get("resource_id"), f"{field}.resource_id", errors)
        event = _non_empty_string(transition.get("event"), f"{field}.event", errors)
        timestamp = _non_negative_number(transition.get("timestamp_s"), f"{field}.timestamp_s", errors)
        from_owner = _non_empty_string(transition.get("from_owner"), f"{field}.from_owner", errors)
        to_owner = _non_empty_string(transition.get("to_owner"), f"{field}.to_owner", errors)
        _non_empty_string(transition.get("reason"), f"{field}.reason", errors)
        if event is not None:
            seen_events.add(event)
        if timestamp is not None:
            if previous_ts is not None and timestamp < previous_ts:
                errors.append(f"{field}.timestamp_s must be non-decreasing")
            previous_ts = timestamp
        if resource_id is None or resource_id not in resources:
            errors.append(f"{field}.resource_id references unknown resource")
            continue
        if from_owner is not None and from_owner not in KNOWN_OWNERS:
            errors.append(f"{field}.from_owner must be a known owner")
        if to_owner is not None and to_owner not in KNOWN_OWNERS:
            errors.append(f"{field}.to_owner must be a known owner")
        current = owners.get(resource_id)
        if current == "released":
            errors.append(f"{field}.resource_id is already released")
        if from_owner is not None and current != from_owner:
            errors.append(f"{field}.from_owner must match current owner {current}")
        if to_owner is not None:
            owners[resource_id] = to_owner
    missing_events = REQUIRED_EVENT_KINDS - seen_events
    if missing_events:
        errors.append(f"ownership_transitions missing required events: {sorted(missing_events)}")
    return owners, seen_events


def _validate_snapshots(
    snapshots: Any,
    resources: dict[str, dict[str, Any]],
    errors: list[str],
) -> bool:
    if not isinstance(snapshots, list) or not snapshots:
        errors.append("ownership_snapshots must be a non-empty list")
        return False
    dual_owner_detected = False
    for index, snapshot in enumerate(snapshots):
        field = f"ownership_snapshots[{index}]"
        if not isinstance(snapshot, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(snapshot.get("snapshot_id"), f"{field}.snapshot_id", errors)
        _non_negative_number(snapshot.get("timestamp_s"), f"{field}.timestamp_s", errors)
        claims = snapshot.get("claims")
        if not isinstance(claims, list) or not claims:
            errors.append(f"{field}.claims must be a non-empty list")
            continue
        active_by_resource: dict[str, set[str]] = {}
        for claim_index, claim in enumerate(claims):
            claim_field = f"{field}.claims[{claim_index}]"
            if not isinstance(claim, dict):
                errors.append(f"{claim_field} must be an object")
                continue
            resource_id = _non_empty_string(claim.get("resource_id"), f"{claim_field}.resource_id", errors)
            owner = _non_empty_string(claim.get("owner"), f"{claim_field}.owner", errors)
            state = _non_empty_string(claim.get("state"), f"{claim_field}.state", errors)
            if resource_id is not None and resource_id not in resources:
                errors.append(f"{claim_field}.resource_id references unknown resource")
            if owner is not None and owner not in KNOWN_OWNERS:
                errors.append(f"{claim_field}.owner must be a known owner")
            if state not in {"active", "released"}:
                errors.append(f"{claim_field}.state must be active or released")
            if resource_id is not None and owner is not None and state == "active":
                active_by_resource.setdefault(resource_id, set()).add(owner)
        for resource_id, owners in active_by_resource.items():
            if len(owners) > 1:
                dual_owner_detected = True
                errors.append(
                    f"{field} has multiple active owners for {resource_id}: {sorted(owners)}"
                )
    return dual_owner_detected


def validate_state_ownership_fixture(data: dict[str, Any]) -> dict[str, Any]:
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
    _non_empty_string(data.get("plan_id"), "plan_id", errors)
    _non_empty_string(data.get("request_id"), "request_id", errors)
    _non_empty_string(data.get("cancel_request_id"), "cancel_request_id", errors)
    _non_empty_string(data.get("model_id"), "model_id", errors)
    policy = data.get("ownership_policy")
    if not isinstance(policy, dict):
        errors.append("ownership_policy must be an object")
        policy = {}
    if policy.get("single_active_owner") is not True:
        errors.append("ownership_policy.single_active_owner must be true")
    if policy.get("terminal_owner") != "released":
        errors.append("ownership_policy.terminal_owner must be released")
    if policy.get("cleanup_required") is not True:
        errors.append("ownership_policy.cleanup_required must be true")
    resources = _resource_map(data.get("resources"), errors)
    final_owners, _ = _validate_transitions(
        data.get("ownership_transitions"), resources, errors
    )
    dual_owner_detected = _validate_snapshots(
        data.get("ownership_snapshots"), resources, errors
    )
    stale_resources = sorted(
        resource_id for resource_id, owner in final_owners.items() if owner != "released"
    )
    if stale_resources:
        errors.append("required resources not released: " + ", ".join(stale_resources))
    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    if result.get("single_owner_preserved") is not True:
        errors.append("result.single_owner_preserved must be true")
    if result.get("all_required_resources_released") is not True:
        errors.append("result.all_required_resources_released must be true")
    if result.get("normal_request_completed") is not True:
        errors.append("result.normal_request_completed must be true")
    if result.get("cancel_request_cleaned") is not True:
        errors.append("result.cancel_request_cleaned must be true")
    if result.get("dual_owner_detected") is not False:
        errors.append("result.dual_owner_detected must be false")
    if result.get("stale_resource_count") != len(stale_resources):
        errors.append("result.stale_resource_count must match final owners")
    if result.get("correctness_passed") is not True:
        errors.append("result.correctness_passed must be true")
    if dual_owner_detected:
        errors.append("result cannot pass with dual active owners")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    transitions = data.get("ownership_transitions") if isinstance(data.get("ownership_transitions"), list) else []
    snapshots = data.get("ownership_snapshots") if isinstance(data.get("ownership_snapshots"), list) else []
    released_count = sum(1 for owner in final_owners.values() if owner == "released")
    if summary.get("resource_count") != len(resources):
        errors.append("summary.resource_count must match resources")
    if summary.get("transition_count") != len(transitions):
        errors.append("summary.transition_count must match ownership_transitions")
    if summary.get("snapshot_count") != len(snapshots):
        errors.append("summary.snapshot_count must match ownership_snapshots")
    if summary.get("terminal_released_count") != released_count:
        errors.append("summary.terminal_released_count must match final owners")
    if summary.get("stale_resource_count") != len(stale_resources):
        errors.append("summary.stale_resource_count must match final owners")
    if summary.get("dual_owner_detected") is not False:
        errors.append("summary.dual_owner_detected must be false")
    if summary.get("normal_request_terminal_owner") != final_owners.get("request:primary"):
        errors.append("summary.normal_request_terminal_owner must match request:primary")
    if summary.get("cancel_request_terminal_owner") != final_owners.get("request:cancel"):
        errors.append("summary.cancel_request_terminal_owner must match request:cancel")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")

    warnings.append(
        "state ownership is simulation evidence, not live serving runtime evidence"
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "resource_count": summary.get("resource_count"),
            "transition_count": summary.get("transition_count"),
            "snapshot_count": summary.get("snapshot_count"),
            "terminal_released_count": released_count,
            "stale_resource_count": len(stale_resources),
            "dual_owner_detected": dual_owner_detected,
            "normal_request_terminal_owner": final_owners.get("request:primary"),
            "cancel_request_terminal_owner": final_owners.get("request:cancel"),
            "correctness_passed": summary.get("correctness_passed") is True,
        },
    }


def validate_state_ownership(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid state ownership artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["state ownership artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_state_ownership_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
