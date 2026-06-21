from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_EVENT_KINDS = {
    "stage_timing",
    "bubble_fraction",
    "queue_depth",
    "backpressure",
    "router_decision",
    "remote_expert_hit",
    "expert_wait",
    "migration",
    "kv_page_count",
    "memory_pressure",
    "allocation_failure",
    "eviction",
    "replay",
    "placement_explanation",
    "bad_plan_reproduction",
}
STAGE_PHASES = {"prefill", "decode", "transfer", "gather"}
PLACEMENT_DECISIONS = {"selected", "excluded", "demoted"}
MEMORY_PRESSURE_LEVELS = {"normal", "warning", "critical"}


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


def _non_negative_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{field} must be a non-negative integer")
        return None
    return value


def _bounded_fraction(value: Any, field: str, errors: list[str]) -> float | None:
    number = _non_negative_number(value, field, errors)
    if number is not None and number > 1:
        errors.append(f"{field} must be between 0 and 1")
    return number


def _string_list(value: Any, field: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return None
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
            return None
        result.append(item)
    return result


def _int_list(value: Any, field: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            errors.append(f"{field}[{index}] must be an integer")
            return None
        result.append(item)
    return result


def _check_event_identity(
    event: dict[str, Any],
    index: int,
    request_id: str | None,
    plan_id: str | None,
    errors: list[str],
) -> None:
    if request_id is not None and event.get("request_id") != request_id:
        errors.append(f"events[{index}].request_id must match fixture request_id")
    if plan_id is not None and event.get("plan_id") != plan_id:
        errors.append(f"events[{index}].plan_id must match fixture plan_id")


def _check_stage_timing(event: dict[str, Any], field: str, errors: list[str]) -> None:
    _non_negative_int(event.get("stage_id"), f"{field}.stage_id", errors)
    phase = event.get("phase")
    if phase not in STAGE_PHASES:
        errors.append(f"{field}.phase must be one of {sorted(STAGE_PHASES)}")
    _non_negative_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)


def _check_router_event(event: dict[str, Any], field: str, errors: list[str]) -> None:
    _non_negative_int(event.get("layer_id"), f"{field}.layer_id", errors)
    _non_negative_int(event.get("token_id"), f"{field}.token_id", errors)
    _int_list(event.get("expert_ids"), f"{field}.expert_ids", errors)
    if event.get("remote") is not None and not isinstance(event.get("remote"), bool):
        errors.append(f"{field}.remote must be a boolean when present")


def _check_placement_event(event: dict[str, Any], field: str, errors: list[str]) -> None:
    _non_empty_string(event.get("node_id"), f"{field}.node_id", errors)
    decision = event.get("decision")
    if decision not in PLACEMENT_DECISIONS:
        errors.append(
            f"{field}.decision must be one of {sorted(PLACEMENT_DECISIONS)}"
        )
    _non_empty_string(event.get("reason"), f"{field}.reason", errors)


def _check_bad_plan_reproduction(
    event: dict[str, Any], field: str, errors: list[str]
) -> None:
    _non_empty_string(event.get("bad_plan_id"), f"{field}.bad_plan_id", errors)
    _non_empty_string(event.get("expected_error"), f"{field}.expected_error", errors)
    command = _string_list(event.get("command"), f"{field}.command", errors)
    if command is not None and command[:3] != ["python3", "-m", "fornax"]:
        errors.append(f"{field}.command must start with python3 -m fornax")


def _check_event(
    event: dict[str, Any],
    index: int,
    request_id: str | None,
    plan_id: str | None,
    errors: list[str],
    warnings: list[str],
) -> str | None:
    kind = event.get("kind")
    field = f"events[{index}]"
    if not isinstance(kind, str) or not kind:
        errors.append(f"{field}.kind must be a non-empty string")
        return None
    _check_event_identity(event, index, request_id, plan_id, errors)

    if kind == "stage_timing":
        _check_stage_timing(event, field, errors)
    elif kind == "bubble_fraction":
        _bounded_fraction(event.get("value"), f"{field}.value", errors)
    elif kind == "queue_depth":
        _non_negative_int(event.get("depth"), f"{field}.depth", errors)
    elif kind == "backpressure":
        _non_negative_int(event.get("queue_depth"), f"{field}.queue_depth", errors)
        _non_empty_string(event.get("reason"), f"{field}.reason", errors)
    elif kind == "router_decision":
        _check_router_event(event, field, errors)
    elif kind == "remote_expert_hit":
        _non_empty_string(event.get("source_stage"), f"{field}.source_stage", errors)
        _non_empty_string(event.get("target_node"), f"{field}.target_node", errors)
        _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
    elif kind == "expert_wait":
        _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
        _non_negative_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)
    elif kind == "migration":
        _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
        _non_empty_string(event.get("from_node"), f"{field}.from_node", errors)
        _non_empty_string(event.get("to_node"), f"{field}.to_node", errors)
        _non_empty_string(event.get("reason"), f"{field}.reason", errors)
    elif kind == "kv_page_count":
        _non_empty_string(event.get("owner"), f"{field}.owner", errors)
        _non_negative_int(event.get("pages"), f"{field}.pages", errors)
    elif kind == "memory_pressure":
        level = event.get("level")
        if level not in MEMORY_PRESSURE_LEVELS:
            errors.append(
                f"{field}.level must be one of {sorted(MEMORY_PRESSURE_LEVELS)}"
            )
        _non_negative_int(event.get("bytes_used"), f"{field}.bytes_used", errors)
        _non_negative_int(event.get("bytes_limit"), f"{field}.bytes_limit", errors)
    elif kind == "allocation_failure":
        _non_negative_int(event.get("bytes_requested"), f"{field}.bytes_requested", errors)
        _non_empty_string(event.get("reason"), f"{field}.reason", errors)
    elif kind == "eviction":
        _non_empty_string(event.get("page_id"), f"{field}.page_id", errors)
        _non_empty_string(event.get("reason"), f"{field}.reason", errors)
    elif kind == "replay":
        _non_empty_string(event.get("page_id"), f"{field}.page_id", errors)
        _non_empty_string(event.get("source"), f"{field}.source", errors)
    elif kind == "placement_explanation":
        _check_placement_event(event, field, errors)
    elif kind == "bad_plan_reproduction":
        _check_bad_plan_reproduction(event, field, errors)
    else:
        warnings.append(f"unknown observability event kind: {kind}")
    return kind


def _check_reproducible_bad_plan(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("reproducible_bad_plan must be an object")
        return
    _non_empty_string(value.get("bad_plan_id"), "reproducible_bad_plan.bad_plan_id", errors)
    _non_empty_string(
        value.get("expected_error"), "reproducible_bad_plan.expected_error", errors
    )
    fixture_path = _non_empty_string(
        value.get("fixture_path"), "reproducible_bad_plan.fixture_path", errors
    )
    command = _string_list(value.get("command"), "reproducible_bad_plan.command", errors)
    if command is not None and command[:3] != ["python3", "-m", "fornax"]:
        errors.append("reproducible_bad_plan.command must start with python3 -m fornax")
    if command is not None and fixture_path is not None and fixture_path not in command:
        errors.append("reproducible_bad_plan.command must include fixture_path")


def validate_observability_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("contract_kind") != "observability":
        errors.append("contract_kind must be observability")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")

    request_id = _non_empty_string(data.get("request_id"), "request_id", errors)
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    _check_reproducible_bad_plan(data.get("reproducible_bad_plan"), errors)

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []

    seen: set[str] = set()
    stage_ids: set[int] = set()
    placement_decisions: set[str] = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"events[{index}] must be an object")
            continue
        kind = _check_event(event, index, request_id, plan_id, errors, warnings)
        if kind is not None:
            seen.add(kind)
        if kind == "stage_timing" and isinstance(event.get("stage_id"), int):
            stage_ids.add(int(event["stage_id"]))
        if kind == "placement_explanation" and isinstance(event.get("decision"), str):
            placement_decisions.add(str(event["decision"]))

    missing = sorted(REQUIRED_EVENT_KINDS - seen)
    if missing:
        errors.append("missing required observability events: " + ", ".join(missing))
    if not {"selected", "excluded", "demoted"}.issubset(placement_decisions):
        errors.append("placement explanations must include selected, excluded, and demoted")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "request_id": request_id,
            "plan_id": plan_id,
            "event_count": len(events),
            "required_events_seen": sorted(seen & REQUIRED_EVENT_KINDS),
            "stage_count": len(stage_ids),
            "placement_decisions": sorted(placement_decisions),
        },
    }


def validate_observability_contract(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing observability fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture failures.
        return {
            "ok": False,
            "errors": [f"invalid observability fixture JSON: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["observability fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_observability_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
