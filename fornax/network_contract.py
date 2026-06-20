from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

REQUIRED_EVENTS = {
    "enqueue",
    "dequeue",
    "backpressure",
    "timeout",
    "cancel",
    "plan_integrity_reject",
}


def _positive_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return None
    return value


def _positive_number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


def validate_network_contract_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    mode = data.get("mode")
    if mode != "simulated":
        errors.append("network contract mode must be simulated")
    max_queue_depth = _positive_int(
        data.get("max_queue_depth"), "max_queue_depth", errors
    )
    timeout_ms = _positive_number(data.get("timeout_ms"), "timeout_ms", errors)
    if not data.get("plan_id"):
        errors.append("plan_id must be non-empty")
    if not data.get("node_id"):
        errors.append("node_id must be non-empty")
    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []

    seen: set[str] = set()
    queue_depth = 0
    max_seen_depth = 0
    saw_over_capacity_enqueue = False
    queued_requests: set[str] = set()
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {idx} must be an object")
            continue
        kind = event.get("kind")
        if not isinstance(kind, str):
            errors.append(f"event {idx}.kind must be a string")
            continue
        seen.add(kind)
        event_plan_id = event.get("plan_id", data.get("plan_id"))
        if kind != "plan_integrity_reject" and event_plan_id != data.get("plan_id"):
            errors.append(f"event {idx} has wrong plan_id for non-reject event")
        request_id = event.get("request_id")
        if kind in {"enqueue", "dequeue", "timeout", "cancel"}:
            if not isinstance(request_id, str) or not request_id:
                errors.append(f"event {idx} {kind} must include request_id")
                continue
        if kind == "enqueue":
            if request_id in queued_requests:
                errors.append(f"event {idx} duplicate enqueue for request_id {request_id}")
            queued_requests.add(request_id)
            queue_depth += 1
            max_seen_depth = max(max_seen_depth, queue_depth)
            if max_queue_depth is not None and queue_depth > max_queue_depth:
                saw_over_capacity_enqueue = True
        elif kind == "dequeue":
            if request_id not in queued_requests:
                errors.append(f"event {idx} dequeues unknown request_id {request_id}")
            else:
                queued_requests.remove(request_id)
            queue_depth -= 1
            if queue_depth < 0:
                errors.append(f"event {idx} dequeues from an empty queue")
                queue_depth = 0
        elif kind == "backpressure":
            depth = event.get("queue_depth")
            if max_queue_depth is not None and depth != max_queue_depth:
                errors.append("backpressure event must report queue_depth at max_queue_depth")
        elif kind == "timeout":
            elapsed = event.get("elapsed_ms")
            if timeout_ms is not None and (
                not isinstance(elapsed, (int, float)) or elapsed < timeout_ms
            ):
                errors.append("timeout event elapsed_ms must be >= timeout_ms")
        elif kind == "cancel":
            if request_id in queued_requests:
                queued_requests.remove(request_id)
                queue_depth = max(0, queue_depth - 1)
            else:
                warnings.append(f"cancel event for non-queued request_id: {request_id}")
        elif kind == "plan_integrity_reject":
            if event.get("plan_id") == data.get("plan_id"):
                errors.append("plan_integrity_reject must carry a mismatched plan_id")
        else:
            warnings.append(f"unknown event kind: {kind}")

    missing = sorted(REQUIRED_EVENTS - seen)
    if missing:
        errors.append("missing required simulated events: " + ", ".join(missing))
    if saw_over_capacity_enqueue:
        errors.append("enqueue exceeded max_queue_depth before backpressure")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "mode": mode,
            "event_count": len(events),
            "max_seen_queue_depth": max_seen_depth,
            "required_events_seen": sorted(seen & REQUIRED_EVENTS),
        },
    }


def validate_network_contract(path: str | Path, mode: str = "simulated") -> dict[str, Any]:
    if mode != "simulated":
        return {
            "ok": False,
            "errors": ["only simulated network-contract mode is implemented in Phase 0"],
            "warnings": [],
            "summary": {"mode": mode},
            "fixture": str(path),
        }
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "simulated.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing network-contract fixture: {fixture_path}"],
            "warnings": [],
            "summary": {"mode": mode},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid network-contract fixture JSON: {exc}"],
            "warnings": [],
            "summary": {"mode": mode},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["network-contract fixture must be a JSON object"],
            "warnings": [],
            "summary": {"mode": mode},
            "fixture": str(fixture_path),
        }
    result = validate_network_contract_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
