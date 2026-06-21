from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json
from .runtime_format import validate_runtime_format_manifest


ALLOWED_ROLES = {"stage_worker", "expert_worker"}
REQUIRED_EVENT_KINDS = (
    "worker_registered",
    "plan_loaded",
    "activation_received",
    "stage_execute_start",
    "stage_execute_end",
    "kv_write",
    "activation_sent",
    "expert_batch_received",
    "expert_execute_start",
    "expert_execute_end",
    "expert_result_sent",
    "stale_plan_reject",
    "cleanup",
)
KNOWN_EVENT_KINDS = set(REQUIRED_EVENT_KINDS)
PLAN_HASH_REQUIRED_EVENTS = {
    "plan_loaded",
    "activation_received",
    "stage_execute_start",
    "stage_execute_end",
    "kv_write",
    "activation_sent",
    "expert_batch_received",
    "expert_execute_start",
    "expert_execute_end",
    "expert_result_sent",
}
STAGE_WORKER_EVENTS = {
    "activation_received",
    "stage_execute_start",
    "stage_execute_end",
    "kv_write",
    "activation_sent",
}
EXPERT_WORKER_EVENTS = {
    "expert_batch_received",
    "expert_execute_start",
    "expert_execute_end",
    "expert_result_sent",
}
SHARED_WORKER_EVENTS = {
    "worker_registered",
    "plan_loaded",
    "stale_plan_reject",
    "cleanup",
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


def _non_negative_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{field} must be a non-negative integer")
        return None
    return value


def _non_negative_number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{field} must be a non-negative number")
        return None
    return float(value)


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


def _default_runtime_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "activation": {
            "dtype": "fp16",
            "shape": [2, 4],
            "layout": "contiguous_row_major",
            "values": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
        },
        "kv_page": {
            "dtype": "fp16",
            "shape": [4, 2, 4],
            "page_size": 4,
            "token_count": 2,
            "owner_stage": 0,
        },
        "expert_batch": {
            "layer_id": 1,
            "expert_ids": [3, 7],
            "token_indices": [0, 1],
            "topk_weights": [0.75, 0.25],
            "hidden_shape": [2, 4],
            "gather_order": [0, 1],
        },
        "tolerances": {"fp16": {"rtol": 0.001, "atol": 0.001}},
    }


def simulated_worker_contract(
    *,
    plan_id: str = "worker-contract-plan",
    request_id: str = "req-worker-contract",
    plan_hash: str = "sha256:worker-contract-plan",
    max_queue_depth: int = 2,
) -> dict[str, Any]:
    """Build a deterministic T1 simulated worker-contract artifact."""

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    if not request_id:
        raise ValueError("request_id must be non-empty")
    if not plan_hash:
        raise ValueError("plan_hash must be non-empty")
    if isinstance(max_queue_depth, bool) or max_queue_depth <= 0:
        raise ValueError("max_queue_depth must be positive")

    events = [
        {
            "kind": "worker_registered",
            "timestamp_s": 0.0,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "role": "stage_worker",
            "node_id": "sim-gpu0",
            "stage_index": 0,
        },
        {
            "kind": "worker_registered",
            "timestamp_s": 0.0,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "role": "expert_worker",
            "node_id": "sim-gpu1",
        },
        {
            "kind": "plan_loaded",
            "timestamp_s": 0.001,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "plan_hash": plan_hash,
        },
        {
            "kind": "plan_loaded",
            "timestamp_s": 0.001,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "plan_hash": plan_hash,
        },
        {
            "kind": "activation_received",
            "timestamp_s": 0.002,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "payload_id": "activation-0",
            "payload_kind": "activation",
            "source": "scheduler",
            "queue_depth": 1,
        },
        {
            "kind": "stage_execute_start",
            "timestamp_s": 0.003,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "stage_index": 0,
        },
        {
            "kind": "kv_write",
            "timestamp_s": 0.004,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "page_id": "kv-0",
            "owner_stage": 0,
            "token_count": 2,
        },
        {
            "kind": "stage_execute_end",
            "timestamp_s": 0.007,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "stage_index": 0,
            "elapsed_ms": 4.0,
        },
        {
            "kind": "activation_sent",
            "timestamp_s": 0.008,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "payload_id": "expert-batch-0",
            "payload_kind": "expert_batch",
            "target_worker_id": "expert-0",
        },
        {
            "kind": "expert_batch_received",
            "timestamp_s": 0.009,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "payload_id": "expert-batch-0",
            "payload_kind": "expert_batch",
            "queue_depth": 1,
        },
        {
            "kind": "expert_execute_start",
            "timestamp_s": 0.010,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "expert_ids": [3, 7],
        },
        {
            "kind": "expert_execute_end",
            "timestamp_s": 0.013,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "expert_ids": [3, 7],
            "elapsed_ms": 3.0,
        },
        {
            "kind": "expert_result_sent",
            "timestamp_s": 0.014,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "payload_id": "expert-result-0",
            "payload_kind": "expert_result",
            "target_worker_id": "stage-0",
        },
        {
            "kind": "activation_sent",
            "timestamp_s": 0.015,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "plan_hash": plan_hash,
            "batch_id": "mb-0",
            "payload_id": "activation-1",
            "payload_kind": "activation",
            "target_worker_id": "stage-1",
        },
        {
            "kind": "stale_plan_reject",
            "timestamp_s": 0.016,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "rejected_plan_hash": "sha256:old-plan",
            "reason": "plan hash mismatch",
        },
        {
            "kind": "cleanup",
            "timestamp_s": 0.017,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "batch_id": "mb-0",
            "released": ["activation", "kv_page"],
        },
        {
            "kind": "cleanup",
            "timestamp_s": 0.017,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "batch_id": "mb-0",
            "released": ["expert_batch", "expert_result"],
        },
    ]
    return {
        "version": 1,
        "record_kind": "worker-simulation-contract",
        "mode": "t1-simulation",
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "request_id": request_id,
        "max_queue_depth": max_queue_depth,
        "workers": [
            {
                "worker_id": "stage-0",
                "role": "stage_worker",
                "node_id": "sim-gpu0",
                "stage_index": 0,
                "supported_payloads": ["activation", "kv_page", "expert_batch"],
            },
            {
                "worker_id": "expert-0",
                "role": "expert_worker",
                "node_id": "sim-gpu1",
                "supported_payloads": ["expert_batch", "expert_result"],
            },
        ],
        "runtime_payload": _default_runtime_payload(),
        "events": events,
        "summary": {
            "worker_count": 2,
            "stage_worker_count": 1,
            "expert_worker_count": 1,
            "event_count": len(events),
            "plan_integrity_reject_count": 1,
            "cleanup_count": 2,
        },
        "note": (
            "T1 simulated worker contract; validates worker identity, plan-integrity "
            "tags, activation/KV/expert payload handoff, and cleanup without real "
            "distributed runtime execution."
        ),
    }


def _worker_map(workers: Any, errors: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(workers, list) or not workers:
        errors.append("workers must be a non-empty list")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, worker in enumerate(workers):
        field = f"workers[{index}]"
        if not isinstance(worker, dict):
            errors.append(f"{field} must be an object")
            continue
        worker_id = _non_empty_string(
            worker.get("worker_id"), f"{field}.worker_id", errors
        )
        role = worker.get("role")
        if role not in ALLOWED_ROLES:
            errors.append(f"{field}.role must be one of {sorted(ALLOWED_ROLES)}")
        _non_empty_string(worker.get("node_id"), f"{field}.node_id", errors)
        if role == "stage_worker":
            _non_negative_int(worker.get("stage_index"), f"{field}.stage_index", errors)
        payloads = _string_list(
            worker.get("supported_payloads"), f"{field}.supported_payloads", errors
        )
        if worker_id is not None:
            if worker_id in result:
                errors.append(f"duplicate worker_id: {worker_id}")
            result[worker_id] = {
                **worker,
                "role": role,
                "supported_payloads": payloads or [],
            }
    return result


def _event_role_allowed(kind: str, role: str | None) -> bool:
    if kind in SHARED_WORKER_EVENTS:
        return True
    if kind in STAGE_WORKER_EVENTS:
        return role == "stage_worker"
    if kind in EXPERT_WORKER_EVENTS:
        return role == "expert_worker"
    return False


def _check_event_payload(
    event: dict[str, Any], field: str, errors: list[str]
) -> str | None:
    payload_kind = event.get("payload_kind")
    if payload_kind is not None and payload_kind not in {
        "activation",
        "kv_page",
        "expert_batch",
        "expert_result",
    }:
        errors.append(f"{field}.payload_kind is unsupported")
    if event.get("payload_id") is not None:
        _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
    return payload_kind if isinstance(payload_kind, str) else None


def validate_worker_contract_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != "worker-simulation-contract":
        errors.append("record_kind must be worker-simulation-contract")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors) or ""
    request_id = _non_empty_string(data.get("request_id"), "request_id", errors) or ""
    plan_hash = _non_empty_string(data.get("plan_hash"), "plan_hash", errors) or ""
    max_queue_depth = (
        _positive_int(data.get("max_queue_depth"), "max_queue_depth", errors) or 0
    )
    workers = _worker_map(data.get("workers"), errors)

    runtime_payload = data.get("runtime_payload")
    if not isinstance(runtime_payload, dict):
        errors.append("runtime_payload must be an object")
    else:
        runtime_result = validate_runtime_format_manifest(runtime_payload)
        errors.extend(f"runtime_payload: {error}" for error in runtime_result["errors"])
        warnings.extend(
            f"runtime_payload: {warning}" for warning in runtime_result["warnings"]
        )

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []

    seen_kinds: set[str] = set()
    registered: set[str] = set()
    loaded: set[str] = set()
    cleanup_workers: set[str] = set()
    stage_open: set[tuple[str, str]] = set()
    expert_open: set[tuple[str, str]] = set()
    max_seen_queue_depth = 0
    plan_integrity_reject_count = 0
    cleanup_count = 0

    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        kind = event.get("kind")
        if not isinstance(kind, str) or not kind:
            errors.append(f"{field}.kind must be a non-empty string")
            continue
        seen_kinds.add(kind)
        if kind not in KNOWN_EVENT_KINDS:
            errors.append(f"{field}.kind is unknown: {kind}")
            continue
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        if event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match root plan_id")
        if event.get("request_id") != request_id:
            errors.append(f"{field}.request_id must match root request_id")
        worker_id = event.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            errors.append(f"{field}.worker_id must be a non-empty string")
            continue
        worker = workers.get(worker_id)
        if worker is None:
            errors.append(f"{field}.worker_id references unknown worker")
            continue
        role = worker.get("role")
        if not _event_role_allowed(kind, role):
            errors.append(f"{field}.kind {kind} is not valid for role {role}")

        if kind != "worker_registered" and worker_id not in registered:
            errors.append(f"{field} occurs before worker_registered")
        if (
            kind in PLAN_HASH_REQUIRED_EVENTS
            and kind != "plan_loaded"
            and worker_id not in loaded
        ):
            errors.append(f"{field} occurs before plan_loaded")
        if kind in PLAN_HASH_REQUIRED_EVENTS and event.get("plan_hash") != plan_hash:
            errors.append(f"{field}.plan_hash must match root plan_hash")
        if kind == "stale_plan_reject":
            rejected_hash = _non_empty_string(
                event.get("rejected_plan_hash"), f"{field}.rejected_plan_hash", errors
            )
            if rejected_hash == plan_hash:
                errors.append(
                    f"{field}.rejected_plan_hash must differ from root plan_hash"
                )
            _non_empty_string(event.get("reason"), f"{field}.reason", errors)
            plan_integrity_reject_count += 1

        queue_depth = event.get("queue_depth")
        if queue_depth is not None:
            depth = _non_negative_int(queue_depth, f"{field}.queue_depth", errors)
            if depth is not None:
                max_seen_queue_depth = max(max_seen_queue_depth, depth)
                if depth > max_queue_depth:
                    errors.append(f"{field}.queue_depth exceeds max_queue_depth")

        if kind == "worker_registered":
            if event.get("role") != role:
                errors.append(f"{field}.role must match worker role")
            registered.add(worker_id)
        elif kind == "plan_loaded":
            loaded.add(worker_id)
        elif kind in {"activation_received", "activation_sent", "expert_batch_received"}:
            payload_kind = _check_event_payload(event, field, errors)
            if payload_kind is not None and payload_kind not in worker.get(
                "supported_payloads", []
            ):
                errors.append(f"{field}.payload_kind is not supported by worker")
            if kind == "activation_sent":
                target = event.get("target_worker_id")
                if not isinstance(target, str) or not target:
                    errors.append(f"{field}.target_worker_id must be non-empty")
        elif kind == "stage_execute_start":
            batch_id = _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            _non_negative_int(event.get("stage_index"), f"{field}.stage_index", errors)
            if batch_id is not None:
                stage_open.add((worker_id, batch_id))
        elif kind == "stage_execute_end":
            batch_id = _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            _non_negative_int(event.get("stage_index"), f"{field}.stage_index", errors)
            _non_negative_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)
            if batch_id is not None:
                key = (worker_id, batch_id)
                if key not in stage_open:
                    errors.append(f"{field} has no matching stage_execute_start")
                else:
                    stage_open.remove(key)
        elif kind == "kv_write":
            _non_empty_string(event.get("page_id"), f"{field}.page_id", errors)
            _non_negative_int(event.get("owner_stage"), f"{field}.owner_stage", errors)
            _non_negative_int(event.get("token_count"), f"{field}.token_count", errors)
        elif kind == "expert_execute_start":
            batch_id = _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            expert_ids = event.get("expert_ids")
            if not isinstance(expert_ids, list) or not expert_ids:
                errors.append(f"{field}.expert_ids must be a non-empty list")
            if batch_id is not None:
                expert_open.add((worker_id, batch_id))
        elif kind == "expert_execute_end":
            batch_id = _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            _non_negative_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)
            if batch_id is not None:
                key = (worker_id, batch_id)
                if key not in expert_open:
                    errors.append(f"{field} has no matching expert_execute_start")
                else:
                    expert_open.remove(key)
        elif kind == "expert_result_sent":
            payload_kind = _check_event_payload(event, field, errors)
            if payload_kind is not None and payload_kind not in worker.get(
                "supported_payloads", []
            ):
                errors.append(f"{field}.payload_kind is not supported by worker")
            target = event.get("target_worker_id")
            if not isinstance(target, str) or not target:
                errors.append(f"{field}.target_worker_id must be non-empty")
        elif kind == "cleanup":
            cleanup_workers.add(worker_id)
            cleanup_count += 1
            _string_list(event.get("released"), f"{field}.released", errors)

    missing = [kind for kind in REQUIRED_EVENT_KINDS if kind not in seen_kinds]
    if missing:
        errors.append("events missing required worker events: " + ", ".join(missing))
    if stage_open:
        errors.append("stage_execute_start without end: " + str(sorted(stage_open)))
    if expert_open:
        errors.append("expert_execute_start without end: " + str(sorted(expert_open)))
    missing_cleanup = sorted(set(workers) - cleanup_workers)
    if missing_cleanup:
        errors.append("workers missing cleanup event: " + ", ".join(missing_cleanup))

    stage_workers = [
        worker for worker in workers.values() if worker.get("role") == "stage_worker"
    ]
    expert_workers = [
        worker for worker in workers.values() if worker.get("role") == "expert_worker"
    ]
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count does not match events")
    if summary.get("worker_count") != len(workers):
        errors.append("summary.worker_count does not match workers")
    if summary.get("stage_worker_count") != len(stage_workers):
        errors.append("summary.stage_worker_count does not match workers")
    if summary.get("expert_worker_count") != len(expert_workers):
        errors.append("summary.expert_worker_count does not match workers")
    if summary.get("plan_integrity_reject_count") != plan_integrity_reject_count:
        errors.append("summary.plan_integrity_reject_count does not match events")
    if summary.get("cleanup_count") != cleanup_count:
        errors.append("summary.cleanup_count does not match events")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "worker_count": len(workers),
            "stage_worker_count": len(stage_workers),
            "expert_worker_count": len(expert_workers),
            "event_count": len(events),
            "required_events_seen": sorted(seen_kinds & set(REQUIRED_EVENT_KINDS)),
            "max_seen_queue_depth": max_seen_queue_depth,
            "plan_integrity_reject_count": plan_integrity_reject_count,
            "cleanup_count": cleanup_count,
        },
    }


def validate_worker_contract(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing worker contract fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid worker contract fixture: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["worker contract fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_worker_contract_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
