from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json
from .scheduler import simulate_scheduler, validate_scheduler_contract
from .transport import (
    simulated_transport_contract,
    validate_transport_contract_fixture,
)
from .workers import simulated_worker_contract, validate_worker_contract_fixture


REQUIRED_EVENT_KINDS = (
    "engine_start",
    "plan_loaded",
    "scheduler_contract_validated",
    "worker_contract_validated",
    "transport_contract_validated",
    "scheduler_dispatch",
    "stage_worker_call",
    "activation_handoff",
    "expert_dispatch",
    "token_emitted",
    "cancel_propagated",
    "request_finished",
    "health_probe",
    "cleanup",
)
KNOWN_EVENT_KINDS = set(REQUIRED_EVENT_KINDS)
PLAN_HASH_REQUIRED_EVENTS = {
    "plan_loaded",
    "stage_worker_call",
    "activation_handoff",
    "expert_dispatch",
    "token_emitted",
    "cancel_propagated",
    "request_finished",
    "cleanup",
}
FINISH_REASONS = {"stop", "length", "cancelled", "error"}
STAGE_METHODS = {"prefill", "decode_step", "kv_evict"}


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


def _default_plan() -> dict[str, Any]:
    return {
        "feasible": True,
        "stages": [
            {"index": 0, "layers": [0], "replicas": ["sim-gpu0"], "mode": "stage"},
            {"index": 1, "layers": [1], "replicas": ["sim-gpu1"], "mode": "stage"},
        ],
        "predicted": {
            "throughput_tok_s": 20.0,
            "per_request_latency_s": 0.25,
            "bubble_fraction": 0.12,
            "stage_effective_times_s": [0.010, 0.014],
        },
    }


def _scheduler_requests(request_id: str) -> list[dict[str, Any]]:
    return [
        {"id": request_id, "prompt_len": 8, "gen_len": 3},
        {
            "id": f"{request_id}-cancel",
            "prompt_len": 8,
            "gen_len": 2,
            "cancel": True,
        },
    ]


def simulated_engine_contract(
    *,
    plan_id: str = "engine-simulated-plan",
    request_id: str = "req-engine-simulated",
    plan_hash: str = "sha256:engine-simulated-plan",
    max_queue_depth: int = 2,
    max_inflight: int = 2,
    microbatch_size: int = 2,
    timeout_ms: float = 50.0,
) -> dict[str, Any]:
    """Build a deterministic T1 FornaxEngine orchestration contract."""

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    if not request_id:
        raise ValueError("request_id must be non-empty")
    if not plan_hash:
        raise ValueError("plan_hash must be non-empty")
    if isinstance(max_queue_depth, bool) or max_queue_depth <= 0:
        raise ValueError("max_queue_depth must be positive")
    if isinstance(max_inflight, bool) or max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    if isinstance(microbatch_size, bool) or microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    if isinstance(timeout_ms, bool) or timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")

    plan = _default_plan()
    scheduler = simulate_scheduler(
        plan,
        _scheduler_requests(request_id),
        plan_id=plan_id,
        max_queue_depth=max_queue_depth,
        max_inflight=max_inflight,
        microbatch_size=microbatch_size,
    )
    worker = simulated_worker_contract(
        plan_id=plan_id,
        request_id=request_id,
        plan_hash=plan_hash,
        max_queue_depth=max_queue_depth,
    )
    transport = simulated_transport_contract(
        plan_id=plan_id,
        request_id=request_id,
        plan_hash=plan_hash,
        max_queue_depth=max_queue_depth,
        timeout_ms=timeout_ms,
    )
    cancel_request_id = f"{request_id}-cancel"
    events = [
        {
            "kind": "engine_start",
            "timestamp_s": 0.000,
            "plan_id": plan_id,
            "request_id": request_id,
            "engine_id": "fornax-engine-sim-0",
            "mode": "t1-simulation",
        },
        {
            "kind": "plan_loaded",
            "timestamp_s": 0.001,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "stage_count": 2,
            "worker_ids": ["stage-0", "stage-1", "expert-0"],
        },
        {
            "kind": "scheduler_contract_validated",
            "timestamp_s": 0.002,
            "plan_id": plan_id,
            "request_id": request_id,
            "ok": True,
            "summary_ref": "scheduler_contract.summary",
        },
        {
            "kind": "worker_contract_validated",
            "timestamp_s": 0.003,
            "plan_id": plan_id,
            "request_id": request_id,
            "ok": True,
            "summary_ref": "worker_contract.summary",
        },
        {
            "kind": "transport_contract_validated",
            "timestamp_s": 0.004,
            "plan_id": plan_id,
            "request_id": request_id,
            "ok": True,
            "summary_ref": "transport_contract.summary",
        },
        {
            "kind": "scheduler_dispatch",
            "timestamp_s": 0.005,
            "plan_id": plan_id,
            "request_id": request_id,
            "batch_id": "mb-0",
            "request_ids": [request_id, cancel_request_id],
            "microbatch_size": 2,
        },
        {
            "kind": "stage_worker_call",
            "timestamp_s": 0.006,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "worker_id": "stage-0",
            "method": "prefill",
            "batch_id": "mb-0",
            "input_payload_id": "activation-0",
            "output_payload_id": "activation-0",
        },
        {
            "kind": "activation_handoff",
            "timestamp_s": 0.007,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "payload_id": "activation-0",
            "source_worker_id": "stage-0",
            "target_worker_id": "stage-1",
            "channel_id": "stage0-stage1",
        },
        {
            "kind": "stage_worker_call",
            "timestamp_s": 0.008,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "worker_id": "stage-1",
            "method": "decode_step",
            "batch_id": "mb-0",
            "input_payload_id": "activation-0",
            "output_payload_id": "activation-1",
        },
        {
            "kind": "expert_dispatch",
            "timestamp_s": 0.009,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "payload_id": "expert-batch-0",
            "source_worker_id": "stage-0",
            "target_worker_id": "expert-0",
            "expert_ids": [3, 7],
        },
        {
            "kind": "token_emitted",
            "timestamp_s": 0.010,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "token_index": 0,
            "token_text": "sim",
            "from_worker_id": "stage-1",
        },
        {
            "kind": "token_emitted",
            "timestamp_s": 0.011,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "token_index": 1,
            "token_text": "ulated",
            "from_worker_id": "stage-1",
        },
        {
            "kind": "cancel_propagated",
            "timestamp_s": 0.012,
            "plan_id": plan_id,
            "request_id": cancel_request_id,
            "plan_hash": plan_hash,
            "targets": ["scheduler", "workers", "transport", "kv_state"],
            "cleanup": {
                "scheduler_released": True,
                "workers_released": True,
                "transport_released": True,
                "kv_released": True,
            },
        },
        {
            "kind": "request_finished",
            "timestamp_s": 0.013,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "finish_reason": "stop",
            "generated_tokens": 2,
        },
        {
            "kind": "request_finished",
            "timestamp_s": 0.014,
            "plan_id": plan_id,
            "request_id": cancel_request_id,
            "plan_hash": plan_hash,
            "finish_reason": "cancelled",
            "generated_tokens": 0,
        },
        {
            "kind": "health_probe",
            "timestamp_s": 0.015,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-0",
            "node_id": "sim-gpu0",
            "stats": {
                "queue_depth": 0,
                "mem_free_bytes": 48_000_000_000,
                "ready": True,
            },
        },
        {
            "kind": "health_probe",
            "timestamp_s": 0.015,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "stage-1",
            "node_id": "sim-gpu1",
            "stats": {
                "queue_depth": 0,
                "mem_free_bytes": 32_000_000_000,
                "ready": True,
            },
        },
        {
            "kind": "health_probe",
            "timestamp_s": 0.015,
            "plan_id": plan_id,
            "request_id": request_id,
            "worker_id": "expert-0",
            "node_id": "sim-gpu1",
            "stats": {
                "queue_depth": 0,
                "mem_free_bytes": 32_000_000_000,
                "ready": True,
            },
        },
        {
            "kind": "cleanup",
            "timestamp_s": 0.016,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "released_components": ["scheduler", "workers", "transport", "kv_state"],
        },
    ]
    token_count = sum(1 for event in events if event["kind"] == "token_emitted")
    return {
        "version": 1,
        "record_kind": "fornax-engine-simulation-contract",
        "mode": "t1-simulation",
        "plan_id": plan_id,
        "request_id": request_id,
        "plan_hash": plan_hash,
        "plan": plan,
        "scheduler_contract": scheduler,
        "worker_contract": worker,
        "transport_contract": transport,
        "events": events,
        "summary": {
            "event_count": len(events),
            "request_count": 2,
            "finished_count": 2,
            "token_count": token_count,
            "cancel_count": 1,
            "health_probe_count": 3,
            "cleanup_count": 1,
            "embedded_contract_count": 3,
            "logical_host_count": 2,
        },
        "note": (
            "T1 simulated FornaxEngine lifecycle; composes scheduler, worker, "
            "and transport contracts without real model execution or distributed "
            "runtime processes."
        ),
    }


def _worker_ids(worker_contract: dict[str, Any]) -> set[str]:
    workers = worker_contract.get("workers")
    if not isinstance(workers, list):
        return set()
    return {
        str(worker.get("worker_id"))
        for worker in workers
        if isinstance(worker, dict) and worker.get("worker_id")
    }


def _transport_worker_ids(transport_contract: dict[str, Any]) -> set[str]:
    endpoints = transport_contract.get("endpoints")
    if not isinstance(endpoints, list):
        return set()
    return {
        str(endpoint.get("worker_id"))
        for endpoint in endpoints
        if isinstance(endpoint, dict) and endpoint.get("worker_id")
    }


def _transport_payload_ids(transport_contract: dict[str, Any]) -> set[str]:
    events = transport_contract.get("events")
    if not isinstance(events, list):
        return set()
    return {
        str(event.get("payload_id"))
        for event in events
        if isinstance(event, dict) and event.get("kind") == "payload_enqueue"
    }


def _validate_embedded_contracts(
    data: dict[str, Any],
    plan_id: str,
    request_id: str,
    plan_hash: str,
    errors: list[str],
    warnings: list[str],
) -> tuple[set[str], set[str], int]:
    embedded_ok = 0
    scheduler = data.get("scheduler_contract")
    worker = data.get("worker_contract")
    transport = data.get("transport_contract")
    if not isinstance(scheduler, dict):
        errors.append("scheduler_contract must be an object")
    else:
        result = validate_scheduler_contract(scheduler)
        if not result["ok"]:
            errors.extend(f"scheduler_contract: {error}" for error in result["errors"])
        else:
            embedded_ok += 1
        warnings.extend(f"scheduler_contract: {warning}" for warning in result["warnings"])
        if scheduler.get("plan_id") != plan_id:
            errors.append("scheduler_contract.plan_id must match root plan_id")
    if not isinstance(worker, dict):
        errors.append("worker_contract must be an object")
        worker = {}
    else:
        result = validate_worker_contract_fixture(worker)
        if not result["ok"]:
            errors.extend(f"worker_contract: {error}" for error in result["errors"])
        else:
            embedded_ok += 1
        warnings.extend(f"worker_contract: {warning}" for warning in result["warnings"])
        if worker.get("plan_id") != plan_id:
            errors.append("worker_contract.plan_id must match root plan_id")
        if worker.get("request_id") != request_id:
            errors.append("worker_contract.request_id must match root request_id")
        if worker.get("plan_hash") != plan_hash:
            errors.append("worker_contract.plan_hash must match root plan_hash")
    if not isinstance(transport, dict):
        errors.append("transport_contract must be an object")
        transport = {}
    else:
        result = validate_transport_contract_fixture(transport)
        if not result["ok"]:
            errors.extend(f"transport_contract: {error}" for error in result["errors"])
        else:
            embedded_ok += 1
        warnings.extend(
            f"transport_contract: {warning}" for warning in result["warnings"]
        )
        if transport.get("plan_id") != plan_id:
            errors.append("transport_contract.plan_id must match root plan_id")
        if transport.get("request_id") != request_id:
            errors.append("transport_contract.request_id must match root request_id")
        if transport.get("plan_hash") != plan_hash:
            errors.append("transport_contract.plan_hash must match root plan_hash")
    return (
        _worker_ids(worker) | _transport_worker_ids(transport),
        _transport_payload_ids(transport),
        embedded_ok,
    )


def validate_engine_simulation_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != "fornax-engine-simulation-contract":
        errors.append("record_kind must be fornax-engine-simulation-contract")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors) or ""
    request_id = _non_empty_string(data.get("request_id"), "request_id", errors) or ""
    plan_hash = _non_empty_string(data.get("plan_hash"), "plan_hash", errors) or ""
    worker_ids, payload_ids, embedded_ok = _validate_embedded_contracts(
        data,
        plan_id,
        request_id,
        plan_hash,
        errors,
        warnings,
    )

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []

    seen: set[str] = set()
    request_ids: set[str] = {request_id}
    finished: set[str] = set()
    token_count = 0
    cancel_count = 0
    health_probe_count = 0
    cleanup_count = 0
    validation_events = 0

    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        kind = event.get("kind")
        if not isinstance(kind, str) or not kind:
            errors.append(f"{field}.kind must be a non-empty string")
            continue
        seen.add(kind)
        if kind not in KNOWN_EVENT_KINDS:
            errors.append(f"{field}.kind is unknown: {kind}")
            continue
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        if event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match root plan_id")
        if kind in PLAN_HASH_REQUIRED_EVENTS and event.get("plan_hash") != plan_hash:
            errors.append(f"{field}.plan_hash must match root plan_hash")
        event_request_id = event.get("request_id")
        if isinstance(event_request_id, str) and event_request_id:
            request_ids.add(event_request_id)
        else:
            errors.append(f"{field}.request_id must be a non-empty string")

        if kind == "engine_start":
            if event.get("mode") != "t1-simulation":
                errors.append(f"{field}.mode must be t1-simulation")
            _non_empty_string(event.get("engine_id"), f"{field}.engine_id", errors)
        elif kind == "plan_loaded":
            _positive_int(event.get("stage_count"), f"{field}.stage_count", errors)
            workers = _string_list(event.get("worker_ids"), f"{field}.worker_ids", errors)
            if workers is not None and not set(workers).issubset(worker_ids):
                errors.append(f"{field}.worker_ids must be known worker IDs")
        elif kind in {
            "scheduler_contract_validated",
            "worker_contract_validated",
            "transport_contract_validated",
        }:
            if event.get("ok") is not True:
                errors.append(f"{field}.ok must be true")
            _non_empty_string(event.get("summary_ref"), f"{field}.summary_ref", errors)
            validation_events += 1
        elif kind == "scheduler_dispatch":
            _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            ids = _string_list(event.get("request_ids"), f"{field}.request_ids", errors)
            if ids is not None:
                request_ids.update(ids)
            _positive_int(
                event.get("microbatch_size"), f"{field}.microbatch_size", errors
            )
        elif kind == "stage_worker_call":
            worker_id = _non_empty_string(event.get("worker_id"), f"{field}.worker_id", errors)
            if worker_id is not None and worker_id not in worker_ids:
                errors.append(f"{field}.worker_id references unknown worker")
            if event.get("method") not in STAGE_METHODS:
                errors.append(f"{field}.method is unsupported")
            _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            _non_empty_string(
                event.get("output_payload_id"), f"{field}.output_payload_id", errors
            )
        elif kind == "activation_handoff":
            payload_id = _non_empty_string(
                event.get("payload_id"), f"{field}.payload_id", errors
            )
            if payload_id is not None and payload_id not in payload_ids:
                errors.append(f"{field}.payload_id references unknown transport payload")
            for side in ("source_worker_id", "target_worker_id"):
                worker_id = _non_empty_string(event.get(side), f"{field}.{side}", errors)
                if worker_id is not None and worker_id not in worker_ids:
                    errors.append(f"{field}.{side} references unknown worker")
            _non_empty_string(event.get("channel_id"), f"{field}.channel_id", errors)
        elif kind == "expert_dispatch":
            payload_id = _non_empty_string(
                event.get("payload_id"), f"{field}.payload_id", errors
            )
            if payload_id is not None and payload_id not in payload_ids:
                errors.append(f"{field}.payload_id references unknown transport payload")
            target = _non_empty_string(
                event.get("target_worker_id"), f"{field}.target_worker_id", errors
            )
            if target is not None and target not in worker_ids:
                errors.append(f"{field}.target_worker_id references unknown worker")
            expert_ids = event.get("expert_ids")
            if not isinstance(expert_ids, list) or not expert_ids:
                errors.append(f"{field}.expert_ids must be a non-empty list")
        elif kind == "token_emitted":
            _non_negative_int(event.get("token_index"), f"{field}.token_index", errors)
            _non_empty_string(event.get("token_text"), f"{field}.token_text", errors)
            worker_id = _non_empty_string(
                event.get("from_worker_id"), f"{field}.from_worker_id", errors
            )
            if worker_id is not None and worker_id not in worker_ids:
                errors.append(f"{field}.from_worker_id references unknown worker")
            if event_request_id == request_id:
                token_count += 1
        elif kind == "cancel_propagated":
            targets = _string_list(event.get("targets"), f"{field}.targets", errors)
            required_targets = {"scheduler", "workers", "transport", "kv_state"}
            if targets is not None and not required_targets.issubset(set(targets)):
                errors.append(
                    f"{field}.targets must include scheduler, workers, transport, kv_state"
                )
            cleanup = event.get("cleanup")
            if not isinstance(cleanup, dict):
                errors.append(f"{field}.cleanup must be an object")
            else:
                for cleanup_field in (
                    "scheduler_released",
                    "workers_released",
                    "transport_released",
                    "kv_released",
                ):
                    if cleanup.get(cleanup_field) is not True:
                        errors.append(f"{field}.cleanup.{cleanup_field} must be true")
            cancel_count += 1
        elif kind == "request_finished":
            finish_reason = event.get("finish_reason")
            if finish_reason not in FINISH_REASONS:
                errors.append(f"{field}.finish_reason is invalid")
            generated = _non_negative_int(
                event.get("generated_tokens"), f"{field}.generated_tokens", errors
            )
            if event_request_id == request_id and generated is not None:
                if generated != token_count:
                    errors.append(
                        f"{field}.generated_tokens must match emitted token count"
                    )
            if finish_reason == "cancelled" and generated not in {None, 0}:
                errors.append(f"{field}.generated_tokens must be 0 when cancelled")
            if isinstance(event_request_id, str):
                finished.add(event_request_id)
        elif kind == "health_probe":
            worker_id = _non_empty_string(event.get("worker_id"), f"{field}.worker_id", errors)
            if worker_id is not None and worker_id not in worker_ids:
                errors.append(f"{field}.worker_id references unknown worker")
            _non_empty_string(event.get("node_id"), f"{field}.node_id", errors)
            stats = event.get("stats")
            if not isinstance(stats, dict):
                errors.append(f"{field}.stats must be an object")
            elif stats.get("ready") is not True:
                errors.append(f"{field}.stats.ready must be true")
            health_probe_count += 1
        elif kind == "cleanup":
            components = _string_list(
                event.get("released_components"),
                f"{field}.released_components",
                errors,
            )
            required = {"scheduler", "workers", "transport", "kv_state"}
            if components is not None and not required.issubset(set(components)):
                errors.append(
                    f"{field}.released_components must include scheduler, workers, transport, kv_state"
                )
            cleanup_count += 1

    missing_events = [kind for kind in REQUIRED_EVENT_KINDS if kind not in seen]
    if missing_events:
        errors.append("events missing required engine events: " + ", ".join(missing_events))
    if request_id not in finished:
        errors.append("primary request missing request_finished event")
    cancelled_requests = {rid for rid in request_ids if rid.endswith("-cancel")}
    missing_cancel_finish = sorted(cancelled_requests - finished)
    if missing_cancel_finish:
        errors.append(
            "cancelled requests missing request_finished event: "
            + ", ".join(missing_cancel_finish)
        )

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    expected = {
        "event_count": len(events),
        "request_count": len(request_ids),
        "finished_count": len(finished),
        "token_count": token_count,
        "cancel_count": cancel_count,
        "health_probe_count": health_probe_count,
        "cleanup_count": cleanup_count,
        "embedded_contract_count": embedded_ok,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            errors.append(f"summary.{field} does not match events")
    if validation_events != 3:
        errors.append("engine must validate scheduler, worker, and transport contracts")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "event_count": len(events),
            "request_count": len(request_ids),
            "finished_count": len(finished),
            "token_count": token_count,
            "cancel_count": cancel_count,
            "health_probe_count": health_probe_count,
            "cleanup_count": cleanup_count,
            "embedded_contract_count": embedded_ok,
            "required_events_seen": sorted(seen & set(REQUIRED_EVENT_KINDS)),
        },
    }


def validate_engine_simulation(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing engine simulation fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid engine simulation fixture: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["engine simulation fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_engine_simulation_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
