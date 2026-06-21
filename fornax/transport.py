from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json
from .inventory.simulated_cluster import SIMULATED_CLUSTER_WARNING
from .runtime_format import validate_runtime_format_manifest


ALLOWED_ENDPOINT_ROLES = {"stage_worker", "expert_worker"}
ALLOWED_PAYLOAD_KINDS = {"activation", "kv_page", "expert_batch", "expert_result"}
RUNTIME_PAYLOAD_KINDS = {"activation", "kv_page", "expert_batch"}
REQUIRED_EVENT_KINDS = (
    "endpoint_registered",
    "channel_open",
    "payload_enqueue",
    "payload_send_start",
    "payload_send_end",
    "payload_receive",
    "payload_ack",
    "backpressure",
    "timeout",
    "cancel",
    "plan_integrity_reject",
    "cleanup",
)
KNOWN_EVENT_KINDS = set(REQUIRED_EVENT_KINDS)
PLAN_HASH_REQUIRED_EVENTS = {
    "channel_open",
    "payload_enqueue",
    "payload_send_start",
    "payload_send_end",
    "payload_receive",
    "payload_ack",
    "backpressure",
    "timeout",
    "cancel",
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


def _positive_number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


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


def simulated_transport_contract(
    *,
    plan_id: str = "transport-contract-plan",
    request_id: str = "req-transport-contract",
    plan_hash: str = "sha256:transport-contract-plan",
    max_queue_depth: int = 2,
    timeout_ms: float = 50.0,
) -> dict[str, Any]:
    """Build a deterministic T1 transport artifact for two logical GPU hosts."""

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    if not request_id:
        raise ValueError("request_id must be non-empty")
    if not plan_hash:
        raise ValueError("plan_hash must be non-empty")
    if isinstance(max_queue_depth, bool) or max_queue_depth <= 0:
        raise ValueError("max_queue_depth must be positive")
    if isinstance(timeout_ms, bool) or timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")

    endpoints = [
        {
            "endpoint_id": "stage-0",
            "worker_id": "stage-0",
            "role": "stage_worker",
            "node_id": "sim-gpu0",
            "logical_host_id": "sim-host-0",
            "physical_device": "cuda:0",
            "worker_environment": {"CUDA_VISIBLE_DEVICES": "0"},
            "supported_payloads": ["activation", "kv_page", "expert_batch", "expert_result"],
        },
        {
            "endpoint_id": "stage-1",
            "worker_id": "stage-1",
            "role": "stage_worker",
            "node_id": "sim-gpu1",
            "logical_host_id": "sim-host-1",
            "physical_device": "cuda:1",
            "worker_environment": {"CUDA_VISIBLE_DEVICES": "1"},
            "supported_payloads": ["activation", "kv_page"],
        },
        {
            "endpoint_id": "expert-0",
            "worker_id": "expert-0",
            "role": "expert_worker",
            "node_id": "sim-gpu1",
            "logical_host_id": "sim-host-1",
            "physical_device": "cuda:1",
            "worker_environment": {"CUDA_VISIBLE_DEVICES": "1"},
            "supported_payloads": ["expert_batch", "expert_result"],
        },
    ]
    events = [
        {
            "kind": "endpoint_registered",
            "timestamp_s": 0.000,
            "plan_id": plan_id,
            "request_id": request_id,
            "endpoint_id": "stage-0",
            "node_id": "sim-gpu0",
            "logical_host_id": "sim-host-0",
        },
        {
            "kind": "endpoint_registered",
            "timestamp_s": 0.000,
            "plan_id": plan_id,
            "request_id": request_id,
            "endpoint_id": "stage-1",
            "node_id": "sim-gpu1",
            "logical_host_id": "sim-host-1",
        },
        {
            "kind": "endpoint_registered",
            "timestamp_s": 0.000,
            "plan_id": plan_id,
            "request_id": request_id,
            "endpoint_id": "expert-0",
            "node_id": "sim-gpu1",
            "logical_host_id": "sim-host-1",
        },
        {
            "kind": "channel_open",
            "timestamp_s": 0.001,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "source_endpoint_id": "stage-0",
            "destination_endpoint_id": "stage-1",
            "allowed_payloads": ["activation", "kv_page"],
        },
        {
            "kind": "channel_open",
            "timestamp_s": 0.001,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-expert0",
            "source_endpoint_id": "stage-0",
            "destination_endpoint_id": "expert-0",
            "allowed_payloads": ["expert_batch"],
        },
        {
            "kind": "channel_open",
            "timestamp_s": 0.001,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "expert0-stage0",
            "source_endpoint_id": "expert-0",
            "destination_endpoint_id": "stage-0",
            "allowed_payloads": ["expert_result"],
        },
        {
            "kind": "payload_enqueue",
            "timestamp_s": 0.002,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "activation-0",
            "payload_kind": "activation",
            "runtime_payload_ref": "activation",
            "source_endpoint_id": "stage-0",
            "destination_endpoint_id": "stage-1",
            "queue_depth": 1,
            "bytes": 16,
        },
        {
            "kind": "payload_send_start",
            "timestamp_s": 0.003,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "activation-0",
        },
        {
            "kind": "payload_send_end",
            "timestamp_s": 0.0034,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "activation-0",
            "bytes": 16,
            "elapsed_ms": 0.4,
        },
        {
            "kind": "payload_receive",
            "timestamp_s": 0.004,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "activation-0",
            "queue_depth": 1,
        },
        {
            "kind": "payload_ack",
            "timestamp_s": 0.0045,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "activation-0",
        },
        {
            "kind": "payload_enqueue",
            "timestamp_s": 0.005,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-expert0",
            "payload_id": "expert-batch-0",
            "payload_kind": "expert_batch",
            "runtime_payload_ref": "expert_batch",
            "source_endpoint_id": "stage-0",
            "destination_endpoint_id": "expert-0",
            "queue_depth": 1,
            "bytes": 64,
        },
        {
            "kind": "payload_send_start",
            "timestamp_s": 0.006,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-expert0",
            "payload_id": "expert-batch-0",
        },
        {
            "kind": "payload_send_end",
            "timestamp_s": 0.0066,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-expert0",
            "payload_id": "expert-batch-0",
            "bytes": 64,
            "elapsed_ms": 0.6,
        },
        {
            "kind": "payload_receive",
            "timestamp_s": 0.007,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-expert0",
            "payload_id": "expert-batch-0",
            "queue_depth": 1,
        },
        {
            "kind": "payload_ack",
            "timestamp_s": 0.0075,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-expert0",
            "payload_id": "expert-batch-0",
        },
        {
            "kind": "payload_enqueue",
            "timestamp_s": 0.008,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "kv-page-timeout",
            "payload_kind": "kv_page",
            "runtime_payload_ref": "kv_page",
            "source_endpoint_id": "stage-0",
            "destination_endpoint_id": "stage-1",
            "queue_depth": 1,
            "bytes": 128,
        },
        {
            "kind": "payload_send_start",
            "timestamp_s": 0.009,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "kv-page-timeout",
        },
        {
            "kind": "timeout",
            "timestamp_s": 0.060,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "stage0-stage1",
            "payload_id": "kv-page-timeout",
            "elapsed_ms": float(timeout_ms) + 5.0,
            "reason": "simulated destination stall",
        },
        {
            "kind": "payload_enqueue",
            "timestamp_s": 0.061,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "expert0-stage0",
            "payload_id": "expert-result-cancel",
            "payload_kind": "expert_result",
            "source_endpoint_id": "expert-0",
            "destination_endpoint_id": "stage-0",
            "queue_depth": 1,
            "bytes": 32,
        },
        {
            "kind": "cancel",
            "timestamp_s": 0.062,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "channel_id": "expert0-stage0",
            "payload_id": "expert-result-cancel",
            "released": True,
            "reason": "simulated scheduler cancellation",
        },
        {
            "kind": "backpressure",
            "timestamp_s": 0.063,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "endpoint_id": "stage-1",
            "channel_id": "stage0-stage1",
            "queue_depth": max_queue_depth,
            "reason": "destination queue at max_queue_depth",
        },
        {
            "kind": "plan_integrity_reject",
            "timestamp_s": 0.064,
            "plan_id": plan_id,
            "request_id": request_id,
            "endpoint_id": "stage-1",
            "channel_id": "stage0-stage1",
            "expected_plan_hash": plan_hash,
            "rejected_plan_hash": "sha256:stale-transport-plan",
            "reason": "plan hash mismatch",
        },
        {
            "kind": "cleanup",
            "timestamp_s": 0.065,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "endpoint_id": "stage-0",
            "released_payload_ids": [
                "activation-0",
                "expert-batch-0",
                "expert-result-cancel",
            ],
        },
        {
            "kind": "cleanup",
            "timestamp_s": 0.065,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "endpoint_id": "stage-1",
            "released_payload_ids": ["activation-0", "kv-page-timeout"],
        },
        {
            "kind": "cleanup",
            "timestamp_s": 0.065,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "endpoint_id": "expert-0",
            "released_payload_ids": ["expert-batch-0", "expert-result-cancel"],
        },
    ]
    return {
        "version": 1,
        "record_kind": "transport-simulation-contract",
        "mode": "t1-simulation",
        "simulation": {
            "method": "two_gpu_logical_hosts",
            "mode": "logical_multi_host",
            "profile": "two-gpu-heterogeneous",
            "logical_host_count": 2,
            "physical_gpu_count": 2,
            "warning": SIMULATED_CLUSTER_WARNING,
        },
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "request_id": request_id,
        "max_queue_depth": max_queue_depth,
        "timeout_ms": float(timeout_ms),
        "transport_protocol": "simulated-logical-host",
        "endpoints": endpoints,
        "links": [
            {
                "a": "sim-gpu0",
                "b": "sim-gpu1",
                "bandwidth_bytes_s": 25.0e9,
                "latency_s": 0.00025,
                "measurement": {
                    "measured": False,
                    "simulated": True,
                    "source": "fornax.transport.simulated_transport_contract",
                },
            }
        ],
        "runtime_payload": _default_runtime_payload(),
        "events": events,
        "summary": {
            "endpoint_count": len(endpoints),
            "logical_host_count": 2,
            "channel_count": 3,
            "payload_count": 4,
            "ack_count": 2,
            "timeout_count": 1,
            "cancel_count": 1,
            "backpressure_count": 1,
            "plan_integrity_reject_count": 1,
            "cleanup_count": 3,
            "event_count": len(events),
        },
        "note": (
            "T1 simulated transport contract; validates activation/KV/expert "
            "payload lifecycle across two local GPUs treated as separate logical "
            "hosts. This is development evidence only, not real heterogeneous "
            "cluster closure evidence."
        ),
    }


def _endpoint_map(endpoints: Any, errors: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(endpoints, list) or not endpoints:
        errors.append("endpoints must be a non-empty list")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, endpoint in enumerate(endpoints):
        field = f"endpoints[{index}]"
        if not isinstance(endpoint, dict):
            errors.append(f"{field} must be an object")
            continue
        endpoint_id = _non_empty_string(
            endpoint.get("endpoint_id"), f"{field}.endpoint_id", errors
        )
        _non_empty_string(endpoint.get("worker_id"), f"{field}.worker_id", errors)
        _non_empty_string(endpoint.get("node_id"), f"{field}.node_id", errors)
        _non_empty_string(
            endpoint.get("logical_host_id"), f"{field}.logical_host_id", errors
        )
        _non_empty_string(
            endpoint.get("physical_device"), f"{field}.physical_device", errors
        )
        worker_environment = endpoint.get("worker_environment")
        if not isinstance(worker_environment, dict):
            errors.append(f"{field}.worker_environment must be an object")
        elif not worker_environment.get("CUDA_VISIBLE_DEVICES"):
            errors.append(
                f"{field}.worker_environment.CUDA_VISIBLE_DEVICES must be non-empty"
            )
        role = endpoint.get("role")
        if role not in ALLOWED_ENDPOINT_ROLES:
            errors.append(f"{field}.role must be one of {sorted(ALLOWED_ENDPOINT_ROLES)}")
        supported_payloads = _string_list(
            endpoint.get("supported_payloads"), f"{field}.supported_payloads", errors
        )
        if supported_payloads is not None:
            unknown = sorted(set(supported_payloads) - ALLOWED_PAYLOAD_KINDS)
            if unknown:
                errors.append(
                    f"{field}.supported_payloads contains unsupported values: "
                    + ", ".join(unknown)
                )
        if endpoint_id is not None:
            if endpoint_id in result:
                errors.append(f"duplicate endpoint_id: {endpoint_id}")
            result[endpoint_id] = {
                **endpoint,
                "role": role,
                "supported_payloads": supported_payloads or [],
            }
    return result


def _check_cluster_simulation(
    data: dict[str, Any], endpoints: dict[str, dict[str, Any]], errors: list[str]
) -> int:
    simulation = data.get("simulation")
    if not isinstance(simulation, dict):
        errors.append("simulation must be an object")
        return 0
    if simulation.get("method") != "two_gpu_logical_hosts":
        errors.append("simulation.method must be two_gpu_logical_hosts")
    if simulation.get("mode") != "logical_multi_host":
        errors.append("simulation.mode must be logical_multi_host")
    physical_count = _positive_int(
        simulation.get("physical_gpu_count"), "simulation.physical_gpu_count", errors
    )
    if physical_count is not None and physical_count < 2:
        errors.append("simulation.physical_gpu_count must be at least 2")
    logical_hosts = {
        str(endpoint.get("logical_host_id"))
        for endpoint in endpoints.values()
        if endpoint.get("logical_host_id")
    }
    if len(logical_hosts) < 2:
        errors.append("endpoints must span at least two logical hosts")
    if simulation.get("logical_host_count") != len(logical_hosts):
        errors.append("simulation.logical_host_count must match endpoint logical hosts")
    return len(logical_hosts)


def _event_payload_id(
    event: dict[str, Any], field: str, errors: list[str]
) -> str | None:
    return _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)


def _event_channel_id(
    event: dict[str, Any], field: str, errors: list[str]
) -> str | None:
    return _non_empty_string(event.get("channel_id"), f"{field}.channel_id", errors)


def _same_channel_endpoints(
    event: dict[str, Any],
    channel: dict[str, Any],
    field: str,
    errors: list[str],
) -> None:
    if event.get("source_endpoint_id") != channel["source_endpoint_id"]:
        errors.append(f"{field}.source_endpoint_id must match channel source")
    if event.get("destination_endpoint_id") != channel["destination_endpoint_id"]:
        errors.append(f"{field}.destination_endpoint_id must match channel destination")


def _validate_summary(
    data: dict[str, Any],
    *,
    endpoints: dict[str, dict[str, Any]],
    logical_host_count: int,
    channels: dict[str, dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    event_count: int,
    ack_count: int,
    timeout_count: int,
    cancel_count: int,
    backpressure_count: int,
    plan_integrity_reject_count: int,
    cleanup_count: int,
    errors: list[str],
) -> None:
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    expected = {
        "endpoint_count": len(endpoints),
        "logical_host_count": logical_host_count,
        "channel_count": len(channels),
        "payload_count": len(payloads),
        "ack_count": ack_count,
        "timeout_count": timeout_count,
        "cancel_count": cancel_count,
        "backpressure_count": backpressure_count,
        "plan_integrity_reject_count": plan_integrity_reject_count,
        "cleanup_count": cleanup_count,
        "event_count": event_count,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            errors.append(f"summary.{field} does not match events")


def validate_transport_contract_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != "transport-simulation-contract":
        errors.append("record_kind must be transport-simulation-contract")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")
    if data.get("transport_protocol") != "simulated-logical-host":
        errors.append("transport_protocol must be simulated-logical-host")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors) or ""
    request_id = _non_empty_string(data.get("request_id"), "request_id", errors) or ""
    plan_hash = _non_empty_string(data.get("plan_hash"), "plan_hash", errors) or ""
    max_queue_depth = (
        _positive_int(data.get("max_queue_depth"), "max_queue_depth", errors) or 0
    )
    timeout_ms = _positive_number(data.get("timeout_ms"), "timeout_ms", errors) or 0.0
    endpoints = _endpoint_map(data.get("endpoints"), errors)
    logical_host_count = _check_cluster_simulation(data, endpoints, errors)

    runtime_payload = data.get("runtime_payload")
    runtime_sections: set[str] = set()
    if not isinstance(runtime_payload, dict):
        errors.append("runtime_payload must be an object")
    else:
        runtime_result = validate_runtime_format_manifest(runtime_payload)
        errors.extend(f"runtime_payload: {error}" for error in runtime_result["errors"])
        warnings.extend(
            f"runtime_payload: {warning}" for warning in runtime_result["warnings"]
        )
        runtime_sections = {
            key for key, value in runtime_payload.items() if isinstance(value, dict)
        }

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []

    seen_kinds: set[str] = set()
    registered: set[str] = set()
    channels: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    cleanup_endpoints: set[str] = set()
    ack_count = 0
    timeout_count = 0
    cancel_count = 0
    backpressure_count = 0
    plan_integrity_reject_count = 0
    cleanup_count = 0
    max_seen_queue_depth = 0

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
        if kind in PLAN_HASH_REQUIRED_EVENTS and event.get("plan_hash") != plan_hash:
            errors.append(f"{field}.plan_hash must match root plan_hash")

        queue_depth = event.get("queue_depth")
        if queue_depth is not None:
            depth = _non_negative_int(queue_depth, f"{field}.queue_depth", errors)
            if depth is not None:
                max_seen_queue_depth = max(max_seen_queue_depth, depth)
                if depth > max_queue_depth:
                    errors.append(f"{field}.queue_depth exceeds max_queue_depth")

        if kind == "endpoint_registered":
            endpoint_id = _non_empty_string(
                event.get("endpoint_id"), f"{field}.endpoint_id", errors
            )
            if endpoint_id is None:
                continue
            endpoint = endpoints.get(endpoint_id)
            if endpoint is None:
                errors.append(f"{field}.endpoint_id references unknown endpoint")
                continue
            if event.get("node_id") != endpoint.get("node_id"):
                errors.append(f"{field}.node_id must match endpoint node_id")
            if event.get("logical_host_id") != endpoint.get("logical_host_id"):
                errors.append(f"{field}.logical_host_id must match endpoint")
            registered.add(endpoint_id)
        elif kind == "channel_open":
            channel_id = _event_channel_id(event, field, errors)
            source = _non_empty_string(
                event.get("source_endpoint_id"), f"{field}.source_endpoint_id", errors
            )
            destination = _non_empty_string(
                event.get("destination_endpoint_id"),
                f"{field}.destination_endpoint_id",
                errors,
            )
            allowed_payloads = _string_list(
                event.get("allowed_payloads"), f"{field}.allowed_payloads", errors
            )
            if source is not None and source not in registered:
                errors.append(f"{field}.source_endpoint_id must be registered first")
            if destination is not None and destination not in registered:
                errors.append(
                    f"{field}.destination_endpoint_id must be registered first"
                )
            if source is not None and destination is not None and source == destination:
                errors.append(f"{field} source and destination must differ")
            if allowed_payloads is not None:
                unknown = sorted(set(allowed_payloads) - ALLOWED_PAYLOAD_KINDS)
                if unknown:
                    errors.append(
                        f"{field}.allowed_payloads contains unsupported values: "
                        + ", ".join(unknown)
                    )
            if channel_id is not None:
                if channel_id in channels:
                    errors.append(f"duplicate channel_id: {channel_id}")
                channels[channel_id] = {
                    "source_endpoint_id": source,
                    "destination_endpoint_id": destination,
                    "allowed_payloads": allowed_payloads or [],
                }
        elif kind == "payload_enqueue":
            payload_id = _event_payload_id(event, field, errors)
            channel_id = _event_channel_id(event, field, errors)
            payload_kind = event.get("payload_kind")
            if payload_kind not in ALLOWED_PAYLOAD_KINDS:
                errors.append(f"{field}.payload_kind is unsupported")
            channel = channels.get(channel_id or "")
            if channel is None:
                errors.append(f"{field}.channel_id references unknown channel")
            else:
                _same_channel_endpoints(event, channel, field, errors)
                if payload_kind not in channel["allowed_payloads"]:
                    errors.append(f"{field}.payload_kind is not allowed on channel")
                source = endpoints.get(str(event.get("source_endpoint_id")))
                destination = endpoints.get(str(event.get("destination_endpoint_id")))
                if source is not None and payload_kind not in source["supported_payloads"]:
                    errors.append(f"{field}.payload_kind is not supported by source")
                if (
                    destination is not None
                    and payload_kind not in destination["supported_payloads"]
                ):
                    errors.append(
                        f"{field}.payload_kind is not supported by destination"
                    )
            if payload_kind in RUNTIME_PAYLOAD_KINDS:
                payload_ref = event.get("runtime_payload_ref")
                if payload_ref != payload_kind:
                    errors.append(f"{field}.runtime_payload_ref must match payload_kind")
                if payload_ref not in runtime_sections:
                    errors.append(f"{field}.runtime_payload_ref missing from runtime_payload")
            _positive_int(event.get("bytes"), f"{field}.bytes", errors)
            if payload_id is not None:
                if payload_id in payloads:
                    errors.append(f"duplicate payload_id enqueue: {payload_id}")
                payloads[payload_id] = {
                    "kind": payload_kind,
                    "channel_id": channel_id,
                    "started": False,
                    "ended": False,
                    "received": False,
                    "terminal": None,
                }
        elif kind in {"payload_send_start", "payload_send_end", "payload_receive", "payload_ack", "timeout", "cancel"}:
            payload_id = _event_payload_id(event, field, errors)
            channel_id = _event_channel_id(event, field, errors)
            payload = payloads.get(payload_id or "")
            if payload is None:
                errors.append(f"{field}.payload_id references unknown payload")
                continue
            if channel_id != payload.get("channel_id"):
                errors.append(f"{field}.channel_id must match payload channel")
            if payload["terminal"] is not None:
                errors.append(f"{field} occurs after terminal payload event")
                continue
            if kind == "payload_send_start":
                if payload["started"]:
                    errors.append(f"{field} duplicates payload_send_start")
                payload["started"] = True
            elif kind == "payload_send_end":
                if not payload["started"]:
                    errors.append(f"{field} occurs before payload_send_start")
                if payload["ended"]:
                    errors.append(f"{field} duplicates payload_send_end")
                _positive_int(event.get("bytes"), f"{field}.bytes", errors)
                _non_negative_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)
                payload["ended"] = True
            elif kind == "payload_receive":
                if not payload["ended"]:
                    errors.append(f"{field} occurs before payload_send_end")
                if payload["received"]:
                    errors.append(f"{field} duplicates payload_receive")
                payload["received"] = True
            elif kind == "payload_ack":
                if not payload["received"]:
                    errors.append(f"{field} occurs before payload_receive")
                payload["terminal"] = "ack"
                ack_count += 1
            elif kind == "timeout":
                if not payload["started"]:
                    errors.append(f"{field} occurs before payload_send_start")
                elapsed = _non_negative_number(
                    event.get("elapsed_ms"), f"{field}.elapsed_ms", errors
                )
                if elapsed is not None and elapsed < timeout_ms:
                    errors.append(f"{field}.elapsed_ms must be >= timeout_ms")
                _non_empty_string(event.get("reason"), f"{field}.reason", errors)
                payload["terminal"] = "timeout"
                timeout_count += 1
            elif kind == "cancel":
                if event.get("released") is not True:
                    errors.append(f"{field}.released must be true")
                _non_empty_string(event.get("reason"), f"{field}.reason", errors)
                payload["terminal"] = "cancel"
                cancel_count += 1
        elif kind == "backpressure":
            endpoint_id = _non_empty_string(
                event.get("endpoint_id"), f"{field}.endpoint_id", errors
            )
            channel_id = _event_channel_id(event, field, errors)
            if endpoint_id is not None and endpoint_id not in endpoints:
                errors.append(f"{field}.endpoint_id references unknown endpoint")
            if channel_id is not None and channel_id not in channels:
                errors.append(f"{field}.channel_id references unknown channel")
            if event.get("queue_depth") != max_queue_depth:
                errors.append(f"{field}.queue_depth must equal max_queue_depth")
            _non_empty_string(event.get("reason"), f"{field}.reason", errors)
            backpressure_count += 1
        elif kind == "plan_integrity_reject":
            endpoint_id = _non_empty_string(
                event.get("endpoint_id"), f"{field}.endpoint_id", errors
            )
            channel_id = _event_channel_id(event, field, errors)
            if endpoint_id is not None and endpoint_id not in endpoints:
                errors.append(f"{field}.endpoint_id references unknown endpoint")
            if channel_id is not None and channel_id not in channels:
                errors.append(f"{field}.channel_id references unknown channel")
            if event.get("expected_plan_hash") != plan_hash:
                errors.append(f"{field}.expected_plan_hash must match root plan_hash")
            rejected_hash = _non_empty_string(
                event.get("rejected_plan_hash"), f"{field}.rejected_plan_hash", errors
            )
            if rejected_hash == plan_hash:
                errors.append(f"{field}.rejected_plan_hash must differ from root plan_hash")
            _non_empty_string(event.get("reason"), f"{field}.reason", errors)
            plan_integrity_reject_count += 1
        elif kind == "cleanup":
            endpoint_id = _non_empty_string(
                event.get("endpoint_id"), f"{field}.endpoint_id", errors
            )
            if endpoint_id is not None:
                if endpoint_id not in endpoints:
                    errors.append(f"{field}.endpoint_id references unknown endpoint")
                cleanup_endpoints.add(endpoint_id)
            released = _string_list(
                event.get("released_payload_ids"),
                f"{field}.released_payload_ids",
                errors,
            )
            if released is not None:
                for payload_id in released:
                    if payload_id not in payloads:
                        errors.append(
                            f"{field}.released_payload_ids references unknown payload"
                        )
            cleanup_count += 1

    missing_events = [kind for kind in REQUIRED_EVENT_KINDS if kind not in seen_kinds]
    if missing_events:
        errors.append("events missing required transport events: " + ", ".join(missing_events))
    for payload_id, payload in payloads.items():
        if payload["terminal"] is None:
            errors.append(f"payload {payload_id} has no terminal ack/timeout/cancel")
        if payload["terminal"] == "ack" and not payload["received"]:
            errors.append(f"payload {payload_id} acked without receive")
        if payload["started"] and not payload["ended"] and payload["terminal"] not in {
            "timeout",
            "cancel",
        }:
            errors.append(f"payload {payload_id} send_start without send_end")
    missing_cleanup = sorted(set(endpoints) - cleanup_endpoints)
    if missing_cleanup:
        errors.append("endpoints missing cleanup event: " + ", ".join(missing_cleanup))

    _validate_summary(
        data,
        endpoints=endpoints,
        logical_host_count=logical_host_count,
        channels=channels,
        payloads=payloads,
        event_count=len(events),
        ack_count=ack_count,
        timeout_count=timeout_count,
        cancel_count=cancel_count,
        backpressure_count=backpressure_count,
        plan_integrity_reject_count=plan_integrity_reject_count,
        cleanup_count=cleanup_count,
        errors=errors,
    )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "endpoint_count": len(endpoints),
            "logical_host_count": logical_host_count,
            "channel_count": len(channels),
            "payload_count": len(payloads),
            "event_count": len(events),
            "required_events_seen": sorted(seen_kinds & set(REQUIRED_EVENT_KINDS)),
            "max_seen_queue_depth": max_seen_queue_depth,
            "ack_count": ack_count,
            "timeout_count": timeout_count,
            "cancel_count": cancel_count,
            "backpressure_count": backpressure_count,
            "plan_integrity_reject_count": plan_integrity_reject_count,
            "cleanup_count": cleanup_count,
        },
    }


def validate_transport_contract(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing transport contract fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid transport contract fixture: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["transport contract fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_transport_contract_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
