from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json
from .runtime_format import validate_runtime_format_manifest


REQUIRED_EVENT_KINDS = (
    "router_topk_start",
    "router_topk_end",
    "expert_bucketed",
    "local_expert_dispatch",
    "remote_expert_dispatch",
    "expert_execute_start",
    "expert_execute_end",
    "expert_result_received",
    "migration_recommendation",
    "weighted_gather_start",
    "weighted_gather_end",
    "expert_trace_recorded",
    "cleanup",
)
KNOWN_EVENT_KINDS = set(REQUIRED_EVENT_KINDS)
PLAN_HASH_REQUIRED_EVENTS = {
    "router_topk_start",
    "router_topk_end",
    "expert_bucketed",
    "local_expert_dispatch",
    "remote_expert_dispatch",
    "expert_execute_start",
    "expert_execute_end",
    "expert_result_received",
    "migration_recommendation",
    "weighted_gather_start",
    "weighted_gather_end",
    "cleanup",
}
LOCAL_ROLES = {"hot_resident", "migrated_hot"}
REMOTE_ROLES = {"warm_remote", "cold_remote"}


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


def _int_list(value: Any, field: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            errors.append(f"{field}[{index}] must be an integer")
            return None
        result.append(item)
    return result


def _number_list(value: Any, field: str, errors: list[str]) -> list[float] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{field}[{index}] must be numeric")
            return None
        result.append(float(item))
    return result


def _default_runtime_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "activation": {
            "dtype": "fp16",
            "shape": [3, 4],
            "layout": "contiguous_row_major",
            "values": [
                0.0,
                0.1,
                0.2,
                0.3,
                1.0,
                1.1,
                1.2,
                1.3,
                2.0,
                2.1,
                2.2,
                2.3,
            ],
        },
        "kv_page": {
            "dtype": "fp16",
            "shape": [4, 2, 4],
            "page_size": 4,
            "token_count": 3,
            "owner_stage": 0,
        },
        "expert_batch": {
            "layer_id": 1,
            "expert_ids": [3, 5, 7],
            "token_indices": [0, 1, 2],
            "topk_weights": [0.75, 0.60, 0.45],
            "hidden_shape": [3, 4],
            "gather_order": [0, 1, 2],
        },
        "tolerances": {"fp16": {"rtol": 0.001, "atol": 0.001}},
    }


def _routing_trace(layer_id: int) -> list[dict[str, Any]]:
    return [
        {
            "token_index": 0,
            "layer_id": layer_id,
            "expert_ids": [3, 7],
            "topk_weights": [0.75, 0.25],
        },
        {
            "token_index": 1,
            "layer_id": layer_id,
            "expert_ids": [3, 5],
            "topk_weights": [0.60, 0.40],
        },
        {
            "token_index": 2,
            "layer_id": layer_id,
            "expert_ids": [5, 7],
            "topk_weights": [0.55, 0.45],
        },
    ]


def _expert_placement(layer_id: int) -> list[dict[str, Any]]:
    return [
        {
            "layer_id": layer_id,
            "expert_id": 3,
            "node_id": "sim-gpu0",
            "worker_id": "stage-0",
            "role": "hot_resident",
            "local_to_stage": True,
        },
        {
            "layer_id": layer_id,
            "expert_id": 5,
            "node_id": "sim-gpu1",
            "worker_id": "expert-0",
            "role": "warm_remote",
            "local_to_stage": False,
        },
        {
            "layer_id": layer_id,
            "expert_id": 7,
            "node_id": "sim-gpu1",
            "worker_id": "expert-0",
            "role": "cold_remote",
            "local_to_stage": False,
        },
    ]


def _buckets(trace: list[dict[str, Any]]) -> dict[int, dict[str, list[Any]]]:
    buckets: dict[int, dict[str, list[Any]]] = {}
    for row in trace:
        token_index = int(row["token_index"])
        for expert_id, weight in zip(row["expert_ids"], row["topk_weights"]):
            bucket = buckets.setdefault(
                int(expert_id), {"token_indices": [], "topk_weights": []}
            )
            bucket["token_indices"].append(token_index)
            bucket["topk_weights"].append(float(weight))
    return buckets


def simulated_moe_contract(
    *,
    plan_id: str = "moe-simulated-plan",
    request_id: str = "req-moe-simulated",
    plan_hash: str = "sha256:moe-simulated-plan",
    layer_id: int = 1,
    top_k: int = 2,
    max_remote_wait_ms: float = 5.0,
    migration_hotness_threshold: float = 0.50,
) -> dict[str, Any]:
    """Build a deterministic T1 MoE expert-runtime simulation contract."""

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    if not request_id:
        raise ValueError("request_id must be non-empty")
    if not plan_hash:
        raise ValueError("plan_hash must be non-empty")
    if isinstance(layer_id, bool) or layer_id < 0:
        raise ValueError("layer_id must be non-negative")
    if isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be positive")
    if isinstance(max_remote_wait_ms, bool) or max_remote_wait_ms <= 0:
        raise ValueError("max_remote_wait_ms must be positive")
    if (
        isinstance(migration_hotness_threshold, bool)
        or not isinstance(migration_hotness_threshold, (int, float))
        or migration_hotness_threshold <= 0
        or migration_hotness_threshold > 1
    ):
        raise ValueError("migration_hotness_threshold must be in (0, 1]")

    trace = _routing_trace(layer_id)
    placement = _expert_placement(layer_id)
    buckets = _buckets(trace)
    events: list[dict[str, Any]] = [
        {
            "kind": "router_topk_start",
            "timestamp_s": 0.000,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "layer_id": layer_id,
            "token_count": len(trace),
            "top_k": top_k,
        },
        {
            "kind": "router_topk_end",
            "timestamp_s": 0.001,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "layer_id": layer_id,
            "routes": trace,
            "elapsed_ms": 1.0,
        },
    ]
    for expert_id in sorted(buckets):
        bucket = buckets[expert_id]
        placement_row = next(item for item in placement if item["expert_id"] == expert_id)
        token_indices = list(bucket["token_indices"])
        topk_weights = list(bucket["topk_weights"])
        events.append(
            {
                "kind": "expert_bucketed",
                "timestamp_s": 0.002 + expert_id * 0.0001,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "token_indices": token_indices,
                "topk_weights": topk_weights,
                "target_worker_id": placement_row["worker_id"],
                "payload_id": f"expert-batch-{expert_id}",
            }
        )
        dispatch_kind = (
            "local_expert_dispatch"
            if placement_row["role"] in LOCAL_ROLES
            else "remote_expert_dispatch"
        )
        dispatch: dict[str, Any] = {
            "kind": dispatch_kind,
            "timestamp_s": 0.004 + expert_id * 0.0001,
            "plan_id": plan_id,
            "request_id": request_id,
            "plan_hash": plan_hash,
            "layer_id": layer_id,
            "expert_id": expert_id,
            "payload_id": f"expert-batch-{expert_id}",
            "worker_id": placement_row["worker_id"],
            "node_id": placement_row["node_id"],
            "token_count": len(token_indices),
        }
        if dispatch_kind == "remote_expert_dispatch":
            dispatch["remote_wait_ms"] = 2.0 if expert_id == 5 else 3.0
            dispatch["max_remote_wait_ms"] = float(max_remote_wait_ms)
        events.append(dispatch)
        events.append(
            {
                "kind": "expert_execute_start",
                "timestamp_s": 0.006 + expert_id * 0.0001,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "worker_id": placement_row["worker_id"],
                "payload_id": f"expert-batch-{expert_id}",
            }
        )
        events.append(
            {
                "kind": "expert_execute_end",
                "timestamp_s": 0.009 + expert_id * 0.0001,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "worker_id": placement_row["worker_id"],
                "payload_id": f"expert-batch-{expert_id}",
                "result_payload_id": f"expert-result-{expert_id}",
                "elapsed_ms": 3.0,
            }
        )
        events.append(
            {
                "kind": "expert_result_received",
                "timestamp_s": 0.010 + expert_id * 0.0001,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "result_payload_id": f"expert-result-{expert_id}",
                "token_indices": token_indices,
            }
        )
    events.extend(
        [
            {
                "kind": "migration_recommendation",
                "timestamp_s": 0.020,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "layer_id": layer_id,
                "expert_id": 5,
                "from_node_id": "sim-gpu1",
                "to_node_id": "sim-gpu0",
                "reason": "expert hotness exceeds migration threshold",
                "hotness": 2 / 3,
                "threshold": float(migration_hotness_threshold),
                "decision": "migrate_next_window",
            },
            {
                "kind": "weighted_gather_start",
                "timestamp_s": 0.021,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "layer_id": layer_id,
                "token_indices": [0, 1, 2],
                "source_result_payloads": [
                    "expert-result-3",
                    "expert-result-5",
                    "expert-result-7",
                ],
            },
            {
                "kind": "weighted_gather_end",
                "timestamp_s": 0.024,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "layer_id": layer_id,
                "token_indices": [0, 1, 2],
                "output_payload_id": "moe-gathered-0",
                "output_checksum": "sha256:00000000000000000000000000000000000000000000000000000000000000ab",
                "elapsed_ms": 3.0,
            },
            {
                "kind": "expert_trace_recorded",
                "timestamp_s": 0.025,
                "plan_id": plan_id,
                "request_id": request_id,
                "layer_id": layer_id,
                "trace_id": "expert-trace-0",
                "hot_experts": [3, 5],
                "remote_hit_rate": 4 / 6,
                "migration_candidates": [5],
            },
            {
                "kind": "cleanup",
                "timestamp_s": 0.026,
                "plan_id": plan_id,
                "request_id": request_id,
                "plan_hash": plan_hash,
                "released_payloads": [
                    "expert-batch-3",
                    "expert-batch-5",
                    "expert-batch-7",
                    "expert-result-3",
                    "expert-result-5",
                    "expert-result-7",
                    "moe-gathered-0",
                ],
            },
        ]
    )
    return {
        "version": 1,
        "record_kind": "moe-expert-runtime-simulation-contract",
        "mode": "t1-simulation",
        "plan_id": plan_id,
        "request_id": request_id,
        "plan_hash": plan_hash,
        "layer_id": layer_id,
        "top_k": top_k,
        "max_remote_wait_ms": float(max_remote_wait_ms),
        "migration_hotness_threshold": float(migration_hotness_threshold),
        "expert_placement": placement,
        "routing_trace": trace,
        "runtime_payload": _default_runtime_payload(),
        "events": events,
        "summary": {
            "event_count": len(events),
            "token_count": len(trace),
            "expert_count": len(placement),
            "remote_dispatch_count": 2,
            "local_dispatch_count": 1,
            "migration_recommendation_count": 1,
            "weighted_gather_count": 1,
            "remote_hit_rate": 4 / 6,
        },
        "note": (
            "T1 simulated MoE expert runtime; validates router top-k, expert "
            "bucketing, local/remote dispatch, migration recommendation, weighted "
            "gather, expert traces, and cleanup without model execution."
        ),
    }


def _placement_map(value: Any, errors: list[str]) -> dict[tuple[int, int], dict[str, Any]]:
    if not isinstance(value, list) or not value:
        errors.append("expert_placement must be a non-empty list")
        return {}
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for index, item in enumerate(value):
        field = f"expert_placement[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{field} must be an object")
            continue
        layer_id = _non_negative_int(item.get("layer_id"), f"{field}.layer_id", errors)
        expert_id = _non_negative_int(item.get("expert_id"), f"{field}.expert_id", errors)
        _non_empty_string(item.get("node_id"), f"{field}.node_id", errors)
        _non_empty_string(item.get("worker_id"), f"{field}.worker_id", errors)
        role = item.get("role")
        if role not in LOCAL_ROLES | REMOTE_ROLES:
            errors.append(f"{field}.role is unsupported")
        if not isinstance(item.get("local_to_stage"), bool):
            errors.append(f"{field}.local_to_stage must be a boolean")
        if layer_id is not None and expert_id is not None:
            key = (layer_id, expert_id)
            if key in result:
                errors.append(f"duplicate expert placement: {key}")
            result[key] = item
    return result


def _routes(value: Any, layer_id: int, top_k: int, errors: list[str]) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        errors.append("routing_trace must be a non-empty list")
        return {}
    result: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(value):
        field = f"routing_trace[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{field} must be an object")
            continue
        token_index = _non_negative_int(item.get("token_index"), f"{field}.token_index", errors)
        if item.get("layer_id") != layer_id:
            errors.append(f"{field}.layer_id must match root layer_id")
        expert_ids = _int_list(item.get("expert_ids"), f"{field}.expert_ids", errors)
        weights = _number_list(item.get("topk_weights"), f"{field}.topk_weights", errors)
        if expert_ids is not None and len(expert_ids) != top_k:
            errors.append(f"{field}.expert_ids length must equal top_k")
        if weights is not None and len(weights) != top_k:
            errors.append(f"{field}.topk_weights length must equal top_k")
        if expert_ids is not None and weights is not None:
            if len(expert_ids) != len(weights):
                errors.append(f"{field}.expert_ids and topk_weights lengths must match")
            if abs(sum(weights) - 1.0) > 1e-6:
                errors.append(f"{field}.topk_weights must sum to 1.0")
        if token_index is not None:
            result[token_index] = item
    return result


def _expected_buckets(routes: dict[int, dict[str, Any]]) -> dict[int, dict[str, list[Any]]]:
    buckets: dict[int, dict[str, list[Any]]] = {}
    for token_index, item in routes.items():
        for expert_id, weight in zip(item["expert_ids"], item["topk_weights"]):
            bucket = buckets.setdefault(
                int(expert_id), {"token_indices": [], "topk_weights": []}
            )
            bucket["token_indices"].append(token_index)
            bucket["topk_weights"].append(float(weight))
    return buckets


def validate_moe_contract_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != "moe-expert-runtime-simulation-contract":
        errors.append("record_kind must be moe-expert-runtime-simulation-contract")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors) or ""
    request_id = _non_empty_string(data.get("request_id"), "request_id", errors) or ""
    plan_hash = _non_empty_string(data.get("plan_hash"), "plan_hash", errors) or ""
    layer_id = _non_negative_int(data.get("layer_id"), "layer_id", errors) or 0
    top_k = _positive_int(data.get("top_k"), "top_k", errors) or 0
    max_remote_wait_ms = _positive_number(
        data.get("max_remote_wait_ms"), "max_remote_wait_ms", errors
    ) or 0.0
    threshold = _positive_number(
        data.get("migration_hotness_threshold"),
        "migration_hotness_threshold",
        errors,
    ) or 0.0
    if threshold > 1:
        errors.append("migration_hotness_threshold must be <= 1")
    placement = _placement_map(data.get("expert_placement"), errors)
    routes = _routes(data.get("routing_trace"), layer_id, top_k, errors)
    expected_buckets = _expected_buckets(routes)

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

    seen: set[str] = set()
    bucketed: dict[int, dict[str, Any]] = {}
    execute_open: set[tuple[int, int, str]] = set()
    executed: set[int] = set()
    result_experts: set[int] = set()
    local_dispatch_count = 0
    remote_dispatch_count = 0
    migration_count = 0
    gather_count = 0
    remote_hits = 0
    total_hits = 0
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
        seen.add(kind)
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
        if event.get("layer_id") is not None and event.get("layer_id") != layer_id:
            errors.append(f"{field}.layer_id must match root layer_id")

        if kind == "router_topk_start":
            if event.get("top_k") != top_k:
                errors.append(f"{field}.top_k must match root top_k")
            if event.get("token_count") != len(routes):
                errors.append(f"{field}.token_count must match routing_trace")
        elif kind == "router_topk_end":
            if event.get("routes") != data.get("routing_trace"):
                errors.append(f"{field}.routes must match routing_trace")
            _positive_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)
        elif kind == "expert_bucketed":
            expert_id = _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
            token_indices = _int_list(event.get("token_indices"), f"{field}.token_indices", errors)
            weights = _number_list(event.get("topk_weights"), f"{field}.topk_weights", errors)
            _non_empty_string(event.get("target_worker_id"), f"{field}.target_worker_id", errors)
            _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
            if expert_id is not None:
                expected = expected_buckets.get(expert_id)
                if expected is None:
                    errors.append(f"{field}.expert_id not present in routing_trace")
                else:
                    if token_indices != expected["token_indices"]:
                        errors.append(f"{field}.token_indices do not match routing_trace")
                    if weights != expected["topk_weights"]:
                        errors.append(f"{field}.topk_weights do not match routing_trace")
                bucketed[expert_id] = event
        elif kind in {"local_expert_dispatch", "remote_expert_dispatch"}:
            expert_id = _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
            _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
            _non_empty_string(event.get("worker_id"), f"{field}.worker_id", errors)
            _non_empty_string(event.get("node_id"), f"{field}.node_id", errors)
            token_count = _positive_int(event.get("token_count"), f"{field}.token_count", errors)
            if expert_id is not None:
                placement_row = placement.get((layer_id, expert_id))
                if placement_row is None:
                    errors.append(f"{field}.expert_id missing placement")
                else:
                    role = placement_row.get("role")
                    if kind == "local_expert_dispatch" and role not in LOCAL_ROLES:
                        errors.append(f"{field}.expert placement role is not local")
                    if kind == "remote_expert_dispatch" and role not in REMOTE_ROLES:
                        errors.append(f"{field}.expert placement role is not remote")
                    if event.get("worker_id") != placement_row.get("worker_id"):
                        errors.append(f"{field}.worker_id must match expert placement")
                bucket = bucketed.get(expert_id)
                if token_count is not None and bucket is not None:
                    if token_count != len(bucket.get("token_indices", [])):
                        errors.append(f"{field}.token_count must match bucket")
                if kind == "remote_expert_dispatch":
                    remote_wait = _non_negative_number(
                        event.get("remote_wait_ms"), f"{field}.remote_wait_ms", errors
                    )
                    if remote_wait is not None and remote_wait > max_remote_wait_ms:
                        errors.append(f"{field}.remote_wait_ms exceeds max_remote_wait_ms")
                    remote_dispatch_count += 1
                    if token_count is not None:
                        remote_hits += token_count
                else:
                    local_dispatch_count += 1
                if token_count is not None:
                    total_hits += token_count
        elif kind == "expert_execute_start":
            expert_id = _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
            worker_id = _non_empty_string(event.get("worker_id"), f"{field}.worker_id", errors)
            payload_id = _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
            if expert_id is not None and worker_id is not None and payload_id is not None:
                execute_open.add((expert_id, layer_id, payload_id))
        elif kind == "expert_execute_end":
            expert_id = _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
            payload_id = _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
            _non_empty_string(event.get("result_payload_id"), f"{field}.result_payload_id", errors)
            _positive_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)
            if expert_id is not None and payload_id is not None:
                key = (expert_id, layer_id, payload_id)
                if key not in execute_open:
                    errors.append(f"{field} has no matching expert_execute_start")
                else:
                    execute_open.remove(key)
                    executed.add(expert_id)
        elif kind == "expert_result_received":
            expert_id = _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
            _non_empty_string(event.get("result_payload_id"), f"{field}.result_payload_id", errors)
            token_indices = _int_list(event.get("token_indices"), f"{field}.token_indices", errors)
            if expert_id is not None:
                if expert_id not in executed:
                    errors.append(f"{field} result received before expert_execute_end")
                expected = expected_buckets.get(expert_id)
                if expected is not None and token_indices != expected["token_indices"]:
                    errors.append(f"{field}.token_indices do not match bucket")
                result_experts.add(expert_id)
        elif kind == "migration_recommendation":
            expert_id = _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
            _non_empty_string(event.get("from_node_id"), f"{field}.from_node_id", errors)
            _non_empty_string(event.get("to_node_id"), f"{field}.to_node_id", errors)
            hotness = _non_negative_number(event.get("hotness"), f"{field}.hotness", errors)
            if hotness is not None and hotness < threshold:
                errors.append(f"{field}.hotness must be >= migration threshold")
            if event.get("decision") != "migrate_next_window":
                errors.append(f"{field}.decision must be migrate_next_window")
            if expert_id is not None and expert_id not in expected_buckets:
                errors.append(f"{field}.expert_id not present in routing_trace")
            migration_count += 1
        elif kind == "weighted_gather_start":
            token_indices = _int_list(event.get("token_indices"), f"{field}.token_indices", errors)
            if token_indices is not None and sorted(token_indices) != sorted(routes):
                errors.append(f"{field}.token_indices must cover routed tokens")
            payloads = event.get("source_result_payloads")
            if not isinstance(payloads, list) or not payloads:
                errors.append(f"{field}.source_result_payloads must be a non-empty list")
        elif kind == "weighted_gather_end":
            token_indices = _int_list(event.get("token_indices"), f"{field}.token_indices", errors)
            if token_indices is not None and sorted(token_indices) != sorted(routes):
                errors.append(f"{field}.token_indices must cover routed tokens")
            _non_empty_string(event.get("output_payload_id"), f"{field}.output_payload_id", errors)
            _non_empty_string(event.get("output_checksum"), f"{field}.output_checksum", errors)
            _positive_number(event.get("elapsed_ms"), f"{field}.elapsed_ms", errors)
            gather_count += 1
        elif kind == "expert_trace_recorded":
            _non_empty_string(event.get("trace_id"), f"{field}.trace_id", errors)
            hot_experts = _int_list(event.get("hot_experts"), f"{field}.hot_experts", errors)
            candidates = _int_list(
                event.get("migration_candidates"), f"{field}.migration_candidates", errors
            )
            remote_hit_rate = _non_negative_number(
                event.get("remote_hit_rate"), f"{field}.remote_hit_rate", errors
            )
            if remote_hit_rate is not None and abs(remote_hit_rate - (remote_hits / max(total_hits, 1))) > 1e-6:
                errors.append(f"{field}.remote_hit_rate does not match dispatch events")
            if hot_experts is not None and candidates is not None:
                if not set(candidates).issubset(set(hot_experts)):
                    errors.append(f"{field}.migration_candidates must be hot experts")
        elif kind == "cleanup":
            payloads = event.get("released_payloads")
            if not isinstance(payloads, list) or not payloads:
                errors.append(f"{field}.released_payloads must be a non-empty list")
            cleanup_count += 1

    missing = [kind for kind in REQUIRED_EVENT_KINDS if kind not in seen]
    if missing:
        errors.append("events missing required MoE events: " + ", ".join(missing))
    if execute_open:
        errors.append("expert_execute_start without end: " + str(sorted(execute_open)))
    if set(expected_buckets) - set(bucketed):
        errors.append("missing expert buckets for routed experts")
    if set(expected_buckets) - result_experts:
        errors.append("missing expert results for routed experts")
    if gather_count != 1:
        errors.append("exactly one weighted gather is required")
    if migration_count < 1:
        errors.append("at least one migration recommendation is required")
    if cleanup_count != 1:
        errors.append("exactly one cleanup event is required")

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    expected = {
        "event_count": len(events),
        "token_count": len(routes),
        "expert_count": len(placement),
        "remote_dispatch_count": remote_dispatch_count,
        "local_dispatch_count": local_dispatch_count,
        "migration_recommendation_count": migration_count,
        "weighted_gather_count": gather_count,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            errors.append(f"summary.{field} does not match events")
    if total_hits and abs(float(summary.get("remote_hit_rate", -1.0)) - (remote_hits / total_hits)) > 1e-6:
        errors.append("summary.remote_hit_rate does not match dispatch events")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "event_count": len(events),
            "token_count": len(routes),
            "expert_count": len(placement),
            "remote_dispatch_count": remote_dispatch_count,
            "local_dispatch_count": local_dispatch_count,
            "migration_recommendation_count": migration_count,
            "weighted_gather_count": gather_count,
            "remote_hit_rate": remote_hits / max(total_hits, 1),
            "required_events_seen": sorted(seen & set(REQUIRED_EVENT_KINDS)),
        },
    }


def validate_moe_contract(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing MoE runtime fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid MoE runtime fixture: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["MoE runtime fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_moe_contract_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
