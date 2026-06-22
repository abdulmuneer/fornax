from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json
from .moe_parity import (
    _checksum,
    _inputs,
    _local_expert_ids,
    _max_abs_error,
    _remote_expert_ids,
    _run_reference_layer,
    _run_split_layer,
    _validate_config,
)

CONTRACT_KIND = "moe-hot-expert-migration-simulation"
REQUIRED_EVENT_KINDS = (
    "placement_snapshot_before",
    "hot_expert_detected",
    "migration_recommendation",
    "migration_plan",
    "drain_started",
    "expert_state_copied",
    "placement_committed",
    "routing_replayed",
    "parity_verified",
    "cleanup",
)


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


def _hot_route_trace(
    token_count: int,
    expert_count: int,
    top_k: int,
    hot_expert_id: int,
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for token in range(token_count):
        expert_ids = [hot_expert_id]
        candidate = (token + 1) % expert_count
        while len(expert_ids) < top_k:
            if candidate not in expert_ids:
                expert_ids.append(candidate)
            candidate = (candidate + 1) % expert_count
        raw = [float(top_k - rank) + 0.2 * ((token + rank) % 2) for rank in range(top_k)]
        total = sum(raw)
        routes.append(
            {
                "token_index": token,
                "expert_ids": expert_ids,
                "topk_weights": [value / total for value in raw],
            }
        )
    return routes


def _token_copies(routes: list[dict[str, Any]], expert_ids: set[int]) -> int:
    return sum(
        1
        for route in routes
        for expert_id in route["expert_ids"]
        if int(expert_id) in expert_ids
    )


def _remote_bucket_count(routes: list[dict[str, Any]], expert_ids: set[int]) -> int:
    return len(
        {
            int(expert_id)
            for route in routes
            for expert_id in route["expert_ids"]
            if int(expert_id) in expert_ids
        }
    )


def _placement(
    expert_count: int,
    local_ids: list[int],
    logical_source_host: str,
    logical_expert_host: str,
) -> list[dict[str, Any]]:
    local_set = set(local_ids)
    return [
        {
            "expert_id": expert_id,
            "logical_host": logical_source_host if expert_id in local_set else logical_expert_host,
            "local_to_source": expert_id in local_set,
            "role": "hot_resident" if expert_id in local_set else "warm_remote",
        }
        for expert_id in range(expert_count)
    ]


def _event(
    kind: str,
    *,
    timestamp_s: float,
    plan_id: str,
    request_id: str,
    plan_hash: str,
    **fields: Any,
) -> dict[str, Any]:
    data = {
        "kind": kind,
        "timestamp_s": timestamp_s,
        "plan_id": plan_id,
        "request_id": request_id,
        "plan_hash": plan_hash,
    }
    data.update(fields)
    return data


def simulated_moe_hot_expert_migration(
    *,
    plan_id: str = "moe-migration-simulated-plan",
    request_id: str = "req-moe-migration-simulated",
    plan_hash: str = "sha256:moe-migration-simulated-plan",
    token_count: int = 6,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    vocab_size: int = 17,
    expert_count: int = 4,
    top_k: int = 2,
    hot_expert_id: int = 1,
    migration_hotness_threshold: float = 0.45,
    tolerance: float = 0.0,
    logical_source_host: str = "logical-host-0",
    logical_expert_host: str = "logical-host-1",
) -> dict[str, Any]:
    """Build a deterministic T1 hot-expert migration simulation contract."""

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    if not request_id:
        raise ValueError("request_id must be non-empty")
    if not plan_hash:
        raise ValueError("plan_hash must be non-empty")
    _validate_config(
        iterations=1,
        warmup=0,
        token_count=token_count,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
        expert_count=expert_count,
        top_k=top_k,
        tolerance=tolerance,
    )
    if isinstance(hot_expert_id, bool) or hot_expert_id < 0 or hot_expert_id >= expert_count:
        raise ValueError("hot_expert_id must be in [0, expert_count)")
    if (
        isinstance(migration_hotness_threshold, bool)
        or not isinstance(migration_hotness_threshold, (int, float))
        or migration_hotness_threshold <= 0
        or migration_hotness_threshold > 1
    ):
        raise ValueError("migration_hotness_threshold must be in (0, 1]")
    if not logical_source_host or not logical_expert_host:
        raise ValueError("logical host names must be non-empty")
    if logical_source_host == logical_expert_host:
        raise ValueError("logical host names must differ")

    local_before = _local_expert_ids(expert_count)
    remote_before = _remote_expert_ids(expert_count)
    if hot_expert_id not in remote_before:
        raise ValueError("hot_expert_id must start on the remote expert host")
    local_after = sorted(local_before + [hot_expert_id])
    remote_after = [expert_id for expert_id in remote_before if expert_id != hot_expert_id]
    routes = _hot_route_trace(token_count, expert_count, top_k, hot_expert_id)
    inputs = _inputs(token_count, hidden_dim)

    reference_layer, reference_logits, reference_next_tokens = _run_reference_layer(
        inputs,
        routes,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
    )
    pre_layer, pre_logits, pre_next_tokens, _ = _run_split_layer(
        inputs,
        routes,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
        remote_expert_ids=set(remote_before),
    )
    post_layer, post_logits, post_next_tokens, _ = _run_split_layer(
        inputs,
        routes,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
        remote_expert_ids=set(remote_after),
    )
    pre_remote_copies = _token_copies(routes, set(remote_before))
    post_remote_copies = _token_copies(routes, set(remote_after))
    hot_copies = _token_copies(routes, {hot_expert_id})
    hotness = hot_copies / float(token_count * top_k)
    pre_remote_batches = _remote_bucket_count(routes, set(remote_before))
    post_remote_batches = _remote_bucket_count(routes, set(remote_after))
    max_pre_layer_abs_error = _max_abs_error(reference_layer, pre_layer)
    max_pre_logit_abs_error = _max_abs_error(reference_logits, pre_logits)
    max_post_layer_abs_error = _max_abs_error(reference_layer, post_layer)
    max_post_logit_abs_error = _max_abs_error(reference_logits, post_logits)
    next_tokens_match = (
        pre_next_tokens == reference_next_tokens
        and post_next_tokens == reference_next_tokens
    )
    correctness_passed = (
        hotness >= migration_hotness_threshold
        and pre_remote_copies > post_remote_copies
        and next_tokens_match
        and max_pre_layer_abs_error <= tolerance
        and max_pre_logit_abs_error <= tolerance
        and max_post_layer_abs_error <= tolerance
        and max_post_logit_abs_error <= tolerance
    )
    placement_before = _placement(
        expert_count,
        local_before,
        logical_source_host,
        logical_expert_host,
    )
    placement_after = _placement(
        expert_count,
        local_after,
        logical_source_host,
        logical_expert_host,
    )
    events = [
        _event(
            "placement_snapshot_before",
            timestamp_s=0.000,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            expert_placement=placement_before,
            remote_token_copies=pre_remote_copies,
        ),
        _event(
            "hot_expert_detected",
            timestamp_s=0.001,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            expert_id=hot_expert_id,
            hotness=hotness,
            threshold=float(migration_hotness_threshold),
        ),
        _event(
            "migration_recommendation",
            timestamp_s=0.002,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            expert_id=hot_expert_id,
            from_logical_host=logical_expert_host,
            to_logical_host=logical_source_host,
            reason="remote hotness exceeds migration threshold",
        ),
        _event(
            "migration_plan",
            timestamp_s=0.003,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            expert_id=hot_expert_id,
            drain_required=True,
            expected_remote_call_reduction=pre_remote_copies - post_remote_copies,
        ),
        _event(
            "drain_started",
            timestamp_s=0.004,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            expert_id=hot_expert_id,
            in_flight_batches=0,
        ),
        _event(
            "expert_state_copied",
            timestamp_s=0.005,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            expert_id=hot_expert_id,
            source_logical_host=logical_expert_host,
            destination_logical_host=logical_source_host,
            bytes_copied=hidden_dim * intermediate_dim * 8 * 2,
        ),
        _event(
            "placement_committed",
            timestamp_s=0.006,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            expert_id=hot_expert_id,
            expert_placement=placement_after,
        ),
        _event(
            "routing_replayed",
            timestamp_s=0.007,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            pre_remote_token_copies=pre_remote_copies,
            post_remote_token_copies=post_remote_copies,
        ),
        _event(
            "parity_verified",
            timestamp_s=0.008,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            max_post_layer_abs_error=max_post_layer_abs_error,
            max_post_logit_abs_error=max_post_logit_abs_error,
            next_tokens_match=next_tokens_match,
        ),
        _event(
            "cleanup",
            timestamp_s=0.009,
            plan_id=plan_id,
            request_id=request_id,
            plan_hash=plan_hash,
            stale_remote_expert_evicted=True,
        ),
    ]

    return {
        "version": 1,
        "contract_kind": CONTRACT_KIND,
        "tier": "T1-simulation",
        "simulation_method": "two-logical-host-hot-expert-migration",
        "plan_id": plan_id,
        "request_id": request_id,
        "plan_hash": plan_hash,
        "config": {
            "token_count": token_count,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "vocab_size": vocab_size,
            "expert_count": expert_count,
            "top_k": top_k,
            "hot_expert_id": hot_expert_id,
            "migration_hotness_threshold": float(migration_hotness_threshold),
            "tolerance": tolerance,
            "logical_source_host": logical_source_host,
            "logical_expert_host": logical_expert_host,
            "local_expert_ids_before": local_before,
            "remote_expert_ids_before": remote_before,
            "local_expert_ids_after": local_after,
            "remote_expert_ids_after": remote_after,
        },
        "routing": {"routes": routes},
        "placement": {"before": placement_before, "after": placement_after},
        "events": events,
        "result": {
            "hotness": hotness,
            "hot_expert_migrated": True,
            "pre_remote_token_copies": pre_remote_copies,
            "post_remote_token_copies": post_remote_copies,
            "remote_token_copy_reduction": pre_remote_copies - post_remote_copies,
            "pre_remote_batches": pre_remote_batches,
            "post_remote_batches": post_remote_batches,
            "pre_next_tokens": pre_next_tokens,
            "post_next_tokens": post_next_tokens,
            "reference_next_tokens": reference_next_tokens,
            "next_tokens_match": next_tokens_match,
            "pre_layer_checksum": _checksum(pre_layer),
            "post_layer_checksum": _checksum(post_layer),
            "reference_layer_checksum": _checksum(reference_layer),
            "pre_logit_checksum": _checksum(pre_logits),
            "post_logit_checksum": _checksum(post_logits),
            "reference_logit_checksum": _checksum(reference_logits),
            "max_pre_layer_abs_error": max_pre_layer_abs_error,
            "max_pre_logit_abs_error": max_pre_logit_abs_error,
            "max_post_layer_abs_error": max_post_layer_abs_error,
            "max_post_logit_abs_error": max_post_logit_abs_error,
            "dropped_tokens": 0,
            "correctness_passed": correctness_passed,
        },
        "summary": {
            "event_count": len(events),
            "migration_count": 1,
            "hot_expert_id": hot_expert_id,
            "remote_token_copy_reduction": pre_remote_copies - post_remote_copies,
            "pre_remote_batches": pre_remote_batches,
            "post_remote_batches": post_remote_batches,
            "max_post_logit_abs_error": max_post_logit_abs_error,
            "correctness_passed": correctness_passed,
        },
        "note": (
            "T1 simulated hot-expert migration contract; validates placement update, "
            "remote-call reduction, and parity preservation without claiming real "
            "multi-host migration evidence."
        ),
    }


def _placement_locality(placement: Any, expert_id: int, errors: list[str], field: str) -> bool | None:
    if not isinstance(placement, list):
        errors.append(f"{field} must be a list")
        return None
    matches = [
        row
        for row in placement
        if isinstance(row, dict) and row.get("expert_id") == expert_id
    ]
    if len(matches) != 1:
        errors.append(f"{field} must contain exactly one row for hot_expert_id")
        return None
    value = matches[0].get("local_to_source")
    if not isinstance(value, bool):
        errors.append(f"{field} hot expert local_to_source must be a boolean")
        return None
    return value


def validate_moe_hot_expert_migration_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("contract_kind") != CONTRACT_KIND:
        errors.append(f"contract_kind must be {CONTRACT_KIND}")
    if data.get("tier") != "T1-simulation":
        errors.append("tier must be T1-simulation")
    _non_empty_string(data.get("simulation_method"), "simulation_method", errors)
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    request_id = _non_empty_string(data.get("request_id"), "request_id", errors)
    plan_hash = _non_empty_string(data.get("plan_hash"), "plan_hash", errors)
    config = data.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
        config = {}
    token_count = _positive_int(config.get("token_count"), "config.token_count", errors)
    hidden_dim = _positive_int(config.get("hidden_dim"), "config.hidden_dim", errors)
    _positive_int(config.get("intermediate_dim"), "config.intermediate_dim", errors)
    _positive_int(config.get("vocab_size"), "config.vocab_size", errors)
    expert_count = _positive_int(config.get("expert_count"), "config.expert_count", errors)
    top_k = _positive_int(config.get("top_k"), "config.top_k", errors)
    hot_expert_id = _non_negative_int(config.get("hot_expert_id"), "config.hot_expert_id", errors)
    threshold = _positive_number(
        config.get("migration_hotness_threshold"),
        "config.migration_hotness_threshold",
        errors,
    )
    if threshold is not None and threshold > 1:
        errors.append("config.migration_hotness_threshold must be <= 1")
    _non_negative_number(config.get("tolerance"), "config.tolerance", errors)
    logical_source_host = _non_empty_string(config.get("logical_source_host"), "config.logical_source_host", errors)
    logical_expert_host = _non_empty_string(config.get("logical_expert_host"), "config.logical_expert_host", errors)
    if logical_source_host == logical_expert_host and logical_source_host is not None:
        errors.append("config.logical_source_host must differ from logical_expert_host")
    local_before = _int_list(config.get("local_expert_ids_before"), "config.local_expert_ids_before", errors)
    remote_before = _int_list(config.get("remote_expert_ids_before"), "config.remote_expert_ids_before", errors)
    local_after = _int_list(config.get("local_expert_ids_after"), "config.local_expert_ids_after", errors)
    remote_after = _int_list(config.get("remote_expert_ids_after"), "config.remote_expert_ids_after", errors)
    if expert_count is not None and hot_expert_id is not None and hot_expert_id >= expert_count:
        errors.append("config.hot_expert_id must be < config.expert_count")
    if hot_expert_id is not None and remote_before is not None and hot_expert_id not in remote_before:
        errors.append("hot expert must start in config.remote_expert_ids_before")
    if hot_expert_id is not None and local_after is not None and hot_expert_id not in local_after:
        errors.append("hot expert must appear in config.local_expert_ids_after")
    if hot_expert_id is not None and remote_after is not None and hot_expert_id in remote_after:
        errors.append("hot expert must be removed from config.remote_expert_ids_after")
    routing = data.get("routing")
    if not isinstance(routing, dict) or not isinstance(routing.get("routes"), list):
        errors.append("routing.routes must be a list")
    elif token_count is not None and len(routing["routes"]) != token_count:
        errors.append("routing.routes length must equal config.token_count")
    placement = data.get("placement")
    if not isinstance(placement, dict):
        errors.append("placement must be an object")
        placement = {}
    if hot_expert_id is not None:
        before_local = _placement_locality(placement.get("before"), hot_expert_id, errors, "placement.before")
        after_local = _placement_locality(placement.get("after"), hot_expert_id, errors, "placement.after")
        if before_local is True:
            errors.append("placement.before hot expert must be remote")
        if after_local is not True:
            errors.append("placement.after hot expert must be local")
    events = data.get("events")
    if not isinstance(events, list):
        errors.append("events must be a list")
        events = []
    kinds = [event.get("kind") for event in events if isinstance(event, dict)]
    if kinds != list(REQUIRED_EVENT_KINDS):
        errors.append("events must contain the required migration sequence in order")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"events[{index}] must be an object")
            continue
        field = f"events[{index}]"
        if event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match plan_id")
        if event.get("request_id") != request_id:
            errors.append(f"{field}.request_id must match request_id")
        if event.get("plan_hash") != plan_hash:
            errors.append(f"{field}.plan_hash must match plan_hash")
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    hotness = _non_negative_number(result.get("hotness"), "result.hotness", errors)
    if hotness is not None and threshold is not None and hotness < threshold:
        errors.append("result.hotness must be >= migration threshold")
    pre_remote = _positive_int(result.get("pre_remote_token_copies"), "result.pre_remote_token_copies", errors)
    post_remote = _non_negative_int(result.get("post_remote_token_copies"), "result.post_remote_token_copies", errors)
    reduction = _positive_int(result.get("remote_token_copy_reduction"), "result.remote_token_copy_reduction", errors)
    if pre_remote is not None and post_remote is not None and reduction is not None and reduction != pre_remote - post_remote:
        errors.append("result.remote_token_copy_reduction must equal pre minus post remote token copies")
    _positive_int(result.get("pre_remote_batches"), "result.pre_remote_batches", errors)
    _non_negative_int(result.get("post_remote_batches"), "result.post_remote_batches", errors)
    if result.get("hot_expert_migrated") is not True:
        errors.append("result.hot_expert_migrated must be true")
    if result.get("next_tokens_match") is not True:
        errors.append("result.next_tokens_match must be true")
    if result.get("dropped_tokens") != 0:
        errors.append("result.dropped_tokens must be 0")
    if result.get("correctness_passed") is not True:
        errors.append("result.correctness_passed must be true")
    for name in (
        "pre_layer_checksum",
        "post_layer_checksum",
        "reference_layer_checksum",
        "pre_logit_checksum",
        "post_logit_checksum",
        "reference_logit_checksum",
    ):
        _non_negative_number(abs(result.get(name, 0.0)) if isinstance(result.get(name), (int, float)) else result.get(name), f"result.{name}", errors)
    for name in (
        "max_pre_layer_abs_error",
        "max_pre_logit_abs_error",
        "max_post_layer_abs_error",
        "max_post_logit_abs_error",
    ):
        _non_negative_number(result.get(name), f"result.{name}", errors)
    if token_count is not None:
        for name in ("pre_next_tokens", "post_next_tokens", "reference_next_tokens"):
            values = result.get(name)
            if not isinstance(values, list) or len(values) != token_count:
                errors.append(f"result.{name} must contain config.token_count tokens")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must equal len(events)")
    if summary.get("migration_count") != 1:
        errors.append("summary.migration_count must be 1")
    if hot_expert_id is not None and summary.get("hot_expert_id") != hot_expert_id:
        errors.append("summary.hot_expert_id must match config.hot_expert_id")
    if reduction is not None and summary.get("remote_token_copy_reduction") != reduction:
        errors.append("summary.remote_token_copy_reduction must match result")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")
    warnings.append("hot-expert migration is simulation evidence, not real cluster migration evidence")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "event_count": len(events),
            "migration_count": summary.get("migration_count"),
            "hot_expert_id": hot_expert_id,
            "remote_token_copy_reduction": reduction,
            "pre_remote_batches": result.get("pre_remote_batches"),
            "post_remote_batches": result.get("post_remote_batches"),
            "max_post_logit_abs_error": result.get("max_post_logit_abs_error"),
            "correctness_passed": result.get("correctness_passed") is True,
        },
    }


def validate_moe_hot_expert_migration(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid MoE hot-expert migration artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["MoE hot-expert migration artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_moe_hot_expert_migration_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
