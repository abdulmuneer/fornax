from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "resilience-replay-simulation-contract"
MODE = "t1-simulation"
REQUIRED_EVENT_KINDS = {
    "request_started",
    "checkpoint_recorded",
    "node_loss_detected",
    "replay_scheduled",
    "replay_started",
    "replay_completed",
    "request_completed",
    "cleanup",
}


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


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _default_requests() -> list[dict[str, Any]]:
    return [
        {"request_id": "res-r0", "prompt_len": 8, "gen_len": 5, "seed": 3},
        {"request_id": "res-r1", "prompt_len": 7, "gen_len": 4, "seed": 11},
        {"request_id": "res-r2", "prompt_len": 6, "gen_len": 3, "seed": 19},
    ]


def _normalize_requests(requests: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    raw = _default_requests() if requests is None else requests
    if not isinstance(raw, list) or not raw:
        raise ValueError("requests must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, request in enumerate(raw):
        if not isinstance(request, dict):
            raise ValueError(f"requests[{index}] must be an object")
        request_id = str(request.get("request_id") or request.get("id") or f"res-r{index}")
        if request_id in seen:
            raise ValueError("request ids must be unique")
        seen.add(request_id)
        prompt_len = int(request.get("prompt_len", request.get("prompt_tokens", 0)))
        gen_len = int(request.get("gen_len", request.get("max_new_tokens", 0)))
        seed = int(request.get("seed", index + 1))
        if prompt_len < 0 or gen_len <= 0:
            raise ValueError(f"{request_id} must have non-negative prompt_len and positive gen_len")
        normalized.append(
            {
                "request_id": request_id,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "seed": seed,
            }
        )
    return normalized


def _tokens(seed: int, gen_len: int, vocab_size: int) -> list[int]:
    return [int((seed + step * 7 + 5) % vocab_size) for step in range(gen_len)]


def _checksum(tokens: list[int]) -> float:
    return sum(float(token) * (1.0 + index * 0.01) for index, token in enumerate(tokens))


def _event(kind: str, *, timestamp_s: float, plan_id: str, **fields: Any) -> dict[str, Any]:
    event = {"kind": kind, "timestamp_s": round(timestamp_s, 9), "plan_id": plan_id}
    event.update(fields)
    return event


def simulate_resilience_replay(
    *,
    plan_id: str = "resilience-replay-plan",
    requests: list[dict[str, Any]] | None = None,
    failed_node_id: str = "logical-host-1",
    replay_node_id: str = "logical-host-0",
    checkpoint_token_index: int = 2,
    node_loss_time_s: float = 0.050,
    replay_delay_s: float = 0.010,
    token_time_s: float = 0.006,
    max_replay_delay_s: float = 0.025,
    vocab_size: int = 97,
) -> dict[str, Any]:
    """Build a deterministic replay simulation for single-node loss."""

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    errors: list[str] = []
    _non_empty_string(failed_node_id, "failed_node_id", errors)
    _non_empty_string(replay_node_id, "replay_node_id", errors)
    _non_negative_int(checkpoint_token_index, "checkpoint_token_index", errors)
    _positive_number(node_loss_time_s, "node_loss_time_s", errors)
    _non_negative_number(replay_delay_s, "replay_delay_s", errors)
    _positive_number(token_time_s, "token_time_s", errors)
    _positive_number(max_replay_delay_s, "max_replay_delay_s", errors)
    _positive_int(vocab_size, "vocab_size", errors)
    if failed_node_id == replay_node_id:
        errors.append("failed_node_id and replay_node_id must differ")
    normalized = _normalize_requests(requests)
    if errors:
        raise ValueError("; ".join(errors))

    events: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, request in enumerate(normalized):
        request_id = request["request_id"]
        gen_len = int(request["gen_len"])
        checkpoint_index = min(checkpoint_token_index, gen_len)
        reference_tokens = _tokens(int(request["seed"]), gen_len, vocab_size)
        emitted_before_failure = reference_tokens[:checkpoint_index]
        replayed_tokens = reference_tokens[checkpoint_index:]
        replay_start = node_loss_time_s + replay_delay_s + index * 0.001
        replay_end = replay_start + len(replayed_tokens) * token_time_s
        completed_tokens = emitted_before_failure + replayed_tokens
        max_abs_error = abs(_checksum(completed_tokens) - _checksum(reference_tokens))
        duplicate_token_count = len(completed_tokens) - len(reference_tokens)
        dropped_token_count = len(reference_tokens) - len(completed_tokens)
        events.append(
            _event(
                "request_started",
                timestamp_s=0.000 + index * 0.001,
                plan_id=plan_id,
                request_id=request_id,
                node_id=failed_node_id,
                prompt_len=request["prompt_len"],
                gen_len=gen_len,
            )
        )
        events.append(
            _event(
                "checkpoint_recorded",
                timestamp_s=max(0.0, node_loss_time_s - 0.010 + index * 0.001),
                plan_id=plan_id,
                request_id=request_id,
                node_id=failed_node_id,
                checkpoint_token_index=checkpoint_index,
                emitted_tokens=emitted_before_failure,
            )
        )
        results.append(
            {
                "request_id": request_id,
                "checkpoint_token_index": checkpoint_index,
                "reference_tokens": reference_tokens,
                "emitted_before_failure": emitted_before_failure,
                "replayed_tokens": replayed_tokens,
                "completed_tokens": completed_tokens,
                "reference_checksum": _checksum(reference_tokens),
                "completed_checksum": _checksum(completed_tokens),
                "max_abs_error": max_abs_error,
                "duplicate_token_count": duplicate_token_count,
                "dropped_token_count": dropped_token_count,
                "replay_start_s": round(replay_start, 9),
                "replay_end_s": round(replay_end, 9),
                "replay_delay_s": round(replay_delay_s, 9),
                "completed": completed_tokens == reference_tokens,
            }
        )
    events.append(
        _event(
            "node_loss_detected",
            timestamp_s=node_loss_time_s,
            plan_id=plan_id,
            failed_node_id=failed_node_id,
            affected_request_ids=[request["request_id"] for request in normalized],
        )
    )
    for result in results:
        request_id = result["request_id"]
        events.append(
            _event(
                "replay_scheduled",
                timestamp_s=node_loss_time_s + replay_delay_s,
                plan_id=plan_id,
                request_id=request_id,
                failed_node_id=failed_node_id,
                replay_node_id=replay_node_id,
                from_checkpoint_token_index=result["checkpoint_token_index"],
            )
        )
        events.append(
            _event(
                "replay_started",
                timestamp_s=result["replay_start_s"],
                plan_id=plan_id,
                request_id=request_id,
                replay_node_id=replay_node_id,
            )
        )
        events.append(
            _event(
                "replay_completed",
                timestamp_s=result["replay_end_s"],
                plan_id=plan_id,
                request_id=request_id,
                replay_node_id=replay_node_id,
                replayed_token_count=len(result["replayed_tokens"]),
            )
        )
        events.append(
            _event(
                "request_completed",
                timestamp_s=result["replay_end_s"],
                plan_id=plan_id,
                request_id=request_id,
                completed_token_count=len(result["completed_tokens"]),
                dropped_token_count=result["dropped_token_count"],
            )
        )
    events.append(
        _event(
            "cleanup",
            timestamp_s=max(result["replay_end_s"] for result in results) + 0.001,
            plan_id=plan_id,
            failed_node_state_released=True,
        )
    )

    dropped_requests = [result["request_id"] for result in results if not result["completed"]]
    duplicate_token_count = sum(max(0, result["duplicate_token_count"]) for result in results)
    dropped_token_count = sum(max(0, result["dropped_token_count"]) for result in results)
    max_abs_error = max(result["max_abs_error"] for result in results)
    max_replay_delay = max(result["replay_delay_s"] for result in results)
    correctness_passed = (
        not dropped_requests
        and duplicate_token_count == 0
        and dropped_token_count == 0
        and max_abs_error == 0.0
        and max_replay_delay <= max_replay_delay_s
    )
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "simulation_method": "single-node-loss-replay",
        "config": {
            "failed_node_id": failed_node_id,
            "replay_node_id": replay_node_id,
            "checkpoint_token_index": checkpoint_token_index,
            "node_loss_time_s": node_loss_time_s,
            "replay_delay_s": replay_delay_s,
            "token_time_s": token_time_s,
            "max_replay_delay_s": max_replay_delay_s,
            "vocab_size": vocab_size,
        },
        "requests": normalized,
        "events": events,
        "results": results,
        "summary": {
            "request_count": len(normalized),
            "in_flight_request_count": len(normalized),
            "replayed_request_count": len(results),
            "dropped_request_count": len(dropped_requests),
            "dropped_token_count": dropped_token_count,
            "duplicate_token_count": duplicate_token_count,
            "max_abs_error": max_abs_error,
            "max_replay_delay_s": max_replay_delay,
            "replay_delay_within_budget": max_replay_delay <= max_replay_delay_s,
            "event_count": len(events),
            "zero_dropped_in_flight": len(dropped_requests) == 0 and dropped_token_count == 0,
            "correctness_passed": correctness_passed,
        },
        "note": (
            "T1 resilience replay simulation: validates single-node-loss replay, "
            "zero dropped in-flight requests, no duplicate tokens, and deterministic "
            "output recovery. Not real T4 fault-tolerance evidence."
        ),
    }


def validate_resilience_replay_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    _non_empty_string(data.get("simulation_method"), "simulation_method", errors)
    config = data.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
        config = {}
    failed_node_id = _non_empty_string(config.get("failed_node_id"), "config.failed_node_id", errors)
    replay_node_id = _non_empty_string(config.get("replay_node_id"), "config.replay_node_id", errors)
    if failed_node_id == replay_node_id and failed_node_id is not None:
        errors.append("config.failed_node_id and replay_node_id must differ")
    _non_negative_int(config.get("checkpoint_token_index"), "config.checkpoint_token_index", errors)
    _positive_number(config.get("node_loss_time_s"), "config.node_loss_time_s", errors)
    _non_negative_number(config.get("replay_delay_s"), "config.replay_delay_s", errors)
    _positive_number(config.get("token_time_s"), "config.token_time_s", errors)
    max_replay_delay_s = _positive_number(config.get("max_replay_delay_s"), "config.max_replay_delay_s", errors)
    _positive_int(config.get("vocab_size"), "config.vocab_size", errors)
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        errors.append("requests must be a non-empty list")
        requests = []
    request_ids: set[str] = set()
    for index, request in enumerate(requests):
        field = f"requests[{index}]"
        if not isinstance(request, dict):
            errors.append(f"{field} must be an object")
            continue
        request_id = _non_empty_string(request.get("request_id"), f"{field}.request_id", errors)
        _non_negative_int(request.get("prompt_len"), f"{field}.prompt_len", errors)
        _positive_int(request.get("gen_len"), f"{field}.gen_len", errors)
        _non_negative_int(request.get("seed"), f"{field}.seed", errors)
        if request_id is not None:
            if request_id in request_ids:
                errors.append("request ids must be unique")
            request_ids.add(request_id)
    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []
    event_kinds = {event.get("kind") for event in events if isinstance(event, dict)}
    missing = REQUIRED_EVENT_KINDS - event_kinds
    if missing:
        errors.append(f"events missing required kinds: {sorted(missing)}")
    node_loss_count = 0
    scheduled_ids: set[str] = set()
    replay_started_ids: set[str] = set()
    replay_completed_ids: set[str] = set()
    completed_ids: set[str] = set()
    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        kind = _non_empty_string(event.get("kind"), f"{field}.kind", errors)
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        if event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match plan_id")
        request_id = event.get("request_id")
        if kind == "node_loss_detected":
            node_loss_count += 1
            if event.get("failed_node_id") != failed_node_id:
                errors.append(f"{field}.failed_node_id must match config.failed_node_id")
            affected = event.get("affected_request_ids")
            if not isinstance(affected, list) or set(affected) != request_ids:
                errors.append(f"{field}.affected_request_ids must match all requests")
        elif kind == "replay_scheduled":
            if request_id in scheduled_ids:
                errors.append(f"{field}.request_id scheduled more than once")
            scheduled_ids.add(str(request_id))
            if event.get("failed_node_id") != failed_node_id:
                errors.append(f"{field}.failed_node_id must match config")
            if event.get("replay_node_id") != replay_node_id:
                errors.append(f"{field}.replay_node_id must match config")
        elif kind == "replay_started":
            replay_started_ids.add(str(request_id))
        elif kind == "replay_completed":
            replay_completed_ids.add(str(request_id))
        elif kind == "request_completed":
            completed_ids.add(str(request_id))
            if event.get("dropped_token_count") != 0:
                errors.append(f"{field}.dropped_token_count must be 0")
    if node_loss_count != 1:
        errors.append("exactly one node_loss_detected event is required")
    if request_ids:
        if scheduled_ids != request_ids:
            errors.append("every request must be scheduled for replay exactly once")
        if replay_started_ids != request_ids:
            errors.append("every request must start replay")
        if replay_completed_ids != request_ids:
            errors.append("every request must complete replay")
        if completed_ids != request_ids:
            errors.append("every request must complete after replay")
    results = data.get("results")
    if not isinstance(results, list) or not results:
        errors.append("results must be a non-empty list")
        results = []
    if request_ids and len(results) != len(request_ids):
        errors.append("results length must equal request count")
    result_ids: set[str] = set()
    for index, result in enumerate(results):
        field = f"results[{index}]"
        if not isinstance(result, dict):
            errors.append(f"{field} must be an object")
            continue
        request_id = _non_empty_string(result.get("request_id"), f"{field}.request_id", errors)
        if request_id is not None:
            result_ids.add(request_id)
        reference_tokens = result.get("reference_tokens")
        completed_tokens = result.get("completed_tokens")
        replayed_tokens = result.get("replayed_tokens")
        emitted_before = result.get("emitted_before_failure")
        if not isinstance(reference_tokens, list) or not all(isinstance(token, int) for token in reference_tokens):
            errors.append(f"{field}.reference_tokens must be a list of integers")
            reference_tokens = []
        if completed_tokens != reference_tokens:
            errors.append(f"{field}.completed_tokens must match reference_tokens")
        if isinstance(emitted_before, list) and isinstance(replayed_tokens, list):
            if emitted_before + replayed_tokens != reference_tokens:
                errors.append(f"{field}.emitted_before_failure plus replayed_tokens must match reference_tokens")
        else:
            errors.append(f"{field}.emitted_before_failure and replayed_tokens must be lists")
        if result.get("duplicate_token_count") != 0:
            errors.append(f"{field}.duplicate_token_count must be 0")
        if result.get("dropped_token_count") != 0:
            errors.append(f"{field}.dropped_token_count must be 0")
        if result.get("max_abs_error") != 0.0:
            errors.append(f"{field}.max_abs_error must be 0.0")
        if result.get("completed") is not True:
            errors.append(f"{field}.completed must be true")
        _non_negative_number(result.get("replay_delay_s"), f"{field}.replay_delay_s", errors)
        if max_replay_delay_s is not None and isinstance(result.get("replay_delay_s"), (int, float)):
            if float(result["replay_delay_s"]) > max_replay_delay_s:
                errors.append(f"{field}.replay_delay_s exceeds max_replay_delay_s")
    if request_ids and result_ids != request_ids:
        errors.append("results must cover every request")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("request_count") != len(requests):
        errors.append("summary.request_count must match requests length")
    if summary.get("in_flight_request_count") != len(requests):
        errors.append("summary.in_flight_request_count must match requests length")
    if summary.get("replayed_request_count") != len(results):
        errors.append("summary.replayed_request_count must match results length")
    if summary.get("dropped_request_count") != 0:
        errors.append("summary.dropped_request_count must be 0")
    if summary.get("dropped_token_count") != 0:
        errors.append("summary.dropped_token_count must be 0")
    if summary.get("duplicate_token_count") != 0:
        errors.append("summary.duplicate_token_count must be 0")
    if summary.get("max_abs_error") != 0.0:
        errors.append("summary.max_abs_error must be 0.0")
    if max_replay_delay_s is not None and isinstance(summary.get("max_replay_delay_s"), (int, float)):
        if float(summary["max_replay_delay_s"]) > max_replay_delay_s:
            errors.append("summary.max_replay_delay_s exceeds config.max_replay_delay_s")
    if summary.get("replay_delay_within_budget") is not True:
        errors.append("summary.replay_delay_within_budget must be true")
    if summary.get("zero_dropped_in_flight") is not True:
        errors.append("summary.zero_dropped_in_flight must be true")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must equal len(events)")
    warnings.append("resilience replay is simulation evidence, not real T4 node-loss evidence")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "request_count": summary.get("request_count"),
            "replayed_request_count": summary.get("replayed_request_count"),
            "dropped_request_count": summary.get("dropped_request_count"),
            "dropped_token_count": summary.get("dropped_token_count"),
            "duplicate_token_count": summary.get("duplicate_token_count"),
            "max_abs_error": summary.get("max_abs_error"),
            "max_replay_delay_s": summary.get("max_replay_delay_s"),
            "correctness_passed": summary.get("correctness_passed") is True,
        },
    }


def validate_resilience_replay(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid resilience replay artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["resilience replay artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_resilience_replay_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
