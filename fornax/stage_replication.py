from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "stage-replication-simulation-contract"
MODE = "t1-simulation"
REQUIRED_EVENT_KINDS = {
    "baseline_plan",
    "replica_added",
    "microbatch_assigned",
    "replica_stage_start",
    "replica_stage_end",
    "output_compared",
    "throughput_compared",
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


def _positive_int_list(value: Any, field: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        parsed = _positive_int(item, f"{field}[{index}]", errors)
        if parsed is None:
            return None
        result.append(parsed)
    return result


def _microbatch_checksum(microbatch_index: int, token_count: int) -> float:
    total = 0.0
    for token in range(token_count):
        for hidden in range(8):
            value = (((microbatch_index + 3) * (token + 5) * (hidden + 7)) % 37 - 18) / 37.0
            total += value * (1.0 + token * 0.01 + hidden * 0.001)
    return total


def _event(
    kind: str,
    *,
    timestamp_s: float,
    plan_id: str,
    **fields: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": kind,
        "timestamp_s": round(timestamp_s, 9),
        "plan_id": plan_id,
    }
    data.update(fields)
    return data


def simulate_stage_replication(
    *,
    plan_id: str = "stage-replication-plan",
    bottleneck_stage_index: int = 1,
    microbatch_token_counts: list[int] | None = None,
    baseline_replica_id: str = "stage-1-replica-0",
    added_replica_id: str = "stage-1-replica-1",
    baseline_stage_time_s_per_token: float = 0.014,
    replicated_stage_time_s_per_token: float = 0.014,
    transfer_overhead_s: float = 0.001,
    speedup_floor: float = 1.25,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Simulate data-parallel replication for one bottleneck pipeline stage."""

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    errors: list[str] = []
    _non_negative_int(bottleneck_stage_index, "bottleneck_stage_index", errors)
    _non_empty_string(baseline_replica_id, "baseline_replica_id", errors)
    _non_empty_string(added_replica_id, "added_replica_id", errors)
    _positive_number(baseline_stage_time_s_per_token, "baseline_stage_time_s_per_token", errors)
    _positive_number(replicated_stage_time_s_per_token, "replicated_stage_time_s_per_token", errors)
    _non_negative_number(transfer_overhead_s, "transfer_overhead_s", errors)
    _positive_number(speedup_floor, "speedup_floor", errors)
    _non_negative_number(tolerance, "tolerance", errors)
    token_counts = list(microbatch_token_counts or [4, 4, 3, 3, 2, 2])
    _positive_int_list(token_counts, "microbatch_token_counts", errors)
    if baseline_replica_id == added_replica_id:
        errors.append("baseline_replica_id and added_replica_id must differ")
    if errors:
        raise ValueError("; ".join(errors))

    replica_ids = [baseline_replica_id, added_replica_id]
    reference_checksums = [
        _microbatch_checksum(index, token_count)
        for index, token_count in enumerate(token_counts)
    ]
    baseline_makespan = sum(
        token_count * baseline_stage_time_s_per_token + transfer_overhead_s
        for token_count in token_counts
    )
    replica_available = {replica_id: 0.0 for replica_id in replica_ids}
    assignments: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [
        _event(
            "baseline_plan",
            timestamp_s=0.0,
            plan_id=plan_id,
            stage_index=bottleneck_stage_index,
            replica_ids=[baseline_replica_id],
            baseline_makespan_s=round(baseline_makespan, 9),
        ),
        _event(
            "replica_added",
            timestamp_s=0.001,
            plan_id=plan_id,
            stage_index=bottleneck_stage_index,
            replica_id=added_replica_id,
            replica_ids=replica_ids,
        ),
    ]
    for index, token_count in enumerate(token_counts):
        replica_id = min(replica_ids, key=lambda item: replica_available[item])
        start = replica_available[replica_id]
        elapsed = token_count * replicated_stage_time_s_per_token + transfer_overhead_s
        end = start + elapsed
        replica_available[replica_id] = end
        output_checksum = _microbatch_checksum(index, token_count)
        max_abs_error = abs(output_checksum - reference_checksums[index])
        assignment = {
            "microbatch_id": f"rep-mb-{index}",
            "microbatch_index": index,
            "token_count": token_count,
            "replica_id": replica_id,
            "start_s": round(start, 9),
            "end_s": round(end, 9),
            "output_checksum": output_checksum,
            "reference_checksum": reference_checksums[index],
            "max_abs_error": max_abs_error,
        }
        assignments.append(assignment)
        events.append(
            _event(
                "microbatch_assigned",
                timestamp_s=start,
                plan_id=plan_id,
                microbatch_id=assignment["microbatch_id"],
                token_count=token_count,
                replica_id=replica_id,
            )
        )
        events.append(
            _event(
                "replica_stage_start",
                timestamp_s=start,
                plan_id=plan_id,
                microbatch_id=assignment["microbatch_id"],
                stage_index=bottleneck_stage_index,
                replica_id=replica_id,
            )
        )
        events.append(
            _event(
                "replica_stage_end",
                timestamp_s=end,
                plan_id=plan_id,
                microbatch_id=assignment["microbatch_id"],
                stage_index=bottleneck_stage_index,
                replica_id=replica_id,
                elapsed_s=round(elapsed, 9),
            )
        )
        events.append(
            _event(
                "output_compared",
                timestamp_s=end,
                plan_id=plan_id,
                microbatch_id=assignment["microbatch_id"],
                replica_id=replica_id,
                max_abs_error=max_abs_error,
            )
        )
    replicated_makespan = max(replica_available.values())
    speedup = baseline_makespan / replicated_makespan
    used_replicas = sorted({assignment["replica_id"] for assignment in assignments})
    max_abs_error = max(assignment["max_abs_error"] for assignment in assignments)
    correctness_passed = (
        len(used_replicas) == len(replica_ids)
        and speedup >= speedup_floor
        and max_abs_error <= tolerance
    )
    events.append(
        _event(
            "throughput_compared",
            timestamp_s=replicated_makespan,
            plan_id=plan_id,
            baseline_makespan_s=round(baseline_makespan, 9),
            replicated_makespan_s=round(replicated_makespan, 9),
            speedup=speedup,
            speedup_floor=speedup_floor,
        )
    )
    events.append(
        _event(
            "cleanup",
            timestamp_s=replicated_makespan + 0.001,
            plan_id=plan_id,
            replica_state_released=True,
        )
    )
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "simulation_method": "deterministic-stage-data-parallel-replication",
        "config": {
            "bottleneck_stage_index": bottleneck_stage_index,
            "baseline_replica_ids": [baseline_replica_id],
            "replicated_replica_ids": replica_ids,
            "microbatch_token_counts": token_counts,
            "baseline_stage_time_s_per_token": baseline_stage_time_s_per_token,
            "replicated_stage_time_s_per_token": replicated_stage_time_s_per_token,
            "transfer_overhead_s": transfer_overhead_s,
            "speedup_floor": speedup_floor,
            "tolerance": tolerance,
        },
        "assignments": assignments,
        "events": events,
        "result": {
            "baseline_makespan_s": round(baseline_makespan, 9),
            "replicated_makespan_s": round(replicated_makespan, 9),
            "speedup": speedup,
            "speedup_passed": speedup >= speedup_floor,
            "used_replica_ids": used_replicas,
            "microbatch_count": len(token_counts),
            "total_tokens": sum(token_counts),
            "baseline_tokens_s": sum(token_counts) / baseline_makespan,
            "replicated_tokens_s": sum(token_counts) / replicated_makespan,
            "max_abs_error": max_abs_error,
            "outputs_match_reference": max_abs_error <= tolerance,
            "correctness_passed": correctness_passed,
        },
        "summary": {
            "event_count": len(events),
            "replica_count": len(replica_ids),
            "microbatch_count": len(token_counts),
            "baseline_makespan_s": round(baseline_makespan, 9),
            "replicated_makespan_s": round(replicated_makespan, 9),
            "speedup": speedup,
            "speedup_passed": speedup >= speedup_floor,
            "max_abs_error": max_abs_error,
            "correctness_passed": correctness_passed,
        },
        "note": (
            "T1 stage replication simulation: validates data-parallel assignment, "
            "replica output parity, and simulated throughput gain. Not real added-node evidence."
        ),
    }


def validate_stage_replication_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    _non_empty_string(data.get("plan_id"), "plan_id", errors)
    _non_empty_string(data.get("simulation_method"), "simulation_method", errors)
    config = data.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
        config = {}
    _non_negative_int(config.get("bottleneck_stage_index"), "config.bottleneck_stage_index", errors)
    baseline_replica_ids = _string_list(config.get("baseline_replica_ids"), "config.baseline_replica_ids", errors)
    replicated_replica_ids = _string_list(config.get("replicated_replica_ids"), "config.replicated_replica_ids", errors)
    token_counts = _positive_int_list(config.get("microbatch_token_counts"), "config.microbatch_token_counts", errors)
    _positive_number(config.get("baseline_stage_time_s_per_token"), "config.baseline_stage_time_s_per_token", errors)
    _positive_number(config.get("replicated_stage_time_s_per_token"), "config.replicated_stage_time_s_per_token", errors)
    _non_negative_number(config.get("transfer_overhead_s"), "config.transfer_overhead_s", errors)
    speedup_floor = _positive_number(config.get("speedup_floor"), "config.speedup_floor", errors)
    tolerance = _non_negative_number(config.get("tolerance"), "config.tolerance", errors)
    if baseline_replica_ids is not None and len(baseline_replica_ids) != 1:
        errors.append("config.baseline_replica_ids must contain exactly one replica")
    if replicated_replica_ids is not None and len(set(replicated_replica_ids)) != len(replicated_replica_ids):
        errors.append("config.replicated_replica_ids must be unique")
    if baseline_replica_ids is not None and replicated_replica_ids is not None:
        if not set(baseline_replica_ids).issubset(set(replicated_replica_ids)):
            errors.append("baseline replica must appear in replicated_replica_ids")
        if len(replicated_replica_ids) < 2:
            errors.append("replicated_replica_ids must contain at least two replicas")
    assignments = data.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        errors.append("assignments must be a non-empty list")
        assignments = []
    if token_counts is not None and len(assignments) != len(token_counts):
        errors.append("assignments length must equal microbatch_token_counts length")
    used_replicas: set[str] = set()
    for index, assignment in enumerate(assignments):
        field = f"assignments[{index}]"
        if not isinstance(assignment, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(assignment.get("microbatch_id"), f"{field}.microbatch_id", errors)
        microbatch_index = _non_negative_int(assignment.get("microbatch_index"), f"{field}.microbatch_index", errors)
        token_count = _positive_int(assignment.get("token_count"), f"{field}.token_count", errors)
        replica_id = _non_empty_string(assignment.get("replica_id"), f"{field}.replica_id", errors)
        start = _non_negative_number(assignment.get("start_s"), f"{field}.start_s", errors)
        end = _positive_number(assignment.get("end_s"), f"{field}.end_s", errors)
        output_checksum = _non_negative_number(abs(assignment.get("output_checksum", 0.0)) if isinstance(assignment.get("output_checksum"), (int, float)) else assignment.get("output_checksum"), f"{field}.output_checksum", errors)
        reference_checksum = _non_negative_number(abs(assignment.get("reference_checksum", 0.0)) if isinstance(assignment.get("reference_checksum"), (int, float)) else assignment.get("reference_checksum"), f"{field}.reference_checksum", errors)
        max_abs_error = _non_negative_number(assignment.get("max_abs_error"), f"{field}.max_abs_error", errors)
        if microbatch_index is not None and microbatch_index != index:
            errors.append(f"{field}.microbatch_index must equal assignment index")
        if token_counts is not None and token_count is not None and index < len(token_counts) and token_count != token_counts[index]:
            errors.append(f"{field}.token_count must match config.microbatch_token_counts")
        if replica_id is not None:
            used_replicas.add(replica_id)
            if replicated_replica_ids is not None and replica_id not in replicated_replica_ids:
                errors.append(f"{field}.replica_id must be in config.replicated_replica_ids")
        if start is not None and end is not None and end <= start:
            errors.append(f"{field}.end_s must be greater than start_s")
        if output_checksum is not None and reference_checksum is not None and max_abs_error is not None:
            if tolerance is not None and max_abs_error > tolerance:
                errors.append(f"{field}.max_abs_error exceeds tolerance")
    if replicated_replica_ids is not None and used_replicas != set(replicated_replica_ids):
        errors.append("assignments must use every replicated replica")
    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []
    event_kinds = {event.get("kind") for event in events if isinstance(event, dict)}
    missing = REQUIRED_EVENT_KINDS - event_kinds
    if missing:
        errors.append(f"events missing required kinds: {sorted(missing)}")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"events[{index}] must be an object")
            continue
        _non_empty_string(event.get("kind"), f"events[{index}].kind", errors)
        _non_negative_number(event.get("timestamp_s"), f"events[{index}].timestamp_s", errors)
        if event.get("plan_id") != data.get("plan_id"):
            errors.append(f"events[{index}].plan_id must match plan_id")
    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    baseline_makespan = _positive_number(result.get("baseline_makespan_s"), "result.baseline_makespan_s", errors)
    replicated_makespan = _positive_number(result.get("replicated_makespan_s"), "result.replicated_makespan_s", errors)
    speedup = _positive_number(result.get("speedup"), "result.speedup", errors)
    _positive_number(result.get("baseline_tokens_s"), "result.baseline_tokens_s", errors)
    _positive_number(result.get("replicated_tokens_s"), "result.replicated_tokens_s", errors)
    total_tokens = _positive_int(result.get("total_tokens"), "result.total_tokens", errors)
    _non_negative_number(result.get("max_abs_error"), "result.max_abs_error", errors)
    if baseline_makespan is not None and replicated_makespan is not None and speedup is not None:
        expected = baseline_makespan / replicated_makespan
        if abs(expected - speedup) > 1e-6:
            errors.append("result.speedup must equal baseline_makespan_s / replicated_makespan_s")
        if replicated_makespan >= baseline_makespan:
            errors.append("result.replicated_makespan_s must be lower than baseline_makespan_s")
    if speedup is not None and speedup_floor is not None and speedup < speedup_floor:
        errors.append("result.speedup must be >= config.speedup_floor")
    if result.get("speedup_passed") is not True:
        errors.append("result.speedup_passed must be true")
    if result.get("outputs_match_reference") is not True:
        errors.append("result.outputs_match_reference must be true")
    if result.get("correctness_passed") is not True:
        errors.append("result.correctness_passed must be true")
    if token_counts is not None and total_tokens is not None and total_tokens != sum(token_counts):
        errors.append("result.total_tokens must equal sum(config.microbatch_token_counts)")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must equal len(events)")
    if replicated_replica_ids is not None and summary.get("replica_count") != len(replicated_replica_ids):
        errors.append("summary.replica_count must match config.replicated_replica_ids")
    if token_counts is not None and summary.get("microbatch_count") != len(token_counts):
        errors.append("summary.microbatch_count must match microbatch count")
    if speedup is not None and abs(float(summary.get("speedup", -1.0)) - speedup) > 1e-6:
        errors.append("summary.speedup must match result.speedup")
    if summary.get("speedup_passed") is not True:
        errors.append("summary.speedup_passed must be true")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")
    warnings.append("stage replication is simulation evidence, not real added-node scaling evidence")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "replica_count": summary.get("replica_count"),
            "microbatch_count": summary.get("microbatch_count"),
            "baseline_makespan_s": result.get("baseline_makespan_s"),
            "replicated_makespan_s": result.get("replicated_makespan_s"),
            "speedup": result.get("speedup"),
            "max_abs_error": result.get("max_abs_error"),
            "correctness_passed": result.get("correctness_passed") is True,
        },
    }


def validate_stage_replication(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid stage replication artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["stage replication artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_stage_replication_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
