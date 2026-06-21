from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_EVENT_KINDS = (
    "enqueue",
    "microbatch_start",
    "stage_start",
    "stage_end",
    "complete",
)
TERMINAL_EVENT_KINDS = {"complete", "cancel", "reject"}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _normalize_scheduler_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} must be an object")
        prompt_len = int(request.get("prompt_len", request.get("prompt_tokens", 0)))
        gen_len = int(request.get("gen_len", request.get("max_new_tokens", 0)))
        if prompt_len < 0 or gen_len < 0:
            raise ValueError(f"request {index} has negative token counts")
        request_id = str(
            request.get("id") or request.get("request_id") or f"req-{index}"
        )
        normalized.append(
            {
                "request_id": request_id,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "arrival_s": float(request.get("arrival_s", 0.0)),
                "cancel": bool(request.get("cancel", False)),
            }
        )
    return normalized


def load_scheduler_requests(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    requests = data.get("requests") if isinstance(data, dict) else data
    if not isinstance(requests, list):
        raise ValueError("request trace must be a list or object with a requests list")
    return _normalize_scheduler_requests(requests)


def _stage_times(plan: dict[str, Any]) -> list[float]:
    stages = plan.get("stages") if isinstance(plan.get("stages"), list) else []
    predicted = plan.get("predicted") if isinstance(plan.get("predicted"), dict) else {}
    raw_times = predicted.get("stage_effective_times_s")
    if isinstance(raw_times, list) and raw_times:
        times = [
            float(value)
            for value in raw_times
            if _numeric(value) and float(value) > 0
        ]
        if times:
            if stages and len(times) < len(stages):
                times.extend([times[-1]] * (len(stages) - len(times)))
            return times[: len(stages) or len(times)]
    throughput = float(predicted.get("throughput_tok_s", 1.0) or 1.0)
    stage_count = max(1, len(stages))
    return [1.0 / max(throughput, 1e-9) / stage_count for _ in range(stage_count)]


def _event(
    events: list[dict[str, Any]],
    *,
    kind: str,
    timestamp_s: float,
    plan_id: str,
    queue_depth: int | None = None,
    inflight_count: int | None = None,
    **extra: Any,
) -> None:
    item: dict[str, Any] = {
        "kind": kind,
        "timestamp_s": round(timestamp_s, 9),
        "plan_id": plan_id,
    }
    if queue_depth is not None:
        item["queue_depth"] = queue_depth
    if inflight_count is not None:
        item["inflight_count"] = inflight_count
    item.update(extra)
    events.append(item)


def simulate_scheduler(
    plan: dict[str, Any],
    requests: list[dict[str, Any]],
    *,
    plan_id: str = "plan-simulated-t1",
    max_queue_depth: int = 4,
    max_inflight: int = 4,
    microbatch_size: int = 2,
) -> dict[str, Any]:
    """Run a deterministic T1 scheduler simulation without workers or hardware."""

    _positive_int("max_queue_depth", max_queue_depth)
    _positive_int("max_inflight", max_inflight)
    _positive_int("microbatch_size", microbatch_size)
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    if plan.get("feasible") is not True or not isinstance(plan.get("predicted"), dict):
        raise ValueError("scheduler simulation requires a feasible plan with predictions")
    if not plan_id:
        raise ValueError("plan_id must be non-empty")

    normalized_requests = _normalize_scheduler_requests(requests)
    pending = sorted(
        (dict(request) for request in normalized_requests),
        key=lambda item: item.get("arrival_s", 0.0),
    )
    queue: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    stage_times = _stage_times(plan)
    stages = plan.get("stages") if isinstance(plan.get("stages"), list) else []
    stage_count = max(len(stage_times), len(stages), 1)
    batch_size = min(microbatch_size, max_inflight)
    clock = 0.0
    batch_index = 0
    completed_count = 0
    cancelled_count = 0
    max_observed_queue_depth = 0
    max_observed_inflight = 0

    while pending or queue:
        while pending and len(queue) < max_queue_depth:
            request = pending.pop(0)
            clock = max(clock, float(request.get("arrival_s", 0.0)))
            queue.append(request)
            max_observed_queue_depth = max(max_observed_queue_depth, len(queue))
            _event(
                events,
                kind="enqueue",
                timestamp_s=clock,
                plan_id=plan_id,
                request_id=request["request_id"],
                queue_depth=len(queue),
            )
        if pending and len(queue) >= max_queue_depth:
            _event(
                events,
                kind="backpressure",
                timestamp_s=clock,
                plan_id=plan_id,
                queue_depth=len(queue),
                pending_request_count=len(pending),
                max_queue_depth=max_queue_depth,
            )
        if not queue:
            continue

        batch = queue[:batch_size]
        queue = queue[batch_size:]
        batch_id = f"mb-{batch_index}"
        batch_index += 1
        request_ids = [str(request["request_id"]) for request in batch]
        max_observed_inflight = max(max_observed_inflight, len(batch))
        _event(
            events,
            kind="microbatch_start",
            timestamp_s=clock,
            plan_id=plan_id,
            batch_id=batch_id,
            request_ids=request_ids,
            queue_depth=len(queue),
            inflight_count=len(batch),
        )

        token_scale = max(1, max(int(request.get("gen_len", 0)) for request in batch))
        for stage_index in range(stage_count):
            stage_time = (
                stage_times[stage_index]
                if stage_index < len(stage_times)
                else stage_times[-1]
            )
            _event(
                events,
                kind="stage_start",
                timestamp_s=clock,
                plan_id=plan_id,
                batch_id=batch_id,
                stage_index=stage_index,
                request_ids=request_ids,
                inflight_count=len(batch),
            )
            elapsed = stage_time * token_scale
            clock += elapsed
            _event(
                events,
                kind="stage_end",
                timestamp_s=clock,
                plan_id=plan_id,
                batch_id=batch_id,
                stage_index=stage_index,
                request_ids=request_ids,
                elapsed_s=round(elapsed, 9),
                inflight_count=len(batch),
            )

        for request in batch:
            if request.get("cancel"):
                cancelled_count += 1
                _event(
                    events,
                    kind="cancel",
                    timestamp_s=clock,
                    plan_id=plan_id,
                    request_id=request["request_id"],
                    cleanup="scheduler_state_released",
                    inflight_count=0,
                )
            else:
                completed_count += 1
                _event(
                    events,
                    kind="complete",
                    timestamp_s=clock,
                    plan_id=plan_id,
                    request_id=request["request_id"],
                    generated_tokens=int(request.get("gen_len", 0)),
                    inflight_count=0,
                )

    backpressure_count = sum(1 for event in events if event.get("kind") == "backpressure")
    return {
        "version": 1,
        "record_kind": "scheduler-simulation-contract",
        "mode": "t1-simulation",
        "plan_id": plan_id,
        "max_queue_depth": max_queue_depth,
        "max_inflight": max_inflight,
        "microbatch_size": microbatch_size,
        "stage_count": stage_count,
        "request_count": len(normalized_requests),
        "events": events,
        "summary": {
            "request_count": len(normalized_requests),
            "completed_count": completed_count,
            "cancelled_count": cancelled_count,
            "backpressure_count": backpressure_count,
            "microbatch_count": batch_index,
            "max_observed_queue_depth": max_observed_queue_depth,
            "max_observed_inflight": max_observed_inflight,
            "makespan_s": round(clock, 9),
        },
        "note": (
            "T1 model-free scheduler simulation; validates admission, microbatch, "
            "bounded queue, and per-stage event contracts without real workers."
        ),
    }


def simulate_scheduler_from_paths(
    plan_path: str | Path,
    requests_path: str | Path,
    *,
    plan_id: str = "plan-simulated-t1",
    max_queue_depth: int = 4,
    max_inflight: int = 4,
    microbatch_size: int = 2,
) -> dict[str, Any]:
    plan = read_json(plan_path)
    requests = load_scheduler_requests(requests_path)
    return simulate_scheduler(
        plan,
        requests,
        plan_id=plan_id,
        max_queue_depth=max_queue_depth,
        max_inflight=max_inflight,
        microbatch_size=microbatch_size,
    )


def validate_scheduler_contract(path_or_data: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_data, (str, Path)):
        path = Path(path_or_data)
        if path.is_dir():
            path = path / "fixture.json"
        try:
            data = read_json(path)
        except Exception as exc:  # noqa: BLE001 - validator reports fixture parse errors.
            return {
                "ok": False,
                "errors": [f"invalid scheduler contract: {exc}"],
                "warnings": [],
            }
    else:
        data = path_or_data

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["scheduler contract must be a JSON object"],
            "warnings": [],
        }
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != "scheduler-simulation-contract":
        errors.append("record_kind must be scheduler-simulation-contract")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")
    plan_id = data.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        errors.append("plan_id must be a non-empty string")
        plan_id = ""

    for field in ("max_queue_depth", "max_inflight", "microbatch_size", "request_count"):
        value = data.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{field} must be a non-negative integer")
    max_queue_depth = int(data.get("max_queue_depth", 0) or 0)
    max_inflight = int(data.get("max_inflight", 0) or 0)
    microbatch_size = int(data.get("microbatch_size", 0) or 0)

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []

    kinds = {event.get("kind") for event in events if isinstance(event, dict)}
    missing = [kind for kind in REQUIRED_EVENT_KINDS if kind not in kinds]
    if missing:
        errors.append("events missing required scheduler events: " + ", ".join(missing))

    enqueued: set[str] = set()
    terminal: set[str] = set()
    stage_starts: set[tuple[str, int]] = set()
    stage_ends: set[tuple[str, int]] = set()
    observed_max_queue = 0
    observed_max_inflight = 0
    backpressure_count = 0
    completed_count = 0
    cancelled_count = 0

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {index} must be an object")
            continue
        kind = event.get("kind")
        if event.get("plan_id") != plan_id:
            errors.append(f"event {index} plan_id must match root plan_id")
        if not _numeric(event.get("timestamp_s")) or float(event.get("timestamp_s")) < 0:
            errors.append(f"event {index} timestamp_s must be non-negative numeric")
        queue_depth = event.get("queue_depth")
        if queue_depth is not None:
            if (
                isinstance(queue_depth, bool)
                or not isinstance(queue_depth, int)
                or queue_depth < 0
            ):
                errors.append(f"event {index} queue_depth must be non-negative integer")
            else:
                observed_max_queue = max(observed_max_queue, queue_depth)
                if max_queue_depth and queue_depth > max_queue_depth:
                    errors.append(f"event {index} queue_depth exceeds max_queue_depth")
        inflight = event.get("inflight_count")
        if inflight is not None:
            if (
                isinstance(inflight, bool)
                or not isinstance(inflight, int)
                or inflight < 0
            ):
                errors.append(f"event {index} inflight_count must be non-negative integer")
            else:
                observed_max_inflight = max(observed_max_inflight, inflight)
                if max_inflight and inflight > max_inflight:
                    errors.append(f"event {index} inflight_count exceeds max_inflight")
        if kind == "enqueue":
            request_id = event.get("request_id")
            if isinstance(request_id, str) and request_id:
                enqueued.add(request_id)
            else:
                errors.append(f"event {index} enqueue missing request_id")
        elif kind == "microbatch_start":
            request_ids = event.get("request_ids")
            if not isinstance(request_ids, list) or not request_ids:
                errors.append(f"event {index} microbatch_start missing request_ids")
            else:
                if len(request_ids) > microbatch_size:
                    errors.append(f"event {index} request_ids exceeds microbatch_size")
                if len(request_ids) > max_inflight:
                    errors.append(f"event {index} request_ids exceeds max_inflight")
        elif kind == "stage_start":
            stage_index = event.get("stage_index")
            if (
                isinstance(stage_index, bool)
                or not isinstance(stage_index, int)
                or stage_index < 0
            ):
                errors.append(f"event {index} stage_index must be non-negative integer")
            else:
                key = (str(event.get("batch_id")), stage_index)
                stage_starts.add(key)
        elif kind == "stage_end":
            stage_index = event.get("stage_index")
            if (
                isinstance(stage_index, bool)
                or not isinstance(stage_index, int)
                or stage_index < 0
            ):
                errors.append(f"event {index} stage_index must be non-negative integer")
                key = (str(event.get("batch_id")), -1)
            else:
                key = (str(event.get("batch_id")), stage_index)
            stage_ends.add(key)
            if not _numeric(event.get("elapsed_s")) or float(event.get("elapsed_s")) <= 0:
                errors.append(f"event {index} stage_end elapsed_s must be positive numeric")
        elif kind in TERMINAL_EVENT_KINDS:
            request_id = event.get("request_id")
            if isinstance(request_id, str) and request_id:
                terminal.add(request_id)
            else:
                errors.append(f"event {index} terminal event missing request_id")
            if kind == "complete":
                completed_count += 1
            if kind == "cancel":
                cancelled_count += 1
                if event.get("cleanup") != "scheduler_state_released":
                    errors.append(f"event {index} cancel must record scheduler cleanup")
        elif kind == "backpressure":
            backpressure_count += 1
            if queue_depth != max_queue_depth:
                warnings.append(f"event {index} backpressure emitted before queue reached limit")
        else:
            errors.append(f"event {index} has unknown kind: {kind}")

    unclosed = sorted(enqueued - terminal)
    if unclosed:
        errors.append("enqueued requests missing terminal event: " + ", ".join(unclosed))
    missing_stage_end = sorted(stage_starts - stage_ends)
    if missing_stage_end:
        errors.append("stage_start without stage_end: " + str(missing_stage_end))
    missing_stage_start = sorted(stage_ends - stage_starts)
    if missing_stage_start:
        errors.append("stage_end without stage_start: " + str(missing_stage_start))

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    if summary.get("completed_count") != completed_count:
        errors.append("summary.completed_count does not match complete events")
    if summary.get("cancelled_count") != cancelled_count:
        errors.append("summary.cancelled_count does not match cancel events")
    if summary.get("backpressure_count") != backpressure_count:
        errors.append("summary.backpressure_count does not match backpressure events")
    if summary.get("max_observed_queue_depth") != observed_max_queue:
        errors.append("summary.max_observed_queue_depth does not match events")
    if summary.get("max_observed_inflight") != observed_max_inflight:
        errors.append("summary.max_observed_inflight does not match events")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "event_count": len(events),
            "request_count": data.get("request_count"),
            "completed_count": completed_count,
            "cancelled_count": cancelled_count,
            "backpressure_count": backpressure_count,
            "max_observed_queue_depth": observed_max_queue,
            "max_observed_inflight": observed_max_inflight,
        },
    }
