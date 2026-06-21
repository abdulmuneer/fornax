from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_EVENT_KINDS = (
    "request_admitted",
    "backpressure",
    "fairness_yield",
    "microbatch_formed",
    "stage_compute_start",
    "stage_compute_end",
    "activation_transfer_start",
    "activation_transfer_end",
    "token_emit",
    "request_complete",
    "bubble_sample",
)
KNOWN_EVENT_KINDS = set(REQUIRED_EVENT_KINDS)


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


def _request_ids(value: Any, field: str, errors: list[str]) -> list[str] | None:
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
            "throughput_tok_s": 18.0,
            "per_request_latency_s": 0.30,
            "bubble_fraction": 0.18,
            "stage_effective_times_s": [0.010, 0.015],
        },
    }


def _default_requests() -> list[dict[str, Any]]:
    return [
        {"id": "cb-r0", "prompt_len": 8, "gen_len": 4, "arrival_s": 0.000},
        {"id": "cb-r1", "prompt_len": 8, "gen_len": 3, "arrival_s": 0.000},
        {"id": "cb-r2", "prompt_len": 8, "gen_len": 3, "arrival_s": 0.000},
        {"id": "cb-r3", "prompt_len": 8, "gen_len": 2, "arrival_s": 0.000},
        {"id": "cb-r4", "prompt_len": 8, "gen_len": 2, "arrival_s": 0.000},
        {"id": "cb-r5", "prompt_len": 8, "gen_len": 1, "arrival_s": 0.000},
    ]


def _normalize_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} must be an object")
        prompt_len = int(request.get("prompt_len", request.get("prompt_tokens", 0)))
        gen_len = int(request.get("gen_len", request.get("max_new_tokens", 0)))
        if prompt_len < 0 or gen_len < 0:
            raise ValueError(f"request {index} has negative token counts")
        normalized.append(
            {
                "request_id": str(
                    request.get("id") or request.get("request_id") or f"req-{index}"
                ),
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "arrival_s": float(request.get("arrival_s", 0.0)),
            }
        )
    return sorted(normalized, key=lambda item: (item["arrival_s"], item["request_id"]))


def _stage_times(plan: dict[str, Any]) -> list[float]:
    predicted = plan.get("predicted") if isinstance(plan.get("predicted"), dict) else {}
    stages = plan.get("stages") if isinstance(plan.get("stages"), list) else []
    raw = predicted.get("stage_effective_times_s")
    if isinstance(raw, list) and raw:
        values = [
            float(value)
            for value in raw
            if not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) > 0
        ]
        if values:
            while len(values) < len(stages):
                values.append(values[-1])
            return values[: len(stages) or len(values)]
    return [0.010 for _ in range(max(2, len(stages)))]


def _event(
    events: list[dict[str, Any]],
    *,
    kind: str,
    timestamp_s: float,
    plan_id: str,
    **extra: Any,
) -> None:
    item: dict[str, Any] = {
        "kind": kind,
        "timestamp_s": round(timestamp_s, 9),
        "plan_id": plan_id,
    }
    item.update(extra)
    events.append(item)


def simulate_continuous_batching(
    plan: dict[str, Any] | None = None,
    requests: list[dict[str, Any]] | None = None,
    *,
    plan_id: str = "continuous-batching-plan",
    max_queue_depth: int = 4,
    max_inflight: int = 4,
    microbatch_size: int = 2,
    fairness_window_s: float = 0.050,
    transfer_s: float = 0.002,
) -> dict[str, Any]:
    """Simulate continuous batching and 1F1B-style stage overlap."""

    errors: list[str] = []
    _positive_int(max_queue_depth, "max_queue_depth", errors)
    _positive_int(max_inflight, "max_inflight", errors)
    _positive_int(microbatch_size, "microbatch_size", errors)
    _positive_number(fairness_window_s, "fairness_window_s", errors)
    _positive_number(transfer_s, "transfer_s", errors)
    if errors:
        raise ValueError("; ".join(errors))
    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    plan = _default_plan() if plan is None else plan
    if not isinstance(plan, dict) or plan.get("feasible") is not True:
        raise ValueError("continuous batching simulation requires a feasible plan")
    normalized = _normalize_requests(_default_requests() if requests is None else requests)
    if not normalized:
        raise ValueError("requests must be non-empty")

    stage_times = _stage_times(plan)
    stage_count = max(2, len(stage_times))
    events: list[dict[str, Any]] = []
    pending = list(normalized)
    queue: list[dict[str, Any]] = []
    admitted_order: list[str] = []
    formed_order: list[str] = []
    microbatches: list[dict[str, Any]] = []
    now = 0.0
    max_seen_queue = 0
    backpressure_count = 0
    fairness_yield_count = 0
    wait_times: list[float] = []

    while pending or queue:
        while pending and len(queue) < max_queue_depth and pending[0]["arrival_s"] <= now:
            request = pending.pop(0)
            queue.append(request)
            admitted_order.append(request["request_id"])
            max_seen_queue = max(max_seen_queue, len(queue))
            _event(
                events,
                kind="request_admitted",
                timestamp_s=now,
                plan_id=plan_id,
                request_id=request["request_id"],
                arrival_s=request["arrival_s"],
                queue_depth=len(queue),
            )
        if pending and len(queue) >= max_queue_depth and pending[0]["arrival_s"] <= now:
            backpressure_count += 1
            _event(
                events,
                kind="backpressure",
                timestamp_s=now,
                plan_id=plan_id,
                queue_depth=len(queue),
                max_queue_depth=max_queue_depth,
                pending_request_count=len(pending),
            )
        if len(queue) >= microbatch_size or (queue and not pending):
            batch = queue[:microbatch_size]
            queue = queue[microbatch_size:]
            batch_id = f"cb-mb-{len(microbatches)}"
            request_ids = [request["request_id"] for request in batch]
            formed_order.extend(request_ids)
            waits = [max(0.0, now - float(request["arrival_s"])) for request in batch]
            wait_times.extend(waits)
            oldest_wait = max(waits) if waits else 0.0
            fairness_yield_count += 1
            _event(
                events,
                kind="fairness_yield",
                timestamp_s=now,
                plan_id=plan_id,
                batch_id=batch_id,
                policy="fifo",
                request_ids=request_ids,
                oldest_wait_s=round(oldest_wait, 9),
                fairness_window_s=fairness_window_s,
            )
            _event(
                events,
                kind="microbatch_formed",
                timestamp_s=now,
                plan_id=plan_id,
                batch_id=batch_id,
                request_ids=request_ids,
                microbatch_size=len(batch),
                queue_depth=len(queue),
                inflight_count=min(len(microbatches) + 1, max_inflight),
                wait_s=[round(wait, 9) for wait in waits],
            )
            microbatches.append({"batch_id": batch_id, "requests": batch})
            continue
        if pending:
            now = max(now, pending[0]["arrival_s"])

    stage_available = [0.0 for _ in range(stage_count)]
    stage_busy = [0.0 for _ in range(stage_count)]
    intervals: list[tuple[str, int, float, float]] = []
    makespan = 0.0
    max_observed_inflight = 0
    for batch_index, microbatch in enumerate(microbatches):
        batch_id = str(microbatch["batch_id"])
        batch_requests = microbatch["requests"]
        request_ids = [request["request_id"] for request in batch_requests]
        token_scale = max(1, max(int(request.get("gen_len", 0)) for request in batch_requests))
        ready_time = 0.0
        for stage_index in range(stage_count):
            start = max(stage_available[stage_index], ready_time)
            idle_s = max(0.0, start - stage_available[stage_index])
            _event(
                events,
                kind="bubble_sample",
                timestamp_s=start,
                plan_id=plan_id,
                batch_id=batch_id,
                stage_index=stage_index,
                idle_s=round(idle_s, 9),
            )
            max_observed_inflight = max(
                max_observed_inflight,
                min(batch_index + 1, max_inflight),
            )
            _event(
                events,
                kind="stage_compute_start",
                timestamp_s=start,
                plan_id=plan_id,
                batch_id=batch_id,
                stage_index=stage_index,
                request_ids=request_ids,
                inflight_count=min(batch_index + 1, max_inflight),
            )
            elapsed = stage_times[stage_index] * token_scale
            end = start + elapsed
            stage_busy[stage_index] += elapsed
            intervals.append((batch_id, stage_index, start, end))
            _event(
                events,
                kind="stage_compute_end",
                timestamp_s=end,
                plan_id=plan_id,
                batch_id=batch_id,
                stage_index=stage_index,
                request_ids=request_ids,
                elapsed_s=round(elapsed, 9),
            )
            stage_available[stage_index] = end
            if stage_index < stage_count - 1:
                transfer_start = end
                transfer_end = transfer_start + transfer_s
                _event(
                    events,
                    kind="activation_transfer_start",
                    timestamp_s=transfer_start,
                    plan_id=plan_id,
                    batch_id=batch_id,
                    source_stage=stage_index,
                    target_stage=stage_index + 1,
                    payload_id=f"{batch_id}-activation-{stage_index}",
                )
                _event(
                    events,
                    kind="activation_transfer_end",
                    timestamp_s=transfer_end,
                    plan_id=plan_id,
                    batch_id=batch_id,
                    source_stage=stage_index,
                    target_stage=stage_index + 1,
                    payload_id=f"{batch_id}-activation-{stage_index}",
                    elapsed_s=round(transfer_s, 9),
                )
                ready_time = transfer_end
            else:
                ready_time = end
        makespan = max(makespan, ready_time)
        for request in batch_requests:
            _event(
                events,
                kind="token_emit",
                timestamp_s=ready_time,
                plan_id=plan_id,
                batch_id=batch_id,
                request_id=request["request_id"],
                token_count=max(1, int(request.get("gen_len", 0))),
            )
            _event(
                events,
                kind="request_complete",
                timestamp_s=ready_time,
                plan_id=plan_id,
                batch_id=batch_id,
                request_id=request["request_id"],
                generated_tokens=int(request.get("gen_len", 0)),
            )

    overlap_observed = _has_stage_overlap(intervals)
    total_busy = sum(stage_busy)
    total_capacity = max(makespan, 1e-9) * stage_count
    bubble_fraction = max(0.0, min(1.0, 1.0 - (total_busy / total_capacity)))
    max_wait = max(wait_times) if wait_times else 0.0
    p95_wait = sorted(wait_times)[max(0, int(len(wait_times) * 0.95) - 1)] if wait_times else 0.0
    return {
        "version": 1,
        "record_kind": "continuous-batching-simulation-contract",
        "mode": "t1-simulation",
        "plan_id": plan_id,
        "max_queue_depth": max_queue_depth,
        "max_inflight": max_inflight,
        "microbatch_size": microbatch_size,
        "fairness_window_s": fairness_window_s,
        "stage_count": stage_count,
        "request_count": len(normalized),
        "events": events,
        "summary": {
            "event_count": len(events),
            "request_count": len(normalized),
            "completed_count": len(normalized),
            "microbatch_count": len(microbatches),
            "backpressure_count": backpressure_count,
            "fairness_yield_count": fairness_yield_count,
            "max_observed_queue_depth": max_seen_queue,
            "max_observed_inflight": max_observed_inflight,
            "overlap_observed": overlap_observed,
            "bubble_fraction": round(bubble_fraction, 9),
            "max_wait_s": round(max_wait, 9),
            "p95_wait_s": round(p95_wait, 9),
            "makespan_s": round(makespan, 9),
            "formed_request_order": formed_order,
            "admitted_request_order": admitted_order,
        },
        "note": (
            "T1 continuous-batching simulation; validates admission, FIFO fairness, "
            "1F1B-style pipeline overlap, activation transfer, and bubble telemetry "
            "without real worker execution."
        ),
    }


def _has_stage_overlap(intervals: list[tuple[str, int, float, float]]) -> bool:
    for left_index, left in enumerate(intervals):
        left_batch, left_stage, left_start, left_end = left
        for right in intervals[left_index + 1 :]:
            right_batch, right_stage, right_start, right_end = right
            if left_batch == right_batch or left_stage == right_stage:
                continue
            if left_start < right_end and right_start < left_end:
                return True
    return False


def validate_continuous_batching_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != "continuous-batching-simulation-contract":
        errors.append("record_kind must be continuous-batching-simulation-contract")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors) or ""
    max_queue_depth = _positive_int(
        data.get("max_queue_depth"), "max_queue_depth", errors
    ) or 0
    max_inflight = _positive_int(data.get("max_inflight"), "max_inflight", errors) or 0
    microbatch_size = _positive_int(
        data.get("microbatch_size"), "microbatch_size", errors
    ) or 0
    fairness_window_s = _positive_number(
        data.get("fairness_window_s"), "fairness_window_s", errors
    ) or 0.0
    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []

    seen: set[str] = set()
    admitted_order: list[str] = []
    formed_order: list[str] = []
    completed: set[str] = set()
    token_emitted: set[str] = set()
    stage_starts: dict[tuple[str, int], dict[str, Any]] = {}
    stage_intervals: list[tuple[str, int, float, float]] = []
    transfer_starts: set[tuple[str, int, int, str]] = set()
    transfer_ends: set[tuple[str, int, int, str]] = set()
    backpressure_count = 0
    fairness_yield_count = 0
    microbatch_count = 0
    max_seen_queue = 0
    max_seen_inflight = 0
    max_wait = 0.0
    bubble_samples = 0

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
        if event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match root plan_id")
        timestamp = _non_negative_number(
            event.get("timestamp_s"), f"{field}.timestamp_s", errors
        )
        queue_depth = event.get("queue_depth")
        if queue_depth is not None:
            depth = _non_negative_int(queue_depth, f"{field}.queue_depth", errors)
            if depth is not None:
                max_seen_queue = max(max_seen_queue, depth)
                if depth > max_queue_depth:
                    errors.append(f"{field}.queue_depth exceeds max_queue_depth")
        inflight = event.get("inflight_count")
        if inflight is not None:
            count = _positive_int(inflight, f"{field}.inflight_count", errors)
            if count is not None:
                max_seen_inflight = max(max_seen_inflight, count)
                if count > max_inflight:
                    errors.append(f"{field}.inflight_count exceeds max_inflight")

        if kind == "request_admitted":
            request_id = _non_empty_string(event.get("request_id"), f"{field}.request_id", errors)
            if request_id is not None:
                admitted_order.append(request_id)
            _non_negative_number(event.get("arrival_s"), f"{field}.arrival_s", errors)
        elif kind == "backpressure":
            if event.get("queue_depth") != max_queue_depth:
                errors.append(f"{field}.queue_depth must equal max_queue_depth")
            _positive_int(
                event.get("pending_request_count"),
                f"{field}.pending_request_count",
                errors,
            )
            backpressure_count += 1
        elif kind == "fairness_yield":
            request_ids = _request_ids(event.get("request_ids"), f"{field}.request_ids", errors)
            if request_ids is not None:
                for request_id in request_ids:
                    if request_id not in admitted_order:
                        errors.append(f"{field}.request_ids contains unadmitted request")
            oldest_wait = _non_negative_number(
                event.get("oldest_wait_s"), f"{field}.oldest_wait_s", errors
            )
            if oldest_wait is not None:
                max_wait = max(max_wait, oldest_wait)
                if oldest_wait > fairness_window_s:
                    errors.append(f"{field}.oldest_wait_s exceeds fairness_window_s")
            if event.get("policy") != "fifo":
                errors.append(f"{field}.policy must be fifo")
            fairness_yield_count += 1
        elif kind == "microbatch_formed":
            request_ids = _request_ids(event.get("request_ids"), f"{field}.request_ids", errors)
            if request_ids is not None:
                if len(request_ids) > microbatch_size:
                    errors.append(f"{field}.request_ids exceeds microbatch_size")
                formed_order.extend(request_ids)
            wait_s = event.get("wait_s")
            if not isinstance(wait_s, list):
                errors.append(f"{field}.wait_s must be a list")
            else:
                for wait in wait_s:
                    observed = _non_negative_number(wait, f"{field}.wait_s[]", errors)
                    if observed is not None:
                        max_wait = max(max_wait, observed)
                        if observed > fairness_window_s:
                            errors.append(f"{field}.wait_s contains wait above fairness_window_s")
            microbatch_count += 1
        elif kind == "stage_compute_start":
            batch_id = _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            stage_index = _non_negative_int(event.get("stage_index"), f"{field}.stage_index", errors)
            _request_ids(event.get("request_ids"), f"{field}.request_ids", errors)
            if batch_id is not None and stage_index is not None and timestamp is not None:
                stage_starts[(batch_id, stage_index)] = event
        elif kind == "stage_compute_end":
            batch_id = _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
            stage_index = _non_negative_int(event.get("stage_index"), f"{field}.stage_index", errors)
            elapsed = _positive_number(event.get("elapsed_s"), f"{field}.elapsed_s", errors)
            if batch_id is not None and stage_index is not None and timestamp is not None:
                start_event = stage_starts.get((batch_id, stage_index))
                if start_event is None:
                    errors.append(f"{field} has no matching stage_compute_start")
                else:
                    start_time = float(start_event["timestamp_s"])
                    if elapsed is not None and abs((timestamp - start_time) - elapsed) > 1e-6:
                        errors.append(f"{field}.elapsed_s does not match timestamps")
                    stage_intervals.append((batch_id, stage_index, start_time, timestamp))
        elif kind == "activation_transfer_start":
            key = _transfer_key(event, field, errors)
            if key is not None:
                transfer_starts.add(key)
        elif kind == "activation_transfer_end":
            key = _transfer_key(event, field, errors)
            _positive_number(event.get("elapsed_s"), f"{field}.elapsed_s", errors)
            if key is not None:
                transfer_ends.add(key)
        elif kind == "token_emit":
            request_id = _non_empty_string(event.get("request_id"), f"{field}.request_id", errors)
            _positive_int(event.get("token_count"), f"{field}.token_count", errors)
            if request_id is not None:
                token_emitted.add(request_id)
        elif kind == "request_complete":
            request_id = _non_empty_string(event.get("request_id"), f"{field}.request_id", errors)
            _non_negative_number(
                event.get("generated_tokens"), f"{field}.generated_tokens", errors
            )
            if request_id is not None:
                completed.add(request_id)
        elif kind == "bubble_sample":
            _non_negative_int(event.get("stage_index"), f"{field}.stage_index", errors)
            _non_negative_number(event.get("idle_s"), f"{field}.idle_s", errors)
            bubble_samples += 1

    missing = [kind for kind in REQUIRED_EVENT_KINDS if kind not in seen]
    if missing:
        errors.append("events missing required continuous-batching events: " + ", ".join(missing))
    if admitted_order != formed_order:
        errors.append("formed_request_order must preserve FIFO admission order")
    if sorted(completed) != sorted(admitted_order):
        errors.append("all admitted requests must complete")
    if not token_emitted.issuperset(completed):
        errors.append("completed requests must emit tokens")
    if transfer_starts != transfer_ends:
        errors.append("activation transfer start/end pairs must match")
    overlap_observed = _has_stage_overlap(stage_intervals)
    if not overlap_observed:
        errors.append("1F1B overlap was not observed across stages")

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    expected = {
        "event_count": len(events),
        "request_count": len(admitted_order),
        "completed_count": len(completed),
        "microbatch_count": microbatch_count,
        "backpressure_count": backpressure_count,
        "fairness_yield_count": fairness_yield_count,
        "max_observed_queue_depth": max_seen_queue,
        "max_observed_inflight": max_seen_inflight,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            errors.append(f"summary.{field} does not match events")
    if summary.get("overlap_observed") is not True:
        errors.append("summary.overlap_observed must be true")
    bubble_fraction = summary.get("bubble_fraction")
    if (
        isinstance(bubble_fraction, bool)
        or not isinstance(bubble_fraction, (int, float))
        or bubble_fraction < 0
        or bubble_fraction >= 1
    ):
        errors.append("summary.bubble_fraction must be in [0, 1)")
    if summary.get("max_wait_s") != round(max_wait, 9):
        errors.append("summary.max_wait_s does not match events")
    if summary.get("formed_request_order") != formed_order:
        errors.append("summary.formed_request_order does not match events")
    if summary.get("admitted_request_order") != admitted_order:
        errors.append("summary.admitted_request_order does not match events")
    if bubble_samples < microbatch_count:
        errors.append("bubble telemetry must cover the simulated microbatches")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "event_count": len(events),
            "request_count": len(admitted_order),
            "completed_count": len(completed),
            "microbatch_count": microbatch_count,
            "backpressure_count": backpressure_count,
            "fairness_yield_count": fairness_yield_count,
            "max_observed_queue_depth": max_seen_queue,
            "max_observed_inflight": max_seen_inflight,
            "overlap_observed": overlap_observed,
            "max_wait_s": round(max_wait, 9),
            "required_events_seen": sorted(seen & set(REQUIRED_EVENT_KINDS)),
        },
    }


def _transfer_key(
    event: dict[str, Any], field: str, errors: list[str]
) -> tuple[str, int, int, str] | None:
    batch_id = _non_empty_string(event.get("batch_id"), f"{field}.batch_id", errors)
    source = _non_negative_int(event.get("source_stage"), f"{field}.source_stage", errors)
    target = _non_negative_int(event.get("target_stage"), f"{field}.target_stage", errors)
    payload = _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
    if source is not None and target is not None and target != source + 1:
        errors.append(f"{field}.target_stage must be source_stage + 1")
    if batch_id is None or source is None or target is None or payload is None:
        return None
    return (batch_id, source, target, payload)


def validate_continuous_batching(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing continuous batching fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid continuous batching fixture: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["continuous batching fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_continuous_batching_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
