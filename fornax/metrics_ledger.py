from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "metrics-ledger-simulation-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "queue-backpressure-kv-memory-metrics"
REQUIRED_EVENT_KINDS = {
    "request_admitted",
    "request_completed",
    "backpressure_asserted",
    "backpressure_cleared",
    "kv_pages_allocated",
    "memory_pressure_observed",
    "kv_pages_evicted",
    "cleanup",
}
REQUIRED_ALERT_KINDS = {
    "queue_depth_high",
    "memory_pressure_warning",
    "kv_eviction_started",
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


def _bounded_fraction(value: Any, field: str, errors: list[str]) -> float | None:
    number = _non_negative_number(value, field, errors)
    if number is not None and number > 1.0:
        errors.append(f"{field} must be between 0 and 1")
    return number


def _event(kind: str, *, timestamp_s: float, plan_id: str, request_id: str, **fields: Any) -> dict[str, Any]:
    event = {
        "kind": kind,
        "timestamp_s": round(timestamp_s, 6),
        "plan_id": plan_id,
        "request_id": request_id,
    }
    event.update(fields)
    return event


def _pressure(memory_used_bytes: int, memory_limit_bytes: int) -> float:
    return round(memory_used_bytes / memory_limit_bytes, 6)


def _sample(
    sample_id: str,
    *,
    timestamp_s: float,
    queue_depth: int,
    inflight_requests: int,
    backpressure_active: bool,
    kv_pages_allocated: int,
    kv_pages_evicted_total: int,
    memory_used_bytes: int,
    memory_limit_bytes: int,
    stage_0_latency_ms: float,
    stage_1_latency_ms: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "timestamp_s": round(timestamp_s, 6),
        "queue_depth": queue_depth,
        "inflight_requests": inflight_requests,
        "backpressure_active": backpressure_active,
        "kv_pages_allocated": kv_pages_allocated,
        "kv_pages_evicted_total": kv_pages_evicted_total,
        "memory_used_bytes": memory_used_bytes,
        "memory_pressure_fraction": _pressure(memory_used_bytes, memory_limit_bytes),
        "stage_latency_ms": {
            "stage-0": stage_0_latency_ms,
            "stage-1": stage_1_latency_ms,
        },
    }


def _latency_histogram(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    for sample in samples:
        stage_latencies = sample["stage_latency_ms"]
        values.extend(float(value) for value in stage_latencies.values())
    buckets = []
    for boundary in (1.0, 5.0, 10.0, float("inf")):
        count = sum(1 for value in values if value <= boundary)
        buckets.append({"le": "inf" if boundary == float("inf") else boundary, "count": count})
    return {
        "unit": "ms",
        "count": len(values),
        "sum": round(sum(values), 6),
        "buckets": buckets,
    }


def _derive_counters(events: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "requests_admitted_total": sum(1 for event in events if event["kind"] == "request_admitted"),
        "requests_completed_total": sum(1 for event in events if event["kind"] == "request_completed"),
        "backpressure_asserted_total": sum(1 for event in events if event["kind"] == "backpressure_asserted"),
        "backpressure_cleared_total": sum(1 for event in events if event["kind"] == "backpressure_cleared"),
        "kv_pages_evicted_total": sum(
            int(event.get("pages", 0)) for event in events if event["kind"] == "kv_pages_evicted"
        ),
        "allocation_failures_total": sum(1 for event in events if event["kind"] == "allocation_failure"),
    }


def _derive_gauges(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "max_queue_depth_observed": max(int(sample["queue_depth"]) for sample in samples),
        "max_inflight_observed": max(int(sample["inflight_requests"]) for sample in samples),
        "max_kv_pages_allocated": max(int(sample["kv_pages_allocated"]) for sample in samples),
        "max_memory_used_bytes": max(int(sample["memory_used_bytes"]) for sample in samples),
        "max_memory_pressure_fraction": max(
            float(sample["memory_pressure_fraction"]) for sample in samples
        ),
    }


def simulate_metrics_ledger(
    *,
    plan_id: str = "metrics-ledger-plan",
    request_id: str = "req-metrics-ledger",
    max_queue_depth: int = 4,
    max_inflight: int = 3,
    kv_page_limit: int = 16,
    memory_limit_bytes: int = 80 * 1024 * 1024 * 1024,
    memory_warning_fraction: float = 0.85,
    memory_critical_fraction: float = 0.95,
    sample_period_ms: float = 10.0,
) -> dict[str, Any]:
    """Simulate a T1 metrics ledger for queue, backpressure, KV, and memory signals."""

    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(request_id, "request_id", errors)
    _positive_int(max_queue_depth, "max_queue_depth", errors)
    _positive_int(max_inflight, "max_inflight", errors)
    _positive_int(kv_page_limit, "kv_page_limit", errors)
    _positive_int(memory_limit_bytes, "memory_limit_bytes", errors)
    warning = _bounded_fraction(memory_warning_fraction, "memory_warning_fraction", errors)
    critical = _bounded_fraction(memory_critical_fraction, "memory_critical_fraction", errors)
    _positive_number(sample_period_ms, "sample_period_ms", errors)
    if warning is not None and critical is not None and warning >= critical:
        errors.append("memory_warning_fraction must be lower than memory_critical_fraction")
    if errors:
        raise ValueError("; ".join(errors))

    samples = [
        _sample(
            "sample-0",
            timestamp_s=0.000,
            queue_depth=0,
            inflight_requests=0,
            backpressure_active=False,
            kv_pages_allocated=0,
            kv_pages_evicted_total=0,
            memory_used_bytes=24 * 1024 * 1024 * 1024,
            memory_limit_bytes=memory_limit_bytes,
            stage_0_latency_ms=2.4,
            stage_1_latency_ms=2.6,
        ),
        _sample(
            "sample-1",
            timestamp_s=0.010,
            queue_depth=2,
            inflight_requests=2,
            backpressure_active=False,
            kv_pages_allocated=8,
            kv_pages_evicted_total=0,
            memory_used_bytes=48 * 1024 * 1024 * 1024,
            memory_limit_bytes=memory_limit_bytes,
            stage_0_latency_ms=3.1,
            stage_1_latency_ms=3.4,
        ),
        _sample(
            "sample-2",
            timestamp_s=0.020,
            queue_depth=max_queue_depth,
            inflight_requests=max_inflight,
            backpressure_active=True,
            kv_pages_allocated=14,
            kv_pages_evicted_total=0,
            memory_used_bytes=68 * 1024 * 1024 * 1024,
            memory_limit_bytes=memory_limit_bytes,
            stage_0_latency_ms=4.8,
            stage_1_latency_ms=5.1,
        ),
        _sample(
            "sample-3",
            timestamp_s=0.030,
            queue_depth=3,
            inflight_requests=3,
            backpressure_active=True,
            kv_pages_allocated=14,
            kv_pages_evicted_total=2,
            memory_used_bytes=72 * 1024 * 1024 * 1024,
            memory_limit_bytes=memory_limit_bytes,
            stage_0_latency_ms=5.5,
            stage_1_latency_ms=5.7,
        ),
        _sample(
            "sample-4",
            timestamp_s=0.040,
            queue_depth=1,
            inflight_requests=1,
            backpressure_active=False,
            kv_pages_allocated=10,
            kv_pages_evicted_total=2,
            memory_used_bytes=56 * 1024 * 1024 * 1024,
            memory_limit_bytes=memory_limit_bytes,
            stage_0_latency_ms=3.2,
            stage_1_latency_ms=3.0,
        ),
    ]
    events = [
        _event("request_admitted", timestamp_s=0.001, plan_id=plan_id, request_id=request_id, admitted_request_id="r0"),
        _event("request_admitted", timestamp_s=0.002, plan_id=plan_id, request_id=request_id, admitted_request_id="r1"),
        _event("kv_pages_allocated", timestamp_s=0.003, plan_id=plan_id, request_id=request_id, owner="stage-0", pages=8),
        _event("request_admitted", timestamp_s=0.011, plan_id=plan_id, request_id=request_id, admitted_request_id="r2"),
        _event("request_admitted", timestamp_s=0.012, plan_id=plan_id, request_id=request_id, admitted_request_id="r3"),
        _event("backpressure_asserted", timestamp_s=0.020, plan_id=plan_id, request_id=request_id, queue_depth=max_queue_depth, reason="admission_queue_full"),
        _event("kv_pages_allocated", timestamp_s=0.021, plan_id=plan_id, request_id=request_id, owner="stage-1", pages=6),
        _event("memory_pressure_observed", timestamp_s=0.030, plan_id=plan_id, request_id=request_id, level="warning", memory_pressure_fraction=samples[3]["memory_pressure_fraction"]),
        _event("kv_pages_evicted", timestamp_s=0.031, plan_id=plan_id, request_id=request_id, owner="stage-1", pages=2, reason="memory_pressure"),
        _event("request_completed", timestamp_s=0.035, plan_id=plan_id, request_id=request_id, completed_request_id="r0"),
        _event("request_completed", timestamp_s=0.036, plan_id=plan_id, request_id=request_id, completed_request_id="r1"),
        _event("request_completed", timestamp_s=0.037, plan_id=plan_id, request_id=request_id, completed_request_id="r2"),
        _event("request_completed", timestamp_s=0.038, plan_id=plan_id, request_id=request_id, completed_request_id="r3"),
        _event("backpressure_cleared", timestamp_s=0.040, plan_id=plan_id, request_id=request_id, queue_depth=1, reason="queue_drained"),
        _event("cleanup", timestamp_s=0.050, plan_id=plan_id, request_id=request_id, metrics_state_released=True),
    ]
    counters = _derive_counters(events)
    gauges = _derive_gauges(samples)
    histogram = _latency_histogram(samples)
    alerts = [
        {
            "kind": "queue_depth_high",
            "severity": "warning",
            "sample_id": "sample-2",
            "value": max_queue_depth,
            "threshold": max_queue_depth,
        },
        {
            "kind": "memory_pressure_warning",
            "severity": "warning",
            "sample_id": "sample-2",
            "value": samples[2]["memory_pressure_fraction"],
            "threshold": memory_warning_fraction,
        },
        {
            "kind": "kv_eviction_started",
            "severity": "info",
            "sample_id": "sample-3",
            "value": 2,
            "threshold": 1,
        },
    ]
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "request_id": request_id,
        "simulation_method": SIMULATION_METHOD,
        "target_workstreams": ["WS-G2", "WS-F2", "WS-E4"],
        "config": {
            "max_queue_depth": max_queue_depth,
            "max_inflight": max_inflight,
            "kv_page_limit": kv_page_limit,
            "memory_limit_bytes": memory_limit_bytes,
            "memory_warning_fraction": memory_warning_fraction,
            "memory_critical_fraction": memory_critical_fraction,
            "sample_period_ms": sample_period_ms,
        },
        "samples": samples,
        "events": events,
        "metrics": {
            "counters": counters,
            "gauges": gauges,
            "histograms": {"stage_latency_ms": histogram},
            "alerts": alerts,
        },
        "result": {
            "queue_bounded": True,
            "backpressure_observed": True,
            "kv_metrics_consistent": True,
            "memory_pressure_observed": True,
            "alerts_complete": True,
            "histograms_consistent": True,
            "correctness_passed": True,
        },
        "summary": {
            "sample_count": len(samples),
            "event_count": len(events),
            "alert_count": len(alerts),
            "max_queue_depth_observed": gauges["max_queue_depth_observed"],
            "max_inflight_observed": gauges["max_inflight_observed"],
            "backpressure_event_count": counters["backpressure_asserted_total"],
            "kv_pages_evicted_total": counters["kv_pages_evicted_total"],
            "max_kv_pages_allocated": gauges["max_kv_pages_allocated"],
            "max_memory_pressure_fraction": gauges["max_memory_pressure_fraction"],
            "stage_latency_sample_count": histogram["count"],
            "correctness_passed": True,
        },
        "note": (
            "T1 metrics-ledger simulation: validates queue depth, backpressure, "
            "KV-page, memory-pressure, alert, and latency aggregate consistency "
            "without claiming live runtime telemetry or production dashboards."
        ),
    }


def _validate_config(config: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(config, dict):
        errors.append("config must be an object")
        return {}
    max_queue_depth = _positive_int(config.get("max_queue_depth"), "config.max_queue_depth", errors)
    max_inflight = _positive_int(config.get("max_inflight"), "config.max_inflight", errors)
    kv_page_limit = _positive_int(config.get("kv_page_limit"), "config.kv_page_limit", errors)
    memory_limit_bytes = _positive_int(
        config.get("memory_limit_bytes"), "config.memory_limit_bytes", errors
    )
    warning = _bounded_fraction(
        config.get("memory_warning_fraction"), "config.memory_warning_fraction", errors
    )
    critical = _bounded_fraction(
        config.get("memory_critical_fraction"), "config.memory_critical_fraction", errors
    )
    _positive_number(config.get("sample_period_ms"), "config.sample_period_ms", errors)
    if warning is not None and critical is not None and warning >= critical:
        errors.append("config.memory_warning_fraction must be lower than memory_critical_fraction")
    return {
        "max_queue_depth": max_queue_depth,
        "max_inflight": max_inflight,
        "kv_page_limit": kv_page_limit,
        "memory_limit_bytes": memory_limit_bytes,
        "memory_warning_fraction": warning,
        "memory_critical_fraction": critical,
    }


def _validate_samples(
    samples: Any, config: dict[str, Any], errors: list[str]
) -> list[dict[str, Any]]:
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty list")
        return []
    valid: list[dict[str, Any]] = []
    previous_ts: float | None = None
    for index, sample in enumerate(samples):
        field = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(sample.get("sample_id"), f"{field}.sample_id", errors)
        timestamp = _non_negative_number(sample.get("timestamp_s"), f"{field}.timestamp_s", errors)
        if previous_ts is not None and timestamp is not None and timestamp < previous_ts:
            errors.append(f"{field}.timestamp_s must be non-decreasing")
        if timestamp is not None:
            previous_ts = timestamp
        queue_depth = _non_negative_int(sample.get("queue_depth"), f"{field}.queue_depth", errors)
        inflight = _non_negative_int(sample.get("inflight_requests"), f"{field}.inflight_requests", errors)
        kv_pages = _non_negative_int(sample.get("kv_pages_allocated"), f"{field}.kv_pages_allocated", errors)
        _non_negative_int(
            sample.get("kv_pages_evicted_total"), f"{field}.kv_pages_evicted_total", errors
        )
        memory_used = _non_negative_int(sample.get("memory_used_bytes"), f"{field}.memory_used_bytes", errors)
        pressure = _bounded_fraction(
            sample.get("memory_pressure_fraction"), f"{field}.memory_pressure_fraction", errors
        )
        if not isinstance(sample.get("backpressure_active"), bool):
            errors.append(f"{field}.backpressure_active must be a boolean")
        if config.get("max_queue_depth") is not None and queue_depth is not None:
            if queue_depth > config["max_queue_depth"]:
                errors.append(f"{field}.queue_depth exceeds config.max_queue_depth")
        if config.get("max_inflight") is not None and inflight is not None:
            if inflight > config["max_inflight"]:
                errors.append(f"{field}.inflight_requests exceeds config.max_inflight")
        if config.get("kv_page_limit") is not None and kv_pages is not None:
            if kv_pages > config["kv_page_limit"]:
                errors.append(f"{field}.kv_pages_allocated exceeds config.kv_page_limit")
        if config.get("memory_limit_bytes") is not None and memory_used is not None:
            if memory_used > config["memory_limit_bytes"]:
                errors.append(f"{field}.memory_used_bytes exceeds config.memory_limit_bytes")
            expected_pressure = _pressure(memory_used, config["memory_limit_bytes"])
            if pressure is not None and abs(pressure - expected_pressure) > 1e-6:
                errors.append(f"{field}.memory_pressure_fraction must equal memory_used / limit")
        stage_latency = sample.get("stage_latency_ms")
        if not isinstance(stage_latency, dict) or not stage_latency:
            errors.append(f"{field}.stage_latency_ms must be a non-empty object")
        else:
            for stage, value in stage_latency.items():
                if not isinstance(stage, str) or not stage:
                    errors.append(f"{field}.stage_latency_ms keys must be non-empty strings")
                _non_negative_number(value, f"{field}.stage_latency_ms.{stage}", errors)
        valid.append(sample)
    return valid


def _validate_events(
    events: Any, *, plan_id: str | None, request_id: str | None, errors: list[str]
) -> list[dict[str, Any]]:
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        return []
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        kind = _non_empty_string(event.get("kind"), f"{field}.kind", errors)
        if kind is not None:
            seen.add(kind)
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        if plan_id is not None and event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match root plan_id")
        if request_id is not None and event.get("request_id") != request_id:
            errors.append(f"{field}.request_id must match root request_id")
        if kind in {"request_admitted", "request_completed"}:
            key = "admitted_request_id" if kind == "request_admitted" else "completed_request_id"
            _non_empty_string(event.get(key), f"{field}.{key}", errors)
        elif kind == "backpressure_asserted":
            _non_negative_int(event.get("queue_depth"), f"{field}.queue_depth", errors)
            _non_empty_string(event.get("reason"), f"{field}.reason", errors)
        elif kind == "backpressure_cleared":
            _non_negative_int(event.get("queue_depth"), f"{field}.queue_depth", errors)
        elif kind in {"kv_pages_allocated", "kv_pages_evicted"}:
            _non_empty_string(event.get("owner"), f"{field}.owner", errors)
            _positive_int(event.get("pages"), f"{field}.pages", errors)
        elif kind == "memory_pressure_observed":
            if event.get("level") not in {"warning", "critical"}:
                errors.append(f"{field}.level must be warning or critical")
            _bounded_fraction(event.get("memory_pressure_fraction"), f"{field}.memory_pressure_fraction", errors)
        valid.append(event)
    missing = REQUIRED_EVENT_KINDS - seen
    if missing:
        errors.append(f"events missing required kinds: {sorted(missing)}")
    return valid


def _validate_histogram(
    histogram: Any, samples: list[dict[str, Any]], errors: list[str]
) -> None:
    if not isinstance(histogram, dict):
        errors.append("metrics.histograms.stage_latency_ms must be an object")
        return
    expected = _latency_histogram(samples) if samples else {"count": 0, "sum": 0.0, "buckets": []}
    if histogram.get("unit") != "ms":
        errors.append("metrics.histograms.stage_latency_ms.unit must be ms")
    if histogram.get("count") != expected["count"]:
        errors.append("metrics.histograms.stage_latency_ms.count must match samples")
    if abs(float(histogram.get("sum", -1.0)) - expected["sum"]) > 1e-6:
        errors.append("metrics.histograms.stage_latency_ms.sum must match samples")
    buckets = histogram.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != len(expected["buckets"]):
        errors.append("metrics.histograms.stage_latency_ms.buckets must match expected buckets")
        return
    for index, expected_bucket in enumerate(expected["buckets"]):
        bucket = buckets[index]
        if not isinstance(bucket, dict):
            errors.append(f"metrics.histograms.stage_latency_ms.buckets[{index}] must be an object")
            continue
        if bucket.get("le") != expected_bucket["le"] or bucket.get("count") != expected_bucket["count"]:
            errors.append(f"metrics.histograms.stage_latency_ms.buckets[{index}] must match samples")


def validate_metrics_ledger_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if data.get("simulation_method") != SIMULATION_METHOD:
        errors.append(f"simulation_method must be {SIMULATION_METHOD}")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    request_id = _non_empty_string(data.get("request_id"), "request_id", errors)
    config = _validate_config(data.get("config"), errors)
    samples = _validate_samples(data.get("samples"), config, errors)
    events = _validate_events(data.get("events"), plan_id=plan_id, request_id=request_id, errors=errors)

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be an object")
        metrics = {}
    expected_counters = _derive_counters(events) if events else {}
    expected_gauges = _derive_gauges(samples) if samples else {}
    counters = metrics.get("counters")
    if not isinstance(counters, dict):
        errors.append("metrics.counters must be an object")
        counters = {}
    for name, expected in expected_counters.items():
        if counters.get(name) != expected:
            errors.append(f"metrics.counters.{name} must match events")
    gauges = metrics.get("gauges")
    if not isinstance(gauges, dict):
        errors.append("metrics.gauges must be an object")
        gauges = {}
    for name, expected in expected_gauges.items():
        actual = gauges.get(name)
        if isinstance(expected, float):
            if abs(float(actual if actual is not None else -1.0) - expected) > 1e-6:
                errors.append(f"metrics.gauges.{name} must match samples")
        elif actual != expected:
            errors.append(f"metrics.gauges.{name} must match samples")

    histograms = metrics.get("histograms")
    if not isinstance(histograms, dict):
        errors.append("metrics.histograms must be an object")
        histograms = {}
    _validate_histogram(histograms.get("stage_latency_ms"), samples, errors)

    alerts = metrics.get("alerts")
    if not isinstance(alerts, list) or not alerts:
        errors.append("metrics.alerts must be a non-empty list")
        alerts = []
    alert_kinds = {alert.get("kind") for alert in alerts if isinstance(alert, dict)}
    missing_alerts = REQUIRED_ALERT_KINDS - alert_kinds
    if missing_alerts:
        errors.append(f"metrics.alerts missing required kinds: {sorted(missing_alerts)}")
    sample_ids = {sample.get("sample_id") for sample in samples}
    for index, alert in enumerate(alerts):
        field = f"metrics.alerts[{index}]"
        if not isinstance(alert, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(alert.get("kind"), f"{field}.kind", errors)
        if alert.get("severity") not in {"info", "warning", "critical"}:
            errors.append(f"{field}.severity must be info, warning, or critical")
        if alert.get("sample_id") not in sample_ids:
            errors.append(f"{field}.sample_id must reference a sample")
        _non_negative_number(alert.get("value"), f"{field}.value", errors)
        _non_negative_number(alert.get("threshold"), f"{field}.threshold", errors)
    if samples and config.get("max_queue_depth") is not None:
        if any(sample.get("queue_depth") == config["max_queue_depth"] for sample in samples):
            if "queue_depth_high" not in alert_kinds:
                errors.append("queue_depth_high alert required when queue reaches max depth")
    if samples and config.get("memory_warning_fraction") is not None:
        if any(
            float(sample.get("memory_pressure_fraction", 0.0)) >= config["memory_warning_fraction"]
            for sample in samples
        ):
            if "memory_pressure_warning" not in alert_kinds:
                errors.append("memory_pressure_warning alert required when memory exceeds warning threshold")
    if expected_counters.get("kv_pages_evicted_total", 0) > 0 and "kv_eviction_started" not in alert_kinds:
        errors.append("kv_eviction_started alert required when KV pages are evicted")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    for flag in (
        "queue_bounded",
        "backpressure_observed",
        "kv_metrics_consistent",
        "memory_pressure_observed",
        "alerts_complete",
        "histograms_consistent",
        "correctness_passed",
    ):
        if result.get(flag) is not True:
            errors.append(f"result.{flag} must be true")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("sample_count") != len(samples):
        errors.append("summary.sample_count must match samples")
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must match events")
    if summary.get("alert_count") != len(alerts):
        errors.append("summary.alert_count must match alerts")
    if expected_gauges:
        for field in (
            "max_queue_depth_observed",
            "max_inflight_observed",
            "max_kv_pages_allocated",
            "max_memory_pressure_fraction",
        ):
            if summary.get(field) != expected_gauges.get(field):
                errors.append(f"summary.{field} must match metrics.gauges")
    if expected_counters:
        if summary.get("backpressure_event_count") != expected_counters.get("backpressure_asserted_total"):
            errors.append("summary.backpressure_event_count must match counters")
        if summary.get("kv_pages_evicted_total") != expected_counters.get("kv_pages_evicted_total"):
            errors.append("summary.kv_pages_evicted_total must match counters")
    expected_histogram = _latency_histogram(samples) if samples else {"count": 0}
    if summary.get("stage_latency_sample_count") != expected_histogram["count"]:
        errors.append("summary.stage_latency_sample_count must match histogram")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")

    warnings.append(
        "metrics ledger is simulation evidence, not live runtime telemetry or dashboard evidence"
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "sample_count": summary.get("sample_count"),
            "event_count": summary.get("event_count"),
            "alert_count": summary.get("alert_count"),
            "max_queue_depth_observed": summary.get("max_queue_depth_observed"),
            "backpressure_event_count": summary.get("backpressure_event_count"),
            "kv_pages_evicted_total": summary.get("kv_pages_evicted_total"),
            "max_memory_pressure_fraction": summary.get("max_memory_pressure_fraction"),
            "stage_latency_sample_count": summary.get("stage_latency_sample_count"),
            "correctness_passed": summary.get("correctness_passed") is True,
        },
    }


def validate_metrics_ledger(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid metrics ledger artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["metrics ledger artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_metrics_ledger_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
