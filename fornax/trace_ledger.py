from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


RECORD_KIND = "trace-correlation-ledger"
MODE = "t1-simulation"
SIMULATION_METHOD = "request-plan-span-correlation"
REQUIRED_COMPONENTS = {
    "serving_gateway",
    "fornax_engine",
    "scheduler",
    "stage-0",
    "transport",
    "stage-1",
    "kv_manager",
    "metrics_ledger",
}
REQUIRED_EVENT_KINDS = {
    "request_received",
    "engine_normalized",
    "scheduler_admitted",
    "microbatch_started",
    "stage_started",
    "router_decision",
    "remote_expert_dispatched",
    "activation_sent",
    "activation_received",
    "kv_read",
    "kv_write",
    "metric_recorded",
    "stream_chunk",
    "request_finished",
    "cleanup",
}
REQUIRED_EDGES = {
    ("serving_gateway", "fornax_engine"),
    ("fornax_engine", "scheduler"),
    ("scheduler", "stage-0"),
    ("stage-0", "transport"),
    ("transport", "stage-1"),
    ("stage-1", "kv_manager"),
    ("stage-1", "metrics_ledger"),
}
CAUSAL_ORDER = [
    "request_received",
    "engine_normalized",
    "scheduler_admitted",
    "microbatch_started",
    "stage_started",
    "router_decision",
    "remote_expert_dispatched",
    "activation_sent",
    "activation_received",
    "kv_read",
    "kv_write",
    "metric_recorded",
    "stream_chunk",
    "request_finished",
    "cleanup",
]


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


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


def _span(
    span_id: str,
    parent_span_id: str | None,
    *,
    component: str,
    logical_host_id: str,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    return {
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "component": component,
        "logical_host_id": logical_host_id,
        "start_s": round(start_s, 6),
        "end_s": round(end_s, 6),
    }


def _event(
    event_id: str,
    span_id: str,
    *,
    kind: str,
    component: str,
    timestamp_s: float,
    logical_host_id: str,
    **fields: Any,
) -> dict[str, Any]:
    data = {
        "event_id": event_id,
        "span_id": span_id,
        "kind": kind,
        "component": component,
        "timestamp_s": round(timestamp_s, 6),
        "logical_host_id": logical_host_id,
    }
    data.update(fields)
    return data


def simulate_trace_ledger(
    *,
    plan_id: str = "trace-ledger-plan",
    request_id: str = "req-trace-ledger",
    trace_id: str = "trace-trace-ledger",
) -> dict[str, Any]:
    """Simulate correlated trace/span propagation through the Fornax lifecycle."""

    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(request_id, "request_id", errors)
    _non_empty_string(trace_id, "trace_id", errors)
    if errors:
        raise ValueError("; ".join(errors))

    components = [
        {"component": "serving_gateway", "logical_host_id": "logical-host-0"},
        {"component": "fornax_engine", "logical_host_id": "logical-host-0"},
        {"component": "scheduler", "logical_host_id": "logical-host-0"},
        {"component": "stage-0", "logical_host_id": "logical-host-0", "stage_id": 0},
        {"component": "transport", "logical_host_id": "logical-link-0-1"},
        {"component": "stage-1", "logical_host_id": "logical-host-1", "stage_id": 1},
        {"component": "kv_manager", "logical_host_id": "logical-host-1"},
        {"component": "metrics_ledger", "logical_host_id": "logical-host-0"},
    ]
    spans = [
        _span("span-serving", None, component="serving_gateway", logical_host_id="logical-host-0", start_s=0.000, end_s=0.030),
        _span("span-engine", "span-serving", component="fornax_engine", logical_host_id="logical-host-0", start_s=0.002, end_s=0.028),
        _span("span-scheduler", "span-engine", component="scheduler", logical_host_id="logical-host-0", start_s=0.004, end_s=0.026),
        _span("span-stage-0", "span-scheduler", component="stage-0", logical_host_id="logical-host-0", start_s=0.006, end_s=0.018),
        _span("span-transport", "span-stage-0", component="transport", logical_host_id="logical-link-0-1", start_s=0.011, end_s=0.015),
        _span("span-stage-1", "span-transport", component="stage-1", logical_host_id="logical-host-1", start_s=0.015, end_s=0.024),
        _span("span-kv", "span-stage-1", component="kv_manager", logical_host_id="logical-host-1", start_s=0.017, end_s=0.022),
        _span("span-metrics", "span-stage-1", component="metrics_ledger", logical_host_id="logical-host-0", start_s=0.019, end_s=0.025),
    ]
    events = [
        _event("evt-001", "span-serving", kind="request_received", component="serving_gateway", timestamp_s=0.001, logical_host_id="logical-host-0", surface="openai_chat_completions"),
        _event("evt-002", "span-engine", kind="engine_normalized", component="fornax_engine", timestamp_s=0.003, logical_host_id="logical-host-0", engine_request_id="engine-req-001"),
        _event("evt-003", "span-scheduler", kind="scheduler_admitted", component="scheduler", timestamp_s=0.005, logical_host_id="logical-host-0", queue_depth=1),
        _event("evt-004", "span-scheduler", kind="microbatch_started", component="scheduler", timestamp_s=0.006, logical_host_id="logical-host-0", microbatch_id="mb-001", request_count=1),
        _event("evt-005", "span-stage-0", kind="stage_started", component="stage-0", timestamp_s=0.007, logical_host_id="logical-host-0", stage_id=0, phase="prefill"),
        _event("evt-006", "span-stage-0", kind="router_decision", component="stage-0", timestamp_s=0.009, logical_host_id="logical-host-0", stage_id=0, layer_id=3, token_id=12, expert_ids=[7, 2], remote=True),
        _event("evt-007", "span-stage-0", kind="remote_expert_dispatched", component="stage-0", timestamp_s=0.010, logical_host_id="logical-host-0", expert_id=7, target_component="stage-1", target_logical_host_id="logical-host-1"),
        _event("evt-008", "span-transport", kind="activation_sent", component="transport", timestamp_s=0.012, logical_host_id="logical-link-0-1", payload_id="activation-001", source_component="stage-0", destination_component="stage-1"),
        _event("evt-009", "span-stage-1", kind="activation_received", component="stage-1", timestamp_s=0.015, logical_host_id="logical-host-1", stage_id=1, payload_id="activation-001"),
        _event("evt-010", "span-stage-1", kind="stage_started", component="stage-1", timestamp_s=0.016, logical_host_id="logical-host-1", stage_id=1, phase="decode"),
        _event("evt-011", "span-kv", kind="kv_read", component="kv_manager", timestamp_s=0.018, logical_host_id="logical-host-1", stage_id=1, page_id="kv-page-001", pages=4),
        _event("evt-012", "span-kv", kind="kv_write", component="kv_manager", timestamp_s=0.021, logical_host_id="logical-host-1", stage_id=1, page_id="kv-page-002", pages=5),
        _event("evt-013", "span-metrics", kind="metric_recorded", component="metrics_ledger", timestamp_s=0.022, logical_host_id="logical-host-0", metric_name="stage_decode_latency_ms", value=6.0, unit="ms"),
        _event("evt-014", "span-serving", kind="stream_chunk", component="serving_gateway", timestamp_s=0.026, logical_host_id="logical-host-0", chunk_index=0, token_count=1),
        _event("evt-015", "span-engine", kind="request_finished", component="fornax_engine", timestamp_s=0.028, logical_host_id="logical-host-0", finish_reason="stop"),
        _event("evt-016", "span-serving", kind="cleanup", component="serving_gateway", timestamp_s=0.030, logical_host_id="logical-host-0", released_spans=len(spans)),
    ]
    for event in events:
        event["trace_id"] = trace_id
        event["request_id"] = request_id
        event["plan_id"] = plan_id
    for span in spans:
        span["trace_id"] = trace_id
        span["request_id"] = request_id
        span["plan_id"] = plan_id
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "simulation_method": SIMULATION_METHOD,
        "trace_id": trace_id,
        "request_id": request_id,
        "plan_id": plan_id,
        "target_workstreams": ["WS-G1", "WS-H1", "WS-C2", "WS-E2"],
        "components": components,
        "spans": spans,
        "events": events,
        "correlation_policy": {
            "require_trace_id": True,
            "require_request_id": True,
            "require_plan_id": True,
            "require_parent_spans": True,
            "require_cleanup": True,
        },
        "summary": {
            "component_count": len(components),
            "span_count": len(spans),
            "event_count": len(events),
            "stage_count": 2,
            "remote_expert_event_count": 1,
            "metric_event_count": 1,
            "cleanup_event_count": 1,
            "required_event_count": len(REQUIRED_EVENT_KINDS),
            "correlation_complete": True,
            "g2_g3_gate_evidence": False,
        },
        "result": {
            "trace_context_preserved": True,
            "parent_child_spans_valid": True,
            "required_events_present": True,
            "causal_order_preserved": True,
            "cleanup_recorded": True,
            "correctness_passed": True,
        },
        "note": (
            "T1 trace-correlation simulation for request, plan, span, stage, "
            "router/expert, KV, metric, and cleanup observability. Not live "
            "runtime telemetry or G2/G3 gate evidence."
        ),
    }


def _component_map(value: Any, errors: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        errors.append("components must be a non-empty list")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, component in enumerate(value):
        field = f"components[{index}]"
        if not isinstance(component, dict):
            errors.append(f"{field} must be an object")
            continue
        name = _non_empty_string(component.get("component"), f"{field}.component", errors)
        _non_empty_string(component.get("logical_host_id"), f"{field}.logical_host_id", errors)
        if name is not None:
            if name in result:
                errors.append(f"duplicate component: {name}")
            result[name] = component
    missing = REQUIRED_COMPONENTS - set(result)
    if missing:
        errors.append(f"components missing required entries: {sorted(missing)}")
    return result


def _span_map(
    value: Any,
    components: dict[str, dict[str, Any]],
    trace_id: str | None,
    request_id: str | None,
    plan_id: str | None,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        errors.append("spans must be a non-empty list")
        return {}
    result: dict[str, dict[str, Any]] = {}
    roots = 0
    for index, span in enumerate(value):
        field = f"spans[{index}]"
        if not isinstance(span, dict):
            errors.append(f"{field} must be an object")
            continue
        span_id = _non_empty_string(span.get("span_id"), f"{field}.span_id", errors)
        parent = span.get("parent_span_id")
        if parent is None:
            roots += 1
        elif not isinstance(parent, str) or not parent:
            errors.append(f"{field}.parent_span_id must be null or a non-empty string")
        component = _non_empty_string(span.get("component"), f"{field}.component", errors)
        if component is not None and component not in components:
            errors.append(f"{field}.component must reference a known component")
        _non_empty_string(span.get("logical_host_id"), f"{field}.logical_host_id", errors)
        start = _non_negative_number(span.get("start_s"), f"{field}.start_s", errors)
        end = _non_negative_number(span.get("end_s"), f"{field}.end_s", errors)
        if start is not None and end is not None and end < start:
            errors.append(f"{field}.end_s must be >= start_s")
        if trace_id is not None and span.get("trace_id") != trace_id:
            errors.append(f"{field}.trace_id must match trace_id")
        if request_id is not None and span.get("request_id") != request_id:
            errors.append(f"{field}.request_id must match request_id")
        if plan_id is not None and span.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match plan_id")
        if span_id is not None:
            if span_id in result:
                errors.append(f"duplicate span_id: {span_id}")
            result[span_id] = span
    if roots != 1:
        errors.append("spans must contain exactly one root span")
    for span_id, span in result.items():
        parent = span.get("parent_span_id")
        if isinstance(parent, str) and parent not in result:
            errors.append(f"span {span_id} parent_span_id references unknown span")
    return result


def _validate_events(
    value: Any,
    spans: dict[str, dict[str, Any]],
    components: dict[str, dict[str, Any]],
    trace_id: str | None,
    request_id: str | None,
    plan_id: str | None,
    errors: list[str],
) -> tuple[set[str], set[int], set[tuple[str, str]]]:
    if not isinstance(value, list) or not value:
        errors.append("events must be a non-empty list")
        return set(), set(), set()
    seen_event_ids: set[str] = set()
    seen_kinds: set[str] = set()
    stage_ids: set[int] = set()
    previous_ts: float | None = None
    edges: set[tuple[str, str]] = set()
    first_index: dict[str, int] = {}
    for index, event in enumerate(value):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        event_id = _non_empty_string(event.get("event_id"), f"{field}.event_id", errors)
        if event_id is not None:
            if event_id in seen_event_ids:
                errors.append(f"duplicate event_id: {event_id}")
            seen_event_ids.add(event_id)
        span_id = _non_empty_string(event.get("span_id"), f"{field}.span_id", errors)
        kind = _non_empty_string(event.get("kind"), f"{field}.kind", errors)
        component = _non_empty_string(event.get("component"), f"{field}.component", errors)
        timestamp = _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        logical_host_id = _non_empty_string(event.get("logical_host_id"), f"{field}.logical_host_id", errors)
        if timestamp is not None:
            if previous_ts is not None and timestamp < previous_ts:
                errors.append(f"{field}.timestamp_s must be non-decreasing")
            previous_ts = timestamp
        if trace_id is not None and event.get("trace_id") != trace_id:
            errors.append(f"{field}.trace_id must match trace_id")
        if request_id is not None and event.get("request_id") != request_id:
            errors.append(f"{field}.request_id must match request_id")
        if plan_id is not None and event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match plan_id")
        span = spans.get(span_id or "")
        if span is None:
            errors.append(f"{field}.span_id references unknown span")
        elif component is not None and span.get("component") != component:
            errors.append(f"{field}.component must match span component")
        elif logical_host_id is not None and span.get("logical_host_id") != logical_host_id:
            errors.append(f"{field}.logical_host_id must match span logical_host_id")
        if kind is not None:
            seen_kinds.add(kind)
            first_index.setdefault(kind, index)
        parent = span.get("parent_span_id") if isinstance(span, dict) else None
        if isinstance(parent, str) and parent in spans:
            parent_component = str(spans[parent].get("component"))
            if component is not None:
                edges.add((parent_component, component))
        if kind in {"stage_started", "router_decision", "activation_received", "kv_read", "kv_write"}:
            stage_id = _non_negative_int(event.get("stage_id"), f"{field}.stage_id", errors)
            if stage_id is not None:
                stage_ids.add(stage_id)
        if kind == "router_decision":
            experts = event.get("expert_ids")
            if not isinstance(experts, list) or not experts:
                errors.append(f"{field}.expert_ids must be a non-empty list")
            elif not all(isinstance(expert, int) for expert in experts):
                errors.append(f"{field}.expert_ids must contain integers")
        elif kind == "remote_expert_dispatched":
            _non_negative_int(event.get("expert_id"), f"{field}.expert_id", errors)
            target_component = _non_empty_string(event.get("target_component"), f"{field}.target_component", errors)
            target_logical_host_id = _non_empty_string(
                event.get("target_logical_host_id"),
                f"{field}.target_logical_host_id",
                errors,
            )
            if target_component is not None and target_component not in components:
                errors.append(f"{field}.target_component must reference a known component")
            elif (
                target_component is not None
                and target_logical_host_id is not None
                and components[target_component].get("logical_host_id") != target_logical_host_id
            ):
                errors.append(
                    f"{field}.target_logical_host_id must match target component logical_host_id"
                )
        elif kind == "activation_sent":
            _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
            source_component = _non_empty_string(event.get("source_component"), f"{field}.source_component", errors)
            destination_component = _non_empty_string(event.get("destination_component"), f"{field}.destination_component", errors)
            if source_component is not None and source_component not in components:
                errors.append(f"{field}.source_component must reference a known component")
            if destination_component is not None and destination_component not in components:
                errors.append(f"{field}.destination_component must reference a known component")
        elif kind == "activation_received":
            _non_empty_string(event.get("payload_id"), f"{field}.payload_id", errors)
        elif kind in {"kv_read", "kv_write"}:
            _non_empty_string(event.get("page_id"), f"{field}.page_id", errors)
            _non_negative_int(event.get("pages"), f"{field}.pages", errors)
        elif kind == "metric_recorded":
            _non_empty_string(event.get("metric_name"), f"{field}.metric_name", errors)
            _non_negative_number(event.get("value"), f"{field}.value", errors)
        elif kind == "cleanup":
            _non_negative_int(event.get("released_spans"), f"{field}.released_spans", errors)
    missing = REQUIRED_EVENT_KINDS - seen_kinds
    if missing:
        errors.append(f"events missing required kinds: {sorted(missing)}")
    previous_kind: str | None = None
    for kind in CAUSAL_ORDER:
        if kind not in first_index:
            continue
        if previous_kind is not None and first_index[kind] < first_index[previous_kind]:
            errors.append(f"event kind {kind} occurs before {previous_kind}")
        previous_kind = kind
    return seen_kinds, stage_ids, edges


def validate_trace_ledger_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "trace ledger is simulation evidence, not live runtime telemetry or G2/G3 evidence"
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if data.get("simulation_method") != SIMULATION_METHOD:
        errors.append(f"simulation_method must be {SIMULATION_METHOD}")
    trace_id = _non_empty_string(data.get("trace_id"), "trace_id", errors)
    request_id = _non_empty_string(data.get("request_id"), "request_id", errors)
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    components = _component_map(data.get("components"), errors)
    spans = _span_map(data.get("spans"), components, trace_id, request_id, plan_id, errors)
    seen_kinds, stage_ids, edges = _validate_events(
        data.get("events"), spans, components, trace_id, request_id, plan_id, errors
    )
    missing_edges = REQUIRED_EDGES - edges
    if missing_edges:
        errors.append(f"spans missing required component edges: {sorted(missing_edges)}")
    if not {0, 1}.issubset(stage_ids):
        errors.append("trace must include stage ids 0 and 1")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    for field in [
        "trace_context_preserved",
        "parent_child_spans_valid",
        "required_events_present",
        "causal_order_preserved",
        "cleanup_recorded",
        "correctness_passed",
    ]:
        if result.get(field) is not True:
            errors.append(f"result.{field} must be true")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    events = data.get("events") if isinstance(data.get("events"), list) else []
    remote_expert_event_count = sum(
        1 for event in events if isinstance(event, dict) and event.get("kind") == "remote_expert_dispatched"
    )
    metric_event_count = sum(
        1 for event in events if isinstance(event, dict) and event.get("kind") == "metric_recorded"
    )
    cleanup_event_count = sum(
        1 for event in events if isinstance(event, dict) and event.get("kind") == "cleanup"
    )
    if summary.get("component_count") != len(components):
        errors.append("summary.component_count must match components")
    if summary.get("span_count") != len(spans):
        errors.append("summary.span_count must match spans")
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must match events")
    if summary.get("stage_count") != len(stage_ids):
        errors.append("summary.stage_count must match observed stages")
    if summary.get("required_event_count") != len(REQUIRED_EVENT_KINDS):
        errors.append("summary.required_event_count must match required events")
    if summary.get("remote_expert_event_count") != remote_expert_event_count:
        errors.append("summary.remote_expert_event_count must match events")
    if summary.get("metric_event_count") != metric_event_count:
        errors.append("summary.metric_event_count must match events")
    if summary.get("cleanup_event_count") != cleanup_event_count:
        errors.append("summary.cleanup_event_count must match events")
    if summary.get("correlation_complete") is not True:
        errors.append("summary.correlation_complete must be true")
    if summary.get("g2_g3_gate_evidence") is not False:
        errors.append("summary.g2_g3_gate_evidence must be false")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "trace_id": trace_id,
            "request_id": request_id,
            "plan_id": plan_id,
            "component_count": len(components),
            "span_count": len(spans),
            "event_count": len(events),
            "stage_count": len(stage_ids),
            "required_events_seen": sorted(seen_kinds & REQUIRED_EVENT_KINDS),
            "required_edge_count": len(edges & REQUIRED_EDGES),
            "remote_expert_event_count": remote_expert_event_count,
            "metric_event_count": metric_event_count,
            "cleanup_event_count": cleanup_event_count,
            "correlation_complete": summary.get("correlation_complete") is True,
        },
    }


def validate_trace_ledger(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid trace ledger artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["trace ledger artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_trace_ledger_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
