from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "stage-host-simulation-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "deterministic-layer-group-stage-host"
BACKEND = "simulated-max-graphlet"
REFERENCE_PATH = "cpu-stdlib-slow-correct"
REQUIRED_BOUNDARY_OPS = {"activation_in", "activation_out", "kv_read", "kv_write"}
REQUIRED_EVENT_KINDS = {
    "stage_loaded",
    "activation_received",
    "kv_read",
    "graphlet_executed",
    "kv_written",
    "activation_emitted",
    "reference_compared",
    "cleanup",
}


def _positive_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return None
    return value


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


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _activation_value(token: int, dim: int) -> float:
    return round((((token + 2) * (dim + 5)) % 31 - 15) / 31.0, 6)


def _weight_value(layer: int, input_dim: int, output_dim: int) -> float:
    return (((layer + 3) * (input_dim + 7) * (output_dim + 11)) % 43 - 21) / 43.0


def _bias_value(layer: int, output_dim: int) -> float:
    return (((layer + 5) * (output_dim + 13)) % 23 - 11) / 230.0


def _activation_matrix(token_count: int, hidden_dim: int) -> list[list[float]]:
    return [
        [_activation_value(token, dim) for dim in range(hidden_dim)]
        for token in range(token_count)
    ]


def _run_layer_group(
    activations: list[list[float]],
    *,
    layer_ids: list[int],
    hidden_dim: int,
) -> list[list[float]]:
    current = [list(row) for row in activations]
    for layer in layer_ids:
        output: list[list[float]] = []
        for row in current:
            next_row: list[float] = []
            for output_dim in range(hidden_dim):
                acc = _bias_value(layer, output_dim)
                for input_dim, value in enumerate(row):
                    acc += value * _weight_value(layer, input_dim, output_dim)
                next_row.append(round(acc if acc > 0.0 else 0.0, 9))
            output.append(next_row)
        current = output
    return current


def _kv_payload(token_count: int, hidden_dim: int, *, offset: int) -> dict[str, list[list[float]]]:
    keys: list[list[float]] = []
    values: list[list[float]] = []
    for token in range(token_count):
        key_row: list[float] = []
        value_row: list[float] = []
        for dim in range(hidden_dim):
            key_row.append(round((((token + offset) * (dim + 3)) % 29 - 14) / 29.0, 6))
            value_row.append(round((((token + offset + 2) * (dim + 7)) % 31 - 15) / 31.0, 6))
        keys.append(key_row)
        values.append(value_row)
    return {"keys": keys, "values": values}


def _checksum(matrix: list[list[float]]) -> float:
    total = 0.0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            total += value * (1.0 + row_index * 0.01 + col_index * 0.001)
    return round(total, 12)


def _matrix_shape(matrix: Any) -> list[int] | None:
    if not isinstance(matrix, list) or not matrix:
        return None
    width: int | None = None
    for row in matrix:
        if not isinstance(row, list) or not row:
            return None
        if width is None:
            width = len(row)
        if len(row) != width:
            return None
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
    return [len(matrix), int(width or 0)]


def _max_abs_error(left: list[list[float]], right: list[list[float]]) -> float:
    max_error = 0.0
    for row_left, row_right in zip(left, right):
        for value_left, value_right in zip(row_left, row_right):
            max_error = max(max_error, abs(float(value_left) - float(value_right)))
    return round(max_error, 12)


def _event(kind: str, *, timestamp_s: float, plan_id: str, **fields: Any) -> dict[str, Any]:
    event = {"kind": kind, "timestamp_s": round(timestamp_s, 9), "plan_id": plan_id}
    event.update(fields)
    return event


def _boundary_op(
    name: str,
    *,
    direction: str,
    dtype: str,
    shape: list[int],
    producer: str,
    consumer: str,
    payload_path: str,
    payload: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "op_kind": "custom-boundary-op",
        "direction": direction,
        "dtype": dtype,
        "shape": shape,
        "producer": producer,
        "consumer": consumer,
        "payload_path": payload_path,
        "payload_hash": _canonical_hash(payload),
    }


def simulate_stage_host(
    *,
    plan_id: str = "stage-host-plan",
    request_id: str = "req-stage-host",
    stage_id: str = "stage-1",
    logical_host_id: str = "logical-host-1",
    predecessor_stage_id: str = "stage-0",
    successor_stage_id: str = "stage-2",
    layer_start: int = 12,
    layer_count: int = 2,
    token_count: int = 3,
    hidden_dim: int = 4,
    dtype: str = "fp16",
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Simulate a layer-group stage host with explicit boundary handoff checks."""

    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(request_id, "request_id", errors)
    _non_empty_string(stage_id, "stage_id", errors)
    _non_empty_string(logical_host_id, "logical_host_id", errors)
    _non_empty_string(predecessor_stage_id, "predecessor_stage_id", errors)
    _non_empty_string(successor_stage_id, "successor_stage_id", errors)
    _positive_int(layer_start + 1, "layer_start", errors)
    _positive_int(layer_count, "layer_count", errors)
    _positive_int(token_count, "token_count", errors)
    _positive_int(hidden_dim, "hidden_dim", errors)
    _non_empty_string(dtype, "dtype", errors)
    _non_negative_number(tolerance, "tolerance", errors)
    if stage_id in {predecessor_stage_id, successor_stage_id}:
        errors.append("stage_id must differ from predecessor_stage_id and successor_stage_id")
    if errors:
        raise ValueError("; ".join(errors))

    layer_ids = list(range(layer_start, layer_start + layer_count))
    activation_input = _activation_matrix(token_count, hidden_dim)
    reference_output = _run_layer_group(
        activation_input, layer_ids=layer_ids, hidden_dim=hidden_dim
    )
    stage_output = _run_layer_group(
        activation_input, layer_ids=layer_ids, hidden_dim=hidden_dim
    )
    kv_read_payload = _kv_payload(token_count, hidden_dim, offset=3)
    kv_write_payload = _kv_payload(token_count, hidden_dim, offset=layer_start + 5)
    max_abs_error = _max_abs_error(stage_output, reference_output)
    output_checksum = _checksum(stage_output)
    reference_checksum = _checksum(reference_output)
    shape = [token_count, hidden_dim]
    graphlet_id = f"{stage_id}-layer-group-{layer_start}-{layer_start + layer_count - 1}"
    boundary_ops = [
        _boundary_op(
            "activation_in",
            direction="receive",
            dtype=dtype,
            shape=shape,
            producer=predecessor_stage_id,
            consumer=stage_id,
            payload_path="tensors.activation_input",
            payload=activation_input,
        ),
        _boundary_op(
            "kv_read",
            direction="read",
            dtype=dtype,
            shape=shape,
            producer=stage_id,
            consumer=stage_id,
            payload_path="tensors.kv_read",
            payload=kv_read_payload,
        ),
        _boundary_op(
            "kv_write",
            direction="write",
            dtype=dtype,
            shape=shape,
            producer=stage_id,
            consumer=stage_id,
            payload_path="tensors.kv_write",
            payload=kv_write_payload,
        ),
        _boundary_op(
            "activation_out",
            direction="send",
            dtype=dtype,
            shape=shape,
            producer=stage_id,
            consumer=successor_stage_id,
            payload_path="tensors.stage_output",
            payload=stage_output,
        ),
    ]
    events = [
        _event(
            "stage_loaded",
            timestamp_s=0.000,
            plan_id=plan_id,
            stage_id=stage_id,
            logical_host_id=logical_host_id,
            graphlet_id=graphlet_id,
            layer_ids=layer_ids,
        ),
        _event(
            "activation_received",
            timestamp_s=0.001,
            plan_id=plan_id,
            stage_id=stage_id,
            request_id=request_id,
            boundary_op="activation_in",
            payload_hash=_canonical_hash(activation_input),
        ),
        _event(
            "kv_read",
            timestamp_s=0.002,
            plan_id=plan_id,
            stage_id=stage_id,
            request_id=request_id,
            boundary_op="kv_read",
            payload_hash=_canonical_hash(kv_read_payload),
        ),
        _event(
            "graphlet_executed",
            timestamp_s=0.003,
            plan_id=plan_id,
            stage_id=stage_id,
            request_id=request_id,
            graphlet_id=graphlet_id,
            backend=BACKEND,
            measured=False,
        ),
        _event(
            "kv_written",
            timestamp_s=0.004,
            plan_id=plan_id,
            stage_id=stage_id,
            request_id=request_id,
            boundary_op="kv_write",
            payload_hash=_canonical_hash(kv_write_payload),
        ),
        _event(
            "activation_emitted",
            timestamp_s=0.005,
            plan_id=plan_id,
            stage_id=stage_id,
            request_id=request_id,
            boundary_op="activation_out",
            payload_hash=_canonical_hash(stage_output),
        ),
        _event(
            "reference_compared",
            timestamp_s=0.006,
            plan_id=plan_id,
            stage_id=stage_id,
            request_id=request_id,
            max_abs_error=max_abs_error,
            tolerance=tolerance,
        ),
        _event(
            "cleanup",
            timestamp_s=0.007,
            plan_id=plan_id,
            stage_id=stage_id,
            request_id=request_id,
            buffers_released=True,
        ),
    ]
    kv_ownership_preserved = True
    outputs_match_reference = max_abs_error <= tolerance
    graphlet_claim_is_simulated = True
    boundary_ops_complete = {op["name"] for op in boundary_ops} == REQUIRED_BOUNDARY_OPS
    correctness_passed = (
        outputs_match_reference
        and kv_ownership_preserved
        and graphlet_claim_is_simulated
        and boundary_ops_complete
    )
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "request_id": request_id,
        "simulation_method": SIMULATION_METHOD,
        "target_workstreams": ["WS-B2", "WS-B3", "WS-B4"],
        "config": {
            "token_count": token_count,
            "hidden_dim": hidden_dim,
            "dtype": dtype,
            "tolerance": tolerance,
        },
        "stage_host": {
            "stage_id": stage_id,
            "logical_host_id": logical_host_id,
            "predecessor_stage_id": predecessor_stage_id,
            "successor_stage_id": successor_stage_id,
            "layer_group": {
                "start": layer_start,
                "end": layer_start + layer_count - 1,
                "layers": layer_ids,
                "layer_count": layer_count,
            },
            "backend": BACKEND,
            "max_graphlet_status": "planned",
            "reference_path": REFERENCE_PATH,
            "measured": False,
            "hardware_accelerated": False,
            "graphlet_id": graphlet_id,
        },
        "graphlet_contract": {
            "graphlet_id": graphlet_id,
            "status": "planned",
            "backend": BACKEND,
            "measured": False,
            "hardware_accelerated": False,
            "claim": "simulation-only placeholder; no MAX graph was executed",
            "forbidden_claims": [
                "measured_max_execution",
                "hardware_accelerated_execution",
                "g2_distributed_correctness",
            ],
        },
        "tensors": {
            "activation_input": activation_input,
            "stage_output": stage_output,
            "reference_output": reference_output,
            "kv_read": kv_read_payload,
            "kv_write": kv_write_payload,
        },
        "boundary_ops": boundary_ops,
        "kv_cache": {
            "cache_id": f"{request_id}:{stage_id}:kv",
            "owner_before": stage_id,
            "owner_after": stage_id,
            "handoff_policy": "stage-local-readwrite",
            "read_count": token_count,
            "write_count": token_count,
            "read_payload_hash": _canonical_hash(kv_read_payload),
            "write_payload_hash": _canonical_hash(kv_write_payload),
        },
        "events": events,
        "result": {
            "output_checksum": output_checksum,
            "reference_checksum": reference_checksum,
            "max_abs_error": max_abs_error,
            "outputs_match_reference": outputs_match_reference,
            "kv_ownership_preserved": kv_ownership_preserved,
            "graphlet_claim_is_simulated": graphlet_claim_is_simulated,
            "boundary_ops_complete": boundary_ops_complete,
            "correctness_passed": correctness_passed,
        },
        "summary": {
            "event_count": len(events),
            "boundary_op_count": len(boundary_ops),
            "token_count": token_count,
            "hidden_dim": hidden_dim,
            "layer_count": layer_count,
            "max_abs_error": max_abs_error,
            "outputs_match_reference": outputs_match_reference,
            "kv_ownership_preserved": kv_ownership_preserved,
            "graphlet_claim_is_simulated": graphlet_claim_is_simulated,
            "correctness_passed": correctness_passed,
        },
        "note": (
            "T1 stage-host simulation: validates layer-group boundary formats, KV ownership, "
            "lifecycle events, and slow-correct reference parity. This is not real MAX graph "
            "execution and does not satisfy G2 distributed correctness."
        ),
    }


def _payload_for_path(data: dict[str, Any], payload_path: str) -> Any:
    current: Any = data
    for part in payload_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _validate_tensor_payloads(
    data: dict[str, Any],
    *,
    token_count: int | None,
    hidden_dim: int | None,
    errors: list[str],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    tensors = data.get("tensors")
    if not isinstance(tensors, dict):
        errors.append("tensors must be an object")
        tensors = {}
    expected_shape = (
        [token_count, hidden_dim]
        if token_count is not None and hidden_dim is not None
        else None
    )
    matrices: dict[str, list[list[float]]] = {}
    for name in ("activation_input", "stage_output", "reference_output"):
        matrix = tensors.get(name)
        shape = _matrix_shape(matrix)
        if shape is None:
            errors.append(f"tensors.{name} must be a non-empty numeric matrix")
            matrix = []
        elif expected_shape is not None and shape != expected_shape:
            errors.append(f"tensors.{name} shape must be {expected_shape}")
        matrices[name] = matrix if isinstance(matrix, list) else []
    for name in ("kv_read", "kv_write"):
        payload = tensors.get(name)
        if not isinstance(payload, dict):
            errors.append(f"tensors.{name} must be an object")
            continue
        for key in ("keys", "values"):
            shape = _matrix_shape(payload.get(key))
            if shape is None:
                errors.append(f"tensors.{name}.{key} must be a non-empty numeric matrix")
            elif expected_shape is not None and shape != expected_shape:
                errors.append(f"tensors.{name}.{key} shape must be {expected_shape}")
    return (
        matrices["activation_input"],
        matrices["stage_output"],
        matrices["reference_output"],
    )


def validate_stage_host_fixture(data: dict[str, Any]) -> dict[str, Any]:
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
    _non_empty_string(data.get("request_id"), "request_id", errors)
    config = data.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
        config = {}
    token_count = _positive_int(config.get("token_count"), "config.token_count", errors)
    hidden_dim = _positive_int(config.get("hidden_dim"), "config.hidden_dim", errors)
    dtype = _non_empty_string(config.get("dtype"), "config.dtype", errors)
    tolerance = _non_negative_number(config.get("tolerance"), "config.tolerance", errors)
    stage_host = data.get("stage_host")
    if not isinstance(stage_host, dict):
        errors.append("stage_host must be an object")
        stage_host = {}
    stage_id = _non_empty_string(stage_host.get("stage_id"), "stage_host.stage_id", errors)
    _non_empty_string(stage_host.get("logical_host_id"), "stage_host.logical_host_id", errors)
    _non_empty_string(
        stage_host.get("predecessor_stage_id"),
        "stage_host.predecessor_stage_id",
        errors,
    )
    _non_empty_string(
        stage_host.get("successor_stage_id"),
        "stage_host.successor_stage_id",
        errors,
    )
    if stage_host.get("backend") != BACKEND:
        errors.append(f"stage_host.backend must be {BACKEND}")
    if stage_host.get("max_graphlet_status") != "planned":
        errors.append("stage_host.max_graphlet_status must be planned")
    if stage_host.get("reference_path") != REFERENCE_PATH:
        errors.append(f"stage_host.reference_path must be {REFERENCE_PATH}")
    if stage_host.get("measured") is not False:
        errors.append("stage_host.measured must be false for simulation evidence")
    if stage_host.get("hardware_accelerated") is not False:
        errors.append("stage_host.hardware_accelerated must be false for simulation evidence")
    _non_empty_string(stage_host.get("graphlet_id"), "stage_host.graphlet_id", errors)
    layer_group = stage_host.get("layer_group")
    if not isinstance(layer_group, dict):
        errors.append("stage_host.layer_group must be an object")
        layer_group = {}
    layer_count = _positive_int(
        layer_group.get("layer_count"), "stage_host.layer_group.layer_count", errors
    )
    layers = layer_group.get("layers")
    if not isinstance(layers, list) or not layers:
        errors.append("stage_host.layer_group.layers must be a non-empty list")
    elif layer_count is not None and len(layers) != layer_count:
        errors.append("stage_host.layer_group.layers length must equal layer_count")

    graphlet = data.get("graphlet_contract")
    if not isinstance(graphlet, dict):
        errors.append("graphlet_contract must be an object")
        graphlet = {}
    if graphlet.get("status") != "planned":
        errors.append("graphlet_contract.status must be planned")
    if graphlet.get("backend") != BACKEND:
        errors.append(f"graphlet_contract.backend must be {BACKEND}")
    if graphlet.get("measured") is not False:
        errors.append("graphlet_contract.measured must be false for simulation evidence")
    if graphlet.get("hardware_accelerated") is not False:
        errors.append(
            "graphlet_contract.hardware_accelerated must be false for simulation evidence"
        )
    if graphlet.get("graphlet_id") != stage_host.get("graphlet_id"):
        errors.append("graphlet_contract.graphlet_id must match stage_host.graphlet_id")

    _, stage_output, reference_output = _validate_tensor_payloads(
        data, token_count=token_count, hidden_dim=hidden_dim, errors=errors
    )
    computed_error = _max_abs_error(stage_output, reference_output) if stage_output and reference_output else None
    computed_output_checksum = _checksum(stage_output) if stage_output else None
    computed_reference_checksum = _checksum(reference_output) if reference_output else None

    boundary_ops = data.get("boundary_ops")
    if not isinstance(boundary_ops, list) or not boundary_ops:
        errors.append("boundary_ops must be a non-empty list")
        boundary_ops = []
    names: list[str] = []
    expected_shape = [token_count, hidden_dim] if token_count is not None and hidden_dim is not None else None
    for index, op in enumerate(boundary_ops):
        field = f"boundary_ops[{index}]"
        if not isinstance(op, dict):
            errors.append(f"{field} must be an object")
            continue
        name = _non_empty_string(op.get("name"), f"{field}.name", errors)
        if name is not None:
            names.append(name)
        if op.get("op_kind") != "custom-boundary-op":
            errors.append(f"{field}.op_kind must be custom-boundary-op")
        _non_empty_string(op.get("direction"), f"{field}.direction", errors)
        if dtype is not None and op.get("dtype") != dtype:
            errors.append(f"{field}.dtype must match config.dtype")
        if expected_shape is not None and op.get("shape") != expected_shape:
            errors.append(f"{field}.shape must match tensor shape")
        _non_empty_string(op.get("producer"), f"{field}.producer", errors)
        _non_empty_string(op.get("consumer"), f"{field}.consumer", errors)
        payload_path = _non_empty_string(op.get("payload_path"), f"{field}.payload_path", errors)
        payload_hash = _non_empty_string(op.get("payload_hash"), f"{field}.payload_hash", errors)
        if payload_path is not None and payload_hash is not None:
            payload = _payload_for_path(data, payload_path)
            if payload is None:
                errors.append(f"{field}.payload_path does not resolve")
            elif _canonical_hash(payload) != payload_hash:
                errors.append(f"{field}.payload_hash must match payload_path")
    missing_ops = REQUIRED_BOUNDARY_OPS - set(names)
    if missing_ops:
        errors.append(f"boundary_ops missing required entries: {sorted(missing_ops)}")
    if len(names) != len(set(names)):
        errors.append("boundary_ops names must be unique")

    kv_cache = data.get("kv_cache")
    if not isinstance(kv_cache, dict):
        errors.append("kv_cache must be an object")
        kv_cache = {}
    _non_empty_string(kv_cache.get("cache_id"), "kv_cache.cache_id", errors)
    if stage_id is not None:
        if kv_cache.get("owner_before") != stage_id:
            errors.append("kv_cache.owner_before must match stage_host.stage_id")
        if kv_cache.get("owner_after") != stage_id:
            errors.append("kv_cache.owner_after must match stage_host.stage_id")
    if kv_cache.get("handoff_policy") != "stage-local-readwrite":
        errors.append("kv_cache.handoff_policy must be stage-local-readwrite")
    if token_count is not None:
        if kv_cache.get("read_count") != token_count:
            errors.append("kv_cache.read_count must equal config.token_count")
        if kv_cache.get("write_count") != token_count:
            errors.append("kv_cache.write_count must equal config.token_count")
    tensors = data.get("tensors") if isinstance(data.get("tensors"), dict) else {}
    if kv_cache.get("read_payload_hash") != _canonical_hash(tensors.get("kv_read")):
        errors.append("kv_cache.read_payload_hash must match tensors.kv_read")
    if kv_cache.get("write_payload_hash") != _canonical_hash(tensors.get("kv_write")):
        errors.append("kv_cache.write_payload_hash must match tensors.kv_write")

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []
    event_kinds = {event.get("kind") for event in events if isinstance(event, dict)}
    missing_events = REQUIRED_EVENT_KINDS - event_kinds
    if missing_events:
        errors.append(f"events missing required kinds: {sorted(missing_events)}")
    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(event.get("kind"), f"{field}.kind", errors)
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        if plan_id is not None and event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match plan_id")
        if event.get("kind") == "graphlet_executed" and event.get("measured") is not False:
            errors.append(f"{field}.measured must be false for simulation evidence")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    result_error = _non_negative_number(result.get("max_abs_error"), "result.max_abs_error", errors)
    if computed_error is not None and result_error is not None:
        if abs(result_error - computed_error) > 1e-9:
            errors.append("result.max_abs_error must match tensors.stage_output/reference_output")
        if tolerance is not None and computed_error > tolerance:
            errors.append("result.max_abs_error exceeds config.tolerance")
    if computed_output_checksum is not None and result.get("output_checksum") != computed_output_checksum:
        errors.append("result.output_checksum must match tensors.stage_output")
    if (
        computed_reference_checksum is not None
        and result.get("reference_checksum") != computed_reference_checksum
    ):
        errors.append("result.reference_checksum must match tensors.reference_output")
    if result.get("outputs_match_reference") is not True:
        errors.append("result.outputs_match_reference must be true")
    if result.get("kv_ownership_preserved") is not True:
        errors.append("result.kv_ownership_preserved must be true")
    if result.get("graphlet_claim_is_simulated") is not True:
        errors.append("result.graphlet_claim_is_simulated must be true")
    if result.get("boundary_ops_complete") is not True:
        errors.append("result.boundary_ops_complete must be true")
    if result.get("correctness_passed") is not True:
        errors.append("result.correctness_passed must be true")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must equal len(events)")
    if summary.get("boundary_op_count") != len(boundary_ops):
        errors.append("summary.boundary_op_count must equal len(boundary_ops)")
    if token_count is not None and summary.get("token_count") != token_count:
        errors.append("summary.token_count must equal config.token_count")
    if hidden_dim is not None and summary.get("hidden_dim") != hidden_dim:
        errors.append("summary.hidden_dim must equal config.hidden_dim")
    if layer_count is not None and summary.get("layer_count") != layer_count:
        errors.append("summary.layer_count must equal stage_host.layer_group.layer_count")
    if computed_error is not None and summary.get("max_abs_error") != computed_error:
        errors.append("summary.max_abs_error must match computed max error")
    if summary.get("outputs_match_reference") is not True:
        errors.append("summary.outputs_match_reference must be true")
    if summary.get("kv_ownership_preserved") is not True:
        errors.append("summary.kv_ownership_preserved must be true")
    if summary.get("graphlet_claim_is_simulated") is not True:
        errors.append("summary.graphlet_claim_is_simulated must be true")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")

    warnings.append(
        "stage host is simulation evidence, not real MAX graphlet execution or G2 evidence"
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "event_count": summary.get("event_count"),
            "boundary_op_count": summary.get("boundary_op_count"),
            "token_count": summary.get("token_count"),
            "hidden_dim": summary.get("hidden_dim"),
            "layer_count": summary.get("layer_count"),
            "max_abs_error": result.get("max_abs_error"),
            "outputs_match_reference": result.get("outputs_match_reference") is True,
            "kv_ownership_preserved": result.get("kv_ownership_preserved") is True,
            "graphlet_claim_is_simulated": result.get("graphlet_claim_is_simulated") is True,
            "correctness_passed": result.get("correctness_passed") is True,
        },
    }


def validate_stage_host(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid stage host artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["stage host artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_stage_host_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
