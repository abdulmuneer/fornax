from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_BACKENDS = ("linux_nvidia", "linux_amd", "macos_apple_silicon")
REQUIRED_OPERATIONS = (
    "attention",
    "dense_mlp",
    "router_topk",
    "expert_gemm_mlp",
    "collect_scatter_gather",
    "kv_operations",
    "sampling_logits",
    "serialization_pack_gather",
    "transport",
)
STATUS_VALUES = {"yes", "no", "unknown", "not_applicable"}
REQUIRED_LEDGER_FIELDS = {
    "hardware",
    "os",
    "driver_runtime",
    "max_mojo_version",
    "model",
    "context",
    "concurrency",
    "quantization",
    "thermals",
    "command",
}


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


def _status(value: Any, field: str, errors: list[str]) -> str | None:
    text = _non_empty_string(value, field, errors)
    if text is not None and text not in STATUS_VALUES:
        errors.append(f"{field} must be one of {sorted(STATUS_VALUES)}")
    return text


def _check_evidence(
    value: Any,
    field: str,
    status_values: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{field}.evidence must be an object")
        return
    kind = _non_empty_string(value.get("kind"), f"{field}.evidence.kind", errors)
    _non_empty_string(value.get("source"), f"{field}.evidence.source", errors)
    if kind == "measured":
        _string_list(value.get("command"), f"{field}.evidence.command", errors)
        _non_empty_string(value.get("artifact"), f"{field}.evidence.artifact", errors)
    if any(status == "yes" for status in status_values) and kind in {
        "placeholder",
        "none",
    }:
        errors.append(f"{field}.evidence cannot be placeholder for a yes status")
    if any(status == "unknown" for status in status_values):
        reason = value.get("reason") or value.get("note")
        if not isinstance(reason, str) or not reason:
            errors.append(f"{field}.evidence must explain unknown status")
    if kind == "fixture":
        warnings.append(f"{field}.evidence is fixture-only, not gate evidence")


def _check_backend_status(
    value: Any,
    field: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return
    supported = _status(value.get("supported"), f"{field}.supported", errors)
    fast_enough = _status(value.get("fast_enough"), f"{field}.fast_enough", errors)
    correct = _status(value.get("correct"), f"{field}.correct", errors)
    used = value.get("used_by_target_model")
    if not isinstance(used, bool):
        errors.append(f"{field}.used_by_target_model must be a boolean")
    _check_evidence(
        value.get("evidence"),
        field,
        [item for item in (supported, fast_enough, correct) if item is not None],
        errors,
        warnings,
    )


def _check_profiler(value: Any, backend: str, errors: list[str]) -> None:
    field = f"profilers.{backend}"
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return
    _non_empty_string(value.get("name"), f"{field}.name", errors)
    _non_empty_string(value.get("role"), f"{field}.role", errors)
    _string_list(value.get("command"), f"{field}.command", errors)
    _non_empty_string(value.get("status"), f"{field}.status", errors)


def validate_backend_coverage_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("contract_kind") != "backend-coverage":
        errors.append("contract_kind must be backend-coverage")
    _non_empty_string(data.get("target_model"), "target_model", errors)
    _non_empty_string(data.get("quantization"), "quantization", errors)
    _non_empty_string(data.get("in_flight_dtype"), "in_flight_dtype", errors)

    profilers = data.get("profilers")
    if not isinstance(profilers, dict):
        errors.append("profilers must be an object")
        profilers = {}
    for backend in REQUIRED_BACKENDS:
        _check_profiler(profilers.get(backend), backend, errors)

    ledger_fields = _string_list(
        data.get("benchmark_ledger_fields"), "benchmark_ledger_fields", errors
    )
    if ledger_fields is not None:
        missing = sorted(REQUIRED_LEDGER_FIELDS - set(ledger_fields))
        if missing:
            errors.append("benchmark_ledger_fields missing: " + ", ".join(missing))

    operations = data.get("operations")
    if not isinstance(operations, list) or not operations:
        errors.append("operations must be a non-empty list")
        operations = []

    seen_operations: set[str] = set()
    for op_index, operation in enumerate(operations):
        field = f"operations[{op_index}]"
        if not isinstance(operation, dict):
            errors.append(f"{field} must be an object")
            continue
        name = _non_empty_string(operation.get("operation"), f"{field}.operation", errors)
        if name is not None:
            seen_operations.add(name)
        backends = operation.get("backends")
        if not isinstance(backends, dict):
            errors.append(f"{field}.backends must be an object")
            continue
        for backend in REQUIRED_BACKENDS:
            _check_backend_status(
                backends.get(backend),
                f"{field}.backends.{backend}",
                errors,
                warnings,
            )

    missing_ops = sorted(set(REQUIRED_OPERATIONS) - seen_operations)
    extra_ops = sorted(seen_operations - set(REQUIRED_OPERATIONS))
    if missing_ops:
        errors.append("operations missing required entries: " + ", ".join(missing_ops))
    if extra_ops:
        warnings.append("operations include non-required entries: " + ", ".join(extra_ops))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "target_model": data.get("target_model"),
            "quantization": data.get("quantization"),
            "operation_count": len(operations),
            "required_operations_seen": [
                name for name in REQUIRED_OPERATIONS if name in seen_operations
            ],
            "required_backends": list(REQUIRED_BACKENDS),
        },
    }


def validate_backend_coverage_contract(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing backend-coverage fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture failures.
        return {
            "ok": False,
            "errors": [f"invalid backend-coverage fixture JSON: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["backend-coverage fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_backend_coverage_fixture(data)
    result["fixture"] = str(fixture_path)
    return result


def render_backend_coverage_report(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    data = read_json(fixture_path)
    if not isinstance(data, dict):
        raise ValueError("backend-coverage fixture must be a JSON object")
    validation = validate_backend_coverage_fixture(data)
    markdown = _render_markdown(data, validation)
    return {
        "ok": bool(validation["ok"]),
        "validation": validation,
        "fixture": str(fixture_path),
        "markdown": markdown,
    }


def _cell(status: dict[str, Any], key: str) -> str:
    value = status.get(key, "missing") if isinstance(status, dict) else "missing"
    return str(value)


def _render_markdown(data: dict[str, Any], validation: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Backend Operation Coverage Matrix",
        "",
        "Status: DRAFT - Phase-0 evidence shape, not G1 closure evidence.",
        "",
        f"- Target model: `{data.get('target_model', 'missing')}`",
        f"- Quantization: `{data.get('quantization', 'missing')}`",
        f"- In-flight dtype: `{data.get('in_flight_dtype', 'missing')}`",
        "",
        "## Operation Matrix",
        "",
        "| Operation | Backend | Supported | Fast enough | Correct | Used by target | Evidence |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for operation in data.get("operations", []):
        if not isinstance(operation, dict):
            continue
        op_name = str(operation.get("operation", "missing"))
        backends = operation.get("backends") if isinstance(operation.get("backends"), dict) else {}
        for backend in REQUIRED_BACKENDS:
            status = backends.get(backend) if isinstance(backends, dict) else {}
            evidence = status.get("evidence") if isinstance(status, dict) else {}
            source = evidence.get("source", "missing") if isinstance(evidence, dict) else "missing"
            lines.append(
                "| "
                + " | ".join(
                    [
                        op_name,
                        backend,
                        _cell(status, "supported"),
                        _cell(status, "fast_enough"),
                        _cell(status, "correct"),
                        str(status.get("used_by_target_model", "missing"))
                        if isinstance(status, dict)
                        else "missing",
                        str(source),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Profilers",
            "",
            "| Backend | Profiler / harness | Role | Status |",
            "|---|---|---|---|",
        ]
    )
    profilers = data.get("profilers") if isinstance(data.get("profilers"), dict) else {}
    for backend in REQUIRED_BACKENDS:
        profiler = profilers.get(backend) if isinstance(profilers, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    backend,
                    str(profiler.get("name", "missing"))
                    if isinstance(profiler, dict)
                    else "missing",
                    str(profiler.get("role", "missing"))
                    if isinstance(profiler, dict)
                    else "missing",
                    str(profiler.get("status", "missing"))
                    if isinstance(profiler, dict)
                    else "missing",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Benchmark Ledger Fields",
            "",
            ", ".join(str(item) for item in data.get("benchmark_ledger_fields", [])),
            "",
            "## Validation",
            "",
            f"- Valid: `{validation['ok']}`",
            "- Errors: " + ("; ".join(validation["errors"]) or "none"),
            "- Warnings: " + ("; ".join(validation["warnings"]) or "none"),
            "",
        ]
    )
    return "\n".join(lines)
