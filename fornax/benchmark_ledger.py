from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_LEDGER_FIELDS = (
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
)
REQUIRED_RESULT_FIELDS = ("elapsed_s", "tokens_s", "checksum")


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _command(value: Any, field: str, errors: list[str]) -> list[str] | None:
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


def _unknown_warning(record: dict[str, Any], field: str, warnings: list[str]) -> None:
    value = record.get(field)
    if isinstance(value, str) and value in {"unknown", "unset", "unmeasured"}:
        warnings.append(f"{field} is {value}; replace before gate evidence")


def build_benchmark_ledger_record(
    benchmark: dict[str, Any],
    *,
    benchmark_id: str,
    command: list[str],
    hardware: str | None = None,
    os_name: str | None = None,
    driver_runtime: str = "unknown",
    max_mojo_version: str = "unknown",
    model: str = "unknown",
    context: str = "unknown",
    concurrency: int | str = "unknown",
    quantization: str = "unknown",
    thermals: str | dict[str, Any] = "unknown",
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Create one benchmark ledger JSONL record from a measured benchmark result."""

    environment = (
        benchmark.get("environment") if isinstance(benchmark.get("environment"), dict) else {}
    )
    if isinstance(concurrency, str) and concurrency.isdigit():
        concurrency = int(concurrency)
    return {
        "version": 1,
        "record_kind": "benchmark-ledger-record",
        "benchmark_id": benchmark_id,
        "timestamp_utc": timestamp_utc
        or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "hardware": hardware or str(environment.get("machine") or "unknown"),
        "os": os_name or str(environment.get("platform") or "unknown"),
        "driver_runtime": driver_runtime,
        "max_mojo_version": max_mojo_version,
        "model": model,
        "context": context,
        "concurrency": concurrency,
        "quantization": quantization,
        "thermals": thermals,
        "command": command,
        "benchmark": benchmark,
    }


def append_benchmark_ledger_record(path: str | Path, record: dict[str, Any]) -> None:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")


def validate_benchmark_ledger_record(record: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if record.get("version") != 1:
        errors.append("version must be 1")
    if record.get("record_kind") != "benchmark-ledger-record":
        errors.append("record_kind must be benchmark-ledger-record")
    _non_empty_string(record.get("benchmark_id"), "benchmark_id", errors)
    _non_empty_string(record.get("timestamp_utc"), "timestamp_utc", errors)
    for field in REQUIRED_LEDGER_FIELDS:
        if field == "command":
            _command(record.get(field), field, errors)
        elif field == "thermals":
            if not isinstance(record.get(field), (str, dict)):
                errors.append("thermals must be a string or object")
        elif field == "concurrency":
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                errors.append("concurrency must be an integer or string")
            if isinstance(value, int) and value < 0:
                errors.append("concurrency must be non-negative")
            if isinstance(value, str) and not value:
                errors.append("concurrency string must be non-empty")
        else:
            _non_empty_string(record.get(field), field, errors)
        _unknown_warning(record, field, warnings)

    benchmark = record.get("benchmark")
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be an object")
    else:
        if benchmark.get("measured") is not True:
            errors.append("benchmark.measured must be true for ledger evidence")
        _non_empty_string(benchmark.get("source"), "benchmark.source", errors)
        result = benchmark.get("result")
        if not isinstance(result, dict):
            errors.append("benchmark.result must be an object")
        else:
            for field in REQUIRED_RESULT_FIELDS:
                value = result.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(f"benchmark.result.{field} must be numeric")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _read_ledger_records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    if isinstance(data, list):
        return data
    return [data]


def validate_benchmark_ledger(path: str | Path) -> dict[str, Any]:
    ledger_path = Path(path)
    if ledger_path.is_dir():
        ledger_path = ledger_path / "ledger.jsonl"
    if not ledger_path.exists():
        return {
            "ok": False,
            "errors": [f"missing benchmark ledger: {ledger_path}"],
            "warnings": [],
            "summary": {"record_count": 0},
            "ledger": str(ledger_path),
        }
    try:
        records = _read_ledger_records(ledger_path)
    except Exception as exc:  # noqa: BLE001 - validator reports ledger parse failures.
        return {
            "ok": False,
            "errors": [f"invalid benchmark ledger: {exc}"],
            "warnings": [],
            "summary": {"record_count": 0},
            "ledger": str(ledger_path),
        }
    errors: list[str] = []
    warnings: list[str] = []
    measured_count = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index} must be an object")
            continue
        result = validate_benchmark_ledger_record(record)
        errors.extend(f"record {index}: {error}" for error in result["errors"])
        warnings.extend(f"record {index}: {warning}" for warning in result["warnings"])
        if (
            isinstance(record.get("benchmark"), dict)
            and record["benchmark"].get("measured") is True
        ):
            measured_count += 1
    if not records:
        errors.append("ledger must contain at least one record")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "record_count": len(records),
            "measured_record_count": measured_count,
        },
        "ledger": str(ledger_path),
    }
