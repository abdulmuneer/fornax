from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


RECORD_KIND = "throughput-scaling-simulation-contract"
MODE = "t1-simulation"
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]


def _positive_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")
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


def parse_concurrency_levels(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return list(DEFAULT_CONCURRENCY_LEVELS)
    levels: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            level = int(item)
        except ValueError as exc:
            raise ValueError("concurrency levels must be comma-separated integers") from exc
        if level <= 0:
            raise ValueError("concurrency levels must be positive")
        levels.append(level)
    if not levels:
        raise ValueError("concurrency levels must be non-empty")
    return levels


def _validate_levels(levels: list[int]) -> list[int]:
    if not isinstance(levels, list) or not levels:
        raise ValueError("concurrency_levels must be a non-empty list")
    normalized: list[int] = []
    previous = 0
    for index, level in enumerate(levels):
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError(f"concurrency_levels[{index}] must be a positive integer")
        if level <= previous:
            raise ValueError("concurrency_levels must be strictly increasing")
        normalized.append(level)
        previous = level
    return normalized


def simulate_throughput_scaling(
    *,
    plan_id: str = "throughput-scaling-plan",
    concurrency_levels: list[int] | None = None,
    contracted_min_concurrency: int = 16,
    saturation_concurrency: int = 8,
    planner_bound_fraction: float = 0.20,
    throughput_efficiency_floor: float = 0.60,
    sum_node_ideal_tokens_s: float = 45.0,
    saturated_pipeline_tokens_s: float = 30.0,
    planner_bias_fraction: float = 0.08,
    jitter_fraction: float = 0.015,
) -> dict[str, Any]:
    """Simulate a concurrency sweep and planner-accuracy check.

    The output is T1 evidence: deterministic simulation of the metric contract,
    not real T3/T4 hardware evidence.
    """

    if not plan_id:
        raise ValueError("plan_id must be non-empty")
    levels = _validate_levels(list(DEFAULT_CONCURRENCY_LEVELS if concurrency_levels is None else concurrency_levels))
    errors: list[str] = []
    _positive_int(contracted_min_concurrency, "contracted_min_concurrency", errors)
    _positive_int(saturation_concurrency, "saturation_concurrency", errors)
    _positive_number(planner_bound_fraction, "planner_bound_fraction", errors)
    _positive_number(throughput_efficiency_floor, "throughput_efficiency_floor", errors)
    _positive_number(sum_node_ideal_tokens_s, "sum_node_ideal_tokens_s", errors)
    _positive_number(saturated_pipeline_tokens_s, "saturated_pipeline_tokens_s", errors)
    _non_negative_number(planner_bias_fraction, "planner_bias_fraction", errors)
    _non_negative_number(jitter_fraction, "jitter_fraction", errors)
    if errors:
        raise ValueError("; ".join(errors))
    if throughput_efficiency_floor >= 1.0:
        raise ValueError("throughput_efficiency_floor must be below 1.0")
    if planner_bias_fraction >= planner_bound_fraction:
        raise ValueError("planner_bias_fraction must be below planner_bound_fraction")
    if jitter_fraction >= planner_bound_fraction:
        raise ValueError("jitter_fraction must be below planner_bound_fraction")
    if contracted_min_concurrency not in levels:
        raise ValueError("contracted_min_concurrency must appear in concurrency_levels")
    if saturation_concurrency not in levels:
        raise ValueError("saturation_concurrency must appear in concurrency_levels")

    rows: list[dict[str, Any]] = []
    for index, concurrency in enumerate(levels):
        utilization = min(1.0, concurrency / float(saturation_concurrency))
        # Deterministic, small jitter makes the sweep realistic while remaining
        # safely inside the planner-accuracy bound.
        jitter = 1.0 - (jitter_fraction if index % 2 and concurrency < saturation_concurrency else 0.0)
        measured_tokens_s = saturated_pipeline_tokens_s * utilization * jitter
        predicted_tokens_s = measured_tokens_s * (1.0 + planner_bias_fraction)
        planner_error_fraction = abs(predicted_tokens_s - measured_tokens_s) / measured_tokens_s
        efficiency_fraction = measured_tokens_s / sum_node_ideal_tokens_s
        bubble_fraction = max(0.0, round(1.0 - utilization, 9))
        rows.append(
            {
                "concurrency": concurrency,
                "predicted_tokens_s": round(predicted_tokens_s, 9),
                "measured_tokens_s": round(measured_tokens_s, 9),
                "planner_error_fraction": round(planner_error_fraction, 9),
                "throughput_efficiency_fraction": round(efficiency_fraction, 9),
                "pipeline_utilization_fraction": round(utilization, 9),
                "bubble_fraction": bubble_fraction,
                "simulation_source": "deterministic_t1_concurrency_sweep",
            }
        )

    max_measured = max(row["measured_tokens_s"] for row in rows)
    saturation_threshold = 0.95 * max_measured
    observed_saturation = next(
        row["concurrency"] for row in rows if row["measured_tokens_s"] >= saturation_threshold
    )
    contract_row = next(row for row in rows if row["concurrency"] == contracted_min_concurrency)
    max_error = max(row["planner_error_fraction"] for row in rows)
    monotonic = all(
        rows[index]["measured_tokens_s"] <= rows[index + 1]["measured_tokens_s"]
        for index in range(len(rows) - 1)
    )
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "measured": True,
        "measurement_kind": "deterministic-simulation",
        "concurrency_levels": levels,
        "contracted_min_concurrency": contracted_min_concurrency,
        "planner_bound_fraction": planner_bound_fraction,
        "throughput_efficiency_floor": throughput_efficiency_floor,
        "sum_node_ideal_tokens_s": sum_node_ideal_tokens_s,
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "monotonic_scaling": monotonic,
            "observed_saturation_concurrency": observed_saturation,
            "saturation_within_contract": observed_saturation <= contracted_min_concurrency,
            "throughput_efficiency_at_contract": contract_row["throughput_efficiency_fraction"],
            "throughput_efficiency_passed": contract_row["throughput_efficiency_fraction"] >= throughput_efficiency_floor,
            "max_abs_planner_error_fraction": round(max_error, 9),
            "planner_accuracy_passed": max_error <= planner_bound_fraction,
            "contracted_measured_tokens_s": contract_row["measured_tokens_s"],
            "contracted_predicted_tokens_s": contract_row["predicted_tokens_s"],
            "target_met": (
                monotonic
                and observed_saturation <= contracted_min_concurrency
                and contract_row["throughput_efficiency_fraction"] >= throughput_efficiency_floor
                and max_error <= planner_bound_fraction
            ),
        },
        "note": (
            "T1 throughput-scaling simulation: validates concurrency scaling, "
            "planner-accuracy bounds, and provisional throughput-efficiency math. "
            "Not T3 hardware evidence."
        ),
    }


def validate_throughput_scaling_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    _non_empty_string(data.get("plan_id"), "plan_id", errors)
    if data.get("measured") is not True:
        errors.append("measured must be true")
    measurement_kind = _non_empty_string(data.get("measurement_kind"), "measurement_kind", errors)
    if measurement_kind == "deterministic-simulation":
        warnings.append("throughput scaling is simulation evidence, not T3 hardware evidence")
    contracted_min = _positive_int(data.get("contracted_min_concurrency"), "contracted_min_concurrency", errors)
    planner_bound = _positive_number(data.get("planner_bound_fraction"), "planner_bound_fraction", errors)
    efficiency_floor = _positive_number(data.get("throughput_efficiency_floor"), "throughput_efficiency_floor", errors)
    sum_ideal = _positive_number(data.get("sum_node_ideal_tokens_s"), "sum_node_ideal_tokens_s", errors)
    levels_raw = data.get("concurrency_levels")
    levels: list[int] = []
    if not isinstance(levels_raw, list) or not levels_raw:
        errors.append("concurrency_levels must be a non-empty list")
    else:
        previous = 0
        for index, level in enumerate(levels_raw):
            parsed = _positive_int(level, f"concurrency_levels[{index}]", errors)
            if parsed is None:
                continue
            if parsed <= previous:
                errors.append("concurrency_levels must be strictly increasing")
            levels.append(parsed)
            previous = parsed
    if contracted_min is not None and levels and contracted_min not in levels:
        errors.append("contracted_min_concurrency must appear in concurrency_levels")

    rows_raw = data.get("rows")
    rows: list[dict[str, Any]] = []
    if not isinstance(rows_raw, list) or not rows_raw:
        errors.append("rows must be a non-empty list")
    else:
        if levels and len(rows_raw) != len(levels):
            errors.append("rows length must match concurrency_levels length")
        for index, row in enumerate(rows_raw):
            field = f"rows[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{field} must be an object")
                continue
            concurrency = _positive_int(row.get("concurrency"), f"{field}.concurrency", errors)
            predicted = _positive_number(row.get("predicted_tokens_s"), f"{field}.predicted_tokens_s", errors)
            measured = _positive_number(row.get("measured_tokens_s"), f"{field}.measured_tokens_s", errors)
            error_fraction = _non_negative_number(row.get("planner_error_fraction"), f"{field}.planner_error_fraction", errors)
            efficiency = _non_negative_number(row.get("throughput_efficiency_fraction"), f"{field}.throughput_efficiency_fraction", errors)
            utilization = _non_negative_number(row.get("pipeline_utilization_fraction"), f"{field}.pipeline_utilization_fraction", errors)
            bubble = _non_negative_number(row.get("bubble_fraction"), f"{field}.bubble_fraction", errors)
            _non_empty_string(row.get("simulation_source"), f"{field}.simulation_source", errors)
            if levels and concurrency is not None and index < len(levels) and concurrency != levels[index]:
                errors.append(f"{field}.concurrency must match concurrency_levels[{index}]")
            if predicted is not None and measured is not None and error_fraction is not None:
                expected = abs(predicted - measured) / measured
                if abs(expected - error_fraction) > 1e-6:
                    errors.append(f"{field}.planner_error_fraction does not match predicted/measured")
            if measured is not None and sum_ideal is not None and efficiency is not None:
                expected_efficiency = measured / sum_ideal
                if abs(expected_efficiency - efficiency) > 1e-6:
                    errors.append(f"{field}.throughput_efficiency_fraction does not match measured/sum ideal")
            if utilization is not None and utilization > 1.0:
                errors.append(f"{field}.pipeline_utilization_fraction must be <= 1.0")
            if bubble is not None and bubble >= 1.0:
                errors.append(f"{field}.bubble_fraction must be in [0, 1)")
            rows.append(row)

    measured_values = [row.get("measured_tokens_s") for row in rows if isinstance(row.get("measured_tokens_s"), (int, float))]
    monotonic = all(
        float(measured_values[index]) <= float(measured_values[index + 1])
        for index in range(len(measured_values) - 1)
    ) if measured_values else False
    max_error = max(
        [float(row.get("planner_error_fraction")) for row in rows if isinstance(row.get("planner_error_fraction"), (int, float))]
        or [0.0]
    )
    observed_saturation = None
    if measured_values and levels:
        threshold = 0.95 * max(float(value) for value in measured_values)
        for level, value in zip(levels, measured_values):
            if float(value) >= threshold:
                observed_saturation = level
                break
    contract_row = next(
        (row for row in rows if row.get("concurrency") == contracted_min), None
    ) if contracted_min is not None else None
    contract_efficiency = None
    if isinstance(contract_row, dict) and isinstance(contract_row.get("throughput_efficiency_fraction"), (int, float)):
        contract_efficiency = float(contract_row["throughput_efficiency_fraction"])

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    else:
        if summary.get("row_count") != len(rows):
            errors.append("summary.row_count must match rows length")
        if summary.get("monotonic_scaling") != monotonic:
            errors.append("summary.monotonic_scaling must match rows")
        if observed_saturation is not None and summary.get("observed_saturation_concurrency") != observed_saturation:
            errors.append("summary.observed_saturation_concurrency must match rows")
        if contracted_min is not None and observed_saturation is not None:
            expected = observed_saturation <= contracted_min
            if summary.get("saturation_within_contract") != expected:
                errors.append("summary.saturation_within_contract must match observed saturation")
        if contract_efficiency is not None:
            if abs(float(summary.get("throughput_efficiency_at_contract", -1.0)) - contract_efficiency) > 1e-6:
                errors.append("summary.throughput_efficiency_at_contract must match contract row")
            if efficiency_floor is not None and summary.get("throughput_efficiency_passed") != (contract_efficiency >= efficiency_floor):
                errors.append("summary.throughput_efficiency_passed must match floor")
        if abs(float(summary.get("max_abs_planner_error_fraction", -1.0)) - max_error) > 1e-6:
            errors.append("summary.max_abs_planner_error_fraction must match rows")
        if planner_bound is not None and summary.get("planner_accuracy_passed") != (max_error <= planner_bound):
            errors.append("summary.planner_accuracy_passed must match planner bound")
        expected_target = (
            monotonic
            and observed_saturation is not None
            and contracted_min is not None
            and observed_saturation <= contracted_min
            and contract_efficiency is not None
            and efficiency_floor is not None
            and contract_efficiency >= efficiency_floor
            and planner_bound is not None
            and max_error <= planner_bound
        )
        if summary.get("target_met") != expected_target:
            errors.append("summary.target_met must match metric gates")
        if summary.get("target_met") is not True:
            errors.append("summary.target_met must be true")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "row_count": len(rows),
            "contracted_min_concurrency": contracted_min,
            "observed_saturation_concurrency": observed_saturation,
            "max_abs_planner_error_fraction": max_error,
            "throughput_efficiency_at_contract": contract_efficiency,
            "target_met": summary.get("target_met") if isinstance(summary, dict) else False,
        },
    }


def validate_throughput_scaling(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid throughput-scaling artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["throughput-scaling artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_throughput_scaling_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
