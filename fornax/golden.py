from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from typing import Any

from .io import read_json
from .planner import Inventory, ModelSpec, Target, plan_placement


@dataclass(frozen=True)
class GoldenResult:
    name: str
    passed: bool
    message: str


def _fixture_paths() -> list[Any]:
    root = importlib.resources.files("fornax.golden_plans")
    return sorted(root.iterdir(), key=lambda p: p.name)


def _check_stage(expected: dict[str, Any], actual: dict[str, Any]) -> str | None:
    if "layers" in expected and expected["layers"] != actual["layers"]:
        return f"layers expected {expected['layers']} got {actual['layers']}"
    if "replicas" in expected and expected["replicas"] != actual["replicas"]:
        return f"replicas expected {expected['replicas']} got {actual['replicas']}"
    if "mode" in expected and expected["mode"] != actual["mode"]:
        return f"mode expected {expected['mode']} got {actual['mode']}"
    return None


def check_fixture(data: dict[str, Any]) -> GoldenResult:
    name = str(data.get("name", "<unnamed>"))
    plan_options = data.get("plan_options", {})
    plan = plan_placement(
        ModelSpec.from_dict(data["model"]),
        Inventory.from_dict(data["inventory"]),
        Target.from_dict(data["target"]),
        min_stages=plan_options.get("min_stages"),
        max_stages=plan_options.get("max_stages"),
    )
    expected = data["expected"]
    if bool(expected["feasible"]) != plan.feasible:
        return GoldenResult(name, False, f"feasible expected {expected['feasible']} got {plan.feasible}")
    if not plan.feasible:
        needle = expected.get("reason_contains")
        reason = plan.infeasible_reason or ""
        if needle and needle not in reason:
            return GoldenResult(name, False, f"reason missing {needle!r}: {reason}")
        return GoldenResult(name, True, "infeasible as expected")

    plan_dict = plan.to_dict()
    if expected.get("stage_count") != len(plan.stages):
        return GoldenResult(
            name,
            False,
            f"stage_count expected {expected.get('stage_count')} got {len(plan.stages)}",
        )
    for idx, expected_stage in enumerate(expected.get("stages", [])):
        err = _check_stage(expected_stage, plan_dict["stages"][idx])
        if err:
            return GoldenResult(name, False, f"stage {idx}: {err}")
    predicted = plan.predicted
    assert predicted is not None
    if "bottleneck_stage" in expected and expected["bottleneck_stage"] != predicted.bottleneck_stage:
        return GoldenResult(
            name,
            False,
            f"bottleneck expected {expected['bottleneck_stage']} got {predicted.bottleneck_stage}",
        )
    if predicted.throughput_tok_s < float(expected.get("throughput_tok_s_min", 0.0)):
        return GoldenResult(
            name,
            False,
            f"throughput below minimum: {predicted.throughput_tok_s}",
        )
    return GoldenResult(name, True, "ok")


def run_golden_plans() -> list[GoldenResult]:
    results: list[GoldenResult] = []
    for path in _fixture_paths():
        if path.name.endswith(".json"):
            with importlib.resources.as_file(path) as local_path:
                results.append(check_fixture(read_json(local_path)))
    return results
