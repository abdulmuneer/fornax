from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .benchmark import DEFAULT_MODE, benchmark_from_plan
from .contracts import load_target_contract
from .doctor import inspect_phase0_bundle
from .inventory import collect_local_inventory, probe_declared_links
from .io import load_inventory, write_json
from .planner import plan_placement
from .simulate import simulation_result, summarize_request_trace
from .validation import validate_target_contract


def _copy_target_contract(target_path: Path, out_dir: Path) -> Path:
    name = "v0-target-contract.md" if target_path.suffix.lower() == ".md" else "target.json"
    copied = out_dir / name
    if target_path.resolve() == copied.resolve():
        return copied
    shutil.copyfile(target_path, copied)
    return copied


def run_phase0_preflight(
    *,
    target_path: str | Path,
    out_dir: str | Path,
    requests_path: str | Path | None = None,
    benchmark_mode: str = DEFAULT_MODE,
    benchmark_iterations: int = 25,
    inventory_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the minimal Phase-0 evidence workflow and write a doctorable bundle."""

    target_file = Path(target_path)
    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    target_artifact = _copy_target_contract(target_file, bundle)
    inventory_data = inventory_data if inventory_data is not None else collect_local_inventory()
    inventory_path = bundle / "inventory.json"
    links_path = bundle / "links.json"
    placement_path = bundle / "placement.json"
    validate_path = bundle / "validate.json"
    simulate_path = bundle / "simulate.json"
    benchmark_path = bundle / "benchmark.json"
    doctor_path = bundle / "doctor.json"

    write_json(inventory_path, inventory_data)
    link_data = probe_declared_links(inventory_data)
    write_json(links_path, link_data)

    model, target, contract_bundle = load_target_contract(target_artifact)
    inventory = load_inventory(inventory_path, links_path)
    plan = plan_placement(model, inventory, target)
    placement_data = plan.to_dict()
    write_json(placement_path, placement_data)

    validation = validate_target_contract(
        model, target, contract_bundle, inventory, plan=plan
    )
    write_json(validate_path, validation)

    request_trace = summarize_request_trace(requests_path) if requests_path else None
    if placement_data.get("predicted") is not None:
        simulate = simulation_result(placement_data["predicted"], request_trace)
    else:
        simulate = {
            "predicted": None,
            "error": placement_data.get("infeasible_reason", "placement is infeasible"),
        }
        if request_trace is not None:
            simulate["requests"] = request_trace
    write_json(simulate_path, simulate)

    try:
        benchmark = benchmark_from_plan(
            placement_data, mode=benchmark_mode, iterations=benchmark_iterations
        )
    except ValueError as exc:
        benchmark = {
            "mode": benchmark_mode,
            "measured": False,
            "error": str(exc),
        }
    write_json(benchmark_path, benchmark)

    doctor = inspect_phase0_bundle(bundle)
    write_json(doctor_path, doctor)
    return {
        "ok": bool(doctor["ok"]) and bool(validation["valid"]) and bool(plan.feasible),
        "bundle": str(bundle),
        "target_artifact": str(target_artifact),
        "artifacts": {
            "inventory": str(inventory_path),
            "links": str(links_path),
            "placement": str(placement_path),
            "validate": str(validate_path),
            "simulate": str(simulate_path),
            "benchmark": str(benchmark_path),
            "doctor": str(doctor_path),
        },
        "doctor": doctor,
    }
