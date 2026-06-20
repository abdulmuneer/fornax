from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .apple_probe import apple_probe_template
from .benchmark import DEFAULT_MODE, benchmark_from_plan
from .calibration import run_local_calibration
from .contracts import load_target_contract
from .doctor import inspect_phase0_bundle
from .inventory import collect_local_inventory, probe_declared_links
from .io import load_inventory, write_json
from .network_security_spec import render_network_security_spec_draft
from .planner import plan_placement
from .program_rebaseline import render_program_rebaseline_draft
from .runtime_format_spec import render_runtime_format_spec_draft
from .substrate_adr import render_substrate_adr_draft
from .simulate import simulation_result, summarize_request_trace
from .validation import validate_target_contract


def _copy_target_contract(target_path: Path, out_dir: Path) -> Path:
    name = "v0-target-contract.md" if target_path.suffix.lower() == ".md" else "target.json"
    copied = out_dir / name
    if target_path.resolve() == copied.resolve():
        return copied
    shutil.copyfile(target_path, copied)
    return copied


def _target_model_name(contract_bundle: dict[str, Any]) -> str:
    model = contract_bundle.get("model") if isinstance(contract_bundle, dict) else None
    if isinstance(model, dict):
        for key in ("name", "family", "model_id"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return "target-model"


def _throughput_threshold(contract_bundle: dict[str, Any]) -> float:
    contract = (
        contract_bundle.get("contract") if isinstance(contract_bundle, dict) else None
    )
    if isinstance(contract, dict):
        value = contract.get("throughput_threshold_tok_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return 1.0


def _write_g1_drafts(
    *,
    bundle: Path,
    contract_bundle: dict[str, Any],
    substrate_pinned_build: str,
    kickoff_date: str | None,
    ker_status: str,
    scope: str,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}

    runtime = render_runtime_format_spec_draft("fornax/golden_vectors/runtime_format")
    runtime_path = bundle / "runtime-format-and-invariants.md"
    runtime_path.write_text(runtime["markdown"], encoding="utf-8")
    artifacts["runtime_format_spec"] = str(runtime_path)

    network = render_network_security_spec_draft("fornax/golden_vectors/network_contract")
    network_path = bundle / "networking-security-and-backpressure.md"
    network_path.write_text(network["markdown"], encoding="utf-8")
    artifacts["network_security_spec"] = str(network_path)

    adr_dir = bundle / "adr"
    adr_dir.mkdir(exist_ok=True)
    substrate = render_substrate_adr_draft(pinned_build=substrate_pinned_build)
    substrate_path = adr_dir / "0001-max-mojo-substrate.md"
    substrate_path.write_text(substrate["markdown"], encoding="utf-8")
    artifacts["substrate_adr"] = str(substrate_path)

    apple_probe_path = bundle / "apple-expert-mlp-probe.json"
    write_json(
        apple_probe_path,
        apple_probe_template(
            target_model=_target_model_name(contract_bundle),
            pinned_build=substrate_pinned_build,
            threshold_tokens_s=_throughput_threshold(contract_bundle),
        ),
    )
    artifacts["apple_probe"] = str(apple_probe_path)

    rebaseline = render_program_rebaseline_draft(
        kickoff_date=kickoff_date,
        ker_status=ker_status,
        scope=scope,
    )
    rebaseline_path = bundle / "roadmap-staffing-rebaseline.md"
    rebaseline_path.write_text(rebaseline["markdown"], encoding="utf-8")
    artifacts["program_rebaseline"] = str(rebaseline_path)
    return artifacts


def run_phase0_preflight(
    *,
    target_path: str | Path,
    out_dir: str | Path,
    requests_path: str | Path | None = None,
    benchmark_mode: str = DEFAULT_MODE,
    benchmark_iterations: int = 25,
    inventory_data: dict[str, Any] | None = None,
    include_g1_drafts: bool = False,
    substrate_pinned_build: str = "unset",
    kickoff_date: str | None = None,
    ker_status: str = "unassigned",
    scope: str = "pending",
    include_calibration: bool = False,
    calibration_torch_python: str | None = None,
    active_local_links: bool = False,
    fabric_torch_python: str | None = None,
    active_local_link_bytes: int = 16 * 1024 * 1024,
    active_local_link_iterations: int = 4,
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
    calibration_path = bundle / "calibration.json"
    generated_g1_artifacts: dict[str, str] = {}

    write_json(inventory_path, inventory_data)
    link_data = probe_declared_links(
        inventory_data,
        active_local=active_local_links,
        torch_python=fabric_torch_python,
        active_local_bytes=active_local_link_bytes,
        active_local_iterations=active_local_link_iterations,
    )
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

    if include_g1_drafts:
        generated_g1_artifacts = _write_g1_drafts(
            bundle=bundle,
            contract_bundle=contract_bundle,
            substrate_pinned_build=substrate_pinned_build,
            kickoff_date=kickoff_date,
            ker_status=ker_status,
            scope=scope,
        )

    if include_calibration:
        calibration = run_local_calibration(torch_python=calibration_torch_python)
        write_json(calibration_path, calibration)

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
            **({"calibration": str(calibration_path)} if include_calibration else {}),
            **generated_g1_artifacts,
        },
        "doctor": doctor,
    }
