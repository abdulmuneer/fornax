from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark import DEFAULT_MODE
from .inventory import build_logical_cluster_inventory, collect_local_inventory
from .io import read_json, write_json
from .phase0_status import render_phase0_status_report
from .preflight import run_phase0_preflight


def run_phase0_simulated_validation(
    *,
    target_path: str | Path,
    out_dir: str | Path,
    source_inventory_path: str | Path | None = None,
    gpu_count: int = 2,
    profile: str = "two-gpu-heterogeneous",
    link_bandwidth_bytes_s: float = 25.0e9,
    link_latency_s: float = 0.00025,
    slow_node_factor: float = 0.65,
    requests_path: str | Path | None = None,
    benchmark_mode: str = DEFAULT_MODE,
    benchmark_iterations: int = 25,
    include_calibration: bool = False,
    calibration_torch_python: str | None = None,
    program_report_date: str | None = None,
    program_plan_version: str = "v3",
    substrate_pinned_build: str = "unset",
    kickoff_date: str | None = None,
    ker_status: str = "unassigned",
    scope: str = "pending",
    simulated_apple_role: str = "capacity-only",
    simulated_apple_reason: str | None = None,
) -> dict[str, Any]:
    """Build a local logical cluster and run the full simulated Phase-0 bundle."""

    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    source_inventory = (
        read_json(source_inventory_path)
        if source_inventory_path is not None
        else collect_local_inventory()
    )
    if not isinstance(source_inventory, dict):
        raise ValueError("source inventory must contain a JSON object")

    source_inventory_artifact = bundle / "source-inventory.json"
    simulated_inventory_artifact = bundle / "simulated-cluster-inventory.json"
    write_json(source_inventory_artifact, source_inventory)

    simulated_inventory = build_logical_cluster_inventory(
        source_inventory,
        gpu_count=gpu_count,
        profile=profile,
        link_bandwidth_bytes_s=link_bandwidth_bytes_s,
        link_latency_s=link_latency_s,
        slow_node_factor=slow_node_factor,
    )
    write_json(simulated_inventory_artifact, simulated_inventory)

    preflight = run_phase0_preflight(
        target_path=target_path,
        out_dir=bundle,
        requests_path=requests_path,
        benchmark_mode=benchmark_mode,
        benchmark_iterations=benchmark_iterations,
        inventory_data=simulated_inventory,
        include_g1_drafts=True,
        substrate_pinned_build=substrate_pinned_build,
        kickoff_date=kickoff_date,
        ker_status=ker_status,
        scope=scope,
        include_calibration=include_calibration,
        calibration_torch_python=calibration_torch_python,
        include_golden_plans=True,
        include_program_reports=True,
        program_report_date=program_report_date,
        program_plan_version=program_plan_version,
        include_simulated_apple_evidence=True,
        simulated_apple_role=simulated_apple_role,
        simulated_apple_reason=simulated_apple_reason,
    )

    status_path = bundle / "phase0-status.json"
    status = (
        read_json(status_path)
        if status_path.exists()
        else render_phase0_status_report(
            bundle,
            report_date=program_report_date,
            plan_version=program_plan_version,
        )
    )
    if not isinstance(status, dict):
        raise ValueError("phase0 status report must contain a JSON object")

    artifacts = dict(preflight.get("artifacts", {}))
    artifacts.update(
        {
            "source_inventory": str(source_inventory_artifact),
            "simulated_cluster_inventory": str(simulated_inventory_artifact),
        }
    )

    return {
        "ok": bool(preflight.get("ok")),
        "bundle": str(bundle),
        "artifacts": artifacts,
        "preflight": preflight,
        "summary": status.get("summary", {}),
        "g1": status.get("g1", {}),
        "simulation": status.get("simulation", {}),
        "apple_simulation": status.get("apple_simulation", {}),
        "status": status,
    }
