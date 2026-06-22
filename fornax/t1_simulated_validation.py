from __future__ import annotations

from pathlib import Path
from typing import Any

from .backend_coverage import validate_backend_coverage_contract
from .benchmark_ledger import validate_benchmark_ledger
from .continuous_batching import (
    simulate_continuous_batching,
    validate_continuous_batching_fixture,
)
from .engine_seam import validate_engine_seam_contract
from .engine_simulation import (
    simulated_engine_contract,
    validate_engine_simulation_fixture,
)
from .golden import run_golden_plans
from .inventory import build_logical_cluster_inventory, collect_local_inventory
from .io import read_json, write_json
from .moe import simulated_moe_contract, validate_moe_contract_fixture
from .model_support import (
    simulated_model_support_matrix,
    validate_model_support_matrix_fixture,
)
from .network_contract import validate_network_contract
from .observability import validate_observability_contract
from .pipeline_probe import (
    run_cpu_pipeline_correctness_probe,
    validate_pipeline_correctness_probe_fixture,
)
from .remote_expert_probe import (
    run_cpu_remote_expert_batch_probe,
    validate_remote_expert_batch_probe_fixture,
)
from .runtime_format import validate_runtime_format_golden
from .throughput_scaling import (
    simulate_throughput_scaling,
    validate_throughput_scaling_fixture,
)
from .scheduler import simulate_scheduler, validate_scheduler_contract
from .transport import (
    simulated_transport_contract,
    validate_transport_contract_fixture,
)
from .workers import simulated_worker_contract, validate_worker_contract_fixture


def _default_scheduler_plan() -> dict[str, Any]:
    return {
        "feasible": True,
        "stages": [
            {"index": 0, "layers": [0], "replicas": ["sim-gpu0"], "mode": "stage"},
            {"index": 1, "layers": [1], "replicas": ["sim-gpu1"], "mode": "stage"},
        ],
        "predicted": {
            "throughput_tok_s": 20.0,
            "per_request_latency_s": 0.25,
            "bubble_fraction": 0.12,
            "stage_effective_times_s": [0.010, 0.014],
        },
    }


def _default_requests() -> list[dict[str, Any]]:
    return [
        {"id": "t1-r0", "prompt_len": 8, "gen_len": 4},
        {"id": "t1-r1", "prompt_len": 8, "gen_len": 3},
        {"id": "t1-r2", "prompt_len": 8, "gen_len": 2},
        {"id": "t1-r3", "prompt_len": 8, "gen_len": 1},
    ]


def _golden_plans_report() -> dict[str, Any]:
    results = run_golden_plans()
    passed = sum(1 for result in results if result.passed)
    return {
        "ok": passed == len(results),
        "errors": [
            f"{result.name}: {result.message}"
            for result in results
            if not result.passed
        ],
        "warnings": [],
        "summary": {
            "passed_count": passed,
            "total_count": len(results),
            "results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "message": result.message,
                }
                for result in results
            ],
        },
    }


def _logical_cluster_report(inventory: dict[str, Any], gpu_count: int) -> dict[str, Any]:
    errors: list[str] = []
    simulation = inventory.get("simulation") if isinstance(inventory, dict) else None
    nodes = inventory.get("nodes") if isinstance(inventory, dict) else None
    links = inventory.get("links") if isinstance(inventory, dict) else None
    logical_hosts = {
        str(node.get("logical_host_id") or node.get("host_id"))
        for node in nodes or []
        if isinstance(node, dict)
    }
    if not isinstance(simulation, dict):
        errors.append("simulation must be recorded")
    else:
        if simulation.get("mode") != "logical_multi_host":
            errors.append("simulation.mode must be logical_multi_host")
        if simulation.get("physical_gpu_count") != gpu_count:
            errors.append("simulation.physical_gpu_count must match gpu_count")
        if simulation.get("logical_host_count") != len(logical_hosts):
            errors.append("simulation.logical_host_count must match logical hosts")
    if len(logical_hosts) < 2:
        errors.append("simulated cluster must contain at least two logical hosts")
    if not isinstance(links, list) or not links:
        errors.append("simulated cluster must contain declared logical links")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": [],
        "summary": {
            "logical_host_count": len(logical_hosts),
            "link_count": len(links) if isinstance(links, list) else 0,
        },
    }


def _check(name: str, tier: str, result: dict[str, Any], artifact: str) -> dict[str, Any]:
    return {
        "name": name,
        "tier": tier,
        "ok": bool(result.get("ok")),
        "artifact": artifact,
        "errors": list(result.get("errors", [])),
        "warnings": list(result.get("warnings", [])),
        "summary": result.get("summary", {}),
    }


def run_t1_simulated_validation(
    *,
    out_dir: str | Path,
    source_inventory_path: str | Path | None = None,
    gpu_count: int = 2,
    profile: str = "two-gpu-heterogeneous",
    link_bandwidth_bytes_s: float = 25.0e9,
    link_latency_s: float = 0.00025,
    slow_node_factor: float = 0.65,
    plan_id: str = "t1-simulated-plan",
    request_id: str = "req-t1-simulated",
    plan_hash: str = "sha256:t1-simulated-plan",
    max_queue_depth: int = 2,
    max_inflight: int = 2,
    microbatch_size: int = 2,
    timeout_ms: float = 50.0,
) -> dict[str, Any]:
    """Build and validate a T1 simulated bundle over two logical GPU hosts."""

    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    source_inventory = (
        read_json(source_inventory_path)
        if source_inventory_path is not None
        else collect_local_inventory()
    )
    if not isinstance(source_inventory, dict):
        raise ValueError("source inventory must contain a JSON object")

    source_inventory_path_out = bundle / "source-inventory.json"
    simulated_inventory_path = bundle / "simulated-cluster-inventory.json"
    scheduler_path = bundle / "scheduler-contract.json"
    worker_path = bundle / "worker-contract.json"
    transport_path = bundle / "transport-contract.json"
    engine_path = bundle / "engine-simulation.json"
    batching_path = bundle / "continuous-batching.json"
    pipeline_path = bundle / "pipeline-correctness.json"
    throughput_path = bundle / "throughput-scaling.json"
    moe_path = bundle / "moe-runtime.json"
    remote_expert_path = bundle / "remote-expert-batch.json"
    model_support_path = bundle / "model-support-matrix.json"
    results_path = bundle / "t1-simulated-validation.json"

    write_json(source_inventory_path_out, source_inventory)
    simulated_inventory = build_logical_cluster_inventory(
        source_inventory,
        gpu_count=gpu_count,
        profile=profile,
        link_bandwidth_bytes_s=link_bandwidth_bytes_s,
        link_latency_s=link_latency_s,
        slow_node_factor=slow_node_factor,
    )
    write_json(simulated_inventory_path, simulated_inventory)

    scheduler = simulate_scheduler(
        _default_scheduler_plan(),
        _default_requests(),
        plan_id=plan_id,
        max_queue_depth=max_queue_depth,
        max_inflight=max_inflight,
        microbatch_size=microbatch_size,
    )
    worker = simulated_worker_contract(
        plan_id=plan_id,
        request_id=request_id,
        plan_hash=plan_hash,
        max_queue_depth=max_queue_depth,
    )
    transport = simulated_transport_contract(
        plan_id=plan_id,
        request_id=request_id,
        plan_hash=plan_hash,
        max_queue_depth=max_queue_depth,
        timeout_ms=timeout_ms,
    )
    engine = simulated_engine_contract(
        plan_id=plan_id,
        request_id=request_id,
        plan_hash=plan_hash,
        max_queue_depth=max_queue_depth,
        max_inflight=max_inflight,
        microbatch_size=microbatch_size,
        timeout_ms=timeout_ms,
    )
    batching = simulate_continuous_batching(
        plan_id=plan_id,
        max_queue_depth=max_queue_depth + 2,
        max_inflight=max(max_inflight, 4),
        microbatch_size=microbatch_size,
    )
    moe = simulated_moe_contract(
        plan_id=plan_id,
        request_id=request_id,
        plan_hash=plan_hash,
    )
    remote_expert = run_cpu_remote_expert_batch_probe(
        iterations=2,
        warmup=1,
        token_count=4,
        hidden_dim=16,
        intermediate_dim=32,
        expert_id=5,
        tolerance=0.0,
    )
    model_support = simulated_model_support_matrix(
        matrix_id=f"{plan_id}-model-support",
        target_model_id="qwen3-moe-class-target",
    )
    pipeline = run_cpu_pipeline_correctness_probe(
        iterations=2,
        warmup=1,
        vocab_size=17,
        hidden_dim=16,
        new_tokens=3,
        tolerance=0.0,
    )
    throughput = simulate_throughput_scaling(
        plan_id=f"{plan_id}-throughput",
        contracted_min_concurrency=16,
        saturation_concurrency=8,
    )
    write_json(scheduler_path, scheduler)
    write_json(worker_path, worker)
    write_json(transport_path, transport)
    write_json(engine_path, engine)
    write_json(batching_path, batching)
    write_json(pipeline_path, pipeline)
    write_json(throughput_path, throughput)
    write_json(moe_path, moe)
    write_json(remote_expert_path, remote_expert)
    write_json(model_support_path, model_support)

    checks = [
        _check(
            "logical-cluster",
            "T1",
            _logical_cluster_report(simulated_inventory, gpu_count),
            str(simulated_inventory_path),
        ),
        _check(
            "golden-plans",
            "T0",
            _golden_plans_report(),
            "fornax/golden_plans",
        ),
        _check(
            "runtime-format",
            "T0/T1",
            validate_runtime_format_golden("fornax/golden_vectors/runtime_format"),
            "fornax/golden_vectors/runtime_format",
        ),
        _check(
            "network-contract",
            "T1",
            validate_network_contract("fornax/golden_vectors/network_contract"),
            "fornax/golden_vectors/network_contract",
        ),
        _check(
            "engine-seam",
            "T1",
            validate_engine_seam_contract("fornax/golden_vectors/engine_seam"),
            "fornax/golden_vectors/engine_seam",
        ),
        _check(
            "observability",
            "T1",
            validate_observability_contract("fornax/golden_vectors/observability"),
            "fornax/golden_vectors/observability",
        ),
        _check(
            "engine-simulation",
            "T1",
            validate_engine_simulation_fixture(engine),
            str(engine_path),
        ),
        _check(
            "continuous-batching",
            "T1",
            validate_continuous_batching_fixture(batching),
            str(batching_path),
        ),
        _check(
            "pipeline-correctness",
            "T1",
            validate_pipeline_correctness_probe_fixture(pipeline),
            str(pipeline_path),
        ),
        _check(
            "throughput-scaling",
            "T1",
            validate_throughput_scaling_fixture(throughput),
            str(throughput_path),
        ),
        _check(
            "moe-runtime",
            "T1",
            validate_moe_contract_fixture(moe),
            str(moe_path),
        ),
        _check(
            "remote-expert-batch",
            "T1",
            validate_remote_expert_batch_probe_fixture(remote_expert),
            str(remote_expert_path),
        ),
        _check(
            "model-support",
            "T1",
            validate_model_support_matrix_fixture(model_support),
            str(model_support_path),
        ),
        _check(
            "scheduler-contract",
            "T1",
            validate_scheduler_contract(scheduler),
            str(scheduler_path),
        ),
        _check(
            "worker-contract",
            "T1",
            validate_worker_contract_fixture(worker),
            str(worker_path),
        ),
        _check(
            "transport-contract",
            "T1",
            validate_transport_contract_fixture(transport),
            str(transport_path),
        ),
        _check(
            "backend-coverage",
            "T1",
            validate_backend_coverage_contract("fornax/golden_vectors/backend_coverage"),
            "fornax/golden_vectors/backend_coverage",
        ),
        _check(
            "benchmark-ledger",
            "T1",
            validate_benchmark_ledger("fornax/golden_vectors/benchmark_ledger"),
            "fornax/golden_vectors/benchmark_ledger",
        ),
    ]
    passed_count = sum(1 for check in checks if check["ok"])
    result = {
        "ok": passed_count == len(checks),
        "bundle": str(bundle),
        "simulation": simulated_inventory.get("simulation", {}),
        "artifacts": {
            "source_inventory": str(source_inventory_path_out),
            "simulated_cluster_inventory": str(simulated_inventory_path),
            "scheduler_contract": str(scheduler_path),
            "worker_contract": str(worker_path),
            "transport_contract": str(transport_path),
            "engine_simulation": str(engine_path),
            "continuous_batching": str(batching_path),
            "pipeline_correctness": str(pipeline_path),
            "throughput_scaling": str(throughput_path),
            "moe_runtime": str(moe_path),
            "remote_expert_batch": str(remote_expert_path),
            "model_support_matrix": str(model_support_path),
            "validation": str(results_path),
        },
        "summary": {
            "check_count": len(checks),
            "passed_count": passed_count,
            "failed_checks": [
                check["name"] for check in checks if not check["ok"]
            ],
            "logical_host_count": simulated_inventory.get("simulation", {}).get(
                "logical_host_count"
            ),
            "physical_gpu_count": simulated_inventory.get("simulation", {}).get(
                "physical_gpu_count"
            ),
        },
        "checks": checks,
        "note": (
            "T1 simulated validation bundle over local GPUs treated as logical "
            "hosts. This is development evidence only; real T3/T4 hardware "
            "validation remains required for distributed correctness and "
            "heterogeneous serving gates."
        ),
    }
    write_json(results_path, result)
    return result
