from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_json
from .planner import Inventory, ModelSpec, Target, plan_placement
from .simulate import simulation_result
from .validation import validate_target_contract


QUICKSTART_TARGET: dict[str, Any] = {
    "model": {
        "hidden_dim": 1024,
        "num_layers": 4,
        "dtype_weight": "q4",
        "dtype_activation": "fp16",
        "layers": [
            {
                "kind": "dense",
                "weight_bytes": 1_000_000,
                "active_flops_per_token": 1_000_000,
            }
            for _ in range(4)
        ],
    },
    "target": {
        "concurrency": 4,
        "prompt_len": 16,
        "gen_len": 8,
        "objective": "balanced",
    },
    "contract": {
        "seed_target_rationale": (
            "Small deterministic teaching fixture that forces a two-stage "
            "heterogeneous placement."
        ),
        "throughput_threshold_tok_s": 1_000.0,
        "memory_headroom_fraction_min": 0.05,
        "concurrency_sweep": [1, 4, 8],
        "persona_min_concurrency": 4,
        "persona_can_supply_concurrency": True,
        "baselines": [
            {"name": "single-node-fit", "status": "infeasible-by-fixture-design"},
            {"name": "naive-sequential-pipeline", "status": "fixture-only"},
        ],
        "kill_metric": "The teaching fixture is invalid if no two-stage plan is feasible.",
    },
    "evidence": {
        "class": "simulation_fixture",
        "physical_measurement": False,
        "warning": (
            "Synthetic teaching inputs and cost-model predictions; this fixture "
            "does not establish physical hardware support or performance."
        ),
    },
}


QUICKSTART_INVENTORY: dict[str, Any] = {
    "nodes": [
        {
            "id": "nvidia-node",
            "vendor": "nvidia",
            "runtime": "max",
            "mem_free_bytes": 3_000_000,
            "compute_class": 4_000_000_000_000.0,
            "mem_bandwidth_bytes_s": 400_000_000_000.0,
            "supports_stage": True,
            "supports_expert_worker": True,
            "supports_kv": True,
            "supported_dtypes": ["fp16"],
        },
        {
            "id": "apple-node",
            "vendor": "apple",
            "runtime": "max",
            "mem_free_bytes": 3_000_000,
            "compute_class": 1_000_000_000_000.0,
            "mem_bandwidth_bytes_s": 100_000_000_000.0,
            "supports_stage": True,
            "supports_expert_worker": True,
            "supports_kv": True,
            "supported_dtypes": ["fp16"],
        },
    ],
    "links": [
        {
            "a": "nvidia-node",
            "b": "apple-node",
            "bandwidth_bytes_s": 12_500_000_000.0,
            "latency_s": 0.00002,
        }
    ],
    "evidence": {
        "class": "simulation_fixture",
        "physical_measurement": False,
        "warning": "Synthetic capacities, compute rates, and link properties.",
    },
}


def run_quickstart(out_dir: str | Path) -> dict[str, Any]:
    """Run a deterministic, no-hardware tour of the Fornax planning loop."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = ModelSpec.from_dict(QUICKSTART_TARGET["model"])
    target = Target.from_dict(QUICKSTART_TARGET["target"])
    inventory = Inventory.from_dict(QUICKSTART_INVENTORY)
    plan = plan_placement(model, inventory, target)
    plan_data = plan.to_dict()
    validation = validate_target_contract(
        model,
        target,
        QUICKSTART_TARGET,
        inventory,
        plan=plan,
    )
    simulation = (
        simulation_result(plan_data["predicted"], None)
        if plan_data["predicted"] is not None
        else {
            "predicted": None,
            "error": plan_data.get("infeasible_reason", "placement is infeasible"),
        }
    )

    target_path = output / "target.json"
    inventory_path = output / "inventory.json"
    plan_path = output / "placement.json"
    simulation_path = output / "simulation.json"
    validation_path = output / "validation.json"
    summary_path = output / "summary.json"

    write_json(target_path, QUICKSTART_TARGET)
    write_json(inventory_path, QUICKSTART_INVENTORY)
    write_json(plan_path, plan_data)
    write_json(simulation_path, simulation)
    write_json(validation_path, validation)

    predicted = plan_data.get("predicted") or {}
    summary = {
        "schema_version": 1,
        "result": "ok" if plan.feasible and validation["valid"] else "failed",
        "evidence_class": "simulation_fixture",
        "physical_measurement": False,
        "warning": QUICKSTART_TARGET["evidence"]["warning"],
        "feasible": plan.feasible,
        "contract_valid": validation["valid"],
        "stage_count": len(plan.stages),
        "stages": [
            {
                "index": stage.index,
                "node": stage.replicas[0],
                "layers": list(stage.layers),
            }
            for stage in plan.stages
        ],
        "predicted": predicted,
        "artifacts": {
            "target": str(target_path),
            "inventory": str(inventory_path),
            "placement": str(plan_path),
            "simulation": str(simulation_path),
            "validation": str(validation_path),
            "summary": str(summary_path),
        },
        "next_commands": [
            f"python3 -m fornax simulate --plan {plan_path}",
            (
                "python3 -m fornax target validate "
                f"{target_path} --inventory {inventory_path}"
            ),
        ],
    }
    write_json(summary_path, summary)
    return summary
