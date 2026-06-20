from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .planner import Inventory, ModelSpec, PlacementPlan, Target, plan_placement
from .planner.cost import stage_memory_bytes


@dataclass(frozen=True)
class ContractCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _contract(bundle: dict[str, Any]) -> dict[str, Any]:
    value = bundle.get("contract")
    return value if isinstance(value, dict) else {}


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        result.append(item)
    return result


def _baselines_present(value: Any) -> bool:
    if isinstance(value, list):
        return any(
            isinstance(item, dict) and _nonempty_string(item.get("name"))
            for item in value
        )
    if isinstance(value, dict):
        return any(_nonempty_string(str(key)) for key in value.keys())
    return False


def _minimum_memory_headroom(
    model: ModelSpec, target: Target, inventory: Inventory, plan: PlacementPlan
) -> tuple[float | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if not plan.feasible:
        return None, rows
    for stage in plan.stages:
        for node_id in stage.replicas:
            node = inventory.node(node_id)
            used = stage_memory_bytes(model, stage.layers, target, stage.mode)
            headroom = (node.mem_free_bytes - used) / node.mem_free_bytes
            rows.append(
                {
                    "stage": stage.index,
                    "node_id": node_id,
                    "mode": stage.mode,
                    "memory_used_bytes": int(used),
                    "memory_free_bytes": node.mem_free_bytes,
                    "headroom_fraction": headroom,
                }
            )
    if not rows:
        return None, rows
    return min(row["headroom_fraction"] for row in rows), rows


def validate_target_contract(
    model: ModelSpec,
    target: Target,
    bundle: dict[str, Any],
    inventory: Inventory,
    *,
    plan: PlacementPlan | None = None,
) -> dict[str, Any]:
    """Validate executable G1-facing pieces of a v0 target contract.

    This does not sign off G1. It checks that the machine-readable contract has
    enough structure for review and that the current planner prediction satisfies
    the declared threshold and memory-headroom gates.
    """

    contract = _contract(bundle)
    checks: list[ContractCheck] = []

    checks.append(
        ContractCheck(
            "contract.metadata_present",
            bool(contract),
            (
                "top-level contract object present"
                if contract
                else "missing top-level contract object"
            ),
        )
    )
    checks.append(
        ContractCheck(
            "contract.seed_target_rationale",
            _nonempty_string(contract.get("seed_target_rationale")),
            "seed acceptance/replacement rationale present",
        )
    )
    checks.append(
        ContractCheck(
            "contract.kill_metric",
            _nonempty_string(contract.get("kill_metric")),
            "kill metric present",
        )
    )
    checks.append(
        ContractCheck(
            "contract.baselines",
            _baselines_present(contract.get("baselines")),
            (
                "at least one baseline named"
                if _baselines_present(contract.get("baselines"))
                else "missing named baselines"
            ),
        )
    )

    throughput_threshold = _number(contract.get("throughput_threshold_tok_s"))
    checks.append(
        ContractCheck(
            "contract.throughput_threshold",
            throughput_threshold is not None and throughput_threshold > 0,
            (
                f"throughput threshold={throughput_threshold}"
                if throughput_threshold is not None
                else "missing positive throughput_threshold_tok_s"
            ),
        )
    )
    memory_headroom_min = _number(contract.get("memory_headroom_fraction_min"))
    checks.append(
        ContractCheck(
            "contract.memory_headroom_threshold",
            memory_headroom_min is not None and 0 <= memory_headroom_min < 1,
            (
                f"memory headroom threshold={memory_headroom_min}"
                if memory_headroom_min is not None
                else "missing memory_headroom_fraction_min in [0,1)"
            ),
        )
    )

    sweep = _int_list(contract.get("concurrency_sweep"))
    checks.append(
        ContractCheck(
            "contract.concurrency_sweep",
            sweep is not None and target.concurrency in sweep,
            (
                f"sweep={sweep}, target concurrency={target.concurrency}"
                if sweep is not None
                else "missing non-empty integer concurrency_sweep"
            ),
        )
    )
    persona_min = contract.get("persona_min_concurrency")
    persona_ok = isinstance(persona_min, int) and not isinstance(persona_min, bool)
    persona_can_supply = bool(contract.get("persona_can_supply_concurrency", False))
    checks.append(
        ContractCheck(
            "contract.persona_concurrency",
            persona_ok and persona_can_supply and target.concurrency >= int(persona_min),
            (
                f"persona_min={persona_min}, persona_can_supply={persona_can_supply}, target={target.concurrency}"
                if persona_ok
                else "missing integer persona_min_concurrency"
            ),
        )
    )

    if plan is None:
        plan = plan_placement(model, inventory, target)
    checks.append(
        ContractCheck(
            "planner.feasible",
            plan.feasible,
            (
                "placement feasible"
                if plan.feasible
                else (plan.infeasible_reason or "placement infeasible")
            ),
        )
    )

    predicted = plan.predicted.to_dict() if plan.predicted else None
    throughput = plan.predicted.throughput_tok_s if plan.predicted else None
    checks.append(
        ContractCheck(
            "planner.throughput_threshold_met",
            throughput is not None and throughput_threshold is not None and throughput >= throughput_threshold,
            (
                f"predicted={throughput:.6f} tok/s >= threshold={throughput_threshold:.6f} tok/s"
                if throughput is not None and throughput_threshold is not None
                else "missing prediction or threshold"
            ),
        )
    )

    min_headroom, memory_rows = _minimum_memory_headroom(model, target, inventory, plan)
    checks.append(
        ContractCheck(
            "planner.memory_headroom_met",
            min_headroom is not None and memory_headroom_min is not None and min_headroom >= memory_headroom_min,
            (
                f"minimum headroom={min_headroom:.6f} >= threshold={memory_headroom_min:.6f}"
                if min_headroom is not None and memory_headroom_min is not None
                else "missing memory headroom or threshold"
            ),
        )
    )

    checks_out = [check.to_dict() for check in checks]
    return {
        "valid": all(check["passed"] for check in checks_out),
        "checks": checks_out,
        "predicted": predicted,
        "memory": {
            "minimum_headroom_fraction": min_headroom,
            "stages": memory_rows,
        },
    }
