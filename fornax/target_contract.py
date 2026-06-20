from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import load_target_contract
from .io import load_inventory, read_json
from .planner import ModelSpec, PlacementPlan, Target, plan_placement
from .validation import validate_target_contract


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.2f} {unit}"


def _format_float(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _contract(bundle: dict[str, Any]) -> dict[str, Any]:
    value = bundle.get("contract")
    return value if isinstance(value, dict) else {}


def _inventory_summary(
    inventory_data: dict[str, Any], links_data: dict[str, Any] | None
) -> dict[str, Any]:
    nodes = inventory_data.get("nodes") if isinstance(inventory_data, dict) else None
    links = None
    measured = None
    active_measurement_count = None
    estimated_link_count = None
    warnings: list[str] = []
    if isinstance(links_data, dict):
        links = links_data.get("links")
        measured = links_data.get("measured")
        active_measurement_count = links_data.get("active_measurement_count")
        estimated_link_count = links_data.get("estimated_link_count")
        if isinstance(links_data.get("warnings"), list):
            warnings = [str(item) for item in links_data["warnings"]]
    elif isinstance(inventory_data, dict):
        links = inventory_data.get("links")
    return {
        "node_count": len(nodes) if isinstance(nodes, list) else None,
        "link_count": len(links) if isinstance(links, list) else None,
        "links_measured": measured,
        "active_measurement_count": active_measurement_count,
        "estimated_link_count": estimated_link_count,
        "warnings": warnings,
    }


def _node_table(inventory_data: dict[str, Any]) -> list[str]:
    rows = [
        "| Node | Vendor | Runtime | Free memory | Stage | Expert | KV |",
        "|---|---|---|---:|---|---|---|",
    ]
    nodes = inventory_data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        rows.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        return rows
    for node in nodes:
        if not isinstance(node, dict):
            continue
        rows.append(
            "| {id} | {vendor} | {runtime} | {memory} | {stage} | {expert} | {kv} |".format(
                id=node.get("id", "n/a"),
                vendor=node.get("vendor", "n/a"),
                runtime=node.get("runtime", "n/a"),
                memory=_format_bytes(node.get("mem_free_bytes")),
                stage="yes" if node.get("supports_stage", True) else "no",
                expert="yes" if node.get("supports_expert_worker", False) else "no",
                kv="yes" if node.get("supports_kv", True) else "no",
            )
        )
    return rows


def _memory_table(validation: dict[str, Any]) -> list[str]:
    rows = [
        "| Stage | Node | Mode | Used memory | Free memory | Headroom |",
        "|---:|---|---|---:|---:|---:|",
    ]
    stages = validation.get("memory", {}).get("stages")
    if not isinstance(stages, list) or not stages:
        rows.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
        return rows
    for row in stages:
        if not isinstance(row, dict):
            continue
        rows.append(
            "| {stage} | {node} | {mode} | {used} | {free} | {headroom} |".format(
                stage=row.get("stage", "n/a"),
                node=row.get("node_id", "n/a"),
                mode=row.get("mode", "n/a"),
                used=_format_bytes(row.get("memory_used_bytes")),
                free=_format_bytes(row.get("memory_free_bytes")),
                headroom=_format_float(row.get("headroom_fraction"), 4),
            )
        )
    return rows


def _checks_table(validation: dict[str, Any]) -> list[str]:
    rows = ["| Check | Status | Detail |", "|---|---|---|"]
    checks = validation.get("checks")
    if not isinstance(checks, list) or not checks:
        rows.append("| n/a | n/a | n/a |")
        return rows
    for check in checks:
        if not isinstance(check, dict):
            continue
        detail = str(check.get("detail", "")).replace("|", "\\|")
        rows.append(
            f"| {check.get('name', 'n/a')} | {'PASS' if check.get('passed') else 'FAIL'} | {detail} |"
        )
    return rows


def _baseline_lines(contract: dict[str, Any]) -> list[str]:
    baselines = contract.get("baselines")
    if not baselines:
        return ["- No baselines declared."]
    if isinstance(baselines, dict):
        return [f"- `{name}`: {value}" for name, value in baselines.items()]
    lines: list[str] = []
    if isinstance(baselines, list):
        for item in baselines:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "unnamed")
            status = item.get("status", "declared")
            lines.append(f"- `{name}`: {status}")
    return lines or ["- No named baselines declared."]


def _machine_bundle_with_evidence(
    bundle: dict[str, Any],
    *,
    source_path: Path,
    inventory_path: Path,
    links_path: Path | None,
    inventory_data: dict[str, Any],
    links_data: dict[str, Any] | None,
    plan: PlacementPlan,
    validation: dict[str, Any],
) -> dict[str, Any]:
    machine = dict(bundle)
    machine["evidence"] = {
        "status": "draft_not_signed_off",
        "generated_by": "fornax target draft",
        "source_contract": str(source_path),
        "inventory": str(inventory_path),
        "links": str(links_path) if links_path is not None else None,
        "inventory_summary": _inventory_summary(inventory_data, links_data),
        "planner_prediction": plan.predicted.to_dict() if plan.predicted else None,
        "placement_feasible": plan.feasible,
        "infeasible_reason": plan.infeasible_reason,
        "memory_budget": validation.get("memory"),
        "validation_checks": validation.get("checks"),
        "valid": bool(validation.get("valid")),
        "honesty_note": (
            "This draft records planner predictions and inventory/fabric provenance. "
            "It is not G1 sign-off and measured-fabric warnings remain binding."
        ),
    }
    return machine


def render_target_contract_draft(
    *,
    source_path: str | Path,
    inventory_path: str | Path,
    links_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_path)
    inventory_file = Path(inventory_path)
    links_file = Path(links_path) if links_path is not None else None
    model, target, bundle = load_target_contract(source)
    inventory_data = read_json(inventory_file)
    links_data = read_json(links_file) if links_file is not None else None
    inventory = load_inventory(inventory_file, links_file)
    plan = plan_placement(model, inventory, target)
    validation = validate_target_contract(model, target, bundle, inventory, plan=plan)
    machine = _machine_bundle_with_evidence(
        bundle,
        source_path=source,
        inventory_path=inventory_file,
        links_path=links_file,
        inventory_data=inventory_data,
        links_data=links_data,
        plan=plan,
        validation=validation,
    )
    markdown = _render_markdown(
        model=model,
        target=target,
        bundle=bundle,
        inventory_data=inventory_data,
        links_data=links_data,
        plan=plan,
        validation=validation,
        machine=machine,
    )
    return {
        "markdown": markdown,
        "valid": bool(validation.get("valid")),
        "validation": validation,
        "placement": plan.to_dict(),
        "machine_bundle": machine,
    }


def _render_markdown(
    *,
    model: ModelSpec,
    target: Target,
    bundle: dict[str, Any],
    inventory_data: dict[str, Any],
    links_data: dict[str, Any] | None,
    plan: PlacementPlan,
    validation: dict[str, Any],
    machine: dict[str, Any],
) -> str:
    contract = _contract(bundle)
    predicted = plan.predicted.to_dict() if plan.predicted else {}
    summary = _inventory_summary(inventory_data, links_data)
    lines: list[str] = [
        "# V0 Target Contract Draft",
        "",
        (
            "Status: DRAFT - generated by `fornax target draft`; "
            "not TL/SP sign-off and not a G1 closure claim."
        ),
        "",
        "## Seed Or Replacement Rationale",
        "",
        str(
            contract.get(
                "seed_target_rationale",
                "Missing seed acceptance or replacement rationale.",
            )
        ),
        "",
        "## Target Summary",
        "",
        f"- Layers: {model.num_layers}",
        f"- Hidden dim: {model.hidden_dim}",
        f"- Weight dtype: {model.dtype_weight}",
        f"- Activation dtype: {model.dtype_activation}",
        f"- Concurrency: {target.concurrency}",
        f"- Prompt length: {target.prompt_len}",
        f"- Generation length: {target.gen_len}",
        f"- Objective: {target.objective}",
        "",
        "## Fleet Summary",
        "",
        *_node_table(inventory_data),
        "",
        "## Fabric Summary",
        "",
        f"- Link count: {summary['link_count']}",
        f"- Links fully active-measured: {summary['links_measured']}",
        f"- Active measurement count: {summary['active_measurement_count']}",
        f"- Estimated link count: {summary['estimated_link_count']}",
    ]
    warnings = summary.get("warnings")
    if warnings:
        lines.extend(["- Warnings:"] + [f"  - {warning}" for warning in warnings])
    lines.extend(
        [
            "",
            "## Planner Prediction",
            "",
            f"- Feasible: {plan.feasible}",
            f"- Throughput tok/s: {_format_float(predicted.get('throughput_tok_s'), 3)}",
            f"- TTFT s: {_format_float(predicted.get('ttft_s'), 6)}",
            (
                "- Per-request latency s: "
                f"{_format_float(predicted.get('per_request_latency_s'), 6)}"
            ),
            f"- Bubble fraction: {_format_float(predicted.get('bubble_fraction'), 4)}",
            (
                "- Remote expert wait s/token: "
                f"{_format_float(predicted.get('remote_expert_wait_s_per_token'), 6)}"
            ),
            "",
            "## Memory Budget",
            "",
            *_memory_table(validation),
            "",
            "## Thresholds And Baselines",
            "",
            (
                "- Throughput threshold tok/s: "
                f"{contract.get('throughput_threshold_tok_s', 'missing')}"
            ),
            (
                "- Memory headroom minimum: "
                f"{contract.get('memory_headroom_fraction_min', 'missing')}"
            ),
            f"- Concurrency sweep: {contract.get('concurrency_sweep', 'missing')}",
            f"- Persona min concurrency: {contract.get('persona_min_concurrency', 'missing')}",
            f"- Persona can supply concurrency: {contract.get('persona_can_supply_concurrency', 'missing')}",
            f"- Kill metric: {contract.get('kill_metric', 'missing')}",
            "",
            "### Baselines",
            "",
            *_baseline_lines(contract),
            "",
            "## Validation Checks",
            "",
            *_checks_table(validation),
            "",
            "## Machine-Readable Contract",
            "",
            "```json fornax-target",
            json.dumps(machine, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
