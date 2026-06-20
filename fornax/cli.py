from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .golden import run_golden_plans
from .inventory import collect_local_inventory, probe_declared_links
from .io import load_inventory, load_model_target, read_json, write_json
from .planner import plan_placement


def _cmd_inventory_collect(args: argparse.Namespace) -> int:
    data = collect_local_inventory()
    write_json(args.out, data)
    print(f"wrote inventory: {args.out}")
    return 0


def _cmd_fabric_probe(args: argparse.Namespace) -> int:
    inventory = read_json(args.inventory)
    data = probe_declared_links(inventory)
    write_json(args.out, data)
    print(f"wrote link probe: {args.out}")
    return 0


def _cmd_target_validate(args: argparse.Namespace) -> int:
    model, target = load_model_target(args.target)
    inventory = load_inventory(args.inventory, args.links)
    plan = plan_placement(model, inventory, target)
    result: dict[str, Any] = {
        "valid": plan.feasible,
        "infeasible_reason": plan.infeasible_reason,
        "predicted": plan.predicted.to_dict() if plan.predicted else None,
    }
    if args.out:
        write_json(args.out, result)
    print("valid" if plan.feasible else f"invalid: {plan.infeasible_reason}")
    return 0 if plan.feasible else 2


def _cmd_plan(args: argparse.Namespace) -> int:
    model, target = load_model_target(args.target)
    inventory = load_inventory(args.inventory, args.links)
    plan = plan_placement(model, inventory, target)
    write_json(args.out, plan.to_dict())
    print(f"wrote placement plan: {args.out}")
    return 0 if plan.feasible else 2


def _cmd_simulate(args: argparse.Namespace) -> int:
    plan = read_json(args.plan)
    predicted = plan.get("predicted")
    if predicted is None:
        print(f"infeasible plan: {plan.get('infeasible_reason')}")
        return 2
    if args.out:
        write_json(args.out, {"predicted": predicted, "trace": args.trace})
    print(
        "simulate: "
        f"throughput={predicted['throughput_tok_s']:.3f} tok/s "
        f"latency={predicted['per_request_latency_s']:.6f}s "
        f"bubble={predicted['bubble_fraction']:.3f}"
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    plan = read_json(args.plan)
    predicted = plan.get("predicted")
    if predicted is None:
        print(f"infeasible plan: {plan.get('infeasible_reason')}")
        return 2
    result = {
        "mode": args.mode,
        "measured": False,
        "source": "planner_prediction",
        "throughput_tok_s": predicted["throughput_tok_s"],
        "ttft_s": predicted["ttft_s"],
        "per_request_latency_s": predicted["per_request_latency_s"],
        "note": "Phase-0 dry benchmark; replace with measured tiny-MoE/expert-MLP result for G1.",
    }
    if args.out:
        write_json(args.out, result)
    print(f"benchmark({args.mode}): dry-run throughput={result['throughput_tok_s']:.3f} tok/s")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle)
    required = ["inventory.json", "links.json", "placement.json"]
    missing = [name for name in required if not (bundle / name).exists()]
    result = {"bundle": str(bundle), "ok": not missing, "missing": missing}
    if args.out:
        write_json(args.out, result)
    if missing:
        print("doctor: missing " + ", ".join(missing))
        return 2
    print("doctor: bundle has required Phase-0 files")
    return 0


def _cmd_test_golden(args: argparse.Namespace) -> int:
    results = run_golden_plans()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}: {result.message}")
    passed = sum(1 for r in results if r.passed)
    print(f"golden plans: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def _cmd_test(args: argparse.Namespace) -> int:
    if args.test_name == "golden-plans":
        return _cmd_test_golden(args)
    raise ValueError(args.test_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fornax")
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory")
    inv_sub = inv.add_subparsers(dest="inventory_command", required=True)
    inv_collect = inv_sub.add_parser("collect")
    inv_collect.add_argument("--out", required=True)
    inv_collect.set_defaults(func=_cmd_inventory_collect)

    fabric = sub.add_parser("fabric")
    fabric_sub = fabric.add_subparsers(dest="fabric_command", required=True)
    fabric_probe = fabric_sub.add_parser("probe")
    fabric_probe.add_argument("--inventory", required=True)
    fabric_probe.add_argument("--out", required=True)
    fabric_probe.set_defaults(func=_cmd_fabric_probe)

    target = sub.add_parser("target")
    target_sub = target.add_subparsers(dest="target_command", required=True)
    target_validate = target_sub.add_parser("validate")
    target_validate.add_argument("target")
    target_validate.add_argument("--inventory", required=True)
    target_validate.add_argument("--links")
    target_validate.add_argument("--out")
    target_validate.set_defaults(func=_cmd_target_validate)

    plan = sub.add_parser("plan")
    plan.add_argument("--target", required=True)
    plan.add_argument("--inventory", required=True)
    plan.add_argument("--links")
    plan.add_argument("--out", required=True)
    plan.set_defaults(func=_cmd_plan)

    simulate = sub.add_parser("simulate")
    simulate.add_argument("--plan", required=True)
    simulate.add_argument("--trace")
    simulate.add_argument("--out")
    simulate.set_defaults(func=_cmd_simulate)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--plan", required=True)
    benchmark.add_argument("--mode", default="tiny-moe-or-expert-mlp")
    benchmark.add_argument("--out")
    benchmark.set_defaults(func=_cmd_benchmark)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--bundle", required=True)
    doctor.add_argument("--out")
    doctor.set_defaults(func=_cmd_doctor)

    tests = sub.add_parser("test")
    tests.add_argument("test_name", choices=["golden-plans"])
    tests.set_defaults(func=_cmd_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if shutil.which("python3") is None:
        print("warning: python3 not found on PATH")
    return int(args.func(args))
