from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .benchmark import benchmark_from_plan
from .doctor import inspect_phase0_bundle
from .golden import run_golden_plans
from .inventory import collect_local_inventory, probe_declared_links
from .contracts import load_target_contract
from .io import load_inventory, load_model_target, read_json, write_json
from .planner import plan_placement
from .preflight import run_phase0_preflight
from .network_contract import validate_network_contract
from .runtime_format import validate_runtime_format_golden
from .simulate import simulation_result, summarize_request_trace
from .target_contract import render_target_contract_draft
from .validation import validate_target_contract


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


def _cmd_target_draft(args: argparse.Namespace) -> int:
    try:
        result = render_target_contract_draft(
            source_path=args.source,
            inventory_path=args.inventory,
            links_path=args.links,
        )
    except (OSError, ValueError) as exc:
        print(f"target draft: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    status = "valid" if result["valid"] else "invalid"
    print(f"wrote target contract draft: {args.out} ({status})")
    return 0 if result["valid"] else 2


def _cmd_target_validate(args: argparse.Namespace) -> int:
    model, target, bundle = load_target_contract(args.target)
    inventory = load_inventory(args.inventory, args.links)
    plan = plan_placement(model, inventory, target)
    result = validate_target_contract(model, target, bundle, inventory, plan=plan)
    if args.out:
        write_json(args.out, result)
    if result["valid"]:
        print("valid")
        return 0
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    print("invalid: " + ", ".join(failed))
    return 2


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
    if args.requests and args.trace and args.requests != args.trace:
        print("simulate: pass only one of --requests or --trace")
        return 2
    trace_path = args.requests or args.trace
    try:
        request_trace = summarize_request_trace(trace_path) if trace_path else None
    except ValueError as exc:
        print(f"simulate: invalid request trace: {exc}")
        return 2
    result = simulation_result(predicted, request_trace)
    if args.out:
        write_json(args.out, result)
    suffix = ""
    if request_trace is not None:
        suffix = (
            f" requests={request_trace['request_count']}"
            f" gen_tokens={request_trace['total_generation_tokens']}"
        )
    print(
        "simulate: "
        f"throughput={predicted['throughput_tok_s']:.3f} tok/s "
        f"latency={predicted['per_request_latency_s']:.6f}s "
        f"bubble={predicted['bubble_fraction']:.3f}"
        f"{suffix}"
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    plan = read_json(args.plan)
    try:
        result = benchmark_from_plan(plan, mode=args.mode, iterations=args.iterations)
    except ValueError as exc:
        print(f"benchmark: {exc}")
        return 2
    if args.out:
        write_json(args.out, result)
    tokens_s = result["result"]["tokens_s"]
    print(
        f"benchmark({args.mode}): measured tiny expert-MLP "
        f"tokens_s={tokens_s:.3f} checksum={result['result']['checksum']:.6f}"
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    result = inspect_phase0_bundle(args.bundle)
    if args.out:
        write_json(args.out, result)
    if not result["ok"]:
        print("doctor: errors: " + "; ".join(result["errors"]))
        return 2
    if result["warnings"]:
        print("doctor: ok with warnings: " + "; ".join(result["warnings"]))
        return 0
    print("doctor: bundle has required Phase-0 evidence files")
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    if args.requests and args.trace and args.requests != args.trace:
        print("preflight: pass only one of --requests or --trace")
        return 2
    trace_path = args.requests or args.trace
    try:
        result = run_phase0_preflight(
            target_path=args.target,
            out_dir=args.out_dir,
            requests_path=trace_path,
            benchmark_mode=args.benchmark_mode,
            benchmark_iterations=args.benchmark_iterations,
        )
    except (OSError, ValueError) as exc:
        print(f"preflight: {exc}")
        return 2
    doctor = result["doctor"]
    warnings = list(doctor.get("warnings", []))
    if result["ok"]:
        suffix = f"; warnings: {'; '.join(warnings)}" if warnings else ""
        print(f"preflight: wrote Phase-0 evidence bundle: {result['bundle']}{suffix}")
        return 0
    problems = list(doctor.get("errors", [])) + warnings
    print(f"preflight: bundle incomplete: {'; '.join(problems)}")
    return 2


def _cmd_test_golden(args: argparse.Namespace) -> int:
    results = run_golden_plans()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}: {result.message}")
    passed = sum(1 for r in results if r.passed)
    print(f"golden plans: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def _cmd_test_runtime_format(args: argparse.Namespace) -> int:
    result = validate_runtime_format_golden(args.golden)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        print(f"PASS runtime-format: {result['manifest']}{suffix}")
        return 0
    print("FAIL runtime-format: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_network_contract(args: argparse.Namespace) -> int:
    result = validate_network_contract(args.fixture, mode=args.mode)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        print(f"PASS network-contract: {result['fixture']}{suffix}")
        return 0
    print("FAIL network-contract: " + "; ".join(result["errors"]))
    return 1


def _cmd_test(args: argparse.Namespace) -> int:
    if args.test_name == "golden-plans":
        return _cmd_test_golden(args)
    if args.test_name == "runtime-format":
        return _cmd_test_runtime_format(args)
    if args.test_name == "network-contract":
        return _cmd_test_network_contract(args)
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

    target_draft = target_sub.add_parser("draft")
    target_draft.add_argument("source")
    target_draft.add_argument("--inventory", required=True)
    target_draft.add_argument("--links")
    target_draft.add_argument("--out", required=True)
    target_draft.set_defaults(func=_cmd_target_draft)

    plan = sub.add_parser("plan")
    plan.add_argument("--target", required=True)
    plan.add_argument("--inventory", required=True)
    plan.add_argument("--links")
    plan.add_argument("--out", required=True)
    plan.set_defaults(func=_cmd_plan)

    simulate = sub.add_parser("simulate")
    simulate.add_argument("--plan", required=True)
    simulate.add_argument("--requests")
    simulate.add_argument("--trace", help="deprecated alias for --requests")
    simulate.add_argument("--out")
    simulate.set_defaults(func=_cmd_simulate)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--plan", required=True)
    benchmark.add_argument("--mode", default="tiny-moe-or-expert-mlp")
    benchmark.add_argument("--iterations", type=int, default=25)
    benchmark.add_argument("--out")
    benchmark.set_defaults(func=_cmd_benchmark)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--bundle", required=True)
    doctor.add_argument("--out")
    doctor.set_defaults(func=_cmd_doctor)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--target", required=True)
    preflight.add_argument("--out-dir", required=True)
    preflight.add_argument("--requests")
    preflight.add_argument("--trace", help="deprecated alias for --requests")
    preflight.add_argument("--benchmark-mode", default="tiny-moe-or-expert-mlp")
    preflight.add_argument("--benchmark-iterations", type=int, default=25)
    preflight.set_defaults(func=_cmd_preflight)

    tests = sub.add_parser("test")
    tests.add_argument("test_name", choices=["golden-plans", "runtime-format", "network-contract"])
    tests.add_argument("--golden", default="fornax/golden_vectors/runtime_format")
    tests.add_argument("--mode", default="simulated")
    tests.add_argument("--fixture", default="fornax/golden_vectors/network_contract")
    tests.set_defaults(func=_cmd_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if shutil.which("python3") is None:
        print("warning: python3 not found on PATH")
    return int(args.func(args))
