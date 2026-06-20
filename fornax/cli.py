from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .apple_probe import (
    apple_probe_template,
    render_apple_role_decision_draft,
    validate_apple_probe_file,
)
from .benchmark import benchmark_from_plan
from .calibration import run_local_calibration
from .doctor import inspect_phase0_bundle
from .golden import run_golden_plans
from .g1_review import render_g1_gate_review_draft
from .inventory import collect_local_inventory, probe_declared_links
from .contracts import load_target_contract
from .io import load_inventory, load_model_target, read_json, write_json
from .planner import plan_placement
from .preflight import run_phase0_preflight
from .program_rebaseline import (
    KER_STATUS_VALUES,
    SCOPE_VALUES,
    render_program_rebaseline_draft,
)
from .network_contract import validate_network_contract
from .network_security_spec import render_network_security_spec_draft
from .runtime_format import validate_runtime_format_golden
from .runtime_format_spec import render_runtime_format_spec_draft
from .simulate import simulation_result, summarize_request_trace
from .substrate_adr import (
    APPLE_ROLE_VALUES,
    STATUS_VALUES,
    render_substrate_adr_draft,
)
from .target_contract import render_target_contract_draft
from .validation import validate_target_contract


def _cmd_apple_probe_template(args: argparse.Namespace) -> int:
    data = apple_probe_template(
        target_model=args.target_model,
        pinned_build=args.pinned_build,
        threshold_tokens_s=args.threshold_tokens_s,
    )
    write_json(args.out, data)
    print(f"wrote Apple expert-MLP probe template: {args.out}")
    return 0


def _cmd_apple_validate_probe(args: argparse.Namespace) -> int:
    try:
        result = validate_apple_probe_file(args.probe)
    except (OSError, ValueError) as exc:
        print(f"apple validate-probe: {exc}")
        return 2
    if args.out:
        write_json(args.out, result)
    role = result.get("recommended_role", "undecided")
    if result.get("valid"):
        print(f"valid Apple probe evidence: role={role} outcome={result.get('outcome')}")
        return 0
    failed = [check["name"] for check in result.get("checks", []) if not check.get("passed")]
    print("invalid Apple probe evidence: " + ", ".join(failed))
    return 2


def _cmd_apple_role_decision(args: argparse.Namespace) -> int:
    try:
        result = render_apple_role_decision_draft(args.probe)
    except (OSError, ValueError) as exc:
        print(f"apple role-decision: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    validation = result["validation"]
    role = validation.get("recommended_role", "undecided")
    if validation.get("valid"):
        print(f"wrote Apple role decision draft: {args.out} (role={role})")
        return 0
    print(f"wrote incomplete Apple role decision draft: {args.out} (role={role})")
    return 2


def _cmd_calibrate_local(args: argparse.Namespace) -> int:
    try:
        result = run_local_calibration(
            cpu_memory_bytes=args.cpu_memory_bytes,
            cpu_memory_iterations=args.cpu_memory_iterations,
            cpu_compute_iterations=args.cpu_compute_iterations,
            try_torch_cuda=not args.no_torch_cuda,
            torch_python=args.torch_python,
            cuda_matrix_dim=args.cuda_matrix_dim,
            cuda_iterations=args.cuda_iterations,
        )
    except ValueError as exc:
        print(f"calibrate local: {exc}")
        return 2
    write_json(args.out, result)
    warnings = result.get("warnings", [])
    suffix = ""
    if warnings:
        suffix = "; warnings: " + "; ".join(str(warning) for warning in warnings)
    print(f"wrote local calibration artifact: {args.out}{suffix}")
    return 0


def _cmd_inventory_collect(args: argparse.Namespace) -> int:
    data = collect_local_inventory()
    write_json(args.out, data)
    print(f"wrote inventory: {args.out}")
    return 0


def _cmd_fabric_probe(args: argparse.Namespace) -> int:
    inventory = read_json(args.inventory)
    try:
        data = probe_declared_links(
            inventory,
            active_local=args.active_local,
            torch_python=args.torch_python,
            active_local_bytes=args.active_local_bytes,
            active_local_iterations=args.active_local_iterations,
        )
    except ValueError as exc:
        print(f"fabric probe: {exc}")
        return 2
    write_json(args.out, data)
    suffix = ""
    warnings = data.get("warnings", [])
    if warnings:
        suffix = "; warnings: " + "; ".join(str(warning) for warning in warnings)
    print(f"wrote link probe: {args.out}{suffix}")
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


def _cmd_program_rebaseline(args: argparse.Namespace) -> int:
    try:
        result = render_program_rebaseline_draft(
            kickoff_date=args.kickoff_date,
            phase0_weeks=args.phase0_weeks,
            ker_status=args.ker_status,
            scope=args.scope,
        )
    except ValueError as exc:
        print(f"program rebaseline: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    warnings = result.get("warnings", [])
    suffix = ""
    if warnings:
        suffix = "; warnings: " + "; ".join(str(warning) for warning in warnings)
    print(f"wrote program rebaseline draft: {args.out}{suffix}")
    return 0


def _cmd_program_g1_review(args: argparse.Namespace) -> int:
    result = render_g1_gate_review_draft(
        args.bundle,
        review_date=args.date,
        plan_version=args.plan_version,
    )
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    blockers = result.get("closure_blockers", [])
    suffix = f"; closure blockers: {len(blockers)}" if blockers else ""
    print(
        "wrote G1 gate review draft: "
        f"{args.out} ({result['recommended_outcome']}{suffix})"
    )
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
            include_g1_drafts=args.include_g1_drafts,
            substrate_pinned_build=args.substrate_pinned_build,
            kickoff_date=args.kickoff_date,
            ker_status=args.ker_status,
            scope=args.scope,
            include_calibration=args.include_calibration,
            calibration_torch_python=args.calibration_torch_python,
            active_local_links=args.active_local_links,
            fabric_torch_python=args.fabric_torch_python,
            active_local_link_bytes=args.active_local_link_bytes,
            active_local_link_iterations=args.active_local_link_iterations,
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


def _cmd_spec_network_security(args: argparse.Namespace) -> int:
    try:
        result = render_network_security_spec_draft(args.fixture)
    except (OSError, ValueError) as exc:
        print(f"spec network-security: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    status = "valid" if result["ok"] else "invalid"
    print(f"wrote network-security spec draft: {args.out} ({status})")
    return 0 if result["ok"] else 2


def _cmd_spec_runtime_format(args: argparse.Namespace) -> int:
    try:
        result = render_runtime_format_spec_draft(args.golden)
    except (OSError, ValueError) as exc:
        print(f"spec runtime-format: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    status = "valid" if result["ok"] else "invalid"
    print(f"wrote runtime-format spec draft: {args.out} ({status})")
    return 0 if result["ok"] else 2


def _cmd_spec_substrate_adr(args: argparse.Namespace) -> int:
    try:
        result = render_substrate_adr_draft(
            pinned_build=args.pinned_build,
            last_checked=args.last_checked,
            status=args.status,
            apple_role=args.apple_role,
        )
    except (OSError, ValueError) as exc:
        print(f"spec substrate-adr: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    warnings = result.get("warnings", [])
    suffix = ""
    if warnings:
        suffix = "; warnings: " + "; ".join(str(warning) for warning in warnings)
    print(f"wrote substrate ADR draft: {args.out}{suffix}")
    return 0


def _cmd_test_golden(args: argparse.Namespace) -> int:
    results = run_golden_plans()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}: {result.message}")
    passed = sum(1 for r in results if r.passed)
    report = {
        "test_tier": "T0",
        "command": "fornax test golden-plans",
        "passed": passed == len(results),
        "passed_count": passed,
        "total_count": len(results),
        "results": [
            {"name": result.name, "passed": result.passed, "message": result.message}
            for result in results
        ],
    }
    if args.out:
        write_json(args.out, report)
    print(f"golden plans: {passed}/{len(results)} passed")
    return 0 if report["passed"] else 1


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

    calibrate = sub.add_parser("calibrate")
    calibrate_sub = calibrate.add_subparsers(dest="calibrate_command", required=True)
    calibrate_local = calibrate_sub.add_parser("local")
    calibrate_local.add_argument("--out", required=True)
    calibrate_local.add_argument("--cpu-memory-bytes", type=int, default=16 * 1024 * 1024)
    calibrate_local.add_argument("--cpu-memory-iterations", type=int, default=8)
    calibrate_local.add_argument("--cpu-compute-iterations", type=int, default=200000)
    calibrate_local.add_argument("--no-torch-cuda", action="store_true")
    calibrate_local.add_argument("--torch-python")
    calibrate_local.add_argument("--cuda-matrix-dim", type=int, default=512)
    calibrate_local.add_argument("--cuda-iterations", type=int, default=10)
    calibrate_local.set_defaults(func=_cmd_calibrate_local)

    apple = sub.add_parser("apple")
    apple_sub = apple.add_subparsers(dest="apple_command", required=True)
    apple_template = apple_sub.add_parser("probe-template")
    apple_template.add_argument("--out", required=True)
    apple_template.add_argument("--target-model", default="target-model")
    apple_template.add_argument("--pinned-build", default="unset")
    apple_template.add_argument("--threshold-tokens-s", type=float, default=1.0)
    apple_template.set_defaults(func=_cmd_apple_probe_template)

    apple_validate = apple_sub.add_parser("validate-probe")
    apple_validate.add_argument("probe")
    apple_validate.add_argument("--out")
    apple_validate.set_defaults(func=_cmd_apple_validate_probe)

    apple_decision = apple_sub.add_parser("role-decision")
    apple_decision.add_argument("--probe", required=True)
    apple_decision.add_argument("--out", required=True)
    apple_decision.set_defaults(func=_cmd_apple_role_decision)

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
    fabric_probe.add_argument("--active-local", action="store_true")
    fabric_probe.add_argument("--torch-python")
    fabric_probe.add_argument("--active-local-bytes", type=int, default=16 * 1024 * 1024)
    fabric_probe.add_argument("--active-local-iterations", type=int, default=4)
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

    program = sub.add_parser("program")
    program_sub = program.add_subparsers(dest="program_command", required=True)
    rebaseline = program_sub.add_parser("rebaseline")
    rebaseline.add_argument("--out", required=True)
    rebaseline.add_argument("--kickoff-date")
    rebaseline.add_argument("--phase0-weeks", type=int, default=4)
    rebaseline.add_argument("--ker-status", choices=KER_STATUS_VALUES, default="unassigned")
    rebaseline.add_argument("--scope", choices=SCOPE_VALUES, default="pending")
    rebaseline.set_defaults(func=_cmd_program_rebaseline)

    g1_review = program_sub.add_parser("g1-review")
    g1_review.add_argument("--bundle", required=True)
    g1_review.add_argument("--out", required=True)
    g1_review.add_argument("--date")
    g1_review.add_argument("--plan-version", default="v3")
    g1_review.set_defaults(func=_cmd_program_g1_review)

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
    preflight.add_argument("--include-g1-drafts", action="store_true")
    preflight.add_argument("--include-calibration", action="store_true")
    preflight.add_argument("--calibration-torch-python")
    preflight.add_argument("--active-local-links", action="store_true")
    preflight.add_argument("--fabric-torch-python")
    preflight.add_argument("--active-local-link-bytes", type=int, default=16 * 1024 * 1024)
    preflight.add_argument("--active-local-link-iterations", type=int, default=4)
    preflight.add_argument("--substrate-pinned-build", default="unset")
    preflight.add_argument("--kickoff-date")
    preflight.add_argument("--ker-status", choices=KER_STATUS_VALUES, default="unassigned")
    preflight.add_argument("--scope", choices=SCOPE_VALUES, default="pending")
    preflight.set_defaults(func=_cmd_preflight)

    spec = sub.add_parser("spec")
    spec_sub = spec.add_subparsers(dest="spec_command", required=True)
    spec_runtime = spec_sub.add_parser("runtime-format")
    spec_runtime.add_argument("--golden", default="fornax/golden_vectors/runtime_format")
    spec_runtime.add_argument("--out", required=True)
    spec_runtime.set_defaults(func=_cmd_spec_runtime_format)

    spec_network = spec_sub.add_parser("network-security")
    spec_network.add_argument("--fixture", default="fornax/golden_vectors/network_contract")
    spec_network.add_argument("--out", required=True)
    spec_network.set_defaults(func=_cmd_spec_network_security)

    spec_substrate = spec_sub.add_parser("substrate-adr")
    spec_substrate.add_argument("--out", required=True)
    spec_substrate.add_argument("--pinned-build", default="unset")
    spec_substrate.add_argument("--last-checked")
    spec_substrate.add_argument("--status", choices=STATUS_VALUES, default="probing")
    spec_substrate.add_argument(
        "--apple-role", choices=APPLE_ROLE_VALUES, default="undecided"
    )
    spec_substrate.set_defaults(func=_cmd_spec_substrate_adr)

    tests = sub.add_parser("test")
    tests.add_argument("test_name", choices=["golden-plans", "runtime-format", "network-contract"])
    tests.add_argument("--golden", default="fornax/golden_vectors/runtime_format")
    tests.add_argument("--mode", default="simulated")
    tests.add_argument("--fixture", default="fornax/golden_vectors/network_contract")
    tests.add_argument("--out")
    tests.set_defaults(func=_cmd_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if shutil.which("python3") is None:
        print("warning: python3 not found on PATH")
    return int(args.func(args))
