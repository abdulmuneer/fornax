from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .accelerator_probe import (
    run_expert_mlp_probe,
    validate_expert_mlp_probe,
)
from .apple_probe import (
    apple_probe_template,
    render_apple_role_decision_draft,
    render_simulated_apple_role_decision,
    simulated_apple_probe_artifact,
    validate_apple_probe_file,
)
from .backend_coverage import (
    render_backend_coverage_report,
    validate_backend_coverage_contract,
)
from .benchmark import benchmark_from_plan
from .benchmark_ledger import (
    append_benchmark_ledger_record,
    build_benchmark_ledger_record,
    validate_benchmark_ledger,
)
from .calibration import run_local_calibration
from .doctor import inspect_phase0_bundle
from .engine_seam import validate_engine_seam_contract
from .engine_simulation import (
    simulated_engine_contract,
    validate_engine_simulation,
)
from .golden import run_golden_plans
from .g1_review import render_g1_gate_review_draft
from .inventory import (
    SIMULATED_CLUSTER_PROFILES,
    build_logical_cluster_inventory,
    collect_local_inventory,
    probe_declared_links,
)
from .continuous_batching import (
    simulate_continuous_batching,
    validate_continuous_batching,
)
from .contracts import load_target_contract
from .io import load_inventory, load_model_target, read_json, write_json
from .planner import plan_placement
from .preflight import run_phase0_preflight
from .program_rebaseline import (
    KER_STATUS_VALUES,
    SCOPE_VALUES,
    render_program_rebaseline_draft,
)
from .moe import simulated_moe_contract, validate_moe_contract
from .model_support import (
    render_model_support_matrix_report,
    simulated_model_support_matrix,
    validate_model_support_matrix,
)
from .network_contract import validate_network_contract
from .network_security_spec import render_network_security_spec_draft
from .observability import validate_observability_contract
from .phase0_status import render_phase0_status_report
from .phase0_simulated_validation import run_phase0_simulated_validation
from .runtime_format import validate_runtime_format_golden
from .runtime_format_spec import render_runtime_format_spec_draft
from .scheduler import simulate_scheduler_from_paths, validate_scheduler_contract
from .simulate import simulation_result, summarize_request_trace
from .substrate_adr import (
    APPLE_ROLE_VALUES,
    STATUS_VALUES,
    render_substrate_adr_draft,
)
from .target_contract import render_target_contract_draft
from .t1_simulated_validation import run_t1_simulated_validation
from .transport import simulated_transport_contract, validate_transport_contract
from .validation import validate_target_contract
from .workers import simulated_worker_contract, validate_worker_contract


def _cmd_accelerator_expert_mlp_probe(args: argparse.Namespace) -> int:
    try:
        result = run_expert_mlp_probe(
            backend=args.backend,
            torch_python=args.torch_python,
            device=args.device,
            dtype=args.dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            batch_tokens=args.batch_tokens,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            experts=args.experts,
            top_k=args.top_k,
            tolerance=args.tolerance,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        print(f"accelerator expert-mlp-probe: {exc}")
        return 2
    write_json(args.out, result)
    if not result.get("measured"):
        print(
            "expert-MLP probe unavailable: "
            f"backend={result.get('backend')} error={result.get('error')}"
        )
        return 2
    summary = validate_expert_mlp_probe(args.out)["summary"]
    suffix = " accelerator" if summary.get("accelerator_measured") else " reference"
    print(
        "expert-MLP probe:"
        f"{suffix} backend={summary.get('backend')} device={summary.get('device')} "
        f"tokens_s={summary.get('tokens_s'):.3f} "
        f"max_abs_error={summary.get('max_abs_error')}"
    )
    return 0


def _cmd_apple_probe_template(args: argparse.Namespace) -> int:
    data = apple_probe_template(
        target_model=args.target_model,
        pinned_build=args.pinned_build,
        threshold_tokens_s=args.threshold_tokens_s,
    )
    write_json(args.out, data)
    print(f"wrote Apple expert-MLP probe template: {args.out}")
    return 0


def _cmd_apple_simulate_probe(args: argparse.Namespace) -> int:
    try:
        artifact = simulated_apple_probe_artifact(
            target_model=args.target_model,
            pinned_build=args.pinned_build,
            recommended_role=args.role,
            reason=args.reason,
        )
    except ValueError as exc:
        print(f"apple simulate-probe: {exc}")
        return 2
    write_json(args.out, artifact)
    if args.decision_out:
        Path(args.decision_out).write_text(
            render_simulated_apple_role_decision(artifact, source=args.out),
            encoding="utf-8",
        )
    print(
        "wrote simulated Apple probe artifact: "
        f"{args.out} (role={args.role}; not G1 closure evidence)"
    )
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


def _cmd_inventory_simulate_cluster(args: argparse.Namespace) -> int:
    try:
        source = (
            read_json(args.source_inventory)
            if args.source_inventory
            else collect_local_inventory()
        )
        data = build_logical_cluster_inventory(
            source,
            gpu_count=args.gpu_count,
            profile=args.profile,
            link_bandwidth_bytes_s=args.link_bandwidth_bytes_s,
            link_latency_s=args.link_latency_s,
            slow_node_factor=args.slow_node_factor,
        )
    except (OSError, ValueError) as exc:
        print(f"inventory simulate-cluster: {exc}")
        return 2
    write_json(args.out, data)
    simulation = data.get("simulation", {})
    print(
        "wrote simulated cluster inventory: "
        f"{args.out}; logical_hosts={simulation.get('logical_host_count')}; "
        "simulation evidence only"
    )
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


def _cmd_engine_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulated_engine_contract(
            plan_id=args.plan_id,
            request_id=args.request_id,
            plan_hash=args.plan_hash,
            max_queue_depth=args.max_queue_depth,
            max_inflight=args.max_inflight,
            microbatch_size=args.microbatch_size,
            timeout_ms=args.timeout_ms,
        )
    except ValueError as exc:
        print(f"engine simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "engine simulate: "
        f"events={summary['event_count']} "
        f"requests={summary['request_count']} "
        f"finished={summary['finished_count']} "
        f"tokens={summary['token_count']} "
        f"embedded_contracts={summary['embedded_contract_count']}"
    )
    return 0


def _cmd_workers_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulated_worker_contract(
            plan_id=args.plan_id,
            request_id=args.request_id,
            plan_hash=args.plan_hash,
            max_queue_depth=args.max_queue_depth,
        )
    except ValueError as exc:
        print(f"workers simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "workers simulate: "
        f"workers={summary['worker_count']} "
        f"events={summary['event_count']} "
        f"plan_rejects={summary['plan_integrity_reject_count']} "
        f"cleanup={summary['cleanup_count']}"
    )
    return 0


def _cmd_transport_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulated_transport_contract(
            plan_id=args.plan_id,
            request_id=args.request_id,
            plan_hash=args.plan_hash,
            max_queue_depth=args.max_queue_depth,
            timeout_ms=args.timeout_ms,
        )
    except ValueError as exc:
        print(f"transport simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "transport simulate: "
        f"logical_hosts={summary['logical_host_count']} "
        f"endpoints={summary['endpoint_count']} "
        f"payloads={summary['payload_count']} "
        f"acks={summary['ack_count']} "
        f"timeouts={summary['timeout_count']} "
        f"cancels={summary['cancel_count']}"
    )
    return 0


def _cmd_scheduler_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_scheduler_from_paths(
            args.plan,
            args.requests,
            plan_id=args.plan_id,
            max_queue_depth=args.max_queue_depth,
            max_inflight=args.max_inflight,
            microbatch_size=args.microbatch_size,
        )
    except (OSError, ValueError) as exc:
        print(f"scheduler simulate: {exc}")
        return 2
    if args.out:
        write_json(args.out, result)
    summary = result["summary"]
    print(
        "scheduler simulate: "
        f"requests={summary['request_count']} "
        f"microbatches={summary['microbatch_count']} "
        f"backpressure={summary['backpressure_count']} "
        f"max_queue={summary['max_observed_queue_depth']} "
        f"max_inflight={summary['max_observed_inflight']}"
    )
    return 0


def _cmd_moe_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulated_moe_contract(
            plan_id=args.plan_id,
            request_id=args.request_id,
            plan_hash=args.plan_hash,
            layer_id=args.layer_id,
            top_k=args.top_k,
            max_remote_wait_ms=args.max_remote_wait_ms,
            migration_hotness_threshold=args.migration_hotness_threshold,
        )
    except ValueError as exc:
        print(f"moe simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "moe simulate: "
        f"tokens={summary['token_count']} "
        f"experts={summary['expert_count']} "
        f"remote_dispatches={summary['remote_dispatch_count']} "
        f"migrations={summary['migration_recommendation_count']} "
        f"remote_hit_rate={summary['remote_hit_rate']:.3f}"
    )
    return 0


def _cmd_model_support_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulated_model_support_matrix(
            matrix_id=args.matrix_id,
            target_model_id=args.target_model_id,
            target_contract=args.target_contract,
        )
    except ValueError as exc:
        print(f"model-support simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "model-support simulate: "
        f"models={summary['model_count']} "
        f"supported={summary['supported_model_count']} "
        f"planned={summary['planned_model_count']} "
        f"capabilities={summary['required_capability_count']}"
    )
    return 0


def _cmd_batching_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_continuous_batching(
            plan_id=args.plan_id,
            max_queue_depth=args.max_queue_depth,
            max_inflight=args.max_inflight,
            microbatch_size=args.microbatch_size,
            fairness_window_s=args.fairness_window_s,
            transfer_s=args.transfer_s,
        )
    except ValueError as exc:
        print(f"batching simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "batching simulate: "
        f"requests={summary['request_count']} "
        f"microbatches={summary['microbatch_count']} "
        f"overlap={summary['overlap_observed']} "
        f"bubble={summary['bubble_fraction']:.3f} "
        f"max_wait={summary['max_wait_s']:.6f}s"
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
    if args.ledger_out:
        command = [
            "python3",
            "-m",
            "fornax",
            "benchmark",
            "--plan",
            args.plan,
            "--mode",
            args.mode,
            "--iterations",
            str(args.iterations),
        ]
        record = build_benchmark_ledger_record(
            result,
            benchmark_id=args.benchmark_id,
            command=command,
            hardware=args.hardware,
            os_name=args.os_name,
            driver_runtime=args.driver_runtime,
            max_mojo_version=args.max_mojo_version,
            model=args.model,
            context=args.context,
            concurrency=args.concurrency,
            quantization=args.quantization,
            thermals=args.thermals,
        )
        append_benchmark_ledger_record(args.ledger_out, record)
    tokens_s = result["result"]["tokens_s"]
    suffix = f" ledger={args.ledger_out}" if args.ledger_out else ""
    print(
        f"benchmark({args.mode}): measured tiny expert-MLP "
        f"tokens_s={tokens_s:.3f} checksum={result['result']['checksum']:.6f}"
        f"{suffix}"
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


def _cmd_program_phase0_status(args: argparse.Namespace) -> int:
    try:
        result = render_phase0_status_report(
            args.bundle,
            report_date=args.date,
            plan_version=args.plan_version,
        )
    except (OSError, ValueError) as exc:
        print(f"program phase0-status: {exc}")
        return 2
    if args.out:
        write_json(args.out, result)
    if args.markdown_out:
        Path(args.markdown_out).write_text(result["markdown"], encoding="utf-8")
    summary = result["summary"]
    print(
        "phase0 status: "
        f"{summary['machine_or_better']}/{summary['total']} "
        "deliverables machine/simulation complete or closed; "
        f"recommended={result['g1']['recommended_outcome']}"
    )
    return 2 if result.get("doctor_errors") else 0


def _cmd_program_simulate_phase0(args: argparse.Namespace) -> int:
    if args.requests and args.trace and args.requests != args.trace:
        print("program simulate-phase0: pass only one of --requests or --trace")
        return 2
    trace_path = args.requests or args.trace
    try:
        result = run_phase0_simulated_validation(
            target_path=args.target,
            out_dir=args.out_dir,
            source_inventory_path=args.source_inventory,
            gpu_count=args.gpu_count,
            profile=args.profile,
            link_bandwidth_bytes_s=args.link_bandwidth_bytes_s,
            link_latency_s=args.link_latency_s,
            slow_node_factor=args.slow_node_factor,
            requests_path=trace_path,
            benchmark_mode=args.benchmark_mode,
            benchmark_iterations=args.benchmark_iterations,
            include_calibration=args.include_calibration,
            calibration_torch_python=args.calibration_torch_python,
            program_report_date=args.program_report_date,
            program_plan_version=args.program_plan_version,
            substrate_pinned_build=args.substrate_pinned_build,
            kickoff_date=args.kickoff_date,
            ker_status=args.ker_status,
            scope=args.scope,
            simulated_apple_role=args.simulated_apple_role,
            simulated_apple_reason=args.simulated_apple_reason,
            include_benchmark_ledger=not args.no_benchmark_ledger,
            benchmark_id=args.benchmark_id,
        )
    except (OSError, ValueError) as exc:
        print(f"program simulate-phase0: {exc}")
        return 2
    summary = result["summary"]
    g1 = result["g1"]
    print(
        "phase0 simulated validation: "
        f"bundle={result['bundle']}; "
        f"{summary.get('machine_or_better', 0)}/{summary.get('total', 0)} "
        "deliverables machine/simulation complete or closed; "
        f"recommended={g1.get('recommended_outcome')}; simulation evidence only"
    )
    return 0 if result["ok"] else 2


def _cmd_program_simulate_t1(args: argparse.Namespace) -> int:
    try:
        result = run_t1_simulated_validation(
            out_dir=args.out_dir,
            source_inventory_path=args.source_inventory,
            gpu_count=args.gpu_count,
            profile=args.profile,
            link_bandwidth_bytes_s=args.link_bandwidth_bytes_s,
            link_latency_s=args.link_latency_s,
            slow_node_factor=args.slow_node_factor,
            plan_id=args.plan_id,
            request_id=args.request_id,
            plan_hash=args.plan_hash,
            max_queue_depth=args.max_queue_depth,
            max_inflight=args.max_inflight,
            microbatch_size=args.microbatch_size,
            timeout_ms=args.timeout_ms,
        )
    except (OSError, ValueError) as exc:
        print(f"program simulate-t1: {exc}")
        return 2
    summary = result["summary"]
    print(
        "t1 simulated validation: "
        f"bundle={result['bundle']}; "
        f"checks={summary['passed_count']}/{summary['check_count']} passed; "
        f"logical_hosts={summary.get('logical_host_count')}; "
        "simulation evidence only"
    )
    return 0 if result["ok"] else 2


def _cmd_preflight(args: argparse.Namespace) -> int:
    if args.requests and args.trace and args.requests != args.trace:
        print("preflight: pass only one of --requests or --trace")
        return 2
    trace_path = args.requests or args.trace
    try:
        inventory_data = None
        if args.inventory:
            inventory_data = read_json(args.inventory)
            if not isinstance(inventory_data, dict):
                raise ValueError("--inventory must contain a JSON object")
        result = run_phase0_preflight(
            target_path=args.target,
            out_dir=args.out_dir,
            requests_path=trace_path,
            benchmark_mode=args.benchmark_mode,
            benchmark_iterations=args.benchmark_iterations,
            inventory_data=inventory_data,
            include_g1_drafts=args.include_g1_drafts,
            substrate_pinned_build=args.substrate_pinned_build,
            kickoff_date=args.kickoff_date,
            ker_status=args.ker_status,
            scope=args.scope,
            include_calibration=args.include_calibration,
            calibration_torch_python=args.calibration_torch_python,
            include_golden_plans=args.include_golden_plans,
            include_program_reports=args.include_program_reports,
            program_report_date=args.program_report_date,
            program_plan_version=args.program_plan_version,
            include_simulated_apple_evidence=args.include_simulated_apple_evidence,
            simulated_apple_role=args.simulated_apple_role,
            simulated_apple_reason=args.simulated_apple_reason,
            include_benchmark_ledger=not args.no_benchmark_ledger,
            benchmark_id=args.benchmark_id,
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


def _cmd_spec_model_support(args: argparse.Namespace) -> int:
    try:
        result = render_model_support_matrix_report(args.fixture)
    except (OSError, ValueError) as exc:
        print(f"spec model-support: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    status = "valid" if result["ok"] else "invalid"
    print(f"wrote model support matrix draft: {args.out} ({status})")
    return 0 if result["ok"] else 2


def _cmd_spec_backend_coverage(args: argparse.Namespace) -> int:
    try:
        result = render_backend_coverage_report(args.fixture)
    except (OSError, ValueError) as exc:
        print(f"spec backend-coverage: {exc}")
        return 2
    Path(args.out).write_text(result["markdown"], encoding="utf-8")
    status = "valid" if result["ok"] else "invalid"
    print(f"wrote backend coverage matrix draft: {args.out} ({status})")
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
    fixture = args.fixture or "fornax/golden_vectors/network_contract"
    result = validate_network_contract(fixture, mode=args.mode)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        print(f"PASS network-contract: {result['fixture']}{suffix}")
        return 0
    print("FAIL network-contract: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_engine_seam(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/engine_seam"
    result = validate_engine_seam_contract(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        print(f"PASS engine-seam: {result['fixture']}{suffix}")
        return 0
    print("FAIL engine-seam: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_engine_simulation(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/engine_simulation"
    result = validate_engine_simulation(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS engine-simulation: {fixture} "
            f"events={summary['event_count']} "
            f"requests={summary['request_count']} "
            f"tokens={summary['token_count']} "
            f"embedded_contracts={summary['embedded_contract_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL engine-simulation: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_observability(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/observability"
    result = validate_observability_contract(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        print(f"PASS observability: {result['fixture']}{suffix}")
        return 0
    print("FAIL observability: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_worker_contract(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/worker_contract"
    result = validate_worker_contract(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS worker-contract: {fixture} "
            f"workers={summary['worker_count']} events={summary['event_count']} "
            f"plan_rejects={summary['plan_integrity_reject_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL worker-contract: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_transport_contract(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/transport_contract"
    result = validate_transport_contract(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS transport-contract: {fixture} "
            f"logical_hosts={summary['logical_host_count']} "
            f"endpoints={summary['endpoint_count']} "
            f"payloads={summary['payload_count']} "
            f"timeouts={summary['timeout_count']} "
            f"cancels={summary['cancel_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL transport-contract: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_moe_runtime(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/moe_runtime"
    result = validate_moe_contract(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS moe-runtime: {fixture} "
            f"experts={summary['expert_count']} "
            f"remote_dispatches={summary['remote_dispatch_count']} "
            f"migrations={summary['migration_recommendation_count']} "
            f"remote_hit_rate={summary['remote_hit_rate']:.3f}"
            f"{suffix}"
        )
        return 0
    print("FAIL moe-runtime: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_model_support(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/model_support"
    result = validate_model_support_matrix(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS model-support: {fixture} "
            f"models={summary['model_count']} "
            f"supported={summary['supported_model_count']} "
            f"planned={summary['planned_model_count']} "
            f"capabilities={summary['required_capability_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL model-support: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_continuous_batching(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/continuous_batching"
    result = validate_continuous_batching(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS continuous-batching: {fixture} "
            f"microbatches={summary['microbatch_count']} "
            f"overlap={summary['overlap_observed']} "
            f"max_wait={summary['max_wait_s']:.6f}s"
            f"{suffix}"
        )
        return 0
    print("FAIL continuous-batching: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_scheduler_contract(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/scheduler_contract"
    result = validate_scheduler_contract(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS scheduler-contract: {fixture} "
            f"events={summary['event_count']} backpressure={summary['backpressure_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL scheduler-contract: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_expert_mlp_probe(args: argparse.Namespace) -> int:
    if not args.fixture:
        print("FAIL expert-mlp-probe: --fixture is required")
        return 1
    result = validate_expert_mlp_probe(args.fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS expert-mlp-probe: {args.fixture} "
            f"backend={summary['backend']} "
            f"accelerator={summary['accelerator_measured']} "
            f"device={summary['device']} "
            f"tokens_s={summary['tokens_s']:.3f}"
            f"{suffix}"
        )
        return 0
    print("FAIL expert-mlp-probe: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_benchmark_ledger(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/benchmark_ledger"
    result = validate_benchmark_ledger(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        print(f"PASS benchmark-ledger: {result['ledger']}{suffix}")
        return 0
    print("FAIL benchmark-ledger: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_backend_coverage(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/backend_coverage"
    result = validate_backend_coverage_contract(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        print(f"PASS backend-coverage: {result['fixture']}{suffix}")
        return 0
    print("FAIL backend-coverage: " + "; ".join(result["errors"]))
    return 1


def _cmd_test(args: argparse.Namespace) -> int:
    if args.test_name == "golden-plans":
        return _cmd_test_golden(args)
    if args.test_name == "runtime-format":
        return _cmd_test_runtime_format(args)
    if args.test_name == "network-contract":
        return _cmd_test_network_contract(args)
    if args.test_name == "engine-seam":
        return _cmd_test_engine_seam(args)
    if args.test_name == "engine-simulation":
        return _cmd_test_engine_simulation(args)
    if args.test_name == "observability":
        return _cmd_test_observability(args)
    if args.test_name == "worker-contract":
        return _cmd_test_worker_contract(args)
    if args.test_name == "transport-contract":
        return _cmd_test_transport_contract(args)
    if args.test_name == "moe-runtime":
        return _cmd_test_moe_runtime(args)
    if args.test_name == "model-support":
        return _cmd_test_model_support(args)
    if args.test_name == "continuous-batching":
        return _cmd_test_continuous_batching(args)
    if args.test_name == "scheduler-contract":
        return _cmd_test_scheduler_contract(args)
    if args.test_name == "backend-coverage":
        return _cmd_test_backend_coverage(args)
    if args.test_name == "benchmark-ledger":
        return _cmd_test_benchmark_ledger(args)
    if args.test_name == "expert-mlp-probe":
        return _cmd_test_expert_mlp_probe(args)
    raise ValueError(args.test_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fornax")
    sub = parser.add_subparsers(dest="command", required=True)

    accelerator = sub.add_parser("accelerator")
    accelerator_sub = accelerator.add_subparsers(
        dest="accelerator_command", required=True
    )
    expert_probe = accelerator_sub.add_parser("expert-mlp-probe")
    expert_probe.add_argument("--out", required=True)
    expert_probe.add_argument("--backend", choices=["cpu-stdlib", "torch"], default="torch")
    expert_probe.add_argument("--torch-python")
    expert_probe.add_argument("--device", default="cuda:0")
    expert_probe.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    expert_probe.add_argument("--iterations", type=int, default=25)
    expert_probe.add_argument("--warmup", type=int, default=3)
    expert_probe.add_argument("--batch-tokens", type=int, default=8)
    expert_probe.add_argument("--hidden-dim", type=int, default=64)
    expert_probe.add_argument("--intermediate-dim", type=int, default=128)
    expert_probe.add_argument("--experts", type=int, default=4)
    expert_probe.add_argument("--top-k", type=int, default=2)
    expert_probe.add_argument("--tolerance", type=float, default=0.1)
    expert_probe.add_argument("--timeout-s", type=float, default=180.0)
    expert_probe.set_defaults(func=_cmd_accelerator_expert_mlp_probe)

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

    apple_simulate = apple_sub.add_parser("simulate-probe")
    apple_simulate.add_argument("--out", required=True)
    apple_simulate.add_argument("--decision-out")
    apple_simulate.add_argument("--target-model", default="target-model")
    apple_simulate.add_argument("--pinned-build", default="unset")
    apple_simulate.add_argument("--role", choices=["capacity-only", "expert-worker"], default="capacity-only")
    apple_simulate.add_argument(
        "--reason",
        default="Simulated development evidence until rank-1 Apple probe is available.",
    )
    apple_simulate.set_defaults(func=_cmd_apple_simulate_probe)

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

    inv_simulate = inv_sub.add_parser("simulate-cluster")
    inv_simulate.add_argument("--out", required=True)
    inv_simulate.add_argument("--source-inventory")
    inv_simulate.add_argument("--gpu-count", type=int, default=2)
    inv_simulate.add_argument(
        "--profile", choices=SIMULATED_CLUSTER_PROFILES, default="two-gpu-heterogeneous"
    )
    inv_simulate.add_argument("--link-bandwidth-bytes-s", type=float, default=25.0e9)
    inv_simulate.add_argument("--link-latency-s", type=float, default=0.00025)
    inv_simulate.add_argument("--slow-node-factor", type=float, default=0.65)
    inv_simulate.set_defaults(func=_cmd_inventory_simulate_cluster)

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

    phase0_status = program_sub.add_parser("phase0-status")
    phase0_status.add_argument("--bundle", required=True)
    phase0_status.add_argument("--out")
    phase0_status.add_argument("--markdown-out")
    phase0_status.add_argument("--date")
    phase0_status.add_argument("--plan-version", default="v3")
    phase0_status.set_defaults(func=_cmd_program_phase0_status)

    simulate_phase0 = program_sub.add_parser("simulate-phase0")
    simulate_phase0.add_argument("--target", required=True)
    simulate_phase0.add_argument("--out-dir", required=True)
    simulate_phase0.add_argument("--source-inventory")
    simulate_phase0.add_argument("--gpu-count", type=int, default=2)
    simulate_phase0.add_argument(
        "--profile", choices=SIMULATED_CLUSTER_PROFILES, default="two-gpu-heterogeneous"
    )
    simulate_phase0.add_argument("--link-bandwidth-bytes-s", type=float, default=25.0e9)
    simulate_phase0.add_argument("--link-latency-s", type=float, default=0.00025)
    simulate_phase0.add_argument("--slow-node-factor", type=float, default=0.65)
    simulate_phase0.add_argument("--requests")
    simulate_phase0.add_argument("--trace", help="deprecated alias for --requests")
    simulate_phase0.add_argument("--benchmark-mode", default="tiny-moe-or-expert-mlp")
    simulate_phase0.add_argument("--benchmark-iterations", type=int, default=25)
    simulate_phase0.add_argument("--include-calibration", action="store_true")
    simulate_phase0.add_argument("--calibration-torch-python")
    simulate_phase0.add_argument("--program-report-date")
    simulate_phase0.add_argument("--program-plan-version", default="v3")
    simulate_phase0.add_argument("--substrate-pinned-build", default="unset")
    simulate_phase0.add_argument("--kickoff-date")
    simulate_phase0.add_argument("--ker-status", choices=KER_STATUS_VALUES, default="unassigned")
    simulate_phase0.add_argument("--scope", choices=SCOPE_VALUES, default="pending")
    simulate_phase0.add_argument(
        "--simulated-apple-role", choices=["capacity-only", "expert-worker"], default="capacity-only"
    )
    simulate_phase0.add_argument("--simulated-apple-reason")
    simulate_phase0.add_argument("--no-benchmark-ledger", action="store_true")
    simulate_phase0.add_argument(
        "--benchmark-id", default="phase0-simulated-validation-tiny-expert-mlp"
    )
    simulate_phase0.set_defaults(func=_cmd_program_simulate_phase0)

    simulate_t1 = program_sub.add_parser("simulate-t1")
    simulate_t1.add_argument("--out-dir", required=True)
    simulate_t1.add_argument("--source-inventory")
    simulate_t1.add_argument("--gpu-count", type=int, default=2)
    simulate_t1.add_argument(
        "--profile", choices=SIMULATED_CLUSTER_PROFILES, default="two-gpu-heterogeneous"
    )
    simulate_t1.add_argument("--link-bandwidth-bytes-s", type=float, default=25.0e9)
    simulate_t1.add_argument("--link-latency-s", type=float, default=0.00025)
    simulate_t1.add_argument("--slow-node-factor", type=float, default=0.65)
    simulate_t1.add_argument("--plan-id", default="t1-simulated-plan")
    simulate_t1.add_argument("--request-id", default="req-t1-simulated")
    simulate_t1.add_argument("--plan-hash", default="sha256:t1-simulated-plan")
    simulate_t1.add_argument("--max-queue-depth", type=int, default=2)
    simulate_t1.add_argument("--max-inflight", type=int, default=2)
    simulate_t1.add_argument("--microbatch-size", type=int, default=2)
    simulate_t1.add_argument("--timeout-ms", type=float, default=50.0)
    simulate_t1.set_defaults(func=_cmd_program_simulate_t1)

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

    engine = sub.add_parser("engine")
    engine_sub = engine.add_subparsers(dest="engine_command", required=True)
    engine_simulate = engine_sub.add_parser("simulate")
    engine_simulate.add_argument("--out", required=True)
    engine_simulate.add_argument("--plan-id", default="engine-simulated-plan")
    engine_simulate.add_argument("--request-id", default="req-engine-simulated")
    engine_simulate.add_argument("--plan-hash", default="sha256:engine-simulated-plan")
    engine_simulate.add_argument("--max-queue-depth", type=int, default=2)
    engine_simulate.add_argument("--max-inflight", type=int, default=2)
    engine_simulate.add_argument("--microbatch-size", type=int, default=2)
    engine_simulate.add_argument("--timeout-ms", type=float, default=50.0)
    engine_simulate.set_defaults(func=_cmd_engine_simulate)


    workers = sub.add_parser("workers")
    workers_sub = workers.add_subparsers(dest="workers_command", required=True)
    workers_simulate = workers_sub.add_parser("simulate")
    workers_simulate.add_argument("--out", required=True)
    workers_simulate.add_argument("--plan-id", default="worker-contract-plan")
    workers_simulate.add_argument("--request-id", default="req-worker-contract")
    workers_simulate.add_argument("--plan-hash", default="sha256:worker-contract-plan")
    workers_simulate.add_argument("--max-queue-depth", type=int, default=2)
    workers_simulate.set_defaults(func=_cmd_workers_simulate)

    transport = sub.add_parser("transport")
    transport_sub = transport.add_subparsers(dest="transport_command", required=True)
    transport_simulate = transport_sub.add_parser("simulate")
    transport_simulate.add_argument("--out", required=True)
    transport_simulate.add_argument("--plan-id", default="transport-contract-plan")
    transport_simulate.add_argument("--request-id", default="req-transport-contract")
    transport_simulate.add_argument(
        "--plan-hash", default="sha256:transport-contract-plan"
    )
    transport_simulate.add_argument("--max-queue-depth", type=int, default=2)
    transport_simulate.add_argument("--timeout-ms", type=float, default=50.0)
    transport_simulate.set_defaults(func=_cmd_transport_simulate)


    scheduler = sub.add_parser("scheduler")
    scheduler_sub = scheduler.add_subparsers(dest="scheduler_command", required=True)
    scheduler_simulate = scheduler_sub.add_parser("simulate")
    scheduler_simulate.add_argument("--plan", required=True)
    scheduler_simulate.add_argument("--requests", required=True)
    scheduler_simulate.add_argument("--plan-id", default="plan-simulated-t1")
    scheduler_simulate.add_argument("--max-queue-depth", type=int, default=4)
    scheduler_simulate.add_argument("--max-inflight", type=int, default=4)
    scheduler_simulate.add_argument("--microbatch-size", type=int, default=2)
    scheduler_simulate.add_argument("--out")
    scheduler_simulate.set_defaults(func=_cmd_scheduler_simulate)

    moe = sub.add_parser("moe")
    moe_sub = moe.add_subparsers(dest="moe_command", required=True)
    moe_simulate = moe_sub.add_parser("simulate")
    moe_simulate.add_argument("--out", required=True)
    moe_simulate.add_argument("--plan-id", default="moe-simulated-plan")
    moe_simulate.add_argument("--request-id", default="req-moe-simulated")
    moe_simulate.add_argument("--plan-hash", default="sha256:moe-simulated-plan")
    moe_simulate.add_argument("--layer-id", type=int, default=1)
    moe_simulate.add_argument("--top-k", type=int, default=2)
    moe_simulate.add_argument("--max-remote-wait-ms", type=float, default=5.0)
    moe_simulate.add_argument("--migration-hotness-threshold", type=float, default=0.50)
    moe_simulate.set_defaults(func=_cmd_moe_simulate)

    model_support = sub.add_parser("model-support")
    model_support_sub = model_support.add_subparsers(
        dest="model_support_command", required=True
    )
    model_support_simulate = model_support_sub.add_parser("simulate")
    model_support_simulate.add_argument("--out", required=True)
    model_support_simulate.add_argument("--matrix-id", default="fornax-model-support-t1")
    model_support_simulate.add_argument(
        "--target-model-id", default="qwen3-moe-class-target"
    )
    model_support_simulate.add_argument(
        "--target-contract",
        default="fornax/golden_plans/v0_target_contract_fixture.md",
    )
    model_support_simulate.set_defaults(func=_cmd_model_support_simulate)

    batching = sub.add_parser("batching")
    batching_sub = batching.add_subparsers(dest="batching_command", required=True)
    batching_simulate = batching_sub.add_parser("simulate")
    batching_simulate.add_argument("--out", required=True)
    batching_simulate.add_argument("--plan-id", default="continuous-batching-plan")
    batching_simulate.add_argument("--max-queue-depth", type=int, default=4)
    batching_simulate.add_argument("--max-inflight", type=int, default=4)
    batching_simulate.add_argument("--microbatch-size", type=int, default=2)
    batching_simulate.add_argument("--fairness-window-s", type=float, default=0.050)
    batching_simulate.add_argument("--transfer-s", type=float, default=0.002)
    batching_simulate.set_defaults(func=_cmd_batching_simulate)


    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--plan", required=True)
    benchmark.add_argument("--mode", default="tiny-moe-or-expert-mlp")
    benchmark.add_argument("--iterations", type=int, default=25)
    benchmark.add_argument("--out")
    benchmark.add_argument("--ledger-out")
    benchmark.add_argument("--benchmark-id", default="tiny-expert-mlp-phase0")
    benchmark.add_argument("--hardware")
    benchmark.add_argument("--os-name")
    benchmark.add_argument("--driver-runtime", default="unknown")
    benchmark.add_argument("--max-mojo-version", default="unknown")
    benchmark.add_argument("--model", default="unknown")
    benchmark.add_argument("--context", default="unknown")
    benchmark.add_argument("--concurrency", default="unknown")
    benchmark.add_argument("--quantization", default="unknown")
    benchmark.add_argument("--thermals", default="unknown")
    benchmark.set_defaults(func=_cmd_benchmark)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--bundle", required=True)
    doctor.add_argument("--out")
    doctor.set_defaults(func=_cmd_doctor)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--target", required=True)
    preflight.add_argument("--out-dir", required=True)
    preflight.add_argument("--inventory")
    preflight.add_argument("--requests")
    preflight.add_argument("--trace", help="deprecated alias for --requests")
    preflight.add_argument("--benchmark-mode", default="tiny-moe-or-expert-mlp")
    preflight.add_argument("--benchmark-iterations", type=int, default=25)
    preflight.add_argument("--include-g1-drafts", action="store_true")
    preflight.add_argument("--include-calibration", action="store_true")
    preflight.add_argument("--calibration-torch-python")
    preflight.add_argument("--include-golden-plans", action="store_true")
    preflight.add_argument("--include-program-reports", action="store_true")
    preflight.add_argument("--program-report-date")
    preflight.add_argument("--program-plan-version", default="v3")
    preflight.add_argument("--include-simulated-apple-evidence", action="store_true")
    preflight.add_argument("--simulated-apple-role", choices=["capacity-only", "expert-worker"], default="capacity-only")
    preflight.add_argument("--simulated-apple-reason")
    preflight.add_argument("--no-benchmark-ledger", action="store_true")
    preflight.add_argument("--benchmark-id", default="phase0-preflight-tiny-expert-mlp")
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

    spec_model_support = spec_sub.add_parser("model-support")
    spec_model_support.add_argument(
        "--fixture", default="fornax/golden_vectors/model_support"
    )
    spec_model_support.add_argument("--out", required=True)
    spec_model_support.set_defaults(func=_cmd_spec_model_support)

    spec_backend = spec_sub.add_parser("backend-coverage")
    spec_backend.add_argument(
        "--fixture", default="fornax/golden_vectors/backend_coverage"
    )
    spec_backend.add_argument("--out", required=True)
    spec_backend.set_defaults(func=_cmd_spec_backend_coverage)

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
    tests.add_argument(
        "test_name",
        choices=[
            "golden-plans",
            "runtime-format",
            "network-contract",
            "engine-seam",
            "engine-simulation",
            "observability",
            "worker-contract",
            "transport-contract",
            "moe-runtime",
            "model-support",
            "continuous-batching",
            "scheduler-contract",
            "backend-coverage",
            "benchmark-ledger",
            "expert-mlp-probe",
        ],
    )
    tests.add_argument("--golden", default="fornax/golden_vectors/runtime_format")
    tests.add_argument("--mode", default="simulated")
    tests.add_argument("--fixture")
    tests.add_argument("--out")
    tests.set_defaults(func=_cmd_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if shutil.which("python3") is None:
        print("warning: python3 not found on PATH")
    return int(args.func(args))
