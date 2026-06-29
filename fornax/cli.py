from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .accelerator_probe import (
    run_activation_transfer_probe,
    run_expert_mlp_probe,
    validate_activation_transfer_probe,
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
from .g1_evidence_packet import (
    build_g1_evidence_packet,
    validate_g1_evidence_packet_fixture,
)
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
from .local_accelerator_smoke import (
    run_local_accelerator_smoke,
    validate_local_accelerator_smoke,
)
from .local_http_serving_smoke import (
    run_local_http_serving_smoke,
    validate_local_http_serving_smoke,
)
from .local_serving_smoke import (
    run_local_serving_smoke,
    validate_local_serving_smoke,
)
from .planner import plan_placement
from .preflight import run_phase0_preflight
from .remote_expert_probe import (
    run_remote_expert_batch_probe,
    validate_remote_expert_batch_probe,
)
from .resilience import (
    simulate_resilience_replay,
    validate_resilience_replay,
)
from .program_governance import (
    simulate_program_governance,
    validate_program_governance,
)
from .program_rebaseline import (
    KER_STATUS_VALUES,
    SCOPE_VALUES,
    render_program_rebaseline_draft,
)
from .moe import simulated_moe_contract, validate_moe_contract
from .moe_migration import (
    simulated_moe_hot_expert_migration,
    validate_moe_hot_expert_migration,
)
from .moe_parity import (
    run_moe_layer_parity_probe,
    validate_moe_layer_parity_probe,
)
from .model_support import (
    render_model_support_matrix_report,
    simulated_model_support_matrix,
    validate_model_support_matrix,
)
from .metrics_ledger import simulate_metrics_ledger, validate_metrics_ledger
from .network_contract import validate_network_contract
from .network_security_spec import render_network_security_spec_draft
from .pipeline_probe import (
    parse_prompts_json,
    run_pipeline_correctness_probe,
    validate_pipeline_correctness_probe,
)
from .observability import validate_observability_contract
from .onboarding import (
    simulate_onboarding_methodology,
    validate_onboarding_methodology,
)
from .ops_lifecycle import (
    simulate_ops_lifecycle,
    validate_ops_lifecycle,
)
from .phase0_status import render_phase0_status_report
from .phase0_simulated_validation import run_phase0_simulated_validation
from .phase3_proxy_gate import (
    build_phase3_proxy_gate_packet,
    validate_phase3_proxy_gate,
    validate_phase3_proxy_gate_packet,
)
from .phase4_resilience_gate import (
    build_phase4_resilience_gate_packet,
    render_phase4_t4_runbook_markdown,
    validate_phase4_resilience_gate,
    validate_phase4_resilience_gate_packet,
)
from .phase5_ga_gate import (
    build_phase5_ga_gate_packet,
    render_phase5_g5_runbook_markdown,
    validate_phase5_ga_gate,
    validate_phase5_ga_gate_packet,
)
from .runtime_format import validate_runtime_format_golden
from .runtime_format_spec import render_runtime_format_spec_draft
from .serving import (
    simulate_serving_adapter,
    validate_serving_adapter,
)
from .state_ownership import simulate_state_ownership, validate_state_ownership
from .stage_host import simulate_stage_host, validate_stage_host
from .stage_replication import (
    simulate_stage_replication,
    validate_stage_replication,
)
from .scheduler import simulate_scheduler_from_paths, validate_scheduler_contract
from .simulate import simulation_result, summarize_request_trace
from .substrate_adr import (
    APPLE_ROLE_VALUES,
    STATUS_VALUES,
    render_substrate_adr_draft,
)
from .target_contract import render_target_contract_draft
from .target_fixture_probe import (
    parse_prompt_tokens_json,
    run_target_fixture_execution_probe,
    validate_target_fixture_execution_probe,
)
from .t1_simulated_validation import run_t1_simulated_validation
from .throughput_scaling import (
    parse_concurrency_levels,
    simulate_throughput_scaling,
    validate_throughput_scaling,
)
from .trace_ledger import simulate_trace_ledger, validate_trace_ledger
from .transport import simulated_transport_contract, validate_transport_contract
from .trust_boundary import simulate_trust_boundary, validate_trust_boundary
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



def _cmd_accelerator_activation_transfer_probe(args: argparse.Namespace) -> int:
    try:
        result = run_activation_transfer_probe(
            backend=args.backend,
            torch_python=args.torch_python,
            source_device=args.source_device,
            destination_device=args.destination_device,
            dtype=args.dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            payload_bytes=args.payload_mib * 1024 * 1024,
            tolerance=args.tolerance,
            logical_source_host=args.logical_source_host,
            logical_destination_host=args.logical_destination_host,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        print(f"accelerator activation-transfer-probe: {exc}")
        return 2
    write_json(args.out, result)
    if not result.get("measured"):
        print(
            "activation-transfer probe unavailable: "
            f"backend={result.get('backend')} error={result.get('error')}"
        )
        return 2
    validation = validate_activation_transfer_probe(args.out)
    if not validation["ok"]:
        print("activation-transfer probe invalid: " + "; ".join(validation["errors"]))
        return 2
    summary = validation["summary"]
    suffix = " accelerator" if summary.get("accelerator_measured") else " reference"
    print(
        "activation-transfer probe:"
        f"{suffix} backend={summary.get('backend')} "
        f"{summary.get('source_device')}->{summary.get('destination_device')} "
        f"bandwidth_gib_s={summary.get('bandwidth_gib_s'):.3f} "
        f"latency_s={summary.get('latency_s_per_transfer'):.6f}"
    )
    return 0


def _cmd_accelerator_target_fixture_probe(args: argparse.Namespace) -> int:
    try:
        prompt_tokens = parse_prompt_tokens_json(args.prompt_tokens_json, args.vocab_size)
        result = run_target_fixture_execution_probe(
            backend=args.backend,
            torch_python=args.torch_python,
            device=args.device,
            dtype=args.dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            vocab_size=args.vocab_size,
            new_tokens=args.new_tokens,
            prompt_tokens=prompt_tokens,
            stop_token_id=args.stop_token_id,
            tolerance=args.tolerance,
            logical_host=args.logical_host,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        print(f"accelerator target-fixture-probe: {exc}")
        return 2
    write_json(args.out, result)
    if not result.get("measured"):
        print(
            "target-fixture probe unavailable: "
            f"backend={result.get('backend')} error={result.get('error')}"
        )
        return 2
    validation = validate_target_fixture_execution_probe(args.out)
    if not validation["ok"]:
        print("target-fixture probe invalid: " + "; ".join(validation["errors"]))
        return 2
    summary = validation["summary"]
    suffix = " accelerator" if summary.get("accelerator_measured") else " reference"
    print(
        "target-fixture probe:"
        f"{suffix} backend={summary.get('backend')} device={summary.get('device')} "
        f"generated={summary.get('generated_text')} "
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


def _cmd_observability_metrics_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_metrics_ledger(
            plan_id=args.plan_id,
            request_id=args.request_id,
            max_queue_depth=args.max_queue_depth,
            max_inflight=args.max_inflight,
            kv_page_limit=args.kv_page_limit,
            memory_limit_bytes=args.memory_limit_bytes,
            memory_warning_fraction=args.memory_warning_fraction,
            memory_critical_fraction=args.memory_critical_fraction,
            sample_period_ms=args.sample_period_ms,
        )
    except ValueError as exc:
        print(f"observability metrics-simulate: {exc}")
        return 2
    write_json(args.out, result)
    validation = validate_metrics_ledger(args.out)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "observability metrics-simulate: "
        f"samples={summary['sample_count']} "
        f"alerts={summary['alert_count']} "
        f"max_queue={summary['max_queue_depth_observed']} "
        f"memory_pressure={summary['max_memory_pressure_fraction']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2




def _cmd_observability_trace_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_trace_ledger(
            plan_id=args.plan_id,
            request_id=args.request_id,
            trace_id=args.trace_id,
        )
    except ValueError as exc:
        print(f"observability trace-simulate: {exc}")
        return 2
    write_json(args.out, result)
    validation = validate_trace_ledger(args.out)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "observability trace-simulate: "
        f"components={summary['component_count']} "
        f"spans={summary['span_count']} "
        f"events={summary['event_count']} "
        f"edges={summary['required_edge_count']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2


def _cmd_runtime_stage_host_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_stage_host(
            plan_id=args.plan_id,
            request_id=args.request_id,
            stage_id=args.stage_id,
            logical_host_id=args.logical_host_id,
            predecessor_stage_id=args.predecessor_stage_id,
            successor_stage_id=args.successor_stage_id,
            layer_start=args.layer_start,
            layer_count=args.layer_count,
            token_count=args.token_count,
            hidden_dim=args.hidden_dim,
            dtype=args.dtype,
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        print(f"runtime stage-host-simulate: {exc}")
        return 2
    write_json(args.out, result)
    validation = validate_stage_host(args.out)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "stage-host simulation: "
        f"events={summary['event_count']} "
        f"boundary_ops={summary['boundary_op_count']} "
        f"max_abs_error={summary['max_abs_error']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2


def _cmd_serving_adapter_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_serving_adapter(
            plan_id=args.plan_id,
            request_id=args.request_id,
            model=args.model,
            stream=args.stream,
            max_tokens=args.max_tokens,
            template_hash=args.template_hash,
            tokenizer_hash=args.tokenizer_hash,
        )
    except ValueError as exc:
        print(f"serving adapter-simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "serving adapter-simulate: "
        f"surfaces={summary['surface_count']} "
        f"chunks={summary['openai_chunk_count']} "
        f"tool_calls={summary['tool_call_count']} "
        f"correctness={summary['correctness_passed']}"
    )
    return 0


def _cmd_serving_state_ownership_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_state_ownership(
            plan_id=args.plan_id,
            request_id=args.request_id,
            cancel_request_id=args.cancel_request_id,
            model_id=args.model,
        )
    except ValueError as exc:
        print(f"serving state-ownership-simulate: {exc}")
        return 2
    write_json(args.out, result)
    validation = validate_state_ownership(args.out)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "serving state-ownership-simulate: "
        f"resources={summary['resource_count']} "
        f"transitions={summary['transition_count']} "
        f"released={summary['terminal_released_count']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2


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


def _cmd_transport_trust_boundary_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_trust_boundary(
            plan_id=args.plan_id,
            request_id=args.request_id,
            plan_hash=args.plan_hash,
            cluster_id=args.cluster_id,
            token_ttl_s=args.token_ttl_s,
        )
    except ValueError as exc:
        print(f"transport trust-boundary-simulate: {exc}")
        return 2
    write_json(args.out, result)
    validation = validate_trust_boundary(args.out)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "transport trust-boundary-simulate: "
        f"identities={summary['identity_count']} "
        f"accepted={summary['accepted_auth_count']} "
        f"rejected={summary['rejected_auth_count']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2


def _cmd_replication_simulate(args: argparse.Namespace) -> int:
    try:
        token_counts = [int(item.strip()) for item in args.microbatch_token_counts.split(",") if item.strip()]
        result = simulate_stage_replication(
            plan_id=args.plan_id,
            bottleneck_stage_index=args.bottleneck_stage_index,
            microbatch_token_counts=token_counts,
            baseline_replica_id=args.baseline_replica_id,
            added_replica_id=args.added_replica_id,
            baseline_stage_time_s_per_token=args.baseline_stage_time_s_per_token,
            replicated_stage_time_s_per_token=args.replicated_stage_time_s_per_token,
            transfer_overhead_s=args.transfer_overhead_s,
            speedup_floor=args.speedup_floor,
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        print(f"replication simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "replication simulate: "
        f"replicas={summary['replica_count']} "
        f"microbatches={summary['microbatch_count']} "
        f"speedup={summary['speedup']:.3f} "
        f"max_abs_error={summary['max_abs_error']}"
    )
    return 0


def _cmd_resilience_replay_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_resilience_replay(
            plan_id=args.plan_id,
            failed_node_id=args.failed_node_id,
            replay_node_id=args.replay_node_id,
            checkpoint_token_index=args.checkpoint_token_index,
            node_loss_time_s=args.node_loss_time_s,
            replay_delay_s=args.replay_delay_s,
            token_time_s=args.token_time_s,
            max_replay_delay_s=args.max_replay_delay_s,
            vocab_size=args.vocab_size,
        )
    except ValueError as exc:
        print(f"resilience replay-simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "resilience replay-simulate: "
        f"requests={summary['request_count']} "
        f"replayed={summary['replayed_request_count']} "
        f"dropped={summary['dropped_request_count']} "
        f"max_replay_delay_s={summary['max_replay_delay_s']}"
    )
    return 0


def _cmd_ops_lifecycle_simulate(args: argparse.Namespace) -> int:
    try:
        node_ids = [item.strip() for item in args.node_ids.split(",") if item.strip()]
        result = simulate_ops_lifecycle(
            plan_id=args.plan_id,
            cluster_id=args.cluster_id,
            model_id=args.model_id,
            initial_version=args.initial_version,
            target_version=args.target_version,
            node_ids=node_ids,
            replacement_node_id=args.replacement_node_id,
            in_flight_requests=args.in_flight_requests,
        )
    except ValueError as exc:
        print(f"ops lifecycle-simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "ops lifecycle-simulate: "
        f"actions={summary['action_count']} "
        f"events={summary['event_count']} "
        f"dropped={summary['dropped_in_flight_count']} "
        f"rollback={summary['rollback_verified']} "
        f"node_replace={summary['node_replace_verified']}"
    )
    return 0


def _cmd_ops_onboarding_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_onboarding_methodology(
            plan_id=args.plan_id,
            package_id=args.package_id,
            benchmark_id=args.benchmark_id,
        )
    except ValueError as exc:
        print(f"ops onboarding-simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "ops onboarding-simulate: "
        f"tracks={summary['track_count']} "
        f"docs={summary['document_count']} "
        f"glossary={summary['glossary_term_count']} "
        f"lab_reference={summary['lab_reference_required']} "
        f"product_ga={summary['product_ga_complete']}"
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


def _cmd_moe_migration_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulated_moe_hot_expert_migration(
            plan_id=args.plan_id,
            request_id=args.request_id,
            plan_hash=args.plan_hash,
            token_count=args.token_count,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            vocab_size=args.vocab_size,
            expert_count=args.expert_count,
            top_k=args.top_k,
            hot_expert_id=args.hot_expert_id,
            migration_hotness_threshold=args.migration_hotness_threshold,
            tolerance=args.tolerance,
            logical_source_host=args.logical_source_host,
            logical_expert_host=args.logical_expert_host,
        )
    except ValueError as exc:
        print(f"moe migration-simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "moe migration-simulate: "
        f"hot_expert={summary['hot_expert_id']} "
        f"remote_reduction={summary['remote_token_copy_reduction']} "
        f"post_remote_batches={summary['post_remote_batches']} "
        f"max_post_logit_abs_error={summary['max_post_logit_abs_error']}"
    )
    return 0


def _cmd_moe_remote_expert_probe(args: argparse.Namespace) -> int:
    try:
        result = run_remote_expert_batch_probe(
            backend=args.backend,
            torch_python=args.torch_python,
            source_device=args.source_device,
            expert_device=args.expert_device,
            dtype=args.dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            token_count=args.token_count,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            expert_id=args.expert_id,
            tolerance=args.tolerance,
            logical_source_host=args.logical_source_host,
            logical_expert_host=args.logical_expert_host,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        print(f"moe remote-expert-probe: {exc}")
        return 2
    write_json(args.out, result)
    if not result.get("measured"):
        print(
            "remote expert batch probe unavailable: "
            f"backend={result.get('backend')} error={result.get('error')}"
        )
        return 2
    validation = validate_remote_expert_batch_probe(args.out)
    if not validation["ok"]:
        print("remote expert batch probe invalid: " + "; ".join(validation["errors"]))
        return 2
    summary = validation["summary"]
    suffix = " accelerator" if summary.get("accelerator_measured") else " reference"
    print(
        "remote expert batch probe:"
        f"{suffix} backend={summary.get('backend')} "
        f"{summary.get('source_device')}->{summary.get('expert_device')} "
        f"batches={summary.get('remote_batches')} "
        f"expert_calls_s={summary.get('expert_calls_s'):.3f} "
        f"max_abs_error={summary.get('max_abs_error')}"
    )
    return 0


def _cmd_moe_parity_probe(args: argparse.Namespace) -> int:
    try:
        result = run_moe_layer_parity_probe(
            backend=args.backend,
            torch_python=args.torch_python,
            source_device=args.source_device,
            expert_device=args.expert_device,
            dtype=args.dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            token_count=args.token_count,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            vocab_size=args.vocab_size,
            expert_count=args.expert_count,
            top_k=args.top_k,
            tolerance=args.tolerance,
            logical_source_host=args.logical_source_host,
            logical_expert_host=args.logical_expert_host,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        print(f"moe parity-probe: {exc}")
        return 2
    write_json(args.out, result)
    if not result.get("measured"):
        print(
            "MoE parity probe unavailable: "
            f"backend={result.get('backend')} error={result.get('error')}"
        )
        return 2
    validation = validate_moe_layer_parity_probe(args.out)
    if not validation["ok"]:
        print("MoE parity probe invalid: " + "; ".join(validation["errors"]))
        return 2
    summary = validation["summary"]
    suffix = " accelerator" if summary.get("accelerator_measured") else " reference"
    print(
        "MoE layer parity probe:"
        f"{suffix} backend={summary.get('backend')} "
        f"{summary.get('source_device')}->{summary.get('expert_device')} "
        f"tokens={summary.get('token_count')} "
        f"experts={summary.get('expert_count')} "
        f"max_layer_abs_error={summary.get('max_layer_abs_error')} "
        f"max_logit_abs_error={summary.get('max_logit_abs_error')}"
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


def _cmd_throughput_scaling_simulate(args: argparse.Namespace) -> int:
    try:
        levels = parse_concurrency_levels(args.concurrency_levels)
        result = simulate_throughput_scaling(
            plan_id=args.plan_id,
            concurrency_levels=levels,
            contracted_min_concurrency=args.contracted_min_concurrency,
            saturation_concurrency=args.saturation_concurrency,
            planner_bound_fraction=args.planner_bound_fraction,
            throughput_efficiency_floor=args.throughput_efficiency_floor,
            sum_node_ideal_tokens_s=args.sum_node_ideal_tokens_s,
            saturated_pipeline_tokens_s=args.saturated_pipeline_tokens_s,
            planner_bias_fraction=args.planner_bias_fraction,
            jitter_fraction=args.jitter_fraction,
        )
    except ValueError as exc:
        print(f"throughput scaling simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "throughput scaling simulate: "
        f"rows={summary['row_count']} "
        f"saturates_at={summary['observed_saturation_concurrency']} "
        f"planner_error={summary['max_abs_planner_error_fraction']:.3f} "
        f"efficiency={summary['throughput_efficiency_at_contract']:.3f}"
    )
    return 0


def _cmd_pipeline_correctness_probe(args: argparse.Namespace) -> int:
    try:
        prompts = parse_prompts_json(args.prompts_json, args.vocab_size)
        result = run_pipeline_correctness_probe(
            backend=args.backend,
            torch_python=args.torch_python,
            source_device=args.source_device,
            destination_device=args.destination_device,
            dtype=args.dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            new_tokens=args.new_tokens,
            prompts=prompts,
            tolerance=args.tolerance,
            logical_source_host=args.logical_source_host,
            logical_destination_host=args.logical_destination_host,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        print(f"pipeline correctness-probe: {exc}")
        return 2
    write_json(args.out, result)
    if not result.get("measured"):
        print(
            "pipeline correctness probe unavailable: "
            f"backend={result.get('backend')} error={result.get('error')}"
        )
        return 2
    validation = validate_pipeline_correctness_probe(args.out)
    if not validation["ok"]:
        print("pipeline correctness probe invalid: " + "; ".join(validation["errors"]))
        return 2
    summary = validation["summary"]
    suffix = " accelerator" if summary.get("accelerator_measured") else " reference"
    print(
        "pipeline correctness probe:"
        f"{suffix} backend={summary.get('backend')} "
        f"{summary.get('source_device')}->{summary.get('destination_device')} "
        f"tokens_generated={summary.get('tokens_generated')} "
        f"max_abs_error={summary.get('max_abs_error')}"
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


def _cmd_program_governance_simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_program_governance(
            plan_id=args.plan_id,
            report_date=args.report_date,
            plan_version=args.plan_version,
            current_gate=args.current_gate,
        )
    except ValueError as exc:
        print(f"program governance-simulate: {exc}")
        return 2
    write_json(args.out, result)
    summary = result["summary"]
    print(
        "program governance-simulate: "
        f"decisions={summary['decision_count']} "
        f"controls={summary['control_count']} "
        f"dec005_pending={summary['dec005_pending']} "
        f"g1_ready={summary['g1_gate_ready']} "
        f"status_drift={summary['status_drift_controlled']}"
    )
    return 0


def _cmd_program_phase3_proxy_gate(args: argparse.Namespace) -> int:
    try:
        result = build_phase3_proxy_gate_packet(
            args.endpoint_artifact,
            packet_date=args.date,
            outcome=args.outcome,
            accepted_by=args.accepted_by,
        )
    except (OSError, ValueError) as exc:
        print(f"program phase3-proxy-gate: {exc}")
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    validation = validate_phase3_proxy_gate_packet(result)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    if validation["ok"]:
        print(
            "wrote Phase 3 proxy gate packet: "
            f"{args.out} proxy_passed={summary['phase3_proxy_passed']} "
            f"formal_g3_passed={summary['formal_g3_passed']} "
            f"checks={summary['passed_count']}/{summary['check_count']}"
            f"{suffix}"
        )
        return 0
    print(
        "wrote invalid Phase 3 proxy gate packet: "
        f"{args.out} errors=" + "; ".join(validation["errors"])
    )
    return 1


def _cmd_program_phase4_resilience_gate(args: argparse.Namespace) -> int:
    try:
        result = build_phase4_resilience_gate_packet(
            args.resilience_artifact,
            args.replication_artifact,
            args.ops_artifact,
            packet_date=args.date,
            outcome=args.outcome,
            accepted_by=args.accepted_by,
            proxy_hardware_name=args.proxy_hardware_name,
            proxy_devices=[item.strip() for item in args.proxy_devices.split(",") if item.strip()],
            proxy_logical_hosts=[item.strip() for item in args.proxy_logical_hosts.split(",") if item.strip()],
        )
    except (OSError, ValueError) as exc:
        print(f"program phase4-resilience-gate: {exc}")
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    if args.runbook_out:
        Path(args.runbook_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.runbook_out).write_text(
            render_phase4_t4_runbook_markdown(result["runbook"]),
            encoding="utf-8",
        )
    validation = validate_phase4_resilience_gate_packet(result)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    if validation["ok"]:
        print(
            "wrote Phase 4 resilience proxy gate packet: "
            f"{args.out} proxy_passed={summary['phase4_proxy_passed']} "
            f"formal_g4_passed={summary['formal_g4_passed']} "
            f"checks={summary['passed_count']}/{summary['check_count']} "
            f"runbook_scenarios={summary['runbook_scenario_count']}"
            f"{suffix}"
        )
        return 0
    print(
        "wrote invalid Phase 4 resilience proxy gate packet: "
        f"{args.out} errors=" + "; ".join(validation["errors"])
    )
    return 1


def _cmd_program_phase5_ga_gate(args: argparse.Namespace) -> int:
    try:
        result = build_phase5_ga_gate_packet(
            args.ops_artifact,
            args.onboarding_artifact,
            args.benchmark_ledger,
            args.phase4_artifact,
            packet_date=args.date,
            outcome=args.outcome,
            accepted_by=args.accepted_by,
            proxy_hardware_name=args.proxy_hardware_name,
            proxy_devices=[item.strip() for item in args.proxy_devices.split(",") if item.strip()],
            proxy_logical_hosts=[item.strip() for item in args.proxy_logical_hosts.split(",") if item.strip()],
        )
    except (OSError, ValueError) as exc:
        print(f"program phase5-ga-gate: {exc}")
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    if args.runbook_out:
        Path(args.runbook_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.runbook_out).write_text(
            render_phase5_g5_runbook_markdown(result["runbook"]),
            encoding="utf-8",
        )
    validation = validate_phase5_ga_gate_packet(result)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    if validation["ok"]:
        print(
            "wrote Phase 5 GA proxy gate packet: "
            f"{args.out} proxy_passed={summary['phase5_proxy_passed']} "
            f"formal_g5_passed={summary['formal_g5_passed']} "
            f"checks={summary['passed_count']}/{summary['check_count']} "
            f"runbook_scenarios={summary['runbook_scenario_count']}"
            f"{suffix}"
        )
        return 0
    print(
        "wrote invalid Phase 5 GA proxy gate packet: "
        f"{args.out} errors=" + "; ".join(validation["errors"])
    )
    return 1


def _cmd_program_g1_evidence_packet(args: argparse.Namespace) -> int:
    try:
        result = build_g1_evidence_packet(
            args.bundle,
            packet_date=args.date,
            plan_version=args.plan_version,
        )
    except (OSError, ValueError) as exc:
        print(f"program g1-evidence-packet: {exc}")
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(result["markdown"], encoding="utf-8")
    validation = validate_g1_evidence_packet_fixture(result)
    summary = validation["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    if validation["ok"]:
        print(
            "wrote G1 evidence packet: "
            f"{args.out} machine_complete={summary['machine_complete']} "
            f"g1_ready={summary['g1_gate_ready']} "
            f"closure_blockers={summary['closure_blocker_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL G1 evidence packet: " + "; ".join(validation["errors"]))
    return 2


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




def _cmd_program_local_accelerator_smoke(args: argparse.Namespace) -> int:
    try:
        result = run_local_accelerator_smoke(
            out_dir=args.out_dir,
            torch_python=args.torch_python,
            expert_backend=args.expert_backend,
            expert_device=args.expert_device,
            expert_dtype=args.expert_dtype,
            expert_iterations=args.expert_iterations,
            expert_warmup=args.expert_warmup,
            expert_batch_tokens=args.expert_batch_tokens,
            expert_hidden_dim=args.expert_hidden_dim,
            expert_intermediate_dim=args.expert_intermediate_dim,
            expert_count=args.expert_count,
            expert_top_k=args.expert_top_k,
            expert_tolerance=args.expert_tolerance,
            include_activation_transfer=not args.skip_activation_transfer,
            transfer_backend=args.transfer_backend,
            transfer_source_device=args.transfer_source_device,
            transfer_destination_device=args.transfer_destination_device,
            transfer_dtype=args.transfer_dtype,
            transfer_iterations=args.transfer_iterations,
            transfer_warmup=args.transfer_warmup,
            transfer_payload_bytes=args.transfer_payload_mib * 1024 * 1024,
            transfer_tolerance=args.transfer_tolerance,
            include_pipeline_correctness=not args.skip_pipeline_correctness,
            pipeline_backend=args.pipeline_backend,
            pipeline_source_device=args.pipeline_source_device,
            pipeline_destination_device=args.pipeline_destination_device,
            pipeline_dtype=args.pipeline_dtype,
            pipeline_iterations=args.pipeline_iterations,
            pipeline_warmup=args.pipeline_warmup,
            pipeline_vocab_size=args.pipeline_vocab_size,
            pipeline_hidden_dim=args.pipeline_hidden_dim,
            pipeline_new_tokens=args.pipeline_new_tokens,
            pipeline_tolerance=args.pipeline_tolerance,
            include_moe_parity=not args.skip_moe_parity,
            moe_backend=args.moe_backend,
            moe_source_device=args.moe_source_device,
            moe_expert_device=args.moe_expert_device,
            moe_dtype=args.moe_dtype,
            moe_iterations=args.moe_iterations,
            moe_warmup=args.moe_warmup,
            moe_token_count=args.moe_token_count,
            moe_hidden_dim=args.moe_hidden_dim,
            moe_intermediate_dim=args.moe_intermediate_dim,
            moe_vocab_size=args.moe_vocab_size,
            moe_expert_count=args.moe_expert_count,
            moe_top_k=args.moe_top_k,
            moe_tolerance=args.moe_tolerance,
            logical_source_host=args.logical_source_host,
            logical_destination_host=args.logical_destination_host,
            require_accelerator=not args.allow_reference,
            timeout_s=args.timeout_s,
        )
    except (OSError, ValueError) as exc:
        print(f"program local-accelerator-smoke: {exc}")
        return 2
    validation = validate_local_accelerator_smoke(result["artifacts"]["validation"])
    summary = result["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "local accelerator smoke: "
        f"bundle={result['bundle']}; "
        f"checks={summary['passed_count']}/{summary['check_count']} passed; "
        f"expert_accelerator={summary['expert_accelerator_measured']}; "
        f"transfer_accelerator={summary['activation_transfer_accelerator_measured']}; "
        f"pipeline_accelerator={summary['pipeline_correctness_accelerator_measured']}; "
        f"moe_accelerator={summary['moe_parity_accelerator_measured']}; "
        f"gate_evidence={summary['g2_g3_gate_evidence']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2


def _cmd_program_local_http_serving_smoke(args: argparse.Namespace) -> int:
    try:
        result = run_local_http_serving_smoke(
            out=args.out,
            host=args.host,
            port=args.port,
            plan_id=args.plan_id,
            plan_hash=args.plan_hash,
            request_id=args.request_id,
            model=args.model,
            max_tokens=args.max_tokens,
            auth_token=args.auth_token,
            max_inflight=args.max_inflight,
            backpressure_delay_ms=args.backpressure_delay_ms,
            timeout_s=args.timeout_s,
            backend_mode=args.backend_mode,
            enable_tls=args.enable_tls,
            enable_mtls=args.enable_mtls,
            include_activation_transfer_probe=args.include_activation_transfer_probe,
            activation_transfer_backend=args.activation_transfer_backend,
            activation_transfer_torch_python=args.activation_transfer_torch_python,
            activation_transfer_source_device=args.activation_transfer_source_device,
            activation_transfer_destination_device=args.activation_transfer_destination_device,
            activation_transfer_dtype=args.activation_transfer_dtype,
            activation_transfer_iterations=args.activation_transfer_iterations,
            activation_transfer_warmup=args.activation_transfer_warmup,
            activation_transfer_payload_bytes=args.activation_transfer_payload_mib * 1024 * 1024,
            activation_transfer_tolerance=args.activation_transfer_tolerance,
            activation_transfer_logical_source_host=args.activation_transfer_logical_source_host,
            activation_transfer_logical_destination_host=args.activation_transfer_logical_destination_host,
            activation_transfer_timeout_s=args.activation_transfer_timeout_s,
            include_runtime_probes=args.include_runtime_probes,
            runtime_probe_backend=args.runtime_probe_backend,
            runtime_probe_torch_python=args.runtime_probe_torch_python,
            runtime_probe_source_device=args.runtime_probe_source_device,
            runtime_probe_destination_device=args.runtime_probe_destination_device,
            runtime_probe_dtype=args.runtime_probe_dtype,
            runtime_probe_iterations=args.runtime_probe_iterations,
            runtime_probe_warmup=args.runtime_probe_warmup,
            runtime_probe_tolerance=args.runtime_probe_tolerance,
            runtime_probe_logical_source_host=args.runtime_probe_logical_source_host,
            runtime_probe_logical_destination_host=args.runtime_probe_logical_destination_host,
            runtime_probe_timeout_s=args.runtime_probe_timeout_s,
            pipeline_probe_vocab_size=args.pipeline_probe_vocab_size,
            pipeline_probe_hidden_dim=args.pipeline_probe_hidden_dim,
            pipeline_probe_new_tokens=args.pipeline_probe_new_tokens,
            moe_probe_token_count=args.moe_probe_token_count,
            moe_probe_hidden_dim=args.moe_probe_hidden_dim,
            moe_probe_intermediate_dim=args.moe_probe_intermediate_dim,
            moe_probe_vocab_size=args.moe_probe_vocab_size,
            moe_probe_expert_count=args.moe_probe_expert_count,
            moe_probe_top_k=args.moe_probe_top_k,
            include_target_fixture_execution_probe=args.include_target_fixture_execution_probe,
            target_fixture_execution_backend=args.target_fixture_execution_backend,
            target_fixture_execution_torch_python=args.target_fixture_execution_torch_python,
            target_fixture_execution_device=args.target_fixture_execution_device,
            target_fixture_execution_dtype=args.target_fixture_execution_dtype,
            target_fixture_execution_iterations=args.target_fixture_execution_iterations,
            target_fixture_execution_warmup=args.target_fixture_execution_warmup,
            target_fixture_execution_vocab_size=args.target_fixture_execution_vocab_size,
            target_fixture_execution_new_tokens=args.target_fixture_execution_new_tokens,
            target_fixture_execution_stop_token_id=args.target_fixture_execution_stop_token_id,
            target_fixture_execution_tolerance=args.target_fixture_execution_tolerance,
            target_fixture_execution_logical_host=args.target_fixture_execution_logical_host,
            target_fixture_execution_timeout_s=args.target_fixture_execution_timeout_s,
        )
    except (OSError, ValueError) as exc:
        print(f"program local-http-serving-smoke: {exc}")
        return 2
    validation = validate_local_http_serving_smoke(args.out)
    summary = result["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "local HTTP serving smoke: "
        f"artifact={args.out}; "
        f"checks={summary['passed_count']}/{summary['check_count']} passed; "
        f"endpoint={summary['endpoint']}; "
        f"sse_chunks={summary['sse_chunk_count']}; "
        f"auth_reject={summary['endpoint_auth_rejected']}; "
        f"tls={summary['tls_enabled']}; "
        f"mtls={summary['mtls_enabled']}; "
        f"backpressure={summary['backpressure_rejected']}; "
        f"lifecycle={summary['lifecycle_all_released']}; "
        f"target_fixture={summary['target_fixture_parity']}; "
        f"activation_transfer={summary['activation_transfer_probe_ok']}; "
        f"runtime_probes={summary['runtime_probes_included']}; "
        f"pipeline_accelerator={summary['pipeline_correctness_accelerator_measured']}; "
        f"moe_accelerator={summary['moe_layer_parity_accelerator_measured']}; "
        f"target_fixture_execution={summary['target_fixture_execution_probe_ok']}; "
        f"plan_reject={summary['plan_integrity_rejected']}; "
        f"target_model_parity={summary['target_model_parity']}; "
        f"gate_evidence={summary['g2_g3_gate_evidence']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2


def _cmd_program_local_serving_smoke(args: argparse.Namespace) -> int:
    try:
        result = run_local_serving_smoke(
            out_dir=args.out_dir,
            torch_python=args.torch_python,
            plan_id=args.plan_id,
            request_id=args.request_id,
            model=args.model,
            stream=not args.no_stream,
            max_tokens=args.max_tokens,
            include_pipeline_correctness=not args.skip_pipeline_correctness,
            pipeline_backend=args.pipeline_backend,
            pipeline_source_device=args.pipeline_source_device,
            pipeline_destination_device=args.pipeline_destination_device,
            pipeline_dtype=args.pipeline_dtype,
            pipeline_iterations=args.pipeline_iterations,
            pipeline_warmup=args.pipeline_warmup,
            pipeline_vocab_size=args.pipeline_vocab_size,
            pipeline_hidden_dim=args.pipeline_hidden_dim,
            pipeline_new_tokens=args.pipeline_new_tokens,
            pipeline_tolerance=args.pipeline_tolerance,
            include_moe_parity=not args.skip_moe_parity,
            moe_backend=args.moe_backend,
            moe_source_device=args.moe_source_device,
            moe_expert_device=args.moe_expert_device,
            moe_dtype=args.moe_dtype,
            moe_iterations=args.moe_iterations,
            moe_warmup=args.moe_warmup,
            moe_token_count=args.moe_token_count,
            moe_hidden_dim=args.moe_hidden_dim,
            moe_intermediate_dim=args.moe_intermediate_dim,
            moe_vocab_size=args.moe_vocab_size,
            moe_expert_count=args.moe_expert_count,
            moe_top_k=args.moe_top_k,
            moe_tolerance=args.moe_tolerance,
            include_target_fixture_probe=not args.skip_target_fixture_probe,
            target_fixture_backend=args.target_fixture_backend,
            target_fixture_device=args.target_fixture_device,
            target_fixture_dtype=args.target_fixture_dtype,
            target_fixture_iterations=args.target_fixture_iterations,
            target_fixture_warmup=args.target_fixture_warmup,
            target_fixture_vocab_size=args.target_fixture_vocab_size,
            target_fixture_new_tokens=args.target_fixture_new_tokens,
            target_fixture_stop_token_id=args.target_fixture_stop_token_id,
            target_fixture_tolerance=args.target_fixture_tolerance,
            logical_source_host=args.logical_source_host,
            logical_destination_host=args.logical_destination_host,
            require_accelerator=not args.allow_reference,
            timeout_s=args.timeout_s,
        )
    except (OSError, ValueError) as exc:
        print(f"program local-serving-smoke: {exc}")
        return 2
    validation = validate_local_serving_smoke(result["artifacts"]["validation"])
    summary = result["summary"]
    suffix = ""
    if validation["warnings"]:
        suffix = "; warnings: " + "; ".join(validation["warnings"])
    print(
        "local serving smoke: "
        f"bundle={result['bundle']}; "
        f"checks={summary['passed_count']}/{summary['check_count']} passed; "
        f"serving_adapter={summary['serving_adapter_valid']}; "
        f"pipeline_accelerator={summary['pipeline_correctness_accelerator_measured']}; "
        f"moe_accelerator={summary['moe_parity_accelerator_measured']}; "
        f"target_fixture_accelerator={summary['target_fixture_accelerator_measured']}; "
        f"target_fixture={summary['target_fixture_generated_text']}; "
        f"live_http={summary['live_http_endpoint']}; "
        f"target_model_parity={summary['target_model_parity']}; "
        f"gate_evidence={summary['g2_g3_gate_evidence']}"
        f"{suffix}"
    )
    return 0 if validation["ok"] else 2


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


def _cmd_test_stage_host(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/stage_host"
    result = validate_stage_host(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS stage-host: {fixture} "
            f"events={summary['event_count']} "
            f"boundary_ops={summary['boundary_op_count']} "
            f"max_abs_error={summary['max_abs_error']}"
            f"{suffix}"
        )
        return 0
    print("FAIL stage-host: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_serving_adapter(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/serving_adapter"
    result = validate_serving_adapter(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS serving-adapter: {fixture} "
            f"surfaces={summary['surface_count']} "
            f"chunks={summary['openai_chunk_count']} "
            f"tool_calls={summary['tool_call_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL serving-adapter: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_local_http_serving_smoke(args: argparse.Namespace) -> int:
    fixture = args.fixture or args.out or "/tmp/fornax_local_http_serving_smoke_test.json"
    if not args.fixture:
        run_local_http_serving_smoke(out=fixture)
    result = validate_local_http_serving_smoke(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS local-http-serving-smoke: {fixture} "
            f"checks={summary['passed_count']}/{summary['check_count']} "
            f"sse_chunks={summary['sse_chunk_count']} "
            f"auth_reject={summary['endpoint_auth_rejected']} "
            f"backpressure={summary['backpressure_rejected']} "
            f"lifecycle={summary['lifecycle_all_released']} "
            f"target_fixture={summary['target_fixture_parity']} "
            f"plan_reject={summary['plan_integrity_rejected']} "
            f"gate_evidence={summary['g2_g3_gate_evidence']}"
            f"{suffix}"
        )
        return 0
    print("FAIL local-http-serving-smoke: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_local_serving_smoke(args: argparse.Namespace) -> int:
    if args.fixture:
        fixture = args.fixture
    else:
        fixture = args.out or "/tmp/fornax_local_serving_smoke_reference_test"
        run_local_serving_smoke(
            out_dir=fixture,
            pipeline_backend="cpu-stdlib",
            pipeline_iterations=1,
            pipeline_warmup=0,
            pipeline_hidden_dim=4,
            pipeline_new_tokens=2,
            moe_backend="cpu-stdlib",
            moe_iterations=1,
            moe_warmup=0,
            moe_token_count=2,
            moe_hidden_dim=4,
            moe_intermediate_dim=6,
            moe_vocab_size=11,
            moe_expert_count=2,
            moe_top_k=1,
            target_fixture_backend="cpu-stdlib",
            target_fixture_iterations=1,
            target_fixture_warmup=0,
            target_fixture_new_tokens=4,
            require_accelerator=False,
        )
    result = validate_local_serving_smoke(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS local-serving-smoke: {fixture} "
            f"checks={summary['passed_count']}/{summary['check_count']} "
            f"pipeline_accelerator={summary['pipeline_correctness_accelerator_measured']} "
            f"moe_accelerator={summary['moe_parity_accelerator_measured']} "
            f"target_fixture_accelerator={summary['target_fixture_accelerator_measured']} "
            f"target_fixture={summary['target_fixture_generated_text']} "
            f"t2_smoke={summary['t2_smoke_passed']}"
            f"{suffix}"
        )
        return 0
    print("FAIL local-serving-smoke: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_state_ownership(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/state_ownership"
    result = validate_state_ownership(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS state-ownership: {fixture} "
            f"resources={summary['resource_count']} "
            f"transitions={summary['transition_count']} "
            f"released={summary['terminal_released_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL state-ownership: " + "; ".join(result["errors"]))
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


def _cmd_test_metrics_ledger(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/metrics_ledger"
    result = validate_metrics_ledger(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS metrics-ledger: {fixture} "
            f"samples={summary['sample_count']} "
            f"alerts={summary['alert_count']} "
            f"max_queue={summary['max_queue_depth_observed']}"
            f"{suffix}"
        )
        return 0
    print("FAIL metrics-ledger: " + "; ".join(result["errors"]))
    return 1




def _cmd_test_trace_ledger(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/trace_ledger"
    result = validate_trace_ledger(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS trace-ledger: {fixture} "
            f"components={summary['component_count']} "
            f"spans={summary['span_count']} "
            f"events={summary['event_count']} "
            f"edges={summary['required_edge_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL trace-ledger: " + "; ".join(result["errors"]))
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


def _cmd_test_trust_boundary(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/trust_boundary"
    result = validate_trust_boundary(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS trust-boundary: {fixture} "
            f"identities={summary['identity_count']} "
            f"accepted={summary['accepted_auth_count']} "
            f"rejected={summary['rejected_auth_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL trust-boundary: " + "; ".join(result["errors"]))
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


def _cmd_test_moe_migration(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/moe_migration"
    result = validate_moe_hot_expert_migration(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS moe-migration: {fixture} "
            f"hot_expert={summary['hot_expert_id']} "
            f"remote_reduction={summary['remote_token_copy_reduction']} "
            f"post_remote_batches={summary['post_remote_batches']}"
            f"{suffix}"
        )
        return 0
    print("FAIL moe-migration: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_remote_expert_probe(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/remote_expert_batch"
    result = validate_remote_expert_batch_probe(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS remote-expert-probe: {fixture} "
            f"backend={summary['backend']} "
            f"accelerator={summary['accelerator_measured']} "
            f"devices={summary['source_device']}->{summary['expert_device']} "
            f"batches={summary['remote_batches']}"
            f"{suffix}"
        )
        return 0
    print("FAIL remote-expert-probe: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_moe_parity_probe(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/moe_layer_parity"
    result = validate_moe_layer_parity_probe(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS moe-parity-probe: {fixture} "
            f"backend={summary['backend']} "
            f"accelerator={summary['accelerator_measured']} "
            f"devices={summary['source_device']}->{summary['expert_device']} "
            f"max_logit_abs_error={summary['max_logit_abs_error']}"
            f"{suffix}"
        )
        return 0
    print("FAIL moe-parity-probe: " + "; ".join(result["errors"]))
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



def _cmd_test_activation_transfer_probe(args: argparse.Namespace) -> int:
    if not args.fixture:
        print("FAIL activation-transfer-probe: --fixture is required")
        return 1
    result = validate_activation_transfer_probe(args.fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS activation-transfer-probe: {args.fixture} "
            f"backend={summary['backend']} "
            f"accelerator={summary['accelerator_measured']} "
            f"devices={summary['source_device']}->{summary['destination_device']} "
            f"bandwidth_gib_s={summary['bandwidth_gib_s']:.3f}"
            f"{suffix}"
        )
        return 0
    print("FAIL activation-transfer-probe: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_stage_replication(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/stage_replication"
    result = validate_stage_replication(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS stage-replication: {fixture} "
            f"replicas={summary['replica_count']} "
            f"microbatches={summary['microbatch_count']} "
            f"speedup={summary['speedup']:.3f}"
            f"{suffix}"
        )
        return 0
    print("FAIL stage-replication: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_resilience_replay(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/resilience_replay"
    result = validate_resilience_replay(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS resilience-replay: {fixture} "
            f"requests={summary['request_count']} "
            f"replayed={summary['replayed_request_count']} "
            f"dropped={summary['dropped_request_count']} "
            f"max_replay_delay_s={summary['max_replay_delay_s']}"
            f"{suffix}"
        )
        return 0
    print("FAIL resilience-replay: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_ops_lifecycle(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/ops_lifecycle"
    result = validate_ops_lifecycle(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS ops-lifecycle: {fixture} "
            f"actions={summary['action_count']} "
            f"events={summary['event_count']} "
            f"dropped={summary['dropped_in_flight_count']} "
            f"rollback={summary['rollback_verified']} "
            f"node_replace={summary['node_replace_verified']}"
            f"{suffix}"
        )
        return 0
    print("FAIL ops-lifecycle: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_program_governance(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/program_governance"
    result = validate_program_governance(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS program-governance: {fixture} "
            f"decisions={summary['decision_count']} "
            f"controls={summary['control_count']} "
            f"dec005_pending={summary['dec005_pending']} "
            f"g1_ready={summary['g1_gate_ready']} "
            f"status_drift={summary['status_drift_controlled']}"
            f"{suffix}"
        )
        return 0
    print("FAIL program-governance: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_onboarding_methodology(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/onboarding_methodology"
    result = validate_onboarding_methodology(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS onboarding-methodology: {fixture} "
            f"tracks={summary['track_count']} "
            f"docs={summary['document_count']} "
            f"glossary={summary['glossary_term_count']} "
            f"lab_reference={summary['lab_reference_required']} "
            f"product_ga={summary['product_ga_complete']}"
            f"{suffix}"
        )
        return 0
    print("FAIL onboarding-methodology: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_throughput_scaling(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/throughput_scaling"
    result = validate_throughput_scaling(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS throughput-scaling: {fixture} "
            f"rows={summary['row_count']} "
            f"saturates_at={summary['observed_saturation_concurrency']} "
            f"planner_error={summary['max_abs_planner_error_fraction']:.3f} "
            f"efficiency={summary['throughput_efficiency_at_contract']:.3f}"
            f"{suffix}"
        )
        return 0
    print("FAIL throughput-scaling: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_pipeline_correctness_probe(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/pipeline_correctness"
    result = validate_pipeline_correctness_probe(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS pipeline-correctness-probe: {fixture} "
            f"backend={summary['backend']} "
            f"accelerator={summary['accelerator_measured']} "
            f"devices={summary['source_device']}->{summary['destination_device']} "
            f"tokens_generated={summary['tokens_generated']}"
            f"{suffix}"
        )
        return 0
    print("FAIL pipeline-correctness-probe: " + "; ".join(result["errors"]))
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


def _cmd_test_phase3_proxy_gate(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/phase3_proxy_gate"
    result = validate_phase3_proxy_gate(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS phase3-proxy-gate: {fixture} "
            f"proxy_passed={summary['phase3_proxy_passed']} "
            f"formal_g3_passed={summary['formal_g3_passed']} "
            f"checks={summary['passed_count']}/{summary['check_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL phase3-proxy-gate: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_phase4_resilience_gate(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/phase4_resilience_gate"
    result = validate_phase4_resilience_gate(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS phase4-resilience-gate: {fixture} "
            f"proxy_passed={summary['phase4_proxy_passed']} "
            f"formal_g4_passed={summary['formal_g4_passed']} "
            f"checks={summary['passed_count']}/{summary['check_count']} "
            f"runbook_scenarios={summary['runbook_scenario_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL phase4-resilience-gate: " + "; ".join(result["errors"]))
    return 1


def _cmd_test_phase5_ga_gate(args: argparse.Namespace) -> int:
    fixture = args.fixture or "fornax/golden_vectors/phase5_ga_gate"
    result = validate_phase5_ga_gate(fixture)
    if result["ok"]:
        suffix = ""
        if result["warnings"]:
            suffix = "; warnings: " + "; ".join(result["warnings"])
        summary = result["summary"]
        print(
            f"PASS phase5-ga-gate: {fixture} "
            f"proxy_passed={summary['phase5_proxy_passed']} "
            f"formal_g5_passed={summary['formal_g5_passed']} "
            f"checks={summary['passed_count']}/{summary['check_count']} "
            f"runbook_scenarios={summary['runbook_scenario_count']}"
            f"{suffix}"
        )
        return 0
    print("FAIL phase5-ga-gate: " + "; ".join(result["errors"]))
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
    if args.test_name == "stage-host":
        return _cmd_test_stage_host(args)
    if args.test_name == "serving-adapter":
        return _cmd_test_serving_adapter(args)
    if args.test_name == "local-serving-smoke":
        return _cmd_test_local_serving_smoke(args)
    if args.test_name == "local-http-serving-smoke":
        return _cmd_test_local_http_serving_smoke(args)
    if args.test_name == "state-ownership":
        return _cmd_test_state_ownership(args)
    if args.test_name == "engine-simulation":
        return _cmd_test_engine_simulation(args)
    if args.test_name == "observability":
        return _cmd_test_observability(args)
    if args.test_name == "metrics-ledger":
        return _cmd_test_metrics_ledger(args)
    if args.test_name == "trace-ledger":
        return _cmd_test_trace_ledger(args)
    if args.test_name == "worker-contract":
        return _cmd_test_worker_contract(args)
    if args.test_name == "transport-contract":
        return _cmd_test_transport_contract(args)
    if args.test_name == "trust-boundary":
        return _cmd_test_trust_boundary(args)
    if args.test_name == "moe-runtime":
        return _cmd_test_moe_runtime(args)
    if args.test_name == "moe-migration":
        return _cmd_test_moe_migration(args)
    if args.test_name == "remote-expert-probe":
        return _cmd_test_remote_expert_probe(args)
    if args.test_name == "moe-parity-probe":
        return _cmd_test_moe_parity_probe(args)
    if args.test_name == "model-support":
        return _cmd_test_model_support(args)
    if args.test_name == "continuous-batching":
        return _cmd_test_continuous_batching(args)
    if args.test_name == "scheduler-contract":
        return _cmd_test_scheduler_contract(args)
    if args.test_name == "stage-replication":
        return _cmd_test_stage_replication(args)
    if args.test_name == "resilience-replay":
        return _cmd_test_resilience_replay(args)
    if args.test_name == "ops-lifecycle":
        return _cmd_test_ops_lifecycle(args)
    if args.test_name == "onboarding-methodology":
        return _cmd_test_onboarding_methodology(args)
    if args.test_name == "program-governance":
        return _cmd_test_program_governance(args)
    if args.test_name == "backend-coverage":
        return _cmd_test_backend_coverage(args)
    if args.test_name == "phase3-proxy-gate":
        return _cmd_test_phase3_proxy_gate(args)
    if args.test_name == "phase4-resilience-gate":
        return _cmd_test_phase4_resilience_gate(args)
    if args.test_name == "phase5-ga-gate":
        return _cmd_test_phase5_ga_gate(args)
    if args.test_name == "benchmark-ledger":
        return _cmd_test_benchmark_ledger(args)
    if args.test_name == "expert-mlp-probe":
        return _cmd_test_expert_mlp_probe(args)
    if args.test_name == "activation-transfer-probe":
        return _cmd_test_activation_transfer_probe(args)
    if args.test_name == "pipeline-correctness-probe":
        return _cmd_test_pipeline_correctness_probe(args)
    if args.test_name == "throughput-scaling":
        return _cmd_test_throughput_scaling(args)
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

    transfer_probe = accelerator_sub.add_parser("activation-transfer-probe")
    transfer_probe.add_argument("--out", required=True)
    transfer_probe.add_argument("--backend", choices=["cpu-stdlib", "torch"], default="torch")
    transfer_probe.add_argument("--torch-python")
    transfer_probe.add_argument("--source-device", default="cuda:0")
    transfer_probe.add_argument("--destination-device", default="cuda:1")
    transfer_probe.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    transfer_probe.add_argument("--iterations", type=int, default=20)
    transfer_probe.add_argument("--warmup", type=int, default=3)
    transfer_probe.add_argument("--payload-mib", type=int, default=16)
    transfer_probe.add_argument("--tolerance", type=float, default=0.0)
    transfer_probe.add_argument("--logical-source-host", default="logical-host-0")
    transfer_probe.add_argument("--logical-destination-host", default="logical-host-1")
    transfer_probe.add_argument("--timeout-s", type=float, default=180.0)
    transfer_probe.set_defaults(func=_cmd_accelerator_activation_transfer_probe)

    target_fixture_probe = accelerator_sub.add_parser("target-fixture-probe")
    target_fixture_probe.add_argument("--out", required=True)
    target_fixture_probe.add_argument("--backend", choices=["cpu-stdlib", "torch"], default="torch")
    target_fixture_probe.add_argument("--torch-python")
    target_fixture_probe.add_argument("--device", default="cuda:0")
    target_fixture_probe.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    target_fixture_probe.add_argument("--iterations", type=int, default=20)
    target_fixture_probe.add_argument("--warmup", type=int, default=3)
    target_fixture_probe.add_argument("--vocab-size", type=int, default=17)
    target_fixture_probe.add_argument("--new-tokens", type=int, default=4)
    target_fixture_probe.add_argument("--prompt-tokens-json")
    target_fixture_probe.add_argument("--stop-token-id", type=int, default=9)
    target_fixture_probe.add_argument("--tolerance", type=float, default=1e-4)
    target_fixture_probe.add_argument("--logical-host", default="logical-host-0")
    target_fixture_probe.add_argument("--timeout-s", type=float, default=180.0)
    target_fixture_probe.set_defaults(func=_cmd_accelerator_target_fixture_probe)

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

    governance = program_sub.add_parser("governance-simulate")
    governance.add_argument("--out", required=True)
    governance.add_argument("--plan-id", default="program-governance-plan")
    governance.add_argument("--report-date", default="2026-06-22")
    governance.add_argument("--plan-version", default="v3")
    governance.add_argument("--current-gate", default="G1")
    governance.set_defaults(func=_cmd_program_governance_simulate)

    phase3_proxy = program_sub.add_parser("phase3-proxy-gate")
    phase3_proxy.add_argument("--endpoint-artifact", required=True)
    phase3_proxy.add_argument("--out", required=True)
    phase3_proxy.add_argument("--date")
    phase3_proxy.add_argument("--outcome", choices=["PROCEED", "ITERATE", "NARROW", "KILL"], default="PROCEED")
    phase3_proxy.add_argument("--accepted-by", default="operator")
    phase3_proxy.set_defaults(func=_cmd_program_phase3_proxy_gate)

    phase4_resilience = program_sub.add_parser("phase4-resilience-gate")
    phase4_resilience.add_argument("--resilience-artifact", required=True)
    phase4_resilience.add_argument("--replication-artifact", required=True)
    phase4_resilience.add_argument("--ops-artifact", required=True)
    phase4_resilience.add_argument("--out", required=True)
    phase4_resilience.add_argument("--runbook-out")
    phase4_resilience.add_argument("--date")
    phase4_resilience.add_argument("--outcome", choices=["PROCEED", "ITERATE", "NARROW", "KILL"], default="PROCEED")
    phase4_resilience.add_argument("--accepted-by", default="operator")
    phase4_resilience.add_argument("--proxy-hardware-name", default="NVIDIA H100 80GB HBM3")
    phase4_resilience.add_argument("--proxy-devices", default="cuda:0,cuda:1")
    phase4_resilience.add_argument("--proxy-logical-hosts", default="logical-host-0,logical-host-1")
    phase4_resilience.set_defaults(func=_cmd_program_phase4_resilience_gate)

    phase5_ga = program_sub.add_parser("phase5-ga-gate")
    phase5_ga.add_argument("--ops-artifact", required=True)
    phase5_ga.add_argument("--onboarding-artifact", required=True)
    phase5_ga.add_argument("--benchmark-ledger", required=True)
    phase5_ga.add_argument("--phase4-artifact", required=True)
    phase5_ga.add_argument("--out", required=True)
    phase5_ga.add_argument("--runbook-out")
    phase5_ga.add_argument("--date")
    phase5_ga.add_argument("--outcome", choices=["PROCEED", "ITERATE", "NARROW", "KILL"], default="PROCEED")
    phase5_ga.add_argument("--accepted-by", default="operator")
    phase5_ga.add_argument("--proxy-hardware-name", default="NVIDIA H100 80GB HBM3")
    phase5_ga.add_argument("--proxy-devices", default="cuda:0,cuda:1")
    phase5_ga.add_argument("--proxy-logical-hosts", default="logical-host-0,logical-host-1")
    phase5_ga.set_defaults(func=_cmd_program_phase5_ga_gate)

    g1_packet = program_sub.add_parser("g1-evidence-packet")
    g1_packet.add_argument("--bundle", required=True)
    g1_packet.add_argument("--out", required=True)
    g1_packet.add_argument("--markdown-out")
    g1_packet.add_argument("--date")
    g1_packet.add_argument("--plan-version", default="v3")
    g1_packet.set_defaults(func=_cmd_program_g1_evidence_packet)

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

    local_accel = program_sub.add_parser("local-accelerator-smoke")
    local_accel.add_argument("--out-dir", required=True)
    local_accel.add_argument("--torch-python")
    local_accel.add_argument("--expert-backend", choices=["cpu-stdlib", "torch"], default="torch")
    local_accel.add_argument("--expert-device", default="cuda:0")
    local_accel.add_argument("--expert-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    local_accel.add_argument("--expert-iterations", type=int, default=25)
    local_accel.add_argument("--expert-warmup", type=int, default=3)
    local_accel.add_argument("--expert-batch-tokens", type=int, default=8)
    local_accel.add_argument("--expert-hidden-dim", type=int, default=64)
    local_accel.add_argument("--expert-intermediate-dim", type=int, default=128)
    local_accel.add_argument("--expert-count", type=int, default=4)
    local_accel.add_argument("--expert-top-k", type=int, default=2)
    local_accel.add_argument("--expert-tolerance", type=float, default=0.1)
    local_accel.add_argument("--skip-activation-transfer", action="store_true")
    local_accel.add_argument("--transfer-backend", choices=["cpu-stdlib", "torch"], default="torch")
    local_accel.add_argument("--transfer-source-device", default="cuda:0")
    local_accel.add_argument("--transfer-destination-device", default="cuda:1")
    local_accel.add_argument("--transfer-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    local_accel.add_argument("--transfer-iterations", type=int, default=20)
    local_accel.add_argument("--transfer-warmup", type=int, default=3)
    local_accel.add_argument("--transfer-payload-mib", type=int, default=16)
    local_accel.add_argument("--transfer-tolerance", type=float, default=0.0)
    local_accel.add_argument("--skip-pipeline-correctness", action="store_true")
    local_accel.add_argument("--pipeline-backend", choices=["cpu-stdlib", "torch"], default="torch")
    local_accel.add_argument("--pipeline-source-device", default="cuda:0")
    local_accel.add_argument("--pipeline-destination-device", default="cuda:1")
    local_accel.add_argument("--pipeline-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    local_accel.add_argument("--pipeline-iterations", type=int, default=5)
    local_accel.add_argument("--pipeline-warmup", type=int, default=1)
    local_accel.add_argument("--pipeline-vocab-size", type=int, default=17)
    local_accel.add_argument("--pipeline-hidden-dim", type=int, default=16)
    local_accel.add_argument("--pipeline-new-tokens", type=int, default=4)
    local_accel.add_argument("--pipeline-tolerance", type=float, default=1e-4)
    local_accel.add_argument("--skip-moe-parity", action="store_true")
    local_accel.add_argument("--moe-backend", choices=["cpu-stdlib", "torch"], default="torch")
    local_accel.add_argument("--moe-source-device", default="cuda:0")
    local_accel.add_argument("--moe-expert-device", default="cuda:1")
    local_accel.add_argument("--moe-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    local_accel.add_argument("--moe-iterations", type=int, default=5)
    local_accel.add_argument("--moe-warmup", type=int, default=1)
    local_accel.add_argument("--moe-token-count", type=int, default=4)
    local_accel.add_argument("--moe-hidden-dim", type=int, default=16)
    local_accel.add_argument("--moe-intermediate-dim", type=int, default=32)
    local_accel.add_argument("--moe-vocab-size", type=int, default=17)
    local_accel.add_argument("--moe-expert-count", type=int, default=4)
    local_accel.add_argument("--moe-top-k", type=int, default=2)
    local_accel.add_argument("--moe-tolerance", type=float, default=1e-4)
    local_accel.add_argument("--logical-source-host", default="logical-host-0")
    local_accel.add_argument("--logical-destination-host", default="logical-host-1")
    local_accel.add_argument("--allow-reference", action="store_true")
    local_accel.add_argument("--timeout-s", type=float, default=180.0)
    local_accel.set_defaults(func=_cmd_program_local_accelerator_smoke)

    local_http_serving = program_sub.add_parser("local-http-serving-smoke")
    local_http_serving.add_argument("--out", required=True)
    local_http_serving.add_argument("--host", default="127.0.0.1")
    local_http_serving.add_argument("--port", type=int, default=0)
    local_http_serving.add_argument("--plan-id", default="local-http-serving-plan")
    local_http_serving.add_argument("--plan-hash", default="sha256:local-http-serving-plan")
    local_http_serving.add_argument("--request-id", default="local-http-serving-request")
    local_http_serving.add_argument("--model", default="qwen3-moe-class-target")
    local_http_serving.add_argument("--max-tokens", type=int, default=64)
    local_http_serving.add_argument("--auth-token", default="local-smoke-token")
    local_http_serving.add_argument("--max-inflight", type=int, default=1)
    local_http_serving.add_argument("--backpressure-delay-ms", type=int, default=250)
    local_http_serving.add_argument("--backend-mode", choices=["adapter", "target-fixture"], default="adapter")
    local_http_serving.add_argument("--enable-tls", action="store_true")
    local_http_serving.add_argument("--enable-mtls", action="store_true")
    local_http_serving.add_argument("--include-activation-transfer-probe", action="store_true")
    local_http_serving.add_argument("--activation-transfer-backend", choices=["cpu-stdlib", "torch"], default="cpu-stdlib")
    local_http_serving.add_argument("--activation-transfer-torch-python")
    local_http_serving.add_argument("--activation-transfer-source-device", default="cuda:0")
    local_http_serving.add_argument("--activation-transfer-destination-device", default="cuda:1")
    local_http_serving.add_argument("--activation-transfer-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    local_http_serving.add_argument("--activation-transfer-iterations", type=int, default=20)
    local_http_serving.add_argument("--activation-transfer-warmup", type=int, default=3)
    local_http_serving.add_argument("--activation-transfer-payload-mib", type=int, default=16)
    local_http_serving.add_argument("--activation-transfer-tolerance", type=float, default=0.0)
    local_http_serving.add_argument("--activation-transfer-logical-source-host", default="logical-host-0")
    local_http_serving.add_argument("--activation-transfer-logical-destination-host", default="logical-host-1")
    local_http_serving.add_argument("--activation-transfer-timeout-s", type=float, default=180.0)
    local_http_serving.add_argument("--include-runtime-probes", action="store_true")
    local_http_serving.add_argument("--runtime-probe-backend", choices=["cpu-stdlib", "torch"], default="cpu-stdlib")
    local_http_serving.add_argument("--runtime-probe-torch-python")
    local_http_serving.add_argument("--runtime-probe-source-device", default="cuda:0")
    local_http_serving.add_argument("--runtime-probe-destination-device", default="cuda:1")
    local_http_serving.add_argument("--runtime-probe-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    local_http_serving.add_argument("--runtime-probe-iterations", type=int, default=5)
    local_http_serving.add_argument("--runtime-probe-warmup", type=int, default=1)
    local_http_serving.add_argument("--runtime-probe-tolerance", type=float, default=1e-4)
    local_http_serving.add_argument("--runtime-probe-logical-source-host", default="logical-host-0")
    local_http_serving.add_argument("--runtime-probe-logical-destination-host", default="logical-host-1")
    local_http_serving.add_argument("--runtime-probe-timeout-s", type=float, default=180.0)
    local_http_serving.add_argument("--pipeline-probe-vocab-size", type=int, default=17)
    local_http_serving.add_argument("--pipeline-probe-hidden-dim", type=int, default=16)
    local_http_serving.add_argument("--pipeline-probe-new-tokens", type=int, default=4)
    local_http_serving.add_argument("--moe-probe-token-count", type=int, default=4)
    local_http_serving.add_argument("--moe-probe-hidden-dim", type=int, default=16)
    local_http_serving.add_argument("--moe-probe-intermediate-dim", type=int, default=32)
    local_http_serving.add_argument("--moe-probe-vocab-size", type=int, default=17)
    local_http_serving.add_argument("--moe-probe-expert-count", type=int, default=4)
    local_http_serving.add_argument("--moe-probe-top-k", type=int, default=2)
    local_http_serving.add_argument("--include-target-fixture-execution-probe", action="store_true")
    local_http_serving.add_argument("--target-fixture-execution-backend", choices=["cpu-stdlib", "torch"], default="cpu-stdlib")
    local_http_serving.add_argument("--target-fixture-execution-torch-python")
    local_http_serving.add_argument("--target-fixture-execution-device", default="cuda:0")
    local_http_serving.add_argument("--target-fixture-execution-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    local_http_serving.add_argument("--target-fixture-execution-iterations", type=int, default=5)
    local_http_serving.add_argument("--target-fixture-execution-warmup", type=int, default=1)
    local_http_serving.add_argument("--target-fixture-execution-vocab-size", type=int, default=17)
    local_http_serving.add_argument("--target-fixture-execution-new-tokens", type=int, default=4)
    local_http_serving.add_argument("--target-fixture-execution-stop-token-id", type=int, default=9)
    local_http_serving.add_argument("--target-fixture-execution-tolerance", type=float, default=1e-4)
    local_http_serving.add_argument("--target-fixture-execution-logical-host", default="logical-host-0")
    local_http_serving.add_argument("--target-fixture-execution-timeout-s", type=float, default=180.0)
    local_http_serving.add_argument("--timeout-s", type=float, default=5.0)
    local_http_serving.set_defaults(func=_cmd_program_local_http_serving_smoke)

    local_serving = program_sub.add_parser("local-serving-smoke")
    local_serving.add_argument("--out-dir", required=True)
    local_serving.add_argument("--torch-python")
    local_serving.add_argument("--plan-id", default="local-serving-smoke-plan")
    local_serving.add_argument("--request-id", default="local-serving-smoke-request")
    local_serving.add_argument("--model", default="qwen3-moe-class-target")
    local_serving.add_argument("--no-stream", action="store_true")
    local_serving.add_argument("--max-tokens", type=int, default=64)
    local_serving.add_argument("--skip-pipeline-correctness", action="store_true")
    local_serving.add_argument("--pipeline-backend", choices=["cpu-stdlib", "torch"], default="torch")
    local_serving.add_argument("--pipeline-source-device", default="cuda:0")
    local_serving.add_argument("--pipeline-destination-device", default="cuda:1")
    local_serving.add_argument("--pipeline-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    local_serving.add_argument("--pipeline-iterations", type=int, default=5)
    local_serving.add_argument("--pipeline-warmup", type=int, default=1)
    local_serving.add_argument("--pipeline-vocab-size", type=int, default=17)
    local_serving.add_argument("--pipeline-hidden-dim", type=int, default=16)
    local_serving.add_argument("--pipeline-new-tokens", type=int, default=4)
    local_serving.add_argument("--pipeline-tolerance", type=float, default=1e-4)
    local_serving.add_argument("--skip-moe-parity", action="store_true")
    local_serving.add_argument("--moe-backend", choices=["cpu-stdlib", "torch"], default="torch")
    local_serving.add_argument("--moe-source-device", default="cuda:0")
    local_serving.add_argument("--moe-expert-device", default="cuda:1")
    local_serving.add_argument("--moe-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    local_serving.add_argument("--moe-iterations", type=int, default=5)
    local_serving.add_argument("--moe-warmup", type=int, default=1)
    local_serving.add_argument("--moe-token-count", type=int, default=4)
    local_serving.add_argument("--moe-hidden-dim", type=int, default=16)
    local_serving.add_argument("--moe-intermediate-dim", type=int, default=32)
    local_serving.add_argument("--moe-vocab-size", type=int, default=17)
    local_serving.add_argument("--moe-expert-count", type=int, default=4)
    local_serving.add_argument("--moe-top-k", type=int, default=2)
    local_serving.add_argument("--moe-tolerance", type=float, default=1e-4)
    local_serving.add_argument("--skip-target-fixture-probe", action="store_true")
    local_serving.add_argument("--target-fixture-backend", choices=["cpu-stdlib", "torch"], default="torch")
    local_serving.add_argument("--target-fixture-device", default="cuda:0")
    local_serving.add_argument("--target-fixture-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    local_serving.add_argument("--target-fixture-iterations", type=int, default=5)
    local_serving.add_argument("--target-fixture-warmup", type=int, default=1)
    local_serving.add_argument("--target-fixture-vocab-size", type=int, default=17)
    local_serving.add_argument("--target-fixture-new-tokens", type=int, default=4)
    local_serving.add_argument("--target-fixture-stop-token-id", type=int, default=9)
    local_serving.add_argument("--target-fixture-tolerance", type=float, default=1e-4)
    local_serving.add_argument("--logical-source-host", default="logical-host-0")
    local_serving.add_argument("--logical-destination-host", default="logical-host-1")
    local_serving.add_argument("--allow-reference", action="store_true")
    local_serving.add_argument("--timeout-s", type=float, default=180.0)
    local_serving.set_defaults(func=_cmd_program_local_serving_smoke)

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

    observability = sub.add_parser("observability")
    observability_sub = observability.add_subparsers(
        dest="observability_command", required=True
    )
    metrics = observability_sub.add_parser("metrics-simulate")
    metrics.add_argument("--out", required=True)
    metrics.add_argument("--plan-id", default="metrics-ledger-plan")
    metrics.add_argument("--request-id", default="req-metrics-ledger")
    metrics.add_argument("--max-queue-depth", type=int, default=4)
    metrics.add_argument("--max-inflight", type=int, default=3)
    metrics.add_argument("--kv-page-limit", type=int, default=16)
    metrics.add_argument("--memory-limit-bytes", type=int, default=80 * 1024 * 1024 * 1024)
    metrics.add_argument("--memory-warning-fraction", type=float, default=0.85)
    metrics.add_argument("--memory-critical-fraction", type=float, default=0.95)
    metrics.add_argument("--sample-period-ms", type=float, default=10.0)
    metrics.set_defaults(func=_cmd_observability_metrics_simulate)

    trace = observability_sub.add_parser("trace-simulate")
    trace.add_argument("--out", required=True)
    trace.add_argument("--plan-id", default="trace-ledger-plan")
    trace.add_argument("--request-id", default="req-trace-ledger")
    trace.add_argument("--trace-id", default="trace-trace-ledger")
    trace.set_defaults(func=_cmd_observability_trace_simulate)

    runtime = sub.add_parser("runtime")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    stage_host = runtime_sub.add_parser("stage-host-simulate")
    stage_host.add_argument("--out", required=True)
    stage_host.add_argument("--plan-id", default="stage-host-plan")
    stage_host.add_argument("--request-id", default="req-stage-host")
    stage_host.add_argument("--stage-id", default="stage-1")
    stage_host.add_argument("--logical-host-id", default="logical-host-1")
    stage_host.add_argument("--predecessor-stage-id", default="stage-0")
    stage_host.add_argument("--successor-stage-id", default="stage-2")
    stage_host.add_argument("--layer-start", type=int, default=12)
    stage_host.add_argument("--layer-count", type=int, default=2)
    stage_host.add_argument("--token-count", type=int, default=3)
    stage_host.add_argument("--hidden-dim", type=int, default=4)
    stage_host.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16"
    )
    stage_host.add_argument("--tolerance", type=float, default=0.0)
    stage_host.set_defaults(func=_cmd_runtime_stage_host_simulate)

    serving = sub.add_parser("serving")
    serving_sub = serving.add_subparsers(dest="serving_command", required=True)
    serving_adapter = serving_sub.add_parser("adapter-simulate")
    serving_adapter.add_argument("--out", required=True)
    serving_adapter.add_argument("--plan-id", default="serving-adapter-plan")
    serving_adapter.add_argument("--request-id", default="req-serving-adapter-001")
    serving_adapter.add_argument("--model", default="qwen3-moe-class-target")
    serving_adapter.add_argument("--max-tokens", type=int, default=64)
    serving_adapter.add_argument("--no-stream", dest="stream", action="store_false")
    serving_adapter.add_argument("--template-hash", default="sha256:" + "a" * 64)
    serving_adapter.add_argument("--tokenizer-hash", default="sha256:" + "b" * 64)
    serving_adapter.set_defaults(stream=True, func=_cmd_serving_adapter_simulate)

    state_ownership = serving_sub.add_parser("state-ownership-simulate")
    state_ownership.add_argument("--out", required=True)
    state_ownership.add_argument("--plan-id", default="state-ownership-plan")
    state_ownership.add_argument("--request-id", default="req-state-ownership")
    state_ownership.add_argument(
        "--cancel-request-id", default="req-state-ownership-cancel"
    )
    state_ownership.add_argument("--model", default="qwen3-moe-class-target")
    state_ownership.set_defaults(func=_cmd_serving_state_ownership_simulate)

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

    trust_boundary = transport_sub.add_parser("trust-boundary-simulate")
    trust_boundary.add_argument("--out", required=True)
    trust_boundary.add_argument("--plan-id", default="trust-boundary-plan")
    trust_boundary.add_argument("--request-id", default="req-trust-boundary")
    trust_boundary.add_argument("--plan-hash", default="sha256:trust-boundary-plan")
    trust_boundary.add_argument("--cluster-id", default="fornax-sim-cluster")
    trust_boundary.add_argument("--token-ttl-s", type=float, default=30.0)
    trust_boundary.set_defaults(func=_cmd_transport_trust_boundary_simulate)

    replication = sub.add_parser("replication")
    replication_sub = replication.add_subparsers(dest="replication_command", required=True)
    replication_simulate = replication_sub.add_parser("simulate")
    replication_simulate.add_argument("--out", required=True)
    replication_simulate.add_argument("--plan-id", default="stage-replication-plan")
    replication_simulate.add_argument("--bottleneck-stage-index", type=int, default=1)
    replication_simulate.add_argument("--microbatch-token-counts", default="4,4,3,3,2,2")
    replication_simulate.add_argument("--baseline-replica-id", default="stage-1-replica-0")
    replication_simulate.add_argument("--added-replica-id", default="stage-1-replica-1")
    replication_simulate.add_argument("--baseline-stage-time-s-per-token", type=float, default=0.014)
    replication_simulate.add_argument("--replicated-stage-time-s-per-token", type=float, default=0.014)
    replication_simulate.add_argument("--transfer-overhead-s", type=float, default=0.001)
    replication_simulate.add_argument("--speedup-floor", type=float, default=1.25)
    replication_simulate.add_argument("--tolerance", type=float, default=0.0)
    replication_simulate.set_defaults(func=_cmd_replication_simulate)

    resilience = sub.add_parser("resilience")
    resilience_sub = resilience.add_subparsers(
        dest="resilience_command", required=True
    )
    replay_simulate = resilience_sub.add_parser("replay-simulate")
    replay_simulate.add_argument("--out", required=True)
    replay_simulate.add_argument("--plan-id", default="resilience-replay-plan")
    replay_simulate.add_argument("--failed-node-id", default="logical-host-1")
    replay_simulate.add_argument("--replay-node-id", default="logical-host-0")
    replay_simulate.add_argument("--checkpoint-token-index", type=int, default=2)
    replay_simulate.add_argument("--node-loss-time-s", type=float, default=0.050)
    replay_simulate.add_argument("--replay-delay-s", type=float, default=0.010)
    replay_simulate.add_argument("--token-time-s", type=float, default=0.006)
    replay_simulate.add_argument("--max-replay-delay-s", type=float, default=0.025)
    replay_simulate.add_argument("--vocab-size", type=int, default=97)
    replay_simulate.set_defaults(func=_cmd_resilience_replay_simulate)

    ops = sub.add_parser("ops")
    ops_sub = ops.add_subparsers(dest="ops_command", required=True)
    lifecycle = ops_sub.add_parser("lifecycle-simulate")
    lifecycle.add_argument("--out", required=True)
    lifecycle.add_argument("--plan-id", default="ops-lifecycle-plan")
    lifecycle.add_argument("--cluster-id", default="fornax-sim-cluster")
    lifecycle.add_argument("--model-id", default="qwen3-moe-class-target")
    lifecycle.add_argument("--initial-version", default="v0.1.0")
    lifecycle.add_argument("--target-version", default="v0.2.0")
    lifecycle.add_argument("--node-ids", default="logical-host-0,logical-host-1")
    lifecycle.add_argument("--replacement-node-id", default="logical-host-2")
    lifecycle.add_argument("--in-flight-requests", type=int, default=4)
    lifecycle.set_defaults(func=_cmd_ops_lifecycle_simulate)

    onboarding = ops_sub.add_parser("onboarding-simulate")
    onboarding.add_argument("--out", required=True)
    onboarding.add_argument("--plan-id", default="onboarding-methodology-plan")
    onboarding.add_argument("--package-id", default="fornax-operator-onboarding")
    onboarding.add_argument("--benchmark-id", default="fornax-benchmark-of-record-methodology")
    onboarding.set_defaults(func=_cmd_ops_onboarding_simulate)

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

    moe_migration = moe_sub.add_parser("migration-simulate")
    moe_migration.add_argument("--out", required=True)
    moe_migration.add_argument("--plan-id", default="moe-migration-simulated-plan")
    moe_migration.add_argument("--request-id", default="req-moe-migration-simulated")
    moe_migration.add_argument("--plan-hash", default="sha256:moe-migration-simulated-plan")
    moe_migration.add_argument("--token-count", type=int, default=6)
    moe_migration.add_argument("--hidden-dim", type=int, default=16)
    moe_migration.add_argument("--intermediate-dim", type=int, default=32)
    moe_migration.add_argument("--vocab-size", type=int, default=17)
    moe_migration.add_argument("--expert-count", type=int, default=4)
    moe_migration.add_argument("--top-k", type=int, default=2)
    moe_migration.add_argument("--hot-expert-id", type=int, default=1)
    moe_migration.add_argument("--migration-hotness-threshold", type=float, default=0.45)
    moe_migration.add_argument("--tolerance", type=float, default=0.0)
    moe_migration.add_argument("--logical-source-host", default="logical-host-0")
    moe_migration.add_argument("--logical-expert-host", default="logical-host-1")
    moe_migration.set_defaults(func=_cmd_moe_migration_simulate)

    remote_probe = moe_sub.add_parser("remote-expert-probe")
    remote_probe.add_argument("--out", required=True)
    remote_probe.add_argument("--backend", choices=["cpu-stdlib", "torch"], default="cpu-stdlib")
    remote_probe.add_argument("--torch-python")
    remote_probe.add_argument("--source-device", default="cuda:0")
    remote_probe.add_argument("--expert-device", default="cuda:1")
    remote_probe.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    remote_probe.add_argument("--iterations", type=int, default=5)
    remote_probe.add_argument("--warmup", type=int, default=1)
    remote_probe.add_argument("--token-count", type=int, default=4)
    remote_probe.add_argument("--hidden-dim", type=int, default=16)
    remote_probe.add_argument("--intermediate-dim", type=int, default=32)
    remote_probe.add_argument("--expert-id", type=int, default=5)
    remote_probe.add_argument("--tolerance", type=float, default=0.0)
    remote_probe.add_argument("--logical-source-host", default="logical-host-0")
    remote_probe.add_argument("--logical-expert-host", default="logical-host-1")
    remote_probe.add_argument("--timeout-s", type=float, default=180.0)
    remote_probe.set_defaults(func=_cmd_moe_remote_expert_probe)

    parity_probe = moe_sub.add_parser("parity-probe")
    parity_probe.add_argument("--out", required=True)
    parity_probe.add_argument("--backend", choices=["cpu-stdlib", "torch"], default="cpu-stdlib")
    parity_probe.add_argument("--torch-python")
    parity_probe.add_argument("--source-device", default="cuda:0")
    parity_probe.add_argument("--expert-device", default="cuda:1")
    parity_probe.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parity_probe.add_argument("--iterations", type=int, default=5)
    parity_probe.add_argument("--warmup", type=int, default=1)
    parity_probe.add_argument("--token-count", type=int, default=4)
    parity_probe.add_argument("--hidden-dim", type=int, default=16)
    parity_probe.add_argument("--intermediate-dim", type=int, default=32)
    parity_probe.add_argument("--vocab-size", type=int, default=17)
    parity_probe.add_argument("--expert-count", type=int, default=4)
    parity_probe.add_argument("--top-k", type=int, default=2)
    parity_probe.add_argument("--tolerance", type=float, default=0.0)
    parity_probe.add_argument("--logical-source-host", default="logical-host-0")
    parity_probe.add_argument("--logical-expert-host", default="logical-host-1")
    parity_probe.add_argument("--timeout-s", type=float, default=180.0)
    parity_probe.set_defaults(func=_cmd_moe_parity_probe)

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

    pipeline = sub.add_parser("pipeline")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command", required=True)
    pipeline_probe = pipeline_sub.add_parser("correctness-probe")
    pipeline_probe.add_argument("--out", required=True)
    pipeline_probe.add_argument("--backend", choices=["cpu-stdlib", "torch"], default="cpu-stdlib")
    pipeline_probe.add_argument("--torch-python")
    pipeline_probe.add_argument("--source-device", default="cuda:0")
    pipeline_probe.add_argument("--destination-device", default="cuda:1")
    pipeline_probe.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    pipeline_probe.add_argument("--iterations", type=int, default=5)
    pipeline_probe.add_argument("--warmup", type=int, default=1)
    pipeline_probe.add_argument("--vocab-size", type=int, default=17)
    pipeline_probe.add_argument("--hidden-dim", type=int, default=16)
    pipeline_probe.add_argument("--new-tokens", type=int, default=4)
    pipeline_probe.add_argument("--prompts-json")
    pipeline_probe.add_argument("--tolerance", type=float, default=0.0)
    pipeline_probe.add_argument("--logical-source-host", default="logical-host-0")
    pipeline_probe.add_argument("--logical-destination-host", default="logical-host-1")
    pipeline_probe.add_argument("--timeout-s", type=float, default=180.0)
    pipeline_probe.set_defaults(func=_cmd_pipeline_correctness_probe)

    throughput = sub.add_parser("throughput")
    throughput_sub = throughput.add_subparsers(dest="throughput_command", required=True)
    throughput_scaling = throughput_sub.add_parser("scaling-simulate")
    throughput_scaling.add_argument("--out", required=True)
    throughput_scaling.add_argument("--plan-id", default="throughput-scaling-plan")
    throughput_scaling.add_argument("--concurrency-levels", default="1,2,4,8,16,32")
    throughput_scaling.add_argument("--contracted-min-concurrency", type=int, default=16)
    throughput_scaling.add_argument("--saturation-concurrency", type=int, default=8)
    throughput_scaling.add_argument("--planner-bound-fraction", type=float, default=0.20)
    throughput_scaling.add_argument("--throughput-efficiency-floor", type=float, default=0.60)
    throughput_scaling.add_argument("--sum-node-ideal-tokens-s", type=float, default=45.0)
    throughput_scaling.add_argument("--saturated-pipeline-tokens-s", type=float, default=30.0)
    throughput_scaling.add_argument("--planner-bias-fraction", type=float, default=0.08)
    throughput_scaling.add_argument("--jitter-fraction", type=float, default=0.015)
    throughput_scaling.set_defaults(func=_cmd_throughput_scaling_simulate)

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
            "stage-host",
            "serving-adapter",
            "local-serving-smoke",
            "local-http-serving-smoke",
            "state-ownership",
            "engine-simulation",
            "observability",
            "metrics-ledger",
            "trace-ledger",
            "worker-contract",
            "transport-contract",
            "trust-boundary",
            "moe-runtime",
            "moe-migration",
            "remote-expert-probe",
            "moe-parity-probe",
            "model-support",
            "continuous-batching",
            "scheduler-contract",
            "stage-replication",
            "resilience-replay",
            "ops-lifecycle",
            "onboarding-methodology",
            "program-governance",
            "backend-coverage",
            "phase3-proxy-gate",
            "benchmark-ledger",
            "expert-mlp-probe",
            "activation-transfer-probe",
            "pipeline-correctness-probe",
            "throughput-scaling",
            "phase4-resilience-gate",
            "phase5-ga-gate",
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
