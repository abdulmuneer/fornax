from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fornax.accelerator_probe import (
    run_cpu_activation_transfer_probe,
    run_cpu_expert_mlp_probe,
    validate_activation_transfer_probe_fixture,
    validate_expert_mlp_probe_fixture,
)
from fornax.apple_probe import (
    apple_probe_template,
    render_apple_role_decision_draft,
    render_simulated_apple_role_decision,
    simulated_apple_probe_artifact,
    validate_apple_probe_artifact,
)
from fornax.backend_coverage import (
    render_backend_coverage_report,
    validate_backend_coverage_contract,
    validate_backend_coverage_fixture,
)
from fornax.benchmark import benchmark_from_plan, run_tiny_expert_mlp_benchmark
from fornax.benchmark_ledger import (
    append_benchmark_ledger_record,
    build_benchmark_ledger_record,
    validate_benchmark_ledger,
    validate_benchmark_ledger_record,
)
from fornax.calibration import run_cpu_memory_copy_probe, run_local_calibration
from fornax.continuous_batching import (
    simulate_continuous_batching,
    validate_continuous_batching,
    validate_continuous_batching_fixture,
)
from fornax.contracts import TargetContractError, load_target_contract
from fornax.doctor import inspect_phase0_bundle
from fornax.engine_seam import (
    validate_engine_seam_contract,
    validate_engine_seam_fixture,
)
from fornax.engine_simulation import (
    simulated_engine_contract,
    validate_engine_simulation,
    validate_engine_simulation_fixture,
)
from fornax.golden import run_golden_plans
from fornax.g1_evidence_packet import (
    build_g1_evidence_packet,
    validate_g1_evidence_packet_fixture,
)
from fornax.g1_review import render_g1_gate_review_draft
from fornax.io import load_inventory, read_json, write_json
from fornax.local_accelerator_smoke import (
    run_local_accelerator_smoke,
    validate_local_accelerator_smoke,
    validate_local_accelerator_smoke_fixture,
)
from fornax.local_http_serving_smoke import (
    run_local_http_serving_smoke,
    validate_local_http_serving_smoke,
    validate_local_http_serving_smoke_fixture,
)
from fornax.local_serving_smoke import (
    run_local_serving_smoke,
    validate_local_serving_smoke,
    validate_local_serving_smoke_fixture,
)
from fornax.inventory import build_logical_cluster_inventory
from fornax.inventory.local import (
    collect_local_inventory,
    parse_nvidia_smi_csv,
    probe_declared_links,
)
from fornax.moe import (
    simulated_moe_contract,
    validate_moe_contract,
    validate_moe_contract_fixture,
)
from fornax.moe_migration import (
    simulated_moe_hot_expert_migration,
    validate_moe_hot_expert_migration,
    validate_moe_hot_expert_migration_fixture,
)
from fornax.moe_parity import (
    run_cpu_moe_layer_parity_probe,
    validate_moe_layer_parity_probe_fixture,
)
from fornax.model_support import (
    render_model_support_matrix_report,
    simulated_model_support_matrix,
    validate_model_support_matrix,
    validate_model_support_matrix_fixture,
)
from fornax.metrics_ledger import (
    simulate_metrics_ledger,
    validate_metrics_ledger,
    validate_metrics_ledger_fixture,
)
from fornax.trace_ledger import (
    simulate_trace_ledger,
    validate_trace_ledger,
    validate_trace_ledger_fixture,
)
from fornax.network_security_spec import render_network_security_spec_draft
from fornax.pipeline_probe import (
    run_cpu_pipeline_correctness_probe,
    validate_pipeline_correctness_probe_fixture,
)
from fornax.observability import (
    validate_observability_contract,
    validate_observability_fixture,
)
from fornax.onboarding import (
    simulate_onboarding_methodology,
    validate_onboarding_methodology,
    validate_onboarding_methodology_fixture,
)
from fornax.ops_lifecycle import (
    simulate_ops_lifecycle,
    validate_ops_lifecycle,
    validate_ops_lifecycle_fixture,
)
from fornax.network_contract import (
    validate_network_contract,
    validate_network_contract_fixture,
)
from fornax.planner import Inventory, ModelSpec, Target, plan_placement
from fornax.phase0_status import render_phase0_status_report
from fornax.phase0_simulated_validation import run_phase0_simulated_validation
from fornax.t1_simulated_validation import run_t1_simulated_validation
from fornax.preflight import run_phase0_preflight
from fornax.program_governance import (
    simulate_program_governance,
    validate_program_governance,
    validate_program_governance_fixture,
)
from fornax.program_rebaseline import render_program_rebaseline_draft
from fornax.remote_expert_probe import (
    run_cpu_remote_expert_batch_probe,
    validate_remote_expert_batch_probe_fixture,
)
from fornax.resilience import (
    simulate_resilience_replay,
    validate_resilience_replay,
    validate_resilience_replay_fixture,
)
from fornax.runtime_format_spec import render_runtime_format_spec_draft
from fornax.serving import (
    simulate_serving_adapter,
    validate_serving_adapter,
    validate_serving_adapter_fixture,
)
from fornax.state_ownership import (
    simulate_state_ownership,
    validate_state_ownership,
    validate_state_ownership_fixture,
)
from fornax.stage_host import (
    simulate_stage_host,
    validate_stage_host,
    validate_stage_host_fixture,
)
from fornax.stage_replication import (
    simulate_stage_replication,
    validate_stage_replication,
    validate_stage_replication_fixture,
)
from fornax.scheduler import (
    simulate_scheduler,
    simulate_scheduler_from_paths,
    validate_scheduler_contract,
)
from fornax.runtime_format import (
    validate_runtime_format_golden,
    validate_runtime_format_manifest,
)
from fornax.simulate import simulation_result, summarize_request_trace
from fornax.substrate_adr import render_substrate_adr_draft
from fornax.target_contract import render_target_contract_draft
from fornax.throughput_scaling import (
    simulate_throughput_scaling,
    validate_throughput_scaling_fixture,
)
from fornax.transport import (
    simulated_transport_contract,
    validate_transport_contract,
    validate_transport_contract_fixture,
)
from fornax.trust_boundary import (
    simulate_trust_boundary,
    validate_trust_boundary,
    validate_trust_boundary_fixture,
)
from fornax.validation import validate_target_contract
from fornax.workers import (
    simulated_worker_contract,
    validate_worker_contract,
    validate_worker_contract_fixture,
)


def dense_model(num_layers: int = 4) -> ModelSpec:
    return ModelSpec.from_dict(
        {
            "hidden_dim": 1024,
            "num_layers": num_layers,
            "dtype_weight": "q4",
            "dtype_activation": "fp16",
            "layers": [
                {
                    "kind": "dense",
                    "weight_bytes": 1000000,
                    "active_flops_per_token": 1000000,
                }
                for _ in range(num_layers)
            ],
        }
    )


def measured_apple_probe(tokens_s: float = 12.0, threshold: float = 10.0) -> dict:
    return {
        "version": 1,
        "probe_kind": "apple-expert-mlp",
        "target_model": {
            "name": "qwen3-moe-target",
            "expert_mlp_shape": {"hidden": 2048, "intermediate": 8192},
            "quantization": "q4",
            "activation_dtype": "fp16",
        },
        "environment": {
            "hardware": "Mac Studio M3 Ultra 512GB",
            "os": "macOS 15.5",
            "max_build": "max-26.4.0",
            "mojo_build": "Mojo 1.0",
            "command": "fornax-apple-expert-mlp-probe --fixture fixture.json",
            "log_path": "/tmp/apple-probe.log",
            "profiler": "Instruments",
        },
        "probe": {
            "rank": 1,
            "measured": True,
            "local_to_target_mac": True,
            "input_fixture": "fixture-sha256",
            "output_checksum": "checksum",
        },
        "correctness": {
            "passed": True,
            "max_abs_error": 0.0001,
            "max_rel_error": 0.001,
            "tolerance_source": "runtime-format-and-invariants.md",
        },
        "throughput": {
            "measured": True,
            "tokens_s": tokens_s,
            "threshold_tokens_s": threshold,
            "warmup_iterations": 5,
            "measurement_iterations": 25,
            "thermal_notes": "steady state",
        },
        "decision": {"requested_role": "expert-worker", "demote_role": "capacity-only"},
    }


def two_gpu_source_inventory() -> dict:
    return {
        "nodes": [
            {
                "id": "gpu0",
                "host_id": "unit-host",
                "vendor": "nvidia",
                "runtime": "max",
                "device": "cuda:0",
                "name": "Unit H100",
                "mem_free_bytes": 64_000_000_000,
                "mem_total_bytes": 80_000_000_000,
                "compute_class": 400_000_000_000_000.0,
                "mem_bandwidth_bytes_s": 2_500_000_000_000.0,
                "supports_stage": True,
                "supports_expert_worker": True,
                "supports_kv": True,
                "supported_dtypes": ["fp16", "bf16"],
            },
            {
                "id": "gpu1",
                "host_id": "unit-host",
                "vendor": "nvidia",
                "runtime": "max",
                "device": "cuda:1",
                "name": "Unit H100",
                "mem_free_bytes": 64_000_000_000,
                "mem_total_bytes": 80_000_000_000,
                "compute_class": 400_000_000_000_000.0,
                "mem_bandwidth_bytes_s": 2_500_000_000_000.0,
                "supports_stage": True,
                "supports_expert_worker": True,
                "supports_kv": True,
                "supported_dtypes": ["fp16", "bf16"],
            },
        ],
        "links": [],
        "host_id": "unit-host",
        "source": "unit-test",
        "measured_fields": ["nvidia.memory_free_mib"],
        "estimated_fields": ["compute_class", "mem_bandwidth_bytes_s"],
        "collection_errors": [],
    }


def scheduler_fixture_plan() -> dict:
    return {
        "feasible": True,
        "stages": [
            {"index": 0, "layers": [0], "replicas": ["sim-gpu0"], "mode": "stage"},
            {"index": 1, "layers": [1], "replicas": ["sim-gpu1"], "mode": "stage"},
        ],
        "predicted": {
            "throughput_tok_s": 10.0,
            "per_request_latency_s": 0.1,
            "bubble_fraction": 0.1,
            "stage_effective_times_s": [0.01, 0.02],
        },
    }


def inventory_with_link(bandwidth: float = 12_500_000_000.0) -> Inventory:
    return Inventory.from_dict(
        {
            "nodes": [
                {
                    "id": "fast",
                    "vendor": "nvidia",
                    "runtime": "max",
                    "mem_free_bytes": 16_000_000,
                    "compute_class": 4_000_000_000_000.0,
                    "mem_bandwidth_bytes_s": 400_000_000_000.0,
                    "supports_stage": True,
                    "supports_expert_worker": True,
                    "supports_kv": True,
                    "supported_dtypes": ["fp16"],
                },
                {
                    "id": "slow",
                    "vendor": "cpu",
                    "runtime": "custom",
                    "mem_free_bytes": 16_000_000,
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
                    "a": "fast",
                    "b": "slow",
                    "bandwidth_bytes_s": bandwidth,
                    "latency_s": 0.00002,
                }
            ],
        }
    )


class FornaxPlannerTest(unittest.TestCase):

    def test_local_calibration_records_measured_cpu_probe(self) -> None:
        memory = run_cpu_memory_copy_probe(size_bytes=1024, iterations=2)
        self.assertTrue(memory["measured"])
        self.assertEqual(2048, memory["copied_bytes"])
        artifact = run_local_calibration(
            cpu_memory_bytes=1024,
            cpu_memory_iterations=2,
            cpu_compute_iterations=10,
            try_torch_cuda=False,
        )
        self.assertTrue(artifact["measured"])
        self.assertTrue(artifact["cpu_memory_copy"]["measured"])
        self.assertFalse(artifact["cuda_microprobe"]["measured"])
        self.assertIn("inventory_summary", artifact)
        self.assertEqual(
            {
                "try_torch_cuda": False,
                "torch_python": None,
                "cuda_matrix_dim": 512,
                "cuda_iterations": 10,
            },
            artifact["calibration_inputs"],
        )

    def test_local_calibration_records_external_torch_python_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            fake_python = Path(d) / "fake-torch-python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "cat <<'JSON'\n"
                '{"measured": false, "backend": "torch", '
                '"available": false, "error": "fake torch unavailable"}\n'
                "JSON\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            artifact = run_local_calibration(
                cpu_memory_bytes=1024,
                cpu_memory_iterations=1,
                cpu_compute_iterations=10,
                torch_python=str(fake_python),
                cuda_matrix_dim=8,
                cuda_iterations=1,
            )
        cuda = artifact["cuda_microprobe"]
        self.assertEqual("external_python", cuda["backend_mode"])
        self.assertEqual(str(fake_python), cuda["python_executable"])
        self.assertFalse(cuda["measured"])
        self.assertIn("torch", cuda["error"])
        self.assertEqual(str(fake_python), artifact["calibration_inputs"]["torch_python"])

    def test_program_rebaseline_unavailable_ker_warns_and_dates_schedule(self) -> None:
        result = render_program_rebaseline_draft(
            kickoff_date="2026-06-20", ker_status="unavailable", scope="pending"
        )
        markdown = result["markdown"]
        self.assertIn("2026-06-27 to 2026-07-18", markdown)
        self.assertIn("KER unavailable; G1 must choose NARROW", markdown)
        self.assertIn("Sponsor should narrow Apple role", "; ".join(result["warnings"]))
        self.assertEqual("unavailable", result["ker_status"])

    def test_program_rebaseline_rejects_unknown_ker_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "ker_status must be"):
            render_program_rebaseline_draft(ker_status="maybe-later")

    def test_g1_review_draft_flags_missing_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "bundle"
            fake_python = Path(d) / "fake-torch-python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "cat <<'JSON'\n"
                '{"measured": false, "backend": "torch", '
                '"available": false, "error": "fake torch unavailable"}\n'
                "JSON\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                benchmark_iterations=1,
                include_g1_drafts=True,
                include_calibration=True,
                calibration_torch_python=str(fake_python),
                substrate_pinned_build="max-26.4.0",
                kickoff_date="2026-06-20",
                ker_status="unavailable",
                scope="pending",
            )
            review = render_g1_gate_review_draft(
                bundle, review_date="2026-06-20", plan_version="v3"
            )
        self.assertFalse(review["machine_complete"])
        self.assertFalse(review["gate_ready"])
        self.assertEqual("ITERATE", review["recommended_outcome"])
        missing = "; ".join(review["machine_missing_criteria"])
        self.assertIn("Apple reversal trigger", missing)
        self.assertIn("golden-plan tests T0 green", missing)
        self.assertIn("missing golden-plans.json", review["markdown"])
        self.assertIn("missing G1-closable Apple probe", review["markdown"])

    def test_g1_review_draft_separates_machine_evidence_from_human_closure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "bundle"
            fake_python = Path(d) / "fake-torch-python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "cat <<'JSON'\n"
                '{"measured": false, "backend": "torch", '
                '"available": false, "error": "fake torch unavailable"}\n'
                "JSON\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                benchmark_iterations=1,
                include_g1_drafts=True,
                include_calibration=True,
                calibration_torch_python=str(fake_python),
                substrate_pinned_build="max-26.4.0",
                kickoff_date="2026-06-20",
                ker_status="unavailable",
                scope="pending",
            )
            probe_path = bundle / "apple-expert-mlp-probe.json"
            write_json(probe_path, measured_apple_probe())
            write_json(
                bundle / "apple-probe-validation.json",
                validate_apple_probe_artifact(measured_apple_probe()),
            )
            decision = render_apple_role_decision_draft(probe_path)
            (bundle / "apple-role-decision.md").write_text(
                decision["markdown"], encoding="utf-8"
            )
            write_json(
                bundle / "golden-plans.json",
                {"passed": True, "results": [{"name": "fixture", "passed": True}]},
            )
            review = render_g1_gate_review_draft(
                bundle, review_date="2026-06-20", plan_version="v3"
            )
        self.assertTrue(review["machine_complete"], review["machine_missing_criteria"])
        self.assertFalse(review["gate_ready"])
        self.assertEqual("SPONSOR_DECISION_REQUIRED", review["recommended_outcome"])
        blockers = "; ".join(review["closure_blockers"])
        self.assertIn("target-contract sign-off", blockers)
        self.assertIn("review sign-off for generated specs", blockers)
        self.assertIn("staffing sign-off", blockers)

    def test_apple_probe_valid_pass_recommends_expert_worker(self) -> None:
        result = validate_apple_probe_artifact(measured_apple_probe())
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["gate_closable"])
        self.assertEqual("expert-worker", result["recommended_role"])
        self.assertEqual("expert-worker-pass", result["outcome"])
        self.assertTrue(result["performance_passed"])

    def test_apple_probe_measured_throughput_miss_demotes_capacity_only(self) -> None:
        result = validate_apple_probe_artifact(
            measured_apple_probe(tokens_s=4.0, threshold=10.0)
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["gate_closable"])
        self.assertEqual("capacity-only", result["recommended_role"])
        self.assertEqual("capacity-only-demotion", result["outcome"])
        self.assertFalse(result["throughput_passed"])

    def test_simulated_apple_probe_is_development_only(self) -> None:
        artifact = simulated_apple_probe_artifact(
            target_model="qwen3-moe-target",
            pinned_build="max-26.4.0",
            recommended_role="capacity-only",
        )
        validation = validate_apple_probe_artifact(artifact)
        markdown = render_simulated_apple_role_decision(artifact)
        self.assertEqual("apple-expert-mlp-simulation", artifact["probe_kind"])
        self.assertFalse(artifact["decision"]["gate_closable"])
        self.assertFalse(validation["gate_closable"])
        self.assertEqual("undecided", validation["recommended_role"])
        self.assertIn("not G1 closure evidence", markdown)

    def test_apple_probe_template_is_not_gate_closable(self) -> None:
        result = validate_apple_probe_artifact(
            apple_probe_template(target_model="qwen3-moe-target")
        )
        self.assertFalse(result["valid"])
        self.assertFalse(result["gate_closable"])
        self.assertEqual("undecided", result["recommended_role"])
        self.assertIn("probe.measured", "; ".join(result["errors"]))

    def test_apple_role_decision_draft_records_demotion(self) -> None:
        probe = measured_apple_probe(tokens_s=4.0, threshold=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "apple_probe.json"
            write_json(path, probe)
            result = render_apple_role_decision_draft(path)
        self.assertEqual("capacity-only", result["validation"]["recommended_role"])
        self.assertIn("Recommended Apple role: `capacity-only`", result["markdown"])
        self.assertIn("Rank 1 local probe", result["markdown"])


    def test_cpu_expert_mlp_probe_validates_reference_not_accelerator(self) -> None:
        artifact = run_cpu_expert_mlp_probe(
            iterations=1,
            batch_tokens=2,
            hidden_dim=4,
            intermediate_dim=6,
            experts=3,
            top_k=2,
        )
        result = validate_expert_mlp_probe_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(artifact["measured"])
        self.assertFalse(artifact["accelerator_measured"])
        self.assertIn("not accelerator evidence", "; ".join(result["warnings"]))

    def test_expert_mlp_probe_rejects_false_accelerator_claim(self) -> None:
        artifact = run_cpu_expert_mlp_probe(iterations=1, batch_tokens=1)
        artifact["tier"] = "T2-single-node-accelerator"
        artifact["accelerator_measured"] = True
        result = validate_expert_mlp_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("hardware.device_type must be cuda", text)
        self.assertIn("cpu-stdlib backend cannot be accelerator_measured", text)

    def test_expert_mlp_probe_rejects_failed_correctness(self) -> None:
        artifact = run_cpu_expert_mlp_probe(iterations=1, batch_tokens=1)
        artifact["result"]["correctness_passed"] = False
        artifact["result"]["max_abs_error"] = artifact["config"]["tolerance"] + 1.0
        result = validate_expert_mlp_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("correctness_passed", "; ".join(result["errors"]))


    def test_local_accelerator_smoke_allows_reference_for_ci(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = run_local_accelerator_smoke(
                out_dir=d,
                expert_backend="cpu-stdlib",
                expert_iterations=1,
                expert_warmup=0,
                expert_batch_tokens=1,
                expert_hidden_dim=4,
                expert_intermediate_dim=6,
                expert_count=2,
                expert_top_k=1,
                include_activation_transfer=False,
                include_pipeline_correctness=True,
                pipeline_backend="cpu-stdlib",
                pipeline_iterations=1,
                pipeline_warmup=0,
                pipeline_hidden_dim=4,
                pipeline_new_tokens=2,
                include_moe_parity=True,
                moe_backend="cpu-stdlib",
                moe_iterations=1,
                moe_warmup=0,
                moe_token_count=2,
                moe_hidden_dim=4,
                moe_intermediate_dim=6,
                moe_vocab_size=11,
                moe_expert_count=2,
                moe_top_k=1,
                require_accelerator=False,
            )
            result = validate_local_accelerator_smoke(d)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(4, bundle["summary"]["check_count"])
        self.assertTrue(bundle["summary"]["local_smoke_passed"])
        self.assertFalse(bundle["summary"]["expert_accelerator_measured"])
        self.assertTrue(bundle["summary"]["pipeline_correctness_included"])
        self.assertFalse(bundle["summary"]["pipeline_correctness_accelerator_measured"])
        self.assertTrue(bundle["summary"]["moe_parity_included"])
        self.assertFalse(bundle["summary"]["moe_parity_accelerator_measured"])
        self.assertFalse(bundle["summary"]["t2_smoke_passed"])
        self.assertFalse(bundle["summary"]["g2_g3_gate_evidence"])

    def test_local_accelerator_smoke_rejects_reference_as_required_accelerator(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = run_local_accelerator_smoke(
                out_dir=d,
                expert_backend="cpu-stdlib",
                expert_iterations=1,
                expert_warmup=0,
                expert_batch_tokens=1,
                expert_hidden_dim=4,
                expert_intermediate_dim=6,
                expert_count=2,
                expert_top_k=1,
                include_activation_transfer=False,
                include_pipeline_correctness=False,
                include_moe_parity=False,
                require_accelerator=True,
            )
            result = validate_local_accelerator_smoke_fixture(bundle)
        self.assertFalse(bundle["ok"])
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("bundle-policy", text)
        self.assertIn("t2_smoke_passed", text)


    def test_local_http_serving_smoke_validates_endpoint_and_sse(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            artifact = Path(d) / "local-http-serving-smoke.json"
            bundle = run_local_http_serving_smoke(out=artifact)
            result = validate_local_http_serving_smoke(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(bundle["summary"]["live_http_endpoint"])
        self.assertTrue(bundle["summary"]["localhost_only"])
        self.assertEqual(7, bundle["summary"]["check_count"])
        self.assertEqual(7, bundle["summary"]["passed_count"])
        self.assertTrue(bundle["summary"]["fornax_backend_integrated"])
        self.assertEqual(2, bundle["summary"]["backend_request_count"])
        self.assertTrue(bundle["summary"]["engine_trait_compatible"])
        self.assertTrue(bundle["summary"]["engine_result_emitted"])
        self.assertFalse(bundle["summary"]["backend_target_model_loaded"])
        self.assertEqual("FornaxBackend", bundle["backend"]["backend"])
        self.assertEqual(200, bundle["summary"]["non_stream_status"])
        self.assertEqual(200, bundle["summary"]["stream_status"])
        self.assertEqual(401, bundle["summary"]["auth_reject_status"])
        self.assertTrue(bundle["summary"]["endpoint_auth_rejected"])
        self.assertTrue(bundle["summary"]["local_auth_enabled"])
        self.assertTrue(bundle["summary"]["auth_token_redacted"])
        self.assertNotIn("auth_token", bundle["config"])
        self.assertEqual("local-bearer-token", bundle["auth"]["mode"])
        self.assertTrue(bundle["auth"]["authorization_header_checked"])
        self.assertEqual("endpoint_auth_required", bundle["responses"]["auth_reject"]["body"]["error"]["code"])
        self.assertTrue(bundle["summary"]["sse_done_seen"])
        self.assertEqual(5, bundle["summary"]["sse_chunk_count"])
        self.assertTrue(bundle["summary"]["plan_integrity_rejected"])
        self.assertTrue(bundle["summary"]["bad_path_rejected"])
        self.assertFalse(bundle["summary"]["tls_enabled"])
        self.assertFalse(bundle["summary"]["production_auth_enabled"])
        self.assertFalse(bundle["summary"]["target_model_parity"])
        self.assertFalse(bundle["summary"]["g2_g3_gate_evidence"])

    def test_local_http_serving_smoke_rejects_gate_overclaim(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            artifact = Path(d) / "local-http-serving-smoke.json"
            bundle = run_local_http_serving_smoke(out=artifact)
        bundle["summary"]["target_model_parity"] = True
        bundle["summary"]["g2_g3_gate_evidence"] = True
        result = validate_local_http_serving_smoke_fixture(bundle)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("target_model_parity", text)
        self.assertIn("g2_g3_gate_evidence", text)


    def test_local_serving_smoke_allows_reference_for_ci(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = run_local_serving_smoke(
                out_dir=d,
                pipeline_backend="cpu-stdlib",
                pipeline_iterations=1,
                pipeline_warmup=0,
                pipeline_hidden_dim=4,
                pipeline_new_tokens=2,
                include_moe_parity=True,
                moe_backend="cpu-stdlib",
                moe_iterations=1,
                moe_warmup=0,
                moe_token_count=2,
                moe_hidden_dim=4,
                moe_intermediate_dim=6,
                moe_vocab_size=11,
                moe_expert_count=2,
                moe_top_k=1,
                require_accelerator=False,
            )
            result = validate_local_serving_smoke(d)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(4, bundle["summary"]["check_count"])
        self.assertTrue(bundle["summary"]["serving_adapter_valid"])
        self.assertTrue(bundle["summary"]["serving_correctness_passed"])
        self.assertTrue(bundle["summary"]["pipeline_correctness_included"])
        self.assertFalse(bundle["summary"]["pipeline_correctness_accelerator_measured"])
        self.assertTrue(bundle["summary"]["moe_parity_included"])
        self.assertFalse(bundle["summary"]["moe_parity_accelerator_measured"])
        self.assertTrue(bundle["summary"]["local_runtime_smoke_passed"])
        self.assertFalse(bundle["summary"]["t2_smoke_passed"])
        self.assertFalse(bundle["summary"]["live_http_endpoint"])
        self.assertFalse(bundle["summary"]["target_model_parity"])
        self.assertFalse(bundle["summary"]["g2_g3_gate_evidence"])

    def test_local_serving_smoke_rejects_reference_as_required_accelerator(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = run_local_serving_smoke(
                out_dir=d,
                pipeline_backend="cpu-stdlib",
                pipeline_iterations=1,
                pipeline_warmup=0,
                pipeline_hidden_dim=4,
                pipeline_new_tokens=2,
                include_moe_parity=False,
                require_accelerator=True,
            )
            result = validate_local_serving_smoke_fixture(bundle)
        self.assertFalse(bundle["ok"])
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("bundle-policy", text)
        self.assertIn("t2_smoke_passed", text)



    def test_cpu_activation_transfer_probe_validates_reference_not_accelerator(self) -> None:
        artifact = run_cpu_activation_transfer_probe(iterations=1, payload_bytes=32)
        result = validate_activation_transfer_probe_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(artifact["measured"])
        self.assertFalse(artifact["accelerator_measured"])
        self.assertIn("not accelerator evidence", "; ".join(result["warnings"]))

    def test_activation_transfer_probe_rejects_false_t3_claim_without_cuda_pair(self) -> None:
        artifact = run_cpu_activation_transfer_probe(iterations=1, payload_bytes=32)
        artifact["tier"] = "T3-same-host-two-gpu-simulation"
        artifact["accelerator_measured"] = True
        result = validate_activation_transfer_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("hardware.device_type must be cuda-pair", text)
        self.assertIn("config.source_device must be cuda:<index>", text)
        self.assertIn("cpu-stdlib backend cannot be accelerator_measured", text)

    def test_activation_transfer_probe_rejects_same_cuda_pair_claim(self) -> None:
        artifact = run_cpu_activation_transfer_probe(iterations=1, payload_bytes=32)
        artifact["tier"] = "T3-same-host-two-gpu-simulation"
        artifact["backend"] = "torch"
        artifact["accelerator_measured"] = True
        artifact["source"] = "fornax.accelerator_probe.torch_activation_transfer"
        artifact["config"].update(
            {
                "source_device": "cuda:0",
                "destination_device": "cuda:0",
                "dtype": "float16",
            }
        )
        artifact["environment"]["torch_version"] = "test-torch"
        artifact["hardware"] = {
            "device_type": "cuda-pair",
            "source_device": "cuda:0",
            "destination_device": "cuda:0",
            "source_name": "test-gpu",
            "destination_name": "test-gpu",
            "source_total_memory_bytes": 1024,
            "destination_total_memory_bytes": 1024,
            "same_physical_host": True,
            "logical_hosts": ["logical-host-0", "logical-host-1"],
        }
        result = validate_activation_transfer_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn(
            "config.source_device and config.destination_device must differ",
            "; ".join(result["errors"]),
        )

    def test_activation_transfer_probe_rejects_failed_correctness(self) -> None:
        artifact = run_cpu_activation_transfer_probe(iterations=1, payload_bytes=32)
        artifact["result"]["correctness_passed"] = False
        artifact["result"]["max_abs_error"] = artifact["config"]["tolerance"] + 1.0
        result = validate_activation_transfer_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("correctness_passed", "; ".join(result["errors"]))


    def test_cpu_pipeline_correctness_probe_validates_reference_not_accelerator(self) -> None:
        artifact = run_cpu_pipeline_correctness_probe(iterations=1, new_tokens=2)
        result = validate_pipeline_correctness_probe_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(artifact["result"]["sequences_match"])
        self.assertFalse(artifact["accelerator_measured"])
        self.assertIn("not accelerator evidence", "; ".join(result["warnings"]))

    def test_pipeline_correctness_probe_rejects_false_t3_claim_without_cuda(self) -> None:
        artifact = run_cpu_pipeline_correctness_probe(iterations=1, new_tokens=2)
        artifact["tier"] = "T3-same-host-two-gpu-simulation"
        artifact["accelerator_measured"] = True
        result = validate_pipeline_correctness_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("hardware.device_type must be cuda-pipeline", text)
        self.assertIn("config.source_device must be cuda:<index>", text)
        self.assertIn("cpu-stdlib backend cannot be accelerator_measured", text)

    def test_pipeline_correctness_probe_rejects_same_cuda_pair_claim(self) -> None:
        artifact = run_cpu_pipeline_correctness_probe(iterations=1, new_tokens=2)
        artifact["tier"] = "T3-same-host-two-gpu-simulation"
        artifact["backend"] = "torch"
        artifact["accelerator_measured"] = True
        artifact["source"] = "fornax.pipeline_probe.torch_pipeline_correctness"
        artifact["config"].update(
            {
                "source_device": "cuda:0",
                "destination_device": "cuda:0",
                "dtype": "float32",
            }
        )
        artifact["environment"]["torch_version"] = "test-torch"
        artifact["hardware"] = {
            "device_type": "cuda-pipeline",
            "source_device": "cuda:0",
            "destination_device": "cuda:0",
            "source_name": "test-gpu",
            "destination_name": "test-gpu",
            "source_total_memory_bytes": 1024,
            "destination_total_memory_bytes": 1024,
            "same_physical_host": True,
            "logical_hosts": ["logical-host-0", "logical-host-1"],
        }
        result = validate_pipeline_correctness_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn(
            "config.source_device and config.destination_device must differ",
            "; ".join(result["errors"]),
        )

    def test_pipeline_correctness_probe_rejects_generation_mismatch(self) -> None:
        artifact = run_cpu_pipeline_correctness_probe(iterations=1, new_tokens=2)
        artifact["result"]["generated_sequences"][0][-1] += 1
        artifact["result"]["sequences_match"] = False
        artifact["result"]["correctness_passed"] = False
        result = validate_pipeline_correctness_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("sequences_match", text)
        self.assertIn("generated_sequences must match", text)


    def test_throughput_scaling_simulation_validates_metric_contract(self) -> None:
        artifact = simulate_throughput_scaling()
        result = validate_throughput_scaling_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(artifact["summary"]["target_met"])
        self.assertLessEqual(
            artifact["summary"]["observed_saturation_concurrency"],
            artifact["contracted_min_concurrency"],
        )
        self.assertGreaterEqual(
            artifact["summary"]["throughput_efficiency_at_contract"],
            artifact["throughput_efficiency_floor"],
        )
        self.assertIn("simulation evidence", "; ".join(result["warnings"]))

    def test_throughput_scaling_rejects_planner_error_out_of_bound(self) -> None:
        artifact = simulate_throughput_scaling()
        artifact["rows"][0]["predicted_tokens_s"] *= 2.0
        artifact["rows"][0]["planner_error_fraction"] = 1.0
        artifact["summary"]["max_abs_planner_error_fraction"] = 1.0
        artifact["summary"]["planner_accuracy_passed"] = False
        artifact["summary"]["target_met"] = False
        result = validate_throughput_scaling_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("planner_error_fraction does not match", text)
        self.assertIn("summary.target_met must be true", text)

    def test_throughput_scaling_rejects_non_monotonic_sweep(self) -> None:
        artifact = simulate_throughput_scaling()
        artifact["rows"][2]["measured_tokens_s"] = artifact["rows"][1]["measured_tokens_s"] / 2.0
        artifact["summary"]["monotonic_scaling"] = False
        artifact["summary"]["target_met"] = False
        result = validate_throughput_scaling_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("summary.target_met must be true", "; ".join(result["errors"]))

    def test_throughput_scaling_rejects_late_saturation(self) -> None:
        artifact = simulate_throughput_scaling(contracted_min_concurrency=8, saturation_concurrency=16)
        result = validate_throughput_scaling_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("summary.target_met must be true", text)
        self.assertFalse(artifact["summary"]["saturation_within_contract"])


    def test_cpu_remote_expert_batch_probe_validates_reference_not_accelerator(self) -> None:
        artifact = run_cpu_remote_expert_batch_probe(iterations=1, token_count=2)
        result = validate_remote_expert_batch_probe_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(artifact["result"]["correctness_passed"])
        self.assertFalse(artifact["accelerator_measured"])
        self.assertIn("not accelerator evidence", "; ".join(result["warnings"]))

    def test_remote_expert_batch_rejects_false_t3_claim_without_cuda(self) -> None:
        artifact = run_cpu_remote_expert_batch_probe(iterations=1, token_count=2)
        artifact["tier"] = "T3-same-host-remote-expert-simulation"
        artifact["accelerator_measured"] = True
        result = validate_remote_expert_batch_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("hardware.device_type must be cuda-remote-expert", text)
        self.assertIn("config.source_device must be cuda:<index>", text)
        self.assertIn("cpu-stdlib backend cannot be accelerator_measured", text)

    def test_remote_expert_batch_rejects_same_cuda_pair_claim(self) -> None:
        artifact = run_cpu_remote_expert_batch_probe(iterations=1, token_count=2)
        artifact["tier"] = "T3-same-host-remote-expert-simulation"
        artifact["backend"] = "torch"
        artifact["accelerator_measured"] = True
        artifact["source"] = "fornax.remote_expert_probe.torch_remote_expert_batch"
        artifact["config"].update({"source_device": "cuda:0", "expert_device": "cuda:0", "dtype": "float32"})
        artifact["environment"]["torch_version"] = "test-torch"
        artifact["hardware"] = {
            "device_type": "cuda-remote-expert",
            "source_device": "cuda:0",
            "expert_device": "cuda:0",
            "source_name": "test-gpu",
            "expert_name": "test-gpu",
            "source_total_memory_bytes": 1024,
            "expert_total_memory_bytes": 1024,
            "same_physical_host": True,
            "logical_hosts": ["logical-host-0", "logical-host-1"],
        }
        result = validate_remote_expert_batch_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("config.source_device and config.expert_device must differ", "; ".join(result["errors"]))

    def test_remote_expert_batch_rejects_failed_correctness(self) -> None:
        artifact = run_cpu_remote_expert_batch_probe(iterations=1, token_count=2)
        artifact["result"]["correctness_passed"] = False
        artifact["result"]["max_abs_error"] = artifact["config"]["tolerance"] + 1.0
        result = validate_remote_expert_batch_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("correctness_passed", "; ".join(result["errors"]))

    def test_cpu_moe_layer_parity_probe_validates_reference_not_accelerator(self) -> None:
        artifact = run_cpu_moe_layer_parity_probe(iterations=1, token_count=2)
        result = validate_moe_layer_parity_probe_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(artifact["result"]["correctness_passed"])
        self.assertTrue(artifact["result"]["next_tokens_match"])
        self.assertFalse(artifact["accelerator_measured"])
        self.assertIn("not accelerator evidence", "; ".join(result["warnings"]))

    def test_moe_layer_parity_rejects_false_t3_claim_without_cuda(self) -> None:
        artifact = run_cpu_moe_layer_parity_probe(iterations=1, token_count=2)
        artifact["tier"] = "T3-same-host-moe-parity-simulation"
        artifact["accelerator_measured"] = True
        result = validate_moe_layer_parity_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("hardware.device_type must be cuda-moe-layer-parity", text)
        self.assertIn("config.source_device must be cuda:<index>", text)
        self.assertIn("cpu-stdlib backend cannot be accelerator_measured", text)

    def test_moe_layer_parity_rejects_same_cuda_pair_claim(self) -> None:
        artifact = run_cpu_moe_layer_parity_probe(iterations=1, token_count=2)
        artifact["tier"] = "T3-same-host-moe-parity-simulation"
        artifact["backend"] = "torch"
        artifact["accelerator_measured"] = True
        artifact["source"] = "fornax.moe_parity.torch_moe_layer_parity"
        artifact["config"].update({"source_device": "cuda:0", "expert_device": "cuda:0", "dtype": "float32"})
        artifact["environment"]["torch_version"] = "test-torch"
        artifact["hardware"] = {
            "device_type": "cuda-moe-layer-parity",
            "source_device": "cuda:0",
            "expert_device": "cuda:0",
            "source_name": "test-gpu",
            "expert_name": "test-gpu",
            "source_total_memory_bytes": 1024,
            "expert_total_memory_bytes": 1024,
            "same_physical_host": True,
            "logical_hosts": ["logical-host-0", "logical-host-1"],
        }
        result = validate_moe_layer_parity_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("config.source_device and config.expert_device must differ", "; ".join(result["errors"]))

    def test_moe_layer_parity_rejects_failed_correctness(self) -> None:
        artifact = run_cpu_moe_layer_parity_probe(iterations=1, token_count=2)
        artifact["result"]["correctness_passed"] = False
        artifact["result"]["max_logit_abs_error"] = artifact["config"]["tolerance"] + 1.0
        result = validate_moe_layer_parity_probe_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("correctness_passed", "; ".join(result["errors"]))

    def test_tiny_expert_mlp_benchmark_records_measurement(self) -> None:
        result = run_tiny_expert_mlp_benchmark(
            iterations=1, batch_tokens=2, hidden_dim=4, intermediate_dim=6
        )
        self.assertTrue(result["measured"])
        self.assertEqual("fornax.benchmark.tiny_expert_mlp.cpu_stdlib", result["source"])
        self.assertTrue(result["config"]["weights_precomputed_before_timing"])
        self.assertGreater(result["result"]["elapsed_ns"], 0)
        self.assertEqual(2, result["result"]["tokens_processed"])
        self.assertEqual(4, result["result"]["expert_calls"])
        self.assertIsInstance(result["result"]["checksum"], float)

    def test_benchmark_ledger_fixture_passes(self) -> None:
        result = validate_benchmark_ledger("fornax/golden_vectors/benchmark_ledger")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(1, result["summary"]["record_count"])
        self.assertEqual(1, result["summary"]["measured_record_count"])

    def test_benchmark_ledger_record_from_tiny_benchmark_validates_and_appends(self) -> None:
        benchmark = run_tiny_expert_mlp_benchmark(
            iterations=1, batch_tokens=2, hidden_dim=4, intermediate_dim=6
        )
        record = build_benchmark_ledger_record(
            benchmark,
            benchmark_id="unit-tiny-expert-mlp",
            command=["python3", "-m", "fornax", "benchmark"],
            hardware="unit-cpu",
            os_name="unit-os",
            driver_runtime="unit-runtime",
            max_mojo_version="unit-max-mojo",
            model="unit-model",
            context="prompt=8,gen=4",
            concurrency=4,
            quantization="q4",
            thermals="unit-stable",
            timestamp_utc="2026-06-21T00:00:00Z",
        )
        self.assertTrue(validate_benchmark_ledger_record(record)["ok"])
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "ledger.jsonl"
            append_benchmark_ledger_record(ledger, record)
            result = validate_benchmark_ledger(ledger)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(1, result["summary"]["measured_record_count"])

    def test_benchmark_ledger_rejects_unmeasured_benchmark(self) -> None:
        benchmark = run_tiny_expert_mlp_benchmark(iterations=1, batch_tokens=1)
        benchmark = dict(benchmark)
        benchmark["measured"] = False
        record = build_benchmark_ledger_record(
            benchmark,
            benchmark_id="unit-unmeasured",
            command=["python3", "-m", "fornax", "benchmark"],
            hardware="unit-cpu",
            os_name="unit-os",
            driver_runtime="unit-runtime",
            max_mojo_version="unit-max-mojo",
            model="unit-model",
            context="prompt=1,gen=1",
            concurrency=1,
            quantization="q4",
            thermals="unit-stable",
        )
        result = validate_benchmark_ledger_record(record)
        self.assertFalse(result["ok"])
        self.assertIn("benchmark.measured must be true", "; ".join(result["errors"]))

    def test_benchmark_from_plan_rejects_infeasible_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "infeasible plan"):
            benchmark_from_plan({"feasible": False, "infeasible_reason": "no fit"})

    def test_engine_seam_fixture_passes(self) -> None:
        result = validate_engine_seam_contract("fornax/golden_vectors/engine_seam")
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["summary"]["template_hash_recorded"])
        self.assertTrue(result["summary"]["tokenizer_hash_recorded"])
        self.assertGreaterEqual(result["summary"]["stream_event_count"], 3)

    def test_engine_seam_rejects_hash_mismatch(self) -> None:
        fixture = read_json("fornax/golden_vectors/engine_seam/fixture.json")
        fixture["result"]["tokenizer_hash"] = (
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        )
        result = validate_engine_seam_fixture(fixture)
        self.assertFalse(result["ok"])
        self.assertIn("result.tokenizer_hash", "; ".join(result["errors"]))

    def test_engine_seam_rejects_speculative_decoding_without_opt_in(self) -> None:
        fixture = read_json("fornax/golden_vectors/engine_seam/fixture.json")
        fixture["speculative_decoding"]["enabled"] = True
        result = validate_engine_seam_fixture(fixture)
        self.assertFalse(result["ok"])
        self.assertIn("speculative decoding is out of v0", "; ".join(result["errors"]))

    def test_stage_host_fixture_passes(self) -> None:
        result = validate_stage_host("fornax/golden_vectors/stage_host")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(4, result["summary"]["boundary_op_count"])
        self.assertEqual(0.0, result["summary"]["max_abs_error"])
        self.assertTrue(result["summary"]["graphlet_claim_is_simulated"])

    def test_simulated_stage_host_validates_boundary_and_reference(self) -> None:
        contract = simulate_stage_host(plan_id="unit-stage-host")
        result = validate_stage_host_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(
            {"activation_in", "activation_out", "kv_read", "kv_write"},
            {op["name"] for op in contract["boundary_ops"]},
        )
        self.assertEqual("planned", contract["stage_host"]["max_graphlet_status"])
        self.assertFalse(contract["stage_host"]["measured"])

    def test_stage_host_rejects_output_mismatch(self) -> None:
        contract = simulate_stage_host()
        contract["tensors"]["stage_output"][0][0] += 1.0
        result = validate_stage_host_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("max_abs_error", "; ".join(result["errors"]))

    def test_stage_host_rejects_missing_boundary_op(self) -> None:
        contract = simulate_stage_host()
        contract["boundary_ops"] = [
            op for op in contract["boundary_ops"] if op["name"] != "kv_write"
        ]
        result = validate_stage_host_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("kv_write", "; ".join(result["errors"]))

    def test_stage_host_rejects_measured_max_claim(self) -> None:
        contract = simulate_stage_host()
        contract["stage_host"]["measured"] = True
        contract["graphlet_contract"]["status"] = "executed"
        contract["graphlet_contract"]["measured"] = True
        for event in contract["events"]:
            if event["kind"] == "graphlet_executed":
                event["measured"] = True
        result = validate_stage_host_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("measured must be false", "; ".join(result["errors"]))

    def test_stage_host_rejects_kv_owner_mismatch(self) -> None:
        contract = simulate_stage_host()
        contract["kv_cache"]["owner_after"] = "stage-99"
        result = validate_stage_host_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("kv_cache.owner_after", "; ".join(result["errors"]))

    def test_backend_coverage_fixture_passes(self) -> None:
        result = validate_backend_coverage_contract("fornax/golden_vectors/backend_coverage")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(9, result["summary"]["operation_count"])
        self.assertIn("expert_gemm_mlp", result["summary"]["required_operations_seen"])
        self.assertIn("macos_apple_silicon", result["summary"]["required_backends"])

    def test_backend_coverage_rejects_missing_required_operation(self) -> None:
        fixture = read_json("fornax/golden_vectors/backend_coverage/fixture.json")
        fixture["operations"] = [
            item for item in fixture["operations"] if item.get("operation") != "attention"
        ]
        result = validate_backend_coverage_fixture(fixture)
        self.assertFalse(result["ok"])
        self.assertIn("operations missing required entries", "; ".join(result["errors"]))
        self.assertIn("attention", "; ".join(result["errors"]))

    def test_backend_coverage_rejects_missing_ledger_field(self) -> None:
        fixture = read_json("fornax/golden_vectors/backend_coverage/fixture.json")
        fixture["benchmark_ledger_fields"].remove("thermals")
        result = validate_backend_coverage_fixture(fixture)
        self.assertFalse(result["ok"])
        self.assertIn("benchmark_ledger_fields missing", "; ".join(result["errors"]))
        self.assertIn("thermals", "; ".join(result["errors"]))

    def test_backend_coverage_report_renders_matrix(self) -> None:
        result = render_backend_coverage_report("fornax/golden_vectors/backend_coverage")
        self.assertTrue(result["ok"], result["validation"]["errors"])
        markdown = result["markdown"]
        self.assertIn("# Backend Operation Coverage Matrix", markdown)
        self.assertIn("Status: DRAFT", markdown)
        self.assertIn("expert_gemm_mlp", markdown)
        self.assertIn("macos_apple_silicon", markdown)
        self.assertIn("Benchmark Ledger Fields", markdown)

    def test_observability_fixture_passes(self) -> None:
        result = validate_observability_contract("fornax/golden_vectors/observability")
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("router_decision", result["summary"]["required_events_seen"])
        self.assertIn("bad_plan_reproduction", result["summary"]["required_events_seen"])
        self.assertEqual(2, result["summary"]["stage_count"])
        self.assertEqual(
            ["demoted", "excluded", "selected"],
            result["summary"]["placement_decisions"],
        )

    def test_observability_rejects_missing_required_events(self) -> None:
        fixture = read_json("fornax/golden_vectors/observability/fixture.json")
        fixture["events"] = [
            event for event in fixture["events"] if event.get("kind") != "router_decision"
        ]
        result = validate_observability_fixture(fixture)
        self.assertFalse(result["ok"])
        self.assertIn("missing required observability events", "; ".join(result["errors"]))
        self.assertIn("router_decision", "; ".join(result["errors"]))

    def test_observability_rejects_plan_id_mismatch(self) -> None:
        fixture = read_json("fornax/golden_vectors/observability/fixture.json")
        fixture["events"][0]["plan_id"] = "wrong-plan"
        result = validate_observability_fixture(fixture)
        self.assertFalse(result["ok"])
        self.assertIn("plan_id must match", "; ".join(result["errors"]))

    def test_metrics_ledger_fixture_passes(self) -> None:
        result = validate_metrics_ledger("fornax/golden_vectors/metrics_ledger")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(5, result["summary"]["sample_count"])
        self.assertEqual(3, result["summary"]["alert_count"])
        self.assertEqual(4, result["summary"]["max_queue_depth_observed"])
        self.assertEqual(2, result["summary"]["kv_pages_evicted_total"])

    def test_simulated_metrics_ledger_validates_derived_metrics(self) -> None:
        contract = simulate_metrics_ledger(plan_id="unit-metrics-plan")
        result = validate_metrics_ledger_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(4, contract["metrics"]["counters"]["requests_admitted_total"])
        self.assertEqual(10, contract["metrics"]["histograms"]["stage_latency_ms"]["count"])

    def test_metrics_ledger_rejects_counter_mismatch(self) -> None:
        contract = simulate_metrics_ledger()
        contract["metrics"]["counters"]["requests_admitted_total"] += 1
        result = validate_metrics_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("metrics.counters.requests_admitted_total", "; ".join(result["errors"]))

    def test_metrics_ledger_rejects_queue_overflow_sample(self) -> None:
        contract = simulate_metrics_ledger(max_queue_depth=4)
        contract["samples"][0]["queue_depth"] = 5
        result = validate_metrics_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("queue_depth exceeds", "; ".join(result["errors"]))

    def test_metrics_ledger_rejects_missing_memory_alert(self) -> None:
        contract = simulate_metrics_ledger()
        contract["metrics"]["alerts"] = [
            alert
            for alert in contract["metrics"]["alerts"]
            if alert["kind"] != "memory_pressure_warning"
        ]
        result = validate_metrics_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("memory_pressure_warning", "; ".join(result["errors"]))

    def test_metrics_ledger_rejects_memory_pressure_mismatch(self) -> None:
        contract = simulate_metrics_ledger()
        contract["samples"][1]["memory_pressure_fraction"] = 0.1
        result = validate_metrics_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("memory_pressure_fraction", "; ".join(result["errors"]))

    def test_metrics_ledger_rejects_histogram_mismatch(self) -> None:
        contract = simulate_metrics_ledger()
        contract["metrics"]["histograms"]["stage_latency_ms"]["count"] += 1
        result = validate_metrics_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("histograms.stage_latency_ms.count", "; ".join(result["errors"]))

    def test_trace_ledger_fixture_passes(self) -> None:
        result = validate_trace_ledger("fornax/golden_vectors/trace_ledger")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(8, result["summary"]["component_count"])
        self.assertEqual(8, result["summary"]["span_count"])
        self.assertEqual(16, result["summary"]["event_count"])
        self.assertEqual(7, result["summary"]["required_edge_count"])
        self.assertTrue(result["summary"]["correlation_complete"])

    def test_simulated_trace_ledger_correlates_request_plan_and_spans(self) -> None:
        contract = simulate_trace_ledger(
            plan_id="unit-trace-plan",
            request_id="unit-trace-request",
            trace_id="unit-trace-id",
        )
        result = validate_trace_ledger_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        kinds = {event["kind"] for event in contract["events"]}
        self.assertIn("router_decision", kinds)
        self.assertIn("remote_expert_dispatched", kinds)
        self.assertIn("kv_write", kinds)
        self.assertIn("cleanup", kinds)
        self.assertEqual(2, result["summary"]["stage_count"])

    def test_trace_ledger_rejects_trace_id_mismatch(self) -> None:
        contract = simulate_trace_ledger()
        contract["events"][0]["trace_id"] = "wrong-trace"
        result = validate_trace_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("trace_id must match", "; ".join(result["errors"]))

    def test_trace_ledger_rejects_missing_parent_span(self) -> None:
        contract = simulate_trace_ledger()
        contract["spans"][1]["parent_span_id"] = "missing-span"
        result = validate_trace_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("parent_span_id references unknown span", "; ".join(result["errors"]))

    def test_trace_ledger_rejects_event_logical_host_mismatch(self) -> None:
        contract = simulate_trace_ledger()
        contract["events"][0]["logical_host_id"] = "logical-host-1"
        result = validate_trace_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("logical_host_id must match span logical_host_id", "; ".join(result["errors"]))

    def test_trace_ledger_rejects_summary_count_mismatch(self) -> None:
        contract = simulate_trace_ledger()
        contract["summary"]["remote_expert_event_count"] = 0
        result = validate_trace_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("summary.remote_expert_event_count", "; ".join(result["errors"]))

    def test_trace_ledger_rejects_causal_order_regression(self) -> None:
        contract = simulate_trace_ledger()
        cleanup = contract["events"].pop()
        contract["events"].insert(0, cleanup)
        result = validate_trace_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        errors = "; ".join(result["errors"])
        self.assertTrue(
            "occurs before" in errors or "timestamp_s must be non-decreasing" in errors
        )

    def test_trace_ledger_rejects_missing_required_edge(self) -> None:
        contract = simulate_trace_ledger()
        for span in contract["spans"]:
            if span["span_id"] == "span-kv":
                span["parent_span_id"] = "span-serving"
                break
        result = validate_trace_ledger_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("spans missing required component edges", "; ".join(result["errors"]))

    def test_worker_contract_fixture_passes(self) -> None:
        result = validate_worker_contract("fornax/golden_vectors/worker_contract")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(2, result["summary"]["worker_count"])
        self.assertEqual(1, result["summary"]["stage_worker_count"])
        self.assertEqual(1, result["summary"]["expert_worker_count"])
        self.assertEqual(1, result["summary"]["plan_integrity_reject_count"])
        self.assertEqual(2, result["summary"]["cleanup_count"])

    def test_simulated_worker_contract_validates_plan_integrity_and_cleanup(self) -> None:
        contract = simulated_worker_contract(
            plan_id="unit-worker-plan",
            request_id="unit-request",
            plan_hash="sha256:unit-worker-plan",
            max_queue_depth=2,
        )
        result = validate_worker_contract_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        kinds = {event["kind"] for event in contract["events"]}
        self.assertIn("stale_plan_reject", kinds)
        self.assertIn("cleanup", kinds)
        self.assertIn("expert_batch_received", kinds)

    def test_worker_contract_rejects_payload_plan_hash_mismatch(self) -> None:
        contract = simulated_worker_contract()
        for event in contract["events"]:
            if event["kind"] == "activation_received":
                event["plan_hash"] = "sha256:wrong-plan"
                break
        result = validate_worker_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("plan_hash must match", "; ".join(result["errors"]))

    def test_worker_contract_rejects_missing_cleanup(self) -> None:
        contract = simulated_worker_contract()
        contract["events"] = [
            event for event in contract["events"] if event["kind"] != "cleanup"
        ]
        contract["summary"]["event_count"] = len(contract["events"])
        contract["summary"]["cleanup_count"] = 0
        result = validate_worker_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("workers missing cleanup event", "; ".join(result["errors"]))

    def test_worker_contract_rejects_role_event_mismatch(self) -> None:
        contract = simulated_worker_contract()
        for event in contract["events"]:
            if event["kind"] == "expert_execute_start":
                event["worker_id"] = "stage-0"
                break
        result = validate_worker_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("not valid for role", "; ".join(result["errors"]))

    def test_worker_contract_rejects_unsupported_payload_kind(self) -> None:
        contract = simulated_worker_contract()
        for worker in contract["workers"]:
            if worker["worker_id"] == "stage-0":
                worker["supported_payloads"] = ["kv_page"]
                break
        result = validate_worker_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("payload_kind is not supported", "; ".join(result["errors"]))



    def test_program_simulate_t1_builds_full_logical_cluster_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source_path = root / "source-inventory.json"
            write_json(source_path, two_gpu_source_inventory())
            result = run_t1_simulated_validation(
                out_dir=root / "bundle",
                source_inventory_path=source_path,
                gpu_count=2,
                profile="two-gpu-heterogeneous",
                link_bandwidth_bytes_s=12_500_000_000.0,
                link_latency_s=0.0004,
                slow_node_factor=0.65,
            )
            bundle = Path(result["bundle"])
            transport = read_json(bundle / "transport-contract.json")
            trust_boundary = read_json(bundle / "trust-boundary.json")
            metrics_ledger = read_json(bundle / "metrics-ledger.json")
            trace_ledger = read_json(bundle / "trace-ledger.json")
            stage_host = read_json(bundle / "stage-host.json")
            state_ownership = read_json(bundle / "state-ownership.json")
            validation = read_json(bundle / "t1-simulated-validation.json")
        self.assertTrue(result["ok"], result["summary"])
        self.assertEqual(31, result["summary"]["check_count"])
        self.assertEqual(31, result["summary"]["passed_count"])
        self.assertEqual(2, result["summary"]["logical_host_count"])
        self.assertEqual("logical_multi_host", result["simulation"]["mode"])
        self.assertEqual("two_gpu_logical_hosts", transport["simulation"]["method"])
        self.assertEqual({"0", "1"}, {
            endpoint["worker_environment"]["CUDA_VISIBLE_DEVICES"]
            for endpoint in transport["endpoints"]
        })
        self.assertFalse(trust_boundary["trust_policy"]["allow_anonymous"])
        self.assertTrue(trust_boundary["summary"]["stale_plan_rejected"])
        self.assertTrue(metrics_ledger["summary"]["correctness_passed"])
        self.assertGreaterEqual(metrics_ledger["summary"]["alert_count"], 3)
        self.assertTrue(trace_ledger["summary"]["correlation_complete"])
        self.assertEqual(2, trace_ledger["summary"]["stage_count"])
        self.assertEqual("planned", stage_host["stage_host"]["max_graphlet_status"])
        self.assertFalse(stage_host["stage_host"]["measured"])
        self.assertTrue(state_ownership["summary"]["correctness_passed"])
        self.assertEqual(11, state_ownership["summary"]["terminal_released_count"])
        self.assertIn("trust-boundary", {check["name"] for check in validation["checks"]})
        self.assertIn("metrics-ledger", {check["name"] for check in validation["checks"]})
        self.assertIn("trace-ledger", {check["name"] for check in validation["checks"]})
        self.assertIn("stage-host", {check["name"] for check in validation["checks"]})
        self.assertIn("engine-simulation", {check["name"] for check in validation["checks"]})
        self.assertIn("serving-adapter", {check["name"] for check in validation["checks"]})
        self.assertIn("state-ownership", {check["name"] for check in validation["checks"]})
        self.assertIn("continuous-batching", {check["name"] for check in validation["checks"]})
        self.assertIn("pipeline-correctness", {check["name"] for check in validation["checks"]})
        self.assertIn("throughput-scaling", {check["name"] for check in validation["checks"]})
        self.assertIn("stage-replication", {check["name"] for check in validation["checks"]})
        self.assertIn("resilience-replay", {check["name"] for check in validation["checks"]})
        self.assertIn("ops-lifecycle", {check["name"] for check in validation["checks"]})
        self.assertIn("onboarding-methodology", {check["name"] for check in validation["checks"]})
        self.assertIn("program-governance", {check["name"] for check in validation["checks"]})
        self.assertIn("moe-runtime", {check["name"] for check in validation["checks"]})
        self.assertIn("moe-migration", {check["name"] for check in validation["checks"]})
        self.assertIn("remote-expert-batch", {check["name"] for check in validation["checks"]})
        self.assertIn("moe-layer-parity", {check["name"] for check in validation["checks"]})
        self.assertIn("model-support", {check["name"] for check in validation["checks"]})
        self.assertIn("transport-contract", {check["name"] for check in validation["checks"]})
        self.assertIn("scheduler-contract", {check["name"] for check in validation["checks"]})
        self.assertIn("worker-contract", {check["name"] for check in validation["checks"]})

    def test_program_simulate_t1_rejects_single_gpu_source_inventory(self) -> None:
        source = two_gpu_source_inventory()
        source["nodes"] = source["nodes"][:1]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source_path = root / "source-inventory.json"
            write_json(source_path, source)
            with self.assertRaisesRegex(ValueError, "NVIDIA GPU"):
                run_t1_simulated_validation(
                    out_dir=root / "bundle",
                    source_inventory_path=source_path,
                    gpu_count=2,
                )


    def test_serving_adapter_fixture_passes(self) -> None:
        result = validate_serving_adapter("fornax/golden_vectors/serving_adapter")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(2, result["summary"]["surface_count"])
        self.assertTrue(result["summary"]["template_hash_recorded"])
        self.assertTrue(result["summary"]["tokenizer_hash_recorded"])

    def test_simulated_serving_adapter_roundtrips_openai_and_engine_surfaces(self) -> None:
        contract = simulate_serving_adapter(plan_id="unit-serving-plan")
        result = validate_serving_adapter_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual("openai_chat_completions", contract["openai_request"]["surface"])
        self.assertEqual("FornaxBackend", contract["adapters"][1]["backend"])
        self.assertEqual(
            contract["engine_result"]["usage"],
            contract["openai_response"]["usage"],
        )
        self.assertEqual(
            len(contract["engine_stream_events"]),
            len(contract["openai_stream_chunks"]),
        )

    def test_serving_adapter_rejects_template_hash_mismatch(self) -> None:
        contract = simulate_serving_adapter()
        contract["engine_result"]["template_hash"] = "sha256:" + "c" * 64
        result = validate_serving_adapter_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("template_hash", "; ".join(result["errors"]))

    def test_serving_adapter_rejects_stream_chunk_mismatch(self) -> None:
        contract = simulate_serving_adapter()
        contract["openai_stream_chunks"][1]["engine_event_kind"] = "finish"
        result = validate_serving_adapter_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("engine_event_kind", "; ".join(result["errors"]))

    def test_serving_adapter_rejects_missing_openai_surface(self) -> None:
        contract = simulate_serving_adapter()
        contract["adapters"] = [adapter for adapter in contract["adapters"] if adapter["surface"] != "openai_chat_completions"]
        contract["summary"]["surface_count"] = len(contract["adapters"])
        result = validate_serving_adapter_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("missing required surfaces", "; ".join(result["errors"]))

    def test_state_ownership_fixture_passes(self) -> None:
        result = validate_state_ownership("fornax/golden_vectors/state_ownership")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(11, result["summary"]["resource_count"])
        self.assertEqual(11, result["summary"]["terminal_released_count"])
        self.assertFalse(result["summary"]["dual_owner_detected"])

    def test_simulated_state_ownership_validates_transitions_and_cleanup(self) -> None:
        contract = simulate_state_ownership(
            plan_id="unit-state-plan",
            request_id="unit-state-request",
            cancel_request_id="unit-state-cancel",
        )
        result = validate_state_ownership_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual("released", result["summary"]["normal_request_terminal_owner"])
        self.assertEqual("released", result["summary"]["cancel_request_terminal_owner"])
        self.assertEqual(11, contract["summary"]["terminal_released_count"])
        resource_kinds = {resource["kind"] for resource in contract["resources"]}
        self.assertIn("request_envelope", resource_kinds)
        self.assertIn("activation_buffer", resource_kinds)
        self.assertIn("kv_cache", resource_kinds)
        self.assertIn("response_stream", resource_kinds)

    def test_state_ownership_rejects_wrong_from_owner(self) -> None:
        contract = simulate_state_ownership()
        contract["ownership_transitions"][1]["from_owner"] = "scheduler"
        result = validate_state_ownership_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("from_owner must match", "; ".join(result["errors"]))

    def test_state_ownership_rejects_missing_cleanup_release(self) -> None:
        contract = simulate_state_ownership()
        contract["ownership_transitions"] = [
            transition
            for transition in contract["ownership_transitions"]
            if not (
                transition["resource_id"] == "kv-cache:primary"
                and transition["event"] == "cleanup"
            )
        ]
        result = validate_state_ownership_fixture(contract)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("required resources not released", text)
        self.assertIn("kv-cache:primary", text)

    def test_state_ownership_rejects_dual_active_owner_snapshot(self) -> None:
        contract = simulate_state_ownership()
        contract["ownership_snapshots"][0]["claims"].append(
            {
                "resource_id": "request:primary",
                "owner": "fornax_engine",
                "state": "active",
            }
        )
        result = validate_state_ownership_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("multiple active owners", "; ".join(result["errors"]))

    def test_state_ownership_rejects_summary_stale_count_mismatch(self) -> None:
        contract = simulate_state_ownership()
        contract["summary"]["terminal_released_count"] = 10
        result = validate_state_ownership_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("summary.terminal_released_count", "; ".join(result["errors"]))

    def test_engine_simulation_fixture_passes(self) -> None:
        result = validate_engine_simulation("fornax/golden_vectors/engine_simulation")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(3, result["summary"]["embedded_contract_count"])
        self.assertEqual(2, result["summary"]["request_count"])
        self.assertEqual(2, result["summary"]["token_count"])
        self.assertIn("activation_handoff", result["summary"]["required_events_seen"])

    def test_simulated_engine_contract_composes_scheduler_worker_transport(self) -> None:
        contract = simulated_engine_contract(
            plan_id="unit-engine-plan",
            request_id="unit-engine-request",
            plan_hash="sha256:unit-engine-plan",
            max_queue_depth=2,
            max_inflight=2,
            microbatch_size=2,
            timeout_ms=50.0,
        )
        result = validate_engine_simulation_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual("unit-engine-plan", contract["scheduler_contract"]["plan_id"])
        self.assertEqual("sha256:unit-engine-plan", contract["worker_contract"]["plan_hash"])
        self.assertEqual("sha256:unit-engine-plan", contract["transport_contract"]["plan_hash"])
        self.assertEqual(3, contract["summary"]["embedded_contract_count"])

    def test_engine_simulation_rejects_embedded_transport_plan_hash_mismatch(self) -> None:
        contract = simulated_engine_contract()
        contract["transport_contract"]["plan_hash"] = "sha256:wrong-plan"
        result = validate_engine_simulation_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("transport_contract.plan_hash", "; ".join(result["errors"]))

    def test_engine_simulation_rejects_missing_request_finished(self) -> None:
        contract = simulated_engine_contract()
        contract["events"] = [
            event
            for event in contract["events"]
            if not (event["kind"] == "request_finished" and event["request_id"] == contract["request_id"])
        ]
        contract["summary"]["event_count"] = len(contract["events"])
        contract["summary"]["finished_count"] = 1
        result = validate_engine_simulation_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("primary request missing", "; ".join(result["errors"]))

    def test_engine_simulation_rejects_token_summary_mismatch(self) -> None:
        contract = simulated_engine_contract()
        contract["summary"]["token_count"] = 99
        result = validate_engine_simulation_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("summary.token_count", "; ".join(result["errors"]))

    def test_engine_simulation_rejects_missing_cleanup(self) -> None:
        contract = simulated_engine_contract()
        contract["events"] = [event for event in contract["events"] if event["kind"] != "cleanup"]
        contract["summary"]["event_count"] = len(contract["events"])
        contract["summary"]["cleanup_count"] = 0
        result = validate_engine_simulation_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("events missing required engine events", "; ".join(result["errors"]))

    def test_transport_contract_fixture_passes(self) -> None:
        result = validate_transport_contract("fornax/golden_vectors/transport_contract")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(2, result["summary"]["logical_host_count"])
        self.assertEqual(3, result["summary"]["endpoint_count"])
        self.assertEqual(4, result["summary"]["payload_count"])
        self.assertEqual(1, result["summary"]["timeout_count"])
        self.assertEqual(1, result["summary"]["cancel_count"])

    def test_simulated_transport_contract_uses_two_gpu_logical_hosts(self) -> None:
        contract = simulated_transport_contract(
            plan_id="unit-transport-plan",
            request_id="unit-request",
            plan_hash="sha256:unit-transport-plan",
            max_queue_depth=2,
            timeout_ms=50.0,
        )
        result = validate_transport_contract_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        logical_hosts = {endpoint["logical_host_id"] for endpoint in contract["endpoints"]}
        cuda_bindings = {
            endpoint["worker_environment"]["CUDA_VISIBLE_DEVICES"]
            for endpoint in contract["endpoints"]
        }
        self.assertEqual({"sim-host-0", "sim-host-1"}, logical_hosts)
        self.assertEqual({"0", "1"}, cuda_bindings)
        self.assertEqual("two_gpu_logical_hosts", contract["simulation"]["method"])

    def test_transport_contract_rejects_plan_hash_mismatch(self) -> None:
        contract = simulated_transport_contract()
        for event in contract["events"]:
            if event["kind"] == "payload_enqueue":
                event["plan_hash"] = "sha256:wrong-plan"
                break
        result = validate_transport_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("plan_hash must match", "; ".join(result["errors"]))

    def test_transport_contract_rejects_queue_overflow(self) -> None:
        contract = simulated_transport_contract(max_queue_depth=2)
        for event in contract["events"]:
            if event["kind"] == "backpressure":
                event["queue_depth"] = 3
                break
        result = validate_transport_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("queue_depth exceeds", "; ".join(result["errors"]))

    def test_transport_contract_rejects_unacked_payload(self) -> None:
        contract = simulated_transport_contract()
        contract["events"] = [
            event
            for event in contract["events"]
            if not (
                event["kind"] == "payload_ack"
                and event.get("payload_id") == "activation-0"
            )
        ]
        contract["summary"]["event_count"] = len(contract["events"])
        contract["summary"]["ack_count"] = 1
        result = validate_transport_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("activation-0 has no terminal", "; ".join(result["errors"]))

    def test_transport_contract_rejects_short_timeout(self) -> None:
        contract = simulated_transport_contract(timeout_ms=50.0)
        for event in contract["events"]:
            if event["kind"] == "timeout":
                event["elapsed_ms"] = 10.0
                break
        result = validate_transport_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("elapsed_ms must be >= timeout_ms", "; ".join(result["errors"]))

    def test_transport_contract_rejects_missing_gpu_binding(self) -> None:
        contract = simulated_transport_contract()
        contract["endpoints"][0]["worker_environment"] = {}
        result = validate_transport_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("CUDA_VISIBLE_DEVICES", "; ".join(result["errors"]))


    def test_trust_boundary_fixture_passes(self) -> None:
        result = validate_trust_boundary("fornax/golden_vectors/trust_boundary")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(3, result["summary"]["identity_count"])
        self.assertEqual(2, result["summary"]["accepted_auth_count"])
        self.assertEqual(4, result["summary"]["rejected_auth_count"])
        self.assertTrue(result["summary"]["stale_plan_rejected"])

    def test_simulated_trust_boundary_validates_identity_auth_and_plan_tags(self) -> None:
        contract = simulate_trust_boundary(
            plan_id="unit-trust-plan",
            request_id="unit-trust-request",
            plan_hash="sha256:unit-trust-plan",
        )
        result = validate_trust_boundary_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(contract["trust_policy"]["allow_anonymous"])
        self.assertEqual(
            {"anonymous_disallowed", "duplicate_nonce", "stale_plan_hash", "unknown_identity"},
            {attempt["reason"] for attempt in contract["auth_attempts"] if attempt["status"] == "rejected"},
        )

    def test_trust_boundary_rejects_signature_tamper(self) -> None:
        contract = simulate_trust_boundary()
        contract["auth_attempts"][0]["signature"] = "sha256:bad"
        result = validate_trust_boundary_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("signature must match", "; ".join(result["errors"]))

    def test_trust_boundary_rejects_missing_stale_plan_reject(self) -> None:
        contract = simulate_trust_boundary()
        contract["auth_attempts"] = [
            attempt
            for attempt in contract["auth_attempts"]
            if attempt.get("reason") != "stale_plan_hash"
        ]
        result = validate_trust_boundary_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("stale_plan_hash", "; ".join(result["errors"]))

    def test_trust_boundary_rejects_anonymous_policy_weakening(self) -> None:
        contract = simulate_trust_boundary()
        contract["trust_policy"]["allow_anonymous"] = True
        result = validate_trust_boundary_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("allow_anonymous must be false", "; ".join(result["errors"]))

    def test_trust_boundary_rejects_unknown_identity_acceptance(self) -> None:
        contract = simulate_trust_boundary()
        for attempt in contract["auth_attempts"]:
            if attempt.get("reason") == "unknown_identity":
                attempt["status"] = "accepted"
                attempt.pop("reason")
                break
        result = validate_trust_boundary_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("unknown identity", "; ".join(result["errors"]))

    def test_trust_boundary_requires_duplicate_nonce_reject(self) -> None:
        contract = simulate_trust_boundary()
        contract["auth_attempts"] = [
            attempt
            for attempt in contract["auth_attempts"]
            if attempt.get("reason") != "duplicate_nonce"
        ]
        result = validate_trust_boundary_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("duplicate_nonce", "; ".join(result["errors"]))

    def test_moe_runtime_fixture_passes(self) -> None:
        result = validate_moe_contract("fornax/golden_vectors/moe_runtime")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(3, result["summary"]["expert_count"])
        self.assertEqual(2, result["summary"]["remote_dispatch_count"])
        self.assertEqual(1, result["summary"]["migration_recommendation_count"])
        self.assertIn("weighted_gather_end", result["summary"]["required_events_seen"])

    def test_simulated_moe_runtime_validates_routing_dispatch_and_gather(self) -> None:
        contract = simulated_moe_contract(
            plan_id="unit-moe-plan",
            request_id="unit-moe-request",
            plan_hash="sha256:unit-moe-plan",
            max_remote_wait_ms=5.0,
        )
        result = validate_moe_contract_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(3, contract["summary"]["expert_count"])
        self.assertGreater(contract["summary"]["remote_hit_rate"], 0.0)
        self.assertEqual("sha256:unit-moe-plan", contract["plan_hash"])

    def test_moe_runtime_rejects_plan_hash_mismatch(self) -> None:
        contract = simulated_moe_contract()
        for event in contract["events"]:
            if event["kind"] == "remote_expert_dispatch":
                event["plan_hash"] = "sha256:wrong-plan"
                break
        result = validate_moe_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("plan_hash must match", "; ".join(result["errors"]))

    def test_moe_runtime_rejects_remote_wait_over_budget(self) -> None:
        contract = simulated_moe_contract(max_remote_wait_ms=5.0)
        for event in contract["events"]:
            if event["kind"] == "remote_expert_dispatch":
                event["remote_wait_ms"] = 12.0
                break
        result = validate_moe_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("remote_wait_ms exceeds", "; ".join(result["errors"]))

    def test_moe_runtime_rejects_missing_weighted_gather(self) -> None:
        contract = simulated_moe_contract()
        contract["events"] = [
            event for event in contract["events"] if not event["kind"].startswith("weighted_gather")
        ]
        contract["summary"]["event_count"] = len(contract["events"])
        contract["summary"]["weighted_gather_count"] = 0
        result = validate_moe_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("weighted_gather", "; ".join(result["errors"]))

    def test_moe_runtime_rejects_bad_topk_weights(self) -> None:
        contract = simulated_moe_contract()
        contract["routing_trace"][0]["topk_weights"] = [0.9, 0.9]
        result = validate_moe_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("topk_weights must sum", "; ".join(result["errors"]))

    def test_moe_runtime_rejects_migration_below_threshold(self) -> None:
        contract = simulated_moe_contract(migration_hotness_threshold=0.5)
        for event in contract["events"]:
            if event["kind"] == "migration_recommendation":
                event["hotness"] = 0.1
                break
        result = validate_moe_contract_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("hotness must be", "; ".join(result["errors"]))



    def test_moe_migration_fixture_passes(self) -> None:
        result = validate_moe_hot_expert_migration("fornax/golden_vectors/moe_migration")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(1, result["summary"]["migration_count"])
        self.assertGreater(result["summary"]["remote_token_copy_reduction"], 0)

    def test_simulated_moe_migration_reduces_remote_calls_and_preserves_parity(self) -> None:
        contract = simulated_moe_hot_expert_migration(
            plan_id="unit-migration-plan",
            request_id="unit-migration-request",
            plan_hash="sha256:unit-migration-plan",
        )
        result = validate_moe_hot_expert_migration_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(contract["result"]["hot_expert_migrated"])
        self.assertGreater(contract["result"]["remote_token_copy_reduction"], 0)
        self.assertEqual(0, contract["result"]["max_post_logit_abs_error"])
        self.assertEqual("sha256:unit-migration-plan", contract["plan_hash"])

    def test_moe_migration_rejects_missing_remote_reduction(self) -> None:
        contract = simulated_moe_hot_expert_migration()
        contract["result"]["post_remote_token_copies"] = contract["result"]["pre_remote_token_copies"]
        contract["result"]["remote_token_copy_reduction"] = 0
        contract["summary"]["remote_token_copy_reduction"] = 0
        result = validate_moe_hot_expert_migration_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("remote_token_copy_reduction", "; ".join(result["errors"]))

    def test_moe_migration_rejects_failed_parity(self) -> None:
        contract = simulated_moe_hot_expert_migration()
        contract["result"]["correctness_passed"] = False
        contract["summary"]["correctness_passed"] = False
        contract["result"]["next_tokens_match"] = False
        result = validate_moe_hot_expert_migration_fixture(contract)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("next_tokens_match", text)
        self.assertIn("correctness_passed", text)

    def test_moe_migration_rejects_missing_placement_commit(self) -> None:
        contract = simulated_moe_hot_expert_migration()
        contract["events"] = [
            event for event in contract["events"] if event["kind"] != "placement_committed"
        ]
        contract["summary"]["event_count"] = len(contract["events"])
        result = validate_moe_hot_expert_migration_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("required migration sequence", "; ".join(result["errors"]))

    def test_model_support_fixture_passes(self) -> None:
        result = validate_model_support_matrix("fornax/golden_vectors/model_support")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(2, result["summary"]["model_count"])
        self.assertEqual(1, result["summary"]["supported_model_count"])
        self.assertIn("tokenizer", result["summary"]["required_capabilities_seen"])
        self.assertIn("required_before_t2", result["summary"]["parity_statuses"])

    def test_simulated_model_support_matrix_records_target_gap(self) -> None:
        matrix = simulated_model_support_matrix(
            matrix_id="unit-model-support",
            target_model_id="unit-qwen3-target",
        )
        result = validate_model_support_matrix_fixture(matrix)
        self.assertTrue(result["ok"], result["errors"])
        target = next(row for row in matrix["models"] if row["role"] == "target_candidate")
        self.assertEqual("planned", target["support_level"])
        self.assertEqual("required_before_t2", target["tokenizer"]["hash_status"])
        self.assertEqual("required_before_t2", target["parity"]["status"])

    def test_model_support_report_renders_matrix(self) -> None:
        report = render_model_support_matrix_report("fornax/golden_vectors/model_support")
        self.assertTrue(report["ok"], report["validation"]["errors"])
        markdown = report["markdown"]
        self.assertIn("# Model Support Matrix", markdown)
        self.assertIn("fornax-tiny-moe-fixture", markdown)
        self.assertIn("qwen3-moe-class-target", markdown)
        self.assertIn("required_before_t2", markdown)

    def test_model_support_rejects_missing_required_capability(self) -> None:
        matrix = simulated_model_support_matrix()
        matrix["required_capabilities"].remove("tool_calling")
        result = validate_model_support_matrix_fixture(matrix)
        self.assertFalse(result["ok"])
        self.assertIn("required_capabilities missing", "; ".join(result["errors"]))
        self.assertIn("tool_calling", "; ".join(result["errors"]))

    def test_model_support_rejects_unresolved_supported_tokenizer_hash(self) -> None:
        matrix = simulated_model_support_matrix()
        fixture = matrix["models"][0]
        fixture["tokenizer"]["hash_status"] = "required_before_t2"
        fixture["tokenizer"].pop("hash", None)
        fixture["tokenizer"]["hash_required_before"] = "later"
        result = validate_model_support_matrix_fixture(matrix)
        self.assertFalse(result["ok"])
        self.assertIn("hash_status must be resolved", "; ".join(result["errors"]))

    def test_model_support_rejects_supported_model_without_tool_support(self) -> None:
        matrix = simulated_model_support_matrix()
        fixture = matrix["models"][0]
        fixture["serving_semantics"]["tool_calling"]["status"] = "planned"
        result = validate_model_support_matrix_fixture(matrix)
        self.assertFalse(result["ok"])
        self.assertIn("tool_calling.status", "; ".join(result["errors"]))

    def test_model_support_rejects_false_parity_claim(self) -> None:
        matrix = simulated_model_support_matrix()
        target = matrix["models"][1]
        target["parity"]["status"] = "passed"
        target["parity"]["evidence"] = {
            "kind": "spec",
            "source": "unit-test",
            "note": "not measured",
        }
        result = validate_model_support_matrix_fixture(matrix)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("evidence must be measured", text)
        self.assertIn("cannot be passed", text)


    def test_stage_replication_fixture_passes(self) -> None:
        result = validate_stage_replication("fornax/golden_vectors/stage_replication")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(2, result["summary"]["replica_count"])
        self.assertGreater(result["summary"]["speedup"], 1.0)

    def test_simulated_stage_replication_uses_all_replicas_and_matches_outputs(self) -> None:
        contract = simulate_stage_replication(plan_id="unit-stage-replication")
        result = validate_stage_replication_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual({"stage-1-replica-0", "stage-1-replica-1"}, set(contract["result"]["used_replica_ids"]))
        self.assertTrue(contract["result"]["outputs_match_reference"])
        self.assertGreaterEqual(contract["result"]["speedup"], contract["config"]["speedup_floor"])

    def test_stage_replication_rejects_unused_replica(self) -> None:
        contract = simulate_stage_replication()
        for assignment in contract["assignments"]:
            assignment["replica_id"] = "stage-1-replica-0"
        contract["result"]["used_replica_ids"] = ["stage-1-replica-0"]
        result = validate_stage_replication_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("use every replicated replica", "; ".join(result["errors"]))

    def test_stage_replication_rejects_speedup_below_floor(self) -> None:
        contract = simulate_stage_replication(speedup_floor=2.1)
        result = validate_stage_replication_fixture(contract)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("speedup", text)
        self.assertIn("correctness_passed", text)

    def test_stage_replication_rejects_output_mismatch(self) -> None:
        contract = simulate_stage_replication()
        contract["assignments"][0]["max_abs_error"] = 1.0
        contract["result"]["max_abs_error"] = 1.0
        contract["result"]["outputs_match_reference"] = False
        contract["result"]["correctness_passed"] = False
        contract["summary"]["max_abs_error"] = 1.0
        contract["summary"]["correctness_passed"] = False
        result = validate_stage_replication_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("outputs_match_reference", "; ".join(result["errors"]))

    def test_resilience_replay_fixture_passes(self) -> None:
        result = validate_resilience_replay("fornax/golden_vectors/resilience_replay")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(3, result["summary"]["request_count"])
        self.assertEqual(0, result["summary"]["dropped_token_count"])

    def test_simulated_resilience_replay_zero_dropped_in_flight(self) -> None:
        contract = simulate_resilience_replay(plan_id="unit-resilience-replay")
        result = validate_resilience_replay_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(contract["summary"]["zero_dropped_in_flight"])
        self.assertEqual(
            contract["summary"]["request_count"],
            contract["summary"]["replayed_request_count"],
        )
        self.assertEqual(0.0, contract["summary"]["max_abs_error"])

    def test_resilience_replay_rejects_dropped_token(self) -> None:
        contract = simulate_resilience_replay()
        contract["results"][0]["completed_tokens"] = contract["results"][0]["completed_tokens"][:-1]
        contract["results"][0]["dropped_token_count"] = 1
        contract["results"][0]["completed"] = False
        contract["summary"]["dropped_token_count"] = 1
        contract["summary"]["zero_dropped_in_flight"] = False
        contract["summary"]["correctness_passed"] = False
        result = validate_resilience_replay_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("dropped_token_count", "; ".join(result["errors"]))

    def test_resilience_replay_rejects_duplicate_schedule(self) -> None:
        contract = simulate_resilience_replay()
        duplicate = dict(
            next(event for event in contract["events"] if event["kind"] == "replay_scheduled")
        )
        contract["events"].append(duplicate)
        contract["summary"]["event_count"] = len(contract["events"])
        result = validate_resilience_replay_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("scheduled more than once", "; ".join(result["errors"]))

    def test_resilience_replay_rejects_late_replay(self) -> None:
        contract = simulate_resilience_replay(
            replay_delay_s=0.050,
            max_replay_delay_s=0.025,
        )
        result = validate_resilience_replay_fixture(contract)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("max_replay_delay_s", text)
        self.assertIn("replay_delay_within_budget", text)

    def test_ops_lifecycle_fixture_passes(self) -> None:
        result = validate_ops_lifecycle("fornax/golden_vectors/ops_lifecycle")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(6, result["summary"]["action_count"])
        self.assertEqual(0, result["summary"]["dropped_in_flight_count"])

    def test_simulated_ops_lifecycle_validates_required_actions(self) -> None:
        contract = simulate_ops_lifecycle(plan_id="unit-ops-lifecycle")
        result = validate_ops_lifecycle_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(contract["summary"]["rollback_verified"])
        self.assertTrue(contract["summary"]["node_replace_verified"])
        self.assertTrue(contract["summary"]["correctness_passed"])
        self.assertEqual(
            {"cluster.yaml", "model.yaml", "placement.json"},
            set(contract["operator_configs"]),
        )

    def test_ops_lifecycle_rejects_mutation_without_drain(self) -> None:
        contract = simulate_ops_lifecycle()
        contract["events"] = [
            event
            for event in contract["events"]
            if not (
                event["kind"] == "drain_completed"
                and event["action"] == "upgrade"
            )
        ]
        contract["summary"]["event_count"] = len(contract["events"])
        result = validate_ops_lifecycle_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("drain_completed", "; ".join(result["errors"]))

    def test_ops_lifecycle_rejects_dropped_in_flight(self) -> None:
        contract = simulate_ops_lifecycle()
        for event in contract["events"]:
            if event["kind"] == "drain_completed":
                event["dropped_in_flight_count"] = 1
                break
        contract["request_accounting"]["dropped_in_flight_total"] = 1
        contract["summary"]["dropped_in_flight_count"] = 1
        contract["summary"]["correctness_passed"] = False
        result = validate_ops_lifecycle_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("dropped_in_flight", "; ".join(result["errors"]))

    def test_ops_lifecycle_rejects_missing_operator_config(self) -> None:
        contract = simulate_ops_lifecycle()
        del contract["operator_configs"]["model.yaml"]
        contract["summary"]["config_artifacts_present"] = False
        result = validate_ops_lifecycle_fixture(contract)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("model.yaml", text)
        self.assertIn("config_artifacts_present", text)

    def test_onboarding_methodology_fixture_passes(self) -> None:
        result = validate_onboarding_methodology("fornax/golden_vectors/onboarding_methodology")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(4, result["summary"]["track_count"])
        self.assertEqual(5, result["summary"]["document_count"])
        self.assertFalse(result["summary"]["product_ga_complete"])

    def test_simulated_onboarding_methodology_validates_required_materials(self) -> None:
        contract = simulate_onboarding_methodology(plan_id="unit-onboarding")
        result = validate_onboarding_methodology_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(contract["summary"]["required_tracks_present"])
        self.assertTrue(contract["summary"]["required_documents_present"])
        self.assertTrue(contract["benchmark_methodology"]["lab_reference_required"])
        self.assertTrue(contract["benchmark_methodology"]["correctness_first"])

    def test_onboarding_methodology_rejects_missing_glossary_term(self) -> None:
        contract = simulate_onboarding_methodology()
        contract["glossary_terms"] = [
            term for term in contract["glossary_terms"] if term["term_id"] != "benchmark_of_record"
        ]
        contract["summary"]["glossary_term_count"] = len(contract["glossary_terms"])
        contract["summary"]["required_glossary_terms_present"] = False
        result = validate_onboarding_methodology_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("benchmark_of_record", "; ".join(result["errors"]))

    def test_onboarding_methodology_rejects_missing_lab_reference_boundary(self) -> None:
        contract = simulate_onboarding_methodology()
        contract["benchmark_methodology"]["lab_reference_required"] = False
        contract["summary"]["lab_reference_required"] = False
        result = validate_onboarding_methodology_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("lab_reference_required", "; ".join(result["errors"]))

    def test_onboarding_methodology_rejects_track_without_first_run_command(self) -> None:
        contract = simulate_onboarding_methodology()
        contract["tracks"][0]["first_run_commands"] = []
        result = validate_onboarding_methodology_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("first_run_commands", "; ".join(result["errors"]))

    def test_onboarding_methodology_rejects_missing_required_document(self) -> None:
        contract = simulate_onboarding_methodology()
        del contract["documents"]["glossary.md"]
        contract["summary"]["document_count"] = len(contract["documents"])
        contract["summary"]["required_documents_present"] = False
        result = validate_onboarding_methodology_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("glossary.md", "; ".join(result["errors"]))

    def test_program_governance_fixture_passes(self) -> None:
        result = validate_program_governance("fornax/golden_vectors/program_governance")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(6, result["summary"]["decision_count"])
        self.assertTrue(result["summary"]["dec005_pending"])
        self.assertFalse(result["summary"]["g1_gate_ready"])

    def test_simulated_program_governance_validates_controls(self) -> None:
        contract = simulate_program_governance(plan_id="unit-program-governance")
        result = validate_program_governance_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(contract["summary"]["g1_gate_ready"])
        self.assertTrue(contract["summary"]["silent_proceed_forbidden"])
        self.assertTrue(contract["external_watch"]["local_probe_required"])
        self.assertEqual({"X1", "X2", "X3"}, set(contract["governance_scope"]["workstreams"]))

    def test_program_governance_rejects_dec005_proceed_claim(self) -> None:
        contract = simulate_program_governance()
        for entry in contract["decision_log"]["entries"]:
            if entry["id"] == "DEC-005":
                entry["status"] = "Accepted"
                entry["decision"] = "G1 PROCEED"
        contract["summary"]["g1_gate_ready"] = True
        contract["summary"]["dec005_pending"] = False
        result = validate_program_governance_fixture(contract)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("DEC-005.status", text)
        self.assertIn("g1_gate_ready", text)

    def test_program_governance_rejects_missing_status_drift_control(self) -> None:
        contract = simulate_program_governance()
        contract["raid_register"]["risks"] = [
            risk for risk in contract["raid_register"]["risks"] if risk["id"] != "R-10"
        ]
        contract["summary"]["risk_count"] = len(contract["raid_register"]["risks"])
        contract["summary"]["status_drift_controlled"] = False
        result = validate_program_governance_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("R-10", "; ".join(result["errors"]))

    def test_program_governance_rejects_blog_as_gate_record(self) -> None:
        contract = simulate_program_governance()
        contract["external_watch"]["source_precedence"][0]["gate_of_record"] = False
        contract["external_watch"]["source_precedence"][3]["gate_of_record"] = True
        result = validate_program_governance_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("rank 1", "; ".join(result["errors"]))

    def test_program_governance_rejects_missing_cadence_artifact(self) -> None:
        contract = simulate_program_governance()
        contract["cadence"]["artifacts"] = [
            item for item in contract["cadence"]["artifacts"] if item["artifact_id"] != "weekly-status"
        ]
        contract["summary"]["cadence_artifact_count"] = len(contract["cadence"]["artifacts"])
        result = validate_program_governance_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("weekly-status", "; ".join(result["errors"]))

    def test_continuous_batching_fixture_passes(self) -> None:
        result = validate_continuous_batching("fornax/golden_vectors/continuous_batching")
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["summary"]["overlap_observed"])
        self.assertEqual(3, result["summary"]["microbatch_count"])
        self.assertIn("bubble_sample", result["summary"]["required_events_seen"])

    def test_simulated_continuous_batching_validates_fairness_and_overlap(self) -> None:
        contract = simulate_continuous_batching(
            plan_id="unit-batching-plan",
            max_queue_depth=4,
            max_inflight=4,
            microbatch_size=2,
            fairness_window_s=0.05,
            transfer_s=0.002,
        )
        result = validate_continuous_batching_fixture(contract)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(
            contract["summary"]["admitted_request_order"],
            contract["summary"]["formed_request_order"],
        )
        self.assertTrue(contract["summary"]["overlap_observed"])
        self.assertLess(contract["summary"]["bubble_fraction"], 1.0)

    def test_continuous_batching_rejects_fifo_fairness_violation(self) -> None:
        contract = simulate_continuous_batching()
        order = contract["summary"]["formed_request_order"]
        contract["summary"]["formed_request_order"] = list(reversed(order))
        result = validate_continuous_batching_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("summary.formed_request_order", "; ".join(result["errors"]))

    def test_continuous_batching_rejects_missing_overlap(self) -> None:
        contract = simulate_continuous_batching()
        contract["events"] = [
            event
            for event in contract["events"]
            if event["kind"] != "stage_compute_start" or event.get("stage_index") != 1
        ]
        contract["summary"]["event_count"] = len(contract["events"])
        result = validate_continuous_batching_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("1F1B overlap", "; ".join(result["errors"]))

    def test_continuous_batching_rejects_wait_above_fairness_window(self) -> None:
        contract = simulate_continuous_batching(fairness_window_s=0.05)
        for event in contract["events"]:
            if event["kind"] == "fairness_yield":
                event["oldest_wait_s"] = 0.25
                break
        contract["summary"]["max_wait_s"] = 0.25
        result = validate_continuous_batching_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("exceeds fairness_window_s", "; ".join(result["errors"]))

    def test_continuous_batching_rejects_bubble_summary_mismatch(self) -> None:
        contract = simulate_continuous_batching()
        contract["summary"]["overlap_observed"] = False
        result = validate_continuous_batching_fixture(contract)
        self.assertFalse(result["ok"])
        self.assertIn("summary.overlap_observed", "; ".join(result["errors"]))

    def test_scheduler_contract_fixture_passes(self) -> None:
        result = validate_scheduler_contract("fornax/golden_vectors/scheduler_contract")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(4, result["summary"]["request_count"])
        self.assertEqual(1, result["summary"]["backpressure_count"])
        self.assertEqual(2, result["summary"]["max_observed_queue_depth"])
        self.assertEqual(2, result["summary"]["max_observed_inflight"])

    def test_scheduler_simulation_enforces_bounded_queue_and_microbatch(self) -> None:
        result = simulate_scheduler(
            scheduler_fixture_plan(),
            [
                {"id": "r0", "prompt_len": 8, "gen_len": 4},
                {"id": "r1", "prompt_len": 8, "gen_len": 3},
                {"id": "r2", "prompt_len": 8, "gen_len": 2},
                {"id": "r3", "prompt_len": 8, "gen_len": 1},
            ],
            plan_id="unit-scheduler",
            max_queue_depth=2,
            max_inflight=2,
            microbatch_size=2,
        )
        validation = validate_scheduler_contract(result)
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(1, result["summary"]["backpressure_count"])
        self.assertLessEqual(result["summary"]["max_observed_queue_depth"], 2)
        self.assertLessEqual(result["summary"]["max_observed_inflight"], 2)
        self.assertEqual(2, result["summary"]["microbatch_count"])

    def test_scheduler_simulation_from_paths_accepts_trace_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan_path = root / "placement.json"
            requests_path = root / "requests.json"
            write_json(plan_path, scheduler_fixture_plan())
            write_json(
                requests_path,
                {
                    "requests": [
                        {"request_id": "a", "prompt_tokens": 5, "max_new_tokens": 2},
                        {"request_id": "b", "prompt_tokens": 5, "max_new_tokens": 2},
                    ]
                },
            )
            result = simulate_scheduler_from_paths(
                plan_path,
                requests_path,
                plan_id="path-scheduler",
                max_queue_depth=2,
                max_inflight=2,
                microbatch_size=2,
            )
        validation = validate_scheduler_contract(result)
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(2, result["summary"]["completed_count"])

    def test_scheduler_contract_rejects_queue_overflow(self) -> None:
        fixture = read_json("fornax/golden_vectors/scheduler_contract/fixture.json")
        fixture["events"][0]["queue_depth"] = fixture["max_queue_depth"] + 1
        fixture["summary"]["max_observed_queue_depth"] = fixture["max_queue_depth"] + 1
        result = validate_scheduler_contract(fixture)
        self.assertFalse(result["ok"])
        self.assertIn("queue_depth exceeds", "; ".join(result["errors"]))

    def test_network_contract_fixture_passes(self) -> None:
        result = validate_network_contract(
            "fornax/golden_vectors/network_contract", mode="simulated"
        )
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("backpressure", result["summary"]["required_events_seen"])

    def test_network_contract_rejects_missing_required_events(self) -> None:
        result = validate_network_contract_fixture(
            {
                "version": 1,
                "mode": "simulated",
                "plan_id": "p1",
                "node_id": "n1",
                "max_queue_depth": 1,
                "timeout_ms": 10,
                "events": [
                    {"kind": "enqueue", "request_id": "r1", "plan_id": "p1"},
                    {"kind": "dequeue", "request_id": "r1", "plan_id": "p1"},
                ],
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("missing required simulated events", "; ".join(result["errors"]))

    def test_network_contract_rejects_queue_overflow(self) -> None:
        result = validate_network_contract_fixture(
            {
                "version": 1,
                "mode": "simulated",
                "plan_id": "p1",
                "node_id": "n1",
                "max_queue_depth": 1,
                "timeout_ms": 10,
                "events": [
                    {"kind": "enqueue", "request_id": "r1", "plan_id": "p1"},
                    {"kind": "enqueue", "request_id": "r2", "plan_id": "p1"},
                    {"kind": "backpressure", "queue_depth": 1, "plan_id": "p1"},
                    {"kind": "dequeue", "request_id": "r1", "plan_id": "p1"},
                    {"kind": "timeout", "request_id": "r3", "elapsed_ms": 10, "plan_id": "p1"},
                    {"kind": "cancel", "request_id": "r2", "plan_id": "p1"},
                    {"kind": "plan_integrity_reject", "request_id": "r4", "plan_id": "p2"},
                ],
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("enqueue exceeded max_queue_depth", "; ".join(result["errors"]))


    def test_network_contract_rejects_unknown_dequeue(self) -> None:
        result = validate_network_contract_fixture(
            {
                "version": 1,
                "mode": "simulated",
                "plan_id": "p1",
                "node_id": "n1",
                "max_queue_depth": 2,
                "timeout_ms": 10,
                "events": [
                    {"kind": "enqueue", "request_id": "r1", "plan_id": "p1"},
                    {"kind": "backpressure", "queue_depth": 2, "plan_id": "p1"},
                    {"kind": "dequeue", "request_id": "missing", "plan_id": "p1"},
                    {"kind": "timeout", "request_id": "r3", "elapsed_ms": 10, "plan_id": "p1"},
                    {"kind": "cancel", "request_id": "r1", "plan_id": "p1"},
                    {"kind": "plan_integrity_reject", "request_id": "r4", "plan_id": "p2"},
                ],
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("dequeues unknown request_id", "; ".join(result["errors"]))

    def test_network_security_spec_draft_includes_review_sections(self) -> None:
        result = render_network_security_spec_draft("fornax/golden_vectors/network_contract")
        self.assertTrue(result["ok"], result["validation"])
        markdown = result["markdown"]
        self.assertIn("Status: DRAFT", markdown)
        self.assertIn("## V0 Trust Boundary", markdown)
        self.assertIn("## Node Identity And Endpoint Auth", markdown)
        self.assertIn("## Backpressure And Queue Contract", markdown)
        self.assertIn("## Timeout, Retry, Cancel, And Partition Semantics", markdown)
        self.assertIn("plan_integrity_reject", markdown)
        self.assertIn("Phase 1b T3 lab hardware", markdown)

    def test_substrate_adr_draft_includes_source_precedence(self) -> None:
        result = render_substrate_adr_draft(
            pinned_build="max-26.4.0",
            last_checked="2026-06-20",
            status="probing",
            apple_role="capacity-only",
            local_probe={
                "platform": "macOS-arm64-lab-placeholder",
                "machine": "arm64",
                "python": "3.12",
                "tools": [
                    {
                        "tool": "max",
                        "available": True,
                        "path": "/opt/modular/bin/max",
                        "version": "MAX 26.4.0",
                    },
                    {
                        "tool": "mojo",
                        "available": True,
                        "path": "/opt/modular/bin/mojo",
                        "version": "Mojo 1.0",
                    },
                ],
                "note": "synthetic unit-test probe",
            },
        )
        self.assertTrue(result["ok"])
        markdown = result["markdown"]
        self.assertIn("Status: DRAFT", markdown)
        self.assertIn("## Source Precedence", markdown)
        self.assertIn("Local probe on the pinned build", markdown)
        self.assertIn("## Apple Plan B And Reversal Trigger", markdown)
        self.assertIn("target-model expert MLP", markdown)
        self.assertIn("capacity-only", markdown)
        self.assertIn("## Rejected Alternatives", markdown)
        self.assertIn("TL/SP review", markdown)

    def test_substrate_adr_rejects_unknown_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "status must be"):
            render_substrate_adr_draft(status="future-promise")

    def test_runtime_format_golden_fixture_passes(self) -> None:
        result = validate_runtime_format_golden("fornax/golden_vectors/runtime_format")
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual([], result["warnings"])

    def test_runtime_format_validator_rejects_bad_activation_length(self) -> None:
        manifest = {
            "version": 1,
            "activation": {
                "dtype": "fp16",
                "shape": [2, 2],
                "layout": "contiguous_row_major",
                "values": [0.0, 1.0],
            },
            "kv_page": {
                "dtype": "fp16",
                "shape": [4, 2, 4],
                "page_size": 4,
                "token_count": 2,
                "owner_stage": 0,
            },
            "expert_batch": {
                "layer_id": 0,
                "expert_ids": [1],
                "token_indices": [0],
                "topk_weights": [1.0],
                "hidden_shape": [1, 4],
                "gather_order": [0],
            },
            "tolerances": {"fp16": {"rtol": 0.001, "atol": 0.001}},
        }
        result = validate_runtime_format_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertIn("activation.values length", "; ".join(result["errors"]))

    def test_runtime_format_validator_rejects_bad_expert_gather(self) -> None:
        manifest = {
            "version": 1,
            "activation": {
                "dtype": "fp16",
                "shape": [1, 2],
                "layout": "contiguous_row_major",
                "values": [0.0, 1.0],
            },
            "kv_page": {
                "dtype": "fp16",
                "shape": [2, 1, 2],
                "page_size": 2,
                "token_count": 1,
                "owner_stage": 0,
            },
            "expert_batch": {
                "layer_id": 0,
                "expert_ids": [1, 2],
                "token_indices": [0, 1],
                "topk_weights": [0.5, 0.5],
                "hidden_shape": [2, 2],
                "gather_order": [0, 0],
            },
            "tolerances": {"fp16": {"rtol": 0.001, "atol": 0.001}},
        }
        result = validate_runtime_format_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertIn("gather_order", "; ".join(result["errors"]))


    def test_runtime_format_validator_rejects_weight_only_payload_dtype(self) -> None:
        manifest = {
            "version": 1,
            "activation": {
                "dtype": "q4",
                "shape": [1, 2],
                "layout": "contiguous_row_major",
                "values": [0.0, 1.0],
            },
            "kv_page": {
                "dtype": "fp16",
                "shape": [2, 1, 2],
                "page_size": 2,
                "token_count": 1,
                "owner_stage": 0,
            },
            "expert_batch": {
                "layer_id": 0,
                "expert_ids": [1],
                "token_indices": [0],
                "topk_weights": [1.0],
                "hidden_shape": [1, 2],
                "gather_order": [0],
            },
            "tolerances": {"fp16": {"rtol": 0.001, "atol": 0.001}},
        }
        result = validate_runtime_format_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertIn("activation.dtype", "; ".join(result["errors"]))

    def test_runtime_format_validator_rejects_kv_shape_page_mismatch(self) -> None:
        manifest = {
            "version": 1,
            "activation": {
                "dtype": "fp16",
                "shape": [1, 2],
                "layout": "contiguous_row_major",
                "values": [0.0, 1.0],
            },
            "kv_page": {
                "dtype": "fp16",
                "shape": [3, 1, 2],
                "page_size": 2,
                "token_count": 1,
                "owner_stage": 0,
            },
            "expert_batch": {
                "layer_id": 0,
                "expert_ids": [1],
                "token_indices": [0],
                "topk_weights": [1.0],
                "hidden_shape": [1, 2],
                "gather_order": [0],
            },
            "tolerances": {"fp16": {"rtol": 0.001, "atol": 0.001}},
        }
        result = validate_runtime_format_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertIn("first dimension must equal page_size", "; ".join(result["errors"]))

    def test_runtime_format_spec_draft_includes_review_sections(self) -> None:
        result = render_runtime_format_spec_draft("fornax/golden_vectors/runtime_format")
        self.assertTrue(result["ok"], result["validation"])
        markdown = result["markdown"]
        self.assertIn("Status: DRAFT", markdown)
        self.assertIn("## Activation Tensor", markdown)
        self.assertIn("## KV Page", markdown)
        self.assertIn("## Expert Batch", markdown)
        self.assertIn("## Failure Modes", markdown)
        self.assertIn("## Reference Path And Golden-Vector Method", markdown)
        self.assertIn("Send/receive reuse", markdown)
        self.assertIn("same placement plan", markdown)
        self.assertIn("q4/q8", markdown)

    def test_phase0_doctor_reports_missing_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            result = inspect_phase0_bundle(d)
        self.assertFalse(result["ok"])
        self.assertIn("missing inventory.json", result["errors"])
        self.assertIn("missing placement.json", result["errors"])

    def test_phase0_doctor_accepts_bundle_with_dry_run_warning(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d)
            (bundle / "inventory.json").write_text('{"nodes": [{"id": "n0"}], "links": []}\n', encoding="utf-8")
            (bundle / "links.json").write_text('{"links": []}\n', encoding="utf-8")
            (bundle / "target.json").write_text('{"model": {}, "target": {}}\n', encoding="utf-8")
            (bundle / "placement.json").write_text('{"feasible": true, "predicted": {}}\n', encoding="utf-8")
            (bundle / "validate.json").write_text('{"valid": true}\n', encoding="utf-8")
            (bundle / "simulate.json").write_text('{"predicted": {}}\n', encoding="utf-8")
            (bundle / "benchmark.json").write_text('{"measured": false}\n', encoding="utf-8")
            result = inspect_phase0_bundle(bundle)
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("benchmark.json is a dry-run prediction, not measured evidence", result["warnings"])
        self.assertIn("missing G1 gate artifact: runtime_format_spec", result["warnings"])
        self.assertTrue(result["artifacts"]["validate.json"]["valid"])

    def test_phase0_doctor_recognizes_g1_gate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d)
            (bundle / "adr").mkdir()
            (bundle / "inventory.json").write_text(
                '{"nodes": [{"id": "n0"}], "links": []}\n', encoding="utf-8"
            )
            (bundle / "links.json").write_text(
                '{"links": [], "measured": true}\n', encoding="utf-8"
            )
            (bundle / "target.json").write_text('{"model": {}, "target": {}}\n', encoding="utf-8")
            (bundle / "placement.json").write_text('{"feasible": true, "predicted": {}}\n', encoding="utf-8")
            (bundle / "validate.json").write_text('{"valid": true}\n', encoding="utf-8")
            (bundle / "simulate.json").write_text('{"predicted": {}}\n', encoding="utf-8")
            (bundle / "benchmark.json").write_text('{"measured": true}\n', encoding="utf-8")
            write_json(
                bundle / "calibration.json",
                run_local_calibration(
                    cpu_memory_bytes=1024,
                    cpu_memory_iterations=1,
                    cpu_compute_iterations=10,
                    try_torch_cuda=False,
                ),
            )
            (bundle / "runtime-format-and-invariants.md").write_text("draft\n", encoding="utf-8")
            (bundle / "networking-security-and-backpressure.md").write_text("draft\n", encoding="utf-8")
            (bundle / "adr" / "0001-max-mojo-substrate.md").write_text("draft\n", encoding="utf-8")
            (bundle / "apple-probe.json").write_text('{"probe_kind": "apple-expert-mlp"}\n', encoding="utf-8")
            (bundle / "apple-probe-validation.json").write_text(
                '{"valid": true, "recommended_role": "capacity-only"}\n',
                encoding="utf-8",
            )
            (bundle / "apple-role-decision.md").write_text("draft\n", encoding="utf-8")
            (bundle / "roadmap-staffing-rebaseline.md").write_text("draft\n", encoding="utf-8")
            result = inspect_phase0_bundle(bundle)
        self.assertTrue(result["ok"], result["errors"])
        warning_text = "; ".join(result["warnings"])
        self.assertNotIn("missing G1 gate artifact", warning_text)
        self.assertNotIn("missing recommended calibration.json", warning_text)
        self.assertEqual(
            "capacity-only",
            result["artifacts"]["apple_probe_validation"]["recommended_role"],
        )

    def test_phase0_doctor_rejects_empty_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d)
            (bundle / "inventory.json").write_text('{"nodes": [], "links": []}\n', encoding="utf-8")
            (bundle / "links.json").write_text('{"links": []}\n', encoding="utf-8")
            (bundle / "placement.json").write_text('{"feasible": true}\n', encoding="utf-8")
            result = inspect_phase0_bundle(bundle)
        self.assertFalse(result["ok"])
        self.assertIn("inventory.json must contain at least one node", result["errors"])

    def test_target_contract_validation_passes_fixture(self) -> None:
        model, target, bundle = load_target_contract(
            "fornax/golden_plans/v0_target_contract_fixture.md"
        )
        inventory = inventory_with_link()
        result = validate_target_contract(model, target, bundle, inventory)
        self.assertTrue(result["valid"], result["checks"])
        self.assertGreater(result["memory"]["minimum_headroom_fraction"], 0.05)

    def test_target_contract_validation_rejects_missing_gate_metadata(self) -> None:
        model = dense_model(2)
        target = Target(4, 16, 8)
        bundle = {"model": model.to_dict(), "target": target.to_dict()}
        result = validate_target_contract(model, target, bundle, inventory_with_link())
        self.assertFalse(result["valid"])
        failed = {check["name"] for check in result["checks"] if not check["passed"]}
        self.assertIn("contract.metadata_present", failed)
        self.assertIn("contract.kill_metric", failed)
        self.assertIn("contract.concurrency_sweep", failed)

    def test_target_contract_validation_rejects_unmet_throughput_threshold(self) -> None:
        model, target, bundle = load_target_contract(
            "fornax/golden_plans/v0_target_contract_fixture.md"
        )
        bundle = dict(bundle)
        bundle["contract"] = dict(bundle["contract"])
        bundle["contract"]["throughput_threshold_tok_s"] = 1e12
        result = validate_target_contract(model, target, bundle, inventory_with_link())
        self.assertFalse(result["valid"])
        failed = {check["name"] for check in result["checks"] if not check["passed"]}
        self.assertIn("planner.throughput_threshold_met", failed)

    def test_target_contract_draft_is_executable_markdown_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inventory_path = root / "inventory.json"
            links_path = root / "links.json"
            write_json(
                inventory_path,
                {
                    "nodes": [
                        {
                            "id": "fast",
                            "vendor": "nvidia",
                            "runtime": "max",
                            "mem_free_bytes": 16_000_000,
                            "compute_class": 4_000_000_000_000.0,
                            "mem_bandwidth_bytes_s": 400_000_000_000.0,
                            "supports_stage": True,
                            "supports_expert_worker": True,
                            "supports_kv": True,
                            "supported_dtypes": ["fp16"],
                        }
                    ],
                    "links": [],
                },
            )
            write_json(links_path, {"links": [], "measured": True})
            result = render_target_contract_draft(
                source_path="fornax/golden_plans/v0_target_contract_fixture.md",
                inventory_path=inventory_path,
                links_path=links_path,
            )
            out = root / "v0-target-contract.md"
            out.write_text(result["markdown"], encoding="utf-8")
            model, target, bundle = load_target_contract(out)
            validation = validate_target_contract(
                model, target, bundle, load_inventory(inventory_path, links_path)
            )
        self.assertTrue(result["valid"], result["validation"]["checks"])
        self.assertTrue(validation["valid"], validation["checks"])
        self.assertIn("Status: DRAFT", result["markdown"])
        self.assertIn("## Memory Budget", result["markdown"])
        self.assertIn("## Placement Explanations", result["markdown"])
        self.assertIn("```json fornax-target", result["markdown"])
        self.assertTrue(result["machine_bundle"]["evidence"]["placement_explanations"])
        self.assertEqual(
            "draft_not_signed_off", result["machine_bundle"]["evidence"]["status"]
        )

    def test_request_trace_summary_supports_phase0_simulate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "requests.json"
            path.write_text(
                '{"requests":[{"prompt_len":10,"gen_len":5},{"prompt_tokens":7,"max_new_tokens":3}]}\n',
                encoding="utf-8",
            )
            summary = summarize_request_trace(path)
        self.assertEqual(2, summary["request_count"])
        self.assertEqual(17, summary["total_prompt_tokens"])
        self.assertEqual(8, summary["total_generation_tokens"])
        result = simulation_result({"throughput_tok_s": 4.0}, summary)
        self.assertEqual(2.0, result["requests"]["predicted_decode_wall_time_s"])

    def test_request_trace_summary_rejects_bad_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text('{"requests":{"not":"a list"}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requests"):
                summarize_request_trace(path)

    def test_nvidia_smi_inventory_parser_discovers_gpu_nodes(self) -> None:
        csv_text = "0, NVIDIA H100 80GB HBM3, 80000, 81559, 575.57.08\n1, NVIDIA H100 80GB HBM3, 79000, 81559, 575.57.08\n"
        rows = parse_nvidia_smi_csv(csv_text)
        self.assertEqual(2, len(rows))
        inventory = collect_local_inventory(nvidia_smi_csv=csv_text)
        gpu_nodes = [node for node in inventory["nodes"] if node["vendor"] == "nvidia"]
        self.assertEqual(["gpu0", "gpu1"], [node["id"] for node in gpu_nodes])
        self.assertEqual("cuda:0", gpu_nodes[0]["device"])
        self.assertEqual(int(80000 * 1024 * 1024 * 0.90), gpu_nodes[0]["mem_free_bytes"])
        self.assertIn("static_estimate", gpu_nodes[0]["measurement"]["compute_class"])
        self.assertIn("nvidia.memory_free_mib", inventory["measured_fields"])
        self.assertIn("compute_class", inventory["estimated_fields"])

    def test_nvidia_smi_parser_rejects_unexpected_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 5"):
            parse_nvidia_smi_csv("0, NVIDIA H100\n")

    def test_simulated_cluster_inventory_splits_two_gpus_into_logical_hosts(self) -> None:
        csv_text = (
            "0, NVIDIA H100 80GB HBM3, 80000, 81559, 575.57.08\n"
            "1, NVIDIA H100 80GB HBM3, 79000, 81559, 575.57.08\n"
        )
        source = collect_local_inventory(nvidia_smi_csv=csv_text)
        result = build_logical_cluster_inventory(
            source,
            link_bandwidth_bytes_s=12.5e9,
            link_latency_s=0.0004,
            slow_node_factor=0.5,
        )
        self.assertEqual("logical_multi_host", result["simulation"]["mode"])
        self.assertEqual("two-gpu-heterogeneous", result["simulation"]["profile"])
        self.assertEqual(["sim-host-0", "sim-host-1"], [n["host_id"] for n in result["nodes"]])
        self.assertEqual(["cuda:0", "cuda:0"], [n["device"] for n in result["nodes"]])
        self.assertEqual("0", result["nodes"][0]["worker_environment"]["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("1", result["nodes"][1]["worker_environment"]["CUDA_VISIBLE_DEVICES"])
        self.assertLess(result["nodes"][1]["compute_class"], result["nodes"][0]["compute_class"])
        self.assertEqual(1, len(result["links"]))
        link = result["links"][0]
        self.assertEqual(12.5e9, link["bandwidth_bytes_s"])
        self.assertEqual(0.0004, link["latency_s"])
        self.assertTrue(link["measurement"]["simulated"])
        self.assertFalse(link["measurement"]["measured"])

    def test_fabric_probe_synthesizes_same_host_local_links(self) -> None:
        inventory = {
            "host_id": "host-a",
            "nodes": [
                {"id": "cpu0", "vendor": "cpu", "host_id": "host-a"},
                {"id": "gpu0", "vendor": "nvidia", "host_id": "host-a"},
                {"id": "gpu1", "vendor": "nvidia", "host_id": "host-a"},
            ],
            "links": [],
        }
        result = probe_declared_links(inventory)
        link_pairs = {tuple(sorted((link["a"], link["b"]))) for link in result["links"]}
        self.assertEqual({("cpu0", "gpu0"), ("cpu0", "gpu1"), ("gpu0", "gpu1")}, link_pairs)
        self.assertFalse(result["measured"])
        self.assertEqual(3, result["estimated_link_count"])
        self.assertIn("no active fabric measurements recorded", result["warnings"])
        gpu_link = next(
            link
            for link in result["links"]
            if tuple(sorted((link["a"], link["b"]))) == ("gpu0", "gpu1")
        )
        self.assertIn("gpu_peer", gpu_link["measurement"]["source"])

    def test_fabric_probe_can_replace_same_host_estimates_with_active_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            fake_python = Path(d) / "fake-torch-python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "req = json.load(sys.stdin)\n"
                "links = []\n"
                "for item in req['links']:\n"
                "    links.append({\n"
                "        'a': item['a'], 'b': item['b'],\n"
                "        'bandwidth_bytes_s': 123456789.0,\n"
                "        'latency_s': 0.000123,\n"
                "        'measurement': {\n"
                "            'measured': True,\n"
                "            'source': 'fornax.inventory.active_local_torch_copy',\n"
                "            'backend': 'torch',\n"
                "            'backend_mode': 'external_python',\n"
                "            'python_executable': sys.executable,\n"
                "            'size_bytes': req['size_bytes'],\n"
                "            'iterations': req['iterations'],\n"
                "        },\n"
                "    })\n"
                "print(json.dumps({'ok': True, 'links': links}))\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            inventory = {
                "host_id": "host-a",
                "nodes": [
                    {"id": "cpu0", "vendor": "cpu", "host_id": "host-a"},
                    {"id": "gpu0", "vendor": "nvidia", "host_id": "host-a", "device": "cuda:0"},
                    {"id": "gpu1", "vendor": "nvidia", "host_id": "host-a", "device": "cuda:1"},
                ],
                "links": [
                    {
                        "a": "cpu0",
                        "b": "gpu0",
                        "bandwidth_bytes_s": 1.0,
                        "latency_s": 1.0,
                    }
                ],
            }
            result = probe_declared_links(
                inventory,
                active_local=True,
                torch_python=str(fake_python),
                active_local_bytes=1024,
                active_local_iterations=2,
            )
        self.assertTrue(result["measured"])
        self.assertEqual(3, result["active_measurement_count"])
        self.assertEqual(0, result["estimated_link_count"])
        self.assertEqual(3, len(result["links"]))
        self.assertNotIn("no active fabric measurements recorded", result["warnings"])
        self.assertTrue(result["active_probe"]["requested"])
        for link in result["links"]:
            self.assertTrue(link["measurement"]["measured"])
            self.assertEqual(
                "fornax.inventory.active_local_torch_copy",
                link["measurement"]["source"],
            )
            self.assertEqual(123456789.0, link["bandwidth_bytes_s"])

    def test_fabric_probe_warns_when_declared_link_remains_unmeasured(self) -> None:
        inventory = {
            "host_id": "host-a",
            "nodes": [
                {"id": "gpu0", "vendor": "nvidia", "host_id": "host-a", "device": "cuda:0"},
                {"id": "remote", "vendor": "nvidia", "host_id": "host-b", "device": "cuda:1"},
            ],
            "links": [
                {
                    "a": "gpu0",
                    "b": "remote",
                    "bandwidth_bytes_s": 1.0,
                    "latency_s": 1.0,
                }
            ],
        }
        result = probe_declared_links(inventory, active_local=True, torch_python="/missing/python")
        self.assertFalse(result["measured"])
        self.assertEqual(0, result["active_measurement_count"])
        self.assertEqual(1, result["estimated_link_count"])
        self.assertIn("links include unmeasured declarations", result["warnings"])

    def test_phase0_preflight_writes_doctorable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trace = root / "requests.json"
            trace.write_text('{"requests":[{"prompt_len":8,"gen_len":4}]}\n', encoding="utf-8")
            bundle = root / "bundle"
            result = run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                requests_path=trace,
                benchmark_iterations=1,
                inventory_data={
                    "nodes": [
                        {
                            "id": "fast",
                            "vendor": "nvidia",
                            "runtime": "max",
                            "mem_free_bytes": 16_000_000,
                            "compute_class": 4_000_000_000_000.0,
                            "mem_bandwidth_bytes_s": 400_000_000_000.0,
                            "supports_stage": True,
                            "supports_expert_worker": True,
                            "supports_kv": True,
                            "supported_dtypes": ["fp16"],
                        }
                    ],
                    "links": [],
                    "source": "test",
                },
            )
            self.assertTrue(result["ok"], result["doctor"])
            self.assertTrue((bundle / "v0-target-contract.md").exists())
            self.assertTrue((bundle / "doctor.json").exists())
            doctor = inspect_phase0_bundle(bundle)
            self.assertTrue(doctor["ok"])
            self.assertIn("missing G1 gate artifact: runtime_format_spec", doctor["warnings"])
            self.assertIn("missing G1 gate artifact: apple_probe", doctor["warnings"])
            self.assertIn("benchmark_ledger", result["artifacts"])
            self.assertTrue(doctor["artifacts"]["benchmark_ledger"]["valid"])
            ledger = validate_benchmark_ledger(bundle / "benchmark-ledger.jsonl")
            self.assertTrue(ledger["ok"], ledger["errors"])
            benchmark = (bundle / "benchmark.json").read_text(encoding="utf-8")
            simulate = (bundle / "simulate.json").read_text(encoding="utf-8")
            self.assertIn('"measured": true', benchmark)
            self.assertIn('"request_count": 1', simulate)

    def test_phase0_preflight_accepts_simulated_logical_cluster_inventory(self) -> None:
        csv_text = (
            "0, NVIDIA H100 80GB HBM3, 80000, 81559, 575.57.08\n"
            "1, NVIDIA H100 80GB HBM3, 79000, 81559, 575.57.08\n"
        )
        source = collect_local_inventory(nvidia_smi_csv=csv_text)
        inventory = build_logical_cluster_inventory(source)
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "bundle"
            result = run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                benchmark_iterations=1,
                inventory_data=inventory,
                include_golden_plans=True,
            )
            doctor = inspect_phase0_bundle(bundle)
            review = render_g1_gate_review_draft(
                bundle, review_date="2026-06-21", plan_version="v3"
            )
            self.assertTrue(result["ok"], result["doctor"])
            self.assertTrue(doctor["ok"], doctor)
            self.assertTrue((bundle / "golden-plans.json").exists())
            self.assertTrue((bundle / "benchmark-ledger.jsonl").exists())
            self.assertIn("golden_plans", result["artifacts"])
            self.assertIn("benchmark_ledger", result["artifacts"])
            self.assertTrue(doctor["artifacts"]["benchmark_ledger"]["valid"])
            self.assertNotIn("golden-plan tests T0 green", review["machine_missing_criteria"])
            self.assertEqual(
                "logical_multi_host",
                doctor["artifacts"]["inventory.json"]["simulation_mode"],
            )
            self.assertIn(
                "inventory.json is simulated logical-cluster evidence, not real multi-host hardware evidence",
                doctor["warnings"],
            )
            self.assertIn(
                "links.json: links include unmeasured declarations",
                doctor["warnings"],
            )
            self.assertIn(
                "links.json: no active fabric measurements recorded",
                doctor["warnings"],
            )

    def test_phase0_status_report_labels_simulated_evidence_and_open_gaps(self) -> None:
        csv_text = (
            "0, NVIDIA H100 80GB HBM3, 80000, 81559, 575.57.08\n"
            "1, NVIDIA H100 80GB HBM3, 79000, 81559, 575.57.08\n"
        )
        source = collect_local_inventory(nvidia_smi_csv=csv_text)
        inventory = build_logical_cluster_inventory(source)
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "bundle"
            run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                benchmark_iterations=1,
                inventory_data=inventory,
                include_g1_drafts=True,
                include_golden_plans=True,
                include_program_reports=True,
                program_report_date="2026-06-21",
                include_simulated_apple_evidence=True,
                simulated_apple_role="capacity-only",
                substrate_pinned_build="max-26.4.0",
                kickoff_date="2026-06-21",
                ker_status="unavailable",
                scope="pending",
            )
            report = render_phase0_status_report(
                bundle, report_date="2026-06-21", plan_version="v3"
            )
            program_report_files = (
                (bundle / "g1-gate-review.md").exists(),
                (bundle / "phase0-status.json").exists(),
                (bundle / "phase0-status.md").exists(),
                (bundle / "apple-probe-simulation.json").exists(),
                (bundle / "apple-role-decision-simulated.md").exists(),
            )
        by_id = {item["id"]: item for item in report["deliverables"]}
        self.assertTrue(all(program_report_files))
        self.assertTrue(report["simulation"]["present"])
        self.assertEqual("closed", by_id["S0-1"]["status"])
        self.assertEqual("simulation_complete", by_id["S0-2"]["status"])
        self.assertEqual("simulation_complete", by_id["S0-7"]["status"])
        self.assertEqual("simulation_complete", by_id["S0-9"]["status"])
        self.assertIn(
            "Apple reversal trigger evaluated from rank-1 local probe",
            report["g1"]["machine_missing_criteria"],
        )
        self.assertIn("simulated Apple role=capacity-only", by_id["S0-7"]["evidence"])
        self.assertIn("R-10", report["markdown"])
        self.assertIn("simulation_complete", report["markdown"])

    def test_g1_evidence_packet_separates_machine_evidence_from_closure(self) -> None:
        csv_text = (
            "0, NVIDIA H100 80GB HBM3, 80000, 81559, 575.57.08\n"
            "1, NVIDIA H100 80GB HBM3, 79000, 81559, 575.57.08\n"
        )
        source = collect_local_inventory(nvidia_smi_csv=csv_text)
        inventory = build_logical_cluster_inventory(source)
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "bundle"
            run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                benchmark_iterations=1,
                inventory_data=inventory,
                include_g1_drafts=True,
                include_golden_plans=True,
                include_program_reports=True,
                program_report_date="2026-06-22",
                include_simulated_apple_evidence=True,
                simulated_apple_role="capacity-only",
                substrate_pinned_build="max-26.4.0",
                kickoff_date="2026-06-22",
                ker_status="unavailable",
                scope="pending",
            )
            packet = read_json(bundle / "g1-evidence-packet.json")
            markdown = (bundle / "g1-evidence-packet.md").read_text(encoding="utf-8")
        result = validate_g1_evidence_packet_fixture(packet)
        signoffs = {item["id"]: item for item in packet["signoff_requirements"]}
        evidence = {item["id"]: item for item in packet["evidence_items"]}
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(packet["summary"]["g1_gate_ready"])
        self.assertFalse(packet["summary"]["human_signoff_complete"])
        self.assertFalse(packet["summary"]["g2_g3_gate_evidence"])
        self.assertIn(
            "Apple reversal trigger evaluated from rank-1 local probe",
            packet["machine_missing_criteria"],
        )
        self.assertIn("target_contract_signoff", signoffs)
        self.assertFalse(signoffs["target_contract_signoff"]["present"])
        self.assertEqual("present", evidence["runtime_format_spec"]["status"])
        self.assertEqual("present", evidence["network_security_spec"]["status"])
        self.assertIn("not Sponsor approval", markdown)

    def test_g1_evidence_packet_validator_rejects_gate_ready_overclaim(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "bundle"
            run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                benchmark_iterations=1,
                inventory_data={
                    "nodes": [
                        {
                            "id": "fast",
                            "vendor": "nvidia",
                            "runtime": "max",
                            "mem_free_bytes": 16_000_000,
                            "compute_class": 4_000_000_000_000.0,
                            "mem_bandwidth_bytes_s": 400_000_000_000.0,
                            "supports_stage": True,
                            "supports_expert_worker": True,
                            "supports_kv": True,
                            "supported_dtypes": ["fp16"],
                        }
                    ],
                    "links": [],
                    "source": "test",
                },
                include_g1_drafts=True,
                include_golden_plans=True,
            )
            packet = build_g1_evidence_packet(
                bundle, packet_date="2026-06-22", plan_version="v3"
            )
        packet["summary"]["g1_gate_ready"] = True
        result = validate_g1_evidence_packet_fixture(packet)
        self.assertFalse(result["ok"])
        self.assertIn("g1_gate_ready cannot be true", "; ".join(result["errors"]))

    def test_program_simulate_phase0_builds_full_simulated_bundle(self) -> None:
        csv_text = (
            "0, NVIDIA H100 80GB HBM3, 80000, 81559, 575.57.08\n"
            "1, NVIDIA H100 80GB HBM3, 79000, 81559, 575.57.08\n"
        )
        source = collect_local_inventory(nvidia_smi_csv=csv_text)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source_path = root / "source-inventory.json"
            bundle = root / "bundle"
            write_json(source_path, source)

            result = run_phase0_simulated_validation(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                source_inventory_path=source_path,
                gpu_count=2,
                link_bandwidth_bytes_s=12.5e9,
                link_latency_s=0.0004,
                slow_node_factor=0.65,
                benchmark_iterations=1,
                program_report_date="2026-06-21",
                substrate_pinned_build="max-26.4.0",
                kickoff_date="2026-06-21",
                ker_status="unavailable",
                scope="pending",
            )

            by_id = {item["id"]: item for item in result["status"]["deliverables"]}
            required_files = (
                bundle / "source-inventory.json",
                bundle / "simulated-cluster-inventory.json",
                bundle / "inventory.json",
                bundle / "benchmark-ledger.jsonl",
                bundle / "g1-gate-review.md",
                bundle / "g1-evidence-packet.json",
                bundle / "g1-evidence-packet.md",
                bundle / "phase0-status.json",
                bundle / "phase0-status.md",
                bundle / "apple-probe-simulation.json",
                bundle / "apple-role-decision-simulated.md",
            )
            self.assertTrue(result["ok"], result["preflight"]["doctor"])
            self.assertTrue(all(path.exists() for path in required_files))
            self.assertIn("source_inventory", result["artifacts"])
            self.assertIn("simulated_cluster_inventory", result["artifacts"])
            self.assertIn("benchmark_ledger", result["artifacts"])
            ledger = validate_benchmark_ledger(bundle / "benchmark-ledger.jsonl")
            self.assertTrue(ledger["ok"], ledger["errors"])
            ledger_text = (bundle / "benchmark-ledger.jsonl").read_text(encoding="utf-8")
            self.assertIn("logical simulated cluster", ledger_text)
            self.assertIn("simulate-phase0", ledger_text)
            self.assertEqual(9, result["summary"]["total"])
            self.assertEqual(9, result["summary"]["machine_or_better"])
            self.assertTrue(result["simulation"]["present"])
            self.assertTrue(result["apple_simulation"]["present"])
            self.assertFalse(result["apple_simulation"]["gate_closable"])
            self.assertEqual("simulation_complete", by_id["S0-2"]["status"])
            self.assertEqual("simulation_complete", by_id["S0-7"]["status"])
            self.assertEqual("simulation_complete", by_id["S0-9"]["status"])
            self.assertEqual("ITERATE", result["g1"]["recommended_outcome"])
            self.assertIn(
                "Apple reversal trigger evaluated from rank-1 local probe",
                result["g1"]["machine_missing_criteria"],
            )

    def test_phase0_preflight_can_include_g1_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "bundle"
            fake_python = Path(d) / "fake-torch-python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "cat <<'JSON'\n"
                '{"measured": false, "backend": "torch", '
                '"available": false, "error": "fake torch unavailable"}\n'
                "JSON\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            result = run_phase0_preflight(
                target_path="fornax/golden_plans/v0_target_contract_fixture.md",
                out_dir=bundle,
                benchmark_iterations=1,
                include_g1_drafts=True,
                include_calibration=True,
                calibration_torch_python=str(fake_python),
                include_golden_plans=True,
                include_program_reports=True,
                program_report_date="2026-06-20",
                substrate_pinned_build="max-26.4.0",
                kickoff_date="2026-06-20",
                ker_status="unavailable",
                scope="pending",
                inventory_data={
                    "nodes": [
                        {
                            "id": "fast",
                            "vendor": "nvidia",
                            "runtime": "max",
                            "mem_free_bytes": 16_000_000,
                            "compute_class": 4_000_000_000_000.0,
                            "mem_bandwidth_bytes_s": 400_000_000_000.0,
                            "supports_stage": True,
                            "supports_expert_worker": True,
                            "supports_kv": True,
                            "supported_dtypes": ["fp16"],
                        }
                    ],
                    "links": [],
                    "source": "test",
                },
            )
            doctor = inspect_phase0_bundle(bundle)
            self.assertTrue(result["ok"], result["doctor"])
            self.assertTrue((bundle / "runtime-format-and-invariants.md").exists())
            self.assertTrue((bundle / "networking-security-and-backpressure.md").exists())
            self.assertTrue((bundle / "adr" / "0001-max-mojo-substrate.md").exists())
            self.assertTrue((bundle / "apple-expert-mlp-probe.json").exists())
            self.assertTrue((bundle / "roadmap-staffing-rebaseline.md").exists())
            self.assertTrue((bundle / "calibration.json").exists())
            self.assertTrue((bundle / "golden-plans.json").exists())
            self.assertTrue((bundle / "benchmark-ledger.jsonl").exists())
            self.assertTrue((bundle / "g1-gate-review.md").exists())
            self.assertTrue((bundle / "g1-evidence-packet.json").exists())
            self.assertTrue((bundle / "g1-evidence-packet.md").exists())
            self.assertTrue((bundle / "phase0-status.json").exists())
            self.assertTrue(doctor["artifacts"]["calibration.json"]["measured"])
            self.assertTrue(doctor["artifacts"]["benchmark_ledger"]["valid"])
            warnings = doctor["warnings"]
            self.assertNotIn("missing G1 gate artifact: runtime_format_spec", warnings)
            self.assertNotIn("missing G1 gate artifact: network_security_spec", warnings)
            self.assertNotIn("missing G1 gate artifact: substrate_adr", warnings)
            self.assertNotIn("missing G1 gate artifact: apple_probe", warnings)
            self.assertNotIn("missing G1 gate artifact: program_rebaseline", warnings)
            self.assertIn("missing G1 gate artifact: apple_probe_validation", warnings)

    def test_markdown_target_contract_fixture_loads(self) -> None:
        model, target, bundle = load_target_contract(
            "fornax/golden_plans/v0_target_contract_fixture.md"
        )
        self.assertEqual(2, model.num_layers)
        self.assertEqual(4, target.concurrency)
        self.assertIn("model", bundle)
        self.assertIn("target", bundle)

    def test_markdown_target_contract_requires_machine_readable_block(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad-contract.md"
            path.write_text("# Missing machine block\n", encoding="utf-8")
            with self.assertRaisesRegex(TargetContractError, "fornax-target"):
                load_target_contract(path)

    def test_golden_plan_fixtures(self) -> None:
        results = run_golden_plans()
        self.assertTrue(results)
        failures = [result for result in results if not result.passed]
        self.assertEqual([], failures)

    def test_slow_node_gets_fewer_layers(self) -> None:
        plan = plan_placement(
            dense_model(4),
            inventory_with_link(),
            Target(concurrency=4, prompt_len=16, gen_len=8, objective="balanced"),
            min_stages=2,
            max_stages=2,
        )
        self.assertTrue(plan.feasible, plan.infeasible_reason)
        self.assertEqual((0, 1, 2), plan.stages[0].layers)
        self.assertEqual((3,), plan.stages[1].layers)

    def test_model_too_big_is_infeasible_with_reason(self) -> None:
        model = dense_model(2)
        tiny = Inventory.from_dict(
            {
                "nodes": [
                    {
                        "id": "tiny",
                        "vendor": "cpu",
                        "runtime": "custom",
                        "mem_free_bytes": 1000,
                        "compute_class": 1e12,
                        "mem_bandwidth_bytes_s": 1e11,
                        "supported_dtypes": ["fp16"],
                    }
                ]
            }
        )
        plan = plan_placement(model, tiny, Target(4, 16, 8))
        self.assertFalse(plan.feasible)
        self.assertIn("no feasible contiguous stage placement", plan.infeasible_reason or "")
        self.assertEqual("excluded", plan.to_dict()["explanations"][0]["decision"])
        self.assertIn("insufficient memory", plan.to_dict()["explanations"][0]["reason"])

    def test_planner_records_selected_demoted_and_excluded_explanations(self) -> None:
        inventory = Inventory.from_dict(
            {
                "nodes": [
                    {
                        "id": "fast",
                        "vendor": "nvidia",
                        "runtime": "max",
                        "mem_free_bytes": 16_000_000,
                        "compute_class": 4_000_000_000_000.0,
                        "mem_bandwidth_bytes_s": 400_000_000_000.0,
                        "supported_dtypes": ["fp16"],
                    },
                    {
                        "id": "slow",
                        "vendor": "cpu",
                        "runtime": "custom",
                        "mem_free_bytes": 16_000_000,
                        "compute_class": 1_000_000_000_000.0,
                        "mem_bandwidth_bytes_s": 100_000_000_000.0,
                        "supported_dtypes": ["fp16"],
                    },
                    {
                        "id": "bad_dtype",
                        "vendor": "cpu",
                        "runtime": "custom",
                        "mem_free_bytes": 16_000_000,
                        "compute_class": 1_000_000_000_000.0,
                        "mem_bandwidth_bytes_s": 100_000_000_000.0,
                        "supported_dtypes": ["bf16"],
                    },
                    {
                        "id": "disabled",
                        "vendor": "cpu",
                        "runtime": "custom",
                        "mem_free_bytes": 16_000_000,
                        "compute_class": 1_000_000_000_000.0,
                        "mem_bandwidth_bytes_s": 100_000_000_000.0,
                        "supports_stage": False,
                        "supported_dtypes": ["fp16"],
                    },
                ],
                "links": [
                    {
                        "a": "fast",
                        "b": "slow",
                        "bandwidth_bytes_s": 12_500_000_000.0,
                        "latency_s": 0.00002,
                    }
                ],
            }
        )
        plan = plan_placement(
            dense_model(4),
            inventory,
            Target(4, 16, 8, "balanced"),
            min_stages=2,
            max_stages=2,
        )
        self.assertTrue(plan.feasible, plan.infeasible_reason)
        explanations = plan.to_dict()["explanations"]
        by_node_decision = {(row["node_id"], row["decision"]): row for row in explanations}
        self.assertIn(("fast", "selected"), by_node_decision)
        self.assertIn(("slow", "selected"), by_node_decision)
        self.assertIn(("slow", "demoted"), by_node_decision)
        self.assertIn("slower", by_node_decision[("slow", "demoted")]["reason"])
        self.assertIn(("bad_dtype", "excluded"), by_node_decision)
        self.assertIn("activation dtype", by_node_decision[("bad_dtype", "excluded")]["reason"])
        self.assertIn(("disabled", "excluded"), by_node_decision)
        self.assertIn("not stage-capable", by_node_decision[("disabled", "excluded")]["reason"])

    def test_adding_stage_capable_node_does_not_reduce_throughput(self) -> None:
        model = dense_model(4)
        target = Target(4, 16, 8, "max_throughput")
        one = Inventory.from_dict(
            {
                "nodes": [
                    {
                        "id": "fast",
                        "vendor": "nvidia",
                        "runtime": "max",
                        "mem_free_bytes": 16_000_000,
                        "compute_class": 4e12,
                        "mem_bandwidth_bytes_s": 4e11,
                        "supported_dtypes": ["fp16"],
                    }
                ]
            }
        )
        two = inventory_with_link()
        p1 = plan_placement(model, one, target)
        p2 = plan_placement(model, two, target)
        self.assertTrue(p1.feasible)
        self.assertTrue(p2.feasible)
        assert p1.predicted and p2.predicted
        self.assertGreaterEqual(p2.predicted.throughput_tok_s, p1.predicted.throughput_tok_s)

    def test_faster_link_does_not_raise_latency_for_forced_two_stage_plan(self) -> None:
        model = dense_model(4)
        target = Target(4, 16, 8, "balanced")
        slow_link = plan_placement(
            model,
            inventory_with_link(1_250_000_000.0),
            target,
            min_stages=2,
            max_stages=2,
        )
        fast_link = plan_placement(
            model,
            inventory_with_link(12_500_000_000.0),
            target,
            min_stages=2,
            max_stages=2,
        )
        self.assertTrue(slow_link.feasible)
        self.assertTrue(fast_link.feasible)
        assert slow_link.predicted and fast_link.predicted
        self.assertLessEqual(
            fast_link.predicted.per_request_latency_s,
            slow_link.predicted.per_request_latency_s,
        )


if __name__ == "__main__":
    unittest.main()
