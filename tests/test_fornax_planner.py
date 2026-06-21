from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fornax.apple_probe import (
    apple_probe_template,
    render_apple_role_decision_draft,
    validate_apple_probe_artifact,
)
from fornax.benchmark import benchmark_from_plan, run_tiny_expert_mlp_benchmark
from fornax.calibration import run_cpu_memory_copy_probe, run_local_calibration
from fornax.contracts import TargetContractError, load_target_contract
from fornax.doctor import inspect_phase0_bundle
from fornax.golden import run_golden_plans
from fornax.g1_review import render_g1_gate_review_draft
from fornax.io import load_inventory, write_json
from fornax.inventory import build_logical_cluster_inventory
from fornax.inventory.local import (
    collect_local_inventory,
    parse_nvidia_smi_csv,
    probe_declared_links,
)
from fornax.network_security_spec import render_network_security_spec_draft
from fornax.network_contract import (
    validate_network_contract,
    validate_network_contract_fixture,
)
from fornax.planner import Inventory, ModelSpec, Target, plan_placement
from fornax.phase0_status import render_phase0_status_report
from fornax.preflight import run_phase0_preflight
from fornax.program_rebaseline import render_program_rebaseline_draft
from fornax.runtime_format_spec import render_runtime_format_spec_draft
from fornax.runtime_format import (
    validate_runtime_format_golden,
    validate_runtime_format_manifest,
)
from fornax.simulate import simulation_result, summarize_request_trace
from fornax.substrate_adr import render_substrate_adr_draft
from fornax.target_contract import render_target_contract_draft
from fornax.validation import validate_target_contract


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

    def test_benchmark_from_plan_rejects_infeasible_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "infeasible plan"):
            benchmark_from_plan({"feasible": False, "infeasible_reason": "no fit"})

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
        self.assertIn("```json fornax-target", result["markdown"])
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
            self.assertIn("golden_plans", result["artifacts"])
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
            )
        by_id = {item["id"]: item for item in report["deliverables"]}
        self.assertTrue(all(program_report_files))
        self.assertTrue(report["simulation"]["present"])
        self.assertEqual("closed", by_id["S0-1"]["status"])
        self.assertEqual("simulation_complete", by_id["S0-2"]["status"])
        self.assertEqual("incomplete", by_id["S0-7"]["status"])
        self.assertEqual("simulation_complete", by_id["S0-9"]["status"])
        self.assertIn(
            "Apple reversal trigger evaluated from rank-1 local probe",
            report["g1"]["machine_missing_criteria"],
        )
        self.assertIn("R-10", report["markdown"])
        self.assertIn("simulation_complete", report["markdown"])

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
            self.assertTrue((bundle / "g1-gate-review.md").exists())
            self.assertTrue((bundle / "phase0-status.json").exists())
            self.assertTrue(doctor["artifacts"]["calibration.json"]["measured"])
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
