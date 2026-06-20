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
from fornax.contracts import TargetContractError, load_target_contract
from fornax.doctor import inspect_phase0_bundle
from fornax.golden import run_golden_plans
from fornax.io import load_inventory, write_json
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
        self.assertEqual([], result["warnings"])
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
