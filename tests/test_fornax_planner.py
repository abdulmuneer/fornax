from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fornax.contracts import TargetContractError, load_target_contract
from fornax.golden import run_golden_plans
from fornax.inventory.local import collect_local_inventory, parse_nvidia_smi_csv
from fornax.planner import Inventory, ModelSpec, Target, plan_placement
from fornax.simulate import simulation_result, summarize_request_trace


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
