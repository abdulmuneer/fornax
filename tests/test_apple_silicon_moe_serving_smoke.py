from __future__ import annotations

import unittest

from fornax.apple_silicon_moe_serving_smoke import (
    validate_apple_silicon_moe_serving_smoke_fixture,
)


def apple_silicon_moe_smoke_fixture() -> dict:
    return {
        "version": 1,
        "record_kind": "apple-silicon-moe-serving-smoke",
        "evidence_scope": "single-mac-max-real-moe-serving-smoke",
        "ok": True,
        "error": None,
        "model": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "model_path": None,
            "model_family": "Qwen",
            "config_source": "/Users/test/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/sha/config.json",
            "architecture": "Qwen3MoeForCausalLM",
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "real_frontier_moe_model": True,
            "synthetic_fixture": False,
            "moe_config": {
                "num_hidden_layers": 48,
                "hidden_size": 2048,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 768,
                "max_position_embeddings": 40960,
            },
        },
        "runtime": {
            "backend": "max",
            "mode": "max-generate",
            "max_command": ["/tmp/max"],
            "max_cwd": None,
            "max_extra_args": [],
            "max_version": "MAX 26.5.0.dev2026062906",
            "max_version_error": None,
            "mojo_version": "Mojo 1.0.0b3.dev2026062906",
            "mojo_version_error": None,
            "devices_requested": "gpu",
            "quantization_encoding": None,
            "max_length": None,
            "allow_download": True,
            "fornax_orchestrated": True,
        },
        "serving": {
            "request": {
                "model": "Qwen/Qwen3-30B-A3B",
                "messages": [{"role": "user", "content": "Define MoE."}],
                "max_new_tokens": 8,
                "stream": False,
            },
            "response": {
                "id": "fornax-apple-silicon-moe-smoke",
                "object": "chat.completion",
                "model": "Qwen/Qwen3-30B-A3B",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "MoE means Mixture of Experts.",
                        },
                        "finish_reason": "length",
                    }
                ],
            },
            "generated_text": "MoE means Mixture of Experts.",
            "openai_compatible_shape": True,
            "live_http_endpoint": False,
        },
        "result": {
            "elapsed_s": 123.4,
            "returncode": 0,
            "stdout_tail": "MoE means Mixture of Experts.",
            "stderr_tail": "",
            "failure_signature": [],
        },
        "hardware": {
            "platform": "macOS-26.0-arm64-arm-64bit",
            "machine": "arm64",
            "processor": "arm",
            "model_name": "MacBook Pro",
            "model_identifier": "Mac15,8",
            "chip": "Apple M3 Max",
            "core_count": "16 (12 Performance and 4 Efficiency)",
            "memory": "128 GB",
        },
        "environment": {
            "python_executable": "/tmp/python",
            "python_version": "3.14.6",
            "platform": "macOS-26.0-arm64-arm-64bit",
            "machine": "arm64",
            "hf_home": "/Users/test/.cache/huggingface",
        },
        "claims": {
            "real_frontier_moe_model": True,
            "synthetic_fixture": False,
            "live_http_endpoint": False,
            "target_model_parity_reference": False,
            "formal_g2_passed": False,
            "formal_g3_passed": False,
            "g2_g3_gate_evidence": False,
            "production_distributed_serving": False,
        },
        "note": "fixture",
    }


class AppleSiliconMoeServingSmokeTest(unittest.TestCase):
    def test_validates_qwen_max_artifact(self) -> None:
        result = validate_apple_silicon_moe_serving_smoke_fixture(
            apple_silicon_moe_smoke_fixture()
        )
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual("Qwen/Qwen3-30B-A3B", result["summary"]["model_id"])
        self.assertEqual("Qwen", result["summary"]["model_family"])
        self.assertEqual(128, result["summary"]["num_experts"])
        self.assertEqual("Apple M3 Max", result["summary"]["chip"])
        self.assertFalse(result["summary"]["g2_g3_gate_evidence"])

    def test_rejects_dense_model(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["model"]["architecture"] = "Qwen3_5ForConditionalGeneration"
        artifact["model"]["model_type"] = "qwen3_5"
        artifact["model"]["moe_config"]["num_experts"] = None
        artifact["model"]["moe_config"]["num_experts_per_tok"] = None
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("MoE", text)
        self.assertIn("num_experts", text)

    def test_accepts_routed_expert_metadata(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["model"]["model_id"] = "moonshotai/Kimi-VL-A3B-Instruct"
        artifact["model"]["model_family"] = "Kimi"
        artifact["model"]["architecture"] = "KimiVLForConditionalGeneration"
        artifact["model"]["architectures"] = ["KimiVLForConditionalGeneration"]
        artifact["model"]["model_type"] = "kimi_vl"
        artifact["model"]["moe_config"]["num_experts"] = 64
        artifact["model"]["moe_config"]["n_routed_experts"] = 64
        artifact["model"]["moe_config"]["n_shared_experts"] = 2
        artifact["model"]["moe_config"]["num_experts_per_tok"] = 6
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual("Kimi", result["summary"]["model_family"])
        self.assertEqual(64, result["summary"]["num_experts"])

    def test_rejects_gate_overclaim(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["claims"]["formal_g3_passed"] = True
        artifact["claims"]["production_distributed_serving"] = True
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("formal_g3_passed", text)
        self.assertIn("production_distributed_serving", text)

    def test_rejects_bad_max_extra_args(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["runtime"]["max_extra_args"] = ["--force", ""]
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("runtime.max_extra_args", "; ".join(result["errors"]))

    def test_accepts_live_serve_artifact(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["runtime"]["mode"] = "max-serve"
        artifact["runtime"]["serve_port"] = 18080
        artifact["serving"]["live_http_endpoint"] = True
        artifact["serving"]["http_status"] = 200
        artifact["claims"]["live_http_endpoint"] = True
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual("max-serve", result["summary"]["runtime_mode"])
        self.assertTrue(result["summary"]["live_http_endpoint"])
        self.assertEqual(200, result["summary"]["http_status"])

    def test_rejects_serve_without_http(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["runtime"]["mode"] = "max-serve"
        artifact["runtime"]["serve_port"] = 18080
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("live_http_endpoint", text)
        self.assertIn("http_status", text)


if __name__ == "__main__":
    unittest.main()
