from __future__ import annotations

import hashlib
import json
import math
import unittest
from unittest.mock import MagicMock, patch

from fornax.apple_silicon_moe_serving_smoke import (
    MAX_HTTP_RESPONSE_BYTES,
    _generated_text_from_stdout,
    _post_chat_completion,
    _run_serve_request,
    run_apple_silicon_moe_serving_smoke,
    validate_apple_silicon_moe_serving_smoke_fixture,
)
from fornax.bounded_subprocess import SubprocessOutputLimitExceeded
from fornax.max_generation_smoke import (
    MAX_TIMEOUT_SECONDS,
    SMOKE_SENTINEL,
    SMOKE_SENTINEL_PROMPT,
)


def apple_silicon_moe_smoke_fixture() -> dict:
    stdout = "Generated text: MoE means Mixture of Experts.\n"
    stderr = ""
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
            "launch_argv": [
                "/tmp/max",
                "generate",
                "--model",
                "Qwen/Qwen3-30B-A3B",
                "--devices",
                "gpu",
                "--max-new-tokens",
                "8",
                "--top-k",
                "1",
                "--prompt",
                "Define MoE.",
            ],
            "max_cwd": None,
            "max_extra_args": [],
            "max_version": "MAX 26.5.0.dev2026062906",
            "max_version_error": None,
            "mojo_version": "Mojo 1.0.0b3.dev2026062906",
            "mojo_version_error": None,
            "devices_requested": "gpu",
            "quantization_encoding": None,
            "top_k": 1,
            "max_length": None,
            "serve_port": None,
            "served_model_name": None,
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
            "server_returncode": None,
            "server_terminated_by_probe": False,
            "preexisting_listener_detected": False,
            "server_ownership_verified": False,
            "stdout": stdout,
            "stdout_chars": len(stdout),
            "stdout_text_sha256": (
                "sha256:" + hashlib.sha256(stdout.encode()).hexdigest()
            ),
            "stderr": stderr,
            "stderr_chars": len(stderr),
            "stderr_text_sha256": (
                "sha256:" + hashlib.sha256(stderr.encode()).hexdigest()
            ),
            "stdout_tail": "MoE means Mixture of Experts.",
            "stderr_tail": "",
            "failure_signature": [],
        },
        "runner": {
            "kind": "live_subprocess",
            "physical_execution_eligible": True,
            "authenticated": False,
        },
        "hardware": {
            "platform": "macOS-26.0-arm64-arm-64bit",
            "machine": "arm64",
            "processor": "arm",
            "model_name": "MacBook Pro",
            "model_identifier": "Mac15,8",
            "model_number": "Z1CM00000",
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
    def test_rejects_unbounded_timeout_before_launching_commands(self) -> None:
        for timeout_s in (
            math.nan,
            math.inf,
            MAX_TIMEOUT_SECONDS + 1,
        ):
            with self.subTest(timeout_s=timeout_s), patch(
                "fornax.apple_silicon_moe_serving_smoke._run_text"
            ) as run_text:
                with self.assertRaisesRegex(ValueError, "timeout_s"):
                    run_apple_silicon_moe_serving_smoke(
                        out="unused.json",
                        timeout_s=timeout_s,
                    )
                run_text.assert_not_called()

    def test_generate_stdout_requires_explicit_or_json_framing(self) -> None:
        prompt = "Define MoE."
        rejected = (
            "Architecture: Qwen3MoeForCausalLM\nOutput size: 0\n",
            "Output size: 8\nCompilation complete\n",
            "Generated text:\nCompilation complete\n",
            "Generated text: Compilation complete\n",
            "Generated text: Define MoE.\n",
        )
        for stdout in rejected:
            with self.subTest(stdout=stdout):
                self.assertEqual(
                    "",
                    _generated_text_from_stdout(stdout, prompt=prompt),
                )
        self.assertEqual(
            "A framed answer.",
            _generated_text_from_stdout(
                "Generated text: A framed answer.\n",
                prompt=prompt,
            ),
        )
        self.assertEqual(
            "A JSON answer.",
            _generated_text_from_stdout(
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "A JSON answer."}}
                        ]
                    }
                ),
                prompt=prompt,
            ),
        )

    def test_generate_accepts_only_exact_metrics_framed_sentinel(self) -> None:
        accepted = (
            f"{SMOKE_SENTINEL}\n"
            "Prompt size: 10\n"
            "Output size: 7\n"
            "Time per output token: 0.01 s\n"
        )
        self.assertEqual(
            SMOKE_SENTINEL,
            _generated_text_from_stdout(
                accepted,
                prompt=SMOKE_SENTINEL_PROMPT,
            ),
        )
        rejected = (
            (
                "Compilation complete\n"
                "Prompt size: 10\n"
                "Output size: 7\n"
            ),
            (
                f"{SMOKE_SENTINEL}\n"
                "Prompt size: 10\n"
                "Output size: 0\n"
            ),
            (
                "diagnostic prefix\n"
                f"{SMOKE_SENTINEL}\n"
                "Prompt size: 10\n"
                "Output size: 7\n"
            ),
        )
        for stdout in rejected:
            with self.subTest(stdout=stdout):
                self.assertEqual(
                    "",
                    _generated_text_from_stdout(
                        stdout,
                        prompt=SMOKE_SENTINEL_PROMPT,
                    ),
                )
        self.assertEqual(
            "",
            _generated_text_from_stdout(
                accepted,
                prompt="Use any answer.",
            ),
        )

    def test_http_response_read_is_hard_bounded(self) -> None:
        response = MagicMock()
        response.status = 200
        response.read.return_value = b"x" * (MAX_HTTP_RESPONSE_BYTES + 1)
        context = MagicMock()
        context.__enter__.return_value = response
        with patch(
            "fornax.apple_silicon_moe_serving_smoke.urllib.request.urlopen",
            return_value=context,
        ):
            status, payload, error = _post_chat_completion(
                port=18080,
                model="fornax-smoke-test",
                prompt=SMOKE_SENTINEL_PROMPT,
                max_new_tokens=8,
                timeout_s=1.0,
            )

        response.read.assert_called_once_with(MAX_HTTP_RESPONSE_BYTES + 1)
        self.assertEqual(200, status)
        self.assertIsNone(payload)
        self.assertIn("bounded", str(error))

    def test_serve_refuses_a_preexisting_listener_without_spawning(self) -> None:
        with patch(
            "fornax.apple_silicon_moe_serving_smoke._listener_present",
            return_value=True,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke.start_bounded_subprocess"
        ) as start:
            result = _run_serve_request(
                command=["max", "serve"],
                env={},
                cwd=None,
                port=18080,
                served_model_name="fornax-smoke-" + "a" * 32,
                prompt="Define MoE.",
                max_new_tokens=8,
                timeout_s=1.0,
            )

        start.assert_not_called()
        self.assertFalse(result["server_ownership_verified"])
        self.assertTrue(result["preexisting_listener_detected"])
        self.assertIn("already owns", result["http_error"])

    def test_serve_rejects_http_output_after_spawned_process_exits(self) -> None:
        process = MagicMock()
        process.poll.side_effect = [None, 1, 1]
        process.returncode = 1
        process.communicate.return_value = ("", "address already in use")
        response = {
            "choices": [
                {"message": {"content": "Response from a stale server."}}
            ]
        }
        with patch(
            "fornax.apple_silicon_moe_serving_smoke._listener_present",
            return_value=False,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke.start_bounded_subprocess",
            return_value=process,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke._post_chat_completion",
            return_value=(200, response, None),
        ):
            result = _run_serve_request(
                command=["max", "serve"],
                env={},
                cwd=None,
                port=18080,
                served_model_name="fornax-smoke-" + "b" * 32,
                prompt="Define MoE.",
                max_new_tokens=8,
                timeout_s=1.0,
            )

        self.assertFalse(result["server_ownership_verified"])
        self.assertEqual("", result["generated_text"])
        self.assertIn("stale-listener", result["http_error"])

    def test_serve_rejects_response_for_a_different_model(self) -> None:
        process = MagicMock()
        process.poll.side_effect = [None, None]
        process.returncode = -15
        process.communicate.return_value = ("", "")
        response = {
            "model": "some-preexisting-model",
            "choices": [
                {"message": {"content": "Response from a stale server."}}
            ],
        }
        with patch(
            "fornax.apple_silicon_moe_serving_smoke._listener_present",
            return_value=False,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke.start_bounded_subprocess",
            return_value=process,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke._post_chat_completion",
            return_value=(200, response, None),
        ):
            result = _run_serve_request(
                command=["max", "serve"],
                env={},
                cwd=None,
                port=18080,
                served_model_name="fornax-smoke-" + "c" * 32,
                prompt="Define MoE.",
                max_new_tokens=8,
                timeout_s=1.0,
            )

        process.terminate.assert_called_once_with()
        self.assertFalse(result["server_ownership_verified"])
        self.assertEqual("", result["generated_text"])
        self.assertIn("unique model name", result["http_error"])

    def test_serve_rejects_prompt_echo_as_generated_text(self) -> None:
        process = MagicMock()
        process.poll.side_effect = [None, None]
        process.returncode = -15
        process.communicate.return_value = ("", "")
        served_model_name = "fornax-smoke-" + "d" * 32
        response = {
            "model": served_model_name,
            "choices": [{"message": {"content": "Define MoE."}}],
        }
        with patch(
            "fornax.apple_silicon_moe_serving_smoke._listener_present",
            return_value=False,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke.start_bounded_subprocess",
            return_value=process,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke._post_chat_completion",
            return_value=(200, response, None),
        ):
            result = _run_serve_request(
                command=["max", "serve"],
                env={},
                cwd=None,
                port=18080,
                served_model_name=served_model_name,
                prompt="Define MoE.",
                max_new_tokens=8,
                timeout_s=1.0,
            )

        process.terminate.assert_called_once_with()
        self.assertFalse(result["server_ownership_verified"])
        self.assertEqual("", result["generated_text"])
        self.assertIn("echoed the prompt", result["http_error"])

    def test_serve_output_overflow_discards_http_success(self) -> None:
        process = MagicMock()
        process.poll.side_effect = [None, None, None]
        process.returncode = -9
        process.communicate.side_effect = SubprocessOutputLimitExceeded(
            cmd=("max", "serve"),
            streams=("stdout",),
            stdout="x" * 128,
            stderr="",
            stdout_limit_bytes=128,
            stderr_limit_bytes=128,
            returncode=-9,
        )
        served_model_name = "fornax-smoke-" + "e" * 32
        response = {
            "model": served_model_name,
            "choices": [{"message": {"content": SMOKE_SENTINEL}}],
        }
        with patch(
            "fornax.apple_silicon_moe_serving_smoke._listener_present",
            return_value=False,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke.start_bounded_subprocess",
            return_value=process,
        ), patch(
            "fornax.apple_silicon_moe_serving_smoke._post_chat_completion",
            return_value=(200, response, None),
        ):
            result = _run_serve_request(
                command=["max", "serve"],
                env={},
                cwd=None,
                port=18080,
                served_model_name=served_model_name,
                prompt=SMOKE_SENTINEL_PROMPT,
                max_new_tokens=8,
                timeout_s=1.0,
            )

        self.assertFalse(result["server_ownership_verified"])
        self.assertEqual("", result["generated_text"])
        self.assertIn("bounded capture", result["http_error"])

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

    def test_rejects_synthetic_runner_and_complete_output_tampering(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["runner"] = {
            "kind": "synthetic_injected_test_runner",
            "physical_execution_eligible": False,
            "authenticated": False,
        }
        artifact["result"]["stdout"] += "tampered"
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("live subprocess", text)
        self.assertIn("stdout_chars", text)
        self.assertIn("stdout_text_sha256", text)

    def test_lone_surrogate_output_fails_validation_without_raising(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["result"]["stdout"] = "\ud800"
        artifact["result"]["stdout_chars"] = 1
        artifact["result"]["stdout_text_sha256"] = "sha256:" + "0" * 64
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertFalse(result["ok"])
        self.assertIn("valid UTF-8", "; ".join(result["errors"]))

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

    def test_accepts_gpt_oss_moe_metadata(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["model"]["model_id"] = "openai/gpt-oss-120b"
        artifact["model"]["model_family"] = "GPT-OSS"
        artifact["model"]["architecture"] = "GptOssForCausalLM"
        artifact["model"]["architectures"] = ["GptOssForCausalLM"]
        artifact["model"]["model_type"] = "gpt_oss"
        artifact["model"]["moe_config"]["num_hidden_layers"] = 36
        artifact["model"]["moe_config"]["hidden_size"] = 2880
        artifact["model"]["moe_config"]["num_experts"] = 128
        artifact["model"]["moe_config"]["num_experts_per_tok"] = 4
        artifact["model"]["moe_config"]["max_position_embeddings"] = 131072
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual("GPT-OSS", result["summary"]["model_family"])
        self.assertEqual(128, result["summary"]["num_experts"])

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
        artifact["runtime"]["served_model_name"] = "fornax-smoke-" + "a" * 32
        artifact["serving"]["request"]["model"] = artifact["runtime"][
            "served_model_name"
        ]
        artifact["serving"]["response"]["model"] = artifact["runtime"][
            "served_model_name"
        ]
        artifact["serving"]["live_http_endpoint"] = True
        artifact["serving"]["http_status"] = 200
        artifact["result"]["server_ownership_verified"] = True
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
        artifact["runtime"]["served_model_name"] = "fornax-smoke-" + "a" * 32
        artifact["serving"]["request"]["model"] = artifact["runtime"][
            "served_model_name"
        ]
        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)
        self.assertFalse(result["ok"])
        text = "; ".join(result["errors"])
        self.assertIn("live_http_endpoint", text)
        self.assertIn("http_status", text)

    def test_rejects_serve_response_for_a_different_model(self) -> None:
        artifact = apple_silicon_moe_smoke_fixture()
        artifact["runtime"]["mode"] = "max-serve"
        artifact["runtime"]["serve_port"] = 18080
        artifact["runtime"]["served_model_name"] = "fornax-smoke-" + "a" * 32
        artifact["serving"]["request"]["model"] = artifact["runtime"][
            "served_model_name"
        ]
        artifact["serving"]["live_http_endpoint"] = True
        artifact["serving"]["http_status"] = 200
        artifact["result"]["server_ownership_verified"] = True
        artifact["claims"]["live_http_endpoint"] = True

        result = validate_apple_silicon_moe_serving_smoke_fixture(artifact)

        self.assertFalse(result["ok"])
        self.assertIn("serving.response.model", "; ".join(result["errors"]))


if __name__ == "__main__":
    unittest.main()
