from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fornax.max_generation_smoke import (
    CLAIM_KEYS,
    LIVE_RUNNER_KIND,
    MAX_GENERATED_TEXT_CHARS,
    MAX_OUTPUT_EXCERPT_CHARS,
    SMOKE_SENTINEL,
    SMOKE_SENTINEL_PROMPT,
    SYNTHETIC_RUNNER_KIND,
    run_max_generation_smoke,
    validate_max_generation_smoke_evidence,
)


class _FixtureRunner:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(self, argv: Sequence[str], timeout_s: float) -> Any:
        self.calls.append((tuple(argv), timeout_s))
        if not self.results:
            raise AssertionError("unexpected fixture command")
        return self.results.pop(0)


def _canonical_digest(report: dict[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "integrity"}
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class MaxGenerationSmokeTest(unittest.TestCase):
    def _run(
        self,
        model_dir: Path,
        *,
        runner: _FixtureRunner | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], _FixtureRunner]:
        prompt = kwargs.pop("prompt", "Define MoE.")
        selected_runner = runner or _FixtureRunner(
            [
                (0, "MAX 26.7.0\n", ""),
                (
                    0,
                    "architecture: Qwen3MoeForCausalLM\n"
                    "Generated text: Mixture of Experts routes tokens.\n"
                    "Output size: 8\n",
                    "",
                ),
            ]
        )
        report = run_max_generation_smoke(
            model_id="Qwen/Qwen3-30B-A3B",
            model_dir=model_dir,
            quantization_encoding="bfloat16",
            prompt=prompt,
            command_runner=selected_runner,
            **kwargs,
        )
        return report, selected_runner

    def test_injected_success_is_shell_free_but_never_a_physical_claim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report, runner = self._run(Path(td))
            resolved = str(Path(td).resolve())

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            [
                (("max", "--version"), 30.0),
                (
                    (
                        "max",
                        "generate",
                        "--model",
                        resolved,
                        "--devices",
                        "gpu:0",
                        "--quantization-encoding",
                        "bfloat16",
                        "--max-new-tokens",
                        "8",
                        "--top-k",
                        "1",
                        "--temperature",
                        "0",
                        "--prompt",
                        "Define MoE.",
                    ),
                    1800.0,
                ),
            ],
            runner.calls,
        )
        self.assertEqual(SYNTHETIC_RUNNER_KIND, report["runner"]["kind"])
        self.assertFalse(report["runner"]["physical_execution_eligible"])
        self.assertFalse(report["claims"]["single_platform_bringup_passed"])
        self.assertEqual(set(CLAIM_KEYS), set(report["claims"]))
        self.assertFalse(any(report["claims"].values()))
        self.assertFalse(report["commands"]["version"]["shell"])
        self.assertFalse(report["commands"]["generation"]["shell"])
        self.assertEqual(
            "explicit_generated_text_marker",
            report["observed"]["generated_text_signal"]["method"],
        )
        json.dumps(report, allow_nan=False)
        validation = validate_max_generation_smoke_evidence(report)
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(
            SYNTHETIC_RUNNER_KIND,
            validation["summary"]["runner_kind"],
        )
        self.assertFalse(validation["summary"]["single_platform_bringup_passed"])

    def test_default_subprocess_path_is_the_only_physical_claim_eligible_runner(
        self,
    ) -> None:
        runner = _FixtureRunner(
            [
                (0, "MAX 26.7.0\n", ""),
                (0, "Generated text: A live-path answer.\n", ""),
            ]
        )
        with tempfile.TemporaryDirectory() as td, patch(
            "fornax.max_generation_smoke._default_command_runner",
            new=runner,
        ):
            report = run_max_generation_smoke(
                model_id="Qwen/Qwen3-30B-A3B",
                model_dir=td,
                quantization_encoding="bfloat16",
                prompt="Define MoE.",
            )

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(LIVE_RUNNER_KIND, report["runner"]["kind"])
        self.assertTrue(report["runner"]["physical_execution_eligible"])
        self.assertTrue(report["claims"]["single_platform_bringup_passed"])
        validation = validate_max_generation_smoke_evidence(report)
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertTrue(validation["summary"]["single_platform_bringup_passed"])

    def test_live_nvidia_uuid_binding_masks_physical_gpu_as_max_gpu_zero(
        self,
    ) -> None:
        gpu_uuid = "GPU-12345678-1234-5678-9abc-def012345678"
        calls: list[tuple[tuple[str, ...], float, str | None]] = []

        def runner(
            argv: Sequence[str],
            timeout_s: float,
            *,
            env: dict[str, str] | None = None,
        ) -> Any:
            calls.append(
                (
                    tuple(argv),
                    timeout_s,
                    env.get("CUDA_VISIBLE_DEVICES") if env is not None else None,
                )
            )
            if argv[-1] == "--version":
                return 0, "MAX 26.7\n", ""
            return 0, "Generated text: A UUID-bound answer.\n", ""

        with tempfile.TemporaryDirectory() as td, patch(
            "fornax.max_generation_smoke._default_command_runner",
            new=runner,
        ):
            report = run_max_generation_smoke(
                model_id="Qwen/Qwen3-30B-A3B",
                model_dir=td,
                quantization_encoding="bfloat16",
                device="gpu:0",
                prompt="Define MoE.",
                nvidia_smi_index=7,
                nvidia_gpu_uuid=gpu_uuid,
            )

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["claims"]["single_platform_bringup_passed"])
        self.assertEqual([gpu_uuid, gpu_uuid], [call[2] for call in calls])
        generation_argv = calls[1][0]
        self.assertEqual(
            "gpu:0",
            generation_argv[generation_argv.index("--devices") + 1],
        )
        self.assertEqual(
            {
                "mode": "nvidia_gpu_uuid_to_visible_ordinal",
                "physical_nvidia_smi_index": 7,
                "physical_nvidia_gpu_uuid": gpu_uuid,
                "cuda_visible_devices": gpu_uuid,
                "max_device": "gpu:0",
                "applied_to_live_subprocess": True,
            },
            report["runner"]["device_binding"],
        )
        validation = validate_max_generation_smoke_evidence(report)
        self.assertTrue(validation["ok"], validation["errors"])

    def test_nvidia_uuid_binding_rejects_partial_or_nonzero_visible_ordinal(
        self,
    ) -> None:
        gpu_uuid = "GPU-12345678-1234-5678-9abc-def012345678"
        cases = (
            {"nvidia_smi_index": 0},
            {"nvidia_gpu_uuid": gpu_uuid},
            {
                "nvidia_smi_index": 0,
                "nvidia_gpu_uuid": "not-a-gpu-uuid",
            },
            {
                "device": "gpu:7",
                "nvidia_smi_index": 7,
                "nvidia_gpu_uuid": gpu_uuid,
            },
        )
        with tempfile.TemporaryDirectory() as td:
            for kwargs in cases:
                runner = _FixtureRunner([])
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        run_max_generation_smoke(
                            model_id="Qwen/Qwen3-30B-A3B",
                            model_dir=td,
                            quantization_encoding="bfloat16",
                            device=kwargs.pop("device", "gpu:0"),
                            prompt="Define MoE.",
                            command_runner=runner,
                            **kwargs,
                        )
                    self.assertEqual([], runner.calls)

    def test_multi_part_prefix_device_encoding_and_timeout_are_exact(self) -> None:
        runner = _FixtureRunner(
            [
                (0, "MAX custom\n", ""),
                (
                    0,
                    json.dumps(
                        {
                            "choices": [
                                {"message": {"content": "A generated answer."}}
                            ]
                        }
                    ),
                    "",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            report, runner = self._run(
                Path(td),
                runner=runner,
                device="gpu:7",
                max_argv_prefix=("pixi", "run", "max"),
                timeout_s=12.5,
                max_new_tokens=32,
                top_k=4,
            )

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            ("pixi", "run", "max", "--version"), runner.calls[0][0]
        )
        generation = runner.calls[1][0]
        self.assertEqual(("pixi", "run", "max", "generate"), generation[:4])
        self.assertEqual("gpu:7", generation[generation.index("--devices") + 1])
        self.assertEqual(
            "bfloat16",
            generation[generation.index("--quantization-encoding") + 1],
        )
        self.assertEqual(12.5, runner.calls[0][1])
        self.assertEqual(12.5, runner.calls[1][1])
        self.assertEqual(
            "json_choices_message_content",
            report["observed"]["generated_text_signal"]["method"],
        )

    def test_refuses_missing_and_non_directory_model_paths_before_execution(
        self,
    ) -> None:
        runner = _FixtureRunner([])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file_path = root / "weights"
            file_path.write_text("not a directory", encoding="utf-8")
            for path in (root / "missing", file_path):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(
                        ValueError, "model_dir must be an existing directory"
                    ):
                        run_max_generation_smoke(
                            model_id="Qwen/Qwen3-30B-A3B",
                            model_dir=path,
                            quantization_encoding="bfloat16",
                            prompt="Define MoE.",
                            command_runner=runner,
                        )
        self.assertEqual([], runner.calls)

    def test_invalid_inputs_fail_before_execution(self) -> None:
        cases = (
            {"model_id": "", "quantization_encoding": "bfloat16", "prompt": "x"},
            {
                "model_id": "model",
                "device": "gpu",
                "quantization_encoding": "bfloat16",
                "prompt": "x",
            },
            {
                "model_id": "model",
                "device": "gpu:0,gpu:1",
                "quantization_encoding": "bfloat16",
                "prompt": "x",
            },
            {
                "model_id": "model",
                "quantization_encoding": "bad encoding",
                "prompt": "x",
            },
            {
                "model_id": "model",
                "quantization_encoding": "bfloat16",
                "prompt": " ",
            },
            {
                "model_id": "model",
                "quantization_encoding": "bfloat16",
                "prompt": "x",
                "max_new_tokens": 0,
            },
            {
                "model_id": "model",
                "quantization_encoding": "bfloat16",
                "prompt": "x",
                "top_k": True,
            },
            {
                "model_id": "model",
                "quantization_encoding": "bfloat16",
                "prompt": "x",
                "timeout_s": math.inf,
            },
            {
                "model_id": "model",
                "quantization_encoding": "bfloat16",
                "prompt": "x",
                "max_argv_prefix": "max",
            },
        )
        with tempfile.TemporaryDirectory() as td:
            for kwargs in cases:
                runner = _FixtureRunner([])
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        run_max_generation_smoke(
                            model_dir=td,
                            command_runner=runner,
                            **kwargs,
                        )
                    self.assertEqual([], runner.calls)

    def test_plain_or_diagnostic_stdout_is_not_generated_text(self) -> None:
        outputs = (
            "hello without an explicit signal\n",
            "architecture: Qwen3MoeForCausalLM\nOutput size: 8\n",
            "Prompt: Define MoE.\nOutput size: 8\n",
            "Output size: 8\nCompilation complete\n",
            "Generated text:\nCompilation complete\n",
            "Generated text: Compilation complete\n",
            "Generated text: Define MoE.\n",
        )
        with tempfile.TemporaryDirectory() as td:
            for stdout in outputs:
                with self.subTest(stdout=stdout):
                    runner = _FixtureRunner(
                        [(0, "MAX 26.7\n", ""), (0, stdout, "")]
                    )
                    report, _ = self._run(Path(td), runner=runner)
                    self.assertFalse(report["ok"])
                    self.assertFalse(
                        report["observed"]["generated_text_signal"]["detected"]
                    )
                    self.assertFalse(any(report["claims"].values()))
                    validation = validate_max_generation_smoke_evidence(report)
                    self.assertTrue(validation["ok"], validation["errors"])

    def test_real_max_metrics_format_requires_exact_sentinel_binding(self) -> None:
        accepted = (
            f"{SMOKE_SENTINEL}\n"
            "Prompt size: 10\n"
            "Output size: 7\n"
            "Time per output token: 0.01 s\n"
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
                "prefix log\n"
                f"{SMOKE_SENTINEL}\n"
                "Prompt size: 10\n"
                "Output size: 7\n"
            ),
            (
                f"{SMOKE_SENTINEL_PROMPT}\n"
                "Prompt size: 10\n"
                "Output size: 7\n"
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            runner = _FixtureRunner(
                [(0, "MAX 26.7\n", ""), (0, accepted, "")]
            )
            report, _ = self._run(
                Path(td),
                runner=runner,
                prompt=SMOKE_SENTINEL_PROMPT,
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(
                "max_metrics_exact_sentinel",
                report["observed"]["generated_text_signal"]["method"],
            )
            self.assertEqual(
                SMOKE_SENTINEL,
                report["observed"]["generated_text_signal"]["text_excerpt"],
            )

            for stdout in rejected:
                with self.subTest(stdout=stdout):
                    runner = _FixtureRunner(
                        [(0, "MAX 26.7\n", ""), (0, stdout, "")]
                    )
                    report, _ = self._run(
                        Path(td),
                        runner=runner,
                        prompt=SMOKE_SENTINEL_PROMPT,
                    )
                    self.assertFalse(report["ok"])
                    self.assertFalse(
                        report["observed"]["generated_text_signal"]["detected"]
                    )

            wrong_prompt_runner = _FixtureRunner(
                [(0, "MAX 26.7\n", ""), (0, accepted, "")]
            )
            wrong_prompt_report, _ = self._run(
                Path(td),
                runner=wrong_prompt_runner,
                prompt="Use any answer.",
            )
            self.assertFalse(wrong_prompt_report["ok"])

    def test_live_generation_output_overflow_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "fake_max.py"
            script.write_text(
                "import sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('MAX 26.7')\n"
                "else:\n"
                "    sys.stdout.write('x' * 1024)\n",
                encoding="utf-8",
            )
            with patch(
                "fornax.max_generation_smoke.MAX_GENERATION_STDOUT_BYTES",
                128,
            ):
                report = run_max_generation_smoke(
                    model_id="Qwen/Qwen3-30B-A3B",
                    model_dir=root,
                    quantization_encoding="bfloat16",
                    prompt=SMOKE_SENTINEL_PROMPT,
                    max_argv_prefix=(sys.executable, str(script)),
                    timeout_s=5.0,
                )

        self.assertFalse(report["ok"])
        self.assertIn(
            "SubprocessOutputLimitExceeded",
            str(report["commands"]["generation"]["launch_error"]),
        )
        self.assertFalse(any(report["claims"].values()))

    def test_nonzero_exit_launch_error_and_missing_version_fail_closed(
        self,
    ) -> None:
        cases = (
            _FixtureRunner(
                [(7, "", "version failure"), (0, "Generated text: x", "")]
            ),
            _FixtureRunner(
                [(0, "MAX 26.7", ""), (9, "", "generation failure")]
            ),
            _FixtureRunner(
                [(0, "", ""), (0, "Generated text: x", "")]
            ),
            _FixtureRunner(
                [
                    RuntimeError("MAX unavailable"),
                    (0, "Generated text: x", ""),
                ]
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            for runner in cases:
                with self.subTest(runner=runner):
                    report, _ = self._run(Path(td), runner=runner)
                    self.assertFalse(report["ok"])
                    self.assertFalse(any(report["claims"].values()))
                    validation = validate_max_generation_smoke_evidence(report)
                    self.assertTrue(validation["ok"], validation["errors"])

    def test_output_and_generated_text_excerpts_are_bounded(self) -> None:
        long_text = "x" * (MAX_GENERATED_TEXT_CHARS + 100)
        stdout = (
            "prefix\n"
            + ("diagnostic\n" * MAX_OUTPUT_EXCERPT_CHARS)
            + "Generated text: "
            + long_text
        )
        stderr = "e" * (MAX_OUTPUT_EXCERPT_CHARS + 50)
        runner = _FixtureRunner(
            [(0, "MAX 26.7", ""), (0, stdout, stderr)]
        )
        with tempfile.TemporaryDirectory() as td:
            report, _ = self._run(Path(td), runner=runner)

        command = report["commands"]["generation"]
        signal = report["observed"]["generated_text_signal"]
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            MAX_OUTPUT_EXCERPT_CHARS, len(command["stdout_excerpt"])
        )
        self.assertEqual(
            MAX_OUTPUT_EXCERPT_CHARS, len(command["stderr_excerpt"])
        )
        self.assertTrue(command["stdout_excerpt_truncated"])
        self.assertTrue(command["stderr_excerpt_truncated"])
        self.assertEqual(
            MAX_GENERATED_TEXT_CHARS, len(signal["text_excerpt"])
        )
        self.assertTrue(signal["text_excerpt_truncated"])
        validation = validate_max_generation_smoke_evidence(report)
        self.assertTrue(validation["ok"], validation["errors"])

    def test_integrity_and_semantic_tampering_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report, _ = self._run(Path(td))

        mutations = []
        changed_argv = copy.deepcopy(report)
        changed_argv["commands"]["generation"]["argv"][0] = "sh"
        mutations.append((changed_argv, "integrity digest"))

        changed_excerpt = copy.deepcopy(report)
        changed_excerpt["observed"]["generated_text_signal"][
            "text_excerpt"
        ] = "tampered"
        mutations.append((changed_excerpt, "integrity digest"))

        forbidden_claim = copy.deepcopy(report)
        forbidden_claim["claims"]["formal_g3_passed"] = True
        forbidden_claim["integrity"]["canonical_payload_sha256"] = (
            _canonical_digest(forbidden_claim)
        )
        mutations.append((forbidden_claim, "claims.formal_g3_passed"))

        false_success = copy.deepcopy(report)
        false_success["commands"]["generation"]["returncode"] = 9
        false_success["commands"]["generation"]["ok"] = False
        false_success["integrity"]["canonical_payload_sha256"] = (
            _canonical_digest(false_success)
        )
        mutations.append((false_success, "report.ok"))

        synthetic_physical_claim = copy.deepcopy(report)
        synthetic_physical_claim["claims"]["single_platform_bringup_passed"] = True
        synthetic_physical_claim["integrity"]["canonical_payload_sha256"] = (
            _canonical_digest(synthetic_physical_claim)
        )
        mutations.append(
            (synthetic_physical_claim, "single_platform_bringup_passed")
        )

        for tampered, expected in mutations:
            with self.subTest(expected=expected):
                validation = validate_max_generation_smoke_evidence(tampered)
                self.assertFalse(validation["ok"])
                self.assertIn(expected, "\n".join(validation["errors"]))
                self.assertFalse(
                    validation["summary"]["single_platform_bringup_passed"]
                )

    def test_validator_fails_closed_on_malformed_reports(self) -> None:
        self.assertFalse(
            validate_max_generation_smoke_evidence(None)["ok"]
        )
        with tempfile.TemporaryDirectory() as td:
            report, _ = self._run(Path(td))

        malformed = copy.deepcopy(report)
        malformed["commands"]["generation"]["elapsed_s"] = float("nan")
        validation = validate_max_generation_smoke_evidence(malformed)
        self.assertFalse(validation["ok"])
        self.assertIn("elapsed_s", "\n".join(validation["errors"]))

        extra_field = copy.deepcopy(report)
        extra_field["unexpected"] = True
        validation = validate_max_generation_smoke_evidence(extra_field)
        self.assertFalse(validation["ok"])
        self.assertIn(
            "fields must exactly match", "\n".join(validation["errors"])
        )

        invalid_unicode = copy.deepcopy(report)
        invalid_unicode["commands"]["generation"]["stdout_excerpt"] = "\ud800"
        validation = validate_max_generation_smoke_evidence(invalid_unicode)
        self.assertFalse(validation["ok"])
        self.assertIn("UTF-8", "\n".join(validation["errors"]))


if __name__ == "__main__":
    unittest.main()
