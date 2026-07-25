from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fornax.max_runtime_probe import (
    MAX_DIAGNOSTIC_CHARS,
    MAX_LIST_OUTPUT_BYTES,
    PHYSICAL_CLAIM_KEYS,
    probe_max_runtime_support,
)


class _FixtureRunner:
    def __init__(self, results: dict[tuple[str, ...], Any]) -> None:
        self.results = results
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> Any:
        command = tuple(argv)
        self.calls.append(command)
        return self.results.get(command, (127, "", "unexpected fixture command"))


def _registry(
    *,
    architecture: str = "Qwen3MoeForCausalLM",
    encodings: list[Any] | None = None,
) -> str:
    return json.dumps(
        {
            "architectures": {
                architecture: {
                    "example_repo_ids": ["Qwen/Qwen3-30B-A3B"],
                    "supported_encodings": (
                        ["bfloat16", "float8_e4m3fn"]
                        if encodings is None
                        else encodings
                    ),
                    "multi_gpu_supported": True,
                }
            }
        }
    )


def _runner(
    *,
    version: Any = (0, "MAX 26.5.0\n", ""),
    listing: Any | None = None,
    command: tuple[str, ...] = ("max",),
) -> _FixtureRunner:
    return _FixtureRunner(
        {
            (*command, "--version"): version,
            (*command, "list", "--json"): (
                (0, _registry(), "") if listing is None else listing
            ),
        }
    )


class MaxRuntimeProbeTest(unittest.TestCase):
    def test_exact_registry_match_is_bounded_c1_evidence(self) -> None:
        runner = _runner()

        report = probe_max_runtime_support(
            "Qwen3MoeForCausalLM",
            "bfloat16",
            command_runner=runner,
        )

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            "injected_fixture_runner",
            report["collection_provenance"]["mode"],
        )
        self.assertFalse(
            report["collection_provenance"]["physical_observation_eligible"]
        )
        self.assertFalse(report["collection_provenance"]["authenticated"])
        self.assertEqual(
            [
                ("max", "--version"),
                ("max", "list", "--json"),
            ],
            runner.calls,
        )
        self.assertEqual("MAX 26.5.0", report["observed"]["max_version"])
        self.assertTrue(report["observed"]["architecture_present"])
        self.assertTrue(report["observed"]["encoding_present"])
        self.assertEqual(
            ["bfloat16", "float8_e4m3fn"],
            report["observed"]["supported_encodings"],
        )
        self.assertEqual("C1_contracted", report["qualification"]["maturity"])
        self.assertTrue(report["qualification"]["registry_match_passed"])
        self.assertFalse(report["qualification"]["runtime_compatibility_passed"])
        self.assertEqual(set(PHYSICAL_CLAIM_KEYS), set(report["physical_claims"]))
        self.assertFalse(any(report["physical_claims"].values()))
        self.assertFalse(report["commands"]["version"]["shell"])
        self.assertFalse(report["commands"]["list_json"]["shell"])
        json.dumps(report, allow_nan=False)

    def test_multi_part_max_command_remains_direct_argv(self) -> None:
        command = ("pixi", "run", "max")
        runner = _runner(command=command)

        report = probe_max_runtime_support(
            "Qwen3MoeForCausalLM",
            "float8_e4m3fn",
            max_command=command,
            command_runner=runner,
        )

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            [
                ("pixi", "run", "max", "--version"),
                ("pixi", "run", "max", "list", "--json"),
            ],
            runner.calls,
        )

    def test_architecture_and_encoding_matches_are_exact(self) -> None:
        cases = (
            (
                "qwen3moeforcausallm",
                "bfloat16",
                "does not advertise exact architecture",
            ),
            (
                "Qwen3MoeForCausalLM",
                "BFLOAT16",
                "does not advertise exact encoding",
            ),
        )
        for architecture, encoding, expected_error in cases:
            with self.subTest(architecture=architecture, encoding=encoding):
                report = probe_max_runtime_support(
                    architecture,
                    encoding,
                    command_runner=_runner(),
                )
                self.assertFalse(report["ok"])
                self.assertIn(expected_error, "\n".join(report["errors"]))
                self.assertFalse(report["qualification"]["registry_match_passed"])
                self.assertFalse(any(report["physical_claims"].values()))

    def test_command_failures_and_invalid_runner_results_fail_closed(self) -> None:
        cases = (
            (
                _runner(version=(7, "", "version failed")),
                "max --version failed: version failed",
            ),
            (
                _runner(listing=(9, "", "list failed")),
                "max list --json failed: list failed",
            ),
            (
                _runner(version={"returncode": "zero", "stdout": ""}),
                "max --version failed: TypeError",
            ),
        )
        for runner, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                report = probe_max_runtime_support(
                    "Qwen3MoeForCausalLM",
                    "bfloat16",
                    command_runner=runner,
                )
                self.assertFalse(report["ok"])
                self.assertIn(expected_error, "\n".join(report["errors"]))
                self.assertFalse(any(report["physical_claims"].values()))

    def test_failure_diagnostics_are_bounded_and_invalid_utf8_fails_closed(
        self,
    ) -> None:
        oversized_stderr = "failure " * MAX_DIAGNOSTIC_CHARS
        bounded = probe_max_runtime_support(
            "Qwen3MoeForCausalLM",
            "bfloat16",
            command_runner=_runner(version=(7, "", oversized_stderr)),
        )
        command = bounded["commands"]["version"]
        self.assertFalse(bounded["ok"])
        self.assertEqual(MAX_DIAGNOSTIC_CHARS, len(command["stderr_excerpt"]))
        self.assertTrue(command["stderr_excerpt_truncated"])
        self.assertLessEqual(
            len("\n".join(bounded["errors"])),
            MAX_DIAGNOSTIC_CHARS + 64,
        )

        invalid_utf8 = probe_max_runtime_support(
            "Qwen3MoeForCausalLM",
            "bfloat16",
            command_runner=_runner(listing=(0, b"\xff", b"")),
        )
        self.assertFalse(invalid_utf8["ok"])
        self.assertIn("UnicodeDecodeError", "\n".join(invalid_utf8["errors"]))
        self.assertFalse(any(invalid_utf8["physical_claims"].values()))

    def test_malformed_duplicate_and_invalid_registry_shapes_fail_closed(self) -> None:
        malformed_cases = (
            ("{", "invalid JSON"),
            (
                '{"architectures":{"A":{"supported_encodings":["x"]},'
                '"A":{"supported_encodings":["x"]}}}',
                "duplicate JSON key: A",
            ),
            ("[]", "top level must be an object"),
            ('{"architectures":[]}', "architectures must be an object"),
            ('{"architectures":{"A":[]}}', "must map to an object"),
            (
                '{"architectures":{"A":{"supported_encodings":"x"}}}',
                "supported_encodings must be a list",
            ),
            (
                '{"architectures":{"A":{"supported_encodings":["x",1]}}}',
                "supported_encodings[1] must be a non-empty string",
            ),
            (
                '{"architectures":{"A":{"supported_encodings":["x","x"]}}}',
                "must not contain duplicates",
            ),
            (
                '{"architectures":{"A":{"supported_encodings":[NaN]}}}',
                "non-finite JSON value is forbidden",
            ),
            (
                '{"architectures":{"A":{"supported_encodings":["x"],'
                '"score":1e9999}}}',
                "non-finite JSON value is forbidden",
            ),
        )
        for payload, expected_error in malformed_cases:
            with self.subTest(expected_error=expected_error):
                report = probe_max_runtime_support(
                    "A",
                    "x",
                    command_runner=_runner(listing=(0, payload, "")),
                )
                self.assertFalse(report["ok"])
                self.assertIn(expected_error, "\n".join(report["errors"]))
                self.assertFalse(any(report["physical_claims"].values()))

    def test_empty_version_and_oversized_registry_fail_closed(self) -> None:
        empty_version = probe_max_runtime_support(
            "Qwen3MoeForCausalLM",
            "bfloat16",
            command_runner=_runner(version=(0, " \n", "")),
        )
        self.assertFalse(empty_version["ok"])
        self.assertIn("returned no version text", "\n".join(empty_version["errors"]))

        oversized = " " * (MAX_LIST_OUTPUT_BYTES + 1)
        oversized_registry = probe_max_runtime_support(
            "Qwen3MoeForCausalLM",
            "bfloat16",
            command_runner=_runner(listing=(0, oversized, "")),
        )
        self.assertFalse(oversized_registry["ok"])
        self.assertIn(
            "output exceeds the bounded limit",
            "\n".join(oversized_registry["errors"]),
        )

    def test_live_registry_overflow_is_killed_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "fake_max.py"
            script.write_text(
                "import sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('MAX 26.7')\n"
                "else:\n"
                "    sys.stdout.write('x' * 1024)\n",
                encoding="utf-8",
            )
            with patch(
                "fornax.max_runtime_probe.MAX_LIST_OUTPUT_BYTES",
                128,
            ):
                report = probe_max_runtime_support(
                    "Qwen3MoeForCausalLM",
                    "bfloat16",
                    max_command=(sys.executable, str(script)),
                )

        self.assertFalse(report["ok"])
        self.assertIn(
            "SubprocessOutputLimitExceeded",
            str(report["commands"]["list_json"]["launch_error"]),
        )
        self.assertFalse(any(report["physical_claims"].values()))

    def test_invalid_probe_arguments_are_rejected_before_commands_run(self) -> None:
        runner = _runner()
        cases = (
            ("", "bfloat16", ("max",)),
            (" Qwen3MoeForCausalLM", "bfloat16", ("max",)),
            ("Qwen3MoeForCausalLM", "", ("max",)),
            ("Qwen3MoeForCausalLM", "bfloat16", ()),
        )
        for architecture, encoding, command in cases:
            with self.subTest(
                architecture=architecture,
                encoding=encoding,
                command=command,
            ):
                with self.assertRaises(ValueError):
                    probe_max_runtime_support(
                        architecture,
                        encoding,
                        max_command=command,
                        command_runner=runner,
                    )
        self.assertEqual([], runner.calls)


if __name__ == "__main__":
    unittest.main()
