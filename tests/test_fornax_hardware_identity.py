from __future__ import annotations

import json
import sys
import unittest
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

from fornax.bounded_subprocess import SubprocessOutputLimitExceeded
from fornax.hardware_identity import (
    NVIDIA_SMI_COMMAND,
    SYSTEM_PROFILER_COMMAND,
    _default_command_runner,
    collect_host_identity,
    match_platform_identity,
)


_GIB = 1024**3


class _FixtureRunner:
    def __init__(self, results: dict[tuple[str, ...], Any]) -> None:
        self.results = results
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> Any:
        command = tuple(argv)
        self.calls.append(command)
        return self.results.get(command, (127, "", "fixture tool unavailable"))


def _apple_identity(
    *,
    chip: str,
    memory_gib: int,
    machine_name: str,
    model_identifier: str,
) -> dict[str, Any]:
    profiler = json.dumps(
        {
            "SPHardwareDataType": [
                {
                    "machine_name": machine_name,
                    "machine_model": model_identifier,
                    "model_number": "FIXTURE-MODEL-NUMBER",
                    "chip_type": chip,
                    "physical_memory": f"{memory_gib} GB",
                }
            ]
        }
    )
    runner = _FixtureRunner(
        {
            SYSTEM_PROFILER_COMMAND: {
                "returncode": 0,
                "stdout": profiler,
                "stderr": "",
            },
            ("sysctl", "-n", "hw.memsize"): (
                0,
                f"{memory_gib * _GIB}\n",
                "",
            ),
            ("sysctl", "-n", "hw.model"): (0, f"{model_identifier}\n", ""),
            ("sysctl", "-n", "machdep.cpu.brand_string"): (
                0,
                f"{chip}\n",
                "",
            ),
            ("sw_vers", "-productName"): (0, "macOS\n", ""),
            ("sw_vers", "-productVersion"): (0, "fixture-version\n", ""),
            ("sw_vers", "-buildVersion"): (0, "fixture-build\n", ""),
            NVIDIA_SMI_COMMAND: (127, "", "nvidia-smi unavailable"),
        }
    )
    return collect_host_identity(runner)


def _nvidia_identity(
    rows: Sequence[tuple[int, str, int, str, str, str]],
) -> dict[str, Any]:
    csv_text = "\n".join(
        ", ".join(
            (
                str(index),
                f"GPU-00000000-0000-0000-0000-{index + 1:012x}",
                name,
                str(memory_mib),
                driver,
                pci_bus_id,
                pci_device_id,
            )
        )
        for index, name, memory_mib, driver, pci_bus_id, pci_device_id in rows
    )
    runner = _FixtureRunner(
        {
            SYSTEM_PROFILER_COMMAND: (127, "", "system_profiler unavailable"),
            NVIDIA_SMI_COMMAND: (0, csv_text, ""),
        }
    )
    return collect_host_identity(runner)


class HardwareIdentityTest(unittest.TestCase):
    def test_live_command_capture_overflow_is_killed(self) -> None:
        with patch(
            "fornax.hardware_identity._COMMAND_STDOUT_LIMIT_BYTES",
            128,
        ):
            with self.assertRaises(SubprocessOutputLimitExceeded) as caught:
                _default_command_runner(
                    (
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.write('x' * 1024)",
                    )
                )

        self.assertIn("stdout", caught.exception.streams)
        self.assertIsNotNone(caught.exception.returncode)

    def test_collect_and_match_selected_apple_profiles(self) -> None:
        cases = (
            (
                "Apple M3 Max",
                128,
                "MacBook Pro",
                "fixture-m3-max",
            ),
            (
                "Apple M5 Max",
                128,
                "MacBook Pro",
                "fixture-m5-max",
            ),
            (
                "Apple M3 Ultra",
                512,
                "Mac Studio",
                "fixture-m3-ultra",
            ),
        )
        for chip, memory_gib, machine_name, model_identifier in cases:
            with self.subTest(chip=chip, memory_gib=memory_gib):
                identity = _apple_identity(
                    chip=chip,
                    memory_gib=memory_gib,
                    machine_name=machine_name,
                    model_identifier=model_identifier,
                )
                self.assertEqual("apple", identity["kind"])
                self.assertEqual([], identity["collection_errors"])
                self.assertEqual(chip, identity["apple"]["chip"])
                self.assertEqual(memory_gib * _GIB, identity["apple"]["memory_bytes"])
                self.assertEqual(model_identifier, identity["apple"]["model_identifier"])
                self.assertEqual("fixture-build", identity["apple"]["os"]["build"])

                report = match_platform_identity(
                    identity,
                    {
                        "vendor": "apple",
                        "chip": chip,
                        "memory_gb": memory_gib,
                        "machine_name": machine_name,
                        "model_identifier": model_identifier,
                        "units": 1,
                    },
                )
                self.assertTrue(report["ok"], report)
                self.assertEqual([], report["errors"])
                self.assertEqual(chip, report["expected"]["chip"])
                self.assertEqual(chip, report["observed"]["chip"])

    def test_collect_and_match_selected_nvidia_profiles(self) -> None:
        cases = (
            (
                "NVIDIA H100 80GB HBM3",
                81559,
                80,
                "0x233010DE",
            ),
            (
                "NVIDIA GeForce RTX 4090",
                24564,
                24,
                "0x268410DE",
            ),
            (
                "NVIDIA GeForce RTX 5090",
                32607,
                32,
                "0x2B8510DE",
            ),
        )
        for name, memory_mib, minimum_gb, pci_device_id in cases:
            with self.subTest(name=name):
                identity = _nvidia_identity(
                    (
                        (
                            0,
                            name,
                            memory_mib,
                            "fixture-driver",
                            "00000000:01:00.0",
                            pci_device_id,
                        ),
                    )
                )
                self.assertEqual("nvidia", identity["kind"])
                self.assertEqual([], identity["collection_errors"])
                gpu = identity["nvidia"]["gpus"][0]
                self.assertEqual(
                    "GPU-00000000-0000-0000-0000-000000000001",
                    gpu["uuid"],
                )
                self.assertEqual(name, gpu["name"])
                self.assertEqual(memory_mib * 1024**2, gpu["memory_total_bytes"])
                self.assertEqual("fixture-driver", gpu["driver_version"])
                self.assertEqual("00000000:01:00.0", gpu["pci_bus_id"])
                self.assertEqual(pci_device_id, gpu["pci_device_id"])

                report = match_platform_identity(
                    identity,
                    {
                        "vendor": "nvidia",
                        "gpu_name": name,
                        "vram_gb": minimum_gb,
                        "units": 1,
                    },
                )
                self.assertTrue(report["ok"], report)
                self.assertEqual([0], report["observed"]["matching_gpu_indices"])
                self.assertEqual(
                    ["GPU-00000000-0000-0000-0000-000000000001"],
                    report["observed"]["matching_gpu_uuids"],
                )

    def test_nvidia_runtime_host_requirements_are_enforced(self) -> None:
        identity = _nvidia_identity(
            (
                (
                    0,
                    "NVIDIA H100 80GB HBM3",
                    81559,
                    "580.0",
                    "00000000:01:00.0",
                    "0x233010DE",
                ),
            )
        )
        profile = {
            "vendor": "nvidia",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "vram_gb": 80,
            "runtime": {
                "os_family": "linux",
                "architecture": "x86_64",
                "minimum_os": "Ubuntu 22.04",
                "minimum_driver": "580",
            },
        }
        identity["host"] = {
            "system": "Windows",
            "machine": "ARM64",
            "node": "fixture",
            "os_release": None,
        }
        mismatched = match_platform_identity(identity, profile)
        self.assertFalse(mismatched["ok"])
        joined = "\n".join(mismatched["errors"])
        self.assertIn("operating-system mismatch", joined)
        self.assertIn("architecture mismatch", joined)
        self.assertIn("distribution mismatch", joined)

        identity["host"] = {
            "system": "Linux",
            "machine": "AMD64",
            "node": "fixture",
            "os_release": {
                "id": "ubuntu",
                "version_id": "24.04",
                "name": "Ubuntu",
                "pretty_name": "Ubuntu 24.04",
            },
        }
        matched = match_platform_identity(identity, profile)
        self.assertTrue(matched["ok"], matched)
        self.assertEqual("linux", matched["observed"]["host_os_family"])
        self.assertEqual("x86_64", matched["observed"]["host_architecture"])
        self.assertEqual(
            "24.04",
            matched["observed"]["host_os_release_version"],
        )

    def test_apple_runtime_host_requirements_are_enforced(self) -> None:
        identity = _apple_identity(
            chip="Apple M3 Max",
            memory_gib=128,
            machine_name="MacBook Pro",
            model_identifier="fixture-m3-max",
        )
        identity["apple"]["os"]["version"] = "15.1"
        profile = {
            "vendor": "apple",
            "chip": "Apple M3 Max",
            "memory_gb": 128,
            "runtime": {
                "os_family": "macos",
                "architecture": "arm64",
                "minimum_os": "macOS 15",
            },
        }

        identity["host"] = {
            "system": "Windows",
            "machine": "x86_64",
            "node": "fixture",
            "os_release": None,
        }
        mismatched = match_platform_identity(identity, profile)
        self.assertFalse(mismatched["ok"])
        joined = "\n".join(mismatched["errors"])
        self.assertIn("Apple host operating-system mismatch", joined)
        self.assertIn("Apple host architecture mismatch", joined)

        identity["host"] = {
            "system": "Darwin",
            "machine": "AARCH64",
            "node": "fixture",
            "os_release": None,
        }
        matched = match_platform_identity(identity, profile)
        self.assertTrue(matched["ok"], matched)
        self.assertEqual("macos", matched["observed"]["host_os_family"])
        self.assertEqual("arm64", matched["observed"]["host_architecture"])
        self.assertEqual("macos", matched["expected"]["os_family"])
        self.assertEqual("arm64", matched["expected"]["architecture"])

    def test_nvidia_uuid_is_required_and_must_be_unique(self) -> None:
        cases = (
            (
                "0, not-a-uuid, NVIDIA H100 80GB HBM3, 81559, "
                "fixture-driver, 00000000:01:00.0, 0x233010DE",
                "invalid physical GPU UUID",
            ),
            (
                "\n".join(
                    (
                        "0, GPU-12345678-1234-5678-9abc-def012345678, "
                        "NVIDIA H100 80GB HBM3, 81559, fixture-driver, "
                        "00000000:01:00.0, 0x233010DE",
                        "1, GPU-12345678-1234-5678-9abc-def012345678, "
                        "NVIDIA H100 80GB HBM3, 81559, fixture-driver, "
                        "00000000:02:00.0, 0x233010DE",
                    )
                ),
                "duplicate physical GPU UUID",
            ),
        )
        for csv_text, expected in cases:
            with self.subTest(expected=expected):
                identity = collect_host_identity(
                    _FixtureRunner(
                        {
                            SYSTEM_PROFILER_COMMAND: (
                                127,
                                "",
                                "system_profiler unavailable",
                            ),
                            NVIDIA_SMI_COMMAND: (0, csv_text, ""),
                        }
                    )
                )
                self.assertIn(expected, "\n".join(identity["collection_errors"]))
                report = match_platform_identity(
                    identity,
                    {
                        "vendor": "nvidia",
                        "gpu_name": "NVIDIA H100 80GB HBM3",
                        "vram_gb": 80,
                    },
                )
                self.assertFalse(report["ok"])

    def test_apple_wrong_chip_and_insufficient_memory_fail(self) -> None:
        identity = _apple_identity(
            chip="Apple M3 Max",
            memory_gib=64,
            machine_name="MacBook Pro",
            model_identifier="fixture-m3-max",
        )
        report = match_platform_identity(
            identity,
            {
                "vendor": "apple",
                "chip": "Apple M5 Max",
                "memory_gb": 128,
            },
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("chip mismatch" in error for error in report["errors"]),
            report,
        )
        self.assertTrue(
            any("below the profile minimum" in error for error in report["errors"]),
            report,
        )

    def test_nvidia_wrong_sku_and_insufficient_memory_fail(self) -> None:
        identity = _nvidia_identity(
            (
                (
                    0,
                    "NVIDIA GeForce RTX 4090",
                    24564,
                    "fixture-driver",
                    "00000000:01:00.0",
                    "0x268410DE",
                ),
            )
        )
        wrong_sku = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "vram_gb": 24,
            },
        )
        self.assertFalse(wrong_sku["ok"])
        self.assertTrue(
            any("no observed NVIDIA GPU matches" in error for error in wrong_sku["errors"]),
            wrong_sku,
        )

        insufficient_memory = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "vram_gb": 32,
            },
        )
        self.assertFalse(insufficient_memory["ok"])
        self.assertTrue(
            any(
                "no observed NVIDIA GPU matches" in error
                for error in insufficient_memory["errors"]
            ),
            insufficient_memory,
        )

    def test_m5_512_profile_is_rejected_before_hardware_support_claim(self) -> None:
        identity = _apple_identity(
            chip="Apple M5 Max",
            memory_gib=128,
            machine_name="MacBook Pro",
            model_identifier="fixture-m5-max",
        )
        report = match_platform_identity(
            identity,
            {
                "vendor": "apple",
                "label": "Apple M5 Max 512GB",
                "chip": "Apple M5 Max",
                "memory_gb": 512,
            },
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("M5 512GB" in error for error in report["errors"]),
            report,
        )
        self.assertNotIn("verified", " ".join(report["errors"]).casefold())
        self.assertNotIn("supported", " ".join(report["errors"]).casefold())

    def test_explicit_multi_unit_profile_requires_count(self) -> None:
        identity = _nvidia_identity(
            (
                (
                    0,
                    "NVIDIA GeForce RTX 5090",
                    32607,
                    "fixture-driver",
                    "00000000:01:00.0",
                    "0x2B8510DE",
                ),
            )
        )
        report = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "vram_gb": 32,
                "units": 2,
            },
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("requires at least 2 GPUs" in error for error in report["errors"]),
            report,
        )

    def test_explicit_multi_unit_profile_requires_homogeneous_gpus(self) -> None:
        identity = _nvidia_identity(
            (
                (
                    0,
                    "NVIDIA GeForce RTX 5090",
                    32607,
                    "fixture-driver",
                    "00000000:01:00.0",
                    "0x2B8510DE",
                ),
                (
                    1,
                    "NVIDIA GeForce RTX 4090",
                    24564,
                    "fixture-driver",
                    "00000000:02:00.0",
                    "0x268410DE",
                ),
            )
        )
        report = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "vram_gb": 32,
                "units": 2,
            },
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("all observed GPUs must match" in error for error in report["errors"]),
            report,
        )
        self.assertEqual([0], report["observed"]["matching_gpu_indices"])

    def test_homogeneous_multi_unit_profile_matches(self) -> None:
        identity = _nvidia_identity(
            (
                (
                    0,
                    "NVIDIA H100 80GB HBM3",
                    81559,
                    "fixture-driver",
                    "00000000:01:00.0",
                    "0x233010DE",
                ),
                (
                    1,
                    "NVIDIA H100 80GB HBM3",
                    81559,
                    "fixture-driver",
                    "00000000:02:00.0",
                    "0x233010DE",
                ),
            )
        )
        report = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "vram_gb": 80,
                "units": 2,
            },
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual([0, 1], report["observed"]["matching_gpu_indices"])

    def test_omitted_units_allows_one_match_on_mixed_host(self) -> None:
        identity = _nvidia_identity(
            (
                (
                    0,
                    "NVIDIA GeForce RTX 5090",
                    32607,
                    "fixture-driver",
                    "00000000:01:00.0",
                    "0x2B8510DE",
                ),
                (
                    1,
                    "NVIDIA GeForce RTX 4090",
                    24564,
                    "fixture-driver",
                    "00000000:02:00.0",
                    "0x268410DE",
                ),
            )
        )
        report = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "vram_gb": 32,
            },
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual([0], report["observed"]["matching_gpu_indices"])
        self.assertTrue(
            any("additional observed GPUs" in warning for warning in report["warnings"]),
            report,
        )

    def test_vendor_collection_errors_fail_closed_with_a_matching_device(self) -> None:
        csv_text = "\n".join(
            (
                "0, GPU-00000000-0000-0000-0000-000000000001, "
                "NVIDIA GeForce RTX 5090, 32607, fixture-driver, "
                "00000000:01:00.0, 0x2B8510DE",
                "malformed row",
            )
        )
        runner = _FixtureRunner(
            {
                SYSTEM_PROFILER_COMMAND: (
                    127,
                    "",
                    "system_profiler unavailable",
                ),
                NVIDIA_SMI_COMMAND: (0, csv_text, ""),
            }
        )
        identity = collect_host_identity(runner)
        self.assertTrue(identity["collection_errors"])

        report = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "vram_gb": 32,
                "units": 1,
            },
        )

        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "nvidia-smi row 2" in error
                for error in report["errors"]
            ),
            report,
        )

    def test_apple_collection_errors_fail_closed_with_a_matching_chip(self) -> None:
        identity = _apple_identity(
            chip="Apple M3 Max",
            memory_gib=128,
            machine_name="MacBook Pro",
            model_identifier="fixture-m3-max",
        )
        identity["collection_errors"].append(
            "apple: sysctl hw.memsize returned incomplete data"
        )

        report = match_platform_identity(
            identity,
            {
                "vendor": "apple",
                "chip": "Apple M3 Max",
                "memory_gb": 128,
            },
        )

        self.assertFalse(report["ok"])
        self.assertTrue(
            any("sysctl hw.memsize" in error for error in report["errors"]),
            report,
        )

    def test_other_vendor_collection_errors_do_not_block_a_match(self) -> None:
        identity = _nvidia_identity(
            (
                (
                    0,
                    "NVIDIA H100 80GB HBM3",
                    81559,
                    "fixture-driver",
                    "00000000:01:00.0",
                    "0x233010DE",
                ),
            )
        )
        identity["collection_errors"].append(
            "apple: system_profiler returned incomplete Apple fields"
        )

        report = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "vram_gb": 80,
            },
        )

        self.assertTrue(report["ok"], report)

    def test_missing_hardware_tools_fail_closed(self) -> None:
        runner = _FixtureRunner({})
        identity = collect_host_identity(runner)
        self.assertEqual(
            "injected_fixture_runner",
            identity["collection_provenance"]["mode"],
        )
        self.assertFalse(
            identity["collection_provenance"]["physical_observation_eligible"]
        )
        self.assertEqual("unknown", identity["kind"])
        self.assertIsNone(identity["apple"])
        self.assertIsNone(identity["nvidia"])
        self.assertTrue(identity["collection_errors"])
        self.assertTrue(identity["diagnostics"]["apple"])
        self.assertTrue(identity["diagnostics"]["nvidia"])
        self.assertEqual(
            [SYSTEM_PROFILER_COMMAND, NVIDIA_SMI_COMMAND],
            runner.calls,
        )

        report = match_platform_identity(
            identity,
            {
                "vendor": "nvidia",
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "vram_gb": 80,
            },
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("no NVIDIA GPU identities" in error for error in report["errors"]),
            report,
        )


if __name__ == "__main__":
    unittest.main()
