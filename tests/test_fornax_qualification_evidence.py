from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fornax.cli as cli_module
from fornax.cli import main as cli_main
from fornax.max_runtime_probe import (
    EVIDENCE_SCOPE as MAX_RUNTIME_EVIDENCE_SCOPE,
    PHYSICAL_CLAIM_KEYS,
    REPORT_KIND as MAX_RUNTIME_REPORT_KIND,
)
from fornax.max_generation_smoke import SMOKE_SENTINEL, SMOKE_SENTINEL_PROMPT
from fornax.model_artifacts import inspect_model_artifacts
from fornax.qualification_evidence import (
    QualificationEvidenceError,
    executable_identity,
    load_json_evidence,
    validate_apple_single_preflights,
    validate_nvidia_single_preflights,
)


REVISION = "a" * 40
CATALOG_SHA256 = "sha256:" + "c" * 64
MODEL_PROFILE_SHA256 = "sha256:" + "d" * 64
PLATFORM_PROFILE_SHA256 = "sha256:" + "e" * 64
GPU_UUID = "GPU-12345678-1234-5678-9abc-def012345678"


class QualificationEvidenceTest(unittest.TestCase):
    def _snapshot_and_report(
        self, root: Path
    ) -> tuple[dict[str, object], dict[str, object]]:
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3MoeForCausalLM"],
                    "hidden_size": 2048,
                    "max_position_embeddings": 40960,
                    "model_type": "qwen3_moe",
                    "num_experts": 128,
                    "num_experts_per_tok": 8,
                    "num_hidden_layers": 48,
                }
            ),
            encoding="utf-8",
        )
        (root / "tokenizer.json").write_text(
            '{"model":{"type":"BPE"}}',
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            '{"tokenizer_class":"Qwen2Tokenizer"}',
            encoding="utf-8",
        )
        shard = root / "model.safetensors"
        shard.write_bytes(b"bounded-test-shard")
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": shard.stat().st_size},
                    "weight_map": {
                        "model.layers.0.mlp.experts.0.weight": shard.name
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / ".fornax-revision").write_text(
            REVISION + "\n",
            encoding="utf-8",
        )
        profile: dict[str, object] = {
            "schema_version": 1,
            "kind": "fornax_model_qualification_profile",
            "model_id": "qwen3-30b-a3b",
            "artifact": {
                "provider": "hugging_face",
                "repository": "Qwen/Qwen3-30B-A3B",
                "revision_kind": "git_commit",
                "revision": REVISION,
                "weight_format": "safetensors",
                "weight_dtype": "bf16",
                "quantization": "none",
                "trust_remote_code_required": False,
                "required_files": [
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
            },
            "architecture": {
                "family": "qwen3_moe",
                "decoder_layers": 48,
                "hidden_size": 2048,
                "total_experts": 128,
                "active_experts_per_token": 8,
                "config_max_position_embeddings": 40960,
            },
        }
        initial = inspect_model_artifacts(root, profile)
        self.assertTrue(initial["ok"], initial["errors"])
        artifact = profile["artifact"]
        self.assertIsInstance(artifact, dict)
        artifact["file_hashes"] = {
            row["path"]: row["sha256"] for row in initial["files"]
        }
        report = inspect_model_artifacts(
            root,
            profile,
            catalog_sha256=CATALOG_SHA256,
            profile_sha256=MODEL_PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )
        self.assertTrue(report["ok"], report["errors"])
        return profile, report

    @staticmethod
    def _file_record(label: str) -> dict[str, object]:
        return {
            "path": f"/evidence/{label}.json",
            "size_bytes": 100,
            "sha256": "sha256:" + "1" * 64,
            "device": 1,
            "inode": 2,
            "authenticated": False,
        }

    @staticmethod
    def _host_report(
        nvidia_smi: dict[str, object],
        *,
        matching_indices: list[int] | None = None,
        gpu_uuid: str = GPU_UUID,
    ) -> dict[str, object]:
        indices = [1] if matching_indices is None else matching_indices
        return {
            "schema_version": 1,
            "record_kind": "fornax_qualification_host_identity",
            "platform_id": "nvidia-h100-sxm-80gb",
            "platform_profile_sha256": PLATFORM_PROFILE_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "identity": {
                "collection_provenance": {
                    "mode": "live_subprocess",
                    "physical_observation_eligible": True,
                    "authenticated": False,
                }
            },
            "match": {
                "ok": True,
                "errors": [],
                "observed": {
                    "gpus": [
                        {
                            "index": 1,
                            "uuid": gpu_uuid,
                            "name": "NVIDIA H100 80GB HBM3",
                            "memory_total_bytes": 80 * 1_000_000_000,
                            "driver_version": "fixture-driver",
                            "pci_bus_id": "00000000:01:00.0",
                            "pci_device_id": "0x233010DE",
                            "name_matches": True,
                            "memory_matches": True,
                        }
                    ],
                    "matching_gpu_indices": indices,
                    "matching_gpu_uuids": (
                        [gpu_uuid] if 1 in indices else []
                    ),
                },
            },
            "collector_executables": {"nvidia-smi": nvidia_smi},
            "evidence_scope": "observed_host_identity_only",
            "qualification": {
                "maturity": "C1_contracted",
                "identity_match_passed": True,
                "runtime_compatibility_passed": False,
                "model_bringup_passed": False,
                "target_model_parity_passed": False,
                "formal_g2_passed": False,
                "production_supported": False,
            },
        }

    @staticmethod
    def _runtime_report(
        max_command: tuple[str, ...],
        max_executable: dict[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "record_kind": MAX_RUNTIME_REPORT_KIND,
            "evidence_scope": MAX_RUNTIME_EVIDENCE_SCOPE,
            "collection_provenance": {
                "mode": "live_subprocess",
                "physical_observation_eligible": True,
                "authenticated": False,
            },
            "ok": True,
            "errors": [],
            "warnings": [],
            "catalog_sha256": CATALOG_SHA256,
            "model": {
                "model_id": "qwen3-30b-a3b",
                "profile_sha256": MODEL_PROFILE_SHA256,
                "repository": "Qwen/Qwen3-30B-A3B",
            },
            "platform": {
                "platform_id": "nvidia-h100-sxm-80gb",
                "profile_sha256": PLATFORM_PROFILE_SHA256,
            },
            "expected": {
                "architecture": "Qwen3MoeForCausalLM",
                "encoding": "bfloat16",
            },
            "observed": {
                "architecture_present": True,
                "encoding_present": True,
            },
            "physical_claims": {key: False for key in PHYSICAL_CLAIM_KEYS},
            "qualification": {
                "maturity": "C1_contracted",
                "authority": "exploratory",
                "registry_match_passed": True,
                "runtime_compatibility_passed": False,
                "physical_execution_status": "not_run",
                "production_supported": False,
            },
            "commands": {
                "version": {"argv": [*max_command, "--version"]},
                "list_json": {"argv": [*max_command, "list", "--json"]},
            },
            "max_executable": max_executable,
            "interpretation": "registry only",
        }

    @staticmethod
    def _apple_host_report(
        collectors: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        identity = {
            "collection_provenance": {
                "mode": "live_subprocess",
                "physical_observation_eligible": True,
                "authenticated": False,
            },
            "kind": "apple",
            "apple": {
                "chip": "Apple M3 Max",
                "memory_bytes": 128 * 1024**3,
                "machine_name": "MacBook Pro",
                "model_identifier": "Mac15,8",
                "model_number": "Z1CM00000",
            },
        }
        observed = {
            "vendor": "apple",
            "chip": "Apple M3 Max",
            "memory_bytes": 128 * 1024**3,
            "machine_name": "MacBook Pro",
            "model_identifier": "Mac15,8",
            "model_number": "Z1CM00000",
        }
        return {
            "schema_version": 1,
            "record_kind": "fornax_qualification_host_identity",
            "platform_id": "apple-m3-max-128",
            "platform_profile_sha256": PLATFORM_PROFILE_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "identity": identity,
            "match": {"ok": True, "errors": [], "observed": observed},
            "collector_executables": collectors,
            "evidence_scope": "observed_host_identity_only",
            "qualification": {
                "maturity": "C1_contracted",
                "identity_match_passed": True,
                "runtime_compatibility_passed": False,
                "model_bringup_passed": False,
                "target_model_parity_passed": False,
                "formal_g2_passed": False,
                "production_supported": False,
            },
        }

    @staticmethod
    def _apple_runtime_report(
        max_command: tuple[str, ...],
        max_executable: dict[str, object],
    ) -> dict[str, object]:
        report = QualificationEvidenceTest._runtime_report(
            max_command,
            max_executable,
        )
        report["platform"] = {
            "platform_id": "apple-m3-max-128",
            "profile_sha256": PLATFORM_PROFILE_SHA256,
            "vendor": "apple",
            "runtime_verification_status": "unverified",
        }
        report["observed"] = {
            "architecture_present": True,
            "encoding_present": True,
            "max_version": "MAX 26.7",
        }
        return report

    def test_evidence_loader_rejects_symlink_duplicate_and_nonfinite_json(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            valid = root / "valid.json"
            valid.write_text('{"ok":true}', encoding="utf-8")
            report, record = load_json_evidence(valid, label="valid")
            self.assertEqual({"ok": True}, report)
            self.assertEqual(valid.resolve(), Path(record["path"]))
            self.assertFalse(record["authenticated"])

            symlink = root / "symlink.json"
            try:
                symlink.symlink_to(valid)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(QualificationEvidenceError):
                load_json_evidence(symlink, label="symlink")

            fifo = root / "fifo.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(
                QualificationEvidenceError,
                "regular file",
            ):
                load_json_evidence(fifo, label="fifo")

            duplicate = root / "duplicate.json"
            duplicate.write_text('{"ok":true,"ok":false}', encoding="utf-8")
            with self.assertRaisesRegex(
                QualificationEvidenceError, "duplicate JSON key"
            ):
                load_json_evidence(duplicate, label="duplicate")

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":1e9999}', encoding="utf-8")
            with self.assertRaisesRegex(
                QualificationEvidenceError, "non-finite JSON number"
            ):
                load_json_evidence(nonfinite, label="nonfinite")

    def test_executable_identity_is_content_bound_and_unauthenticated(self) -> None:
        identity = executable_identity(sys.executable)
        self.assertEqual(str(Path(sys.executable).resolve()), identity["resolved_path"])
        self.assertRegex(identity["sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertGreater(identity["size_bytes"], 0)
        self.assertFalse(identity["authenticated"])

    def test_apple_envelope_writer_refuses_symlink_and_nonregular_targets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            victim = root / "victim.json"
            victim.write_text("unchanged\n", encoding="utf-8")
            symlink = root / "result.json"
            try:
                symlink.symlink_to(victim)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                cli_module._write_new_json_no_follow(
                    symlink,
                    {"ok": False},
                )
            self.assertEqual("unchanged\n", victim.read_text(encoding="utf-8"))

            fifo = root / "result.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "non-regular"):
                cli_module._write_new_json_no_follow(
                    fifo,
                    {"ok": False},
                )
            self.assertTrue(fifo.exists())

            existing = root / "existing.json"
            existing.write_text("first\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "existing file"):
                cli_module._write_new_json_no_follow(
                    existing,
                    {"ok": True},
                )
            self.assertEqual("first\n", existing.read_text(encoding="utf-8"))

            real_parent = root / "real-parent"
            real_parent.mkdir()
            ancestor_link = root / "ancestor-link"
            ancestor_link.symlink_to(real_parent, target_is_directory=True)
            redirected = ancestor_link / "redirected.json"
            with self.assertRaisesRegex(ValueError, "parent components"):
                cli_module._write_new_json_no_follow(
                    redirected,
                    {"ok": False},
                )
            self.assertFalse((real_parent / "redirected.json").exists())

    def test_runtime_probe_rejects_multipart_command_prefix(self) -> None:
        for vendor, platform_id in (
            ("apple", "apple-m3-max-128"),
            ("nvidia", "nvidia-h100-sxm-80gb"),
        ):
            with self.subTest(vendor=vendor):
                catalog = SimpleNamespace(
                    model=lambda _model_id: object(),
                    platform=lambda _platform_id: SimpleNamespace(
                        to_dict=lambda: {"vendor": vendor}
                    ),
                )
                args = SimpleNamespace(
                    model="qwen3-30b-a3b",
                    platform=platform_id,
                    max_command="pixi run max",
                    out="/unused",
                )
                with mock.patch.object(
                    cli_module,
                    "load_qualification_catalog",
                    return_value=catalog,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "one direct executable",
                    ):
                        cli_module._cmd_recipe_probe_runtime(args)

    def test_runtime_probe_refuses_to_publish_through_symlink(self) -> None:
        catalog = SimpleNamespace(
            model=lambda _model_id: object(),
            platform=lambda _platform_id: object(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            victim = root / "victim.json"
            victim.write_text("preserve\n", encoding="utf-8")
            output = root / "runtime.json"
            output.symlink_to(victim)
            args = SimpleNamespace(
                model="qwen3-30b-a3b",
                platform="nvidia-h100-sxm-80gb",
                max_command=sys.executable,
                out=str(output),
            )
            with (
                mock.patch.object(
                    cli_module,
                    "load_qualification_catalog",
                    return_value=catalog,
                ),
                mock.patch.object(
                    cli_module,
                    "executable_identity",
                    return_value={"resolved_path": sys.executable},
                ),
                mock.patch.object(
                    cli_module,
                    "_collect_qualification_runtime_report",
                    return_value={"ok": True, "errors": []},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "symbolic link"):
                    cli_module._cmd_recipe_probe_runtime(args)
            self.assertEqual("preserve\n", victim.read_text(encoding="utf-8"))

    def test_nvidia_cli_requires_exact_sentinel_prompt(self) -> None:
        catalog = SimpleNamespace(
            catalog_sha256=CATALOG_SHA256,
            model=lambda _model_id: SimpleNamespace(
                model_id="qwen3-30b-a3b",
                profile_sha256=MODEL_PROFILE_SHA256,
                to_dict=lambda: {
                    "artifact": {
                        "repository": "Qwen/Qwen3-30B-A3B",
                        "revision": REVISION,
                    },
                    "runtime": {
                        "max_architecture": "Qwen3MoeForCausalLM",
                        "max_weight_encoding": "bfloat16",
                    },
                },
            ),
            platform=lambda _platform_id: SimpleNamespace(
                platform_id="nvidia-h100-sxm-80gb",
                profile_sha256=PLATFORM_PROFILE_SHA256,
                to_dict=lambda: {"vendor": "nvidia"},
            ),
        )
        args = SimpleNamespace(
            model="qwen3-30b-a3b",
            platform="nvidia-h100-sxm-80gb",
            model_dir="/unused",
            model_artifact_report="/unused/artifact.json",
            host_report="/unused/host.json",
            runtime_report="/unused/runtime.json",
            max_command=sys.executable,
            device="gpu:0",
            prompt="Define MoE.",
            max_new_tokens=8,
            top_k=1,
            timeout_s=10.0,
            out="/unused/out.json",
        )
        with (
            mock.patch.object(
                cli_module,
                "load_qualification_catalog",
                return_value=catalog,
            ),
            mock.patch.object(
                cli_module,
                "compose_qualification_recipe",
                return_value={
                    "lock": {"capacity_estimate": {"minimum_units": 1}}
                },
            ),
            mock.patch.object(
                cli_module,
                "run_max_generation_smoke",
            ) as generation,
        ):
            with self.assertRaisesRegex(ValueError, "exact bounded sentinel"):
                cli_module._cmd_recipe_run_nvidia_single(args)
        generation.assert_not_called()

    def test_valid_binding_requires_exact_device_and_all_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "model"
            _profile, artifact = self._snapshot_and_report(root)
            max_command = (sys.executable,)
            max_executable = executable_identity(sys.executable)
            nvidia_smi = {
                "requested_argv0": "nvidia-smi",
                "resolved_path": "/usr/bin/nvidia-smi",
                "size_bytes": 42,
                "sha256": "sha256:" + "2" * 64,
                "authenticated": False,
            }
            host_report = self._host_report(nvidia_smi)
            runtime_report = self._runtime_report(
                max_command,
                max_executable,
            )
            fresh_host_identity = {
                "collection_provenance": {
                    "mode": "live_subprocess",
                    "physical_observation_eligible": True,
                    "authenticated": False,
                }
            }
            preflight_kwargs = {
                "artifact_report": artifact,
                "fresh_artifact_report": artifact,
                "host_report": host_report,
                "runtime_report": runtime_report,
                "fresh_host_identity": fresh_host_identity,
                "fresh_host_match": host_report["match"],
                "fresh_runtime_report": runtime_report,
                "artifact_file": self._file_record("artifact"),
                "host_file": self._file_record("host"),
                "runtime_file": self._file_record("runtime"),
                "expected_catalog_sha256": CATALOG_SHA256,
                "expected_model_id": "qwen3-30b-a3b",
                "expected_model_profile_sha256": MODEL_PROFILE_SHA256,
                "expected_repository": "Qwen/Qwen3-30B-A3B",
                "expected_revision": REVISION,
                "expected_model_dir": root,
                "expected_platform_id": "nvidia-h100-sxm-80gb",
                "expected_platform_profile_sha256": PLATFORM_PROFILE_SHA256,
                "expected_architecture": "Qwen3MoeForCausalLM",
                "expected_encoding": "bfloat16",
                "expected_device": "gpu:1",
                "expected_max_command": max_command,
                "expected_max_executable": max_executable,
                "expected_nvidia_smi_executable": nvidia_smi,
                "minimum_units": 1,
            }
            binding = validate_nvidia_single_preflights(**preflight_kwargs)
            self.assertTrue(binding["ok"], binding["errors"])
            self.assertFalse(binding["authenticated"])
            self.assertRegex(binding["binding_sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(
                {
                    "physical_selector": "gpu:1",
                    "nvidia_smi_index": 1,
                    "nvidia_gpu_uuid": GPU_UUID,
                    "cuda_visible_devices": GPU_UUID,
                    "max_launch_device": "gpu:0",
                    "verified": True,
                },
                binding["device_binding"],
            )

            wrong_device = validate_nvidia_single_preflights(
                **{
                    **preflight_kwargs,
                    "expected_device": "gpu:0",
                    "minimum_units": 2,
                }
            )
            self.assertFalse(wrong_device["ok"])
            self.assertIn(
                "does not match the requested device",
                "\n".join(wrong_device["errors"]),
            )
            self.assertIn(
                "capacity-only minimum is 2",
                "\n".join(wrong_device["errors"]),
            )

            mismatched_fresh_match = json.loads(
                json.dumps(host_report["match"])
            )
            other_uuid = "GPU-87654321-4321-8765-cba9-876543210fed"
            mismatched_fresh_match["observed"]["gpus"][0]["uuid"] = other_uuid
            mismatched_fresh_match["observed"]["matching_gpu_uuids"] = [
                other_uuid
            ]
            mismatched_uuid = validate_nvidia_single_preflights(
                **{
                    **preflight_kwargs,
                    "fresh_host_match": mismatched_fresh_match,
                }
            )
            self.assertFalse(mismatched_uuid["ok"])
            self.assertIn(
                "different physical GPU UUIDs",
                "\n".join(mismatched_uuid["errors"]),
            )
            self.assertFalse(mismatched_uuid["device_binding"]["verified"])

    def test_apple_binding_requires_exact_live_host_collectors_and_lineage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "model"
            _profile, artifact = self._snapshot_and_report(root)
            max_command = (sys.executable,)
            max_executable = executable_identity(sys.executable)
            collectors = {
                command: {
                    "requested_argv0": command,
                    "resolved_path": f"/usr/bin/{command}",
                    "size_bytes": 42,
                    "sha256": "sha256:" + str(index) * 64,
                    "authenticated": False,
                }
                for index, command in enumerate(
                    ("system_profiler", "sysctl", "sw_vers"),
                    start=3,
                )
            }
            host_report = self._apple_host_report(collectors)
            runtime_report = self._apple_runtime_report(
                max_command,
                max_executable,
            )
            kwargs = {
                "artifact_report": artifact,
                "fresh_artifact_report": artifact,
                "host_report": host_report,
                "runtime_report": runtime_report,
                "fresh_host_identity": host_report["identity"],
                "fresh_host_match": host_report["match"],
                "fresh_runtime_report": runtime_report,
                "artifact_file": self._file_record("artifact"),
                "host_file": self._file_record("host"),
                "runtime_file": self._file_record("runtime"),
                "expected_catalog_sha256": CATALOG_SHA256,
                "expected_model_id": "qwen3-30b-a3b",
                "expected_model_profile_sha256": MODEL_PROFILE_SHA256,
                "expected_repository": "Qwen/Qwen3-30B-A3B",
                "expected_revision": REVISION,
                "expected_model_dir": root,
                "expected_platform_id": "apple-m3-max-128",
                "expected_platform_profile_sha256": PLATFORM_PROFILE_SHA256,
                "expected_platform_vendor": "apple",
                "expected_platform_runtime_verification_status": "unverified",
                "expected_apple_chip": "Apple M3 Max",
                "expected_apple_memory_bytes": 128 * 1024**3,
                "expected_architecture": "Qwen3MoeForCausalLM",
                "expected_encoding": "bfloat16",
                "expected_max_command": max_command,
                "expected_max_executable": max_executable,
                "expected_collector_executables": collectors,
                "minimum_units": 1,
            }
            binding = validate_apple_single_preflights(**kwargs)
            self.assertTrue(binding["ok"], binding["errors"])
            self.assertEqual(
                host_report["match"]["observed"],
                binding["fresh_host_observation"],
            )
            self.assertRegex(
                binding["binding_sha256"],
                r"^sha256:[0-9a-f]{64}$",
            )
            self.assertFalse(binding["authenticated"])

            replaced_collectors = dict(collectors)
            replaced_collectors["sysctl"] = {
                **replaced_collectors["sysctl"],
                "sha256": "sha256:" + "9" * 64,
            }
            failed = validate_apple_single_preflights(
                **{
                    **kwargs,
                    "expected_collector_executables": replaced_collectors,
                    "minimum_units": 2,
                }
            )
            self.assertFalse(failed["ok"])
            joined = "\n".join(failed["errors"])
            self.assertIn("collector executable identities", joined)
            self.assertIn("capacity-only minimum is 2", joined)

            oversized_host = json.loads(json.dumps(host_report))
            oversized_host["identity"]["apple"]["memory_bytes"] = 512 * 1024**3
            oversized_host["match"]["observed"]["memory_bytes"] = 512 * 1024**3
            oversized = validate_apple_single_preflights(
                **{
                    **kwargs,
                    "host_report": oversized_host,
                    "fresh_host_identity": oversized_host["identity"],
                    "fresh_host_match": oversized_host["match"],
                }
            )
            self.assertFalse(oversized["ok"])
            self.assertIn(
                "configured memory bytes",
                "\n".join(oversized["errors"]),
            )

    def test_cli_does_not_launch_when_prerequisite_binding_fails(self) -> None:
        model_data = {
            "artifact": {
                "repository": "Qwen/Qwen3-30B-A3B",
                "revision": REVISION,
            },
            "runtime": {
                "max_architecture": "Qwen3MoeForCausalLM",
                "max_weight_encoding": "bfloat16",
            },
        }
        platform_data = {"vendor": "nvidia"}
        catalog = SimpleNamespace(
            catalog_sha256=CATALOG_SHA256,
            model=lambda _model_id: SimpleNamespace(
                model_id="qwen3-30b-a3b",
                profile_sha256=MODEL_PROFILE_SHA256,
                to_dict=lambda: model_data,
            ),
            platform=lambda _platform_id: SimpleNamespace(
                platform_id="nvidia-h100-sxm-80gb",
                profile_sha256=PLATFORM_PROFILE_SHA256,
                to_dict=lambda: platform_data,
            ),
        )
        failed_preflight = {
            "ok": False,
            "errors": ["host report does not match requested device"],
            "warnings": [],
        }
        file_record = self._file_record("fixture")
        stdout = StringIO()
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            out = root / "nvidia-generation.json"
            with (
                mock.patch.object(
                    cli_module,
                    "load_qualification_catalog",
                    return_value=catalog,
                ),
                mock.patch.object(
                    cli_module,
                    "compose_qualification_recipe",
                    return_value={
                        "lock": {
                            "capacity_estimate": {
                                "minimum_units": 1,
                            }
                        }
                    },
                ),
                mock.patch.object(
                    cli_module,
                    "executable_identity",
                    return_value={
                        "requested_argv0": "fixture",
                        "resolved_path": "/fixture",
                        "size_bytes": 1,
                        "sha256": "sha256:" + "3" * 64,
                        "authenticated": False,
                    },
                ),
                mock.patch.object(
                    cli_module,
                    "_collect_qualification_runtime_report",
                    return_value={
                        "max_executable": {
                            "requested_argv0": "fixture",
                            "resolved_path": "/fixture",
                            "size_bytes": 1,
                            "sha256": "sha256:" + "3" * 64,
                            "authenticated": False,
                        }
                    },
                ),
                mock.patch.object(
                    cli_module,
                    "load_json_evidence",
                    side_effect=[
                        ({}, file_record),
                        ({}, file_record),
                        ({}, file_record),
                    ],
                ),
                mock.patch.object(
                    cli_module,
                    "inspect_model_artifacts",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "collect_host_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "match_platform_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "validate_nvidia_single_preflights",
                    return_value=failed_preflight,
                ),
                mock.patch.object(
                    cli_module,
                    "run_max_generation_smoke",
                ) as generation,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli_main(
                    [
                        "recipe",
                        "run-nvidia-single",
                        "--model",
                        "qwen3-30b-a3b",
                        "--platform",
                        "nvidia-h100-sxm-80gb",
                        "--model-dir",
                        directory,
                        "--model-artifact-report",
                        "artifact.json",
                        "--host-report",
                        "host.json",
                        "--runtime-report",
                        "runtime.json",
                        "--max-command",
                        sys.executable,
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(2, exit_code, stderr.getvalue())
            generation.assert_not_called()
            envelope = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(envelope["evidence"])
            self.assertFalse(
                envelope["qualification"]["single_platform_bringup_passed"]
            )

    def test_apple_cli_blocks_unversioned_precision_conversion_before_launch(
        self,
    ) -> None:
        args = SimpleNamespace(
            model="qwen3-30b-a3b",
            platform="apple-m3-max-128",
        )
        with mock.patch.object(
            cli_module,
            "run_apple_silicon_moe_serving_smoke",
        ) as generation:
            with self.assertRaisesRegex(
                ValueError,
                "no versioned evidence input exists",
            ):
                cli_module._cmd_recipe_run_apple_single(args)
        generation.assert_not_called()

    def test_apple_cli_does_not_launch_when_prerequisite_binding_fails(
        self,
    ) -> None:
        model_data = {
            "artifact": {
                "repository": "Qwen/Qwen3-30B-A3B",
                "revision": REVISION,
            },
            "runtime": {
                "max_architecture": "Qwen3MoeForCausalLM",
                "max_weight_encoding": "bfloat16",
            },
        }
        platform_data = {
            "vendor": "apple",
            "identity": {"chip": "Apple M3 Max"},
            "capacity_policy": {"sizing_memory_bytes": 128 * 1024**3},
            "runtime": {"verification_status": "unverified"},
        }
        catalog = SimpleNamespace(
            catalog_sha256=CATALOG_SHA256,
            model=lambda _model_id: SimpleNamespace(
                model_id="qwen3-30b-a3b",
                profile_sha256=MODEL_PROFILE_SHA256,
                to_dict=lambda: model_data,
            ),
            platform=lambda _platform_id: SimpleNamespace(
                platform_id="apple-m3-max-128",
                profile_sha256=PLATFORM_PROFILE_SHA256,
                to_dict=lambda: platform_data,
            ),
        )
        failed_preflight = {
            "ok": False,
            "errors": ["fresh host observation does not match Apple M3 Max"],
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reports = []
            for name in ("artifact", "host", "runtime"):
                path = root / f"{name}.json"
                path.write_text("{}\n", encoding="utf-8")
                reports.append(path)
            out = root / "apple-generation.json"
            args = SimpleNamespace(
                model="qwen3-30b-a3b",
                platform="apple-m3-max-128",
                model_dir=directory,
                model_artifact_report=str(reports[0]),
                host_report=str(reports[1]),
                runtime_report=str(reports[2]),
                max_command=sys.executable,
                prompt=SMOKE_SENTINEL_PROMPT,
                max_new_tokens=8,
                top_k=1,
                timeout_s=10.0,
                out=str(out),
            )
            with (
                mock.patch.object(
                    cli_module,
                    "load_qualification_catalog",
                    return_value=catalog,
                ),
                mock.patch.object(
                    cli_module,
                    "compose_qualification_recipe",
                    return_value={
                        "lock": {
                            "capacity_estimate": {"minimum_units": 1},
                            "precision_contract": {
                                "conversion_or_custom_kernel_required": False
                            },
                        }
                    },
                ),
                mock.patch.object(
                    cli_module,
                    "_collect_qualification_runtime_report",
                    return_value={"max_executable": {"resolved_path": "/max"}},
                ),
                mock.patch.object(
                    cli_module,
                    "executable_identity",
                    return_value={"resolved_path": "/collector"},
                ),
                mock.patch.object(
                    cli_module,
                    "inspect_model_artifacts",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "collect_host_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "match_platform_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "validate_apple_single_preflights",
                    return_value=failed_preflight,
                ),
                mock.patch.object(
                    cli_module,
                    "run_apple_silicon_moe_serving_smoke",
                ) as generation,
                redirect_stdout(StringIO()),
            ):
                exit_code = cli_module._cmd_recipe_run_apple_single(args)
            self.assertEqual(2, exit_code)
            generation.assert_not_called()
            envelope = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(envelope["evidence"])
            self.assertFalse(
                envelope["qualification"]["single_platform_bringup_passed"]
            )

    def test_apple_cli_synthetic_runner_cannot_promote_physical_bringup(
        self,
    ) -> None:
        model_data = {
            "artifact": {
                "repository": "Qwen/Qwen3-30B-A3B",
                "revision": REVISION,
            },
            "runtime": {
                "max_architecture": "Qwen3MoeForCausalLM",
                "max_weight_encoding": "bfloat16",
            },
        }
        platform_data = {
            "vendor": "apple",
            "identity": {"chip": "Apple M3 Max"},
            "capacity_policy": {"sizing_memory_bytes": 128 * 1024**3},
            "runtime": {"verification_status": "unverified"},
        }
        catalog = SimpleNamespace(
            catalog_sha256=CATALOG_SHA256,
            model=lambda _model_id: SimpleNamespace(
                model_id="qwen3-30b-a3b",
                profile_sha256=MODEL_PROFILE_SHA256,
                to_dict=lambda: model_data,
            ),
            platform=lambda _platform_id: SimpleNamespace(
                platform_id="apple-m3-max-128",
                profile_sha256=PLATFORM_PROFILE_SHA256,
                to_dict=lambda: platform_data,
            ),
        )
        observed_host = {
            "vendor": "apple",
            "chip": "Apple M3 Max",
            "memory_bytes": 128 * 1024**3,
            "machine_name": "MacBook Pro",
            "model_identifier": "Mac15,8",
            "model_number": "fixture",
        }
        preflight = {
            "ok": True,
            "errors": [],
            "warnings": [],
            "fresh_host_observation": observed_host,
        }
        bound_max = str(Path(sys.executable).resolve())
        runtime_report = {
            "max_executable": {"resolved_path": bound_max},
            "observed": {"max_version": "MAX 26.7"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reports = []
            for name in ("artifact", "host", "runtime"):
                path = root / f"{name}.json"
                path.write_text("{}\n", encoding="utf-8")
                reports.append(path)
            out = root / "apple-generation.json"
            resolved_model_dir = str(root.resolve())
            stdout = (
                f"{SMOKE_SENTINEL}\n"
                "Prompt size: 10\n"
                "Output size: 7\n"
            )
            stderr = ""
            evidence = {
                "version": 1,
                "record_kind": "apple-silicon-moe-serving-smoke",
                "evidence_scope": "single-mac-max-real-moe-serving-smoke",
                "ok": True,
                "error": None,
                "model": {
                    "model_id": "Qwen/Qwen3-30B-A3B",
                    "model_path": resolved_model_dir,
                    "architecture": "Qwen3MoeForCausalLM",
                },
                "runtime": {
                    "backend": "max",
                    "mode": "max-generate",
                    "max_command": [bound_max],
                    "launch_argv": [
                        bound_max,
                        "generate",
                        "--model",
                        resolved_model_dir,
                        "--devices",
                        "gpu",
                        "--max-new-tokens",
                        "8",
                        "--top-k",
                        "1",
                        "--prompt",
                        SMOKE_SENTINEL_PROMPT,
                        "--quantization-encoding",
                        "bfloat16",
                    ],
                    "max_cwd": None,
                    "max_extra_args": [],
                    "max_version": "MAX 26.7",
                    "max_version_error": None,
                    "devices_requested": "gpu",
                    "quantization_encoding": "bfloat16",
                    "top_k": 1,
                    "max_length": None,
                    "allow_download": False,
                    "fornax_orchestrated": True,
                },
                "serving": {
                    "request": {
                        "model": "Qwen/Qwen3-30B-A3B",
                        "messages": [
                            {"role": "user", "content": SMOKE_SENTINEL_PROMPT}
                        ],
                        "max_new_tokens": 8,
                        "stream": False,
                    },
                    "generated_text": SMOKE_SENTINEL,
                    "response": {
                        "id": "fixture",
                        "object": "chat.completion",
                        "model": "Qwen/Qwen3-30B-A3B",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": SMOKE_SENTINEL,
                                },
                                "finish_reason": "length",
                            }
                        ],
                    },
                },
                "result": {
                    "returncode": 0,
                    "stdout": stdout,
                    "stdout_chars": len(stdout),
                    "stdout_text_sha256": (
                        "sha256:" + hashlib.sha256(stdout.encode()).hexdigest()
                    ),
                    "stderr": stderr,
                    "stderr_chars": 0,
                    "stderr_text_sha256": (
                        "sha256:" + hashlib.sha256(b"").hexdigest()
                    ),
                    "failure_signature": [],
                },
                "runner": {
                    "kind": "synthetic_injected_test_runner",
                    "physical_execution_eligible": False,
                    "authenticated": False,
                },
                "hardware": {
                    "chip": "Apple M3 Max",
                    "memory": "128 GB",
                    "model_name": "MacBook Pro",
                    "model_identifier": "Mac15,8",
                    "model_number": "fixture",
                },
                "environment": {},
                "claims": {},
                "note": "synthetic fixture",
            }

            def run_smoke(**kwargs: object) -> dict[str, object]:
                Path(str(kwargs["out"])).write_text(
                    json.dumps(evidence) + "\n",
                    encoding="utf-8",
                )
                return evidence

            args = SimpleNamespace(
                model="qwen3-30b-a3b",
                platform="apple-m3-max-128",
                model_dir=directory,
                model_artifact_report=str(reports[0]),
                host_report=str(reports[1]),
                runtime_report=str(reports[2]),
                max_command=sys.executable,
                prompt=SMOKE_SENTINEL_PROMPT,
                max_new_tokens=8,
                top_k=1,
                timeout_s=10.0,
                out=str(out),
            )
            with (
                mock.patch.object(
                    cli_module,
                    "load_qualification_catalog",
                    return_value=catalog,
                ),
                mock.patch.object(
                    cli_module,
                    "compose_qualification_recipe",
                    return_value={
                        "lock": {
                            "capacity_estimate": {"minimum_units": 1},
                            "precision_contract": {
                                "conversion_or_custom_kernel_required": False
                            },
                        }
                    },
                ),
                mock.patch.object(
                    cli_module,
                    "_collect_qualification_runtime_report",
                    return_value=runtime_report,
                ),
                mock.patch.object(
                    cli_module,
                    "executable_identity",
                    return_value={"resolved_path": bound_max},
                ),
                mock.patch.object(
                    cli_module,
                    "inspect_model_artifacts",
                    return_value={"ok": True},
                ),
                mock.patch.object(
                    cli_module,
                    "collect_host_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "match_platform_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "validate_apple_single_preflights",
                    return_value=preflight,
                ),
                mock.patch.object(
                    cli_module,
                    "run_apple_silicon_moe_serving_smoke",
                    side_effect=run_smoke,
                ) as generation,
                mock.patch.object(
                    cli_module,
                    "validate_apple_silicon_moe_serving_smoke_fixture",
                    return_value={"ok": True, "errors": [], "warnings": []},
                ) as fixture_validator,
                redirect_stdout(StringIO()),
            ):
                exit_code = cli_module._cmd_recipe_run_apple_single(args)
            self.assertEqual(2, exit_code)
            generation.assert_called_once()
            fixture_validator.assert_called_once_with(evidence)
            call = generation.call_args.kwargs
            self.assertEqual(resolved_model_dir, call["model_path"])
            self.assertEqual("gpu", call["devices"])
            self.assertFalse(call["allow_download"])
            self.assertEqual(SMOKE_SENTINEL_PROMPT, call["prompt"])
            envelope = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(
                envelope["qualification"]["single_platform_bringup_passed"]
            )
            self.assertTrue(envelope["host_rebind"]["ok"])
            self.assertFalse(
                envelope["generation_contract_rebind"][
                    "physical_runner_matched"
                ]
            )
            self.assertFalse(
                envelope["smoke_artifact_binding"]["ephemeral_path_retained"]
            )

    def test_cli_launches_uuid_masked_max_gpu_zero_after_binding(self) -> None:
        model_data = {
            "artifact": {
                "repository": "Qwen/Qwen3-30B-A3B",
                "revision": REVISION,
            },
            "runtime": {
                "max_architecture": "Qwen3MoeForCausalLM",
                "max_weight_encoding": "bfloat16",
            },
        }
        platform_data = {"vendor": "nvidia"}
        catalog = SimpleNamespace(
            catalog_sha256=CATALOG_SHA256,
            model=lambda _model_id: SimpleNamespace(
                model_id="qwen3-30b-a3b",
                profile_sha256=MODEL_PROFILE_SHA256,
                to_dict=lambda: model_data,
            ),
            platform=lambda _platform_id: SimpleNamespace(
                platform_id="nvidia-h100-sxm-80gb",
                profile_sha256=PLATFORM_PROFILE_SHA256,
                to_dict=lambda: platform_data,
            ),
        )
        preflight = {
            "ok": True,
            "errors": [],
            "warnings": [],
            "device_binding": {
                "physical_selector": "gpu:7",
                "nvidia_smi_index": 7,
                "nvidia_gpu_uuid": GPU_UUID,
                "cuda_visible_devices": GPU_UUID,
                "max_launch_device": "gpu:0",
                "verified": True,
            },
        }
        generation_binding = {
            "mode": "nvidia_gpu_uuid_to_visible_ordinal",
            "physical_nvidia_smi_index": 7,
            "physical_nvidia_gpu_uuid": GPU_UUID,
            "cuda_visible_devices": GPU_UUID,
            "max_device": "gpu:0",
            "applied_to_live_subprocess": True,
        }
        evidence = {
            "ok": True,
            "errors": [],
            "runner": {"device_binding": generation_binding},
            "observed": {
                "max_version": "MAX 26.7",
                "generated_text_signal": {
                    "detected": True,
                    "text_excerpt": SMOKE_SENTINEL,
                    "text_excerpt_truncated": False,
                },
            },
            "claims": {"single_platform_bringup_passed": True},
        }
        runtime_report = {
            "max_executable": {},
            "observed": {"max_version": "MAX 26.7"},
        }
        file_record = self._file_record("fixture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            out = root / "nvidia-generation.json"
            resolved_max = str(Path(sys.executable).resolve())
            max_executable = {
                "requested_argv0": resolved_max,
                "resolved_path": resolved_max,
                "size_bytes": 1,
                "sha256": "sha256:" + "3" * 64,
                "authenticated": False,
            }
            nvidia_smi_executable = {
                "requested_argv0": "nvidia-smi",
                "resolved_path": "/fixture/nvidia-smi",
                "size_bytes": 1,
                "sha256": "sha256:" + "4" * 64,
                "authenticated": False,
            }
            runtime_report["max_executable"] = max_executable
            evidence["inputs"] = {
                "max_argv_prefix": [resolved_max],
                "model_dir": str(root),
                "prompt": SMOKE_SENTINEL_PROMPT,
            }

            def executable(command: str) -> dict[str, object]:
                if command == "nvidia-smi":
                    return nvidia_smi_executable
                if command in {sys.executable, resolved_max}:
                    return max_executable
                raise AssertionError(f"unexpected executable: {command}")

            args = SimpleNamespace(
                model="qwen3-30b-a3b",
                platform="nvidia-h100-sxm-80gb",
                model_dir=str(root),
                model_artifact_report="artifact.json",
                host_report="host.json",
                runtime_report="runtime.json",
                max_command=sys.executable,
                device="gpu:7",
                prompt=SMOKE_SENTINEL_PROMPT,
                max_new_tokens=8,
                top_k=1,
                timeout_s=10.0,
                out=str(out),
            )
            with (
                mock.patch.object(
                    cli_module,
                    "load_qualification_catalog",
                    return_value=catalog,
                ),
                mock.patch.object(
                    cli_module,
                    "compose_qualification_recipe",
                    return_value={
                        "lock": {
                            "capacity_estimate": {
                                "minimum_units": 1,
                            }
                        }
                    },
                ),
                mock.patch.object(
                    cli_module,
                    "_collect_qualification_runtime_report",
                    return_value=runtime_report,
                ),
                mock.patch.object(
                    cli_module,
                    "executable_identity",
                    side_effect=executable,
                ),
                mock.patch.object(
                    cli_module,
                    "load_json_evidence",
                    side_effect=[
                        ({}, file_record),
                        ({}, file_record),
                        ({}, file_record),
                    ],
                ),
                mock.patch.object(
                    cli_module,
                    "inspect_model_artifacts",
                    return_value={"ok": True},
                ),
                mock.patch.object(
                    cli_module,
                    "collect_host_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "match_platform_identity",
                    return_value={},
                ),
                mock.patch.object(
                    cli_module,
                    "validate_nvidia_single_preflights",
                    return_value=preflight,
                ),
                mock.patch.object(
                    cli_module,
                    "run_max_generation_smoke",
                    return_value=evidence,
                ) as generation,
                mock.patch.object(
                    cli_module,
                    "validate_max_generation_smoke_evidence",
                    return_value={"ok": True, "errors": []},
                ),
                redirect_stdout(StringIO()),
            ):
                exit_code = cli_module._cmd_recipe_run_nvidia_single(args)
            self.assertEqual(0, exit_code)
            generation.assert_called_once()
            call = generation.call_args.kwargs
            self.assertEqual("gpu:0", call["device"])
            self.assertEqual(7, call["nvidia_smi_index"])
            self.assertEqual(GPU_UUID, call["nvidia_gpu_uuid"])
            envelope = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(
                envelope["device_rebind"]["matched_generation_evidence"]
            )
            self.assertTrue(envelope["model_rebind"]["path_matched"])
            self.assertTrue(envelope["continuity_rebind"]["ok"])
            self.assertTrue(
                envelope["runtime_rebind"]["executable_continuity_matched"]
            )
            self.assertTrue(
                envelope["generation_contract_rebind"]["sentinel_matched"]
            )
            self.assertEqual(
                "gpu:7",
                envelope["device_rebind"]["physical_selector"],
            )


if __name__ == "__main__":
    unittest.main()
