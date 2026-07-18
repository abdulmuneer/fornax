from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dependencies.reconstruct_max_lineage import reconstruct
from fornax.g2_validation import (
    PHYSICAL_STEPS,
    capture_fornax_source,
    run_g2_validation,
    validate_g2_run_manifest,
    validate_physical_step_result,
    verify_max_lineage,
)
import fornax.g2_validation as g2_validation


REMOTE = "https://github.com/modular/modular.git"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8").strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class G2ValidationTest(unittest.TestCase):
    def test_source_capture_is_bounded_to_execution_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            _git(root, "config", "user.email", "fornax-tests@example.invalid")
            _git(root, "config", "user.name", "Fornax Tests")
            (root / "fornax").mkdir()
            (root / "fornax" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs" / "large-unrelated.txt").write_text(
                "not execution source\n", encoding="utf-8"
            )
            _git(root, "add", "fornax/runtime.py")
            _git(root, "commit", "-q", "-m", "initial")

            snapshot = capture_fornax_source(root)
            self.assertTrue(snapshot["execution_source_clean"])
            self.assertFalse(snapshot["dirty"])
            self.assertEqual([], snapshot["status"])

            (root / "fornax" / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
            snapshot = capture_fornax_source(root)
            self.assertFalse(snapshot["execution_source_clean"])
            self.assertEqual([" M fornax/runtime.py"], snapshot["status"])

    def _fixture_repository(self, directory: Path) -> tuple[Path, Path, dict[str, Any]]:
        root = directory / "fornax-root"
        max_repo = root / "external" / "modular"
        max_repo.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(max_repo)], check=True)
        _git(max_repo, "config", "user.email", "fornax-tests@example.invalid")
        _git(max_repo, "config", "user.name", "Fornax Tests")
        _git(max_repo, "remote", "add", "origin", REMOTE)
        source = max_repo / "kernel.mojo"
        source.write_text("base\n", encoding="utf-8")
        _git(max_repo, "add", "kernel.mojo")
        _git(max_repo, "commit", "-q", "-m", "base")
        base = str(_git(max_repo, "rev-parse", "HEAD"))
        base_tree = str(_git(max_repo, "rev-parse", "HEAD^{tree}"))
        source.write_text("base\npatch\n", encoding="utf-8")
        _git(max_repo, "add", "kernel.mojo")
        _git(max_repo, "commit", "-q", "-m", "patch")
        patch = str(_git(max_repo, "rev-parse", "HEAD"))
        patch_tree = str(_git(max_repo, "rev-parse", "HEAD^{tree}"))
        commit_object = str(_git(max_repo, "cat-file", "-p", patch))
        headers, _, commit_message = commit_object.partition("\n\n")
        author_line = next(
            line for line in headers.splitlines() if line.startswith("author ")
        )
        committer_line = next(
            line for line in headers.splitlines() if line.startswith("committer ")
        )
        author_name, author_tail = author_line[len("author ") :].split(" <", 1)
        author_email, author_date = author_tail.split("> ", 1)
        committer_name, committer_tail = committer_line[len("committer ") :].split(
            " <", 1
        )
        committer_email, committer_date = committer_tail.split("> ", 1)
        diff = _git(max_repo, "diff", "--binary", base, patch, binary=True)
        assert isinstance(diff, bytes)
        diff_hash = "sha256:" + hashlib.sha256(diff).hexdigest()
        patch_relative = f"dependencies/max-patches/{patch}.diff"
        patch_path = root / patch_relative
        patch_path.parent.mkdir(parents=True)
        patch_path.write_bytes(diff)
        reconstruction_relative = "dependencies/reconstruct_max_lineage.py"
        reconstruction_path = root / reconstruction_relative
        reconstruction_path.write_text(
            "# synthetic reconstruction fixture\n", encoding="utf-8"
        )
        reconstruction_hash = (
            "sha256:" + hashlib.sha256(reconstruction_path.read_bytes()).hexdigest()
        )

        pin = {
            "schema_version": 1,
            "dependency": "modular-max",
            "repository": {
                "url": REMOTE,
                "remote": "origin",
                "checkout_path": "external/modular",
            },
            "lineage": {
                "upstream_base_commit": base,
                "upstream_base_tree": base_tree,
                "patch_commit": patch,
                "patch_tree": patch_tree,
                "patch_diff_sha256": diff_hash,
                "patch_file": patch_relative,
                "patch_series": [
                    {
                        "commit": patch,
                        "parent": base,
                        "tree": patch_tree,
                        "subject": "patch",
                        "diff_sha256": diff_hash,
                    }
                ],
                "reconstruction": {
                    "author_name": author_name,
                    "author_email": author_email,
                    "author_date": author_date,
                    "committer_name": committer_name,
                    "committer_email": committer_email,
                    "committer_date": committer_date,
                    "message": commit_message,
                },
                "reconstruction_script": reconstruction_relative,
                "reconstruction_script_sha256": reconstruction_hash,
            },
            "build": {
                "accepted_cli_version": "MAX test-build-1",
                "primary_target": "//max:test",
                "source_cli_path": "external/modular/bazel-bin/max",
            },
            "fetch_instructions": [
                {"argv": ["git", "clone", REMOTE, "external/modular"]}
            ],
        }
        pin_path = root / "dependencies" / "max-lineage.json"
        _write_json(pin_path, pin)

        subprocess.run(["git", "init", "-q", str(root)], check=True)
        _git(root, "config", "user.email", "fornax-tests@example.invalid")
        _git(root, "config", "user.name", "Fornax Tests")
        _git(root, "add", "dependencies")
        _git(root, "commit", "-q", "-m", "pin MAX")
        return root, pin_path, pin

    def _run_manifest(
        self, *, patch_commit: str, emitter: Path, manifest_path: Path
    ) -> dict[str, Any]:
        model = {
            "model_id": "fornax/test-model",
            "snapshot_id": "test-snapshot-1",
            "model_config_sha256": SHA_A,
            "weights_manifest_sha256": SHA_D,
            "tokenizer_sha256": SHA_B,
            "template_sha256": SHA_C,
            "prompt_corpus_sha256": SHA_D,
        }
        nodes = [
            {
                "role": "nvidia",
                "physical_host_id": "linux-node-01",
                "hostname": "linux-node-01.lab.invalid",
                "os_build": "Test Linux 1",
                "architecture": "x86_64",
                "device_identity": "test-nvidia-device-0",
                "driver_runtime": "test-cuda-driver-1",
                "max_cli_version": "MAX test-build-1",
                "mojo_version": "Mojo test-build-1",
                "bazel_version": "Bazel test-build-1",
                "bazelisk_version": "Bazelisk test-build-1",
                "python_version": "Python test-build-1",
                "compiler_version": "Compiler test-build-1",
                "toolchain_version": "CUDA toolchain test-build-1",
                "build_target": "//max:test",
                "build_flags_sha256": SHA_C,
                "build_environment_sha256": SHA_D,
                "max_patch_commit": patch_commit,
                "max_binary_sha256": SHA_A,
                "memory_bytes": 8589934592,
            },
            {
                "role": "apple",
                "physical_host_id": "apple-node-01",
                "hostname": "apple-node-01.lab.invalid",
                "os_build": "Test macOS 1",
                "architecture": "arm64",
                "device_identity": "test-apple-device-0",
                "driver_runtime": "test-metal-driver-1",
                "max_cli_version": "MAX test-build-1",
                "mojo_version": "Mojo test-build-1",
                "bazel_version": "Bazel test-build-1",
                "bazelisk_version": "Bazelisk test-build-1",
                "python_version": "Python test-build-1",
                "compiler_version": "Compiler test-build-1",
                "toolchain_version": "Metal toolchain test-build-1",
                "build_target": "//max:test",
                "build_flags_sha256": SHA_C,
                "build_environment_sha256": SHA_D,
                "max_patch_commit": patch_commit,
                "max_binary_sha256": SHA_B,
                "memory_bytes": 17179869184,
            },
        ]
        steps = {
            definition["step_id"]: {
                "status": "READY",
                "argv": [
                    sys.executable,
                    str(emitter),
                    str(manifest_path),
                    definition["step_id"],
                    "{step_dir}",
                ],
                "cwd": "{repo_root}",
                "timeout_seconds": 1801
                if definition["step_id"] == "V9_STABILITY"
                else 10,
                "result_artifact": "result.json",
            }
            for definition in PHYSICAL_STEPS
        }
        plan_id = str(uuid.uuid4())
        plan_stages = [
            {
                "stage_id": "stage-0",
                "stage_index": 0,
                "layer_start": 0,
                "layer_end": 1,
                "node_role": "nvidia",
                "physical_host_id": "linux-node-01",
            },
            {
                "stage_id": "stage-1",
                "stage_index": 1,
                "layer_start": 2,
                "layer_end": 3,
                "node_role": "apple",
                "physical_host_id": "apple-node-01",
            },
        ]
        evidence_rows = [
            ("model:test-v1", "model"),
            ("quantization:test-v1", "quantization"),
            ("expert-trace:test-v1", "expert_trace"),
            ("capability:nvidia-test-v1", "capability"),
            ("capability:apple-test-v1", "capability"),
            ("measurement:test-v1", "measurement"),
            ("calibration:test-v1", "calibration"),
            ("route:test-v1", "route"),
        ]
        registry_records = []
        registry_dir = manifest_path.parent / "planner-evidence"
        for index, (source_id, evidence_type) in enumerate(evidence_rows):
            evidence_path = registry_dir / f"record-{index}.json"
            _write_json(
                evidence_path,
                {"source_id": source_id, "evidence_type": evidence_type},
            )
            registry_records.append(
                {
                    "source_id": source_id,
                    "evidence_type": evidence_type,
                    "artifact_path": evidence_path.name,
                    "artifact_sha256": hashlib.sha256(
                        evidence_path.read_bytes()
                    ).hexdigest(),
                    "status": "active",
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            )
        registry_path = registry_dir / "registry.json"
        _write_json(
            registry_path,
            {
                "schema_version": "fornax.planner-evidence-registry.v1",
                "records": registry_records,
            },
        )
        registry_hash = "sha256:" + hashlib.sha256(registry_path.read_bytes()).hexdigest()
        plan_artifact = {
            "schema_version": 1,
            "plan_id": plan_id,
            "model": model,
            "feasible": True,
            "authority": {
                "requested_mode": "deployment",
                "status": "deployment_authoritative",
                "deployment_authorized": True,
                "confidence": "high",
                "prediction_expected_relative_error": 0.1,
                "input_max_expected_relative_error": 0.1,
                "source_ids": [source_id for source_id, _ in evidence_rows],
                "evidence_registry_sha256": registry_hash,
                "reasons": ["all deployment authority checks passed"],
            },
            "stages": plan_stages,
            "frozen_predictions": [
                {
                    "inflight": value,
                    "aggregate_tokens_per_second": float(value) * 0.9,
                }
                for value in (1, 4, 8)
            ],
        }
        plan_artifact_path = manifest_path.parent / "plan-artifact.json"
        _write_json(plan_artifact_path, plan_artifact)
        plan_hash = "sha256:" + hashlib.sha256(plan_artifact_path.read_bytes()).hexdigest()
        stage_paths: list[str] = []
        stage_hashes: list[str] = []
        for stage in plan_stages:
            role = stage["node_role"]
            node = next(node for node in nodes if node["role"] == role)
            stage_artifact = {
                "manifest_version": 1,
                "model_id": model["model_id"],
                "model_snapshot": model["snapshot_id"],
                "model_config_hash": model["model_config_sha256"],
                "tokenizer_hash": model["tokenizer_sha256"],
                "template_hash": model["template_sha256"],
                "plan_id": plan_id,
                "plan_hash": plan_hash,
                "stage_id": stage["stage_id"],
                "stage_index": stage["stage_index"],
                "layer_start": stage["layer_start"],
                "layer_end": stage["layer_end"],
                "node_binding": {
                    "role": role,
                    "physical_host_id": stage["physical_host_id"],
                },
                "max_patch_commit": patch_commit,
                "max_binary_sha256": node["max_binary_sha256"],
            }
            relative = f"stage-manifest-{stage['stage_index']}.json"
            stage_path = manifest_path.parent / relative
            _write_json(stage_path, stage_artifact)
            stage_paths.append(relative)
            stage_hashes.append(
                "sha256:" + hashlib.sha256(stage_path.read_bytes()).hexdigest()
            )
        return {
            "schema_version": 1,
            "model": model,
            "plan": {
                "plan_id": plan_id,
                "plan_hash": plan_hash,
                "plan_artifact": "plan-artifact.json",
                "evidence_registry_artifact": "planner-evidence/registry.json",
                "stage_manifest_sha256": stage_hashes,
                "stage_manifest_artifacts": stage_paths,
            },
            "nodes": nodes,
            "network": {
                "source_host_id": "linux-node-01",
                "destination_host_id": "apple-node-01",
                "route": "isolated-test-lan",
                "interface": "test0",
                "mtu_bytes": 1500,
                "declared_link_bits_per_second": 100000000000,
            },
            "correctness": {
                "reference": {
                    "reference_id": "reference:test-v1",
                    "implementation": "synthetic-cpu-reference-v1",
                    "artifact_sha256": SHA_A,
                },
                "tolerance_policy": {
                    "approval_id": "approval:test-v1",
                    "dtype_tolerances": {
                        "bf16": {"atol": 0.01, "rtol": 0.01}
                    },
                    "nonfinite": "reject",
                    "top1": "exact",
                    "routing": "exact",
                },
                "corpus": {
                    "prompt_count": 20,
                    "context_tokens": [16, 128, 512, 4096],
                    "generated_tokens_per_prompt": 128,
                },
            },
            "stability": {
                "duration_seconds": 1800,
                "sample_interval_seconds": 60,
                "target_inflight": 8,
                "minimum_completed_requests": 20,
                "post_drain_timeout_seconds": 30,
            },
            "steps": steps,
        }

    @staticmethod
    def _passing_prerequisites() -> list[dict[str, Any]]:
        return [
            {
                "step_id": "TEST_T0_T1",
                "tier": "T0/T1-contract",
                "title": "synthetic prerequisite for runner orchestration",
                "argv": [sys.executable, "-c", "print('pass')"],
            }
        ]

    def _emitted_step(
        self, base: Path, step_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], Path, str, dict[str, Any]]:
        manifest_path = base / "manifest.json"
        emitter = Path(__file__).with_name("g2_fixture_emitter.py").resolve()
        manifest = self._run_manifest(
            patch_commit="1" * 40,
            emitter=emitter,
            manifest_path=manifest_path,
        )
        _write_json(manifest_path, manifest)
        step_dir = base / "step"
        nonce = str(uuid.uuid4())
        environment = dict(os.environ)
        environment["FORNAX_G2_RUN_NONCE"] = nonce
        subprocess.run(
            [
                sys.executable,
                str(emitter),
                str(manifest_path),
                step_id,
                str(step_dir),
            ],
            check=True,
            env=environment,
        )
        validation = validate_g2_run_manifest(
            manifest, pinned_max_commit="1" * 40, manifest_base=base
        )
        self.assertTrue(validation["ok"], validation["errors"])
        result = json.loads((step_dir / "result.json").read_text(encoding="utf-8"))
        return manifest, result, step_dir, nonce, validation

    @staticmethod
    def _rebind_plan_artifacts(base: Path, manifest: dict[str, Any]) -> None:
        plan_path = base / manifest["plan"]["plan_artifact"]
        plan_hash = "sha256:" + hashlib.sha256(plan_path.read_bytes()).hexdigest()
        manifest["plan"]["plan_hash"] = plan_hash
        for index, relative in enumerate(manifest["plan"]["stage_manifest_artifacts"]):
            stage_path = base / relative
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage["plan_hash"] = plan_hash
            _write_json(stage_path, stage)
            manifest["plan"]["stage_manifest_sha256"][index] = (
                "sha256:" + hashlib.sha256(stage_path.read_bytes()).hexdigest()
            )

    def test_lineage_pin_is_verified_against_checkout_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, pin_path, pin = self._fixture_repository(Path(directory))
            report = verify_max_lineage(root, pin_path)
            self.assertTrue(report["ok"], report["errors"])
            self.assertTrue(all(row["status"] == "PASSED" for row in report["checks"]))

            pin["lineage"]["patch_diff_sha256"] = SHA_A
            bad_path = Path(directory) / "tampered-pin.json"
            _write_json(bad_path, pin)
            failed = verify_max_lineage(root, bad_path)
            self.assertFalse(failed["ok"])
            self.assertTrue(
                any("patch-diff-hash" in error for error in failed["errors"])
            )

    def test_tracked_diff_reconstructs_the_exact_pinned_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, pin_path, pin = self._fixture_repository(base)
            result = reconstruct(
                repository_root=root,
                checkout=base / "reconstructed-max",
                manifest_path=pin_path,
                source_repository=str(root / "external" / "modular"),
            )
            self.assertTrue(result["ok"])
            self.assertEqual(pin["lineage"]["patch_commit"], result["patch_commit"])
            self.assertEqual(pin["lineage"]["patch_tree"], result["patch_tree"])
            self.assertTrue(result["used_local_object_source"])

    def test_readiness_run_records_blocked_physical_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, pin_path, _ = self._fixture_repository(base)
            bundle = base / "readiness-bundle"
            result = run_g2_validation(
                repo_root=root,
                out_dir=bundle,
                max_lineage_manifest=pin_path,
                prerequisite_commands=self._passing_prerequisites(),
            )
            self.assertTrue(result["summary"]["max_lineage_passed"])
            self.assertTrue(result["summary"]["t0_t1_prerequisites_passed"])
            self.assertFalse(result["summary"]["technical_g2_packet_passed"])
            self.assertEqual(
                {"BLOCKED"}, {row["status"] for row in result["physical_steps"]}
            )
            self.assertTrue((bundle / "g2-evidence.json").is_file())
            self.assertTrue((bundle / "g2-summary.md").is_file())
            manifest = json.loads(
                (bundle / "artifact-manifest.json").read_text(encoding="utf-8")
            )
            paths = {row["path"] for row in manifest["files"]}
            self.assertIn("g2-evidence.json", paths)
            self.assertIn("inputs/max-lineage.json", paths)

    def test_complete_synthetic_packet_passes_only_with_runner_observed_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, pin_path, pin = self._fixture_repository(base)
            run_manifest_path = base / "g2-run-manifest.json"
            emitter = Path(__file__).with_name("g2_fixture_emitter.py").resolve()
            run_manifest = self._run_manifest(
                patch_commit=pin["lineage"]["patch_commit"],
                emitter=emitter,
                manifest_path=run_manifest_path,
            )
            _write_json(run_manifest_path, run_manifest)
            real_run_process = g2_validation._run_process

            def test_clock_process(*args, **kwargs):  # type: ignore[no-untyped-def]
                process = real_run_process(*args, **kwargs)
                environment = kwargs.get("environment") or {}
                if environment.get("FORNAX_G2_STEP_ID") == "V9_STABILITY":
                    process["elapsed_seconds"] = 1801.0
                return process

            with patch(
                "fornax.g2_validation._run_process", side_effect=test_clock_process
            ):
                result = run_g2_validation(
                    repo_root=root,
                    out_dir=base / "physical-bundle",
                    max_lineage_manifest=pin_path,
                    run_manifest_path=run_manifest_path,
                    run_physical=True,
                    prerequisite_commands=self._passing_prerequisites(),
                )
            self.assertTrue(result["summary"]["technical_g2_packet_passed"])
            self.assertEqual("PASS_PENDING_GATE_REVIEW", result["summary"]["status"])
            self.assertEqual(
                {"PASSED"}, {row["status"] for row in result["physical_steps"]}
            )
            self.assertFalse(result["claim_boundary"]["formal_g2_closed"])
            self.assertFalse(result["summary"]["gate_decision_authority"])

    def test_declared_thirty_minutes_cannot_replace_observed_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, pin_path, pin = self._fixture_repository(base)
            run_manifest_path = base / "g2-run-manifest.json"
            emitter = Path(__file__).with_name("g2_fixture_emitter.py").resolve()
            _write_json(
                run_manifest_path,
                self._run_manifest(
                    patch_commit=pin["lineage"]["patch_commit"],
                    emitter=emitter,
                    manifest_path=run_manifest_path,
                ),
            )
            result = run_g2_validation(
                repo_root=root,
                out_dir=base / "short-physical-bundle",
                max_lineage_manifest=pin_path,
                run_manifest_path=run_manifest_path,
                run_physical=True,
                prerequisite_commands=self._passing_prerequisites(),
            )
            stability = next(
                row
                for row in result["physical_steps"]
                if row["step_id"] == "V9_STABILITY"
            )
            self.assertEqual("FAILED", stability["status"])
            self.assertIn("runner-observed", stability["reason"])
            self.assertFalse(result["summary"]["technical_g2_packet_passed"])

    def test_summary_booleans_cannot_override_raw_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest_path = base / "manifest.json"
            emitter = Path(__file__).with_name("g2_fixture_emitter.py").resolve()
            manifest = self._run_manifest(
                patch_commit="1" * 40,
                emitter=emitter,
                manifest_path=manifest_path,
            )
            _write_json(manifest_path, manifest)
            step_dir = base / "step"
            nonce = str(uuid.uuid4())
            environment = dict(os.environ)
            environment["FORNAX_G2_RUN_NONCE"] = nonce
            subprocess.run(
                [
                    sys.executable,
                    str(emitter),
                    str(manifest_path),
                    "V8_LOAD_CALIBRATION",
                    str(step_dir),
                ],
                check=True,
                env=environment,
            )
            raw_path = step_dir / "raw-measurements.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            for row in raw["observations"]:
                if row.get("kind") == "concurrency" and row.get("inflight") == 8:
                    row["aggregate_tokens_per_second"] = 0.5
            _write_json(raw_path, raw)
            result_data = json.loads(
                (step_dir / "result.json").read_text(encoding="utf-8")
            )
            manifest_validation = validate_g2_run_manifest(
                manifest, pinned_max_commit="1" * 40, manifest_base=base
            )
            self.assertTrue(manifest_validation["ok"], manifest_validation["errors"])
            errors = validate_physical_step_result(
                "V8_LOAD_CALIBRATION",
                result_data,
                manifest,
                pinned_max_commit="1" * 40,
                artifact_base=step_dir,
                observed_process_elapsed_seconds=1,
                expected_nonce=nonce,
                bound_inputs=manifest_validation["bound_inputs"],
            )
            self.assertTrue(
                any("not derived" in error for error in errors), errors
            )

    def test_plan_artifact_authority_is_semantically_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest_path = base / "manifest.json"
            emitter = Path(__file__).with_name("g2_fixture_emitter.py").resolve()
            manifest = self._run_manifest(
                patch_commit="1" * 40,
                emitter=emitter,
                manifest_path=manifest_path,
            )
            plan_path = base / manifest["plan"]["plan_artifact"]
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["authority"] = {
                "requested_mode": "exploratory",
                "status": "exploratory",
                "deployment_authorized": False,
            }
            _write_json(plan_path, plan)
            self._rebind_plan_artifacts(base, manifest)
            validation = validate_g2_run_manifest(
                manifest, pinned_max_commit="1" * 40, manifest_base=base
            )
            self.assertFalse(validation["ok"])
            self.assertTrue(
                any("deployment-authoritative" in error for error in validation["errors"]),
                validation["errors"],
            )

    def test_evidence_registry_missing_stale_and_tampered_artifacts_fail(self) -> None:
        for mode in ("missing", "stale", "tampered"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                manifest_path = base / "manifest.json"
                emitter = Path(__file__).with_name("g2_fixture_emitter.py").resolve()
                manifest = self._run_manifest(
                    patch_commit="1" * 40,
                    emitter=emitter,
                    manifest_path=manifest_path,
                )
                registry_path = base / manifest["plan"][
                    "evidence_registry_artifact"
                ]
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                if mode == "tampered":
                    artifact = registry_path.parent / registry["records"][0][
                        "artifact_path"
                    ]
                    artifact.write_text("tampered\n", encoding="utf-8")
                else:
                    if mode == "missing":
                        registry["records"][0]["artifact_path"] = "missing.json"
                    else:
                        registry["records"][0]["expires_at"] = (
                            "2000-01-01T00:00:00Z"
                        )
                    _write_json(registry_path, registry)
                    plan_path = base / manifest["plan"]["plan_artifact"]
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    plan["authority"]["evidence_registry_sha256"] = (
                        "sha256:"
                        + hashlib.sha256(registry_path.read_bytes()).hexdigest()
                    )
                    _write_json(plan_path, plan)
                    self._rebind_plan_artifacts(base, manifest)
                validation = validate_g2_run_manifest(
                    manifest, pinned_max_commit="1" * 40, manifest_base=base
                )
                self.assertFalse(validation["ok"])
                joined = " ".join(validation["errors"])
                if mode == "missing":
                    self.assertIn("missing", joined)
                elif mode == "stale":
                    self.assertIn("stale", joined)
                else:
                    self.assertIn("SHA-256 mismatch", joined)

    def test_preapproved_top1_policy_cannot_be_overridden_by_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, result, step_dir, nonce, validation = self._emitted_step(
                Path(directory), "V6_NVIDIA"
            )
            raw_path = step_dir / "raw-measurements.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw["observations"][0]["observed_top1"] = 99
            _write_json(raw_path, raw)
            errors = validate_physical_step_result(
                "V6_NVIDIA",
                result,
                manifest,
                pinned_max_commit="1" * 40,
                artifact_base=step_dir,
                observed_process_elapsed_seconds=1,
                expected_nonce=nonce,
                bound_inputs=validation["bound_inputs"],
            )
            self.assertTrue(any("numerical summary" in error for error in errors), errors)

    def test_stability_rejects_float_counters_and_unclean_post_drain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, result, step_dir, nonce, validation = self._emitted_step(
                Path(directory), "V9_STABILITY"
            )
            raw_path = step_dir / "raw-measurements.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw["observations"][0]["queue_depth"] = 8.0
            raw["observations"][-1]["kv_bytes"] = 1
            _write_json(raw_path, raw)
            errors = validate_physical_step_result(
                "V9_STABILITY",
                result,
                manifest,
                pinned_max_commit="1" * 40,
                artifact_base=step_dir,
                observed_process_elapsed_seconds=1801,
                expected_nonce=nonce,
                bound_inputs=validation["bound_inputs"],
            )
            self.assertTrue(any("must be an integer" in error for error in errors), errors)
            self.assertTrue(any("post-drain" in error for error in errors), errors)

    def test_fault_outcome_cannot_hide_duplicate_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, result, step_dir, nonce, validation = self._emitted_step(
                Path(directory), "V10_FAILURES"
            )
            raw_path = step_dir / "raw-measurements.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            link_loss = next(
                row
                for row in raw["observations"]
                if row.get("name") == "link_loss"
            )
            link_loss["replay_execution_count"] = 1
            _write_json(raw_path, raw)
            errors = validate_physical_step_result(
                "V10_FAILURES",
                result,
                manifest,
                pinned_max_commit="1" * 40,
                artifact_base=step_dir,
                observed_process_elapsed_seconds=1,
                expected_nonce=nonce,
                bound_inputs=validation["bound_inputs"],
            )
            self.assertTrue(any("faults summary" in error for error in errors), errors)

    def test_ready_physical_commands_are_not_run_without_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, pin_path, pin = self._fixture_repository(base)
            run_manifest_path = base / "g2-run-manifest.json"
            emitter = Path(__file__).with_name("g2_fixture_emitter.py").resolve()
            run_manifest = self._run_manifest(
                patch_commit=pin["lineage"]["patch_commit"],
                emitter=emitter,
                manifest_path=run_manifest_path,
            )
            _write_json(run_manifest_path, run_manifest)
            result = run_g2_validation(
                repo_root=root,
                out_dir=base / "not-run-bundle",
                max_lineage_manifest=pin_path,
                run_manifest_path=run_manifest_path,
                run_physical=False,
                prerequisite_commands=self._passing_prerequisites(),
            )
            self.assertEqual(
                {"NOT_RUN"}, {row["status"] for row in result["physical_steps"]}
            )
            self.assertTrue(
                all(row["process"] is None for row in result["physical_steps"])
            )
            self.assertFalse((base / "not-run-bundle" / "physical").exists())

    def test_manifest_rejects_same_host_and_placeholder_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "manifest.json"
            manifest = self._run_manifest(
                patch_commit="1" * 40,
                emitter=Path("emitter.py"),
                manifest_path=path,
            )
            manifest["nodes"][1]["physical_host_id"] = "linux-node-01"
            manifest["nodes"][1]["device_identity"] = "TBD"
            validation = validate_g2_run_manifest(
                manifest, pinned_max_commit="1" * 40, manifest_base=base
            )
            self.assertFalse(validation["ok"])
            self.assertTrue(
                any(
                    "different physical_host_id" in error
                    for error in validation["errors"]
                )
            )
            self.assertTrue(
                any("device_identity" in error for error in validation["errors"])
            )

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, pin_path, _ = self._fixture_repository(base)
            existing = base / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_g2_validation(
                    repo_root=root,
                    out_dir=existing,
                    max_lineage_manifest=pin_path,
                    prerequisite_commands=self._passing_prerequisites(),
                )


if __name__ == "__main__":
    unittest.main()
