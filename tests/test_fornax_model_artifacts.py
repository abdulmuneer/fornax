from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fornax.model_artifacts as model_artifacts_module
from fornax.cli import _apply_remote_code_review, main as cli_main
from fornax.model_artifacts import (
    MAX_REPORTED_FILES,
    inspect_model_artifacts,
    validate_model_artifact_report,
)


REVISION_A = "a" * 40
REVISION_B = "b" * 40
CATALOG_SHA256 = "sha256:" + "c" * 64
PROFILE_SHA256 = "sha256:" + "d" * 64


class ModelArtifactInspectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name).resolve()

    @staticmethod
    def _config(*, hidden_size: int = 2048) -> dict[str, object]:
        return {
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": hidden_size,
            "max_position_embeddings": 40960,
            "model_type": "qwen3_moe",
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 48,
        }

    @staticmethod
    def _profile(
        *,
        revision: str | None = REVISION_A,
        estimated_weight_bytes: int | None = None,
        required_files: list[str] | None = None,
        trust_remote_code: bool = False,
    ) -> dict[str, object]:
        artifact: dict[str, object] = {
            "provider": "hugging_face",
            "repository": "Qwen/Qwen3-30B-A3B",
            "revision_kind": "git_commit",
            "weight_format": "safetensors",
            "weight_dtype": "bf16",
            "quantization": "none",
            "trust_remote_code_required": trust_remote_code,
            "required_files": required_files
            if required_files is not None
            else [
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        }
        if revision is not None:
            artifact["revision"] = revision
        if estimated_weight_bytes is not None:
            artifact["estimated_weight_bytes"] = estimated_weight_bytes
        return {
            "schema_version": 1,
            "kind": "fornax_model_qualification_profile",
            "model_id": "qwen3-30b-a3b",
            "artifact": artifact,
            "architecture": {
                "family": "qwen3_moe",
                "decoder_layers": 48,
                "hidden_size": 2048,
                "total_experts": 128,
                "active_experts_per_token": 8,
                "config_max_position_embeddings": 40960,
            },
        }

    def _write_metadata(self, root: Path, *, hidden_size: int = 2048) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(
            json.dumps(self._config(hidden_size=hidden_size)),
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

    def _write_indexed_weights(
        self,
        root: Path,
        *,
        shard_payloads: dict[str, bytes] | None = None,
        weight_map: dict[str, str] | None = None,
    ) -> int:
        payloads = shard_payloads or {
            "model-00001-of-00002.safetensors": b"first-shard",
            "model-00002-of-00002.safetensors": b"second-shard-is-longer",
        }
        for name, payload in payloads.items():
            (root / name).write_bytes(payload)
        mapping = weight_map or {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.experts.0.weight": (
                "model-00002-of-00002.safetensors"
            ),
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 123456789},
                    "weight_map": mapping,
                }
            ),
            encoding="utf-8",
        )
        return sum(len(payload) for payload in payloads.values())

    def _write_passing_snapshot(self, root: Path | None = None) -> tuple[Path, int]:
        target = root or self.root
        self._write_metadata(target)
        (target / ".fornax-revision").write_text(REVISION_A + "\n", encoding="utf-8")
        return target, self._write_indexed_weights(target)

    def _pin_selected_file_hashes(
        self,
        root: Path,
        profile: dict[str, object],
    ) -> None:
        initial = inspect_model_artifacts(root, profile)
        self.assertTrue(initial["ok"], initial["errors"])
        artifact = profile["artifact"]
        self.assertIsInstance(artifact, dict)
        artifact["file_hashes"] = {
            row["path"]: row["sha256"] for row in initial["files"]
        }

    @staticmethod
    def _write_hf_local_metadata(
        root: Path,
        artifact_paths: list[str],
        *,
        revision: str = REVISION_A,
    ) -> None:
        timestamp = time.time()
        metadata_root = root / ".cache" / "huggingface" / "download"
        for path in artifact_paths:
            metadata_path = metadata_root / Path(path + ".metadata")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                f"{revision}\netag-{path}\n{timestamp}\n",
                encoding="utf-8",
            )

    def _write_remote_code_review(
        self,
        *,
        model_id: str = "qwen3-30b-a3b",
        revision: str = REVISION_A,
        decision: str = "allow_pinned_sha256",
        files: dict[str, str] | None = None,
        review_id: str = "review-qwen-synthetic-1",
        suffix: str = "valid",
    ) -> Path:
        review_path = self.root / f"remote-code-review-{suffix}.json"
        review_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_kind": "fornax_remote_code_review",
                    "review_id": review_id,
                    "model_id": model_id,
                    "revision": revision,
                    "decision": decision,
                    "files": files
                    if files is not None
                    else {"configuration_qwen.py": "sha256:" + "1" * 64},
                }
            ),
            encoding="utf-8",
        )
        return review_path

    @staticmethod
    def _review_digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def test_qwen_like_indexed_checkpoint_passes_offline_inspection(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        profile = self._profile(estimated_weight_bytes=weight_bytes)

        report = inspect_model_artifacts(root, profile)
        validation = validate_model_artifact_report(report)

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(2, report["weights"]["shard_count"])
        self.assertEqual(weight_bytes, report["weights"]["exact_shard_bytes"])
        self.assertEqual(123456789, report["weights"]["index_declared_tensor_bytes"])
        self.assertNotEqual(
            report["weights"]["exact_shard_bytes"],
            report["weights"]["index_declared_tensor_bytes"],
        )
        self.assertEqual(7, report["summary"]["profile_check_count"])
        self.assertTrue(
            all(row["sha256"].startswith("sha256:") for row in report["files"])
        )

    def test_strict_hash_coverage_and_catalog_lineage_pass(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        profile = self._profile(estimated_weight_bytes=weight_bytes)
        self._pin_selected_file_hashes(root, profile)

        report = inspect_model_artifacts(
            root,
            profile,
            catalog_sha256=CATALOG_SHA256,
            profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )
        validation = validate_model_artifact_report(
            report,
            expected_catalog_sha256=CATALOG_SHA256,
            expected_profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(CATALOG_SHA256, report["catalog_sha256"])
        self.assertEqual(PROFILE_SHA256, report["profile_sha256"])
        self.assertTrue(report["hash_coverage"]["required_by_profile"])
        self.assertTrue(report["hash_coverage"]["required_by_inspection"])
        self.assertTrue(report["hash_coverage"]["complete"])
        self.assertEqual(
            len(report["files"]),
            report["hash_coverage"]["expected_file_count"],
        )

    def test_strict_hash_coverage_rejects_absent_or_incomplete_manifest(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        profile = self._profile(estimated_weight_bytes=weight_bytes)

        absent = inspect_model_artifacts(
            root,
            profile,
            catalog_sha256=CATALOG_SHA256,
            profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )
        self.assertFalse(absent["ok"])
        self.assertFalse(absent["hash_coverage"]["required_by_profile"])
        self.assertIn(
            "strict inspection requires complete pinned SHA-256 coverage",
            "\n".join(absent["errors"]),
        )

        self._pin_selected_file_hashes(root, profile)
        artifact = profile["artifact"]
        self.assertIsInstance(artifact, dict)
        file_hashes = artifact["file_hashes"]
        self.assertIsInstance(file_hashes, dict)
        file_hashes.pop("model-00002-of-00002.safetensors")
        incomplete = inspect_model_artifacts(
            root,
            profile,
            catalog_sha256=CATALOG_SHA256,
            profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )

        self.assertFalse(incomplete["ok"])
        self.assertIn(
            "selected artifacts lack pinned profile SHA-256 values",
            "\n".join(incomplete["errors"]),
        )
        self.assertEqual(
            ["model-00002-of-00002.safetensors"],
            incomplete["hash_coverage"]["unpinned_selected_files"],
        )

    def test_expected_file_hash_and_serialized_lineage_tampering_block(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        profile = self._profile(estimated_weight_bytes=weight_bytes)
        self._pin_selected_file_hashes(root, profile)
        artifact = profile["artifact"]
        self.assertIsInstance(artifact, dict)
        file_hashes = artifact["file_hashes"]
        self.assertIsInstance(file_hashes, dict)
        file_hashes["config.json"] = "sha256:" + "0" * 64

        mismatched = inspect_model_artifacts(
            root,
            profile,
            catalog_sha256=CATALOG_SHA256,
            profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )
        self.assertFalse(mismatched["ok"])
        self.assertIn("sha256 mismatch for config.json", "\n".join(mismatched["errors"]))

        file_hashes["config.json"] = next(
            row["sha256"]
            for row in mismatched["files"]
            if row["path"] == "config.json"
        )
        valid = inspect_model_artifacts(
            root,
            profile,
            catalog_sha256=CATALOG_SHA256,
            profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )
        self.assertTrue(valid["ok"], valid["errors"])
        tampered = copy.deepcopy(valid)
        tampered["catalog_sha256"] = "sha256:" + "e" * 64
        tampered["hash_coverage"]["required_by_inspection"] = False

        validation = validate_model_artifact_report(
            tampered,
            expected_catalog_sha256=CATALOG_SHA256,
            expected_profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )

        self.assertFalse(validation["ok"])
        errors = "\n".join(validation["errors"])
        self.assertIn(
            "catalog_sha256 does not match expected_catalog_sha256",
            errors,
        )
        self.assertIn(
            "hash_coverage.required_by_inspection must be true",
            errors,
        )

    def test_snapshot_path_provides_immutable_revision_without_profile_pin(self) -> None:
        snapshot = (
            self.root
            / "models--Qwen--Qwen3-30B-A3B"
            / "snapshots"
            / REVISION_A
        )
        root, _ = self._write_passing_snapshot(snapshot)
        (root / ".fornax-revision").unlink()
        profile = self._profile(revision=None)

        report = inspect_model_artifacts(root, profile)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(REVISION_A, report["revision"]["value"])
        self.assertEqual(
            "hugging_face_snapshot_path",
            report["revision"]["source"],
        )

    def test_profile_revision_alone_does_not_resolve_arbitrary_directory(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        (root / ".fornax-revision").unlink()

        report = inspect_model_artifacts(
            root,
            self._profile(estimated_weight_bytes=weight_bytes),
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["revision"]["resolved"])
        self.assertEqual(REVISION_A, report["revision"]["expected"])
        self.assertIsNone(report["revision"]["value"])
        self.assertIn(
            "no locally observed Hugging Face revision",
            "\n".join(report["errors"]),
        )

    def test_hf_local_dir_metadata_must_cover_every_selected_artifact(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        (root / ".fornax-revision").unlink()
        artifact_paths = [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        self._write_hf_local_metadata(root, artifact_paths)
        profile = self._profile(estimated_weight_bytes=weight_bytes)

        report = inspect_model_artifacts(root, profile)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            "hugging_face_local_dir_metadata",
            report["revision"]["source"],
        )

        missing_metadata = (
            root
            / ".cache"
            / "huggingface"
            / "download"
            / "model-00002-of-00002.safetensors.metadata"
        )
        missing_metadata.unlink()
        blocked = inspect_model_artifacts(root, profile)
        self.assertFalse(blocked["ok"])
        self.assertIn(
            "local metadata is missing",
            "\n".join(blocked["errors"]),
        )

    def test_revision_and_config_mismatches_block(self) -> None:
        snapshot = self.root / "snapshots" / REVISION_A
        root, weight_bytes = self._write_passing_snapshot(snapshot)
        profile = self._profile(
            revision=REVISION_B,
            estimated_weight_bytes=weight_bytes,
        )
        profile["architecture"]["hidden_size"] = 4096

        report = inspect_model_artifacts(root, profile)

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn("resolved revision mismatch", errors)
        self.assertIn("architecture.hidden_size", errors)

    def test_missing_required_tokenizer_and_shard_files_block(self) -> None:
        scenarios = ("required", "tokenizer", "shard")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                case_root = self.root / scenario
                root, weight_bytes = self._write_passing_snapshot(case_root)
                required_files = None
                if scenario == "required":
                    required_files = [
                        "config.json",
                        "generation_config.json",
                        "model.safetensors.index.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ]
                elif scenario == "tokenizer":
                    (root / "tokenizer.json").unlink()
                else:
                    (root / "model-00002-of-00002.safetensors").unlink()
                profile = self._profile(
                    estimated_weight_bytes=weight_bytes,
                    required_files=required_files,
                )

                report = inspect_model_artifacts(root, profile)

                self.assertFalse(report["ok"])
                errors = "\n".join(report["errors"])
                if scenario == "required":
                    self.assertIn("generation_config.json", errors)
                elif scenario == "tokenizer":
                    self.assertIn("tokenizer payload", errors)
                else:
                    self.assertIn(
                        "model-00002-of-00002.safetensors",
                        errors,
                    )

    def test_index_traversal_multiple_indexes_and_weight_size_mismatch_block(self) -> None:
        root = self.root / "traversal"
        self._write_metadata(root)
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 1},
                    "weight_map": {"model.weight": "../escape.safetensors"},
                }
            ),
            encoding="utf-8",
        )
        traversal_report = inspect_model_artifacts(root, self._profile())
        self.assertFalse(traversal_report["ok"])
        self.assertIn("unsafe", "\n".join(traversal_report["errors"]))

        root, weight_bytes = self._write_passing_snapshot(self.root / "multiple")
        (root / "alternate.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": "other.safetensors"}}),
            encoding="utf-8",
        )
        (root / "other.safetensors").write_bytes(b"other")
        multiple_report = inspect_model_artifacts(
            root,
            self._profile(estimated_weight_bytes=weight_bytes),
        )
        self.assertFalse(multiple_report["ok"])
        self.assertIn(
            "multiple top-level safetensors index",
            "\n".join(multiple_report["errors"]),
        )

        root, weight_bytes = self._write_passing_snapshot(self.root / "size")
        size_report = inspect_model_artifacts(
            root,
            self._profile(estimated_weight_bytes=weight_bytes + 1),
        )
        self.assertFalse(size_report["ok"])
        self.assertIn(
            "artifact.estimated_weight_bytes",
            "\n".join(size_report["errors"]),
        )

    def test_model_root_and_artifact_symlinks_fail_closed(self) -> None:
        real_root, weight_bytes = self._write_passing_snapshot(
            self.root / "real-model"
        )
        root_link = self.root / "model-link"
        root_link.symlink_to(real_root, target_is_directory=True)

        linked_root_report = inspect_model_artifacts(
            root_link,
            self._profile(estimated_weight_bytes=weight_bytes),
        )

        self.assertFalse(linked_root_report["ok"])
        self.assertIn(
            "no-follow directory",
            "\n".join(linked_root_report["errors"]),
        )

        escaped_config = self.root / "escaped-config.json"
        escaped_config.write_text(
            json.dumps(self._config()),
            encoding="utf-8",
        )
        (real_root / "config.json").unlink()
        (real_root / "config.json").symlink_to(escaped_config)

        linked_file_report = inspect_model_artifacts(
            real_root,
            self._profile(estimated_weight_bytes=weight_bytes),
        )

        self.assertFalse(linked_file_report["ok"])
        linked_errors = "\n".join(linked_file_report["errors"])
        self.assertIn("cannot read artifact file config.json", linked_errors)
        self.assertIn("no-follow regular file", linked_errors)

    def test_intermediate_directory_symlink_cannot_escape_model_root(self) -> None:
        root = self.root / "nested-link"
        self._write_metadata(root)
        (root / ".fornax-revision").write_text(
            REVISION_A + "\n",
            encoding="utf-8",
        )
        outside = self.root / "outside-weights"
        outside.mkdir()
        payload = b"outside-shard"
        (outside / "model.safetensors").write_bytes(payload)
        (root / "weights").symlink_to(outside, target_is_directory=True)
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.weight": "weights/model.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )

        report = inspect_model_artifacts(
            root,
            self._profile(estimated_weight_bytes=len(payload)),
        )

        self.assertFalse(report["ok"])
        self.assertIn(
            "cannot preflight artifact file weights/model.safetensors",
            "\n".join(report["errors"]),
        )
        self.assertEqual(0, report["weights"]["exact_shard_bytes"])

    def test_config_hash_and_parse_use_the_same_captured_bytes(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        original_config = (root / "config.json").read_bytes()
        original_digest = "sha256:" + hashlib.sha256(original_config).hexdigest()
        original_parser = model_artifacts_module._parse_json_object
        config_mutated = False

        def mutate_after_capture(
            value: bytes,
            *,
            field: str,
            blockers: list[str],
        ) -> dict[str, object] | None:
            nonlocal config_mutated
            if field == "config.json" and not config_mutated:
                config_mutated = True
                (root / "config.json").write_text(
                    json.dumps(self._config(hidden_size=4096)),
                    encoding="utf-8",
                )
            return original_parser(value, field=field, blockers=blockers)

        with mock.patch.object(
            model_artifacts_module,
            "_parse_json_object",
            side_effect=mutate_after_capture,
        ):
            report = inspect_model_artifacts(
                root,
                self._profile(estimated_weight_bytes=weight_bytes),
            )

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(config_mutated)
        self.assertEqual(
            2048,
            report["config"]["parsed"]["values"]["hidden_size"],
        )
        self.assertEqual(original_digest, report["config"]["sha256"])
        self.assertEqual(
            4096,
            json.loads((root / "config.json").read_text(encoding="utf-8"))[
                "hidden_size"
            ],
        )

    def test_weight_size_preflight_blocks_shard_hashing_on_mismatch(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        read_paths: list[str] = []
        original_reader = model_artifacts_module._ModelRootReader.read_regular_file

        def track_reads(
            reader: object,
            relative_path: str,
            **kwargs: object,
        ) -> tuple[int, str, bytes | None, object]:
            read_paths.append(relative_path)
            return original_reader(reader, relative_path, **kwargs)

        with mock.patch.object(
            model_artifacts_module._ModelRootReader,
            "read_regular_file",
            new=track_reads,
        ):
            report = inspect_model_artifacts(
                root,
                self._profile(estimated_weight_bytes=weight_bytes + 1),
            )

        self.assertFalse(report["ok"])
        self.assertFalse(
            any(path.endswith(".safetensors") for path in read_paths),
            read_paths,
        )
        self.assertIn(
            "during no-follow size preflight",
            "\n".join(report["errors"]),
        )

    def test_trust_remote_code_requires_policy_and_pinned_code_hashes(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        code = b"class QwenConfig: pass\n"
        (root / "configuration_qwen.py").write_bytes(code)
        required_files = [
            "config.json",
            "configuration_qwen.py",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        profile = self._profile(
            estimated_weight_bytes=weight_bytes,
            required_files=required_files,
            trust_remote_code=True,
        )

        blocked = inspect_model_artifacts(root, profile)

        self.assertFalse(blocked["ok"])
        blocked_errors = "\n".join(blocked["errors"])
        self.assertIn("no explicit resolved review/allowlist policy", blocked_errors)
        self.assertIn("no expected code files with sha256 hashes", blocked_errors)

        digest = "sha256:" + hashlib.sha256(code).hexdigest()
        profile["artifact"]["remote_code"] = {
            "policy": "pinned_sha256_allowlist_reviewed",
            "files": {"configuration_qwen.py": digest},
        }
        allowed = inspect_model_artifacts(root, profile)
        self.assertTrue(allowed["ok"], allowed["errors"])
        self.assertTrue(allowed["remote_code"]["ready"])

    def test_trust_remote_code_rejects_unreviewed_nested_python_file(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        code_files = {
            "configuration_qwen.py": (
                b"from remote_helpers.router import select_experts\n"
            ),
            "remote_helpers/__init__.py": b"",
            "remote_helpers/router.py": b"def select_experts(): return []\n",
        }
        for path, payload in code_files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        profile = self._profile(
            estimated_weight_bytes=weight_bytes,
            required_files=[
                "config.json",
                "configuration_qwen.py",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            trust_remote_code=True,
        )
        profile["artifact"]["remote_code"] = {
            "policy": "pinned_sha256_allowlist_reviewed",
            "files": {
                "configuration_qwen.py": (
                    "sha256:"
                    + hashlib.sha256(
                        code_files["configuration_qwen.py"]
                    ).hexdigest()
                )
            },
        }

        report = inspect_model_artifacts(root, profile)

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn(
            "discovered remote-code Python files lack pinned sha256 entries",
            errors,
        )
        self.assertIn("remote_helpers/__init__.py", errors)
        self.assertIn("remote_helpers/router.py", errors)
        evidence = {row["path"]: row for row in report["files"]}
        for path in code_files:
            self.assertIn(path, evidence)
            self.assertIn("remote_code", evidence[path]["roles"])

    def test_trust_remote_code_reviews_complete_recursive_python_inventory(
        self,
    ) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        code_files = {
            "configuration_qwen.py": (
                b"from remote_helpers.router import select_experts\n"
            ),
            "remote_helpers/__init__.py": b"",
            "remote_helpers/router.py": b"def select_experts(): return []\n",
        }
        for path, payload in code_files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        profile = self._profile(
            estimated_weight_bytes=weight_bytes,
            required_files=[
                "config.json",
                "configuration_qwen.py",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            trust_remote_code=True,
        )
        profile["artifact"]["remote_code"] = {
            "policy": "pinned_sha256_allowlist_reviewed",
            "files": {
                path: "sha256:" + hashlib.sha256(payload).hexdigest()
                for path, payload in code_files.items()
            },
        }
        self._pin_selected_file_hashes(root, profile)

        report = inspect_model_artifacts(
            root,
            profile,
            catalog_sha256=CATALOG_SHA256,
            profile_sha256=PROFILE_SHA256,
            require_complete_hash_coverage=True,
        )

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["remote_code"]["ready"])
        self.assertEqual(
            set(code_files),
            {
                row["path"]
                for row in report["remote_code"]["expected_files"]
            },
        )
        evidence = {row["path"]: row for row in report["files"]}
        for path in code_files:
            self.assertIn(path, evidence)
            self.assertIn("remote_code", evidence[path]["roles"])

    def test_remote_code_review_exact_schema_enables_synthetic_fixture(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        code = b"class QwenConfig: pass\n"
        (root / "configuration_qwen.py").write_bytes(code)
        digest = "sha256:" + hashlib.sha256(code).hexdigest()
        profile = self._profile(
            estimated_weight_bytes=weight_bytes,
            required_files=[
                "config.json",
                "configuration_qwen.py",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            trust_remote_code=True,
        )
        selected_paths = [
            "config.json",
            "configuration_qwen.py",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            *sorted(path.name for path in root.glob("*.safetensors")),
        ]
        profile["artifact"]["file_hashes"] = {
            path: "sha256:"
            + hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in selected_paths
        }
        review_path = self._write_remote_code_review(files={
            "configuration_qwen.py": digest
        })

        _apply_remote_code_review(
            profile,
            review_path,
            self._review_digest(review_path),
        )
        report = inspect_model_artifacts(root, profile)

        remote_code = profile["artifact"]["remote_code"]
        self.assertEqual(
            {"configuration_qwen.py": digest},
            remote_code["files"],
        )
        self.assertTrue(
            remote_code["policy"].startswith(
                "pinned_sha256_allowlist_operator_acknowledged:"
                "review-qwen-synthetic-1:sha256:"
            )
        )
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["remote_code"]["ready"])

    def test_recipe_inspect_model_cli_applies_remote_code_review(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        code = b"class QwenConfig: pass\n"
        (root / "configuration_qwen.py").write_bytes(code)
        digest = "sha256:" + hashlib.sha256(code).hexdigest()
        profile = self._profile(
            estimated_weight_bytes=weight_bytes,
            required_files=[
                "config.json",
                "configuration_qwen.py",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            trust_remote_code=True,
        )
        selected_paths = [
            "config.json",
            "configuration_qwen.py",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            *sorted(path.name for path in root.glob("*.safetensors")),
        ]
        profile["artifact"]["file_hashes"] = {
            path: "sha256:"
            + hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in selected_paths
        }
        review_path = self._write_remote_code_review(
            files={"configuration_qwen.py": digest},
            suffix="cli",
        )
        out = self.root / "inspection.json"
        profile_sha256 = "sha256:" + "a" * 64
        catalog = SimpleNamespace(
            catalog_sha256="sha256:" + "b" * 64,
            model=lambda model_id: SimpleNamespace(
                profile_sha256=profile_sha256,
                to_dict=lambda: copy.deepcopy(profile),
            )
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch("fornax.cli.load_qualification_catalog", return_value=catalog),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = cli_main(
                [
                    "recipe",
                    "inspect-model",
                    "--model",
                    "qwen3-30b-a3b",
                    "--model-dir",
                    str(root),
                    "--remote-code-review",
                    str(review_path),
                    "--expected-remote-code-review-sha256",
                    self._review_digest(review_path),
                    "--out",
                    str(out),
                ]
            )

        self.assertEqual(
            0,
            exit_code,
            stderr.getvalue()
            + stdout.getvalue()
            + (out.read_text(encoding="utf-8") if out.exists() else ""),
        )
        self.assertIn("PASS model artifacts", stdout.getvalue())
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(report["validation"]["ok"], report["validation"]["errors"])
        self.assertTrue(
            report["remote_code"]["policy"].startswith(
                "pinned_sha256_allowlist_operator_acknowledged:"
                "review-qwen-synthetic-1:sha256:"
            )
        )

    def test_remote_code_review_rejects_identity_and_decision_mismatches(self) -> None:
        profile = self._profile()
        cases = (
            (
                {"model_id": "different-model"},
                "model_id does not match",
            ),
            (
                {"revision": REVISION_B},
                "revision does not match",
            ),
            (
                {"decision": "deny"},
                "decision must be allow_pinned_sha256",
            ),
        )
        for index, (overrides, expected_error) in enumerate(cases):
            with self.subTest(overrides=overrides):
                review_path = self._write_remote_code_review(
                    suffix=f"identity-{index}",
                    **overrides,
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    _apply_remote_code_review(
                        copy.deepcopy(profile),
                        review_path,
                        self._review_digest(review_path),
                    )

    def test_remote_code_review_requires_matching_out_of_band_digest(self) -> None:
        profile = self._profile()
        review_path = self._write_remote_code_review(suffix="digest-anchor")

        with self.assertRaisesRegex(
            ValueError,
            "do not match the expected out-of-band digest",
        ):
            _apply_remote_code_review(
                profile,
                review_path,
                "sha256:" + "0" * 64,
            )

    def test_remote_code_review_rejects_unsafe_paths_and_invalid_hashes(self) -> None:
        profile = self._profile()
        path_cases = (
            "../configuration.py",
            "/configuration.py",
            "subdir//configuration.py",
            "subdir/../configuration.py",
            "configuration.json",
            "..\\configuration.py",
        )
        for index, path in enumerate(path_cases):
            with self.subTest(path=path):
                review_path = self._write_remote_code_review(
                    files={path: "sha256:" + "1" * 64},
                    suffix=f"path-{index}",
                )
                with self.assertRaisesRegex(ValueError, "safe relative Python paths"):
                    _apply_remote_code_review(
                        copy.deepcopy(profile),
                        review_path,
                        self._review_digest(review_path),
                    )

        digest_cases = (
            "1" * 64,
            "sha512:" + "1" * 64,
            "sha256:" + "1" * 63,
            "sha256:" + "z" * 64,
            "sha256:" + "A" * 64,
        )
        for index, digest in enumerate(digest_cases):
            with self.subTest(digest=digest):
                review_path = self._write_remote_code_review(
                    files={"configuration.py": digest},
                    suffix=f"digest-{index}",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "sha256:<64 lowercase hex>|SHA-256 hex must be lowercase",
                ):
                    _apply_remote_code_review(
                        copy.deepcopy(profile),
                        review_path,
                        self._review_digest(review_path),
                    )

    def test_remote_code_review_rejects_extra_or_missing_fields(self) -> None:
        profile = self._profile()
        valid_path = self._write_remote_code_review(suffix="schema-base")
        valid = json.loads(valid_path.read_text(encoding="utf-8"))
        cases = []
        missing = dict(valid)
        missing.pop("decision")
        cases.append(("missing", missing))
        extra = dict(valid)
        extra["reviewer"] = "not-in-schema"
        cases.append(("extra", extra))
        for suffix, review in cases:
            with self.subTest(suffix=suffix):
                review_path = self.root / f"remote-code-review-{suffix}.json"
                review_path.write_text(json.dumps(review), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "must contain exactly"):
                    _apply_remote_code_review(
                        copy.deepcopy(profile),
                        review_path,
                        self._review_digest(review_path),
                    )

    def test_validator_rejects_duplicate_files_and_summary_tampering(self) -> None:
        root, weight_bytes = self._write_passing_snapshot()
        report = inspect_model_artifacts(
            root,
            self._profile(estimated_weight_bytes=weight_bytes),
        )
        self.assertTrue(report["ok"], report["errors"])

        tampered = copy.deepcopy(report)
        tampered["files"].append(copy.deepcopy(tampered["files"][0]))
        tampered["summary"]["file_count"] += 1
        tampered["summary"]["exact_selected_file_bytes"] += 1

        validation = validate_model_artifact_report(tampered)

        self.assertFalse(validation["ok"])
        errors = "\n".join(validation["errors"])
        self.assertIn("path is duplicated", errors)
        self.assertIn("exact_selected_file_bytes does not match", errors)

    def test_bounded_file_report_blocks_oversized_shard_set(self) -> None:
        root = self.root / "many-shards"
        self._write_metadata(root)
        shard_count = MAX_REPORTED_FILES + 1
        weight_map: dict[str, str] = {}
        for index in range(shard_count):
            name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            (root / name).write_bytes(bytes([index % 251]))
            weight_map[f"model.layers.{index}.weight"] = name
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": shard_count}, "weight_map": weight_map}),
            encoding="utf-8",
        )

        report = inspect_model_artifacts(root, self._profile())

        self.assertFalse(report["ok"])
        self.assertLessEqual(len(report["files"]), MAX_REPORTED_FILES)
        self.assertIn(
            "safetensors shard count exceeds bounded report limit",
            "\n".join(report["errors"]),
        )


if __name__ == "__main__":
    unittest.main()
