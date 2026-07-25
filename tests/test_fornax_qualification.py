from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import fornax.qualification as qualification_module
from fornax.cli import main as cli_main
from fornax.hardware_identity import match_platform_identity
from fornax.qualification import (
    PHYSICAL_CLAIM_KEYS,
    QualificationCatalogError,
    canonical_sha256,
    compose_all_qualification_recipes,
    compose_qualification_recipe,
    load_qualification_catalog,
    materialize_qualification_recipe,
    qualification_catalog_root,
    verify_materialized_qualification_recipe,
)
from fornax.recipe_packet import verify_recipe_packet


EXPECTED_MINIMUM_UNITS = {
    "deepseek-r1--apple-m3-max-128": 8,
    "deepseek-r1--apple-m3-ultra-512": 2,
    "deepseek-r1--apple-m5-max-128": 8,
    "deepseek-r1--nvidia-h100-sxm-80gb": 11,
    "deepseek-r1--nvidia-rtx-4090-24gb": 35,
    "deepseek-r1--nvidia-rtx-5090-32gb": 26,
    "gpt-oss-120b--apple-m3-max-128": 1,
    "gpt-oss-120b--apple-m3-ultra-512": 1,
    "gpt-oss-120b--apple-m5-max-128": 1,
    "gpt-oss-120b--nvidia-h100-sxm-80gb": 1,
    "gpt-oss-120b--nvidia-rtx-4090-24gb": 4,
    "gpt-oss-120b--nvidia-rtx-5090-32gb": 3,
    "qwen3-30b-a3b--apple-m3-max-128": 1,
    "qwen3-30b-a3b--apple-m3-ultra-512": 1,
    "qwen3-30b-a3b--apple-m5-max-128": 1,
    "qwen3-30b-a3b--nvidia-h100-sxm-80gb": 1,
    "qwen3-30b-a3b--nvidia-rtx-4090-24gb": 4,
    "qwen3-30b-a3b--nvidia-rtx-5090-32gb": 3,
}

EXPECTED_MODEL_ARTIFACT_CONTRACTS = {
    "deepseek-r1": {
        "file_count": 170,
        "shard_count": 163,
        "max_architecture": "DeepseekV3ForCausalLM",
        "max_weight_encoding": "float8_e4m3fn",
    },
    "gpt-oss-120b": {
        "file_count": 22,
        "shard_count": 15,
        "max_architecture": "GptOssForCausalLM",
        "max_weight_encoding": "float4_e2m1fnx2",
    },
    "qwen3-30b-a3b": {
        "file_count": 23,
        "shard_count": 16,
        "max_architecture": "Qwen3MoeForCausalLM",
        "max_weight_encoding": "bfloat16",
    },
}


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli_main(argv)
    return result, stdout.getvalue(), stderr.getvalue()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return value


class QualificationCatalogTest(unittest.TestCase):
    def test_strict_catalog_loads_exact_three_by_six_cross_product(self) -> None:
        catalog = load_qualification_catalog()
        recipes = compose_all_qualification_recipes(catalog)

        self.assertEqual(3, len(catalog.models))
        self.assertEqual(6, len(catalog.platforms))
        self.assertEqual(18, len(recipes))
        self.assertEqual(18, len(catalog.recipe_ids()))
        self.assertEqual(
            catalog.recipe_ids(),
            tuple(recipe["recipe_id"] for recipe in recipes),
        )
        self.assertEqual(18, len(set(catalog.recipe_ids())))

    def test_strict_catalog_rejects_unknown_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "catalog"
            shutil.copytree(qualification_catalog_root(), copied)
            profile_path = copied / "models" / "qwen3-30b-a3b.json"
            profile = _read_json(profile_path)
            profile["unsupported_field"] = "must fail closed"
            profile_path.write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                QualificationCatalogError,
                r"unknown keys: unsupported_field",
            ):
                load_qualification_catalog(copied)

    def test_model_profiles_pin_complete_hash_and_max_runtime_contracts(self) -> None:
        catalog = load_qualification_catalog()

        for profile in catalog.models:
            expected = EXPECTED_MODEL_ARTIFACT_CONTRACTS[profile.model_id]
            data = profile.to_dict()
            artifact = data["artifact"]
            runtime = data["runtime"]
            hashes = artifact["file_hashes"]
            provenance = {
                row["field"]: row for row in data["provenance"]
            }
            source = next(
                row
                for row in data["sources"]
                if row["source_id"] == "modular-max-model-formats"
            )
            with self.subTest(model_id=profile.model_id):
                self.assertEqual(expected["file_count"], len(hashes))
                self.assertEqual(
                    expected["shard_count"],
                    sum(path.endswith(".safetensors") for path in hashes),
                )
                self.assertTrue(set(artifact["required_files"]).issubset(hashes))
                self.assertTrue(
                    all(
                        digest.startswith("sha256:") and len(digest) == 71
                        for digest in hashes.values()
                    )
                )
                self.assertEqual(
                    expected["max_architecture"],
                    runtime["max_architecture"],
                )
                self.assertEqual(
                    expected["max_weight_encoding"],
                    runtime["max_weight_encoding"],
                )
                self.assertEqual(
                    {
                        "runtime.max_architecture",
                        "runtime.max_weight_encoding",
                    },
                    set(source["supports"]),
                )
                for field in source["supports"]:
                    self.assertEqual(
                        "published_candidate_contract_unverified",
                        provenance[field]["status"],
                    )
                self.assertEqual(
                    "pinned_expected_sha256_unobserved",
                    data["qualification"]["artifact_hash_status"],
                )
                if profile.model_id == "gpt-oss-120b":
                    self.assertFalse(
                        any(
                            path.startswith(("metal/", "original/"))
                            for path in hashes
                        )
                    )

    def test_strict_catalog_rejects_incomplete_or_excluded_hash_manifests(self) -> None:
        cases = (
            ("missing-field", "qwen3-30b-a3b", "missing keys: file_hashes"),
            ("missing-required", "qwen3-30b-a3b", "missing required files: config.json"),
            ("invalid-digest", "qwen3-30b-a3b", "must be sha256:<64 lowercase hex>"),
            ("excluded-file", "gpt-oss-120b", "include excluded files: metal/"),
        )
        for case, model_id, expected_error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "catalog"
                shutil.copytree(qualification_catalog_root(), copied)
                profile_path = copied / "models" / f"{model_id}.json"
                profile = _read_json(profile_path)
                hashes = profile["artifact"]["file_hashes"]
                if case == "missing-field":
                    profile["artifact"].pop("file_hashes")
                elif case == "missing-required":
                    hashes.pop("config.json")
                elif case == "invalid-digest":
                    hashes["config.json"] = "sha256:" + "A" * 64
                else:
                    hashes["metal/model.safetensors"] = "sha256:" + "0" * 64
                profile_path.write_text(
                    json.dumps(profile, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    QualificationCatalogError,
                    expected_error,
                ):
                    load_qualification_catalog(copied)

    def test_strict_catalog_rejects_missing_or_wrong_max_runtime_contract(self) -> None:
        cases = (
            ("missing", None, "missing keys: max_architecture"),
            ("wrong", "LlamaForCausalLM", "runtime MAX contract must be"),
        )
        for case, architecture, expected_error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "catalog"
                shutil.copytree(qualification_catalog_root(), copied)
                profile_path = copied / "models" / "qwen3-30b-a3b.json"
                profile = _read_json(profile_path)
                if architecture is None:
                    profile["runtime"].pop("max_architecture")
                else:
                    profile["runtime"]["max_architecture"] = architecture
                profile_path.write_text(
                    json.dumps(profile, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    QualificationCatalogError,
                    expected_error,
                ):
                    load_qualification_catalog(copied)

    def test_catalog_and_recipe_hashes_are_deterministic(self) -> None:
        first_catalog = load_qualification_catalog()
        second_catalog = load_qualification_catalog()
        first = compose_all_qualification_recipes(first_catalog)
        second = compose_all_qualification_recipes(second_catalog)

        self.assertEqual(first_catalog.catalog_sha256, second_catalog.catalog_sha256)
        self.assertEqual(first, second)
        for recipe in first:
            lock_payload = dict(recipe["lock"])
            observed_hash = lock_payload.pop("lock_content_sha256")
            self.assertEqual(canonical_sha256(lock_payload), observed_hash)
            self.assertEqual(
                first_catalog.catalog_sha256,
                recipe["lock"]["catalog_sha256"],
            )

    def test_all_recipes_remain_c1_with_false_physical_claims(self) -> None:
        for recipe in compose_all_qualification_recipes():
            with self.subTest(recipe_id=recipe["recipe_id"]):
                lock = recipe["lock"]
                qualification = lock["qualification"]
                claims = lock["physical_claims"]

                self.assertEqual("C1_contracted", qualification["maturity"])
                self.assertEqual(
                    "contract_validated",
                    qualification["support_state"],
                )
                self.assertEqual("not_run", qualification["physical_execution_status"])
                self.assertEqual("exploratory", qualification["authority"])
                self.assertEqual(set(PHYSICAL_CLAIM_KEYS), set(claims))
                self.assertFalse(any(claims.values()))
                self.assertEqual(claims, recipe["commands"]["physical_claims"])
                self.assertIn("not a supported-hardware", recipe["runbook_markdown"])

    def test_capacity_only_minimum_units_are_pinned(self) -> None:
        observed = {
            recipe["recipe_id"]: recipe["lock"]["capacity_estimate"]["minimum_units"]
            for recipe in compose_all_qualification_recipes()
        }
        self.assertEqual(EXPECTED_MINIMUM_UNITS, observed)

        for recipe in compose_all_qualification_recipes():
            capacity = recipe["lock"]["capacity_estimate"]
            with self.subTest(recipe_id=recipe["recipe_id"]):
                self.assertTrue(capacity["capacity_only"])
                self.assertFalse(capacity["performance_feasibility_evaluated"])
                self.assertTrue(capacity["capacity_sufficient_by_estimate"])
                self.assertEqual(
                    capacity["minimum_units"],
                    capacity["selected_units"],
                )

    def test_precision_contract_flags_apple_conversion_without_claiming_bf16(self) -> None:
        catalog = load_qualification_catalog()
        for platform_id in catalog.platform_ids:
            profile = catalog.platform(platform_id).to_dict()
            if profile["vendor"] == "apple":
                self.assertNotIn("bf16", profile["runtime"]["hardware_precision"])
                self.assertNotIn(
                    "bf16",
                    profile["runtime"]["runtime_precision_candidates"],
                )

        for recipe in compose_all_qualification_recipes(catalog):
            platform_id = recipe["lock"]["inputs"]["platform"]["platform_id"]
            precision = recipe["lock"]["precision_contract"]
            with self.subTest(recipe_id=recipe["recipe_id"]):
                if platform_id.startswith("apple-"):
                    self.assertEqual(
                        [],
                        precision["direct_activation_precision_overlap"],
                    )
                    self.assertTrue(
                        precision["conversion_or_custom_kernel_required"]
                    )
                    self.assertTrue(
                        any(
                            "no direct overlap" in blocker
                            for blocker in recipe["lock"]["blockers"]
                        )
                    )
                else:
                    self.assertIn(
                        "bf16",
                        precision["direct_activation_precision_overlap"],
                    )
                    self.assertFalse(
                        precision["conversion_or_custom_kernel_required"]
                    )

    def test_insufficient_unit_selection_is_explicitly_blocked(self) -> None:
        recipe = compose_qualification_recipe(
            "deepseek-r1",
            "nvidia-rtx-4090-24gb",
            units=34,
        )
        capacity = recipe["lock"]["capacity_estimate"]

        self.assertEqual(35, capacity["minimum_units"])
        self.assertEqual(34, capacity["selected_units"])
        self.assertFalse(capacity["capacity_sufficient_by_estimate"])
        self.assertIn(
            "Selected unit count is below the capacity-only minimum",
            recipe["lock"]["blockers"][0],
        )
        self.assertFalse(any(recipe["lock"]["physical_claims"].values()))

    def test_nonexistent_apple_m5_512_configuration_is_rejected(self) -> None:
        catalog = load_qualification_catalog()
        with self.assertRaisesRegex(
            QualificationCatalogError,
            r"unsupported marketed configuration apple-m5-512.*"
            r"Use apple-m3-ultra-512",
        ):
            catalog.platform("apple-m5-512")
        with self.assertRaisesRegex(
            QualificationCatalogError,
            r"Use apple-m3-ultra-512",
        ):
            compose_qualification_recipe(
                "qwen3-30b-a3b",
                "apple-m5-512",
                catalog=catalog,
            )

    def test_materialization_uses_contract_filenames_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = materialize_qualification_recipe(
                "gpt-oss-120b",
                "nvidia-h100-sxm-80gb",
                directory,
            )
            output = Path(directory)

            self.assertEqual(
                {
                    "recipe_lock": output / "recipe-lock.json",
                    "commands": output / "commands.json",
                    "runbook": output / "RUNBOOK.md",
                    "bundle_manifest": output / "bundle-manifest.json",
                },
                paths,
            )
            self.assertEqual(
                {
                    "recipe-lock.json",
                    "commands.json",
                    "RUNBOOK.md",
                    "bundle-manifest.json",
                },
                {path.name for path in output.iterdir()},
            )

            lock = _read_json(paths["recipe_lock"])
            commands = _read_json(paths["commands"])
            runbook = paths["runbook"].read_text(encoding="utf-8")
            lock_payload = dict(lock)
            observed_hash = lock_payload.pop("lock_content_sha256")

            self.assertEqual(canonical_sha256(lock_payload), observed_hash)
            self.assertEqual("fornax_qualification_recipe_lock", lock["record_kind"])
            self.assertEqual(
                "fornax_qualification_recipe_commands",
                commands["record_kind"],
            )
            self.assertEqual(lock["recipe_id"], commands["recipe_id"])
            self.assertFalse(any(lock["physical_claims"].values()))
            self.assertFalse(any(commands["physical_claims"].values()))
            self.assertIn(lock["recipe_id"], runbook)
            self.assertIn(observed_hash, runbook)
            for command in commands["commands"]:
                self.assertIsInstance(command["argv"], list)
                self.assertTrue(command["argv"])
                self.assertTrue(all(isinstance(item, str) for item in command["argv"]))
            verification = verify_materialized_qualification_recipe(output)
            self.assertTrue(verification["ok"], verification["errors"])
            self.assertTrue(verification["self_consistent"])
            self.assertTrue(verification["current_catalog_match"])
            self.assertFalse(verification["authenticated"])

    def test_catalog_binding_uses_the_verifier_snapshot_without_reopening(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            materialize_qualification_recipe(
                "qwen3-30b-a3b",
                "apple-m3-max-128",
                output,
            )
            lock_path = output / "recipe-lock.json"

            def verify_then_replace_lock(*args: Any, **kwargs: Any) -> dict:
                report = verify_recipe_packet(*args, **kwargs)
                replacement = output / "replacement-lock.json"
                replacement.write_text("[]\n", encoding="utf-8")
                os.replace(replacement, lock_path)
                return report

            with patch(
                "fornax.qualification.verify_recipe_packet",
                side_effect=verify_then_replace_lock,
            ):
                verification = verify_materialized_qualification_recipe(
                    output
                )

        self.assertTrue(verification["ok"], verification["errors"])
        self.assertTrue(verification["current_catalog_match"])
        self.assertEqual(
            "qwen3-30b-a3b",
            verification["recipe_lock_binding"]["model_id"],
        )
        self.assertEqual(
            "apple-m3-max-128",
            verification["recipe_lock_binding"]["platform_id"],
        )

    def test_materialization_no_overwrite_preserves_late_entries(self) -> None:
        target_names = (
            "recipe-lock.json",
            "commands.json",
            "RUNBOOK.md",
            "bundle-manifest.json",
        )
        real_publish = qualification_module._publish_recipe_file_at
        for target_name in target_names:
            with self.subTest(target_name=target_name):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "packet"
                    sentinel = b"late creator owns this entry\n"
                    injected = False

                    def inject_then_publish(**kwargs: Any) -> None:
                        nonlocal injected
                        if kwargs["name"] == target_name and not injected:
                            injected = True
                            (output / target_name).write_bytes(sentinel)
                        real_publish(**kwargs)

                    with patch(
                        "fornax.qualification._publish_recipe_file_at",
                        side_effect=inject_then_publish,
                    ):
                        with self.assertRaisesRegex(
                            QualificationCatalogError,
                            "appeared during no-overwrite publication",
                        ):
                            materialize_qualification_recipe(
                                "qwen3-30b-a3b",
                                "apple-m3-max-128",
                                output,
                                overwrite=False,
                            )

                    self.assertTrue(injected)
                    self.assertEqual(
                        sentinel,
                        (output / target_name).read_bytes(),
                    )

    def test_materialization_no_overwrite_never_follows_late_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "packet"
            victim = root / "victim.txt"
            victim.write_bytes(b"victim sentinel\n")
            target = output / "recipe-lock.json"
            real_publish = qualification_module._publish_recipe_file_at
            injected = False

            def inject_then_publish(**kwargs: Any) -> None:
                nonlocal injected
                if kwargs["name"] == "recipe-lock.json" and not injected:
                    injected = True
                    target.symlink_to(victim)
                real_publish(**kwargs)

            with patch(
                "fornax.qualification._publish_recipe_file_at",
                side_effect=inject_then_publish,
            ):
                with self.assertRaisesRegex(
                    QualificationCatalogError,
                    "appeared during no-overwrite publication",
                ):
                    materialize_qualification_recipe(
                        "qwen3-30b-a3b",
                        "apple-m3-max-128",
                        output,
                        overwrite=False,
                    )

            self.assertTrue(injected)
            self.assertTrue(target.is_symlink())
            self.assertEqual(b"victim sentinel\n", victim.read_bytes())

    def test_materialization_output_swap_never_writes_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "packet"
            displaced = root / "packet.displaced"
            victim = root / "victim"
            victim.mkdir()
            sentinel = victim / "recipe-lock.json"
            sentinel.write_bytes(b"victim sentinel\n")
            real_publish = qualification_module._publish_recipe_file_at
            swapped = False

            def swap_then_publish(**kwargs: Any) -> None:
                nonlocal swapped
                if not swapped:
                    swapped = True
                    output.rename(displaced)
                    output.symlink_to(victim, target_is_directory=True)
                real_publish(**kwargs)

            with patch(
                "fornax.qualification._publish_recipe_file_at",
                side_effect=swap_then_publish,
            ):
                with self.assertRaisesRegex(
                    QualificationCatalogError,
                    "changed identity",
                ):
                    materialize_qualification_recipe(
                        "qwen3-30b-a3b",
                        "apple-m3-max-128",
                        output,
                    )

            self.assertTrue(swapped)
            self.assertEqual(b"victim sentinel\n", sentinel.read_bytes())
            self.assertEqual(
                {"recipe-lock.json"},
                {path.name for path in victim.iterdir()},
            )

    def test_materialization_postverify_uses_the_retained_directory_fd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "packet"
            displaced = root / "packet.displaced"
            replacement = root / "replacement"
            replacement.mkdir()
            sentinel = replacement / "bundle-manifest.json"
            sentinel.write_bytes(b"replacement sentinel\n")
            captured_report: dict[str, Any] | None = None

            def swap_then_verify(*args: Any, **kwargs: Any) -> dict:
                nonlocal captured_report
                self.assertIsInstance(kwargs.get("_directory_fd"), int)
                output.rename(displaced)
                replacement.rename(output)
                captured_report = verify_recipe_packet(*args, **kwargs)
                return captured_report

            with patch(
                "fornax.qualification.verify_recipe_packet",
                side_effect=swap_then_verify,
            ):
                with self.assertRaisesRegex(
                    QualificationCatalogError,
                    "changed identity",
                ):
                    materialize_qualification_recipe(
                        "qwen3-30b-a3b",
                        "apple-m3-max-128",
                        output,
                    )

            self.assertIsNotNone(captured_report)
            self.assertTrue(
                captured_report["ok"],
                captured_report["errors"],
            )
            self.assertEqual(
                b"replacement sentinel\n",
                (output / "bundle-manifest.json").read_bytes(),
            )
            self.assertEqual(
                {"bundle-manifest.json"},
                {path.name for path in output.iterdir()},
            )

    def test_rendered_operator_commands_are_complete_and_shell_free(self) -> None:
        shell_executables = {
            "bash",
            "cmd",
            "fish",
            "powershell",
            "pwsh",
            "sh",
            "zsh",
        }
        shell_operator_tokens = {"&&", "||", ";", "|"}

        for recipe in compose_all_qualification_recipes():
            recipe_id = recipe["recipe_id"]
            model_id = recipe["lock"]["inputs"]["model"]["model_id"]
            commands = recipe["commands"]["commands"]
            steps = {command["step_id"]: command for command in commands}
            with self.subTest(recipe_id=recipe_id):
                self.assertIn("inspect_local_model", steps)
                self.assertIn("probe_host_identity", steps)
                self.assertIn("probe_max_runtime_registry", steps)
                self.assertIn("inspect-model", steps["inspect_local_model"]["argv"])
                self.assertIn("probe-host", steps["probe_host_identity"]["argv"])
                self.assertIn(
                    "<EVIDENCE_DIR>/hosts/<HOST_ID>/host-identity.json",
                    steps["probe_host_identity"]["argv"],
                )
                self.assertIn(
                    "probe-runtime",
                    steps["probe_max_runtime_registry"]["argv"],
                )

                selected_units = recipe["lock"]["capacity_estimate"][
                    "selected_units"
                ]
                if selected_units == 1:
                    self.assertIn("single_platform_model_bringup", steps)
                    self.assertNotIn("capacity_spanning_readiness", steps)
                    bringup = steps["single_platform_model_bringup"]["argv"]
                    vendor = recipe["lock"]["inputs"]["platform"]["vendor"]
                    if vendor == "apple":
                        self.assertIn("run-apple-single", bringup)
                        self.assertNotIn(
                            "apple-silicon-moe-serving-smoke",
                            bringup,
                        )
                        for required in (
                            "--model-artifact-report",
                            "--host-report",
                            "--runtime-report",
                            "--max-command",
                        ):
                            self.assertIn(required, bringup)
                else:
                    self.assertIn("capacity_spanning_readiness", steps)
                    self.assertNotIn("single_platform_model_bringup", steps)
                    self.assertFalse(
                        any(
                            "generate" in command["argv"]
                            or "apple-silicon-moe-serving-smoke"
                            in command["argv"]
                            for command in commands
                        )
                    )

                for command in commands:
                    argv = command["argv"]
                    self.assertIsInstance(argv, list)
                    self.assertTrue(argv)
                    self.assertTrue(
                        all(isinstance(argument, str) for argument in argv)
                    )
                    self.assertNotIn(argv[0], shell_executables)
                    self.assertFalse(
                        any(argument in shell_operator_tokens for argument in argv)
                    )
                    self.assertFalse(
                        any(
                            "$(" in argument or "`" in argument
                            for argument in argv
                        )
                    )

                acquisition = steps["acquire_pinned_model"]["argv"]
                inspect = steps["inspect_local_model"]["argv"]
                excludes = [
                    acquisition[index + 1]
                    for index, argument in enumerate(acquisition[:-1])
                    if argument == "--exclude"
                ]
                if model_id == "gpt-oss-120b":
                    self.assertEqual(["metal/*", "original/*"], excludes)
                else:
                    self.assertEqual([], excludes)
                    self.assertNotIn("metal/*", acquisition)
                    self.assertNotIn("original/*", acquisition)

                if model_id == "deepseek-r1":
                    review_index = inspect.index("--remote-code-review")
                    self.assertEqual(
                        "<REMOTE_CODE_REVIEW_JSON>",
                        inspect[review_index + 1],
                    )
                else:
                    self.assertNotIn("--remote-code-review", inspect)
                    self.assertNotIn("<REMOTE_CODE_REVIEW_JSON>", inspect)

    def test_actual_platform_profiles_match_six_synthetic_identities(self) -> None:
        catalog = load_qualification_catalog()
        cases = {
            "apple-m3-max-128": {
                "vendor": "apple",
                "chip": "Apple M3 Max",
                "memory_bytes": 128 * 1024**3,
                "host": {"system": "Darwin", "machine": "arm64"},
                "os": {"version": "15.0"},
            },
            "apple-m3-ultra-512": {
                "vendor": "apple",
                "chip": "Apple M3 Ultra",
                "memory_bytes": 512 * 1024**3,
                "host": {"system": "Darwin", "machine": "arm64"},
                "os": {"version": "15.0"},
            },
            "apple-m5-max-128": {
                "vendor": "apple",
                "chip": "Apple M5 Max",
                "memory_bytes": 128 * 1024**3,
                "host": {"system": "Darwin", "machine": "arm64"},
                "os": {"version": "15.0"},
            },
            "nvidia-h100-sxm-80gb": {
                "vendor": "nvidia",
                "host": {
                    "system": "Linux",
                    "machine": "x86_64",
                    "os_release": {
                        "id": "ubuntu",
                        "version_id": "22.04",
                        "name": "Ubuntu",
                    },
                },
                "gpus": [
                    {
                        "name": "NVIDIA H100 80GB HBM3",
                        "memory_total_mib": 81559,
                        "driver_version": "580.0",
                    }
                ],
            },
            "nvidia-rtx-4090-24gb": {
                "vendor": "nvidia",
                "host": {
                    "system": "Linux",
                    "machine": "x86_64",
                    "os_release": {
                        "id": "ubuntu",
                        "version_id": "22.04",
                        "name": "Ubuntu",
                    },
                },
                "gpus": [
                    {
                        "name": "NVIDIA GeForce RTX 4090",
                        "memory_total_mib": 24564,
                        "driver_version": "580.0",
                    }
                ],
            },
            "nvidia-rtx-5090-32gb": {
                "vendor": "nvidia",
                "host": {
                    "system": "Linux",
                    "machine": "x86_64",
                    "os_release": {
                        "id": "ubuntu",
                        "version_id": "22.04",
                        "name": "Ubuntu",
                    },
                },
                "gpus": [
                    {
                        "name": "NVIDIA GeForce RTX 5090",
                        "memory_total_mib": 32607,
                        "driver_version": "580.0",
                    }
                ],
            },
        }

        for platform_id, identity in cases.items():
            with self.subTest(platform_id=platform_id):
                profile = catalog.platform(platform_id).to_dict()
                report = match_platform_identity(identity, profile)
                self.assertTrue(report["ok"], report)
                self.assertEqual([], report["errors"])


class QualificationCliTest(unittest.TestCase):
    def test_cli_list_and_validate_emit_machine_readable_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            listing_path = output / "listing.json"
            validation_path = output / "validation.json"

            code, stdout, stderr = _run_cli(
                ["recipe", "list", "--out", str(listing_path)]
            )
            self.assertEqual(0, code)
            self.assertEqual("", stderr)
            self.assertIn("3 models x 6 platforms = 18 C1 recipes", stdout)
            listing = _read_json(listing_path)
            self.assertEqual(3, len(listing["models"]))
            self.assertEqual(6, len(listing["platforms"]))
            self.assertEqual(18, len(listing["recipes"]))
            self.assertTrue(
                all(
                    not any(recipe["physical_claims"].values())
                    for recipe in listing["recipes"]
                )
            )

            code, stdout, stderr = _run_cli(
                ["recipe", "validate", "--out", str(validation_path)]
            )
            self.assertEqual(0, code)
            self.assertEqual("", stderr)
            self.assertIn("PASS qualification-recipes", stdout)
            validation = _read_json(validation_path)
            self.assertTrue(validation["ok"])
            self.assertEqual(18, validation["summary"]["recipe_count"])
            self.assertTrue(validation["summary"]["all_physical_claims_false"])
            self.assertEqual(
                EXPECTED_MINIMUM_UNITS,
                validation["summary"]["minimum_units"],
            )

    def test_cli_golden_route_validates_qualification_recipes(self) -> None:
        code, stdout, stderr = _run_cli(["test", "qualification-recipes"])

        self.assertEqual(0, code)
        self.assertEqual("", stderr)
        self.assertIn(
            "PASS qualification-recipes: 3 models x 6 platforms = 18 C1 recipes",
            stdout,
        )

    def test_cli_render_refuses_overwrite_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            args = [
                "recipe",
                "render",
                "--model",
                "qwen3-30b-a3b",
                "--platform",
                "nvidia-h100-sxm-80gb",
                "--out-dir",
                str(output),
            ]

            code, stdout, stderr = _run_cli(args)
            self.assertEqual(0, code)
            self.assertEqual("", stderr)
            self.assertIn("maturity=C1_contracted", stdout)
            self.assertTrue((output / "recipe-lock.json").is_file())
            self.assertTrue((output / "commands.json").is_file())
            self.assertTrue((output / "RUNBOOK.md").is_file())

            unrelated = output / "operator-note.txt"
            unrelated.write_text("preserve me\n", encoding="utf-8")
            code, stdout, stderr = _run_cli(args)
            self.assertEqual(2, code)
            self.assertEqual("", stderr)
            self.assertIn("refusing to overwrite existing recipe files", stdout)
            self.assertEqual("preserve me\n", unrelated.read_text(encoding="utf-8"))

            code, stdout, stderr = _run_cli([*args, "--force"])
            self.assertEqual(0, code)
            self.assertEqual("", stderr)
            self.assertIn("rendered qwen3-30b-a3b--nvidia-h100-sxm-80gb", stdout)
            self.assertEqual("preserve me\n", unrelated.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "RUNBOOK.md",
                    "bundle-manifest.json",
                    "commands.json",
                    "operator-note.txt",
                    "recipe-lock.json",
                },
                {path.name for path in output.iterdir()},
            )

    def test_cli_render_refuses_insufficient_units_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            code, stdout, stderr = _run_cli(
                [
                    "recipe",
                    "render",
                    "--model",
                    "deepseek-r1",
                    "--platform",
                    "nvidia-rtx-4090-24gb",
                    "--units",
                    "34",
                    "--out-dir",
                    str(output),
                ]
            )

            self.assertEqual(2, code)
            self.assertEqual("", stderr)
            self.assertIn(
                "selected units are below the capacity-only minimum (34 < 35)",
                stdout,
            )
            self.assertFalse(output.exists())

    def test_cli_verify_detects_command_tampering_and_can_anchor_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            render_args = [
                "recipe",
                "render",
                "--model",
                "qwen3-30b-a3b",
                "--platform",
                "apple-m3-max-128",
                "--out-dir",
                str(output),
            ]
            code, _stdout, stderr = _run_cli(render_args)
            self.assertEqual(0, code, stderr)
            manifest = _read_json(output / "bundle-manifest.json")

            code, stdout, stderr = _run_cli(
                [
                    "recipe",
                    "verify",
                    "--packet-dir",
                    str(output),
                    "--expected-bundle-sha256",
                    manifest["bundle_content_sha256"],
                ]
            )
            self.assertEqual(0, code, stderr)
            self.assertIn("current_catalog_match=true", stdout)
            self.assertIn("authenticated=false", stdout)

            commands_path = output / "commands.json"
            commands_path.write_bytes(
                commands_path.read_bytes().replace(
                    b'"python3"',
                    b'"python4"',
                    1,
                )
            )
            code, stdout, stderr = _run_cli(
                [
                    "recipe",
                    "verify",
                    "--packet-dir",
                    str(output),
                ]
            )
            self.assertEqual(2, code, stderr)
            self.assertIn("digest mismatch: commands.json", stdout)

    def test_cli_verify_output_is_exclusive_and_outside_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            packet = root / "packet"
            code, _stdout, stderr = _run_cli(
                [
                    "recipe",
                    "render",
                    "--model",
                    "qwen3-30b-a3b",
                    "--platform",
                    "apple-m3-max-128",
                    "--out-dir",
                    str(packet),
                ]
            )
            self.assertEqual(0, code, stderr)

            recipe_lock = packet / "recipe-lock.json"
            original_lock = recipe_lock.read_bytes()
            code, stdout, stderr = _run_cli(
                [
                    "recipe",
                    "verify",
                    "--packet-dir",
                    str(packet),
                    "--out",
                    str(recipe_lock),
                ]
            )
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("output must be outside the packet directory", stderr)
            self.assertEqual(original_lock, recipe_lock.read_bytes())

            inside_output = packet / "verification.json"
            code, stdout, stderr = _run_cli(
                [
                    "recipe",
                    "verify",
                    "--packet-dir",
                    str(packet),
                    "--out",
                    str(inside_output),
                ]
            )
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("output must be outside the packet directory", stderr)
            self.assertFalse(inside_output.exists())

            report_output = root / "verification.json"
            verify_args = [
                "recipe",
                "verify",
                "--packet-dir",
                str(packet),
                "--out",
                str(report_output),
            ]
            code, stdout, stderr = _run_cli(verify_args)
            self.assertEqual(0, code, stderr)
            self.assertIn("PASS recipe packet", stdout)
            original_report = report_output.read_bytes()

            code, stdout, stderr = _run_cli(verify_args)
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("refuses to replace existing file", stderr)
            self.assertEqual(original_report, report_output.read_bytes())

            symlink_output = root / "verification-link.json"
            symlink_output.symlink_to(recipe_lock)
            code, stdout, stderr = _run_cli(
                [
                    "recipe",
                    "verify",
                    "--packet-dir",
                    str(packet),
                    "--out",
                    str(symlink_output),
                ]
            )
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("refuses to replace symbolic link", stderr)
            self.assertTrue(symlink_output.is_symlink())
            self.assertEqual(original_lock, recipe_lock.read_bytes())


if __name__ == "__main__":
    unittest.main()
