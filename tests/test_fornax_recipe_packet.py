from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from fornax.qualification import compose_qualification_recipe
from fornax.recipe_packet import (
    MANAGED_PACKET_FILENAMES,
    PACKET_MANIFEST_FILENAME,
    RecipePacketError,
    build_recipe_packet_manifest,
    verify_recipe_packet,
)


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _managed_files(recipe_id: str) -> dict[str, bytes]:
    model_id, platform_id = recipe_id.split("--", 1)
    bundle = compose_qualification_recipe(model_id, platform_id)
    return {
        "recipe-lock.json": _pretty_json(bundle["lock"]),
        "commands.json": _pretty_json(bundle["commands"]),
        "RUNBOOK.md": bundle["runbook_markdown"].encode("utf-8"),
    }


def _write_packet(root: Path, recipe_id: str) -> None:
    managed = _managed_files(recipe_id)
    manifest = build_recipe_packet_manifest(recipe_id, managed)
    for name, value in managed.items():
        (root / name).write_bytes(value)
    (root / PACKET_MANIFEST_FILENAME).write_bytes(_pretty_json(manifest))


class RecipePacketTest(unittest.TestCase):
    def test_manifest_and_packet_verify_deterministically(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        managed = _managed_files(recipe_id)
        first = build_recipe_packet_manifest(recipe_id, managed)
        second = build_recipe_packet_manifest(recipe_id, managed)

        self.assertEqual(first, second)
        self.assertEqual(set(MANAGED_PACKET_FILENAMES), set(first["files"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packet(root, recipe_id)
            report = verify_recipe_packet(root)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(recipe_id, report["recipe_id"])
        self.assertEqual([], report["unmanaged_entries"])

    def test_managed_file_tampering_fails_digest_and_semantic_checks(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packet(root, recipe_id)
            commands_path = root / "commands.json"
            commands_path.write_bytes(
                commands_path.read_bytes().replace(
                    b'"python3"',
                    b'"python4"',
                    1,
                )
            )

            report = verify_recipe_packet(root)

        self.assertFalse(report["ok"])
        self.assertIn(
            "managed packet file digest mismatch: commands.json",
            report["errors"],
        )

    def test_manifest_content_hash_tampering_fails_closed(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packet(root, recipe_id)
            manifest_path = root / PACKET_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["recipe_id"] = "different--recipe"
            manifest_path.write_bytes(_pretty_json(manifest))

            report = verify_recipe_packet(root)

        self.assertFalse(report["ok"])
        self.assertIn(
            "bundle manifest content hash does not match its payload",
            report["errors"],
        )
        self.assertTrue(
            any("recipe_id does not match" in error for error in report["errors"])
        )

    def test_coherently_rehashed_shell_command_still_fails_c1_semantics(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            managed = _managed_files(recipe_id)
            commands = json.loads(managed["commands.json"])
            commands["commands"][0]["argv"] = ["sh", "-c", "echo unsafe"]
            managed["commands.json"] = _pretty_json(commands)
            manifest = build_recipe_packet_manifest(recipe_id, managed)
            for name, value in managed.items():
                (root / name).write_bytes(value)
            (root / PACKET_MANIFEST_FILENAME).write_bytes(
                _pretty_json(manifest)
            )

            report = verify_recipe_packet(root)

        self.assertFalse(report["ok"])
        self.assertTrue(
            any("forbidden shell" in error for error in report["errors"]),
            report,
        )

    def test_expected_digest_is_separate_from_authentication(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packet(root, recipe_id)
            manifest = json.loads(
                (root / PACKET_MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            matching = verify_recipe_packet(
                root,
                expected_bundle_content_sha256=manifest[
                    "bundle_content_sha256"
                ],
            )
            mismatching = verify_recipe_packet(
                root,
                expected_bundle_content_sha256="sha256:" + "0" * 64,
            )

        self.assertTrue(matching["ok"], matching["errors"])
        self.assertTrue(matching["expected_digest_matched"])
        self.assertFalse(matching["authenticated"])
        self.assertFalse(mismatching["ok"])
        self.assertFalse(mismatching["expected_digest_matched"])
        self.assertFalse(mismatching["authenticated"])

    def test_nonfinite_exponent_is_reported_instead_of_escaping(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packet(root, recipe_id)
            manifest_path = root / PACKET_MANIFEST_FILENAME
            manifest_path.write_bytes(
                manifest_path.read_bytes().replace(
                    b'"schema_version": 1',
                    b'"schema_version": 1e9999',
                    1,
                )
            )

            report = verify_recipe_packet(root)

        self.assertFalse(report["ok"])
        self.assertTrue(
            any("non-finite JSON number" in error for error in report["errors"]),
            report,
        )

    def test_manifest_builder_rejects_missing_or_unknown_managed_files(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        managed = _managed_files(recipe_id)
        managed.pop("RUNBOOK.md")
        managed["operator-note.txt"] = b"unmanaged\n"

        with self.assertRaisesRegex(
            RecipePacketError,
            r"missing RUNBOOK.md; unknown operator-note.txt",
        ):
            build_recipe_packet_manifest(recipe_id, managed)

    def test_unmanaged_operator_evidence_requires_explicit_allowance(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packet(root, recipe_id)
            (root / "host-identity.json").write_text("{}\n", encoding="utf-8")

            strict_report = verify_recipe_packet(root)
            allowed_report = verify_recipe_packet(
                root,
                allow_unmanaged_entries=True,
            )

        self.assertFalse(strict_report["ok"])
        self.assertIn(
            "packet contains unmanaged entries: host-identity.json",
            strict_report["errors"],
        )
        self.assertTrue(allowed_report["ok"], allowed_report["errors"])
        self.assertEqual(
            ["host-identity.json"],
            allowed_report["unmanaged_entries"],
        )

    def test_borrowed_directory_fd_verifies_one_pinned_directory_view(
        self,
    ) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "packet"
            displaced = parent / "packet.displaced"
            root.mkdir()
            _write_packet(root, recipe_id)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            directory_fd = os.open(root, flags)
            try:
                root.rename(displaced)
                root.mkdir()
                (root / PACKET_MANIFEST_FILENAME).write_text(
                    "{}\n",
                    encoding="utf-8",
                )

                report = verify_recipe_packet(
                    root,
                    _directory_fd=directory_fd,
                )
            finally:
                os.close(directory_fd)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(recipe_id, report["recipe_id"])
        self.assertEqual(
            "qwen3-30b-a3b",
            report["recipe_lock_binding"]["model_id"],
        )
        self.assertEqual(
            "apple-m3-max-128",
            report["recipe_lock_binding"]["platform_id"],
        )

    def test_fifo_managed_entry_fails_without_blocking_for_a_writer(self) -> None:
        recipe_id = "qwen3-30b-a3b--apple-m3-max-128"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packet(root, recipe_id)
            fifo = root / "recipe-lock.json"
            fifo.unlink()
            os.mkfifo(fifo)

            outcome: list[object] = []

            def verify() -> None:
                try:
                    outcome.append(verify_recipe_packet(root))
                except BaseException as exc:  # Preserve failures from the worker.
                    outcome.append(exc)

            worker = threading.Thread(target=verify, daemon=True)
            worker.start()
            worker.join(timeout=2.0)
            self.assertFalse(
                worker.is_alive(),
                "packet verification blocked while opening a FIFO with no writer",
            )
            self.assertEqual(1, len(outcome))
            if isinstance(outcome[0], BaseException):
                raise outcome[0]
            report = outcome[0]
            self.assertIsInstance(report, dict)
            assert isinstance(report, dict)

        self.assertFalse(report["ok"])
        self.assertIn(
            "recipe-lock.json must be a regular file",
            report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
