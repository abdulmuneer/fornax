from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fornax.backends import (
    STAGE_BACKEND_API_VERSION,
    BackendCapabilities,
    StageManifest,
    check_stage_backend,
    check_stage_backend_factory,
)
from fornax.cli import main
from fornax.stage_runtime import ReferenceStageBackend


PLAN_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64


class TestMaxBackend(ReferenceStageBackend):
    __test__ = False

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__()
        self.options = options
        self.load_calls = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend="test-max",
            build_id=str(self.options.get("build_id", "test-max-build")),
            device_identity=str(self.options.get("device_identity", "test-device")),
            supported_dtypes=("bf16",),
            memory_bytes=1 << 30,
            supported_operations=("stage_execute",),
        )

    def load(self, stage_manifest):  # type: ignore[no-untyped-def]
        self.load_calls += 1
        return super().load(stage_manifest)


def create_test_backend(options: dict[str, Any]) -> TestMaxBackend:
    return TestMaxBackend(options)


class LegacyBackendWithoutRelease:
    def __init__(self, options: dict[str, Any]) -> None:
        self.delegate = TestMaxBackend(options)

    def capabilities(self):  # type: ignore[no-untyped-def]
        return self.delegate.capabilities()

    def load(self, manifest):  # type: ignore[no-untyped-def]
        return self.delegate.load(manifest)

    def health(self, handle):  # type: ignore[no-untyped-def]
        return self.delegate.health(handle)

    def execute(self, handle, request):  # type: ignore[no-untyped-def]
        return self.delegate.execute(handle, request)

    def cancel(self, handle, request_id, reason):  # type: ignore[no-untyped-def]
        return self.delegate.cancel(handle, request_id, reason)

    def drain(self, handle, deadline_ns):  # type: ignore[no-untyped-def]
        return self.delegate.drain(handle, deadline_ns)

    def unload(self, handle):  # type: ignore[no-untyped-def]
        return self.delegate.unload(handle)


def create_legacy_backend(options: dict[str, Any]) -> LegacyBackendWithoutRelease:
    return LegacyBackendWithoutRelease(options)


def stage_manifest(*, build_id: str = "test-max-build") -> StageManifest:
    return StageManifest(
        manifest_version=1,
        model_id="fornax/backend-sdk-test",
        model_snapshot="snapshot-v1",
        model_config_hash=HASH_A,
        tokenizer_hash=HASH_B,
        template_hash=HASH_C,
        max_build_id=build_id,
        fornax_abi_major=1,
        fornax_abi_minor=0,
        plan_id=PLAN_ID,
        plan_hash=HASH_D,
        stage_id="stage-0",
        stage_index=0,
        layer_start=0,
        layer_end=1,
        input_contract={
            "kind": "activation",
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": 4,
        },
        output_contract={
            "kind": "activation",
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": 4,
        },
        kv_policy="stage_local",
        device_requirement={
            "backend": "test-max",
            "device_identity": "test-device",
            "minimum_memory_bytes": 1024,
            "dtypes": ["bf16"],
            "operations": ["stage_execute"],
        },
    )


class StageBackendSdkTest(unittest.TestCase):
    def test_direct_backend_lifecycle_smoke_passes(self) -> None:
        report = check_stage_backend(
            TestMaxBackend({}), stage_manifest(), factory="test:factory"
        )
        self.assertTrue(report["ok"], report["checks"])
        self.assertEqual(12, report["passed_count"])
        self.assertFalse(report["closes_g2"])
        self.assertEqual(
            "backend", report["capability_attestation"]["observed"]["source"]
        )
        release_check = next(
            item
            for item in report["checks"]
            if item["name"] == "release-request-state"
        )
        self.assertTrue(release_check["ok"])
        self.assertIn("idempotency_results_released", release_check["evidence"])
        bounded_check = next(
            item
            for item in report["checks"]
            if item["name"] == "bounded-state-retention"
        )
        self.assertTrue(bounded_check["ok"])
        self.assertIn("max_live_requests", bounded_check["evidence"])

    def test_capability_mismatch_fails_before_load(self) -> None:
        backend = TestMaxBackend({})
        report = check_stage_backend(backend, stage_manifest(build_id="other-build"))
        self.assertFalse(report["ok"])
        self.assertEqual(0, backend.load_calls)
        self.assertFalse(report["checks"][0]["ok"])
        self.assertIn("build_id", report["checks"][0]["evidence"])

    def test_backend_capability_requires_exact_abi_minor(self) -> None:
        class FutureMinorOnlyBackend(TestMaxBackend):
            def capabilities(self) -> BackendCapabilities:
                base = super().capabilities()
                return BackendCapabilities(
                    backend=base.backend,
                    build_id=base.build_id,
                    device_identity=base.device_identity,
                    supported_dtypes=base.supported_dtypes,
                    abi_versions=((1, 1),),
                    memory_bytes=base.memory_bytes,
                    supported_operations=base.supported_operations,
                )

        backend = FutureMinorOnlyBackend({})
        report = check_stage_backend(backend, stage_manifest())
        self.assertFalse(report["ok"])
        self.assertEqual(0, backend.load_calls)
        self.assertIn("ABI 1.0 unsupported", report["checks"][0]["evidence"])

    def test_factory_is_importable_and_fails_closed(self) -> None:
        factory = f"{__name__}:create_test_backend"
        report = check_stage_backend_factory(factory, {}, stage_manifest())
        self.assertTrue(report["ok"], report["checks"])
        failed = check_stage_backend_factory(
            "missing_fornax_backend:create", {}, stage_manifest()
        )
        self.assertFalse(failed["ok"])
        self.assertTrue(all(not item["ok"] for item in failed["checks"]))

        legacy = check_stage_backend_factory(
            f"{__name__}:create_legacy_backend", {}, stage_manifest()
        )
        self.assertFalse(legacy["ok"])
        self.assertIn("missing required methods: release", legacy["checks"][0]["evidence"])

    def test_cli_writes_limited_evidence_report(self) -> None:
        factory = f"{__name__}:create_test_backend"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            options_path = root / "options.json"
            report_path = root / "report.json"
            manifest_path.write_text(
                json.dumps(stage_manifest().to_dict()), encoding="utf-8"
            )
            options_path.write_text("{}", encoding="utf-8")
            code = main(
                [
                    "runtime",
                    "backend-conformance",
                    "--factory",
                    factory,
                    "--manifest",
                    str(manifest_path),
                    "--options",
                    str(options_path),
                    "--out",
                    str(report_path),
                ]
            )
            self.assertEqual(0, code)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual("functional_contract_smoke", report["evidence_class"])
            self.assertFalse(report["closes_g2"])
            self.assertEqual(2, STAGE_BACKEND_API_VERSION)


if __name__ == "__main__":
    unittest.main()
