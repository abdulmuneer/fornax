from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from fornax.cli import main as cli_main
from fornax.planner import (
    EVIDENCE_REGISTRY_SCHEMA,
    EvidenceRegistry,
    Inventory,
    ModelSpec,
    Target,
    plan_placement,
)


def _measured(source_id: str, error: float = 0.05) -> dict[str, object]:
    return {
        "status": "measured",
        "source_id": source_id,
        "confidence": "high",
        "expected_relative_error": error,
    }


def _model(*, layers: int = 1, attributed: bool = False) -> dict[str, object]:
    data: dict[str, object] = {
        "hidden_dim": 8,
        "dtype_weight": "q4",
        "dtype_activation": "fp16",
        "layers": [
            {
                "kind": "dense",
                "weight_bytes": 1_000,
                "active_flops_per_token": 1_000,
            }
            for _ in range(layers)
        ],
    }
    if attributed:
        data["source_id"] = "model:sha256:model"
        data["quantization_source_id"] = "quantization:sha256:q4"
    return data


def _node(
    node_id: str = "node-0",
    *,
    authoritative: bool = False,
) -> dict[str, object]:
    data: dict[str, object] = {
        "id": node_id,
        "vendor": "nvidia",
        "runtime": "max",
        "mem_free_bytes": 1_000_000,
        "compute_class": 1_000_000_000.0,
        "mem_bandwidth_bytes_s": 1_000_000_000.0,
        "supported_dtypes": ["fp16"],
    }
    if authoritative:
        data.update(
            {
                "build_id": "max-build-1",
                "capabilities_complete": True,
                "supported_operations": ["stage_execute"],
                "supported_quantizations": ["q4"],
                "capability_source_id": f"capability:{node_id}",
                "measurement_provenance": {
                    "mem_free_bytes": _measured(f"memory:{node_id}"),
                    "compute_class": _measured(f"compute:{node_id}"),
                    "mem_bandwidth_bytes_s": _measured(f"bandwidth:{node_id}"),
                },
            }
        )
    return data


def _target(*, deployment: bool = False) -> dict[str, object]:
    data: dict[str, object] = {
        "concurrency": 1,
        "prompt_len": 1,
        "gen_len": 1,
    }
    if deployment:
        data.update(
            {
                "authority_mode": "deployment",
                "required_runtime": "max",
                "accepted_build_ids": ["max-build-1"],
                "required_operations": ["stage_execute"],
                "prediction_calibration": _measured("calibration:g2", 0.10),
                "max_expected_relative_error": 0.20,
            }
        )
    return data


def _evidence_types(
    *node_ids: str,
    include_link: bool = False,
) -> list[tuple[str, str]]:
    rows = [
        ("model:sha256:model", "model"),
        ("quantization:sha256:q4", "quantization"),
        ("calibration:g2", "calibration"),
    ]
    for node_id in node_ids:
        rows.extend(
            [
                (f"capability:{node_id}", "capability"),
                (f"memory:{node_id}", "measurement"),
                (f"compute:{node_id}", "measurement"),
                (f"bandwidth:{node_id}", "measurement"),
            ]
        )
    if include_link:
        rows.extend(
            [
                ("link:bandwidth", "route"),
                ("link:latency", "route"),
            ]
        )
    return rows


def _write_evidence_registry(
    root: Path,
    rows: list[tuple[str, str]],
    *,
    omit: frozenset[str] = frozenset(),
    expires_at: dict[str, str] | None = None,
    corrupt_hash_for: str | None = None,
) -> tuple[EvidenceRegistry, Path]:
    records: list[dict[str, object]] = []
    for index, (source_id, evidence_type) in enumerate(rows):
        if source_id in omit:
            continue
        artifact = root / f"planner-evidence-{index}.json"
        artifact.write_text(
            json.dumps({"evidence_type": evidence_type, "source_id": source_id}),
            encoding="utf-8",
        )
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        record: dict[str, object] = {
            "source_id": source_id,
            "evidence_type": evidence_type,
            "artifact_path": artifact.name,
            "artifact_sha256": (
                "0" * 64 if source_id == corrupt_hash_for else digest
            ),
            "status": "active",
            "expires_at": (expires_at or {}).get(
                source_id, "2099-01-01T00:00:00Z"
            ),
        }
        records.append(record)
    registry_path = root / "planner-evidence-registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": EVIDENCE_REGISTRY_SCHEMA,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return EvidenceRegistry.from_file(registry_path), registry_path


class PlannerAuthorityTest(unittest.TestCase):
    def test_numeric_parsers_reject_non_finite_and_boolean_values(self) -> None:
        model_with_trace = _model(attributed=True)
        model_with_trace["layers"] = [
            {
                "kind": "moe",
                "weight_bytes": 1_000,
                "active_flops_per_token": 1_000,
                "num_experts": 2,
                "experts_active": 1,
                "expert_bytes": 100,
                "expert_flops_per_token": 100,
            }
        ]
        model_with_trace["expert_traces"] = [
            {
                "layer_id": 0,
                "expert_id": 0,
                "hit_rate_prefill": float("nan"),
                "hit_rate_decode": 0.5,
            }
        ]

        cases = {
            "model integer boolean": lambda: ModelSpec.from_dict(
                {**_model(), "hidden_dim": True}
            ),
            "layer integer boolean": lambda: ModelSpec.from_dict(
                {
                    **_model(),
                    "layers": [
                        {
                            "kind": "dense",
                            "weight_bytes": False,
                            "active_flops_per_token": 1_000,
                        }
                    ],
                }
            ),
            "expert trace NaN": lambda: ModelSpec.from_dict(model_with_trace),
            "node compute NaN": lambda: Inventory.from_dict(
                {"nodes": [{**_node(), "compute_class": float("nan")}]}
            ),
            "node bandwidth infinity": lambda: Inventory.from_dict(
                {
                    "nodes": [
                        {
                            **_node(),
                            "mem_bandwidth_bytes_s": float("inf"),
                        }
                    ]
                }
            ),
            "node memory boolean": lambda: Inventory.from_dict(
                {"nodes": [{**_node(), "mem_free_bytes": True}]}
            ),
            "link latency NaN": lambda: Inventory.from_dict(
                {
                    "nodes": [_node("node-0"), _node("node-1")],
                    "links": [
                        {
                            "a": "node-0",
                            "b": "node-1",
                            "bandwidth_bytes_s": 1_000_000.0,
                            "latency_s": float("nan"),
                        }
                    ],
                }
            ),
            "link bandwidth infinity": lambda: Inventory.from_dict(
                {
                    "nodes": [_node("node-0"), _node("node-1")],
                    "links": [
                        {
                            "a": "node-0",
                            "b": "node-1",
                            "bandwidth_bytes_s": float("inf"),
                            "latency_s": 0.001,
                        }
                    ],
                }
            ),
            "target integer boolean": lambda: Target.from_dict(
                {**_target(), "concurrency": True}
            ),
            "target float boolean": lambda: Target.from_dict(
                {**_target(), "memory_reserve_fraction": False}
            ),
            "target float infinity": lambda: Target.from_dict(
                {**_target(), "routing_metadata_bytes_per_token": float("inf")}
            ),
            "target optional NaN": lambda: Target.from_dict(
                {**_target(), "remote_expert_wait_slo_s": float("nan")}
            ),
        }
        for name, parse in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "finite|boolean"):
                    parse()

    def test_non_finite_authoritative_node_fails_closed_and_serializes(self) -> None:
        model = ModelSpec.from_dict(_model(attributed=True))
        inventory = Inventory.from_dict(
            {"nodes": [_node(authoritative=True)]}
        )
        target = Target.from_dict(_target(deployment=True))

        # Model constructors reject NaN. This mutation simulates an invalid value
        # injected by a caller after validation and exercises the authority seam.
        object.__setattr__(inventory.nodes[0], "compute_class", float("nan"))
        plan = plan_placement(model, inventory, target)

        self.assertFalse(plan.feasible)
        self.assertEqual("rejected", plan.authority.status)
        self.assertFalse(plan.authority.deployment_authorized)
        self.assertIn("deployment authority rejected", plan.infeasible_reason or "")
        self.assertIn("compute_class is not finite", plan.explanations[0].reason)
        payload = plan.to_dict()
        self.assertIsNone(payload["explanations"][0]["metrics"]["compute_class"])
        json.dumps(payload, allow_nan=False)

    def test_legacy_inputs_are_automatically_labeled_exploratory(self) -> None:
        plan = plan_placement(
            ModelSpec.from_dict(_model()),
            Inventory.from_dict({"nodes": [_node()]}),
            Target.from_dict(_target()),
        )

        self.assertTrue(plan.feasible, plan.infeasible_reason)
        self.assertEqual("exploratory", plan.authority.status)
        self.assertFalse(plan.authority.deployment_authorized)
        self.assertIn("model has no source_id", plan.authority.reasons)
        self.assertIsNone(plan.to_dict()["predicted"]["prediction_intervals"])

    def test_declared_capability_mismatches_fail_even_in_exploratory_mode(self) -> None:
        target_data = _target()
        target_data.update(
            {
                "required_runtime": "max",
                "accepted_build_ids": ["max-build-1"],
                "required_operations": ["stage_execute"],
            }
        )
        base = _node(authoritative=True)
        cases = {
            "runtime": ("runtime", "custom", "runtime='custom'"),
            "build": ("build_id", "other-build", "is not accepted"),
            "operation": ("supported_operations", [], "lacks required operations"),
            "quantization": (
                "supported_quantizations",
                ["fp16"],
                "does not support weight quantization q4",
            ),
        }
        for name, (field_name, value, reason) in cases.items():
            with self.subTest(name=name):
                node = copy.deepcopy(base)
                node[field_name] = value
                plan = plan_placement(
                    ModelSpec.from_dict(_model()),
                    Inventory.from_dict({"nodes": [node]}),
                    Target.from_dict(target_data),
                )
                self.assertFalse(plan.feasible)
                self.assertIn(reason, plan.explanations[0].reason)
                self.assertEqual("exploratory", plan.authority.status)

    def test_deployment_mode_fails_closed_without_evidence(self) -> None:
        plan = plan_placement(
            ModelSpec.from_dict(_model()),
            Inventory.from_dict({"nodes": [_node()]}),
            Target.from_dict(_target()),
            authority_mode="deployment",
        )

        self.assertFalse(plan.feasible)
        self.assertEqual("rejected", plan.authority.status)
        self.assertFalse(plan.authority.deployment_authorized)
        self.assertIn("deployment authority rejected", plan.infeasible_reason or "")
        self.assertIn("model has no source_id", plan.authority.reasons)

    def test_invented_source_strings_cannot_authorize_without_registry(self) -> None:
        plan = plan_placement(
            ModelSpec.from_dict(_model(attributed=True)),
            Inventory.from_dict({"nodes": [_node(authoritative=True)]}),
            Target.from_dict(_target(deployment=True)),
        )

        self.assertFalse(plan.feasible)
        self.assertEqual("rejected", plan.authority.status)
        self.assertFalse(plan.authority.deployment_authorized)
        self.assertIn(
            "deployment evidence registry is required",
            plan.authority.reasons,
        )
        self.assertIsNone(plan.authority.evidence_registry_sha256)

    def test_complete_sha_bound_registry_can_authorize_simple_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            evidence_registry, _ = _write_evidence_registry(
                Path(td), _evidence_types("node-0")
            )
            plan = plan_placement(
                ModelSpec.from_dict(_model(attributed=True)),
                Inventory.from_dict({"nodes": [_node(authoritative=True)]}),
                Target.from_dict(_target(deployment=True)),
                evidence_registry=evidence_registry,
            )

        self.assertTrue(plan.feasible, plan.infeasible_reason)
        self.assertEqual("deployment_authoritative", plan.authority.status)
        self.assertTrue(plan.authority.deployment_authorized)
        self.assertEqual(0.10, plan.authority.prediction_expected_relative_error)
        self.assertEqual(0.05, plan.authority.input_max_expected_relative_error)
        self.assertIn("calibration:g2", plan.authority.source_ids)
        self.assertEqual(
            evidence_registry.manifest_sha256,
            plan.authority.evidence_registry_sha256,
        )
        predicted = plan.to_dict()["predicted"]
        self.assertEqual("calibration:g2", predicted["prediction_provenance"]["source_id"])
        throughput = predicted["throughput_tok_s"]
        self.assertEqual(
            [throughput * 0.9, throughput * 1.1],
            predicted["prediction_intervals"]["throughput_tok_s"],
        )

    def test_missing_and_stale_registry_records_fail_closed(self) -> None:
        model = ModelSpec.from_dict(_model(attributed=True))
        inventory = Inventory.from_dict({"nodes": [_node(authoritative=True)]})
        target = Target.from_dict(_target(deployment=True))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing_registry, _ = _write_evidence_registry(
                root,
                _evidence_types("node-0"),
                omit=frozenset({"calibration:g2"}),
            )
            missing = plan_placement(
                model,
                inventory,
                target,
                evidence_registry=missing_registry,
            )
            self.assertFalse(missing.feasible)
            self.assertTrue(
                any(
                    "calibration:g2" in reason and "absent" in reason
                    for reason in missing.authority.reasons
                )
            )

            stale_registry, _ = _write_evidence_registry(
                root,
                _evidence_types("node-0"),
                expires_at={"calibration:g2": "2000-01-01T00:00:00Z"},
            )
            stale = plan_placement(
                model,
                inventory,
                target,
                evidence_registry=stale_registry,
            )
            self.assertFalse(stale.feasible)
            self.assertTrue(
                any(
                    "calibration:g2" in reason and "stale" in reason
                    for reason in stale.authority.reasons
                )
            )

    def test_registry_artifact_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            evidence_registry, _ = _write_evidence_registry(
                Path(td),
                _evidence_types("node-0"),
                corrupt_hash_for="model:sha256:model",
            )
            plan = plan_placement(
                ModelSpec.from_dict(_model(attributed=True)),
                Inventory.from_dict({"nodes": [_node(authoritative=True)]}),
                Target.from_dict(_target(deployment=True)),
                evidence_registry=evidence_registry,
            )

        self.assertFalse(plan.feasible)
        self.assertTrue(
            any(
                "model:sha256:model" in reason and "SHA-256 mismatch" in reason
                for reason in plan.authority.reasons
            )
        )

    def test_registry_rechecks_artifact_bytes_when_plan_is_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence_registry, _ = _write_evidence_registry(
                root, _evidence_types("node-0")
            )
            (root / "planner-evidence-0.json").write_text(
                '{"tampered":true}', encoding="utf-8"
            )
            plan = plan_placement(
                ModelSpec.from_dict(_model(attributed=True)),
                Inventory.from_dict({"nodes": [_node(authoritative=True)]}),
                Target.from_dict(_target(deployment=True)),
                evidence_registry=evidence_registry,
            )

        self.assertFalse(plan.feasible)
        self.assertTrue(
            any("SHA-256 mismatch" in reason for reason in plan.authority.reasons)
        )

    def test_registry_record_type_must_match_reference_role(self) -> None:
        rows = [
            (source_id, "measurement" if source_id == "calibration:g2" else kind)
            for source_id, kind in _evidence_types("node-0")
        ]
        with tempfile.TemporaryDirectory() as td:
            evidence_registry, _ = _write_evidence_registry(Path(td), rows)
            plan = plan_placement(
                ModelSpec.from_dict(_model(attributed=True)),
                Inventory.from_dict({"nodes": [_node(authoritative=True)]}),
                Target.from_dict(_target(deployment=True)),
                evidence_registry=evidence_registry,
            )

        self.assertFalse(plan.feasible)
        self.assertTrue(
            any(
                "calibration:g2" in reason
                and "expected 'calibration'" in reason
                for reason in plan.authority.reasons
            )
        )

    def test_deployment_mode_requires_measured_boundary_route(self) -> None:
        nodes = [_node("node-0", authoritative=True), _node("node-1", authoritative=True)]
        link = {
            "a": "node-0",
            "b": "node-1",
            "bandwidth_bytes_s": 1_000_000_000.0,
            "latency_s": 0.00001,
        }
        model = ModelSpec.from_dict(_model(layers=2, attributed=True))
        target = Target.from_dict(_target(deployment=True))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            node_registry, _ = _write_evidence_registry(
                root, _evidence_types("node-0", "node-1")
            )
            rejected = plan_placement(
                model,
                Inventory.from_dict({"nodes": nodes, "links": [link]}),
                target,
                min_stages=2,
                max_stages=2,
                evidence_registry=node_registry,
            )
            self.assertFalse(rejected.feasible)
            self.assertEqual("rejected", rejected.authority.status)

            link["measurement_provenance"] = {
                "bandwidth_bytes_s": _measured("link:bandwidth"),
                "latency_s": _measured("link:latency"),
            }
            route_registry, _ = _write_evidence_registry(
                root,
                _evidence_types("node-0", "node-1", include_link=True),
            )
            admitted = plan_placement(
                model,
                Inventory.from_dict({"nodes": nodes, "links": [link]}),
                target,
                min_stages=2,
                max_stages=2,
                evidence_registry=route_registry,
            )
        self.assertTrue(admitted.feasible, admitted.infeasible_reason)
        self.assertTrue(admitted.authority.deployment_authorized)
        self.assertIn("link:bandwidth", admitted.authority.source_ids)

    def test_cli_deployment_override_writes_rejected_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target_path = root / "target.json"
            inventory_path = root / "inventory.json"
            out_path = root / "plan.json"
            target_path.write_text(
                json.dumps({"model": _model(), "target": _target()}),
                encoding="utf-8",
            )
            inventory_path.write_text(
                json.dumps({"nodes": [_node()]}), encoding="utf-8"
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "plan",
                        "--target",
                        str(target_path),
                        "--inventory",
                        str(inventory_path),
                        "--out",
                        str(out_path),
                        "--authority-mode",
                        "deployment",
                    ]
                )

            self.assertEqual(2, exit_code)
            result = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("rejected", result["authority"]["status"])
            self.assertIn("authority=rejected", stdout.getvalue())

    def test_cli_deployment_accepts_separate_evidence_registry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target_path = root / "target.json"
            inventory_path = root / "inventory.json"
            out_path = root / "plan.json"
            target_path.write_text(
                json.dumps(
                    {
                        "model": _model(attributed=True),
                        "target": _target(deployment=True),
                    }
                ),
                encoding="utf-8",
            )
            inventory_path.write_text(
                json.dumps({"nodes": [_node(authoritative=True)]}),
                encoding="utf-8",
            )
            registry, registry_path = _write_evidence_registry(
                root, _evidence_types("node-0")
            )
            exit_code = cli_main(
                [
                    "plan",
                    "--target",
                    str(target_path),
                    "--inventory",
                    str(inventory_path),
                    "--out",
                    str(out_path),
                    "--authority-mode",
                    "deployment",
                    "--evidence-registry",
                    str(registry_path),
                ]
            )

            self.assertEqual(0, exit_code)
            result = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("deployment_authoritative", result["authority"]["status"])
            self.assertEqual(
                registry.manifest_sha256,
                result["authority"]["evidence_registry_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
