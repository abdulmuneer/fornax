"""Synthetic physical-result emitter used only by G2 runner unit tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


TIERS = {
    "V6_NVIDIA": "T2-physical-single-node",
    "V6_APPLE": "T2-physical-single-node",
    "V7_PIPELINE": "T3-physical-multinode",
    "V8_LOAD_CALIBRATION": "T3-physical-multinode",
    "V9_STABILITY": "T3-physical-multinode",
    "V10_FAILURES": "T3-physical-multinode",
}

NODE_IDENTITY_FIELDS = (
    "role",
    "physical_host_id",
    "hostname",
    "os_build",
    "architecture",
    "device_identity",
    "driver_runtime",
    "max_cli_version",
    "mojo_version",
    "bazel_version",
    "bazelisk_version",
    "python_version",
    "compiler_version",
    "toolchain_version",
    "build_target",
    "build_flags_sha256",
    "build_environment_sha256",
    "max_patch_commit",
    "max_binary_sha256",
    "memory_bytes",
)


def emit(manifest_path: Path, step_id: str, step_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = {node["role"]: node for node in manifest["nodes"]}
    roles = (
        ("nvidia",)
        if step_id == "V6_NVIDIA"
        else ("apple",)
        if step_id == "V6_APPLE"
        else ("nvidia", "apple")
    )
    result = {
        "schema_version": 1,
        "step_id": step_id,
        "evidence_class": TIERS[step_id],
        "measured": True,
        "physical": True,
        "same_host_proxy": False,
        "passed": True,
        "model": manifest["model"],
        "plan": manifest["plan"],
        "max_patch_commit": nodes["nvidia"]["max_patch_commit"],
        "observed_nodes": [nodes[role] for role in roles],
        "physical_host_ids": [nodes[role]["physical_host_id"] for role in roles],
        "raw_artifacts": ["raw-measurements.json"],
        "raw_measurements_artifact": "raw-measurements.json",
        "checks": {},
    }
    observations = []
    limits = None
    if step_id in {"V6_NVIDIA", "V6_APPLE"}:
        result["checks"] = {
            "operator_stage_parity": True,
            "numerical_parity": True,
        }
        result["numerical"] = {
            "reference_id": manifest["correctness"]["reference"]["reference_id"],
            "tolerance_approval_id": manifest["correctness"]["tolerance_policy"]["approval_id"],
            "max_abs_error": 0.001,
            "nonfinite_count": 0,
            "top1_mismatch_count": 0,
            "routing_mismatch_count": 0,
            "all_within_tolerance": True,
        }
        observations.extend(
            {
                "kind": "numerical",
                "scope": scope,
                "dtype": "bf16",
                "reference_id": manifest["correctness"]["reference"]["reference_id"],
                "reference": [0.0, 1.0],
                "observed": [0.001, 1.0],
                "reference_top1": 1,
                "observed_top1": 1,
                "reference_routes": [2, 3],
                "observed_routes": [2, 3],
            }
            for scope in ("operator", "stage")
        )
        if step_id == "V6_APPLE":
            result["apple_role_decision"] = {
                "role": "pipeline-stage",
                "evidence": "derived-from-raw",
                "criteria_id": "g2-apple-role-v1",
            }
            observations.append(
                {
                    "kind": "apple_role_criteria",
                    "operator_cases_total": 10,
                    "operator_cases_passed": 10,
                    "stage_cases_total": 20,
                    "stage_cases_passed": 20,
                    "expert_cases_total": 10,
                    "expert_cases_passed": 10,
                    "decode_context_tokens": [16, 128, 512, 4096],
                    "memory_high_water_bytes": 4096,
                    "runtime_error_count": 0,
                }
            )
    elif step_id == "V7_PIPELINE":
        result["checks"] = {
            "prefill": True,
            "decode": True,
            "greedy_output": True,
            "boundary_parity": True,
            "final_logits_parity": True,
            "correctness_corpus": True,
        }
        result["numerical"] = {
            "reference_id": manifest["correctness"]["reference"]["reference_id"],
            "tolerance_approval_id": manifest["correctness"]["tolerance_policy"]["approval_id"],
            "max_abs_error": 0.001,
            "nonfinite_count": 0,
            "top1_mismatch_count": 0,
            "routing_mismatch_count": 0,
            "all_within_tolerance": True,
        }
        result["timings_ms"] = _timings()
        observations.extend(
            [
                {
                    "kind": "numerical",
                    "scope": scope,
                    "dtype": "bf16",
                    "reference_id": manifest["correctness"]["reference"]["reference_id"],
                    "reference": [0.0, 1.0],
                    "observed": [0.001, 1.0],
                    "reference_top1": 1,
                    "observed_top1": 1,
                    "reference_routes": [2, 3],
                    "observed_routes": [2, 3],
                }
                for scope in ("boundary", "final_logits")
            ]
        )
        observations.extend(
            {
                "kind": "generation",
                "phase": phase,
                "expected_token_ids": [1, 2],
                "observed_token_ids": [1, 2],
            }
            for phase in ("prefill", "decode")
        )
        observations.extend(_correctness_observations("correctness", manifest))
        observations.extend(_timing_observations())
    elif step_id == "V8_LOAD_CALIBRATION":
        result["checks"] = {
            "throughput_scales": True,
            "timing_attribution_recorded": True,
            "correctness_at_1_4_8": True,
        }
        result["planner_max_relative_error"] = 0.1
        result["concurrency"] = [
            {
                "inflight": value,
                "aggregate_tokens_per_second": float(value),
                "predicted_tokens_per_second": float(value) * 0.9,
            }
            for value in (1, 4, 8)
        ]
        result["timings_ms"] = _timings()
        observations.extend(
            {
                "kind": "concurrency",
                "inflight": value,
                "aggregate_tokens_per_second": float(value),
                "predicted_tokens_per_second": float(value) * 0.9,
            }
            for value in (1, 4, 8)
        )
        result["planner_max_relative_error"] = max(
            abs(row["predicted_tokens_per_second"] - row["aggregate_tokens_per_second"])
            / row["aggregate_tokens_per_second"]
            for row in observations
            if row["kind"] == "concurrency"
        )
        for value in (1, 4, 8):
            observations.extend(
                _correctness_observations(
                    "load_correctness", manifest, inflight=value
                )
            )
        observations.extend(_timing_observations())
    elif step_id == "V9_STABILITY":
        result["duration_seconds"] = 1800
        result["bounds"] = {
            "queue": True,
            "credit": True,
            "retained_state": True,
            "native_buffer": True,
            "kv": True,
            "rss": True,
        }
        result["high_water"] = {
            "queue_depth": 8,
            "credit_bytes_inflight": 1024,
            "retained_state_bytes": 2048,
            "native_buffer_bytes": 4096,
            "kv_bytes": 8192,
            "rss_bytes": 16384,
        }
        result["checks"] = {
            "sampling_cadence": True,
            "load_held": True,
            "request_accounting": True,
            "lifecycle_accounting": True,
            "post_drain_cleanup": True,
            "no_silent_divergence": True,
        }
        limits = {
            "queue_depth": 8,
            "credit_bytes_inflight": 1024,
            "retained_state_bytes": 2048,
            "native_buffer_bytes": 4096,
            "kv_bytes": 8192,
            "rss_bytes": 16384,
        }
        observations.extend(
            {
                "kind": "stability",
                "sample_index": sample_index,
                "elapsed_seconds": sample_index * 60,
                "phase": "load",
                "target_inflight": 8,
                "active_inflight": 8,
                "requests_started": sample_index * 10 + 8,
                "requests_completed": sample_index * 10,
                "requests_failed": 0,
                "explicit_releases": sample_index * 10,
                "expiry_releases": 0,
                "live_requests": 8,
                "queue_depth": 8,
                "credit_bytes_inflight": 1024,
                "retained_state_bytes": 2048,
                "native_buffer_bytes": 4096,
                "kv_bytes": 8192,
                "rss_bytes": 16384,
                "divergence_count": 0,
            }
            for sample_index in range(31)
        )
        observations.append(
            {
                "kind": "post_drain",
                "elapsed_seconds": 1801,
                "drain_wait_seconds": 1,
                "requests_started": 308,
                "requests_completed": 308,
                "requests_failed": 0,
                "explicit_releases": 308,
                "expiry_releases": 0,
                "live_requests": 0,
                "queue_depth": 0,
                "credit_bytes_inflight": 0,
                "retained_state_bytes": 0,
                "native_buffer_bytes": 0,
                "kv_bytes": 0,
                "inflight": 0,
            }
        )
    elif step_id == "V10_FAILURES":
        policies = {
            "cancel": ("CANCELLED", "terminal_rejected", 0),
            "timeout": ("DEADLINE_EXCEEDED", "terminal_rejected", 0),
            "stale_plan": ("STALE_PLAN", "terminal_rejected", 0),
            "crc": ("CRC_MISMATCH", "frame_rejected", 0),
            "link_loss": ("LINK_LOSS_RECOVERED", "replay_deduplicated", 1),
        }
        result["faults"] = {
            name: {
                "outcome_code": outcome,
                "passed": True,
                "cleanup_passed": True,
                "replay_passed": True,
                "mutation_passed": True,
            }
            for name, (outcome, _, _) in policies.items()
        }
        observations.extend(
            {
                "kind": "fault",
                "name": name,
                "expected_outcome_code": outcome,
                "observed_outcome_code": outcome,
                "state_before_sha256": "sha256:" + "1" * 64,
                "state_after_sha256": "sha256:" + ("2" if mutations else "1") * 64,
                "mutation_count": mutations,
                "replay_attempt_count": 1,
                "replay_execution_count": 0,
                "replay_disposition": disposition,
                "cleanup": {
                    "live_requests": 0,
                    "kv_bytes": 0,
                    "retained_state_bytes": 0,
                    "native_buffer_bytes": 0,
                    "inflight": 0,
                    "queue_depth": 0,
                    "credit_bytes_inflight": 0,
                },
            }
            for name, (outcome, disposition, mutations) in policies.items()
        )
    step_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "schema_version": 1,
        "step_id": step_id,
        "runner_nonce": os.environ.get("FORNAX_G2_RUN_NONCE"),
        "model": manifest["model"],
        "plan": manifest["plan"],
        "max_patch_commit": nodes["nvidia"]["max_patch_commit"],
        "node_observations": [
            {field: nodes[role][field] for field in NODE_IDENTITY_FIELDS}
            for role in roles
        ],
        "observations": observations,
    }
    if len(roles) > 1:
        raw["network_observation"] = {
            **{
                field: manifest["network"][field]
                for field in (
                    "source_host_id",
                    "destination_host_id",
                    "route",
                    "interface",
                    "mtu_bytes",
                    "declared_link_bits_per_second",
                )
            },
            "transfer_samples": [{"bytes": 1024, "duration_seconds": 0.001}],
        }
    if limits is not None:
        raw["limits"] = limits
    (step_dir / "raw-measurements.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (step_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _timings() -> dict[str, dict[str, float]]:
    return {
        name: {"median": 1.0, "p95": 1.5}
        for name in (
            "stage_0",
            "pack",
            "transfer",
            "unpack",
            "stage_1",
            "exposed_wait",
            "end_to_end",
        )
    }


def _timing_observations() -> list[dict[str, float | str]]:
    return [
        {"kind": "timing", "component": name, "milliseconds": value}
        for name in (
            "stage_0",
            "pack",
            "transfer",
            "unpack",
            "stage_1",
            "exposed_wait",
            "end_to_end",
        )
        for value in (1.0, 1.5)
    ]


def _correctness_observations(
    kind: str, manifest: dict, *, inflight: int | None = None
) -> list[dict]:
    contexts = (16, 128, 512, 4096)
    reference_id = manifest["correctness"]["reference"]["reference_id"]
    rows = []
    for index in range(20):
        row = {
            "kind": kind,
            "prompt_id": f"prompt-{index:02d}",
            "context_tokens": contexts[index % len(contexts)],
            "reference_id": reference_id,
            "expected_token_ids": [index + 1] * 128,
            "observed_token_ids": [index + 1] * 128,
            "reference_top1": index + 1,
            "observed_top1": index + 1,
            "reference_routes": [2, 3],
            "observed_routes": [2, 3],
        }
        if inflight is not None:
            row["inflight"] = inflight
        rows.append(row)
    return rows


if __name__ == "__main__":
    emit(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
