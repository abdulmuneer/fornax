"""Golden validation for the FNX2 ragged reference and loopback contract."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .io import read_json
from .ragged_runtime import (
    IntegratedRaggedScheduler,
    RaggedGenerationRequest,
    RaggedReferenceStageBackend,
    start_ragged_engine,
    stop_ragged_engine,
)
from .stage_abi_v2 import (
    BatchDescriptor,
    LogicalTensor,
    RaggedBatchRequest,
    RaggedFrame,
    RaggedStageManifest,
    decode_ragged_frame,
    derive_execution_lease_id,
    encode_ragged_frame,
)


def _fixture_file(path: str | Path) -> Path:
    value = Path(path)
    return value / "fixture.json" if value.is_dir() else value


def _tensor_digest(tensor: LogicalTensor | None) -> str | None:
    return None if tensor is None else tensor.digest


def _load_case(data: dict[str, Any]) -> tuple[
    tuple[RaggedStageManifest, ...],
    BatchDescriptor,
    LogicalTensor,
]:
    manifests = tuple(
        RaggedStageManifest.from_dict(item) for item in data.get("manifests", [])
    )
    descriptor = BatchDescriptor.from_dict(data.get("prefill", {}).get("batch", {}))
    raw_tensor = data.get("prefill", {}).get("tensor", {})
    tensor = LogicalTensor.from_values(
        raw_tensor.get("values", []),
        kind=str(raw_tensor.get("kind", "")),
        dtype=str(raw_tensor.get("dtype", "")),
        shape=tuple(int(item) for item in raw_tensor.get("shape", [])),
    )
    return manifests, descriptor, tensor


def validate_stage_abi_v2_golden(
    path_or_data: str | Path | dict[str, Any],
    *,
    run_loopback: bool = True,
) -> dict[str, Any]:
    fixture = "<memory>"
    try:
        if isinstance(path_or_data, dict):
            data = path_or_data
        else:
            fixture_path = _fixture_file(path_or_data)
            fixture = str(fixture_path)
            data = read_json(fixture_path)
        if not isinstance(data, dict):
            raise ValueError("fixture must be a JSON object")
        manifests, descriptor, tensor = _load_case(data)
    except Exception as exc:  # noqa: BLE001 - stable validator error report.
        return {
            "ok": False,
            "fixture": fixture,
            "errors": [f"invalid FNX2 fixture: {exc}"],
            "warnings": [],
            "summary": {},
        }

    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence[:512]})
        if not passed:
            errors.append(f"{name}: {evidence}")

    check("schema-version", data.get("schema_version") == 1, "schema_version must be 1")
    check(
        "record-kind",
        data.get("record_kind") == "fnx2-ragged-golden",
        "record_kind must be fnx2-ragged-golden",
    )
    check("two-stage-route", len(manifests) == 2, f"manifest_count={len(manifests)}")
    plan_generation_id = data.get("plan_generation_id")
    if len(manifests) != 2:
        return {
            "ok": False,
            "fixture": fixture,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
            "summary": {"check_count": len(checks), "passed_count": 0},
        }

    request = RaggedBatchRequest(
        plan_id=manifests[0].plan_id,
        plan_hash=manifests[0].plan_hash,
        manifest_hash=manifests[0].manifest_hash,
        descriptor=descriptor,
        input_tensor=tensor,
    )
    frame = RaggedFrame.from_batch(
        request,
        sequence_no=int(data.get("prefill", {}).get("wire_sequence_no", 0)),
        source_stage="gateway",
        destination_stage=manifests[0].stage_id,
    )
    encoded = encode_ragged_frame(frame)
    encoded_sha256 = "sha256:" + hashlib.sha256(encoded).hexdigest()
    expected_wire = data.get("expected", {}).get("wire", {})
    check(
        "exact-wire-bytes",
        encoded_sha256 == expected_wire.get("sha256")
        and len(encoded) == expected_wire.get("bytes"),
        f"sha256={encoded_sha256} bytes={len(encoded)}",
    )
    decoded = decode_ragged_frame(encoded)
    check(
        "wire-round-trip",
        decoded.batch_request() == request,
        "decoded logical request equals encoded request",
    )

    expected_leases = {
        item.request_id: derive_execution_lease_id(
            plan_generation_id,
            manifests[0].plan_id,
            manifests[0].plan_hash,
            item.request_id,
        )
        for item in descriptor.sequences
    }
    check(
        "generation-bound-leases",
        all(
            item.execution_lease_id == expected_leases[item.request_id]
            for item in descriptor.sequences
        ),
        f"plan_generation_id={plan_generation_id}",
    )
    direct = RaggedReferenceStageBackend(
        manifests[0], plan_generation_id=plan_generation_id
    ).execute(request)
    expected_direct = data.get("expected", {}).get("direct_stage0", {})
    check(
        "direct-unequal-prefill",
        [item.input_row_count for item in direct.results]
        == expected_direct.get("input_row_counts")
        and [item.status for item in direct.results]
        == expected_direct.get("statuses"),
        f"rows={[item.input_row_count for item in direct.results]} statuses={[item.status for item in direct.results]}",
    )
    check(
        "direct-output",
        _tensor_digest(direct.output_tensor) == expected_direct.get("output_digest")
        and list(direct.output_tensor.descriptor.shape if direct.output_tensor else ())
        == expected_direct.get("output_shape"),
        f"digest={_tensor_digest(direct.output_tensor)} shape={list(direct.output_tensor.descriptor.shape if direct.output_tensor else ())}",
    )

    worker_count = 0
    worker_pids: list[int] = []
    loopback_validated = False
    if run_loopback:
        workers = []
        channels = []
        try:
            workers, channels, orchestrator = start_ragged_engine(
                manifests, plan_generation_id=plan_generation_id
            )
            worker_pids = [int(worker.pid or 0) for worker in workers]
            worker_count = len(set(worker_pids))
            scheduler = IntegratedRaggedScheduler(
                manifests[0].plan_id,
                max_sequences=int(data.get("limits", {}).get("max_sequences", 4)),
                max_rows=int(data.get("limits", {}).get("max_rows", 32)),
            )
            generation_requests: list[RaggedGenerationRequest] = []
            for raw in data.get("requests", []):
                generation = RaggedGenerationRequest(
                    request_id=str(raw["request_id"]),
                    execution_lease_id=str(raw["execution_lease_id"]),
                    prompt_token_ids=tuple(int(item) for item in raw["prompt_token_ids"]),
                    deadline_ns=time.monotonic_ns() + 60_000_000_000,
                    max_new_tokens=int(raw["max_new_tokens"]),
                )
                generation_requests.append(generation)
                check(
                    f"scheduler-admit-{generation.request_id}",
                    scheduler.submit(generation),
                    "request admitted",
                )
            prefill = scheduler.run_prefill(orchestrator)
            if prefill is None:
                raise AssertionError("scheduler did not form the prefill batch")
            expected_loopback = data.get("expected", {}).get("loopback", {})
            check(
                "independent-workers",
                worker_count == 2,
                f"worker_pids={worker_pids}",
            )
            check(
                "loopback-prefill",
                _tensor_digest(prefill.output_tensor)
                == expected_loopback.get("prefill_output_digest")
                and list(prefill.output_tensor.descriptor.shape if prefill.output_tensor else ())
                == expected_loopback.get("prefill_output_shape"),
                f"digest={_tensor_digest(prefill.output_tensor)} shape={list(prefill.output_tensor.descriptor.shape if prefill.output_tensor else ())}",
            )
            decode_tokens = {
                str(key): int(value)
                for key, value in dict(data.get("decode_token_ids", {})).items()
            }
            decode = scheduler.run_decode(orchestrator, decode_tokens)
            if decode is None:
                raise AssertionError("scheduler did not form the decode batch")
            check(
                "loopback-independent-decode",
                _tensor_digest(decode.output_tensor)
                == expected_loopback.get("decode_output_digest")
                and list(decode.output_tensor.descriptor.shape if decode.output_tensor else ())
                == expected_loopback.get("decode_output_shape")
                and all(item.kv_epoch_after == 2 for item in decode.results),
                f"digest={_tensor_digest(decode.output_tensor)} kv={[item.kv_epoch_after for item in decode.results]}",
            )
            released = [scheduler.release(orchestrator, item) for item in generation_requests]
            check(
                "loopback-final-release",
                all(all(stage.get("released") for stage in result) for result in released),
                f"released={released}",
            )
            check(
                "integrated-scheduler-events",
                {"ragged_prefill_batch", "ragged_decode_batch", "ragged_request_released"}
                <= {item.get("kind") for item in scheduler.events},
                f"event_count={len(scheduler.events)}",
            )
            loopback_validated = True
        except Exception as exc:  # noqa: BLE001 - validator returns bounded errors.
            check("loopback-execution", False, f"{type(exc).__name__}: {exc}")
        finally:
            if workers or channels:
                stop_ragged_engine(workers, channels)
    else:
        warnings.append("loopback worker golden was not run")

    passed_count = sum(item["passed"] for item in checks)
    return {
        "ok": not errors,
        "fixture": fixture,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_count": passed_count,
            "manifest_count": len(manifests),
            "sequence_count": len(descriptor.sequences),
            "input_row_count": descriptor.input_row_count,
            "worker_count": worker_count,
            "worker_pids": worker_pids,
            "loopback_executed": bool(run_loopback),
            "loopback_validated": loopback_validated,
            "evidence_class": (
                "t0_t1_reference_loopback"
                if loopback_validated
                else "t0_reference_only"
            ),
            "physical_g2_passed": False,
        },
    }
