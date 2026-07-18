"""Public Stage Backend API and bounded functional conformance smoke."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from .stage_runtime import (
    BackendCapabilities,
    BufferAdapterHealth,
    CancelResult,
    DrainResult,
    ImportedTensorBuffer,
    LifecycleSweepResult,
    PythonTensorBufferAdapter,
    ReleaseResult,
    StageBackendSpec,
    StageExecutable,
    StageHandle,
    StageHealth,
    StageManifest,
    StageRequest,
    StageResult,
    StageRuntimeError,
    Tensor,
    TensorBufferAdapter,
    TensorDescriptor,
    UnloadResult,
    attest_backend_capabilities,
    create_stage_backend,
)


STAGE_BACKEND_API_VERSION = 2
_CHECK_NAMES = (
    "capability-attestation-before-load",
    "load-identity",
    "health-ready",
    "execute-success",
    "output-contract",
    "duplicate-at-most-once",
    "deadline-before-execution",
    "cancel-before-execution",
    "release-request-state",
    "bounded-state-retention",
    "drain",
    "unload",
)
_LIMITATIONS = (
    "No numerical-parity conclusion.",
    "No throughput or latency conclusion.",
    "No multi-node, cross-vendor, or supported-platform conclusion.",
    "This smoke does not close G2.",
)


def _report(
    *,
    manifest: StageManifest,
    factory: str,
    checks: list[dict[str, Any]],
    attestation: dict[str, Any] | None,
) -> dict[str, Any]:
    passed = sum(1 for item in checks if item["ok"])
    return {
        "schema_version": 1,
        "record_kind": "stage-backend-conformance",
        "backend_api_version": STAGE_BACKEND_API_VERSION,
        "evidence_class": "functional_contract_smoke",
        "ok": len(checks) == len(_CHECK_NAMES)
        and passed == len(checks),
        "manifest_hash": manifest.manifest_hash,
        "factory": factory,
        "capability_attestation": attestation or {},
        "checks": checks,
        "passed_count": passed,
        "check_count": len(checks),
        "closes_g2": False,
        "limitations": list(_LIMITATIONS),
    }


def _request(
    manifest: StageManifest,
    tensor: Tensor,
    *,
    request_id: str,
    sequence_no: int = 0,
    deadline_ns: int | None = None,
) -> StageRequest:
    return StageRequest(
        plan_id=manifest.plan_id,
        plan_hash=manifest.plan_hash,
        request_id=request_id,
        microbatch_id="backend-conformance-0",
        sequence_no=sequence_no,
        phase="prefill",
        token_start=0,
        token_count=tensor.descriptor.shape[0],
        input_activation=tensor,
        kv_epoch=0,
        deadline_ns=(
            time.monotonic_ns() + 60_000_000_000
            if deadline_ns is None
            else deadline_ns
        ),
        trace_context={
            "trace_id": "backend-conformance",
            "span_id": f"request-{sequence_no}",
        },
    )


def _input_tensor(manifest: StageManifest, rows: int) -> Tensor:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    width = int(manifest.input_contract["hidden_size"])
    values = [(((index * 17) % 31) - 15) / 32 for index in range(rows * width)]
    return Tensor.from_values(
        values,
        kind="activation",
        dtype=str(manifest.input_contract["dtype"]),
        shape=(rows, width),
    )


def check_stage_backend(
    backend: StageExecutable,
    manifest: StageManifest,
    *,
    rows: int = 2,
    factory: str = "<direct>",
) -> dict[str, Any]:
    """Exercise one backend lifecycle without making physical-evidence claims."""

    checks: list[dict[str, Any]] = []
    attestation: dict[str, Any] | None = None
    handle: StageHandle | None = None

    def record(name: str, ok: bool, evidence: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "evidence": evidence[:512]})

    try:
        attestation = attest_backend_capabilities(backend, manifest)
        record(
            "capability-attestation-before-load",
            attestation["compatible"]
            and attestation["checked_before_load"]
            and attestation["observed"]["source"] == "backend",
            "backend-originated capabilities satisfy the manifest",
        )
    except Exception as exc:  # noqa: BLE001 - report stable conformance failure.
        record("capability-attestation-before-load", False, str(exc))

    if checks[-1]["ok"]:
        try:
            handle = backend.load(manifest)
            record(
                "load-identity",
                handle.stage_id == manifest.stage_id
                and handle.manifest_hash == manifest.manifest_hash,
                "loaded handle binds the requested stage and manifest",
            )
        except Exception as exc:  # noqa: BLE001
            record("load-identity", False, str(exc))

    if handle is not None:
        try:
            health = backend.health(handle)
            ready = health.state == "READY" and health.inflight == 0
            record("health-ready", ready, f"state={health.state} inflight={health.inflight}")
        except Exception as exc:  # noqa: BLE001
            record("health-ready", False, str(exc))
            ready = False

        result: StageResult | None = None
        tensor: Tensor | None = None
        if ready:
            try:
                tensor = _input_tensor(manifest, rows)
                valid_request = _request(
                    manifest,
                    tensor,
                    request_id="11111111-1111-4111-8111-111111111111",
                )
                result = backend.execute(handle, valid_request)
                record(
                    "execute-success",
                    result.status == "ok"
                    and result.request_id == valid_request.request_id
                    and result.plan_id == manifest.plan_id
                    and result.plan_hash == manifest.plan_hash
                    and result.kv_epoch_before == 0
                    and result.kv_epoch_after == 1,
                    f"status={result.status} kv={result.kv_epoch_before}->{result.kv_epoch_after}",
                )
            except Exception as exc:  # noqa: BLE001
                record("execute-success", False, str(exc))

        if result is not None and tensor is not None:
            output = result.output_tensor
            expected_kind = str(manifest.output_contract.get("kind", "activation"))
            expected_shape = (
                rows,
                int(manifest.output_contract["hidden_size"]),
            )
            record(
                "output-contract",
                output is not None
                and output.descriptor.kind == expected_kind
                and output.descriptor.dtype == manifest.output_contract["dtype"]
                and output.descriptor.layout == manifest.output_contract["layout"]
                and output.descriptor.shape == expected_shape,
                "output tensor matches the declared kind/dtype/layout/shape",
            )
            try:
                duplicate = backend.execute(
                    handle,
                    _request(
                        manifest,
                        tensor,
                        request_id="11111111-1111-4111-8111-111111111111",
                    ),
                )
                record(
                    "duplicate-at-most-once",
                    duplicate == result and duplicate.kv_epoch_after == 1,
                    "identical request returns the cached StageResult",
                )
            except Exception as exc:  # noqa: BLE001
                record("duplicate-at-most-once", False, str(exc))

            try:
                expired = backend.execute(
                    handle,
                    _request(
                        manifest,
                        tensor,
                        request_id="22222222-2222-4222-8222-222222222222",
                        sequence_no=1,
                        deadline_ns=1,
                    ),
                )
                record(
                    "deadline-before-execution",
                    expired.status == "deadline"
                    and expired.kv_epoch_before == expired.kv_epoch_after == 0,
                    f"status={expired.status} kv={expired.kv_epoch_before}->{expired.kv_epoch_after}",
                )
            except Exception as exc:  # noqa: BLE001
                record("deadline-before-execution", False, str(exc))

            cancel_id = "33333333-3333-4333-8333-333333333333"
            try:
                cancelled_ack = backend.cancel(handle, cancel_id, "conformance")
                cancelled = backend.execute(
                    handle,
                    _request(
                        manifest,
                        tensor,
                        request_id=cancel_id,
                        sequence_no=2,
                    ),
                )
                record(
                    "cancel-before-execution",
                    cancelled_ack.cancelled
                    and not cancelled_ack.kv_mutated
                    and cancelled.status == "cancelled"
                    and cancelled.kv_epoch_before == cancelled.kv_epoch_after == 0,
                    f"status={cancelled.status} kv_mutated={cancelled_ack.kv_mutated}",
                )
            except Exception as exc:  # noqa: BLE001
                record("cancel-before-execution", False, str(exc))

        attempted = {item["name"] for item in checks}
        for name in _CHECK_NAMES[2:8]:
            if name not in attempted:
                record(name, False, "not run because a prerequisite failed")

        try:
            successful_release = backend.release(
                handle, "11111111-1111-4111-8111-111111111111"
            )
            cancelled_release = backend.release(
                handle, "33333333-3333-4333-8333-333333333333"
            )
            release_counts = {
                "successful": asdict(successful_release),
                "cancelled": asdict(cancelled_release),
            }
            record(
                "release-request-state",
                successful_release.released
                and successful_release.kv_state_released == 1
                and successful_release.execution_state_released == 1
                and successful_release.idempotency_results_released >= 1
                and cancelled_release.released
                and cancelled_release.cancel_state_released == 1,
                json.dumps(release_counts, sort_keys=True, separators=(",", ":")),
            )
        except Exception as exc:  # noqa: BLE001
            record("release-request-state", False, str(exc))

        try:
            retention = backend.health(handle)
            limits = {
                "max_live_requests": retention.max_live_requests,
                "max_completed_results_per_request": (
                    retention.max_completed_results_per_request
                ),
                "max_transform_cache_entries": (
                    retention.max_transform_cache_entries
                ),
                "max_completed_result_bytes": (
                    retention.max_completed_result_bytes
                ),
                "max_transform_cache_bytes": (
                    retention.max_transform_cache_bytes
                ),
            }
            counts = {
                "live_requests": retention.live_requests,
                "completed_results": retention.completed_results,
                "transform_cache_entries": retention.transform_cache_entries,
                "completed_result_bytes": retention.completed_result_bytes,
                "completed_result_high_water_bytes": (
                    retention.completed_result_high_water_bytes
                ),
                "transform_cache_bytes": retention.transform_cache_bytes,
                "transform_cache_high_water_bytes": (
                    retention.transform_cache_high_water_bytes
                ),
            }
            record(
                "bounded-state-retention",
                all(value > 0 for value in limits.values())
                and counts["live_requests"] == 0
                and counts["completed_results"] == 0
                and counts["transform_cache_entries"]
                <= limits["max_transform_cache_entries"]
                and counts["completed_result_bytes"]
                <= limits["max_completed_result_bytes"]
                and counts["completed_result_high_water_bytes"]
                <= limits["max_completed_result_bytes"]
                and counts["transform_cache_bytes"]
                <= limits["max_transform_cache_bytes"]
                and counts["transform_cache_high_water_bytes"]
                <= limits["max_transform_cache_bytes"],
                json.dumps(
                    {"counts": counts, "limits": limits},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            record("bounded-state-retention", False, str(exc))

        try:
            drained = backend.drain(handle, time.monotonic_ns() + 5_000_000_000)
            record("drain", drained.drained and drained.inflight == 0, f"inflight={drained.inflight}")
        except Exception as exc:  # noqa: BLE001
            record("drain", False, str(exc))
        try:
            unloaded = backend.unload(handle)
            record("unload", unloaded.unloaded, f"unloaded={unloaded.unloaded}")
        except Exception as exc:  # noqa: BLE001
            record("unload", False, str(exc))
    else:
        attempted = {item["name"] for item in checks}
        for name in _CHECK_NAMES:
            if name not in attempted:
                record(name, False, "not run because a prerequisite failed")

    checks.sort(key=lambda item: _CHECK_NAMES.index(item["name"]))
    return _report(
        manifest=manifest,
        factory=factory,
        checks=checks,
        attestation=attestation,
    )


def check_stage_backend_factory(
    factory: str,
    options: dict[str, Any],
    manifest: StageManifest,
    *,
    rows: int = 2,
) -> dict[str, Any]:
    """Import and check a physical adapter factory with no fallback backend."""

    try:
        spec = StageBackendSpec(kind="max", factory=factory, options=options)
        backend = create_stage_backend(spec)
    except Exception as exc:  # noqa: BLE001
        checks = [
            {
                "name": name,
                "ok": False,
                "evidence": str(exc)[:512]
                if index == 0
                else "not run because factory construction failed",
            }
            for index, name in enumerate(_CHECK_NAMES)
        ]
        return _report(
            manifest=manifest,
            factory=factory,
            checks=checks,
            attestation=None,
        )
    return check_stage_backend(
        backend,
        manifest,
        rows=rows,
        factory=factory,
    )


__all__ = [
    "STAGE_BACKEND_API_VERSION",
    "BackendCapabilities",
    "BufferAdapterHealth",
    "CancelResult",
    "DrainResult",
    "ImportedTensorBuffer",
    "LifecycleSweepResult",
    "PythonTensorBufferAdapter",
    "ReleaseResult",
    "StageBackendSpec",
    "StageExecutable",
    "StageHandle",
    "StageHealth",
    "StageManifest",
    "StageRequest",
    "StageResult",
    "StageRuntimeError",
    "Tensor",
    "TensorBufferAdapter",
    "TensorDescriptor",
    "UnloadResult",
    "attest_backend_capabilities",
    "check_stage_backend",
    "check_stage_backend_factory",
    "create_stage_backend",
]
