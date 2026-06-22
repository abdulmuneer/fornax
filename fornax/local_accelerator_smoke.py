from __future__ import annotations

from pathlib import Path
from typing import Any

from .accelerator_probe import (
    run_activation_transfer_probe,
    run_expert_mlp_probe,
    validate_activation_transfer_probe_fixture,
    validate_expert_mlp_probe_fixture,
)
from .io import read_json, write_json
from .moe_parity import (
    run_moe_layer_parity_probe,
    validate_moe_layer_parity_probe_fixture,
)
from .pipeline_probe import (
    run_pipeline_correctness_probe,
    validate_pipeline_correctness_probe_fixture,
)


RECORD_KIND = "local-accelerator-smoke-bundle"
EVIDENCE_SCOPE = "single-node-accelerator-smoke"


def _validation_entry(name: str, result: dict[str, Any], artifact: str) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(result.get("ok")),
        "artifact": artifact,
        "errors": list(result.get("errors", [])),
        "warnings": list(result.get("warnings", [])),
        "summary": result.get("summary", {}),
    }


def run_local_accelerator_smoke(
    *,
    out_dir: str | Path,
    torch_python: str | None = None,
    expert_backend: str = "torch",
    expert_device: str = "cuda:0",
    expert_dtype: str = "float16",
    expert_iterations: int = 25,
    expert_warmup: int = 3,
    expert_batch_tokens: int = 8,
    expert_hidden_dim: int = 64,
    expert_intermediate_dim: int = 128,
    expert_count: int = 4,
    expert_top_k: int = 2,
    expert_tolerance: float = 0.1,
    include_activation_transfer: bool = True,
    transfer_backend: str = "torch",
    transfer_source_device: str = "cuda:0",
    transfer_destination_device: str = "cuda:1",
    transfer_dtype: str = "float16",
    transfer_iterations: int = 20,
    transfer_warmup: int = 3,
    transfer_payload_bytes: int = 16 * 1024 * 1024,
    transfer_tolerance: float = 0.0,
    include_pipeline_correctness: bool = True,
    pipeline_backend: str = "torch",
    pipeline_source_device: str = "cuda:0",
    pipeline_destination_device: str = "cuda:1",
    pipeline_dtype: str = "float32",
    pipeline_iterations: int = 5,
    pipeline_warmup: int = 1,
    pipeline_vocab_size: int = 17,
    pipeline_hidden_dim: int = 16,
    pipeline_new_tokens: int = 4,
    pipeline_tolerance: float = 1e-4,
    include_moe_parity: bool = True,
    moe_backend: str = "torch",
    moe_source_device: str = "cuda:0",
    moe_expert_device: str = "cuda:1",
    moe_dtype: str = "float32",
    moe_iterations: int = 5,
    moe_warmup: int = 1,
    moe_token_count: int = 4,
    moe_hidden_dim: int = 16,
    moe_intermediate_dim: int = 32,
    moe_vocab_size: int = 17,
    moe_expert_count: int = 4,
    moe_top_k: int = 2,
    moe_tolerance: float = 1e-4,
    logical_source_host: str = "logical-host-0",
    logical_destination_host: str = "logical-host-1",
    require_accelerator: bool = True,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run and validate local H100/Torch accelerator smoke evidence."""

    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    expert_path = bundle / "expert-mlp-probe.json"
    transfer_path = bundle / "activation-transfer-probe.json"
    pipeline_path = bundle / "pipeline-correctness-probe.json"
    moe_path = bundle / "moe-layer-parity-probe.json"
    result_path = bundle / "local-accelerator-smoke.json"

    expert = run_expert_mlp_probe(
        backend=expert_backend,
        torch_python=torch_python,
        device=expert_device,
        dtype=expert_dtype,
        iterations=expert_iterations,
        warmup=expert_warmup,
        batch_tokens=expert_batch_tokens,
        hidden_dim=expert_hidden_dim,
        intermediate_dim=expert_intermediate_dim,
        experts=expert_count,
        top_k=expert_top_k,
        tolerance=expert_tolerance,
        timeout_s=timeout_s,
    )
    write_json(expert_path, expert)
    expert_validation = validate_expert_mlp_probe_fixture(expert)

    checks = [
        _validation_entry(
            "expert-mlp-probe",
            expert_validation,
            str(expert_path),
        )
    ]
    transfer: dict[str, Any] | None = None
    transfer_validation: dict[str, Any] | None = None
    if include_activation_transfer:
        transfer = run_activation_transfer_probe(
            backend=transfer_backend,
            torch_python=torch_python,
            source_device=transfer_source_device,
            destination_device=transfer_destination_device,
            dtype=transfer_dtype,
            iterations=transfer_iterations,
            warmup=transfer_warmup,
            payload_bytes=transfer_payload_bytes,
            tolerance=transfer_tolerance,
            logical_source_host=logical_source_host,
            logical_destination_host=logical_destination_host,
            timeout_s=timeout_s,
        )
        write_json(transfer_path, transfer)
        transfer_validation = validate_activation_transfer_probe_fixture(transfer)
        checks.append(
            _validation_entry(
                "activation-transfer-probe",
                transfer_validation,
                str(transfer_path),
            )
        )


    pipeline_validation: dict[str, Any] | None = None
    if include_pipeline_correctness:
        pipeline = run_pipeline_correctness_probe(
            backend=pipeline_backend,
            torch_python=torch_python,
            source_device=pipeline_source_device,
            destination_device=pipeline_destination_device,
            dtype=pipeline_dtype,
            iterations=pipeline_iterations,
            warmup=pipeline_warmup,
            vocab_size=pipeline_vocab_size,
            hidden_dim=pipeline_hidden_dim,
            new_tokens=pipeline_new_tokens,
            tolerance=pipeline_tolerance,
            logical_source_host=logical_source_host,
            logical_destination_host=logical_destination_host,
            timeout_s=timeout_s,
        )
        write_json(pipeline_path, pipeline)
        pipeline_validation = validate_pipeline_correctness_probe_fixture(pipeline)
        checks.append(
            _validation_entry(
                "pipeline-correctness-probe",
                pipeline_validation,
                str(pipeline_path),
            )
        )

    moe_validation: dict[str, Any] | None = None
    if include_moe_parity:
        moe = run_moe_layer_parity_probe(
            backend=moe_backend,
            torch_python=torch_python,
            source_device=moe_source_device,
            expert_device=moe_expert_device,
            dtype=moe_dtype,
            iterations=moe_iterations,
            warmup=moe_warmup,
            token_count=moe_token_count,
            hidden_dim=moe_hidden_dim,
            intermediate_dim=moe_intermediate_dim,
            vocab_size=moe_vocab_size,
            expert_count=moe_expert_count,
            top_k=moe_top_k,
            tolerance=moe_tolerance,
            logical_source_host=logical_source_host,
            logical_expert_host=logical_destination_host,
            timeout_s=timeout_s,
        )
        write_json(moe_path, moe)
        moe_validation = validate_moe_layer_parity_probe_fixture(moe)
        checks.append(
            _validation_entry(
                "moe-layer-parity-probe",
                moe_validation,
                str(moe_path),
            )
        )

    bundle_policy_errors: list[str] = []
    expert_summary = expert_validation.get("summary", {})
    if require_accelerator and expert_summary.get("accelerator_measured") is not True:
        bundle_policy_errors.append(
            "expert-mlp-probe must be measured accelerator evidence"
        )
    transfer_summary = (
        transfer_validation.get("summary", {}) if isinstance(transfer_validation, dict) else {}
    )
    if include_activation_transfer and require_accelerator:
        if transfer_summary.get("accelerator_measured") is not True:
            bundle_policy_errors.append(
                "activation-transfer-probe must be measured accelerator evidence"
            )
    pipeline_summary = (
        pipeline_validation.get("summary", {}) if isinstance(pipeline_validation, dict) else {}
    )
    if include_pipeline_correctness and require_accelerator:
        if pipeline_summary.get("accelerator_measured") is not True:
            bundle_policy_errors.append(
                "pipeline-correctness-probe must be measured accelerator evidence"
            )
    moe_summary = (
        moe_validation.get("summary", {}) if isinstance(moe_validation, dict) else {}
    )
    if include_moe_parity and require_accelerator:
        if moe_summary.get("accelerator_measured") is not True:
            bundle_policy_errors.append(
                "moe-layer-parity-probe must be measured accelerator evidence"
            )
    checks.append(
        {
            "name": "bundle-policy",
            "ok": not bundle_policy_errors,
            "artifact": str(result_path),
            "errors": bundle_policy_errors,
            "warnings": [
                "local accelerator smoke is not Apple rank-1 probe evidence",
                "same-host activation transfer is not real multi-host T3 evidence",
            ],
            "summary": {
                "require_accelerator": require_accelerator,
                "include_activation_transfer": include_activation_transfer,
                "include_pipeline_correctness": include_pipeline_correctness,
                "include_moe_parity": include_moe_parity,
            },
        }
    )

    passed_count = sum(1 for check in checks if check["ok"])
    accelerator_probe_count = sum(
        1
        for check in checks
        if check["name"] != "bundle-policy"
        and check.get("summary", {}).get("accelerator_measured") is True
    )
    required_accelerator_probe_count = (
        1
        + int(include_activation_transfer)
        + int(include_pipeline_correctness)
        + int(include_moe_parity)
    )
    local_smoke_passed = passed_count == len(checks)
    t2_smoke_passed = (
        local_smoke_passed
        and accelerator_probe_count == required_accelerator_probe_count
    )
    summary = {
        "accelerator_required": require_accelerator,
        "check_count": len(checks),
        "passed_count": passed_count,
        "expert_measured": bool(expert_summary.get("measured")),
        "expert_accelerator_measured": bool(
            expert_summary.get("accelerator_measured")
        ),
        "expert_device": expert_summary.get("device"),
        "expert_device_name": expert_summary.get("device_name"),
        "expert_tokens_s": expert_summary.get("tokens_s"),
        "activation_transfer_included": include_activation_transfer,
        "activation_transfer_accelerator_measured": bool(
            transfer_summary.get("accelerator_measured")
        ),
        "activation_transfer_source_device": transfer_summary.get("source_device"),
        "activation_transfer_destination_device": transfer_summary.get(
            "destination_device"
        ),
        "activation_transfer_bandwidth_gib_s": transfer_summary.get(
            "bandwidth_gib_s"
        ),
        "pipeline_correctness_included": include_pipeline_correctness,
        "pipeline_correctness_accelerator_measured": bool(
            pipeline_summary.get("accelerator_measured")
        ),
        "pipeline_correctness_source_device": pipeline_summary.get("source_device"),
        "pipeline_correctness_destination_device": pipeline_summary.get(
            "destination_device"
        ),
        "pipeline_correctness_tokens_s": pipeline_summary.get("tokens_s"),
        "pipeline_correctness_max_abs_error": pipeline_summary.get("max_abs_error"),
        "moe_parity_included": include_moe_parity,
        "moe_parity_accelerator_measured": bool(
            moe_summary.get("accelerator_measured")
        ),
        "moe_parity_source_device": moe_summary.get("source_device"),
        "moe_parity_expert_device": moe_summary.get("expert_device"),
        "moe_parity_tokens_s": moe_summary.get("tokens_s"),
        "moe_parity_expert_calls_s": moe_summary.get("expert_calls_s"),
        "moe_parity_max_logit_abs_error": moe_summary.get("max_logit_abs_error"),
        "accelerator_probe_count": accelerator_probe_count,
        "required_accelerator_probe_count": required_accelerator_probe_count,
        "local_smoke_passed": local_smoke_passed,
        "t2_smoke_passed": t2_smoke_passed,
        "g2_g3_gate_evidence": False,
    }
    result = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "bundle": str(bundle),
        "artifacts": {
            "expert_mlp_probe": str(expert_path),
            "activation_transfer_probe": str(transfer_path)
            if include_activation_transfer
            else None,
            "pipeline_correctness_probe": str(pipeline_path)
            if include_pipeline_correctness
            else None,
            "moe_layer_parity_probe": str(moe_path)
            if include_moe_parity
            else None,
            "validation": str(result_path),
        },
        "summary": summary,
        "checks": checks,
        "ok": passed_count == len(checks),
        "note": (
            "Local accelerator smoke evidence for tiny probe paths. This can "
            "advance T2 readiness on the current machine, but it is not target "
            "model parity, Apple rank-1 probe evidence, real 2-3 node pipeline "
            "evidence, or G2/G3 closure evidence."
        ),
    }
    write_json(result_path, result)
    return result


def validate_local_accelerator_smoke_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "local accelerator smoke is not target-model, Apple, or multi-host gate evidence"
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must be {EVIDENCE_SCOPE}")
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("checks must be a non-empty list")
        checks = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"checks[{index}] must be an object")
            continue
        if not check.get("name"):
            errors.append(f"checks[{index}].name must be set")
        if check.get("ok") is not True:
            errors.append(
                f"checks[{index}] {check.get('name', '<unknown>')} must pass"
            )
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    passed_count = sum(
        1 for check in checks if isinstance(check, dict) and check.get("ok") is True
    )
    if summary.get("check_count") != len(checks):
        errors.append("summary.check_count must match checks")
    if summary.get("passed_count") != passed_count:
        errors.append("summary.passed_count must match checks")
    if summary.get("expert_measured") is not True:
        errors.append("summary.expert_measured must be true")
    if summary.get("local_smoke_passed") is not True:
        errors.append("summary.local_smoke_passed must be true")
    if (
        summary.get("accelerator_required") is True
        and summary.get("t2_smoke_passed") is not True
    ):
        errors.append("summary.t2_smoke_passed must be true when accelerator is required")
    if summary.get("g2_g3_gate_evidence") is not False:
        errors.append("summary.g2_g3_gate_evidence must be false")
    if data.get("ok") is not True:
        errors.append("ok must be true")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "check_count": summary.get("check_count"),
            "passed_count": passed_count,
            "expert_accelerator_measured": summary.get(
                "expert_accelerator_measured"
            ),
            "activation_transfer_included": summary.get(
                "activation_transfer_included"
            ),
            "activation_transfer_accelerator_measured": summary.get(
                "activation_transfer_accelerator_measured"
            ),
            "pipeline_correctness_included": summary.get(
                "pipeline_correctness_included"
            ),
            "pipeline_correctness_accelerator_measured": summary.get(
                "pipeline_correctness_accelerator_measured"
            ),
            "moe_parity_included": summary.get("moe_parity_included"),
            "moe_parity_accelerator_measured": summary.get(
                "moe_parity_accelerator_measured"
            ),
            "accelerator_probe_count": summary.get("accelerator_probe_count"),
            "required_accelerator_probe_count": summary.get(
                "required_accelerator_probe_count"
            ),
            "local_smoke_passed": summary.get("local_smoke_passed") is True,
            "t2_smoke_passed": summary.get("t2_smoke_passed") is True,
        },
    }


def validate_local_accelerator_smoke(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "local-accelerator-smoke.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid local accelerator smoke artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["local accelerator smoke artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_local_accelerator_smoke_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
