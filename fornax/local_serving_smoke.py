from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .moe_parity import (
    run_moe_layer_parity_probe,
    validate_moe_layer_parity_probe_fixture,
)
from .pipeline_probe import (
    run_pipeline_correctness_probe,
    validate_pipeline_correctness_probe_fixture,
)
from .serving import simulate_serving_adapter, validate_serving_adapter_fixture


RECORD_KIND = "local-serving-runtime-smoke-bundle"
EVIDENCE_SCOPE = "single-node-serving-runtime-smoke"


def _validation_entry(name: str, result: dict[str, Any], artifact: str) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(result.get("ok")),
        "artifact": artifact,
        "errors": list(result.get("errors", [])),
        "warnings": list(result.get("warnings", [])),
        "summary": result.get("summary", {}),
    }


def run_local_serving_smoke(
    *,
    out_dir: str | Path,
    torch_python: str | None = None,
    plan_id: str = "local-serving-smoke-plan",
    request_id: str = "local-serving-smoke-request",
    model: str = "qwen3-moe-class-target",
    stream: bool = True,
    max_tokens: int = 64,
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
    """Run a local serving/runtime smoke bundle with explicit evidence boundaries."""

    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    serving_path = bundle / "serving-adapter.json"
    pipeline_path = bundle / "pipeline-correctness-probe.json"
    moe_path = bundle / "moe-layer-parity-probe.json"
    result_path = bundle / "local-serving-smoke.json"

    serving = simulate_serving_adapter(
        plan_id=plan_id,
        request_id=request_id,
        model=model,
        stream=stream,
        max_tokens=max_tokens,
    )
    write_json(serving_path, serving)
    serving_validation = validate_serving_adapter_fixture(serving)
    checks = [
        _validation_entry(
            "serving-adapter",
            serving_validation,
            str(serving_path),
        )
    ]

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

    pipeline_summary = (
        pipeline_validation.get("summary", {}) if isinstance(pipeline_validation, dict) else {}
    )
    moe_summary = (
        moe_validation.get("summary", {}) if isinstance(moe_validation, dict) else {}
    )
    bundle_policy_errors: list[str] = []
    if require_accelerator and include_pipeline_correctness:
        if pipeline_summary.get("accelerator_measured") is not True:
            bundle_policy_errors.append(
                "pipeline-correctness-probe must be measured accelerator evidence"
            )
    if require_accelerator and include_moe_parity:
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
                "serving adapter is not live HTTP endpoint evidence",
                "local runtime smoke is not target-model parity evidence",
                "same-host accelerator probes are not real multi-host T3 evidence",
            ],
            "summary": {
                "require_accelerator": require_accelerator,
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
    required_accelerator_probe_count = int(include_pipeline_correctness) + int(
        include_moe_parity
    )
    local_runtime_smoke_passed = passed_count == len(checks)
    t2_smoke_passed = (
        local_runtime_smoke_passed
        and accelerator_probe_count == required_accelerator_probe_count
    )
    serving_summary = serving_validation.get("summary", {})
    summary = {
        "accelerator_required": require_accelerator,
        "check_count": len(checks),
        "passed_count": passed_count,
        "serving_adapter_valid": bool(serving_validation.get("ok")),
        "serving_surface_count": serving_summary.get("surface_count"),
        "serving_openai_chunk_count": serving_summary.get("openai_chunk_count"),
        "serving_engine_stream_event_count": serving_summary.get(
            "engine_stream_event_count"
        ),
        "serving_tool_call_count": serving_summary.get("tool_call_count"),
        "serving_correctness_passed": bool(serving_summary.get("correctness_passed")),
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
        "local_runtime_smoke_passed": local_runtime_smoke_passed,
        "t2_smoke_passed": t2_smoke_passed,
        "live_http_endpoint": False,
        "target_model_parity": False,
        "g2_g3_gate_evidence": False,
    }
    result = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "bundle": str(bundle),
        "artifacts": {
            "serving_adapter": str(serving_path),
            "pipeline_correctness_probe": str(pipeline_path)
            if include_pipeline_correctness
            else None,
            "moe_layer_parity_probe": str(moe_path) if include_moe_parity else None,
            "validation": str(result_path),
        },
        "summary": summary,
        "checks": checks,
        "ok": local_runtime_smoke_passed,
        "note": (
            "Local serving/runtime smoke evidence for the OpenAI/Ignis serving "
            "adapter plus tiny local accelerator runtime probes. This can advance "
            "T2 development readiness on the current machine, but it is not a live "
            "HTTP endpoint, target-model parity, real multi-host pipeline evidence, "
            "or G2/G3 closure evidence."
        ),
    }
    write_json(result_path, result)
    return result


def validate_local_serving_smoke_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "local serving smoke is not live endpoint, target-model, or multi-host gate evidence"
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
    if summary.get("serving_adapter_valid") is not True:
        errors.append("summary.serving_adapter_valid must be true")
    if summary.get("serving_correctness_passed") is not True:
        errors.append("summary.serving_correctness_passed must be true")
    if summary.get("local_runtime_smoke_passed") is not True:
        errors.append("summary.local_runtime_smoke_passed must be true")
    if (
        summary.get("accelerator_required") is True
        and summary.get("t2_smoke_passed") is not True
    ):
        errors.append("summary.t2_smoke_passed must be true when accelerator is required")
    if summary.get("live_http_endpoint") is not False:
        errors.append("summary.live_http_endpoint must be false")
    if summary.get("target_model_parity") is not False:
        errors.append("summary.target_model_parity must be false")
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
            "serving_adapter_valid": summary.get("serving_adapter_valid") is True,
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
            "local_runtime_smoke_passed": summary.get("local_runtime_smoke_passed")
            is True,
            "t2_smoke_passed": summary.get("t2_smoke_passed") is True,
        },
    }


def validate_local_serving_smoke(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "local-serving-smoke.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid local serving smoke artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["local serving smoke artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_local_serving_smoke_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
