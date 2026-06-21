
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_CAPABILITIES = (
    "architecture",
    "tokenizer",
    "chat_template",
    "quantization",
    "context_length",
    "moe_routing",
    "stop_behavior",
    "streaming",
    "tool_calling",
    "structured_output",
)
ROLE_VALUES = {"reference_fixture", "target_candidate", "fallback_candidate"}
SUPPORT_LEVELS = {"supported", "limited", "planned", "not_supported"}
STATUS_VALUES = {
    "supported",
    "limited",
    "planned",
    "not_supported",
    "unknown",
    "not_applicable",
}
EVIDENCE_KINDS = {"fixture", "spec", "measured", "required", "manual_review"}
HASH_STATUS_VALUES = {"resolved", "required_before_t2", "not_applicable"}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _fixture_evidence(source: str, note: str) -> dict[str, Any]:
    return {"kind": "fixture", "source": source, "note": note}


def _required_evidence(source: str, requirement: str) -> dict[str, Any]:
    return {
        "kind": "required",
        "source": source,
        "requirement": requirement,
        "blocking_tier": "T2/T3",
    }


def _fixture_hash(char: str) -> str:
    return "sha256:" + char * 64


def simulated_model_support_matrix(
    *,
    matrix_id: str = "fornax-model-support-t1",
    target_model_id: str = "qwen3-moe-class-target",
    target_contract: str = "fornax/golden_plans/v0_target_contract_fixture.md",
) -> dict[str, Any]:
    """Build a deterministic T1 model-support matrix contract."""

    if not matrix_id:
        raise ValueError("matrix_id must be non-empty")
    if not target_model_id:
        raise ValueError("target_model_id must be non-empty")
    if not target_contract:
        raise ValueError("target_contract must be non-empty")

    fixture_source = "fornax.golden_vectors.model_support"
    plan_source = "docs/fornax/project-plan-v3.md#phase-2-5"
    engine_seam_source = "fornax/golden_vectors/engine_seam/fixture.json"
    return {
        "version": 1,
        "contract_kind": "model-support-matrix",
        "mode": "t1-simulation",
        "matrix_id": matrix_id,
        "owner": "fornax.serving",
        "target_contract": target_contract,
        "required_capabilities": list(REQUIRED_CAPABILITIES),
        "models": [
            {
                "model_id": "fornax-tiny-moe-fixture",
                "role": "reference_fixture",
                "support_level": "supported",
                "architecture": {
                    "status": "supported",
                    "kind": "decoder_only_sparse_moe",
                    "moe": True,
                    "num_layers": 2,
                    "num_experts": 4,
                    "active_experts": 2,
                    "evidence": _fixture_evidence(
                        fixture_source,
                        "Tiny model-free MoE fixture used to validate support semantics.",
                    ),
                },
                "tokenizer": {
                    "status": "supported",
                    "owner": "fornax.model_support",
                    "source": "fixture://fornax-tiny-moe-tokenizer",
                    "version": "fixture-v1",
                    "hash_status": "resolved",
                    "hash": _fixture_hash("1"),
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "special_tokens": ["<s>", "</s>", "<tool_call>"],
                    "evidence": _fixture_evidence(
                        engine_seam_source,
                        "Engine seam fixture records tokenizer hash and request/result propagation.",
                    ),
                },
                "chat_template": {
                    "status": "supported",
                    "owner": "fornax.model_support",
                    "source": "fixture://fornax-tiny-moe-chat-template",
                    "version": "fixture-v1",
                    "hash_status": "resolved",
                    "hash": _fixture_hash("2"),
                    "roles": ["system", "user", "assistant", "tool"],
                    "tool_call_format": "openai_compatible_json",
                    "evidence": _fixture_evidence(
                        engine_seam_source,
                        "Engine seam fixture covers roles, tools, response format, streaming, and cancellation.",
                    ),
                },
                "quantization": {
                    "status": "supported",
                    "weight_dtypes": ["fp16"],
                    "activation_dtypes": ["fp16"],
                    "kv_dtypes": ["fp16"],
                    "evidence": _fixture_evidence(
                        "fornax/golden_vectors/runtime_format/manifest.json",
                        "Runtime-format golden vector validates dtype and tensor layout shape.",
                    ),
                },
                "context_length": {
                    "status": "supported",
                    "max_context_tokens": 64,
                    "max_batch_tokens": 128,
                    "owner": "target_contract",
                    "evidence": _fixture_evidence(
                        target_contract,
                        "Fixture context is synthetic; real target context must be supplied by G1/G2 evidence.",
                    ),
                },
                "moe_routing": {
                    "status": "supported",
                    "router": "top_k",
                    "top_k": 2,
                    "routing_weighting": "softmax",
                    "expert_trace_required": True,
                    "evidence": _fixture_evidence(
                        "fornax/golden_vectors/moe_runtime/fixture.json",
                        "MoE runtime fixture validates top-k, expert buckets, dispatch, gather, and traces.",
                    ),
                },
                "serving_semantics": {
                    "stop_behavior": {
                        "status": "supported",
                        "stop_sequences": ["</s>"],
                        "stop_token_ids": [2],
                        "evidence": _fixture_evidence(
                            engine_seam_source,
                            "Engine seam fixture validates stop sequence propagation.",
                        ),
                    },
                    "streaming": {
                        "status": "supported",
                        "events": ["start", "token", "finish", "error", "cancelled"],
                        "evidence": _fixture_evidence(
                            engine_seam_source,
                            "Engine seam fixture validates stream start/token/finish events.",
                        ),
                    },
                    "tool_calling": {
                        "status": "supported",
                        "format": "openai_compatible_function_call",
                        "evidence": _fixture_evidence(
                            engine_seam_source,
                            "Engine seam fixture validates tool-call result shape.",
                        ),
                    },
                    "structured_output": {
                        "status": "supported",
                        "formats": ["json_object", "json_schema"],
                        "evidence": _fixture_evidence(
                            engine_seam_source,
                            "Engine seam fixture validates structured-output object shape.",
                        ),
                    },
                },
                "parity": {
                    "status": "fixture_only",
                    "reference_path": "fornax.model_support.fixture",
                    "layer_logit_parity": "not_applicable_to_model_free_fixture",
                    "required_before_tier": "T2/T3",
                    "evidence": _fixture_evidence(
                        fixture_source,
                        "Model-free support contract only; real layer/logit parity remains required.",
                    ),
                },
            },
            {
                "model_id": target_model_id,
                "role": "target_candidate",
                "support_level": "planned",
                "architecture": {
                    "status": "planned",
                    "kind": "decoder_only_sparse_moe",
                    "moe": True,
                    "num_layers": None,
                    "num_experts": None,
                    "active_experts": None,
                    "evidence": _required_evidence(
                        plan_source,
                        "Resolve exact target architecture from the signed v0 target contract before T2 parity.",
                    ),
                },
                "tokenizer": {
                    "status": "planned",
                    "owner": "fornax.model_support",
                    "source": "model_artifact_required",
                    "version": "unresolved",
                    "hash_status": "required_before_t2",
                    "hash_required_before": "T2 single-node accelerator parity",
                    "bos_token": None,
                    "eos_token": None,
                    "special_tokens": [],
                    "evidence": _required_evidence(
                        plan_source,
                        "Record canonical tokenizer version and sha256 hash before executing target model requests.",
                    ),
                },
                "chat_template": {
                    "status": "planned",
                    "owner": "fornax.model_support",
                    "source": "model_artifact_required",
                    "version": "unresolved",
                    "hash_status": "required_before_t2",
                    "hash_required_before": "T2 single-node accelerator parity",
                    "roles": ["system", "user", "assistant", "tool"],
                    "tool_call_format": "openai_compatible_json_required",
                    "evidence": _required_evidence(
                        plan_source,
                        "Record canonical chat-template source and sha256 hash before target serving claims.",
                    ),
                },
                "quantization": {
                    "status": "planned",
                    "weight_dtypes": ["q4"],
                    "activation_dtypes": ["fp16", "bf16"],
                    "kv_dtypes": ["fp16", "bf16"],
                    "evidence": _required_evidence(
                        target_contract,
                        "Signed target contract must declare quantization and dtype choices.",
                    ),
                },
                "context_length": {
                    "status": "planned",
                    "max_context_tokens": 24,
                    "max_batch_tokens": 96,
                    "owner": "target_contract",
                    "evidence": _required_evidence(
                        target_contract,
                        "Signed target contract must replace fixture prompt/gen lengths with real target context.",
                    ),
                },
                "moe_routing": {
                    "status": "planned",
                    "router": "top_k",
                    "top_k": 2,
                    "routing_weighting": "model_artifact_required",
                    "expert_trace_required": True,
                    "evidence": _required_evidence(
                        "fornax/golden_vectors/moe_runtime/fixture.json",
                        "Attach target-model routing traces and layer/logit parity before Phase 2.5 exit.",
                    ),
                },
                "serving_semantics": {
                    "stop_behavior": {
                        "status": "planned",
                        "stop_sequences": [],
                        "stop_token_ids": [],
                        "evidence": _required_evidence(
                            engine_seam_source,
                            "Target model stop strings and stop token IDs must be recorded before T2.",
                        ),
                    },
                    "streaming": {
                        "status": "planned",
                        "events": ["start", "token", "finish", "error", "cancelled"],
                        "evidence": _required_evidence(
                            engine_seam_source,
                            "Target streaming chunk boundaries must match Engine seam acceptance tests.",
                        ),
                    },
                    "tool_calling": {
                        "status": "planned",
                        "format": "openai_compatible_function_call",
                        "evidence": _required_evidence(
                            engine_seam_source,
                            "Tool-call pass-through must be validated for the target template.",
                        ),
                    },
                    "structured_output": {
                        "status": "planned",
                        "formats": ["json_object", "json_schema"],
                        "evidence": _required_evidence(
                            engine_seam_source,
                            "Structured-output behavior must be validated for the target template.",
                        ),
                    },
                },
                "parity": {
                    "status": "required_before_t2",
                    "reference_path": "slow_reference_target_model_required",
                    "layer_logit_parity": "required_before_phase_2_5_exit",
                    "required_before_tier": "T2/T3",
                    "evidence": _required_evidence(
                        plan_source,
                        "Layer/logit parity against the reference path is the Phase 2.5 exit criterion.",
                    ),
                },
            },
        ],
        "summary": {
            "model_count": 2,
            "supported_model_count": 1,
            "planned_model_count": 1,
            "required_capability_count": len(REQUIRED_CAPABILITIES),
            "target_model_id": target_model_id,
        },
        "note": (
            "T1 model-support matrix contract; validates ownership and evidence "
            "shape for architecture, tokenizer, chat template, quantization, "
            "context, MoE routing, stop behavior, streaming, tools, and "
            "structured output without claiming real target-model parity."
        ),
    }


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _bool(value: Any, field: str, errors: list[str]) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{field} must be a boolean")
        return None
    return value


def _positive_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return None
    return value


def _string_list(
    value: Any,
    field: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> list[str] | None:
    if not isinstance(value, list) or (not value and not allow_empty):
        errors.append(f"{field} must be a {'list' if allow_empty else 'non-empty list'}")
        return None
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
            return None
        result.append(item)
    return result


def _int_list(
    value: Any,
    field: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> list[int] | None:
    if not isinstance(value, list) or (not value and not allow_empty):
        errors.append(f"{field} must be a {'list' if allow_empty else 'non-empty list'}")
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            errors.append(f"{field}[{index}] must be an integer")
            return None
        result.append(item)
    return result


def _hash(value: Any, field: str, errors: list[str]) -> str | None:
    text = _non_empty_string(value, field, errors)
    if text is not None and not SHA256_RE.match(text):
        errors.append(f"{field} must be a sha256:<64 lowercase hex chars> hash")
    return text


def _status(value: Any, field: str, errors: list[str]) -> str | None:
    status = _non_empty_string(value, field, errors)
    if status is not None and status not in STATUS_VALUES:
        errors.append(f"{field} must be one of {sorted(STATUS_VALUES)}")
    return status


def _evidence(
    value: Any,
    field: str,
    status: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{field}.evidence must be an object")
        return
    kind = _non_empty_string(value.get("kind"), f"{field}.evidence.kind", errors)
    _non_empty_string(value.get("source"), f"{field}.evidence.source", errors)
    if kind is not None and kind not in EVIDENCE_KINDS:
        errors.append(f"{field}.evidence.kind must be one of {sorted(EVIDENCE_KINDS)}")
    if kind == "measured":
        _string_list(value.get("command"), f"{field}.evidence.command", errors)
        _non_empty_string(value.get("artifact"), f"{field}.evidence.artifact", errors)
    if kind == "required":
        _non_empty_string(value.get("requirement"), f"{field}.evidence.requirement", errors)
        _non_empty_string(value.get("blocking_tier"), f"{field}.evidence.blocking_tier", errors)
    if status in {"supported", "limited"} and kind == "required":
        errors.append(f"{field}.evidence cannot be required-only for {status} status")
    if status in {"planned", "unknown"}:
        reason = value.get("requirement") or value.get("note")
        if not isinstance(reason, str) or not reason:
            errors.append(f"{field}.evidence must explain {status} status")
    if kind == "fixture":
        warnings.append(f"{field}.evidence is fixture-only, not gate evidence")


def _check_hash_status(
    value: dict[str, Any],
    field: str,
    status: str | None,
    errors: list[str],
) -> None:
    hash_status = _non_empty_string(value.get("hash_status"), f"{field}.hash_status", errors)
    if hash_status is not None and hash_status not in HASH_STATUS_VALUES:
        errors.append(f"{field}.hash_status must be one of {sorted(HASH_STATUS_VALUES)}")
    if hash_status == "resolved":
        _hash(value.get("hash"), f"{field}.hash", errors)
    elif hash_status == "required_before_t2":
        _non_empty_string(
            value.get("hash_required_before"),
            f"{field}.hash_required_before",
            errors,
        )
        if status in {"supported", "limited"}:
            errors.append(f"{field}.hash_status must be resolved for {status} status")
    elif hash_status == "not_applicable" and status in {"supported", "limited"}:
        errors.append(f"{field}.hash_status cannot be not_applicable for {status} status")


def _check_architecture(value: Any, field: str, errors: list[str], warnings: list[str]) -> str | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    status = _status(value.get("status"), f"{field}.status", errors)
    _non_empty_string(value.get("kind"), f"{field}.kind", errors)
    _bool(value.get("moe"), f"{field}.moe", errors)
    for number_field in ("num_layers", "num_experts", "active_experts"):
        number = value.get(number_field)
        if number is None:
            if status in {"supported", "limited"}:
                errors.append(f"{field}.{number_field} is required for {status} status")
        else:
            _positive_int(number, f"{field}.{number_field}", errors)
    _evidence(value.get("evidence"), field, status, errors, warnings)
    return status


def _check_tokenizer_or_template(
    value: Any,
    field: str,
    errors: list[str],
    warnings: list[str],
) -> str | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    status = _status(value.get("status"), f"{field}.status", errors)
    _non_empty_string(value.get("owner"), f"{field}.owner", errors)
    _non_empty_string(value.get("source"), f"{field}.source", errors)
    _non_empty_string(value.get("version"), f"{field}.version", errors)
    _check_hash_status(value, field, status, errors)
    if field.endswith("tokenizer"):
        _string_list(value.get("special_tokens"), f"{field}.special_tokens", errors, allow_empty=status == "planned")
    else:
        roles = _string_list(value.get("roles"), f"{field}.roles", errors)
        if roles is not None and not {"system", "user", "assistant"}.issubset(set(roles)):
            errors.append(f"{field}.roles must include system, user, and assistant")
        _non_empty_string(value.get("tool_call_format"), f"{field}.tool_call_format", errors)
    _evidence(value.get("evidence"), field, status, errors, warnings)
    return status


def _check_quantization(value: Any, field: str, errors: list[str], warnings: list[str]) -> str | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    status = _status(value.get("status"), f"{field}.status", errors)
    _string_list(value.get("weight_dtypes"), f"{field}.weight_dtypes", errors)
    _string_list(value.get("activation_dtypes"), f"{field}.activation_dtypes", errors)
    _string_list(value.get("kv_dtypes"), f"{field}.kv_dtypes", errors)
    _evidence(value.get("evidence"), field, status, errors, warnings)
    return status


def _check_context(value: Any, field: str, errors: list[str], warnings: list[str]) -> str | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    status = _status(value.get("status"), f"{field}.status", errors)
    _positive_int(value.get("max_context_tokens"), f"{field}.max_context_tokens", errors)
    _positive_int(value.get("max_batch_tokens"), f"{field}.max_batch_tokens", errors)
    _non_empty_string(value.get("owner"), f"{field}.owner", errors)
    _evidence(value.get("evidence"), field, status, errors, warnings)
    return status


def _check_moe_routing(value: Any, field: str, errors: list[str], warnings: list[str]) -> str | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    status = _status(value.get("status"), f"{field}.status", errors)
    if value.get("router") != "top_k":
        errors.append(f"{field}.router must be top_k")
    _positive_int(value.get("top_k"), f"{field}.top_k", errors)
    _non_empty_string(value.get("routing_weighting"), f"{field}.routing_weighting", errors)
    if value.get("expert_trace_required") is not True:
        errors.append(f"{field}.expert_trace_required must be true")
    _evidence(value.get("evidence"), field, status, errors, warnings)
    return status


def _check_serving_semantics(
    value: Any,
    field: str,
    support_level: str | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, str | None]:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return {}
    statuses: dict[str, str | None] = {}
    for name in ("stop_behavior", "streaming", "tool_calling", "structured_output"):
        item = value.get(name)
        item_field = f"{field}.{name}"
        if not isinstance(item, dict):
            errors.append(f"{item_field} must be an object")
            continue
        status = _status(item.get("status"), f"{item_field}.status", errors)
        statuses[name] = status
        if support_level == "supported" and status not in {"supported", "limited"}:
            errors.append(f"{item_field}.status must be supported or limited for supported model")
        if name == "stop_behavior":
            _string_list(
                item.get("stop_sequences"),
                f"{item_field}.stop_sequences",
                errors,
                allow_empty=status == "planned",
            )
            _int_list(
                item.get("stop_token_ids"),
                f"{item_field}.stop_token_ids",
                errors,
                allow_empty=status == "planned",
            )
        elif name == "streaming":
            events = _string_list(item.get("events"), f"{item_field}.events", errors)
            if events is not None and not {"start", "token", "finish"}.issubset(set(events)):
                errors.append(f"{item_field}.events must include start, token, and finish")
        elif name == "tool_calling":
            _non_empty_string(item.get("format"), f"{item_field}.format", errors)
        elif name == "structured_output":
            _string_list(
                item.get("formats"),
                f"{item_field}.formats",
                errors,
                allow_empty=status == "planned",
            )
        _evidence(item.get("evidence"), item_field, status, errors, warnings)
    return statuses


def _check_parity(
    value: Any,
    field: str,
    role: str | None,
    support_level: str | None,
    errors: list[str],
    warnings: list[str],
) -> str | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    status = _non_empty_string(value.get("status"), f"{field}.status", errors)
    _non_empty_string(value.get("reference_path"), f"{field}.reference_path", errors)
    _non_empty_string(value.get("layer_logit_parity"), f"{field}.layer_logit_parity", errors)
    _non_empty_string(value.get("required_before_tier"), f"{field}.required_before_tier", errors)
    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        errors.append(f"{field}.evidence must be an object")
    else:
        _evidence(evidence, field, None, errors, warnings)
        if status == "passed" and evidence.get("kind") != "measured":
            errors.append(f"{field}.evidence must be measured when parity status is passed")
    if role == "target_candidate" and status == "passed" and support_level != "supported":
        errors.append(f"{field}.status cannot be passed for non-supported target candidate")
    return status


def validate_model_support_matrix_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("contract_kind") != "model-support-matrix":
        errors.append("contract_kind must be model-support-matrix")
    if data.get("mode") != "t1-simulation":
        errors.append("mode must be t1-simulation")
    _non_empty_string(data.get("matrix_id"), "matrix_id", errors)
    _non_empty_string(data.get("owner"), "owner", errors)
    target_contract = _non_empty_string(data.get("target_contract"), "target_contract", errors)

    capabilities = _string_list(data.get("required_capabilities"), "required_capabilities", errors)
    if capabilities is not None:
        missing = sorted(set(REQUIRED_CAPABILITIES) - set(capabilities))
        if missing:
            errors.append("required_capabilities missing: " + ", ".join(missing))

    models = data.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models must be a non-empty list")
        models = []

    supported_count = 0
    planned_count = 0
    target_count = 0
    model_ids: set[str] = set()
    capability_statuses: dict[str, set[str]] = {name: set() for name in REQUIRED_CAPABILITIES}
    parity_statuses: set[str] = set()

    for index, model in enumerate(models):
        field = f"models[{index}]"
        if not isinstance(model, dict):
            errors.append(f"{field} must be an object")
            continue
        model_id = _non_empty_string(model.get("model_id"), f"{field}.model_id", errors)
        if model_id is not None:
            if model_id in model_ids:
                errors.append(f"duplicate model_id: {model_id}")
            model_ids.add(model_id)
        role = _non_empty_string(model.get("role"), f"{field}.role", errors)
        if role is not None and role not in ROLE_VALUES:
            errors.append(f"{field}.role must be one of {sorted(ROLE_VALUES)}")
        support_level = _non_empty_string(model.get("support_level"), f"{field}.support_level", errors)
        if support_level is not None and support_level not in SUPPORT_LEVELS:
            errors.append(f"{field}.support_level must be one of {sorted(SUPPORT_LEVELS)}")
        if support_level == "supported":
            supported_count += 1
        if support_level == "planned":
            planned_count += 1
        if role == "target_candidate":
            target_count += 1

        status_map = {
            "architecture": _check_architecture(model.get("architecture"), f"{field}.architecture", errors, warnings),
            "tokenizer": _check_tokenizer_or_template(model.get("tokenizer"), f"{field}.tokenizer", errors, warnings),
            "chat_template": _check_tokenizer_or_template(model.get("chat_template"), f"{field}.chat_template", errors, warnings),
            "quantization": _check_quantization(model.get("quantization"), f"{field}.quantization", errors, warnings),
            "context_length": _check_context(model.get("context_length"), f"{field}.context_length", errors, warnings),
            "moe_routing": _check_moe_routing(model.get("moe_routing"), f"{field}.moe_routing", errors, warnings),
        }
        serving = _check_serving_semantics(
            model.get("serving_semantics"),
            f"{field}.serving_semantics",
            support_level,
            errors,
            warnings,
        )
        status_map.update(serving)
        parity_status = _check_parity(
            model.get("parity"),
            f"{field}.parity",
            role,
            support_level,
            errors,
            warnings,
        )
        if parity_status is not None:
            parity_statuses.add(parity_status)

        for name in REQUIRED_CAPABILITIES:
            status = status_map.get(name)
            if status is not None:
                capability_statuses[name].add(status)
        if support_level == "supported":
            for name, status in status_map.items():
                if name in REQUIRED_CAPABILITIES and status not in {"supported", "limited"}:
                    errors.append(f"{field}.{name}.status must be supported or limited for supported model")

    if target_contract is not None and not str(target_contract).endswith(('.md', '.json')):
        warnings.append("target_contract is not a markdown or JSON path")
    if supported_count < 1:
        errors.append("at least one supported reference model is required")
    if target_count < 1:
        errors.append("at least one target_candidate row is required")
    if not any(status in {"required_before_t2", "fixture_only"} for status in parity_statuses):
        errors.append("parity must record fixture-only or required-before-T2 status")

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    expected = {
        "model_count": len(models),
        "supported_model_count": supported_count,
        "planned_model_count": planned_count,
        "required_capability_count": len(REQUIRED_CAPABILITIES),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary.{key} does not match matrix")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "model_count": len(models),
            "supported_model_count": supported_count,
            "planned_model_count": planned_count,
            "target_candidate_count": target_count,
            "required_capability_count": len(REQUIRED_CAPABILITIES),
            "required_capabilities_seen": [
                name for name in REQUIRED_CAPABILITIES if capability_statuses[name]
            ],
            "target_model_id": summary.get("target_model_id"),
            "parity_statuses": sorted(parity_statuses),
        },
    }


def validate_model_support_matrix(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing model-support fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid model-support fixture JSON: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["model-support fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_model_support_matrix_fixture(data)
    result["fixture"] = str(fixture_path)
    return result


def render_model_support_matrix_report(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    data = read_json(fixture_path)
    if not isinstance(data, dict):
        raise ValueError("model-support fixture must be a JSON object")
    validation = validate_model_support_matrix_fixture(data)
    lines = [
        "# Model Support Matrix",
        "",
        "Status: DRAFT - T1 simulation evidence, not target-model parity evidence.",
        "",
        f"- Matrix ID: `{data.get('matrix_id', 'missing')}`",
        f"- Owner: `{data.get('owner', 'missing')}`",
        f"- Target contract: `{data.get('target_contract', 'missing')}`",
        "",
        "| Model | Role | Support | Tokenizer hash | Template hash | Routing | Parity |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in data.get("models", []):
        if not isinstance(model, dict):
            continue
        tokenizer = model.get("tokenizer") if isinstance(model.get("tokenizer"), dict) else {}
        template = model.get("chat_template") if isinstance(model.get("chat_template"), dict) else {}
        routing = model.get("moe_routing") if isinstance(model.get("moe_routing"), dict) else {}
        parity = model.get("parity") if isinstance(model.get("parity"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model.get("model_id", "missing")),
                    str(model.get("role", "missing")),
                    str(model.get("support_level", "missing")),
                    str(tokenizer.get("hash_status", "missing")),
                    str(template.get("hash_status", "missing")),
                    str(routing.get("router", "missing")) + ":" + str(routing.get("top_k", "missing")),
                    str(parity.get("status", "missing")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Valid: `{validation['ok']}`",
            "- Errors: " + ("; ".join(validation["errors"]) or "none"),
            "- Warnings: " + ("; ".join(validation["warnings"]) or "none"),
            "",
        ]
    )
    return {
        "ok": bool(validation["ok"]),
        "validation": validation,
        "fixture": str(fixture_path),
        "markdown": "\n".join(lines),
    }
