from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import read_json


FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "structured_output",
    "content_filter",
    "error",
    "cancelled",
}
REQUIRED_STREAM_EVENTS = {"start", "token", "finish"}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _object(value: Any, field: str, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    return value


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _hash(value: Any, field: str, errors: list[str]) -> str | None:
    text = _non_empty_string(value, field, errors)
    if text is not None and not SHA256_RE.match(text):
        errors.append(f"{field} must be a sha256:<64 lowercase hex chars> hash")
    return text


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


def _non_negative_int(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{field} must be a non-negative integer")
        return None
    return value


def _string_list(value: Any, field: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
            return None
        result.append(item)
    return result


def _check_template_or_tokenizer(
    data: dict[str, Any],
    field: str,
    errors: list[str],
) -> str | None:
    _non_empty_string(data.get("name"), f"{field}.name", errors)
    _non_empty_string(data.get("version"), f"{field}.version", errors)
    return _hash(data.get("hash"), f"{field}.hash", errors)


def _check_messages(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("request.messages must be a non-empty list")
        return
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            errors.append(f"request.messages[{index}] must be an object")
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            errors.append(f"request.messages[{index}].role is invalid")
        if not isinstance(message.get("content"), (str, list)):
            errors.append(f"request.messages[{index}].content must be a string or list")


def _check_tools(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("request.tools must be a list")
        return
    for index, tool in enumerate(value):
        if not isinstance(tool, dict):
            errors.append(f"request.tools[{index}] must be an object")
            continue
        if tool.get("type") != "function":
            errors.append(f"request.tools[{index}].type must be function")
        function = tool.get("function")
        if not isinstance(function, dict):
            errors.append(f"request.tools[{index}].function must be an object")
            continue
        _non_empty_string(
            function.get("name"), f"request.tools[{index}].function.name", errors
        )
        if not isinstance(function.get("parameters"), dict):
            errors.append(f"request.tools[{index}].function.parameters must be an object")


def _check_response_format(value: Any, errors: list[str]) -> None:
    data = _object(value, "request.response_format", errors)
    if data is None:
        return
    kind = data.get("type")
    if kind not in {"text", "json_object", "json_schema"}:
        errors.append("request.response_format.type is invalid")
    if kind == "json_schema":
        schema = data.get("json_schema")
        if not isinstance(schema, dict):
            errors.append("request.response_format.json_schema must be an object")
        elif not isinstance(schema.get("schema"), dict):
            errors.append("request.response_format.json_schema.schema must be an object")


def _check_sampling(value: Any, errors: list[str]) -> None:
    data = _object(value, "request.sampling", errors)
    if data is None:
        return
    for field in ("temperature", "top_p"):
        number = data.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            errors.append(f"request.sampling.{field} must be numeric")
    seed = data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        errors.append("request.sampling.seed must be an integer when present")


def _check_request(data: dict[str, Any], errors: list[str]) -> dict[str, str | None]:
    request_id = _non_empty_string(data.get("request_id"), "request.request_id", errors)
    _non_empty_string(data.get("plan_id"), "request.plan_id", errors)
    _check_messages(data.get("messages"), errors)
    _check_tools(data.get("tools"), errors)
    _check_response_format(data.get("response_format"), errors)
    _string_list(data.get("stop"), "request.stop", errors)
    _check_sampling(data.get("sampling"), errors)
    _positive_int(data.get("max_tokens"), "request.max_tokens", errors)
    _bool(data.get("stream"), "request.stream", errors)

    cancellation = _object(data.get("cancellation"), "request.cancellation", errors)
    if cancellation is not None:
        _bool(cancellation.get("supported"), "request.cancellation.supported", errors)
        targets = _string_list(
            cancellation.get("propagates_to"),
            "request.cancellation.propagates_to",
            errors,
        )
        required_targets = {"scheduler", "workers", "kv_state"}
        if targets is not None and not required_targets.issubset(set(targets)):
            errors.append(
                "request.cancellation.propagates_to must include scheduler, "
                "workers, and kv_state"
            )

    template = _object(data.get("template"), "request.template", errors)
    tokenizer = _object(data.get("tokenizer"), "request.tokenizer", errors)
    template_hash = (
        _check_template_or_tokenizer(template, "request.template", errors)
        if template is not None
        else None
    )
    tokenizer_hash = (
        _check_template_or_tokenizer(tokenizer, "request.tokenizer", errors)
        if tokenizer is not None
        else None
    )
    return {
        "request_id": request_id,
        "template_hash": template_hash,
        "tokenizer_hash": tokenizer_hash,
    }


def _check_usage(value: Any, field: str, errors: list[str]) -> None:
    data = _object(value, field, errors)
    if data is None:
        return
    _non_negative_int(data.get("prompt_tokens"), f"{field}.prompt_tokens", errors)
    _non_negative_int(
        data.get("completion_tokens"), f"{field}.completion_tokens", errors
    )
    _non_negative_int(data.get("total_tokens"), f"{field}.total_tokens", errors)
    if (
        isinstance(data.get("prompt_tokens"), int)
        and isinstance(data.get("completion_tokens"), int)
        and isinstance(data.get("total_tokens"), int)
        and data["total_tokens"] != data["prompt_tokens"] + data["completion_tokens"]
    ):
        errors.append(f"{field}.total_tokens must equal prompt_tokens + completion_tokens")


def _check_tool_calls(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return
    for index, call in enumerate(value):
        if not isinstance(call, dict):
            errors.append(f"{field}[{index}] must be an object")
            continue
        _non_empty_string(call.get("id"), f"{field}[{index}].id", errors)
        _non_empty_string(call.get("name"), f"{field}[{index}].name", errors)
        if not isinstance(call.get("arguments"), dict):
            errors.append(f"{field}[{index}].arguments must be an object")


def _check_result(
    data: dict[str, Any],
    request_context: dict[str, str | None],
    errors: list[str],
) -> None:
    request_id = _non_empty_string(data.get("request_id"), "result.request_id", errors)
    if request_id is not None and request_context["request_id"] is not None:
        if request_id != request_context["request_id"]:
            errors.append("result.request_id must match request.request_id")
    finish_reason = data.get("finish_reason")
    if finish_reason not in FINISH_REASONS - {"cancelled", "error"}:
        errors.append("result.finish_reason is invalid for a successful result")
    _check_usage(data.get("usage"), "result.usage", errors)
    template_hash = _hash(data.get("template_hash"), "result.template_hash", errors)
    tokenizer_hash = _hash(data.get("tokenizer_hash"), "result.tokenizer_hash", errors)
    if template_hash and request_context["template_hash"] and template_hash != request_context["template_hash"]:
        errors.append("result.template_hash must match request.template.hash")
    if tokenizer_hash and request_context["tokenizer_hash"] and tokenizer_hash != request_context["tokenizer_hash"]:
        errors.append("result.tokenizer_hash must match request.tokenizer.hash")
    if finish_reason == "tool_calls":
        _check_tool_calls(data.get("tool_calls"), "result.tool_calls", errors)
    if data.get("structured_output") is not None and not isinstance(
        data.get("structured_output"), dict
    ):
        errors.append("result.structured_output must be an object when present")


def _check_stream_events(
    value: Any,
    request_id: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(value, list) or not value:
        errors.append("stream_events must be a non-empty list")
        return
    seen: set[str] = set()
    finished = False
    for index, event in enumerate(value):
        if not isinstance(event, dict):
            errors.append(f"stream_events[{index}] must be an object")
            continue
        kind = event.get("kind")
        if not isinstance(kind, str):
            errors.append(f"stream_events[{index}].kind must be a string")
            continue
        seen.add(kind)
        event_request_id = event.get("request_id")
        if request_id is not None and event_request_id != request_id:
            errors.append(f"stream_events[{index}].request_id must match request")
        if kind == "token":
            if "token_text" not in event and "token_id" not in event:
                errors.append("token stream event must include token_text or token_id")
        elif kind == "finish":
            if event.get("finish_reason") not in FINISH_REASONS:
                errors.append("finish stream event must carry a valid finish_reason")
            finished = True
        elif kind == "error":
            if not isinstance(event.get("error"), dict):
                errors.append("error stream event must include error object")
        elif kind == "cancelled":
            if event.get("finish_reason") != "cancelled":
                errors.append("cancelled stream event must finish as cancelled")
        elif kind not in {"start", "tool_call_delta", "structured_delta"}:
            warnings.append(f"unknown stream event kind: {kind}")
    missing = sorted(REQUIRED_STREAM_EVENTS - seen)
    if missing:
        errors.append("missing required stream events: " + ", ".join(missing))
    if not finished:
        errors.append("stream_events must include a finish event")


def _check_cancellation_result(value: Any, errors: list[str]) -> None:
    data = _object(value, "cancellation_result", errors)
    if data is None:
        return
    _non_empty_string(data.get("request_id"), "cancellation_result.request_id", errors)
    if data.get("finish_reason") != "cancelled":
        errors.append("cancellation_result.finish_reason must be cancelled")
    if data.get("cancelled") is not True:
        errors.append("cancellation_result.cancelled must be true")
    cleanup = _object(data.get("cleanup"), "cancellation_result.cleanup", errors)
    if cleanup is not None:
        for field in ("scheduler_released", "workers_released", "kv_released"):
            if cleanup.get(field) is not True:
                errors.append(f"cancellation_result.cleanup.{field} must be true")


def _check_error_result(value: Any, errors: list[str]) -> None:
    data = _object(value, "error_result", errors)
    if data is None:
        return
    _non_empty_string(data.get("request_id"), "error_result.request_id", errors)
    if data.get("finish_reason") != "error":
        errors.append("error_result.finish_reason must be error")
    error = _object(data.get("error"), "error_result.error", errors)
    if error is not None:
        _non_empty_string(error.get("code"), "error_result.error.code", errors)
        _non_empty_string(error.get("message"), "error_result.error.message", errors)


def validate_engine_seam_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("contract_kind") != "engine-seam":
        errors.append("contract_kind must be engine-seam")

    speculative = _object(data.get("speculative_decoding"), "speculative_decoding", errors)
    if speculative is not None:
        enabled = _bool(speculative.get("enabled"), "speculative_decoding.enabled", errors)
        opted_in = _bool(
            speculative.get("target_contract_opt_in"),
            "speculative_decoding.target_contract_opt_in",
            errors,
        )
        if enabled and opted_in is False:
            errors.append(
                "speculative decoding is out of v0 unless target_contract_opt_in is true"
            )

    request = _object(data.get("request"), "request", errors)
    request_context = (
        _check_request(request, errors)
        if request is not None
        else {"request_id": None, "template_hash": None, "tokenizer_hash": None}
    )
    result = _object(data.get("result"), "result", errors)
    if result is not None:
        _check_result(result, request_context, errors)
    _check_stream_events(
        data.get("stream_events"),
        request_context["request_id"],
        errors,
        warnings,
    )
    _check_cancellation_result(data.get("cancellation_result"), errors)
    _check_error_result(data.get("error_result"), errors)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "request_id": request_context["request_id"],
            "template_hash_recorded": bool(request_context["template_hash"]),
            "tokenizer_hash_recorded": bool(request_context["tokenizer_hash"]),
            "stream_event_count": (
                len(data.get("stream_events", []))
                if isinstance(data.get("stream_events"), list)
                else 0
            ),
            "speculative_decoding_enabled": bool(
                isinstance(data.get("speculative_decoding"), dict)
                and data["speculative_decoding"].get("enabled")
            ),
        },
    }


def validate_engine_seam_contract(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    if not fixture_path.exists():
        return {
            "ok": False,
            "errors": [f"missing engine-seam fixture: {fixture_path}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    try:
        data = read_json(fixture_path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture failures.
        return {
            "ok": False,
            "errors": [f"invalid engine-seam fixture JSON: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["engine-seam fixture must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_engine_seam_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
