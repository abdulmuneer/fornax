from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine_seam import validate_engine_seam_fixture
from .io import read_json

RECORD_KIND = "serving-adapter-simulation-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "openai-and-ignis-to-engine-seam-roundtrip"
REQUIRED_SURFACES = {"openai_chat_completions", "ignis_engine"}
REQUIRED_EVENT_KINDS = {
    "request_received",
    "request_normalized",
    "engine_submitted",
    "stream_chunk_emitted",
    "response_finalized",
    "cleanup",
}
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
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


def _bool(value: Any, field: str, errors: list[str]) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{field} must be a boolean")
        return None
    return value


def _string_list(value: Any, field: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
            return None
        out.append(item)
    return out


def _event(kind: str, *, timestamp_s: float, plan_id: str, request_id: str, **fields: Any) -> dict[str, Any]:
    event = {
        "kind": kind,
        "timestamp_s": round(timestamp_s, 9),
        "plan_id": plan_id,
        "request_id": request_id,
    }
    event.update(fields)
    return event


def _openai_request(model: str, stream: bool, max_tokens: int) -> dict[str, Any]:
    return {
        "surface": "openai_chat_completions",
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "model": model,
        "messages": [
            {"role": "system", "content": "Return structured JSON and call tools when needed."},
            {"role": "user", "content": "Look up alpha and return a score."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_score",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "score_response",
                "schema": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                    "required": ["score"],
                },
            },
        },
        "stop": ["</final>"],
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 7,
        "max_tokens": max_tokens,
        "stream": stream,
        "user": "simulation-user",
    }


def _engine_request(
    *,
    request_id: str,
    plan_id: str,
    openai_request: dict[str, Any],
    template_hash: str,
    tokenizer_hash: str,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "plan_id": plan_id,
        "messages": openai_request["messages"],
        "tools": openai_request["tools"],
        "response_format": openai_request["response_format"],
        "stop": openai_request["stop"],
        "sampling": {
            "temperature": openai_request["temperature"],
            "top_p": openai_request["top_p"],
            "seed": openai_request["seed"],
        },
        "max_tokens": openai_request["max_tokens"],
        "stream": openai_request["stream"],
        "cancellation": {
            "supported": True,
            "propagates_to": ["scheduler", "workers", "kv_state"],
        },
        "template": {
            "name": "qwen3-chat",
            "version": "phase0-fixture-2026-06-22",
            "hash": template_hash,
        },
        "tokenizer": {
            "name": "qwen3-tokenizer",
            "version": "phase0-fixture-2026-06-22",
            "hash": tokenizer_hash,
        },
    }


def _engine_result(request_id: str, template_hash: str, tokenizer_hash: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "finish_reason": "tool_calls",
        "content": "",
        "tool_calls": [
            {
                "id": "call-001",
                "name": "lookup_score",
                "arguments": {"key": "alpha"},
            }
        ],
        "structured_output": {"score": 0.82},
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 9,
            "total_tokens": 51,
        },
        "template_hash": template_hash,
        "tokenizer_hash": tokenizer_hash,
    }


def _engine_stream_events(request_id: str, plan_id: str) -> list[dict[str, Any]]:
    return [
        {"kind": "start", "request_id": request_id, "plan_id": plan_id},
        {"kind": "token", "request_id": request_id, "token_id": 1001, "token_text": "{"},
        {
            "kind": "tool_call_delta",
            "request_id": request_id,
            "tool_call_id": "call-001",
            "delta": {"name": "lookup_score"},
        },
        {
            "kind": "structured_delta",
            "request_id": request_id,
            "delta": {"score": 0.82},
        },
        {"kind": "finish", "request_id": request_id, "finish_reason": "tool_calls"},
    ]


def _openai_response(request_id: str, model: str, engine_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": engine_result["content"],
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in engine_result["tool_calls"]
                    ],
                    "structured_output": engine_result["structured_output"],
                },
                "finish_reason": engine_result["finish_reason"],
            }
        ],
        "usage": engine_result["usage"],
    }


def _openai_stream_chunks(request_id: str, model: str, engine_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, event in enumerate(engine_events):
        kind = event["kind"]
        chunk: dict[str, Any] = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "model": model,
            "index": index,
            "engine_event_kind": kind,
        }
        if kind == "start":
            chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        elif kind == "token":
            chunk["choices"] = [{"index": 0, "delta": {"content": event["token_text"]}, "finish_reason": None}]
        elif kind == "tool_call_delta":
            chunk["choices"] = [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": event["tool_call_id"],
                                "type": "function",
                                "function": event["delta"],
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        elif kind == "structured_delta":
            chunk["choices"] = [
                {
                    "index": 0,
                    "delta": {"structured_output": event["delta"]},
                    "finish_reason": None,
                }
            ]
        elif kind == "finish":
            chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": event["finish_reason"]}]
        else:
            chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "error"}]
        chunks.append(chunk)
    return chunks


def simulate_serving_adapter(
    *,
    plan_id: str = "serving-adapter-plan",
    request_id: str = "req-serving-adapter-001",
    model: str = "qwen3-moe-class-target",
    stream: bool = True,
    max_tokens: int = 64,
    template_hash: str = SHA_A,
    tokenizer_hash: str = SHA_B,
) -> dict[str, Any]:
    if not plan_id or not request_id or not model:
        raise ValueError("plan_id, request_id, and model must be non-empty")
    errors: list[str] = []
    _positive_int(max_tokens, "max_tokens", errors)
    _bool(stream, "stream", errors)
    _non_empty_string(template_hash, "template_hash", errors)
    _non_empty_string(tokenizer_hash, "tokenizer_hash", errors)
    if errors:
        raise ValueError("; ".join(errors))

    openai_request = _openai_request(model, stream, max_tokens)
    engine_request = _engine_request(
        request_id=request_id,
        plan_id=plan_id,
        openai_request=openai_request,
        template_hash=template_hash,
        tokenizer_hash=tokenizer_hash,
    )
    engine_result = _engine_result(request_id, template_hash, tokenizer_hash)
    engine_events = _engine_stream_events(request_id, plan_id)
    openai_response = _openai_response(request_id, model, engine_result)
    openai_chunks = _openai_stream_chunks(request_id, model, engine_events)
    cancellation_result = {
        "request_id": f"{request_id}-cancel",
        "finish_reason": "cancelled",
        "cancelled": True,
        "cleanup": {
            "scheduler_released": True,
            "workers_released": True,
            "kv_released": True,
        },
    }
    error_result = {
        "request_id": f"{request_id}-error",
        "finish_reason": "error",
        "error": {
            "code": "template_mismatch",
            "message": "Template hash did not match the target contract.",
        },
    }
    error_mapping = {
        "engine_error_code": "template_mismatch",
        "openai_status_code": 409,
        "openai_error": {
            "type": "invalid_request_error",
            "code": "template_mismatch",
            "message": "Template hash did not match the target contract.",
        },
    }
    events = [
        _event(
            "request_received",
            timestamp_s=0.000,
            plan_id=plan_id,
            request_id=request_id,
            surface="openai_chat_completions",
            endpoint=openai_request["endpoint"],
        ),
        _event(
            "request_normalized",
            timestamp_s=0.001,
            plan_id=plan_id,
            request_id=request_id,
            source_surface="openai_chat_completions",
            target_surface="ignis_engine",
            template_hash=template_hash,
            tokenizer_hash=tokenizer_hash,
        ),
        _event(
            "engine_submitted",
            timestamp_s=0.002,
            plan_id=plan_id,
            request_id=request_id,
            backend="FornaxBackend",
        ),
        *[
            _event(
                "stream_chunk_emitted",
                timestamp_s=0.003 + index * 0.001,
                plan_id=plan_id,
                request_id=request_id,
                engine_event_kind=chunk["engine_event_kind"],
                chunk_index=index,
            )
            for index, chunk in enumerate(openai_chunks)
        ],
        _event(
            "response_finalized",
            timestamp_s=0.010,
            plan_id=plan_id,
            request_id=request_id,
            finish_reason=engine_result["finish_reason"],
        ),
        _event(
            "cleanup",
            timestamp_s=0.011,
            plan_id=plan_id,
            request_id=request_id,
            scheduler_released=True,
            workers_released=True,
            kv_released=True,
        ),
    ]
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "simulation_method": SIMULATION_METHOD,
        "adapters": [
            {
                "surface": "openai_chat_completions",
                "endpoint": "/v1/chat/completions",
                "protocol": "HTTP/JSON",
                "streaming": stream,
                "trust_boundary": "client_to_serving",
            },
            {
                "surface": "ignis_engine",
                "backend": "FornaxBackend",
                "trait": "Engine",
                "streaming": stream,
                "trust_boundary": "serving_to_scheduler",
            },
        ],
        "openai_request": openai_request,
        "engine_request": engine_request,
        "engine_result": engine_result,
        "engine_stream_events": engine_events,
        "openai_response": openai_response,
        "openai_stream_chunks": openai_chunks,
        "cancellation_result": cancellation_result,
        "error_result": error_result,
        "error_mapping": error_mapping,
        "events": events,
        "summary": {
            "surface_count": 2,
            "openai_chunk_count": len(openai_chunks),
            "engine_stream_event_count": len(engine_events),
            "tool_call_count": len(engine_result["tool_calls"]),
            "structured_output": True,
            "template_hash_recorded": True,
            "tokenizer_hash_recorded": True,
            "cancellation_mapped": True,
            "error_mapped": True,
            "event_count": len(events),
            "correctness_passed": True,
        },
        "note": (
            "T1 serving adapter simulation: validates OpenAI chat-completions and "
            "Ignis Engine surfaces normalize into the same Fornax Engine seam. "
            "Not a live HTTP server or production serving claim."
        ),
    }


def _engine_seam_projection(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_kind": "engine-seam",
        "version": 1,
        "speculative_decoding": {
            "enabled": False,
            "target_contract_opt_in": False,
        },
        "request": data.get("engine_request"),
        "result": data.get("engine_result"),
        "stream_events": data.get("engine_stream_events"),
        "cancellation_result": data.get("cancellation_result"),
        "error_result": data.get("error_result"),
    }


def _check_openai_request(data: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        errors.append("openai_request must be an object")
        return {}
    if data.get("surface") != "openai_chat_completions":
        errors.append("openai_request.surface must be openai_chat_completions")
    if data.get("endpoint") != "/v1/chat/completions":
        errors.append("openai_request.endpoint must be /v1/chat/completions")
    _non_empty_string(data.get("model"), "openai_request.model", errors)
    if not isinstance(data.get("messages"), list) or not data["messages"]:
        errors.append("openai_request.messages must be a non-empty list")
    if not isinstance(data.get("tools"), list):
        errors.append("openai_request.tools must be a list")
    if not isinstance(data.get("response_format"), dict):
        errors.append("openai_request.response_format must be an object")
    _string_list(data.get("stop"), "openai_request.stop", errors)
    _positive_int(data.get("max_tokens"), "openai_request.max_tokens", errors)
    _bool(data.get("stream"), "openai_request.stream", errors)
    return data


def _check_openai_response(
    response: Any,
    *,
    request_id: str | None,
    model: str | None,
    engine_result: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(response, dict):
        errors.append("openai_response must be an object")
        return
    if request_id is not None and response.get("id") != f"chatcmpl-{request_id}":
        errors.append("openai_response.id must derive from request_id")
    if model is not None and response.get("model") != model:
        errors.append("openai_response.model must match openai_request.model")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        errors.append("openai_response.choices must contain exactly one choice")
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        errors.append("openai_response.choices[0] must be an object")
        return
    if choice.get("finish_reason") != engine_result.get("finish_reason"):
        errors.append("openai_response finish_reason must match engine_result")
    message = choice.get("message")
    if not isinstance(message, dict):
        errors.append("openai_response.choices[0].message must be an object")
        return
    if message.get("content") != engine_result.get("content"):
        errors.append("openai_response message content must match engine_result")
    if message.get("structured_output") != engine_result.get("structured_output"):
        errors.append("openai_response structured_output must match engine_result")
    tool_calls = message.get("tool_calls")
    engine_tool_calls = engine_result.get("tool_calls")
    if not isinstance(tool_calls, list) or not isinstance(engine_tool_calls, list):
        errors.append("openai_response and engine_result tool calls must be lists")
    elif len(tool_calls) != len(engine_tool_calls):
        errors.append("openai_response tool call count must match engine_result")
    if response.get("usage") != engine_result.get("usage"):
        errors.append("openai_response.usage must match engine_result.usage")


def _check_openai_chunks(
    chunks: Any,
    engine_events: Any,
    request_id: str | None,
    model: str | None,
    errors: list[str],
) -> None:
    if not isinstance(chunks, list) or not chunks:
        errors.append("openai_stream_chunks must be a non-empty list")
        return
    if not isinstance(engine_events, list):
        errors.append("engine_stream_events must be a list")
        return
    if len(chunks) != len(engine_events):
        errors.append("openai_stream_chunks length must equal engine_stream_events length")
    for index, chunk in enumerate(chunks):
        field = f"openai_stream_chunks[{index}]"
        if not isinstance(chunk, dict):
            errors.append(f"{field} must be an object")
            continue
        if request_id is not None and chunk.get("id") != f"chatcmpl-{request_id}":
            errors.append(f"{field}.id must derive from request_id")
        if model is not None and chunk.get("model") != model:
            errors.append(f"{field}.model must match openai_request.model")
        if chunk.get("object") != "chat.completion.chunk":
            errors.append(f"{field}.object must be chat.completion.chunk")
        if index < len(engine_events) and isinstance(engine_events[index], dict):
            if chunk.get("engine_event_kind") != engine_events[index].get("kind"):
                errors.append(f"{field}.engine_event_kind must match engine event")
        choices = chunk.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            errors.append(f"{field}.choices must contain exactly one choice")


def validate_serving_adapter_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    if data.get("simulation_method") != SIMULATION_METHOD:
        errors.append(f"simulation_method must be {SIMULATION_METHOD}")

    adapters = data.get("adapters")
    if not isinstance(adapters, list):
        errors.append("adapters must be a list")
        adapters = []
    surfaces = {
        adapter.get("surface")
        for adapter in adapters
        if isinstance(adapter, dict)
    }
    if REQUIRED_SURFACES - surfaces:
        errors.append(f"adapters missing required surfaces: {sorted(REQUIRED_SURFACES - surfaces)}")
    for index, adapter in enumerate(adapters):
        field = f"adapters[{index}]"
        if not isinstance(adapter, dict):
            errors.append(f"{field} must be an object")
            continue
        _bool(adapter.get("streaming"), f"{field}.streaming", errors)
        _non_empty_string(adapter.get("trust_boundary"), f"{field}.trust_boundary", errors)

    seam_result = validate_engine_seam_fixture(_engine_seam_projection(data))
    errors.extend(f"engine_seam: {error}" for error in seam_result["errors"])
    warnings.extend(f"engine_seam: {warning}" for warning in seam_result["warnings"])

    openai_request = _check_openai_request(data.get("openai_request"), errors)
    engine_request = data.get("engine_request") if isinstance(data.get("engine_request"), dict) else {}
    engine_result = data.get("engine_result") if isinstance(data.get("engine_result"), dict) else {}
    request_id = engine_request.get("request_id") if isinstance(engine_request.get("request_id"), str) else None
    model = openai_request.get("model") if isinstance(openai_request.get("model"), str) else None
    if plan_id is not None and engine_request.get("plan_id") != plan_id:
        errors.append("engine_request.plan_id must match plan_id")
    for field in ("messages", "tools", "response_format", "stop", "max_tokens", "stream"):
        if field in openai_request and engine_request.get(field) != openai_request.get(field):
            errors.append(f"engine_request.{field} must match openai_request.{field}")
    if isinstance(engine_request.get("sampling"), dict):
        sampling = engine_request["sampling"]
        for source, target in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed")):
            if sampling.get(target) != openai_request.get(source):
                errors.append(f"engine_request.sampling.{target} must match openai_request.{source}")
    if engine_result.get("template_hash") != engine_request.get("template", {}).get("hash"):
        errors.append("engine_result.template_hash must match engine_request.template.hash")
    if engine_result.get("tokenizer_hash") != engine_request.get("tokenizer", {}).get("hash"):
        errors.append("engine_result.tokenizer_hash must match engine_request.tokenizer.hash")
    _check_openai_response(
        data.get("openai_response"),
        request_id=request_id,
        model=model,
        engine_result=engine_result,
        errors=errors,
    )
    _check_openai_chunks(
        data.get("openai_stream_chunks"),
        data.get("engine_stream_events"),
        request_id,
        model,
        errors,
    )

    error_mapping = data.get("error_mapping")
    if not isinstance(error_mapping, dict):
        errors.append("error_mapping must be an object")
    else:
        if error_mapping.get("engine_error_code") != data.get("error_result", {}).get("error", {}).get("code"):
            errors.append("error_mapping.engine_error_code must match error_result.error.code")
        _positive_int(error_mapping.get("openai_status_code"), "error_mapping.openai_status_code", errors)
        if not isinstance(error_mapping.get("openai_error"), dict):
            errors.append("error_mapping.openai_error must be an object")

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []
    event_kinds = {event.get("kind") for event in events if isinstance(event, dict)}
    missing_events = REQUIRED_EVENT_KINDS - event_kinds
    if missing_events:
        errors.append(f"events missing required kinds: {sorted(missing_events)}")
    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(event.get("kind"), f"{field}.kind", errors)
        if event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match plan_id")
        if request_id is not None and event.get("request_id") != request_id:
            errors.append(f"{field}.request_id must match engine_request.request_id")
        _non_negative_int(event.get("chunk_index"), f"{field}.chunk_index", errors) if event.get("kind") == "stream_chunk_emitted" else None

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("surface_count") != len(surfaces):
        errors.append("summary.surface_count must match unique adapter surfaces")
    if isinstance(data.get("openai_stream_chunks"), list) and summary.get("openai_chunk_count") != len(data["openai_stream_chunks"]):
        errors.append("summary.openai_chunk_count must match openai_stream_chunks")
    if isinstance(data.get("engine_stream_events"), list) and summary.get("engine_stream_event_count") != len(data["engine_stream_events"]):
        errors.append("summary.engine_stream_event_count must match engine_stream_events")
    if isinstance(engine_result.get("tool_calls"), list) and summary.get("tool_call_count") != len(engine_result["tool_calls"]):
        errors.append("summary.tool_call_count must match engine_result.tool_calls")
    for field in (
        "structured_output",
        "template_hash_recorded",
        "tokenizer_hash_recorded",
        "cancellation_mapped",
        "error_mapped",
        "correctness_passed",
    ):
        if summary.get(field) is not True:
            errors.append(f"summary.{field} must be true")
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must equal len(events)")
    warnings.append("serving adapter is simulation evidence, not a live HTTP endpoint")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "surface_count": summary.get("surface_count"),
            "openai_chunk_count": summary.get("openai_chunk_count"),
            "engine_stream_event_count": summary.get("engine_stream_event_count"),
            "tool_call_count": summary.get("tool_call_count"),
            "template_hash_recorded": summary.get("template_hash_recorded") is True,
            "tokenizer_hash_recorded": summary.get("tokenizer_hash_recorded") is True,
            "cancellation_mapped": summary.get("cancellation_mapped") is True,
            "error_mapped": summary.get("error_mapped") is True,
            "correctness_passed": summary.get("correctness_passed") is True,
        },
    }


def validate_serving_adapter(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid serving adapter artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["serving adapter artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_serving_adapter_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
