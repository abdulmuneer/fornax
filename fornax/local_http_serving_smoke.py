from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .serving import simulate_serving_adapter, validate_serving_adapter_fixture


RECORD_KIND = "local-http-serving-smoke"
EVIDENCE_SCOPE = "local-http-sse-serving-smoke"
PLAN_ID_HEADER = "x-fornax-plan-id"
PLAN_HASH_HEADER = "x-fornax-plan-hash"


class LocalFornaxBackend:
    """Local smoke implementation of the FornaxBackend Engine seam."""

    def __init__(self, *, plan_id: str, request_id: str, model: str, max_tokens: int) -> None:
        self.plan_id = plan_id
        self.request_id = request_id
        self.model = model
        self.max_tokens = max_tokens
        self.call_count = 0

    def complete(self, request: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        self.call_count += 1
        model = str(request.get("model", self.model))
        max_tokens = int(request.get("max_tokens", self.max_tokens))
        return simulate_serving_adapter(
            plan_id=self.plan_id,
            request_id=self.request_id,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "FornaxBackend",
            "mode": "local-http-smoke",
            "engine_trait_compatible": True,
            "request_count": self.call_count,
            "target_model_loaded": False,
            "target_model_parity": False,
        }


class _SmokeHandler(BaseHTTPRequestHandler):
    server: "_SmokeServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        config = self.server.config
        if self.path != "/v1/chat/completions":
            self._json_response(
                404,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "not_found",
                        "message": "Only /v1/chat/completions is available in local smoke.",
                    }
                },
            )
            return
        plan_id = self.headers.get(PLAN_ID_HEADER)
        plan_hash = self.headers.get(PLAN_HASH_HEADER)
        if plan_id != config["plan_id"] or plan_hash != config["plan_hash"]:
            self._json_response(
                409,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "plan_integrity_mismatch",
                        "message": "Plan identity or hash did not match the local smoke server.",
                    }
                },
            )
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            request = json.loads(body) if body else {}
        except Exception:
            self._json_response(
                400,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                        "message": "Request body must be valid JSON.",
                    }
                },
            )
            return
        stream = bool(request.get("stream", False))
        max_tokens = int(request.get("max_tokens", config["max_tokens"]))
        model = str(request.get("model", config["model"]))
        adapter = self.server.backend.complete(
            {**request, "model": model, "max_tokens": max_tokens},
            stream=stream,
        )
        if stream:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()
            for chunk in adapter["openai_stream_chunks"]:
                line = "data: " + json.dumps(chunk) + "\n\n"
                self.wfile.write(line.encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json_response(200, adapter["openai_response"])


class _SmokeServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config: dict[str, Any]) -> None:
        super().__init__(server_address, _SmokeHandler)
        self.config = config
        self.backend = LocalFornaxBackend(
            plan_id=config["plan_id"],
            request_id=config["request_id"],
            model=config["model"],
            max_tokens=config["max_tokens"],
        )


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    plan_id: str,
    plan_hash: str,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            PLAN_ID_HEADER: plan_id,
            PLAN_HASH_HEADER: plan_hash,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = response.read().decode("utf-8")
            return {
                "status": int(response.status),
                "content_type": response.headers.get("content-type"),
                "body": json.loads(data),
            }
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8")
        return {
            "status": int(exc.code),
            "content_type": exc.headers.get("content-type"),
            "body": json.loads(data) if data else {},
        }


def _post_sse(
    url: str,
    payload: dict[str, Any],
    *,
    plan_id: str,
    plan_hash: str,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            PLAN_ID_HEADER: plan_id,
            PLAN_HASH_HEADER: plan_hash,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        text = response.read().decode("utf-8")
    events: list[dict[str, Any]] = []
    done_seen = False
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if not block.startswith("data: "):
            continue
        raw = block[len("data: ") :]
        if raw == "[DONE]":
            done_seen = True
            continue
        events.append(json.loads(raw))
    return {
        "status": 200,
        "content_type": "text/event-stream",
        "raw_event_count": text.count("data: "),
        "chunk_count": len(events),
        "done_seen": done_seen,
        "events": events,
    }


def run_local_http_serving_smoke(
    *,
    out: str | Path,
    host: str = "127.0.0.1",
    port: int = 0,
    plan_id: str = "local-http-serving-plan",
    plan_hash: str = "sha256:local-http-serving-plan",
    request_id: str = "local-http-serving-request",
    model: str = "qwen3-moe-class-target",
    max_tokens: int = 64,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    if not host or not plan_id or not plan_hash or not request_id or not model:
        raise ValueError("host, plan_id, plan_hash, request_id, and model must be non-empty")
    if isinstance(port, bool) or not isinstance(port, int) or port < 0:
        raise ValueError("port must be a non-negative integer")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive number")

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "request_id": request_id,
        "model": model,
        "max_tokens": max_tokens,
    }
    server = _SmokeServer((host, port), config)
    server_host, server_port = server.server_address
    thread = threading.Thread(target=server.serve_forever, name="fornax-http-smoke", daemon=True)
    started_ns = time.perf_counter_ns()
    thread.start()
    endpoint = f"http://{server_host}:{server_port}/v1/chat/completions"
    try:
        adapter = simulate_serving_adapter(
            plan_id=plan_id,
            request_id=request_id,
            model=model,
            stream=True,
            max_tokens=max_tokens,
        )
        adapter_validation = validate_serving_adapter_fixture(adapter)
        non_stream = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash=plan_hash,
            timeout_s=float(timeout_s),
        )
        stream = _post_sse(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": True},
            plan_id=plan_id,
            plan_hash=plan_hash,
            timeout_s=float(timeout_s),
        )
        plan_reject = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash="sha256:mismatch",
            timeout_s=float(timeout_s),
        )
        bad_path = _post_json(
            f"http://{server_host}:{server_port}/bad/path",
            {"model": model},
            plan_id=plan_id,
            plan_hash=plan_hash,
            timeout_s=float(timeout_s),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=float(timeout_s))
    backend_summary = server.backend.summary()
    elapsed_ns = time.perf_counter_ns() - started_ns

    non_stream_ok = (
        non_stream.get("status") == 200
        and non_stream.get("body", {}).get("object") == "chat.completion"
        and non_stream.get("body", {}).get("model") == model
    )
    stream_ok = (
        stream.get("status") == 200
        and stream.get("done_seen") is True
        and stream.get("chunk_count") == adapter_validation["summary"].get("openai_chunk_count")
    )
    plan_reject_ok = (
        plan_reject.get("status") == 409
        and plan_reject.get("body", {}).get("error", {}).get("code") == "plan_integrity_mismatch"
    )
    bad_path_ok = bad_path.get("status") == 404
    backend_ok = (
        backend_summary.get("backend") == "FornaxBackend"
        and backend_summary.get("engine_trait_compatible") is True
        and backend_summary.get("request_count") == 2
        and backend_summary.get("target_model_loaded") is False
        and backend_summary.get("target_model_parity") is False
    )
    checks = [
        {"name": "serving-adapter", "ok": bool(adapter_validation.get("ok")), "errors": adapter_validation.get("errors", []), "warnings": adapter_validation.get("warnings", [])},
        {"name": "fornax-backend-integration", "ok": backend_ok, "errors": [] if backend_ok else ["FornaxBackend local integration invalid"], "warnings": []},
        {"name": "non-stream-http", "ok": non_stream_ok, "errors": [] if non_stream_ok else ["non-stream HTTP response invalid"], "warnings": []},
        {"name": "stream-sse", "ok": stream_ok, "errors": [] if stream_ok else ["SSE stream response invalid"], "warnings": []},
        {"name": "plan-integrity-reject", "ok": plan_reject_ok, "errors": [] if plan_reject_ok else ["plan integrity rejection invalid"], "warnings": []},
        {"name": "bad-path-reject", "ok": bad_path_ok, "errors": [] if bad_path_ok else ["bad path rejection invalid"], "warnings": []},
    ]
    passed_count = sum(1 for check in checks if check["ok"])
    summary = {
        "check_count": len(checks),
        "passed_count": passed_count,
        "http_endpoint_started": True,
        "endpoint": endpoint,
        "host": server_host,
        "port": server_port,
        "non_stream_status": non_stream.get("status"),
        "stream_status": stream.get("status"),
        "sse_chunk_count": stream.get("chunk_count"),
        "sse_done_seen": stream.get("done_seen"),
        "plan_integrity_rejected": plan_reject_ok,
        "bad_path_rejected": bad_path_ok,
        "fornax_backend_integrated": backend_ok,
        "backend_request_count": backend_summary.get("request_count"),
        "engine_trait_compatible": backend_summary.get("engine_trait_compatible"),
        "engine_result_emitted": non_stream_ok and stream_ok,
        "backend_target_model_loaded": backend_summary.get("target_model_loaded"),
        "elapsed_s": elapsed_ns / 1_000_000_000.0,
        "live_http_endpoint": True,
        "localhost_only": server_host in {"127.0.0.1", "localhost"},
        "tls_enabled": False,
        "production_auth_enabled": False,
        "target_model_parity": False,
        "g2_g3_gate_evidence": False,
        "correctness_passed": passed_count == len(checks),
    }
    result = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "endpoint": endpoint,
        "config": config,
        "backend": backend_summary,
        "serving_adapter": adapter,
        "responses": {
            "non_stream": non_stream,
            "stream": stream,
            "plan_reject": plan_reject,
            "bad_path": bad_path,
        },
        "checks": checks,
        "summary": summary,
        "ok": passed_count == len(checks),
        "note": (
            "Local HTTP/SSE serving smoke for the OpenAI-compatible endpoint path. "
            "This proves local endpoint request/response behavior and plan-integrity "
            "rejection only; it is not TLS/product auth, target-model parity, real "
            "multi-host serving, or G2/G3 closure evidence."
        ),
    }
    write_json(output_path, result)
    return result


def validate_local_http_serving_smoke_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "local HTTP serving smoke is localhost-only and not target-model or multi-host gate evidence"
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must be {EVIDENCE_SCOPE}")
    adapter = data.get("serving_adapter")
    if not isinstance(adapter, dict):
        errors.append("serving_adapter must be an object")
    else:
        adapter_result = validate_serving_adapter_fixture(adapter)
        errors.extend(f"serving_adapter: {error}" for error in adapter_result["errors"])
        warnings.extend(f"serving_adapter: {warning}" for warning in adapter_result["warnings"])
    backend = data.get("backend")
    if not isinstance(backend, dict):
        errors.append("backend must be an object")
        backend = {}
    if backend.get("backend") != "FornaxBackend":
        errors.append("backend.backend must be FornaxBackend")
    if backend.get("mode") != "local-http-smoke":
        errors.append("backend.mode must be local-http-smoke")
    if backend.get("engine_trait_compatible") is not True:
        errors.append("backend.engine_trait_compatible must be true")
    if backend.get("request_count") != 2:
        errors.append("backend.request_count must be 2")
    if backend.get("target_model_loaded") is not False:
        errors.append("backend.target_model_loaded must be false")
    if backend.get("target_model_parity") is not False:
        errors.append("backend.target_model_parity must be false")
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
            errors.append(f"checks[{index}] {check.get('name', '<unknown>')} must pass")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    passed_count = sum(1 for check in checks if isinstance(check, dict) and check.get("ok") is True)
    if summary.get("check_count") != len(checks):
        errors.append("summary.check_count must match checks")
    if summary.get("passed_count") != passed_count:
        errors.append("summary.passed_count must match checks")
    if summary.get("http_endpoint_started") is not True:
        errors.append("summary.http_endpoint_started must be true")
    if summary.get("non_stream_status") != 200:
        errors.append("summary.non_stream_status must be 200")
    if summary.get("stream_status") != 200:
        errors.append("summary.stream_status must be 200")
    if summary.get("sse_done_seen") is not True:
        errors.append("summary.sse_done_seen must be true")
    if summary.get("plan_integrity_rejected") is not True:
        errors.append("summary.plan_integrity_rejected must be true")
    if summary.get("bad_path_rejected") is not True:
        errors.append("summary.bad_path_rejected must be true")
    if summary.get("fornax_backend_integrated") is not True:
        errors.append("summary.fornax_backend_integrated must be true")
    if summary.get("backend_request_count") != 2:
        errors.append("summary.backend_request_count must be 2")
    if summary.get("engine_trait_compatible") is not True:
        errors.append("summary.engine_trait_compatible must be true")
    if summary.get("engine_result_emitted") is not True:
        errors.append("summary.engine_result_emitted must be true")
    if summary.get("backend_target_model_loaded") is not False:
        errors.append("summary.backend_target_model_loaded must be false")
    if summary.get("live_http_endpoint") is not True:
        errors.append("summary.live_http_endpoint must be true")
    if summary.get("localhost_only") is not True:
        errors.append("summary.localhost_only must be true")
    if summary.get("tls_enabled") is not False:
        errors.append("summary.tls_enabled must be false for local smoke")
    if summary.get("production_auth_enabled") is not False:
        errors.append("summary.production_auth_enabled must be false for local smoke")
    if summary.get("target_model_parity") is not False:
        errors.append("summary.target_model_parity must be false")
    if summary.get("g2_g3_gate_evidence") is not False:
        errors.append("summary.g2_g3_gate_evidence must be false")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")
    if data.get("ok") is not True:
        errors.append("ok must be true")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "check_count": summary.get("check_count"),
            "passed_count": passed_count,
            "endpoint": summary.get("endpoint"),
            "sse_chunk_count": summary.get("sse_chunk_count"),
            "plan_integrity_rejected": summary.get("plan_integrity_rejected") is True,
            "fornax_backend_integrated": summary.get("fornax_backend_integrated") is True,
            "backend_request_count": summary.get("backend_request_count"),
            "engine_trait_compatible": summary.get("engine_trait_compatible") is True,
            "engine_result_emitted": summary.get("engine_result_emitted") is True,
            "live_http_endpoint": summary.get("live_http_endpoint") is True,
            "target_model_parity": summary.get("target_model_parity") is True,
            "g2_g3_gate_evidence": summary.get("g2_g3_gate_evidence") is True,
        },
    }


def validate_local_http_serving_smoke(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid local HTTP serving smoke artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["local HTTP serving smoke artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_local_http_serving_smoke_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
