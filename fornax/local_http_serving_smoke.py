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
AUTH_HEADER = "authorization"
LOCAL_LIFECYCLE_RESOURCE_KINDS = (
    "request_envelope",
    "engine_context",
    "scheduler_slot",
    "response_stream",
    "kv_cache",
)


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
        self._inflight_lock = threading.Lock()
        self._inflight_count = 0
        self.max_observed_inflight = 0
        self.backpressure_reject_count = 0
        self.inflight_cleanup_count = 0
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_sequence = 0
        self._lifecycle_active_resources: set[tuple[str, str]] = set()
        self._lifecycle_accepted: set[str] = set()
        self._lifecycle_cleanup_count = 0
        self._lifecycle_rejected_count = 0
        self.lifecycle_events: list[dict[str, Any]] = []

    def try_admit(self) -> bool:
        with self._inflight_lock:
            if self._inflight_count >= int(self.config["max_inflight"]):
                self.backpressure_reject_count += 1
                return False
            self._inflight_count += 1
            self.max_observed_inflight = max(self.max_observed_inflight, self._inflight_count)
            return True

    def release_inflight(self) -> None:
        with self._inflight_lock:
            if self._inflight_count > 0:
                self._inflight_count -= 1
            self.inflight_cleanup_count += 1

    def wait_for_inflight(self, count: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._inflight_lock:
                if self._inflight_count >= count:
                    return True
            time.sleep(0.005)
        return False

    def backpressure_summary(self) -> dict[str, Any]:
        with self._inflight_lock:
            current_inflight = self._inflight_count
        return {
            "max_inflight": self.config["max_inflight"],
            "current_inflight": current_inflight,
            "max_observed_inflight": self.max_observed_inflight,
            "backpressure_reject_count": self.backpressure_reject_count,
            "inflight_cleanup_count": self.inflight_cleanup_count,
        }

    def _append_lifecycle_event(
        self,
        *,
        kind: str,
        request_label: str,
        resource_kind: str | None,
        owner: str,
        state: str,
        reason: str,
    ) -> None:
        self.lifecycle_events.append(
            {
                "index": len(self.lifecycle_events),
                "kind": kind,
                "request_label": request_label,
                "resource_kind": resource_kind,
                "owner": owner,
                "state": state,
                "reason": reason,
            }
        )

    def record_rejection(self, reason: str) -> None:
        with self._lifecycle_lock:
            self._lifecycle_rejected_count += 1
            self._append_lifecycle_event(
                kind="request_rejected",
                request_label=f"rejected-{self._lifecycle_rejected_count}",
                resource_kind=None,
                owner="serving_gateway",
                state="rejected",
                reason=reason,
            )

    def allocate_lifecycle(self, *, stream: bool) -> str:
        with self._lifecycle_lock:
            self._lifecycle_sequence += 1
            request_label = f"accepted-{self._lifecycle_sequence}"
            self._lifecycle_accepted.add(request_label)
            for resource_kind in LOCAL_LIFECYCLE_RESOURCE_KINDS:
                self._lifecycle_active_resources.add((request_label, resource_kind))
            self._append_lifecycle_event(
                kind="request_received",
                request_label=request_label,
                resource_kind="request_envelope",
                owner="serving_gateway",
                state="active",
                reason="OpenAI-compatible request accepted",
            )
            self._append_lifecycle_event(
                kind="engine_request_normalized",
                request_label=request_label,
                resource_kind="engine_context",
                owner="fornax_engine",
                state="active",
                reason="request normalized into local EngineRequest",
            )
            self._append_lifecycle_event(
                kind="scheduler_admitted",
                request_label=request_label,
                resource_kind="scheduler_slot",
                owner="scheduler",
                state="active",
                reason="local admission slot allocated",
            )
            self._append_lifecycle_event(
                kind="stream_opened" if stream else "response_opened",
                request_label=request_label,
                resource_kind="response_stream",
                owner="serving_gateway",
                state="active",
                reason="serving response state opened",
            )
            self._append_lifecycle_event(
                kind="kv_read_granted",
                request_label=request_label,
                resource_kind="kv_cache",
                owner="kv_manager",
                state="active",
                reason="local smoke KV ownership placeholder opened",
            )
            return request_label

    def release_lifecycle(self, request_label: str) -> None:
        with self._lifecycle_lock:
            released_any = False
            for resource_kind in LOCAL_LIFECYCLE_RESOURCE_KINDS:
                key = (request_label, resource_kind)
                if key in self._lifecycle_active_resources:
                    self._lifecycle_active_resources.remove(key)
                    released_any = True
                    self._append_lifecycle_event(
                        kind="cleanup",
                        request_label=request_label,
                        resource_kind=resource_kind,
                        owner="released",
                        state="released",
                        reason="local endpoint request cleanup",
                    )
            if released_any:
                self._lifecycle_cleanup_count += 1

    def lifecycle_summary(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            event_count = len(self.lifecycle_events)
            active_resource_count = len(self._lifecycle_active_resources)
            accepted_request_count = len(self._lifecycle_accepted)
            rejected_request_count = self._lifecycle_rejected_count
            cleanup_count = self._lifecycle_cleanup_count
            resource_allocated_count = accepted_request_count * len(LOCAL_LIFECYCLE_RESOURCE_KINDS)
            resource_released_count = sum(1 for event in self.lifecycle_events if event["kind"] == "cleanup")
            events = list(self.lifecycle_events)
        return {
            "mode": "local-http-lifecycle-smoke",
            "resource_kinds": list(LOCAL_LIFECYCLE_RESOURCE_KINDS),
            "event_count": event_count,
            "accepted_request_count": accepted_request_count,
            "rejected_request_count": rejected_request_count,
            "cleanup_count": cleanup_count,
            "resource_allocated_count": resource_allocated_count,
            "resource_released_count": resource_released_count,
            "active_resource_count": active_resource_count,
            "all_required_resources_released": active_resource_count == 0,
            "single_owner_preserved": True,
            "events": events,
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
            self.server.record_rejection("bad_path")
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
        expected_auth = f"Bearer {config['auth_token']}"
        if self.headers.get(AUTH_HEADER) != expected_auth:
            self.server.record_rejection("endpoint_auth_required")
            self._json_response(
                401,
                {
                    "error": {
                        "type": "authentication_error",
                        "code": "endpoint_auth_required",
                        "message": "Local smoke endpoint requires the configured bearer token.",
                    }
                },
            )
            return
        plan_id = self.headers.get(PLAN_ID_HEADER)
        plan_hash = self.headers.get(PLAN_HASH_HEADER)
        if plan_id != config["plan_id"] or plan_hash != config["plan_hash"]:
            self.server.record_rejection("plan_integrity_mismatch")
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
            self.server.record_rejection("invalid_json")
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
        simulate_work_ms = max(0, int(request.get("simulate_work_ms", 0)))
        if not self.server.try_admit():
            self.server.record_rejection("backpressure_queue_full")
            self._json_response(
                429,
                {
                    "error": {
                        "type": "rate_limit_error",
                        "code": "backpressure_queue_full",
                        "message": "Local smoke inflight capacity is exhausted.",
                    },
                    "retry_after_ms": config["retry_after_ms"],
                },
            )
            return
        request_label = self.server.allocate_lifecycle(stream=stream)
        try:
            if simulate_work_ms:
                time.sleep(simulate_work_ms / 1000.0)
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
        finally:
            self.server.release_lifecycle(request_label)
            self.server.release_inflight()


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    plan_id: str,
    plan_hash: str,
    auth_token: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        PLAN_ID_HEADER: plan_id,
        PLAN_HASH_HEADER: plan_hash,
    }
    if auth_token is not None:
        headers[AUTH_HEADER] = f"Bearer {auth_token}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
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
    auth_token: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        PLAN_ID_HEADER: plan_id,
        PLAN_HASH_HEADER: plan_hash,
    }
    if auth_token is not None:
        headers[AUTH_HEADER] = f"Bearer {auth_token}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
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
    auth_token: str = "local-smoke-token",
    max_inflight: int = 1,
    backpressure_delay_ms: int = 250,
    retry_after_ms: int = 25,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    if not host or not plan_id or not plan_hash or not request_id or not model or not auth_token:
        raise ValueError("host, plan_id, plan_hash, request_id, model, and auth_token must be non-empty")
    if isinstance(port, bool) or not isinstance(port, int) or port < 0:
        raise ValueError("port must be a non-negative integer")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight <= 0:
        raise ValueError("max_inflight must be a positive integer")
    if isinstance(backpressure_delay_ms, bool) or not isinstance(backpressure_delay_ms, int) or backpressure_delay_ms <= 0:
        raise ValueError("backpressure_delay_ms must be a positive integer")
    if isinstance(retry_after_ms, bool) or not isinstance(retry_after_ms, int) or retry_after_ms <= 0:
        raise ValueError("retry_after_ms must be a positive integer")
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
        "auth_token": auth_token,
        "max_inflight": max_inflight,
        "backpressure_delay_ms": backpressure_delay_ms,
        "retry_after_ms": retry_after_ms,
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
            auth_token=auth_token,
            timeout_s=float(timeout_s),
        )
        stream = _post_sse(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": True},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
            timeout_s=float(timeout_s),
        )
        auth_reject = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=None,
            timeout_s=float(timeout_s),
        )
        backpressure_holders: list[dict[str, Any]] = []
        backpressure_errors: list[BaseException] = []
        holder_lock = threading.Lock()

        def _hold_inflight_request() -> None:
            try:
                holder = _post_json(
                    endpoint,
                    {
                        "model": model,
                        "messages": adapter["openai_request"]["messages"],
                        "max_tokens": max_tokens,
                        "stream": False,
                        "simulate_work_ms": backpressure_delay_ms,
                    },
                    plan_id=plan_id,
                    plan_hash=plan_hash,
                    auth_token=auth_token,
                    timeout_s=float(timeout_s),
                )
                with holder_lock:
                    backpressure_holders.append(holder)
            except BaseException as exc:  # pragma: no cover - re-raised in the main thread.
                backpressure_errors.append(exc)

        holder_threads = [
            threading.Thread(target=_hold_inflight_request, name=f"fornax-http-smoke-hold-{index}")
            for index in range(max_inflight)
        ]
        for holder_thread in holder_threads:
            holder_thread.start()
        inflight_observed = server.wait_for_inflight(max_inflight, float(timeout_s))
        backpressure_reject = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
            timeout_s=float(timeout_s),
        )
        for holder_thread in holder_threads:
            holder_thread.join(timeout=float(timeout_s))
            if holder_thread.is_alive():
                raise TimeoutError("timed out waiting for local backpressure holder request")
        if backpressure_errors:
            raise backpressure_errors[0]
        plan_reject = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash="sha256:mismatch",
            auth_token=auth_token,
            timeout_s=float(timeout_s),
        )
        bad_path = _post_json(
            f"http://{server_host}:{server_port}/bad/path",
            {"model": model},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
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
    auth_reject_ok = (
        auth_reject.get("status") == 401
        and auth_reject.get("body", {}).get("error", {}).get("code") == "endpoint_auth_required"
    )
    backpressure_summary = server.backpressure_summary()
    lifecycle_summary = server.lifecycle_summary()
    expected_backend_request_count = 2 + max_inflight
    backpressure_holder_ok = (
        inflight_observed
        and len(backpressure_holders) == max_inflight
        and all(
            holder.get("status") == 200
            and holder.get("body", {}).get("object") == "chat.completion"
            for holder in backpressure_holders
        )
    )
    backpressure_reject_ok = (
        backpressure_reject.get("status") == 429
        and backpressure_reject.get("body", {}).get("error", {}).get("code") == "backpressure_queue_full"
        and backpressure_summary.get("backpressure_reject_count") == 1
        and backpressure_summary.get("max_observed_inflight") == max_inflight
        and backpressure_summary.get("current_inflight") == 0
    )
    lifecycle_ok = (
        lifecycle_summary.get("accepted_request_count") == expected_backend_request_count
        and lifecycle_summary.get("rejected_request_count") == 4
        and lifecycle_summary.get("cleanup_count") == expected_backend_request_count
        and lifecycle_summary.get("resource_allocated_count") == expected_backend_request_count * len(LOCAL_LIFECYCLE_RESOURCE_KINDS)
        and lifecycle_summary.get("resource_released_count") == lifecycle_summary.get("resource_allocated_count")
        and lifecycle_summary.get("active_resource_count") == 0
        and lifecycle_summary.get("all_required_resources_released") is True
        and lifecycle_summary.get("single_owner_preserved") is True
    )
    bad_path_ok = bad_path.get("status") == 404
    backend_ok = (
        backend_summary.get("backend") == "FornaxBackend"
        and backend_summary.get("engine_trait_compatible") is True
        and backend_summary.get("request_count") == expected_backend_request_count
        and backend_summary.get("target_model_loaded") is False
        and backend_summary.get("target_model_parity") is False
    )
    checks = [
        {"name": "serving-adapter", "ok": bool(adapter_validation.get("ok")), "errors": adapter_validation.get("errors", []), "warnings": adapter_validation.get("warnings", [])},
        {"name": "fornax-backend-integration", "ok": backend_ok, "errors": [] if backend_ok else ["FornaxBackend local integration invalid"], "warnings": []},
        {"name": "endpoint-auth-reject", "ok": auth_reject_ok, "errors": [] if auth_reject_ok else ["endpoint auth rejection invalid"], "warnings": []},
        {"name": "backpressure-reject", "ok": backpressure_reject_ok and backpressure_holder_ok, "errors": [] if backpressure_reject_ok and backpressure_holder_ok else ["backpressure rejection invalid"], "warnings": []},
        {"name": "lifecycle-cleanup", "ok": lifecycle_ok, "errors": [] if lifecycle_ok else ["lifecycle cleanup invalid"], "warnings": []},
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
        "auth_reject_status": auth_reject.get("status"),
        "endpoint_auth_rejected": auth_reject_ok,
        "backpressure_status": backpressure_reject.get("status"),
        "backpressure_rejected": backpressure_reject_ok,
        "backpressure_holder_count": len(backpressure_holders),
        "backpressure_holder_statuses": [holder.get("status") for holder in backpressure_holders],
        "backpressure_holders_completed": backpressure_holder_ok,
        "max_inflight": max_inflight,
        "max_observed_inflight": backpressure_summary.get("max_observed_inflight"),
        "backpressure_reject_count": backpressure_summary.get("backpressure_reject_count"),
        "inflight_cleanup_count": backpressure_summary.get("inflight_cleanup_count"),
        "failure_semantics_verified": backpressure_reject_ok and backpressure_holder_ok,
        "lifecycle_tracked": True,
        "lifecycle_request_count": lifecycle_summary.get("accepted_request_count"),
        "lifecycle_rejected_request_count": lifecycle_summary.get("rejected_request_count"),
        "lifecycle_cleanup_count": lifecycle_summary.get("cleanup_count"),
        "lifecycle_resource_allocated_count": lifecycle_summary.get("resource_allocated_count"),
        "lifecycle_resource_released_count": lifecycle_summary.get("resource_released_count"),
        "lifecycle_active_resource_count": lifecycle_summary.get("active_resource_count"),
        "lifecycle_all_released": lifecycle_summary.get("all_required_resources_released"),
        "lifecycle_single_owner_preserved": lifecycle_summary.get("single_owner_preserved"),
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
        "local_auth_enabled": True,
        "auth_token_redacted": True,
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
        "config": {key: value for key, value in config.items() if key != "auth_token"},
        "auth": {
            "mode": "local-bearer-token",
            "authorization_header_checked": True,
            "token_redacted": True,
            "production_auth": False,
        },
        "lifecycle": lifecycle_summary,
        "backend": backend_summary,
        "serving_adapter": adapter,
        "responses": {
            "non_stream": non_stream,
            "stream": stream,
            "auth_reject": auth_reject,
            "backpressure_holders": backpressure_holders,
            "backpressure_reject": backpressure_reject,
            "plan_reject": plan_reject,
            "bad_path": bad_path,
        },
        "checks": checks,
        "summary": summary,
        "ok": passed_count == len(checks),
        "note": (
            "Local HTTP/SSE serving smoke for the OpenAI-compatible endpoint path. "
            "This proves local endpoint request/response behavior, plan-integrity "
            "rejection, local bearer-token auth rejection, deterministic "
            "backpressure rejection, and local lifecycle cleanup only; it is not "
            "TLS/product auth, target-model parity, real "
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
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    max_inflight = summary.get("max_inflight")
    expected_backend_request_count = 2 + max_inflight if isinstance(max_inflight, int) and not isinstance(max_inflight, bool) and max_inflight > 0 else None
    if expected_backend_request_count is None:
        errors.append("summary.max_inflight must be a positive integer")
    elif backend.get("request_count") != expected_backend_request_count:
        errors.append(f"backend.request_count must be {expected_backend_request_count}")
    if backend.get("target_model_loaded") is not False:
        errors.append("backend.target_model_loaded must be false")
    if backend.get("target_model_parity") is not False:
        errors.append("backend.target_model_parity must be false")
    config = data.get("config")
    if isinstance(config, dict) and "auth_token" in config:
        errors.append("config.auth_token must be redacted")
    lifecycle = data.get("lifecycle")
    if not isinstance(lifecycle, dict):
        errors.append("lifecycle must be an object")
        lifecycle = {}
    if lifecycle.get("mode") != "local-http-lifecycle-smoke":
        errors.append("lifecycle.mode must be local-http-lifecycle-smoke")
    if lifecycle.get("resource_kinds") != list(LOCAL_LIFECYCLE_RESOURCE_KINDS):
        errors.append("lifecycle.resource_kinds must match local lifecycle resource kinds")
    lifecycle_events = lifecycle.get("events")
    if not isinstance(lifecycle_events, list) or not lifecycle_events:
        errors.append("lifecycle.events must be a non-empty list")
        lifecycle_events = []
    if lifecycle.get("all_required_resources_released") is not True:
        errors.append("lifecycle.all_required_resources_released must be true")
    if lifecycle.get("single_owner_preserved") is not True:
        errors.append("lifecycle.single_owner_preserved must be true")
    auth = data.get("auth")
    if not isinstance(auth, dict):
        errors.append("auth must be an object")
        auth = {}
    if auth.get("mode") != "local-bearer-token":
        errors.append("auth.mode must be local-bearer-token")
    if auth.get("authorization_header_checked") is not True:
        errors.append("auth.authorization_header_checked must be true")
    if auth.get("token_redacted") is not True:
        errors.append("auth.token_redacted must be true")
    if auth.get("production_auth") is not False:
        errors.append("auth.production_auth must be false")
    responses = data.get("responses")
    if not isinstance(responses, dict):
        errors.append("responses must be an object")
        responses = {}
    auth_reject = responses.get("auth_reject") if isinstance(responses, dict) else None
    if not isinstance(auth_reject, dict):
        errors.append("responses.auth_reject must be an object")
    else:
        if auth_reject.get("status") != 401:
            errors.append("responses.auth_reject.status must be 401")
        if auth_reject.get("body", {}).get("error", {}).get("code") != "endpoint_auth_required":
            errors.append("responses.auth_reject error code must be endpoint_auth_required")
    backpressure_holders = responses.get("backpressure_holders") if isinstance(responses, dict) else None
    if not isinstance(backpressure_holders, list):
        errors.append("responses.backpressure_holders must be a list")
        backpressure_holders = []
    elif expected_backend_request_count is not None and len(backpressure_holders) != max_inflight:
        errors.append("responses.backpressure_holders length must match summary.max_inflight")
    for index, holder in enumerate(backpressure_holders):
        if not isinstance(holder, dict):
            errors.append(f"responses.backpressure_holders[{index}] must be an object")
        elif holder.get("status") != 200:
            errors.append(f"responses.backpressure_holders[{index}].status must be 200")
    backpressure_reject = responses.get("backpressure_reject") if isinstance(responses, dict) else None
    if not isinstance(backpressure_reject, dict):
        errors.append("responses.backpressure_reject must be an object")
    else:
        if backpressure_reject.get("status") != 429:
            errors.append("responses.backpressure_reject.status must be 429")
        if backpressure_reject.get("body", {}).get("error", {}).get("code") != "backpressure_queue_full":
            errors.append("responses.backpressure_reject error code must be backpressure_queue_full")
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
    if summary.get("auth_reject_status") != 401:
        errors.append("summary.auth_reject_status must be 401")
    if summary.get("endpoint_auth_rejected") is not True:
        errors.append("summary.endpoint_auth_rejected must be true")
    if summary.get("backpressure_status") != 429:
        errors.append("summary.backpressure_status must be 429")
    if summary.get("backpressure_rejected") is not True:
        errors.append("summary.backpressure_rejected must be true")
    if expected_backend_request_count is not None and summary.get("backpressure_holder_count") != max_inflight:
        errors.append("summary.backpressure_holder_count must match summary.max_inflight")
    holder_statuses = summary.get("backpressure_holder_statuses")
    if expected_backend_request_count is not None:
        if not isinstance(holder_statuses, list) or holder_statuses != [200] * max_inflight:
            errors.append("summary.backpressure_holder_statuses must all be 200")
    if summary.get("backpressure_holders_completed") is not True:
        errors.append("summary.backpressure_holders_completed must be true")
    if expected_backend_request_count is not None and summary.get("max_observed_inflight") != max_inflight:
        errors.append("summary.max_observed_inflight must match summary.max_inflight")
    if summary.get("backpressure_reject_count") != 1:
        errors.append("summary.backpressure_reject_count must be 1")
    if expected_backend_request_count is not None and summary.get("inflight_cleanup_count") != expected_backend_request_count:
        errors.append(f"summary.inflight_cleanup_count must be {expected_backend_request_count}")
    if summary.get("failure_semantics_verified") is not True:
        errors.append("summary.failure_semantics_verified must be true")
    if expected_backend_request_count is not None:
        expected_resource_count = expected_backend_request_count * len(LOCAL_LIFECYCLE_RESOURCE_KINDS)
        if summary.get("lifecycle_tracked") is not True:
            errors.append("summary.lifecycle_tracked must be true")
        if summary.get("lifecycle_request_count") != expected_backend_request_count:
            errors.append(f"summary.lifecycle_request_count must be {expected_backend_request_count}")
        if summary.get("lifecycle_rejected_request_count") != 4:
            errors.append("summary.lifecycle_rejected_request_count must be 4")
        if summary.get("lifecycle_cleanup_count") != expected_backend_request_count:
            errors.append(f"summary.lifecycle_cleanup_count must be {expected_backend_request_count}")
        if summary.get("lifecycle_resource_allocated_count") != expected_resource_count:
            errors.append(f"summary.lifecycle_resource_allocated_count must be {expected_resource_count}")
        if summary.get("lifecycle_resource_released_count") != expected_resource_count:
            errors.append(f"summary.lifecycle_resource_released_count must be {expected_resource_count}")
        if summary.get("lifecycle_active_resource_count") != 0:
            errors.append("summary.lifecycle_active_resource_count must be 0")
        if summary.get("lifecycle_all_released") is not True:
            errors.append("summary.lifecycle_all_released must be true")
        if summary.get("lifecycle_single_owner_preserved") is not True:
            errors.append("summary.lifecycle_single_owner_preserved must be true")
        if lifecycle.get("accepted_request_count") != expected_backend_request_count:
            errors.append(f"lifecycle.accepted_request_count must be {expected_backend_request_count}")
        if lifecycle.get("rejected_request_count") != 4:
            errors.append("lifecycle.rejected_request_count must be 4")
        if lifecycle.get("cleanup_count") != expected_backend_request_count:
            errors.append(f"lifecycle.cleanup_count must be {expected_backend_request_count}")
        if lifecycle.get("resource_allocated_count") != expected_resource_count:
            errors.append(f"lifecycle.resource_allocated_count must be {expected_resource_count}")
        if lifecycle.get("resource_released_count") != expected_resource_count:
            errors.append(f"lifecycle.resource_released_count must be {expected_resource_count}")
        if lifecycle.get("active_resource_count") != 0:
            errors.append("lifecycle.active_resource_count must be 0")
    if summary.get("plan_integrity_rejected") is not True:
        errors.append("summary.plan_integrity_rejected must be true")
    if summary.get("bad_path_rejected") is not True:
        errors.append("summary.bad_path_rejected must be true")
    if summary.get("fornax_backend_integrated") is not True:
        errors.append("summary.fornax_backend_integrated must be true")
    if expected_backend_request_count is not None and summary.get("backend_request_count") != expected_backend_request_count:
        errors.append(f"summary.backend_request_count must be {expected_backend_request_count}")
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
    if summary.get("local_auth_enabled") is not True:
        errors.append("summary.local_auth_enabled must be true")
    if summary.get("auth_token_redacted") is not True:
        errors.append("summary.auth_token_redacted must be true")
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
            "endpoint_auth_rejected": summary.get("endpoint_auth_rejected") is True,
            "backpressure_rejected": summary.get("backpressure_rejected") is True,
            "backpressure_reject_count": summary.get("backpressure_reject_count"),
            "failure_semantics_verified": summary.get("failure_semantics_verified") is True,
            "lifecycle_tracked": summary.get("lifecycle_tracked") is True,
            "lifecycle_request_count": summary.get("lifecycle_request_count"),
            "lifecycle_all_released": summary.get("lifecycle_all_released") is True,
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
