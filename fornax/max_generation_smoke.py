"""Bounded evidence for one local-model ``max generate`` invocation.

This module deliberately establishes only a single-device MAX generation
smoke.  It does not infer numerical parity, distributed execution, formal gate
closure, performance, or production support.

The command runner seam keeps the contract testable without MAX or accelerator
hardware.  A runner receives a direct argv sequence and a timeout in seconds;
shell command strings are never constructed or executed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .bounded_subprocess import run_bounded_subprocess
from .hardware_identity import NVIDIA_GPU_UUID_RE


CommandRunner = Callable[[Sequence[str], float], Any]

SCHEMA_VERSION = 1
RECORD_KIND = "fornax_max_generation_smoke"
EVIDENCE_SCOPE = "single_device_local_model_max_generation"

DEFAULT_DEVICE = "gpu:0"
DEFAULT_MAX_ARGV_PREFIX = ("max",)
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_TOP_K = 1
DEFAULT_TIMEOUT_SECONDS = 1800.0
SMOKE_SENTINEL = "FORNAX_MOE_SMOKE_OK"
SMOKE_SENTINEL_PROMPT = (
    "Reply with exactly FORNAX_MOE_SMOKE_OK and nothing else."
)

MAX_MODEL_ID_CHARS = 512
MAX_PATH_CHARS = 4096
MAX_DEVICE_CHARS = 64
MAX_ENCODING_CHARS = 128
MAX_PROMPT_CHARS = 16 * 1024
MAX_NEW_TOKENS_LIMIT = 4096
MAX_TOP_K_LIMIT = 4096
MAX_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_ARGV_PREFIX_PARTS = 32
MAX_ARGV_PART_CHARS = 4096
MAX_VERSION_SECONDS = 30.0
MAX_VERSION_BYTES = 64 * 1024
MAX_GENERATION_STDOUT_BYTES = 4 * 1024 * 1024
MAX_GENERATION_STDERR_BYTES = 4 * 1024 * 1024
MAX_VERSION_STDERR_BYTES = 256 * 1024
MAX_VERSION_CHARS = 512
MAX_OUTPUT_EXCERPT_CHARS = 8192
MAX_GENERATED_TEXT_CHARS = 2048
MAX_SIGNAL_ANALYSIS_CHARS = 64 * 1024
MAX_ERROR_CHARS = 2048

CLAIM_KEYS = (
    "single_platform_bringup_passed",
    "distributed_execution_passed",
    "target_model_parity_passed",
    "formal_g2_passed",
    "formal_g3_passed",
    "production_supported",
    "production_distributed_serving",
)

_DEVICE_RE = re.compile(r"^gpu:(0|[1-9][0-9]*)$")
_ENCODING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GENERATED_MARKER_RE = re.compile(
    r"^(?:generated(?:\s+(?:text|output))?|response|assistant)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_PROMPT_SIZE_RE = re.compile(r"^Prompt size:\s*([0-9]+)\s*$", re.MULTILINE)
_OUTPUT_SIZE_RE = re.compile(r"^Output size:\s*([0-9]+)\s*$", re.MULTILINE)

LIVE_RUNNER_KIND = "live_subprocess"
SYNTHETIC_RUNNER_KIND = "synthetic_injected_test_runner"


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _bounded_text(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be valid UTF-8 text") from exc
    if len(value) > limit:
        raise ValueError(f"{field} exceeds the {limit}-character limit")
    return value


def _validated_prompt(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("prompt must contain non-whitespace text")
    if "\x00" in value:
        raise ValueError("prompt must not contain NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("prompt must be valid UTF-8 text") from exc
    if len(value) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"prompt exceeds the {MAX_PROMPT_CHARS}-character limit"
        )
    return value


def _positive_bounded_int(name: str, value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if value > maximum:
        raise ValueError(f"{name} exceeds the bounded limit {maximum}")
    return value


def _validated_timeout(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("timeout_s must be a finite positive number")
    timeout = float(value)
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_s exceeds the bounded limit {MAX_TIMEOUT_SECONDS}"
        )
    return timeout


def _validated_max_argv_prefix(value: Sequence[str] | None) -> tuple[str, ...]:
    raw = DEFAULT_MAX_ARGV_PREFIX if value is None else value
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("max_argv_prefix must be a non-empty argv sequence")
    prefix = tuple(raw)
    if not prefix:
        raise ValueError("max_argv_prefix must be a non-empty argv sequence")
    if len(prefix) > MAX_ARGV_PREFIX_PARTS:
        raise ValueError(
            "max_argv_prefix exceeds the bounded argument-count limit "
            f"{MAX_ARGV_PREFIX_PARTS}"
        )
    for index, part in enumerate(prefix):
        if not isinstance(part, str) or not part:
            raise ValueError(
                f"max_argv_prefix[{index}] must be a non-empty string"
            )
        if "\x00" in part:
            raise ValueError(f"max_argv_prefix[{index}] must not contain NUL")
        try:
            part.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"max_argv_prefix[{index}] must be valid UTF-8 text"
            ) from exc
        if len(part) > MAX_ARGV_PART_CHARS:
            raise ValueError(
                f"max_argv_prefix[{index}] exceeds the "
                f"{MAX_ARGV_PART_CHARS}-character limit"
            )
    return prefix


def _coerce_output_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"{field} must be text or bytes")


def _coerce_returncode(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("command returncode must be an integer")
    return value


def _coerce_command_result(result: Any) -> tuple[int, str, str]:
    if isinstance(result, (str, bytes)):
        return 0, _coerce_output_text(result, "stdout"), ""
    if isinstance(result, Mapping):
        return (
            _coerce_returncode(result.get("returncode", result.get("code", 0))),
            _coerce_output_text(result.get("stdout", ""), "stdout"),
            _coerce_output_text(result.get("stderr", ""), "stderr"),
        )
    if isinstance(result, tuple):
        if len(result) == 2:
            return (
                _coerce_returncode(result[0]),
                _coerce_output_text(result[1], "stdout"),
                "",
            )
        if len(result) == 3:
            return (
                _coerce_returncode(result[0]),
                _coerce_output_text(result[1], "stdout"),
                _coerce_output_text(result[2], "stderr"),
            )
        raise TypeError("command-result tuples must contain two or three items")
    if hasattr(result, "returncode"):
        return (
            _coerce_returncode(getattr(result, "returncode")),
            _coerce_output_text(getattr(result, "stdout", ""), "stdout"),
            _coerce_output_text(getattr(result, "stderr", ""), "stderr"),
        )
    raise TypeError(
        "command runner must return text, a mapping, a "
        "(returncode, stdout[, stderr]) tuple, or a subprocess-style result"
    )


def _default_command_runner(
    argv: Sequence[str],
    timeout_s: float,
    *,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    generation_command = "generate" in argv
    return run_bounded_subprocess(
        list(argv),
        timeout_s=timeout_s,
        stdout_limit_bytes=(
            MAX_GENERATION_STDOUT_BYTES
            if generation_command
            else MAX_VERSION_BYTES
        ),
        stderr_limit_bytes=(
            MAX_GENERATION_STDERR_BYTES
            if generation_command
            else MAX_VERSION_STDERR_BYTES
        ),
        env=env,
    )


def _excerpt(value: str, limit: int = MAX_OUTPUT_EXCERPT_CHARS) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[-limit:], True


def _error_excerpt(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_ERROR_CHARS:
        return value, False
    return value[:MAX_ERROR_CHARS], True


def _run_command(
    runner: CommandRunner,
    argv: Sequence[str],
    timeout_s: float,
) -> tuple[dict[str, Any], str, str]:
    started = time.perf_counter()
    returncode: int | None = None
    stdout = ""
    stderr = ""
    launch_error: str | None = None
    try:
        result = runner(tuple(argv), timeout_s)
        returncode, stdout, stderr = _coerce_command_result(result)
    except Exception as exc:
        launch_error, launch_error_truncated = _error_excerpt(
            f"{type(exc).__name__}: {exc}"
        )
    else:
        launch_error_truncated = False
    elapsed_s = round(time.perf_counter() - started, 6)

    stdout_excerpt, stdout_excerpt_truncated = _excerpt(stdout)
    stderr_excerpt, stderr_excerpt_truncated = _excerpt(stderr)
    record = {
        "argv": list(argv),
        "shell": False,
        "timeout_s": timeout_s,
        "elapsed_s": elapsed_s,
        "returncode": returncode,
        "launch_error": launch_error,
        "launch_error_truncated": launch_error_truncated,
        "stdout_chars": len(stdout),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stdout_text_sha256": _sha256_text(stdout),
        "stdout_excerpt": stdout_excerpt,
        "stdout_excerpt_truncated": stdout_excerpt_truncated,
        "stderr_chars": len(stderr),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stderr_text_sha256": _sha256_text(stderr),
        "stderr_excerpt": stderr_excerpt,
        "stderr_excerpt_truncated": stderr_excerpt_truncated,
        "ok": launch_error is None and returncode == 0,
    }
    return record, stdout, stderr


def _json_generated_text(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    for key in ("generated_text", "output_text"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip(), f"json_{key}"
    choices = value.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip(), "json_choices_message_content"
        text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip(), "json_choices_text"
    return None


def extract_exact_max_metrics_sentinel(stdout: str, prompt: str) -> str:
    """Recognize MAX's raw-token output only for the exact smoke sentinel.

    MAX 26.x ``generate`` writes the decoded token text directly to stdout,
    followed by ``Prompt size`` and ``Output size`` metric lines.  Generic
    residual stdout is not trustworthy generation evidence, so this format is
    accepted only when the invoked prompt is the exact sentinel instruction,
    the complete pre-metrics segment is the exact sentinel, and the reported
    output size is positive.
    """

    if prompt != SMOKE_SENTINEL_PROMPT:
        return ""
    prompt_matches = tuple(_PROMPT_SIZE_RE.finditer(stdout))
    output_matches = tuple(_OUTPUT_SIZE_RE.finditer(stdout))
    if len(prompt_matches) != 1 or len(output_matches) != 1:
        return ""
    prompt_metric = prompt_matches[0]
    output_metric = output_matches[0]
    if output_metric.start() <= prompt_metric.end():
        return ""
    if int(prompt_metric.group(1)) <= 0 or int(output_metric.group(1)) <= 0:
        return ""
    if stdout[: prompt_metric.start()].strip() != SMOKE_SENTINEL:
        return ""
    between_metrics = stdout[prompt_metric.end() : output_metric.start()]
    if between_metrics.strip():
        return ""
    return SMOKE_SENTINEL


def is_diagnostic_generated_text(value: str) -> bool:
    """Return whether an explicitly framed candidate is a known status line."""

    normalized = " ".join(value.split()).casefold()
    if normalized in {
        "compilation complete",
        "compilation completed",
        "model loaded",
        "generation complete",
    }:
        return True
    return bool(
        re.fullmatch(r"(?:prompt|output) size:\s*[0-9]+", normalized)
        or normalized.startswith("architecture:")
    )


def _generated_text_signal(stdout: str, prompt: str) -> dict[str, Any]:
    analysis_text = stdout[-MAX_SIGNAL_ANALYSIS_CHARS:]
    candidate: str | None = None
    method: str | None = None

    if len(analysis_text) == len(stdout):
        try:
            payload = json.loads(analysis_text)
        except (json.JSONDecodeError, UnicodeError):
            payload = None
        extracted = _json_generated_text(payload)
        if extracted is not None:
            candidate, method = extracted

    lines = [line.strip() for line in analysis_text.splitlines() if line.strip()]
    if candidate is None:
        for index in range(len(lines) - 1, -1, -1):
            match = _GENERATED_MARKER_RE.match(lines[index])
            if match is None:
                continue
            marked = match.group(1).strip()
            if marked:
                candidate, method = marked, "explicit_generated_text_marker"
                break
    if candidate is None and len(analysis_text) == len(stdout):
        sentinel = extract_exact_max_metrics_sentinel(stdout, prompt)
        if sentinel:
            candidate = sentinel
            method = "max_metrics_exact_sentinel"

    normalized = candidate.strip() if candidate is not None else ""
    if (
        normalized == prompt.strip()
        or (normalized and is_diagnostic_generated_text(normalized))
    ):
        normalized = ""
        method = None
    bounded = normalized[:MAX_GENERATED_TEXT_CHARS]
    return {
        "detected": bool(bounded),
        "method": method if bounded else None,
        "text_excerpt": bounded,
        "text_excerpt_chars": len(bounded),
        "text_excerpt_sha256": _sha256_text(bounded),
        "text_excerpt_truncated": len(normalized) > len(bounded),
        "analysis_tail_truncated": len(stdout) > len(analysis_text),
    }


def _generation_argv(
    prefix: Sequence[str],
    *,
    model_dir: str,
    device: str,
    quantization_encoding: str,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
) -> tuple[str, ...]:
    return (
        *prefix,
        "generate",
        "--model",
        model_dir,
        "--devices",
        device,
        "--quantization-encoding",
        quantization_encoding,
        "--max-new-tokens",
        str(max_new_tokens),
        "--top-k",
        str(top_k),
        "--temperature",
        "0",
        "--prompt",
        prompt,
    )


def _integrity_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "integrity"}


def run_max_generation_smoke(
    *,
    model_id: str,
    model_dir: str | Path,
    quantization_encoding: str,
    device: str = DEFAULT_DEVICE,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    max_argv_prefix: Sequence[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    command_runner: CommandRunner | None = None,
    nvidia_smi_index: int | None = None,
    nvidia_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Run a bounded, single-device MAX generation smoke.

    ``model_dir`` must already exist locally.  The generated command never uses
    ``model_id`` as a downloadable model reference.
    """

    validated_model_id = _bounded_text(
        model_id, "model_id", MAX_MODEL_ID_CHARS
    )
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        raise ValueError("model_dir must be an existing directory")
    resolved_model_dir = str(path.resolve())
    try:
        resolved_model_dir.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("resolved model_dir must be valid UTF-8 text") from exc
    if len(resolved_model_dir) > MAX_PATH_CHARS:
        raise ValueError(
            f"resolved model_dir exceeds the {MAX_PATH_CHARS}-character limit"
        )
    validated_device = _bounded_text(device, "device", MAX_DEVICE_CHARS)
    if _DEVICE_RE.fullmatch(validated_device) is None:
        raise ValueError("device must be one exact MAX GPU device such as gpu:0")
    encoding = _bounded_text(
        quantization_encoding,
        "quantization_encoding",
        MAX_ENCODING_CHARS,
    )
    if _ENCODING_RE.fullmatch(encoding) is None:
        raise ValueError(
            "quantization_encoding contains unsupported characters"
        )
    validated_prompt = _validated_prompt(prompt)
    validated_tokens = _positive_bounded_int(
        "max_new_tokens", max_new_tokens, MAX_NEW_TOKENS_LIMIT
    )
    validated_top_k = _positive_bounded_int(
        "top_k", top_k, MAX_TOP_K_LIMIT
    )
    validated_timeout = _validated_timeout(timeout_s)
    prefix = _validated_max_argv_prefix(max_argv_prefix)
    binding_requested = (
        nvidia_smi_index is not None or nvidia_gpu_uuid is not None
    )
    if (nvidia_smi_index is None) != (nvidia_gpu_uuid is None):
        raise ValueError(
            "nvidia_smi_index and nvidia_gpu_uuid must be supplied together"
        )
    if binding_requested:
        if (
            isinstance(nvidia_smi_index, bool)
            or not isinstance(nvidia_smi_index, int)
            or nvidia_smi_index < 0
        ):
            raise ValueError("nvidia_smi_index must be a non-negative integer")
        if (
            not isinstance(nvidia_gpu_uuid, str)
            or NVIDIA_GPU_UUID_RE.fullmatch(nvidia_gpu_uuid) is None
        ):
            raise ValueError("nvidia_gpu_uuid must be an exact physical GPU UUID")
        if validated_device != "gpu:0":
            raise ValueError(
                "UUID-bound NVIDIA launches must use MAX device gpu:0"
            )
    live_runner = command_runner is None
    launch_environment: dict[str, str] | None = None
    if live_runner and binding_requested:
        assert nvidia_gpu_uuid is not None
        launch_environment = os.environ.copy()
        launch_environment["CUDA_VISIBLE_DEVICES"] = nvidia_gpu_uuid

        def runner(argv: Sequence[str], command_timeout_s: float) -> Any:
            return _default_command_runner(
                argv,
                command_timeout_s,
                env=launch_environment,
            )

    else:
        runner = _default_command_runner if live_runner else command_runner
    runner_kind = LIVE_RUNNER_KIND if live_runner else SYNTHETIC_RUNNER_KIND
    device_binding = {
        "mode": (
            "nvidia_gpu_uuid_to_visible_ordinal"
            if binding_requested
            else "unbound_max_ordinal"
        ),
        "physical_nvidia_smi_index": nvidia_smi_index,
        "physical_nvidia_gpu_uuid": nvidia_gpu_uuid,
        "cuda_visible_devices": nvidia_gpu_uuid,
        "max_device": validated_device,
        "applied_to_live_subprocess": bool(live_runner and binding_requested),
    }

    version_argv = (*prefix, "--version")
    generation_argv = _generation_argv(
        prefix,
        model_dir=resolved_model_dir,
        device=validated_device,
        quantization_encoding=encoding,
        prompt=validated_prompt,
        max_new_tokens=validated_tokens,
        top_k=validated_top_k,
    )
    version_timeout = min(validated_timeout, MAX_VERSION_SECONDS)
    version_record, version_stdout, _ = _run_command(
        runner, version_argv, version_timeout
    )
    generation_record, generation_stdout, _ = _run_command(
        runner, generation_argv, validated_timeout
    )

    errors: list[str] = []
    max_version: str | None = None
    version_text = version_stdout.strip()
    if not version_record["ok"]:
        detail = (
            version_record["launch_error"]
            or version_record["stderr_excerpt"].strip()
            or f"exit status {version_record['returncode']}"
        )
        errors.append(f"MAX version probe failed: {detail}")
    elif version_record["stdout_bytes"] > MAX_VERSION_BYTES:
        errors.append(
            "MAX version output exceeds the bounded limit "
            f"{MAX_VERSION_BYTES} bytes"
        )
    elif not version_text:
        errors.append("MAX version probe returned no version text")
    elif len(version_text) > MAX_VERSION_CHARS:
        errors.append(
            "MAX version text exceeds the bounded limit "
            f"{MAX_VERSION_CHARS} characters"
        )
    else:
        max_version = version_text

    generated_text = _generated_text_signal(
        generation_stdout, validated_prompt
    )
    if not generation_record["ok"]:
        detail = (
            generation_record["launch_error"]
            or generation_record["stderr_excerpt"].strip()
            or f"exit status {generation_record['returncode']}"
        )
        errors.append(f"MAX generation failed: {detail}")
    elif not generated_text["detected"]:
        errors.append(
            "MAX generation returned zero but emitted no conservative "
            "non-empty generated-text signal"
        )

    passed = (
        not errors
        and max_version is not None
        and generation_record["returncode"] == 0
        and generated_text["detected"] is True
    )
    claims = {key: False for key in CLAIM_KEYS}
    claims["single_platform_bringup_passed"] = passed and live_runner
    warnings = [
        "This is bounded single-device generation evidence only. It does "
        "not establish numerical parity, distributed execution, formal "
        "G2/G3 closure, performance, or production support."
    ]
    if not live_runner:
        warnings.append(
            "The command runner was injected, so this record is synthetic/test "
            "evidence and cannot establish a physical bring-up claim."
        )
    if not binding_requested:
        warnings.append(
            "No physical NVIDIA GPU UUID binding was requested. inputs.device "
            "is only a MAX-visible ordinal and must not be interpreted as an "
            "nvidia-smi index."
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": RECORD_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "ok": passed,
        "errors": errors,
        "warnings": warnings,
        "runner": {
            "kind": runner_kind,
            "physical_execution_eligible": live_runner,
            "device_binding": device_binding,
        },
        "inputs": {
            "model_id": validated_model_id,
            "model_dir": resolved_model_dir,
            "device": validated_device,
            "quantization_encoding": encoding,
            "prompt": validated_prompt,
            "prompt_sha256": _sha256_text(validated_prompt),
            "max_new_tokens": validated_tokens,
            "top_k": validated_top_k,
            "max_argv_prefix": list(prefix),
            "timeout_s": validated_timeout,
        },
        "observed": {
            "max_version": max_version,
            "generated_text_signal": generated_text,
        },
        "commands": {
            "version": version_record,
            "generation": generation_record,
        },
        "claims": claims,
        "interpretation": (
            (
                "A passing live-subprocess record proves only that the named "
                "local model invocation returned zero and exposed an explicitly "
                "framed, bounded non-empty generated-text signal on the exact "
                "requested single device."
            )
            if live_runner
            else (
                "The injected runner makes this synthetic contract evidence. "
                "Even when the simulated invocation succeeds, no physical "
                "bring-up claim is permitted."
            )
        ),
    }
    report["integrity"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": _canonical_sha256(
            _integrity_payload(report)
        ),
    }
    return report


def _validation_string(
    value: Any,
    field: str,
    errors: list[str],
    *,
    non_empty: bool = True,
) -> str | None:
    if not isinstance(value, str) or (non_empty and not value):
        errors.append(f"{field} must be a{' non-empty' if non_empty else ''} string")
        return None
    if "\x00" in value:
        errors.append(f"{field} must not contain NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        errors.append(f"{field} must be valid UTF-8 text")
        return None
    return value


def _validate_command_record(
    value: Any,
    field: str,
    expected_argv: Sequence[str] | None,
    expected_timeout: float | None,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    expected_keys = {
        "argv",
        "shell",
        "timeout_s",
        "elapsed_s",
        "returncode",
        "launch_error",
        "launch_error_truncated",
        "stdout_chars",
        "stdout_bytes",
        "stdout_text_sha256",
        "stdout_excerpt",
        "stdout_excerpt_truncated",
        "stderr_chars",
        "stderr_bytes",
        "stderr_text_sha256",
        "stderr_excerpt",
        "stderr_excerpt_truncated",
        "ok",
    }
    if set(value) != expected_keys:
        errors.append(f"{field} fields must exactly match the schema")
    argv = value.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        errors.append(f"{field}.argv must be a non-empty argv string list")
    elif expected_argv is not None and argv != list(expected_argv):
        errors.append(f"{field}.argv does not match the declared inputs")
    if value.get("shell") is not False:
        errors.append(f"{field}.shell must be false")
    timeout = value.get("timeout_s")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        errors.append(f"{field}.timeout_s must be a finite positive number")
    elif expected_timeout is not None and float(timeout) != expected_timeout:
        errors.append(f"{field}.timeout_s does not match the declared timeout")
    elapsed = value.get("elapsed_s")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        errors.append(f"{field}.elapsed_s must be a finite non-negative number")
    returncode = value.get("returncode")
    if returncode is not None and (
        isinstance(returncode, bool) or not isinstance(returncode, int)
    ):
        errors.append(f"{field}.returncode must be an integer or null")
    launch_error = value.get("launch_error")
    if launch_error is not None and not isinstance(launch_error, str):
        errors.append(f"{field}.launch_error must be a string or null")
    for suffix in (
        "launch_error_truncated",
        "stdout_excerpt_truncated",
        "stderr_excerpt_truncated",
    ):
        if not isinstance(value.get(suffix), bool):
            errors.append(f"{field}.{suffix} must be boolean")
    for stream in ("stdout", "stderr"):
        chars = value.get(f"{stream}_chars")
        byte_count = value.get(f"{stream}_bytes")
        excerpt = value.get(f"{stream}_excerpt")
        digest = value.get(f"{stream}_text_sha256")
        truncated = value.get(f"{stream}_excerpt_truncated")
        excerpt_valid_utf8 = isinstance(excerpt, str)
        if isinstance(chars, bool) or not isinstance(chars, int) or chars < 0:
            errors.append(f"{field}.{stream}_chars must be a non-negative integer")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            errors.append(f"{field}.{stream}_bytes must be a non-negative integer")
        if not isinstance(excerpt, str):
            errors.append(f"{field}.{stream}_excerpt must be a string")
        elif len(excerpt) > MAX_OUTPUT_EXCERPT_CHARS:
            errors.append(
                f"{field}.{stream}_excerpt exceeds the bounded limit"
            )
        else:
            try:
                excerpt.encode("utf-8")
            except UnicodeEncodeError:
                excerpt_valid_utf8 = False
                errors.append(
                    f"{field}.{stream}_excerpt must be valid UTF-8 text"
                )
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            errors.append(f"{field}.{stream}_text_sha256 must be sha256:<hex>")
        if (
            isinstance(chars, int)
            and not isinstance(chars, bool)
            and isinstance(excerpt, str)
            and excerpt_valid_utf8
            and isinstance(truncated, bool)
        ):
            if truncated != (chars > MAX_OUTPUT_EXCERPT_CHARS):
                errors.append(
                    f"{field}.{stream}_excerpt_truncated is inconsistent"
                )
            if not truncated:
                if len(excerpt) != chars:
                    errors.append(f"{field}.{stream}_chars is inconsistent")
                if (
                    isinstance(digest, str)
                    and digest != _sha256_text(excerpt)
                ):
                    errors.append(
                        f"{field}.{stream}_text_sha256 is inconsistent"
                    )
                if (
                    isinstance(byte_count, int)
                    and not isinstance(byte_count, bool)
                    and byte_count != len(excerpt.encode("utf-8"))
                ):
                    errors.append(f"{field}.{stream}_bytes is inconsistent")
    derived_ok = launch_error is None and returncode == 0
    if value.get("ok") is not derived_ok:
        errors.append(f"{field}.ok is inconsistent with command outcome")
    return value


def validate_max_generation_smoke_evidence(
    report: Any,
) -> dict[str, Any]:
    """Validate structure, integrity, command binding, and claim boundaries."""

    errors: list[str] = []
    if not isinstance(report, dict):
        return {
            "ok": False,
            "errors": ["report must be an object"],
            "summary": {},
        }

    expected_top_keys = {
        "schema_version",
        "record_kind",
        "evidence_scope",
        "ok",
        "errors",
        "warnings",
        "runner",
        "inputs",
        "observed",
        "commands",
        "claims",
        "interpretation",
        "integrity",
    }
    if set(report) != expected_top_keys:
        errors.append("report fields must exactly match the schema")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND!r}")
    if report.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must be {EVIDENCE_SCOPE!r}")

    runner = report.get("runner")
    physical_execution_eligible = False
    runner_kind: str | None = None
    device_binding: dict[str, Any] | None = None
    if not isinstance(runner, dict) or set(runner) != {
        "kind",
        "physical_execution_eligible",
        "device_binding",
    }:
        errors.append("runner must exactly match the provenance schema")
    else:
        raw_runner_kind = runner.get("kind")
        if raw_runner_kind not in {LIVE_RUNNER_KIND, SYNTHETIC_RUNNER_KIND}:
            errors.append("runner.kind is unsupported")
        else:
            runner_kind = raw_runner_kind
        eligibility = runner.get("physical_execution_eligible")
        if not isinstance(eligibility, bool):
            errors.append("runner.physical_execution_eligible must be boolean")
        else:
            physical_execution_eligible = eligibility
        expected_eligibility = raw_runner_kind == LIVE_RUNNER_KIND
        if isinstance(eligibility, bool) and eligibility is not expected_eligibility:
            errors.append(
                "runner.physical_execution_eligible is inconsistent with runner.kind"
            )
        raw_binding = runner.get("device_binding")
        binding_keys = {
            "mode",
            "physical_nvidia_smi_index",
            "physical_nvidia_gpu_uuid",
            "cuda_visible_devices",
            "max_device",
            "applied_to_live_subprocess",
        }
        if not isinstance(raw_binding, dict) or set(raw_binding) != binding_keys:
            errors.append(
                "runner.device_binding must exactly match the binding schema"
            )
        else:
            device_binding = raw_binding
            binding_mode = raw_binding.get("mode")
            physical_index = raw_binding.get("physical_nvidia_smi_index")
            physical_uuid = raw_binding.get("physical_nvidia_gpu_uuid")
            cuda_visible = raw_binding.get("cuda_visible_devices")
            max_device = raw_binding.get("max_device")
            applied = raw_binding.get("applied_to_live_subprocess")
            if binding_mode == "unbound_max_ordinal":
                if (
                    physical_index is not None
                    or physical_uuid is not None
                    or cuda_visible is not None
                    or applied is not False
                ):
                    errors.append(
                        "unbound runner.device_binding must not claim a physical "
                        "NVIDIA device or environment override"
                    )
            elif binding_mode == "nvidia_gpu_uuid_to_visible_ordinal":
                if (
                    isinstance(physical_index, bool)
                    or not isinstance(physical_index, int)
                    or physical_index < 0
                ):
                    errors.append(
                        "runner.device_binding physical index must be non-negative"
                    )
                if (
                    not isinstance(physical_uuid, str)
                    or NVIDIA_GPU_UUID_RE.fullmatch(physical_uuid) is None
                ):
                    errors.append(
                        "runner.device_binding physical UUID is invalid"
                    )
                if cuda_visible != physical_uuid:
                    errors.append(
                        "runner.device_binding CUDA_VISIBLE_DEVICES must equal "
                        "the selected physical UUID"
                    )
                if max_device != "gpu:0":
                    errors.append(
                        "runner.device_binding MAX launch device must be gpu:0"
                    )
                expected_applied = raw_runner_kind == LIVE_RUNNER_KIND
                if applied is not expected_applied:
                    errors.append(
                        "runner.device_binding applied flag is inconsistent "
                        "with runner provenance"
                    )
            else:
                errors.append("runner.device_binding mode is unsupported")
            if not isinstance(max_device, str) or _DEVICE_RE.fullmatch(max_device) is None:
                errors.append("runner.device_binding max_device is invalid")
            if not isinstance(applied, bool):
                errors.append(
                    "runner.device_binding applied_to_live_subprocess must be boolean"
                )

    integrity = report.get("integrity")
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "canonical_payload_sha256",
    }:
        errors.append("integrity must exactly match the schema")
    else:
        if integrity.get("algorithm") != "sha256":
            errors.append("integrity.algorithm must be sha256")
        digest = integrity.get("canonical_payload_sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            errors.append(
                "integrity.canonical_payload_sha256 must be sha256:<hex>"
            )
        else:
            try:
                expected_digest = _canonical_sha256(
                    _integrity_payload(report)
                )
            except (
                TypeError,
                ValueError,
                OverflowError,
                UnicodeError,
            ) as exc:
                errors.append(f"report is not canonical JSON: {type(exc).__name__}")
            else:
                if digest != expected_digest:
                    errors.append("integrity digest does not match report payload")

    inputs = report.get("inputs")
    expected_version_argv: tuple[str, ...] | None = None
    expected_generation_argv: tuple[str, ...] | None = None
    declared_timeout: float | None = None
    prompt: str | None = None
    if not isinstance(inputs, dict):
        errors.append("inputs must be an object")
    else:
        expected_input_keys = {
            "model_id",
            "model_dir",
            "device",
            "quantization_encoding",
            "prompt",
            "prompt_sha256",
            "max_new_tokens",
            "top_k",
            "max_argv_prefix",
            "timeout_s",
        }
        if set(inputs) != expected_input_keys:
            errors.append("inputs fields must exactly match the schema")
        model_id = _validation_string(
            inputs.get("model_id"), "inputs.model_id", errors
        )
        model_dir = _validation_string(
            inputs.get("model_dir"), "inputs.model_dir", errors
        )
        device = _validation_string(
            inputs.get("device"), "inputs.device", errors
        )
        encoding = _validation_string(
            inputs.get("quantization_encoding"),
            "inputs.quantization_encoding",
            errors,
        )
        prompt = _validation_string(
            inputs.get("prompt"), "inputs.prompt", errors
        )
        if model_id is not None:
            if len(model_id) > MAX_MODEL_ID_CHARS:
                errors.append("inputs.model_id exceeds the bounded limit")
            if model_id != model_id.strip():
                errors.append(
                    "inputs.model_id must not have surrounding whitespace"
                )
        if model_dir is not None:
            if len(model_dir) > MAX_PATH_CHARS:
                errors.append("inputs.model_dir exceeds the bounded limit")
            if not Path(model_dir).is_absolute():
                errors.append("inputs.model_dir must be an absolute path")
        if device is not None and _DEVICE_RE.fullmatch(device) is None:
            errors.append("inputs.device must be one exact gpu:<index> device")
        if (
            device is not None
            and device_binding is not None
            and device_binding.get("max_device") != device
        ):
            errors.append(
                "runner.device_binding.max_device must match inputs.device"
            )
        if encoding is not None and _ENCODING_RE.fullmatch(encoding) is None:
            errors.append("inputs.quantization_encoding is malformed")
        if prompt is not None:
            if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
                errors.append("inputs.prompt is empty or exceeds the bounded limit")
            if "\x00" in prompt:
                errors.append("inputs.prompt must not contain NUL")
            if inputs.get("prompt_sha256") != _sha256_text(prompt):
                errors.append("inputs.prompt_sha256 is inconsistent")
        tokens = inputs.get("max_new_tokens")
        top_k = inputs.get("top_k")
        for field, value, maximum in (
            ("max_new_tokens", tokens, MAX_NEW_TOKENS_LIMIT),
            ("top_k", top_k, MAX_TOP_K_LIMIT),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > maximum
            ):
                errors.append(f"inputs.{field} must be a bounded positive integer")
        timeout = inputs.get("timeout_s")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
            or float(timeout) > MAX_TIMEOUT_SECONDS
        ):
            errors.append("inputs.timeout_s must be a bounded finite positive number")
        else:
            declared_timeout = float(timeout)
        try:
            prefix = _validated_max_argv_prefix(inputs.get("max_argv_prefix"))
        except ValueError as exc:
            errors.append(f"inputs.max_argv_prefix is invalid: {exc}")
        else:
            expected_version_argv = (*prefix, "--version")
            if (
                model_dir is not None
                and device is not None
                and encoding is not None
                and prompt is not None
                and isinstance(tokens, int)
                and not isinstance(tokens, bool)
                and isinstance(top_k, int)
                and not isinstance(top_k, bool)
            ):
                expected_generation_argv = _generation_argv(
                    prefix,
                    model_dir=model_dir,
                    device=device,
                    quantization_encoding=encoding,
                    prompt=prompt,
                    max_new_tokens=tokens,
                    top_k=top_k,
                )

    commands = report.get("commands")
    version_record: dict[str, Any] | None = None
    generation_record: dict[str, Any] | None = None
    if not isinstance(commands, dict) or set(commands) != {
        "version",
        "generation",
    }:
        errors.append("commands must contain exactly version and generation")
    else:
        version_record = _validate_command_record(
            commands.get("version"),
            "commands.version",
            expected_version_argv,
            (
                min(declared_timeout, MAX_VERSION_SECONDS)
                if declared_timeout is not None
                else None
            ),
            errors,
        )
        generation_record = _validate_command_record(
            commands.get("generation"),
            "commands.generation",
            expected_generation_argv,
            declared_timeout,
            errors,
        )

    observed = report.get("observed")
    max_version: str | None = None
    generated_detected = False
    if not isinstance(observed, dict) or set(observed) != {
        "max_version",
        "generated_text_signal",
    }:
        errors.append(
            "observed must contain exactly max_version and generated_text_signal"
        )
    else:
        raw_version = observed.get("max_version")
        if raw_version is not None:
            max_version = _validation_string(
                raw_version, "observed.max_version", errors
            )
            if max_version is not None and len(max_version) > MAX_VERSION_CHARS:
                errors.append("observed.max_version exceeds the bounded limit")
        signal = observed.get("generated_text_signal")
        signal_keys = {
            "detected",
            "method",
            "text_excerpt",
            "text_excerpt_chars",
            "text_excerpt_sha256",
            "text_excerpt_truncated",
            "analysis_tail_truncated",
        }
        if not isinstance(signal, dict) or set(signal) != signal_keys:
            errors.append(
                "observed.generated_text_signal must exactly match the schema"
            )
        else:
            generated_detected = signal.get("detected") is True
            if not isinstance(signal.get("detected"), bool):
                errors.append(
                    "observed.generated_text_signal.detected must be boolean"
                )
            method = signal.get("method")
            text_excerpt = signal.get("text_excerpt")
            chars = signal.get("text_excerpt_chars")
            if generated_detected:
                if not isinstance(method, str) or not method:
                    errors.append(
                        "generated-text method must be present when detected"
                    )
                if not isinstance(text_excerpt, str) or not text_excerpt.strip():
                    errors.append(
                        "generated-text excerpt must be non-empty when detected"
                    )
            else:
                if method is not None:
                    errors.append(
                        "generated-text method must be null when not detected"
                    )
                if text_excerpt != "":
                    errors.append(
                        "generated-text excerpt must be empty when not detected"
                    )
            if not isinstance(text_excerpt, str):
                errors.append("generated-text excerpt must be a string")
            else:
                text_excerpt_valid_utf8 = True
                try:
                    text_excerpt.encode("utf-8")
                except UnicodeEncodeError:
                    text_excerpt_valid_utf8 = False
                    errors.append(
                        "generated-text excerpt must be valid UTF-8 text"
                    )
                if len(text_excerpt) > MAX_GENERATED_TEXT_CHARS:
                    errors.append("generated-text excerpt exceeds the bounded limit")
                if chars != len(text_excerpt):
                    errors.append("generated-text excerpt character count is inconsistent")
                if (
                    text_excerpt_valid_utf8
                    and signal.get("text_excerpt_sha256")
                    != _sha256_text(text_excerpt)
                ):
                    errors.append("generated-text excerpt digest is inconsistent")
            for field in ("text_excerpt_truncated", "analysis_tail_truncated"):
                if not isinstance(signal.get(field), bool):
                    errors.append(f"generated-text {field} must be boolean")

            if (
                prompt is not None
                and generation_record is not None
                and generation_record.get("stdout_excerpt_truncated") is False
                and isinstance(
                    generation_record.get("stdout_excerpt"), str
                )
            ):
                try:
                    generation_record["stdout_excerpt"].encode("utf-8")
                except UnicodeEncodeError:
                    pass
                else:
                    expected_signal = _generated_text_signal(
                        generation_record["stdout_excerpt"], prompt
                    )
                    if signal != expected_signal:
                        errors.append(
                            "generated-text signal is inconsistent with "
                            "complete generation stdout"
                        )

    if (
        version_record is not None
        and version_record.get("stdout_excerpt_truncated") is False
        and isinstance(version_record.get("stdout_excerpt"), str)
    ):
        complete_version_stdout = version_record["stdout_excerpt"]
        derived_version = complete_version_stdout.strip()
        expected_version = (
            derived_version
            if version_record.get("ok") is True
            and version_record.get("stdout_bytes", MAX_VERSION_BYTES + 1)
            <= MAX_VERSION_BYTES
            and derived_version
            and len(derived_version) <= MAX_VERSION_CHARS
            else None
        )
        if max_version != expected_version:
            errors.append(
                "observed.max_version is inconsistent with complete version stdout"
            )

    claims = report.get("claims")
    single_platform_claim = False
    if not isinstance(claims, dict) or set(claims) != set(CLAIM_KEYS):
        errors.append("claims fields must exactly match the schema")
    else:
        for key in CLAIM_KEYS:
            if not isinstance(claims.get(key), bool):
                errors.append(f"claims.{key} must be boolean")
        single_platform_claim = (
            claims.get("single_platform_bringup_passed") is True
        )
        for key in CLAIM_KEYS:
            if key != "single_platform_bringup_passed" and claims.get(key) is not False:
                errors.append(f"claims.{key} must be false")

    derived_pass = bool(
        version_record is not None
        and version_record.get("ok") is True
        and max_version
        and generation_record is not None
        and generation_record.get("ok") is True
        and generated_detected
    )
    derived_physical_pass = bool(derived_pass and physical_execution_eligible)
    if single_platform_claim != derived_physical_pass:
        errors.append(
            "claims.single_platform_bringup_passed is inconsistent with evidence"
        )
    if report.get("ok") is not derived_pass:
        errors.append("report.ok is inconsistent with bounded evidence")
    report_errors = report.get("errors")
    if (
        not isinstance(report_errors, list)
        or any(not isinstance(item, str) or not item for item in report_errors)
    ):
        errors.append("report.errors must be a string list")
    elif bool(report_errors) == derived_pass:
        errors.append("report.errors is inconsistent with report outcome")
    warnings = report.get("warnings")
    if (
        not isinstance(warnings, list)
        or not warnings
        or any(not isinstance(item, str) or not item for item in warnings)
    ):
        errors.append("report.warnings must be a non-empty string list")
    _validation_string(
        report.get("interpretation"), "interpretation", errors
    )

    errors = list(dict.fromkeys(errors))
    return {
        "ok": not errors,
        "errors": errors,
        "summary": {
            "model_id": inputs.get("model_id") if isinstance(inputs, dict) else None,
            "device": inputs.get("device") if isinstance(inputs, dict) else None,
            "quantization_encoding": (
                inputs.get("quantization_encoding")
                if isinstance(inputs, dict)
                else None
            ),
            "max_version": max_version,
            "runner_kind": runner_kind,
            "physical_execution_eligible": physical_execution_eligible,
            "device_binding_mode": (
                device_binding.get("mode")
                if isinstance(device_binding, dict)
                else None
            ),
            "physical_nvidia_smi_index": (
                device_binding.get("physical_nvidia_smi_index")
                if isinstance(device_binding, dict)
                else None
            ),
            "physical_nvidia_gpu_uuid": (
                device_binding.get("physical_nvidia_gpu_uuid")
                if isinstance(device_binding, dict)
                else None
            ),
            "generated_text_detected": generated_detected,
            "single_platform_bringup_passed": (
                single_platform_claim if not errors else False
            ),
        },
    }


__all__ = [
    "CLAIM_KEYS",
    "DEFAULT_DEVICE",
    "DEFAULT_MAX_ARGV_PREFIX",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_TOP_K",
    "EVIDENCE_SCOPE",
    "LIVE_RUNNER_KIND",
    "MAX_GENERATED_TEXT_CHARS",
    "MAX_OUTPUT_EXCERPT_CHARS",
    "RECORD_KIND",
    "SCHEMA_VERSION",
    "SMOKE_SENTINEL",
    "SMOKE_SENTINEL_PROMPT",
    "SYNTHETIC_RUNNER_KIND",
    "extract_exact_max_metrics_sentinel",
    "is_diagnostic_generated_text",
    "run_max_generation_smoke",
    "validate_max_generation_smoke_evidence",
]
