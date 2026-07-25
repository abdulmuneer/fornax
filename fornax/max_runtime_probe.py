"""Bounded, hardware-free inspection of the MAX runtime model registry.

The probe follows the documented ``max list --json`` contract:
https://docs.modular.com/max/cli/list/

A successful result establishes only that the invoked MAX command identifies
itself and advertises one exact architecture/encoding pair. It does not infer
device, operating-system, driver, model-load, correctness, performance, or
product-support compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .bounded_subprocess import run_bounded_subprocess


CommandRunner = Callable[[Sequence[str]], Any]

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "fornax_max_runtime_registry_probe"
EVIDENCE_SCOPE = "max_runtime_registry_identity_only"

COMMAND_TIMEOUT_SECONDS = 30
MAX_VERSION_OUTPUT_BYTES = 64 * 1024
MAX_LIST_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_STDERR_OUTPUT_BYTES = 256 * 1024
MAX_DIAGNOSTIC_CHARS = 4096
MAX_ARCHITECTURES = 4096
MAX_ENCODINGS_PER_ARCHITECTURE = 128
MAX_ARCHITECTURE_NAME_CHARS = 256
MAX_ENCODING_NAME_CHARS = 128

PHYSICAL_CLAIM_KEYS = (
    "distributed_runtime_passed",
    "formal_g2_passed",
    "formal_g3_passed",
    "hardware_detected",
    "model_artifact_verified",
    "production_supported",
    "single_platform_bringup_passed",
    "target_model_parity_passed",
)


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON value is forbidden: {value}")
    return parsed


def _default_command_runner(
    argv: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    stdout_limit = (
        MAX_LIST_OUTPUT_BYTES
        if tuple(argv[-2:]) == ("list", "--json")
        else MAX_VERSION_OUTPUT_BYTES
    )
    return run_bounded_subprocess(
        list(argv),
        timeout_s=COMMAND_TIMEOUT_SECONDS,
        stdout_limit_bytes=stdout_limit,
        stderr_limit_bytes=MAX_STDERR_OUTPUT_BYTES,
    )


def _output_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"{field} must be text or bytes")


def _returncode(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("command returncode must be an integer")
    return value


def _coerce_command_result(result: Any) -> tuple[int, str, str]:
    if isinstance(result, (str, bytes)):
        return 0, _output_text(result, "stdout"), ""
    if isinstance(result, Mapping):
        return (
            _returncode(result.get("returncode", result.get("code", 0))),
            _output_text(result.get("stdout", ""), "stdout"),
            _output_text(result.get("stderr", ""), "stderr"),
        )
    if isinstance(result, tuple):
        if len(result) == 2:
            return (
                _returncode(result[0]),
                _output_text(result[1], "stdout"),
                "",
            )
        if len(result) == 3:
            return (
                _returncode(result[0]),
                _output_text(result[1], "stdout"),
                _output_text(result[2], "stderr"),
            )
        raise TypeError("command-result tuples must contain two or three items")
    if hasattr(result, "returncode"):
        return (
            _returncode(getattr(result, "returncode")),
            _output_text(getattr(result, "stdout", ""), "stdout"),
            _output_text(getattr(result, "stderr", ""), "stderr"),
        )
    raise TypeError(
        "command runner must return text, a mapping, a "
        "(returncode, stdout[, stderr]) tuple, or a subprocess-style result"
    )


def _text_sha256(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _diagnostic_excerpt(value: str) -> tuple[str, bool]:
    stripped = value.strip().encode("utf-8", errors="replace").decode("utf-8")
    if len(stripped) <= MAX_DIAGNOSTIC_CHARS:
        return stripped, False
    return stripped[:MAX_DIAGNOSTIC_CHARS], True


def _run_command(
    runner: CommandRunner,
    argv: Sequence[str],
) -> tuple[dict[str, Any], str, str]:
    launch_error: str | None = None
    returncode: int | None = None
    stdout = ""
    stderr = ""
    try:
        result = runner(list(argv))
        returncode, stdout, stderr = _coerce_command_result(result)
    except Exception as exc:
        launch_error, launch_error_truncated = _diagnostic_excerpt(
            f"{type(exc).__name__}: {exc}"
        )
    else:
        launch_error_truncated = False

    stderr_excerpt, stderr_truncated = _diagnostic_excerpt(stderr)
    record = {
        "argv": list(argv),
        "shell": False,
        "returncode": returncode,
        "ok": launch_error is None and returncode == 0,
        "launch_error": launch_error,
        "launch_error_truncated": launch_error_truncated,
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stdout_text_sha256": _text_sha256(stdout),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stderr_text_sha256": _text_sha256(stderr),
        "stderr_excerpt": stderr_excerpt,
        "stderr_excerpt_truncated": stderr_truncated,
    }
    return record, stdout, stderr


def _bounded_exact_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")
    if len(value) > maximum:
        raise ValueError(f"{field} exceeds the {maximum}-character limit")
    return value


def _validated_command(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("max_command must be a non-empty sequence of argv strings")
    command = tuple(value)
    if not command:
        raise ValueError("max_command must be a non-empty sequence of argv strings")
    for index, item in enumerate(command):
        if not isinstance(item, str) or not item:
            raise ValueError(f"max_command[{index}] must be a non-empty string")
        if "\x00" in item:
            raise ValueError(f"max_command[{index}] must not contain NUL")
    return command


def _parse_registry(
    text: str,
) -> tuple[dict[str, tuple[str, ...]] | None, list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite_json,
            parse_float=_parse_finite_float,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        return None, [f"max list --json output is invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, ["max list --json top level must be an object"]
    architectures = payload.get("architectures")
    if not isinstance(architectures, dict):
        return None, ["max list --json architectures must be an object"]
    if len(architectures) > MAX_ARCHITECTURES:
        return None, [
            "max list --json architecture count exceeds the bounded limit "
            f"{MAX_ARCHITECTURES}"
        ]

    normalized: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_entry in architectures.items():
        try:
            name = _bounded_exact_text(
                raw_name,
                "max list --json architecture name",
                MAX_ARCHITECTURE_NAME_CHARS,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(raw_entry, dict):
            errors.append(f"architecture {name!r} must map to an object")
            continue
        raw_encodings = raw_entry.get("supported_encodings")
        if not isinstance(raw_encodings, list):
            errors.append(
                f"architecture {name!r}.supported_encodings must be a list"
            )
            continue
        if len(raw_encodings) > MAX_ENCODINGS_PER_ARCHITECTURE:
            errors.append(
                f"architecture {name!r}.supported_encodings exceeds the bounded "
                f"limit {MAX_ENCODINGS_PER_ARCHITECTURE}"
            )
            continue
        encodings: list[str] = []
        entry_invalid = False
        for index, raw_encoding in enumerate(raw_encodings):
            try:
                encoding = _bounded_exact_text(
                    raw_encoding,
                    f"architecture {name!r}.supported_encodings[{index}]",
                    MAX_ENCODING_NAME_CHARS,
                )
            except ValueError as exc:
                errors.append(str(exc))
                entry_invalid = True
                continue
            encodings.append(encoding)
        if len(encodings) != len(set(encodings)):
            errors.append(
                f"architecture {name!r}.supported_encodings must not contain duplicates"
            )
            entry_invalid = True
        if not entry_invalid:
            normalized[name] = tuple(encodings)
    return normalized, errors


def probe_max_runtime_support(
    expected_architecture: str,
    expected_encoding: str,
    *,
    max_command: Sequence[str] = ("max",),
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Probe one exact architecture/encoding pair in the invoked MAX registry.

    ``command_runner`` receives direct argv sequences and may return text, a
    mapping, a ``(returncode, stdout[, stderr])`` tuple, or a
    ``subprocess.CompletedProcess``-style object.
    """

    architecture = _bounded_exact_text(
        expected_architecture,
        "expected_architecture",
        MAX_ARCHITECTURE_NAME_CHARS,
    )
    encoding = _bounded_exact_text(
        expected_encoding,
        "expected_encoding",
        MAX_ENCODING_NAME_CHARS,
    )
    command = _validated_command(max_command)
    runner = (
        _default_command_runner if command_runner is None else command_runner
    )

    version_argv = (*command, "--version")
    list_argv = (*command, "list", "--json")
    version_record, version_stdout, version_stderr = _run_command(
        runner, version_argv
    )
    list_record, list_stdout, list_stderr = _run_command(runner, list_argv)

    errors: list[str] = []
    if not version_record["ok"]:
        detail = (
            version_record["launch_error"]
            or version_record["stderr_excerpt"]
            or f"exit status {version_record['returncode']}"
        )
        errors.append(f"max --version failed: {detail}")
    version_bytes = len(version_stdout.encode("utf-8"))
    max_version = version_stdout.strip() if version_record["ok"] else None
    if version_bytes > MAX_VERSION_OUTPUT_BYTES:
        errors.append(
            f"max --version output exceeds the bounded limit "
            f"{MAX_VERSION_OUTPUT_BYTES} bytes"
        )
        max_version = None
    elif version_record["ok"] and not max_version:
        errors.append("max --version returned no version text")

    registry: dict[str, tuple[str, ...]] | None = None
    registry_errors: list[str] = []
    if not list_record["ok"]:
        detail = (
            list_record["launch_error"]
            or list_record["stderr_excerpt"]
            or f"exit status {list_record['returncode']}"
        )
        errors.append(f"max list --json failed: {detail}")
    elif len(list_stdout.encode("utf-8")) > MAX_LIST_OUTPUT_BYTES:
        errors.append(
            "max list --json output exceeds the bounded limit "
            f"{MAX_LIST_OUTPUT_BYTES} bytes"
        )
    else:
        registry, registry_errors = _parse_registry(list_stdout)
        errors.extend(registry_errors)

    architecture_present = registry is not None and architecture in registry
    supported_encodings = (
        list(registry[architecture]) if architecture_present and registry else []
    )
    encoding_present = architecture_present and encoding in supported_encodings
    if registry is not None and not registry_errors:
        if not architecture_present:
            errors.append(
                f"MAX registry does not advertise exact architecture {architecture!r}"
            )
        elif not encoding_present:
            errors.append(
                f"MAX architecture {architecture!r} does not advertise exact "
                f"encoding {encoding!r}"
            )

    errors = list(dict.fromkeys(errors))
    ok = not errors
    physical_claims = {key: False for key in PHYSICAL_CLAIM_KEYS}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "record_kind": REPORT_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "collection_provenance": {
            "mode": (
                "live_subprocess"
                if command_runner is None
                else "injected_fixture_runner"
            ),
            "physical_observation_eligible": command_runner is None,
            "authenticated": False,
        },
        "ok": ok,
        "errors": errors,
        "warnings": [
            "A registry match is C1 runtime-advertisement evidence only; it does "
            "not establish device, OS, driver, model-load, numerical-correctness, "
            "performance, distributed-runtime, or product-support compatibility."
        ],
        "expected": {
            "architecture": architecture,
            "encoding": encoding,
        },
        "observed": {
            "max_version": max_version,
            "architecture_count": len(registry) if registry is not None else None,
            "architecture_present": architecture_present,
            "supported_encodings": supported_encodings,
            "encoding_present": encoding_present,
        },
        "commands": {
            "version": version_record,
            "list_json": list_record,
        },
        "qualification": {
            "maturity": "C1_contracted",
            "authority": "exploratory",
            "registry_match_passed": ok,
            "runtime_compatibility_passed": False,
            "physical_execution_status": "not_run",
            "production_supported": False,
        },
        "physical_claims": physical_claims,
        "interpretation": (
            "Exact MAX registry architecture/encoding identity only; no hardware "
            "or support inference is permitted."
        ),
    }


__all__ = [
    "COMMAND_TIMEOUT_SECONDS",
    "EVIDENCE_SCOPE",
    "MAX_LIST_OUTPUT_BYTES",
    "MAX_VERSION_OUTPUT_BYTES",
    "PHYSICAL_CLAIM_KEYS",
    "REPORT_KIND",
    "REPORT_SCHEMA_VERSION",
    "probe_max_runtime_support",
]
