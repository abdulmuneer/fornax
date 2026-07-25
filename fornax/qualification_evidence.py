"""Strict evidence loading and prerequisite binding for qualification launches.

These helpers keep a bounded single-device launch from being relabelled with
unrelated catalog identities.  Every prerequisite is read once through a
no-follow file descriptor, validated against the current catalog, and bound by
its raw byte digest.  The records remain unauthenticated local evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .hardware_identity import NVIDIA_GPU_UUID_RE
from .max_runtime_probe import (
    EVIDENCE_SCOPE as MAX_RUNTIME_EVIDENCE_SCOPE,
    PHYSICAL_CLAIM_KEYS as MAX_RUNTIME_PHYSICAL_CLAIM_KEYS,
    REPORT_KIND as MAX_RUNTIME_REPORT_KIND,
)
from .model_artifacts import validate_model_artifact_report


MAX_EVIDENCE_FILE_BYTES = 32 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 512 * 1024 * 1024
HOST_REPORT_KIND = "fornax_qualification_host_identity"
NVIDIA_SINGLE_PREFLIGHT_KIND = "fornax_nvidia_single_preflight_binding"
APPLE_SINGLE_PREFLIGHT_KIND = "fornax_apple_single_preflight_binding"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEVICE_RE = re.compile(r"^gpu:(0|[1-9][0-9]*)$")


class QualificationEvidenceError(ValueError):
    """Raised when qualification evidence cannot be read safely."""


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise QualificationEvidenceError(
        f"non-finite JSON number is forbidden: {value}"
    )


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise QualificationEvidenceError(
            f"non-finite JSON number is forbidden: {value}"
        )
    return parsed


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _read_bounded_descriptor(
    descriptor: int,
    *,
    label: str,
    maximum: int,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise QualificationEvidenceError(f"{label} must be a regular file")
    if before.st_nlink != 1:
        raise QualificationEvidenceError(
            f"{label} must have exactly one hard link; observed {before.st_nlink}"
        )
    if before.st_size > maximum:
        raise QualificationEvidenceError(
            f"{label} exceeds the bounded limit {maximum} bytes"
        )
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    if len(value) > maximum:
        raise QualificationEvidenceError(
            f"{label} exceeds the bounded limit {maximum} bytes"
        )
    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise QualificationEvidenceError(f"{label} changed while being read")
    if len(value) != before.st_size:
        raise QualificationEvidenceError(f"{label} changed size while being read")
    return value, before


def load_json_evidence(
    path: str | Path,
    *,
    label: str,
    maximum: int = MAX_EVIDENCE_FILE_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one bounded, regular, non-symlink JSON evidence file exactly once."""

    evidence_path = Path(path).expanduser()
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise QualificationEvidenceError(
            "safe evidence loading requires O_NOFOLLOW and O_NONBLOCK"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
    )
    try:
        descriptor = os.open(evidence_path, flags)
    except OSError as exc:
        raise QualificationEvidenceError(
            f"{label} cannot be opened without following a symlink: {exc}"
        ) from exc
    try:
        value, metadata = _read_bounded_descriptor(
            descriptor,
            label=label,
            maximum=maximum,
        )
    finally:
        os.close(descriptor)
    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except QualificationEvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise QualificationEvidenceError(
            f"{label} is not valid bounded UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise QualificationEvidenceError(f"{label} top level must be an object")
    return parsed, {
        "path": str(evidence_path.resolve()),
        "size_bytes": len(value),
        "sha256": _sha256_bytes(value),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "authenticated": False,
    }


def executable_identity(command: str) -> dict[str, Any]:
    """Resolve and hash the executable selected for one argv[0] value."""

    if not isinstance(command, str) or not command or "\x00" in command:
        raise QualificationEvidenceError(
            "executable command must be a non-empty NUL-free string"
        )
    candidate = (
        shutil.which(command)
        if os.sep not in command and (os.altsep is None or os.altsep not in command)
        else command
    )
    if candidate is None:
        raise QualificationEvidenceError(f"executable is not available: {command}")
    try:
        resolved = Path(candidate).expanduser().resolve(strict=True)
    except OSError as exc:
        raise QualificationEvidenceError(
            f"executable cannot be resolved: {command}: {exc}"
        ) from exc
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise QualificationEvidenceError(
            "safe executable inspection requires O_NOFOLLOW and O_NONBLOCK"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise QualificationEvidenceError(
            f"resolved executable cannot be opened safely: {resolved}: {exc}"
        ) from exc
    try:
        value_hash = hashlib.sha256()
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationEvidenceError(
                f"resolved executable is not a regular file: {resolved}"
            )
        if before.st_size > MAX_EXECUTABLE_BYTES:
            raise QualificationEvidenceError(
                f"resolved executable exceeds {MAX_EXECUTABLE_BYTES} bytes: {resolved}"
            )
        remaining = MAX_EXECUTABLE_BYTES + 1
        observed_bytes = 0
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            observed_bytes += len(chunk)
            remaining -= len(chunk)
            value_hash.update(chunk)
        after = os.fstat(descriptor)
        if observed_bytes > MAX_EXECUTABLE_BYTES:
            raise QualificationEvidenceError(
                f"resolved executable exceeds {MAX_EXECUTABLE_BYTES} bytes: {resolved}"
            )
        if (
            observed_bytes != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise QualificationEvidenceError(
                f"resolved executable changed while being hashed: {resolved}"
            )
    finally:
        os.close(descriptor)
    return {
        "requested_argv0": command,
        "resolved_path": str(resolved),
        "size_bytes": observed_bytes,
        "sha256": "sha256:" + value_hash.hexdigest(),
        "authenticated": False,
    }


def _exact_live_provenance(
    value: Any,
    *,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} collection_provenance must be an object")
        return
    if value.get("mode") != "live_subprocess":
        errors.append(f"{label} must come from the live subprocess collector")
    if value.get("physical_observation_eligible") is not True:
        errors.append(
            f"{label} must be marked physical_observation_eligible=true"
        )
    if value.get("authenticated") is not False:
        errors.append(f"{label} must explicitly remain unauthenticated")


def _device_index(device: str) -> int:
    if not isinstance(device, str) or _DEVICE_RE.fullmatch(device) is None:
        raise QualificationEvidenceError(
            "device must be one exact MAX GPU device such as gpu:0"
        )
    return int(device.split(":", 1)[1])


def _matching_gpu_uuid(
    observed: Any,
    *,
    device_index: int,
    label: str,
    missing_message: str,
    errors: list[str],
) -> str | None:
    """Resolve one profile-matching nvidia-smi index to its physical UUID."""

    if not isinstance(observed, Mapping):
        errors.append(f"{label} observed GPU match must be an object")
        return None
    indices = observed.get("matching_gpu_indices")
    normalized_indices: set[int] = set()
    if isinstance(indices, list):
        for value in indices:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                normalized_indices.add(value)
            elif isinstance(value, str) and value.isdecimal():
                normalized_indices.add(int(value))
    if device_index not in normalized_indices:
        errors.append(missing_message)

    gpus = observed.get("gpus")
    if not isinstance(gpus, list):
        errors.append(f"{label} observed.gpus must be a list")
        return None
    candidates: list[Mapping[str, Any]] = []
    for gpu in gpus:
        if not isinstance(gpu, Mapping):
            continue
        raw_index = gpu.get("index")
        normalized_index: int | None = None
        if (
            isinstance(raw_index, int)
            and not isinstance(raw_index, bool)
            and raw_index >= 0
        ):
            normalized_index = raw_index
        elif isinstance(raw_index, str) and raw_index.isdecimal():
            normalized_index = int(raw_index)
        if normalized_index == device_index:
            candidates.append(gpu)
    if len(candidates) != 1:
        errors.append(
            f"{label} must contain exactly one GPU row for nvidia-smi "
            f"index {device_index}; observed {len(candidates)}"
        )
        return None
    candidate = candidates[0]
    if (
        candidate.get("name_matches") is not True
        or candidate.get("memory_matches") is not True
    ):
        errors.append(
            f"{label} GPU index {device_index} is not an exact profile match"
        )
        return None
    gpu_uuid = candidate.get("uuid")
    if (
        not isinstance(gpu_uuid, str)
        or NVIDIA_GPU_UUID_RE.fullmatch(gpu_uuid) is None
    ):
        errors.append(
            f"{label} GPU index {device_index} is missing a valid physical UUID"
        )
        return None
    matching_uuids = observed.get("matching_gpu_uuids")
    if (
        not isinstance(matching_uuids, list)
        or gpu_uuid not in matching_uuids
    ):
        errors.append(
            f"{label} matching_gpu_uuids does not bind GPU index "
            f"{device_index} to {gpu_uuid}"
        )
        return None
    return gpu_uuid


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def validate_apple_single_preflights(
    *,
    artifact_report: dict[str, Any],
    fresh_artifact_report: dict[str, Any],
    host_report: dict[str, Any],
    runtime_report: dict[str, Any],
    fresh_host_identity: Mapping[str, Any],
    fresh_host_match: Mapping[str, Any],
    fresh_runtime_report: Mapping[str, Any],
    artifact_file: Mapping[str, Any],
    host_file: Mapping[str, Any],
    runtime_file: Mapping[str, Any],
    expected_catalog_sha256: str,
    expected_model_id: str,
    expected_model_profile_sha256: str,
    expected_repository: str,
    expected_revision: str,
    expected_model_dir: str | Path,
    expected_platform_id: str,
    expected_platform_profile_sha256: str,
    expected_platform_vendor: str,
    expected_platform_runtime_verification_status: str,
    expected_apple_chip: str,
    expected_apple_memory_bytes: int,
    expected_architecture: str,
    expected_encoding: str,
    expected_max_command: Sequence[str],
    expected_max_executable: Mapping[str, Any],
    expected_collector_executables: Mapping[str, Mapping[str, Any]],
    minimum_units: int,
) -> dict[str, Any]:
    """Validate and bind every prerequisite for a one-SoC Apple launch."""

    errors: list[str] = []
    warnings = [
        "The prerequisite files and executable hashes are unauthenticated local "
        "evidence; operator custody is still required."
    ]
    if minimum_units != 1:
        errors.append(
            "single-device generation is forbidden because the capacity-only "
            f"minimum is {minimum_units}, not 1"
        )
    if expected_platform_vendor != "apple":
        errors.append("Apple single-device preflight requires vendor=apple")
    if not isinstance(expected_apple_chip, str) or not expected_apple_chip:
        errors.append("selected Apple profile must provide one exact chip")
    if (
        isinstance(expected_apple_memory_bytes, bool)
        or not isinstance(expected_apple_memory_bytes, int)
        or expected_apple_memory_bytes <= 0
    ):
        errors.append(
            "selected Apple profile must provide exact positive memory bytes"
        )
    model_dir = str(Path(expected_model_dir).expanduser().resolve())

    artifact_validations: dict[str, dict[str, Any]] = {}
    for label, report in (
        ("recorded", artifact_report),
        ("fresh", fresh_artifact_report),
    ):
        validation = validate_model_artifact_report(
            report,
            expected_catalog_sha256=expected_catalog_sha256,
            expected_profile_sha256=expected_model_profile_sha256,
            require_complete_hash_coverage=True,
        )
        artifact_validations[label] = validation
        if not validation["ok"]:
            errors.extend(
                f"{label} artifact report: {message}"
                for message in validation["errors"]
            )
        if report.get("ok") is not True:
            errors.append(f"{label} artifact report must have ok=true")
        reported_model_dir = report.get("model_dir")
        try:
            normalized_reported_model_dir = (
                str(Path(reported_model_dir).expanduser().resolve())
                if isinstance(reported_model_dir, str) and reported_model_dir
                else None
            )
        except OSError:
            normalized_reported_model_dir = None
        if normalized_reported_model_dir != model_dir:
            errors.append(
                f"{label} artifact report model_dir does not match launch model_dir"
            )
        profile_identity = report.get("profile_identity")
        if not isinstance(profile_identity, Mapping):
            errors.append(f"{label} artifact profile_identity must be an object")
        else:
            expected_identity = {
                "model_id": expected_model_id,
                "repository": expected_repository,
                "expected_revision": expected_revision,
            }
            for field, expected in expected_identity.items():
                if profile_identity.get(field) != expected:
                    errors.append(
                        f"{label} artifact profile_identity.{field} does not "
                        "match the selected catalog profile"
                    )
        revision = report.get("revision")
        if not isinstance(revision, Mapping):
            errors.append(f"{label} artifact revision must be an object")
        elif (
            revision.get("expected") != expected_revision
            or revision.get("value") != expected_revision
            or revision.get("resolved") is not True
        ):
            errors.append(
                f"{label} artifact revision does not resolve to the selected pin"
            )

    for field in (
        "artifact_manifest_sha256",
        "files",
        "hash_coverage",
        "revision",
        "profile_identity",
    ):
        if artifact_report.get(field) != fresh_artifact_report.get(field):
            errors.append(
                f"recorded artifact report {field} does not match fresh inspection"
            )

    host_keys = {
        "schema_version",
        "record_kind",
        "platform_id",
        "platform_profile_sha256",
        "catalog_sha256",
        "identity",
        "match",
        "collector_executables",
        "evidence_scope",
        "qualification",
    }
    if set(host_report) != host_keys:
        errors.append("host report fields must exactly match the live report schema")
    if host_report.get("schema_version") != 1:
        errors.append("host report schema_version must be 1")
    if host_report.get("record_kind") != HOST_REPORT_KIND:
        errors.append(f"host report record_kind must be {HOST_REPORT_KIND}")
    if host_report.get("evidence_scope") != "observed_host_identity_only":
        errors.append("host report evidence_scope is invalid")
    if host_report.get("catalog_sha256") != expected_catalog_sha256:
        errors.append("host report catalog_sha256 does not match current catalog")
    if host_report.get("platform_id") != expected_platform_id:
        errors.append("host report platform_id does not match selected platform")
    if (
        host_report.get("platform_profile_sha256")
        != expected_platform_profile_sha256
    ):
        errors.append("host report platform profile hash does not match")
    identity = host_report.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("host report identity must be an object")
    else:
        _exact_live_provenance(
            identity.get("collection_provenance"),
            label="host report",
            errors=errors,
        )
    match = host_report.get("match")
    if not isinstance(match, Mapping):
        errors.append("host report match must be an object")
        recorded_observed = None
    else:
        if match.get("ok") is not True:
            errors.append("host report exact platform match must have ok=true")
        if match.get("errors") != []:
            errors.append("host report exact platform match must have no errors")
        recorded_observed = match.get("observed")
        if (
            not isinstance(recorded_observed, Mapping)
            or recorded_observed.get("vendor") != "apple"
        ):
            errors.append(
                "host report observed identity must be the selected Apple profile"
            )
        elif (
            recorded_observed.get("chip") != expected_apple_chip
            or recorded_observed.get("memory_bytes")
            != expected_apple_memory_bytes
        ):
            errors.append(
                "host report must match the exact selected Apple chip and "
                "configured memory bytes"
            )
    collector_executables = host_report.get("collector_executables")
    expected_collectors = {
        name: dict(value)
        for name, value in expected_collector_executables.items()
    }
    if set(expected_collectors) != {"system_profiler", "sysctl", "sw_vers"}:
        errors.append(
            "current Apple collector executable identities must contain exactly "
            "system_profiler, sysctl, and sw_vers"
        )
    if not isinstance(collector_executables, Mapping):
        errors.append("host report collector_executables must be an object")
    elif dict(collector_executables) != expected_collectors:
        errors.append(
            "host report Apple collector executable identities do not match "
            "the current launch-host executables"
        )
    expected_host_qualification = {
        "maturity": "C1_contracted",
        "identity_match_passed": True,
        "runtime_compatibility_passed": False,
        "model_bringup_passed": False,
        "target_model_parity_passed": False,
        "formal_g2_passed": False,
        "production_supported": False,
    }
    if host_report.get("qualification") != expected_host_qualification:
        errors.append("host report qualification fields are not fail-closed C1")
    _exact_live_provenance(
        fresh_host_identity.get("collection_provenance"),
        label="fresh host observation",
        errors=errors,
    )
    if (
        fresh_host_match.get("ok") is not True
        or fresh_host_match.get("errors") != []
    ):
        errors.append("fresh host observation does not match the selected platform")
    fresh_observed = fresh_host_match.get("observed")
    if (
        not isinstance(fresh_observed, Mapping)
        or fresh_observed.get("vendor") != "apple"
    ):
        errors.append(
            "fresh host observation must identify the selected Apple profile"
        )
    elif (
        fresh_observed.get("chip") != expected_apple_chip
        or fresh_observed.get("memory_bytes") != expected_apple_memory_bytes
    ):
        errors.append(
            "fresh host observation must match the exact selected Apple chip "
            "and configured memory bytes"
        )
    if recorded_observed != fresh_observed:
        errors.append(
            "recorded host identity does not match the fresh launch-host observation"
        )
    if isinstance(identity, Mapping) and dict(identity) != dict(fresh_host_identity):
        errors.append(
            "recorded host report identity does not match the full fresh "
            "launch-host observation"
        )

    runtime_keys = {
        "schema_version",
        "record_kind",
        "evidence_scope",
        "collection_provenance",
        "ok",
        "errors",
        "warnings",
        "expected",
        "observed",
        "commands",
        "qualification",
        "physical_claims",
        "interpretation",
        "catalog_sha256",
        "model",
        "platform",
        "max_executable",
    }
    if set(runtime_report) != runtime_keys:
        errors.append(
            "runtime report fields must exactly match the live report schema"
        )
    if runtime_report.get("schema_version") != 1:
        errors.append("runtime report schema_version must be 1")
    if runtime_report.get("record_kind") != MAX_RUNTIME_REPORT_KIND:
        errors.append(
            f"runtime report record_kind must be {MAX_RUNTIME_REPORT_KIND}"
        )
    if runtime_report.get("evidence_scope") != MAX_RUNTIME_EVIDENCE_SCOPE:
        errors.append("runtime report evidence_scope is invalid")
    if runtime_report.get("catalog_sha256") != expected_catalog_sha256:
        errors.append("runtime report catalog_sha256 does not match current catalog")
    if runtime_report.get("model") != {
        "model_id": expected_model_id,
        "profile_sha256": expected_model_profile_sha256,
        "repository": expected_repository,
    }:
        errors.append("runtime report model lineage does not match selected model")
    if runtime_report.get("platform") != {
        "platform_id": expected_platform_id,
        "profile_sha256": expected_platform_profile_sha256,
        "vendor": expected_platform_vendor,
        "runtime_verification_status": (
            expected_platform_runtime_verification_status
        ),
    }:
        errors.append(
            "runtime report platform lineage does not match selected platform"
        )
    _exact_live_provenance(
        runtime_report.get("collection_provenance"),
        label="runtime report",
        errors=errors,
    )
    if runtime_report.get("ok") is not True or runtime_report.get("errors") != []:
        errors.append("runtime report exact registry probe must pass without errors")
    if runtime_report.get("expected") != {
        "architecture": expected_architecture,
        "encoding": expected_encoding,
    }:
        errors.append("runtime report expected architecture/encoding do not match")
    runtime_observed = runtime_report.get("observed")
    if not isinstance(runtime_observed, Mapping) or (
        runtime_observed.get("architecture_present") is not True
        or runtime_observed.get("encoding_present") is not True
    ):
        errors.append("runtime report did not observe the exact registry pair")
    physical_claims = runtime_report.get("physical_claims")
    if (
        not isinstance(physical_claims, Mapping)
        or set(physical_claims) != set(MAX_RUNTIME_PHYSICAL_CLAIM_KEYS)
        or any(value is not False for value in physical_claims.values())
    ):
        errors.append("runtime report physical claims must all remain false")
    expected_runtime_qualification = {
        "maturity": "C1_contracted",
        "authority": "exploratory",
        "registry_match_passed": True,
        "runtime_compatibility_passed": False,
        "physical_execution_status": "not_run",
        "production_supported": False,
    }
    if runtime_report.get("qualification") != expected_runtime_qualification:
        errors.append("runtime report qualification fields are not fail-closed C1")

    max_command = tuple(expected_max_command)
    commands = runtime_report.get("commands")
    if not isinstance(commands, Mapping):
        errors.append("runtime report commands must be an object")
    else:
        version = commands.get("version")
        listing = commands.get("list_json")
        if not isinstance(version, Mapping) or version.get("argv") != [
            *max_command,
            "--version",
        ]:
            errors.append("runtime report MAX version argv does not match launch")
        if not isinstance(listing, Mapping) or listing.get("argv") != [
            *max_command,
            "list",
            "--json",
        ]:
            errors.append("runtime report MAX registry argv does not match launch")
    if runtime_report.get("max_executable") != dict(expected_max_executable):
        errors.append(
            "runtime report MAX executable identity does not match the launch executable"
        )
    if set(fresh_runtime_report) != runtime_keys:
        errors.append(
            "fresh runtime report fields must exactly match the live report schema"
        )
    if fresh_runtime_report.get("schema_version") != 1:
        errors.append("fresh runtime report schema_version must be 1")
    if fresh_runtime_report.get("record_kind") != MAX_RUNTIME_REPORT_KIND:
        errors.append(
            f"fresh runtime report record_kind must be {MAX_RUNTIME_REPORT_KIND}"
        )
    if fresh_runtime_report.get("evidence_scope") != MAX_RUNTIME_EVIDENCE_SCOPE:
        errors.append("fresh runtime report evidence_scope is invalid")
    _exact_live_provenance(
        fresh_runtime_report.get("collection_provenance"),
        label="fresh runtime report",
        errors=errors,
    )
    if (
        fresh_runtime_report.get("ok") is not True
        or fresh_runtime_report.get("errors") != []
    ):
        errors.append("fresh MAX runtime registry probe did not pass")
    for field in (
        "schema_version",
        "record_kind",
        "evidence_scope",
        "catalog_sha256",
        "model",
        "platform",
        "expected",
        "observed",
        "commands",
        "qualification",
        "physical_claims",
        "max_executable",
    ):
        if fresh_runtime_report.get(field) != runtime_report.get(field):
            errors.append(
                f"recorded runtime report {field} does not match fresh observation"
            )

    binding = {
        "schema_version": 1,
        "record_kind": APPLE_SINGLE_PREFLIGHT_KIND,
        "ok": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": warnings,
        "catalog_sha256": expected_catalog_sha256,
        "model_id": expected_model_id,
        "model_profile_sha256": expected_model_profile_sha256,
        "platform_id": expected_platform_id,
        "platform_profile_sha256": expected_platform_profile_sha256,
        "expected_apple_identity": {
            "chip": expected_apple_chip,
            "memory_bytes": expected_apple_memory_bytes,
        },
        "model_dir": model_dir,
        "max_command": list(max_command),
        "max_executable": dict(expected_max_executable),
        "collector_executables": expected_collectors,
        "evidence_files": {
            "model_artifacts": dict(artifact_file),
            "host_identity": dict(host_file),
            "max_runtime": dict(runtime_file),
        },
        "fresh_artifact_manifest_sha256": fresh_artifact_report.get(
            "artifact_manifest_sha256"
        ),
        "fresh_host_observation": (
            dict(fresh_observed)
            if isinstance(fresh_observed, Mapping)
            else None
        ),
        "fresh_host_observation_sha256": _canonical_mapping_sha256(
            {
                "identity": fresh_host_identity,
                "match": fresh_host_match,
            }
        ),
        "fresh_runtime_report_sha256": _canonical_mapping_sha256(
            fresh_runtime_report
        ),
        "artifact_validations": artifact_validations,
        "authenticated": False,
    }
    canonical = json.dumps(
        binding,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    binding["binding_sha256"] = _sha256_bytes(canonical)
    return binding


def validate_nvidia_single_preflights(
    *,
    artifact_report: dict[str, Any],
    fresh_artifact_report: dict[str, Any],
    host_report: dict[str, Any],
    runtime_report: dict[str, Any],
    fresh_host_identity: Mapping[str, Any],
    fresh_host_match: Mapping[str, Any],
    fresh_runtime_report: Mapping[str, Any],
    artifact_file: Mapping[str, Any],
    host_file: Mapping[str, Any],
    runtime_file: Mapping[str, Any],
    expected_catalog_sha256: str,
    expected_model_id: str,
    expected_model_profile_sha256: str,
    expected_repository: str,
    expected_revision: str,
    expected_model_dir: str | Path,
    expected_platform_id: str,
    expected_platform_profile_sha256: str,
    expected_architecture: str,
    expected_encoding: str,
    expected_device: str,
    expected_max_command: Sequence[str],
    expected_max_executable: Mapping[str, Any],
    expected_nvidia_smi_executable: Mapping[str, Any],
    minimum_units: int,
) -> dict[str, Any]:
    """Validate and bind every prerequisite for a one-GPU NVIDIA launch."""

    errors: list[str] = []
    warnings = [
        "The prerequisite files and executable hashes are unauthenticated local "
        "evidence; operator custody is still required."
    ]
    if minimum_units != 1:
        errors.append(
            "single-device generation is forbidden because the capacity-only "
            f"minimum is {minimum_units}, not 1"
        )
    try:
        device_index = _device_index(expected_device)
    except QualificationEvidenceError as exc:
        errors.append(str(exc))
        device_index = -1
    model_dir = str(Path(expected_model_dir).expanduser().resolve())

    artifact_validations: dict[str, dict[str, Any]] = {}
    for label, report in (
        ("recorded", artifact_report),
        ("fresh", fresh_artifact_report),
    ):
        validation = validate_model_artifact_report(
            report,
            expected_catalog_sha256=expected_catalog_sha256,
            expected_profile_sha256=expected_model_profile_sha256,
            require_complete_hash_coverage=True,
        )
        artifact_validations[label] = validation
        if not validation["ok"]:
            errors.extend(
                f"{label} artifact report: {message}"
                for message in validation["errors"]
            )
        if report.get("ok") is not True:
            errors.append(f"{label} artifact report must have ok=true")
        reported_model_dir = report.get("model_dir")
        try:
            normalized_reported_model_dir = (
                str(Path(reported_model_dir).expanduser().resolve())
                if isinstance(reported_model_dir, str) and reported_model_dir
                else None
            )
        except OSError:
            normalized_reported_model_dir = None
        if normalized_reported_model_dir != model_dir:
            errors.append(
                f"{label} artifact report model_dir does not match launch model_dir"
            )
        profile_identity = report.get("profile_identity")
        if not isinstance(profile_identity, Mapping):
            errors.append(f"{label} artifact profile_identity must be an object")
        else:
            expected_identity = {
                "model_id": expected_model_id,
                "repository": expected_repository,
                "expected_revision": expected_revision,
            }
            for field, expected in expected_identity.items():
                if profile_identity.get(field) != expected:
                    errors.append(
                        f"{label} artifact profile_identity.{field} does not "
                        "match the selected catalog profile"
                    )
        revision = report.get("revision")
        if not isinstance(revision, Mapping):
            errors.append(f"{label} artifact revision must be an object")
        elif (
            revision.get("expected") != expected_revision
            or revision.get("value") != expected_revision
            or revision.get("resolved") is not True
        ):
            errors.append(
                f"{label} artifact revision does not resolve to the selected pin"
            )

    comparison_fields = (
        "artifact_manifest_sha256",
        "files",
        "hash_coverage",
        "revision",
        "profile_identity",
    )
    for field in comparison_fields:
        if artifact_report.get(field) != fresh_artifact_report.get(field):
            errors.append(
                f"recorded artifact report {field} does not match fresh inspection"
            )

    host_keys = {
        "schema_version",
        "record_kind",
        "platform_id",
        "platform_profile_sha256",
        "catalog_sha256",
        "identity",
        "match",
        "collector_executables",
        "evidence_scope",
        "qualification",
    }
    if set(host_report) != host_keys:
        errors.append("host report fields must exactly match the live report schema")
    if host_report.get("schema_version") != 1:
        errors.append("host report schema_version must be 1")
    if host_report.get("record_kind") != HOST_REPORT_KIND:
        errors.append(f"host report record_kind must be {HOST_REPORT_KIND}")
    if host_report.get("evidence_scope") != "observed_host_identity_only":
        errors.append("host report evidence_scope is invalid")
    if host_report.get("catalog_sha256") != expected_catalog_sha256:
        errors.append("host report catalog_sha256 does not match current catalog")
    if host_report.get("platform_id") != expected_platform_id:
        errors.append("host report platform_id does not match selected platform")
    if (
        host_report.get("platform_profile_sha256")
        != expected_platform_profile_sha256
    ):
        errors.append("host report platform profile hash does not match")
    identity = host_report.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("host report identity must be an object")
    else:
        _exact_live_provenance(
            identity.get("collection_provenance"),
            label="host report",
            errors=errors,
        )
    match = host_report.get("match")
    if not isinstance(match, Mapping):
        errors.append("host report match must be an object")
    else:
        if match.get("ok") is not True:
            errors.append("host report exact platform match must have ok=true")
        if match.get("errors") != []:
            errors.append("host report exact platform match must have no errors")
        recorded_observed = match.get("observed")
        recorded_gpu_uuid = (
            _matching_gpu_uuid(
                recorded_observed,
                device_index=device_index,
                label="host report",
                missing_message=(
                    "host report does not match the requested device "
                    f"gpu:{device_index}"
                ),
                errors=errors,
            )
            if device_index >= 0
            else None
        )
    if not isinstance(match, Mapping):
        recorded_observed = None
        recorded_gpu_uuid = None
    collector_executables = host_report.get("collector_executables")
    if not isinstance(collector_executables, Mapping):
        errors.append("host report collector_executables must be an object")
    elif collector_executables.get("nvidia-smi") != dict(
        expected_nvidia_smi_executable
    ):
        errors.append(
            "host report nvidia-smi executable identity does not match the "
            "current launch-host executable"
        )
    expected_host_qualification = {
        "maturity": "C1_contracted",
        "identity_match_passed": True,
        "runtime_compatibility_passed": False,
        "model_bringup_passed": False,
        "target_model_parity_passed": False,
        "formal_g2_passed": False,
        "production_supported": False,
    }
    if host_report.get("qualification") != expected_host_qualification:
        errors.append("host report qualification fields are not fail-closed C1")
    _exact_live_provenance(
        fresh_host_identity.get("collection_provenance"),
        label="fresh host observation",
        errors=errors,
    )
    if fresh_host_match.get("ok") is not True or fresh_host_match.get("errors") != []:
        errors.append("fresh host observation does not match the selected platform")
    fresh_observed = fresh_host_match.get("observed")
    fresh_gpu_uuid = (
        _matching_gpu_uuid(
            fresh_observed,
            device_index=device_index,
            label="fresh host observation",
            missing_message=(
                "fresh host observation does not match requested device "
                f"gpu:{device_index}"
            ),
            errors=errors,
        )
        if device_index >= 0
        else None
    )
    if (
        recorded_gpu_uuid is not None
        and fresh_gpu_uuid is not None
        and recorded_gpu_uuid.casefold() != fresh_gpu_uuid.casefold()
    ):
        errors.append(
            "recorded and fresh host observations bind the requested "
            "nvidia-smi index to different physical GPU UUIDs"
        )
    if recorded_observed != fresh_observed:
        errors.append(
            "recorded host identity does not match the fresh launch-host observation"
        )
    selected_gpu_uuid = (
        fresh_gpu_uuid
        if (
            recorded_gpu_uuid is not None
            and fresh_gpu_uuid is not None
            and recorded_gpu_uuid.casefold() == fresh_gpu_uuid.casefold()
        )
        else None
    )
    if selected_gpu_uuid is not None:
        warnings.append(
            "The requested gpu:N is a physical nvidia-smi selector. Launch "
            f"must set CUDA_VISIBLE_DEVICES={selected_gpu_uuid} and address "
            "the resulting MAX-visible device as gpu:0."
        )

    runtime_keys = {
        "schema_version",
        "record_kind",
        "evidence_scope",
        "collection_provenance",
        "ok",
        "errors",
        "warnings",
        "expected",
        "observed",
        "commands",
        "qualification",
        "physical_claims",
        "interpretation",
        "catalog_sha256",
        "model",
        "platform",
        "max_executable",
    }
    if set(runtime_report) != runtime_keys:
        errors.append(
            "runtime report fields must exactly match the live report schema"
        )
    if runtime_report.get("schema_version") != 1:
        errors.append("runtime report schema_version must be 1")
    if runtime_report.get("record_kind") != MAX_RUNTIME_REPORT_KIND:
        errors.append(
            f"runtime report record_kind must be {MAX_RUNTIME_REPORT_KIND}"
        )
    if runtime_report.get("evidence_scope") != MAX_RUNTIME_EVIDENCE_SCOPE:
        errors.append("runtime report evidence_scope is invalid")
    if runtime_report.get("catalog_sha256") != expected_catalog_sha256:
        errors.append("runtime report catalog_sha256 does not match current catalog")
    runtime_model = runtime_report.get("model")
    if not isinstance(runtime_model, Mapping) or (
        runtime_model.get("model_id") != expected_model_id
        or runtime_model.get("profile_sha256")
        != expected_model_profile_sha256
        or runtime_model.get("repository") != expected_repository
    ):
        errors.append("runtime report model lineage does not match selected model")
    runtime_platform = runtime_report.get("platform")
    if not isinstance(runtime_platform, Mapping) or (
        runtime_platform.get("platform_id") != expected_platform_id
        or runtime_platform.get("profile_sha256")
        != expected_platform_profile_sha256
    ):
        errors.append(
            "runtime report platform lineage does not match selected platform"
        )
    _exact_live_provenance(
        runtime_report.get("collection_provenance"),
        label="runtime report",
        errors=errors,
    )
    if runtime_report.get("ok") is not True or runtime_report.get("errors") != []:
        errors.append("runtime report exact registry probe must pass without errors")
    if runtime_report.get("expected") != {
        "architecture": expected_architecture,
        "encoding": expected_encoding,
    }:
        errors.append("runtime report expected architecture/encoding do not match")
    runtime_observed = runtime_report.get("observed")
    if not isinstance(runtime_observed, Mapping) or (
        runtime_observed.get("architecture_present") is not True
        or runtime_observed.get("encoding_present") is not True
    ):
        errors.append("runtime report did not observe the exact registry pair")
    physical_claims = runtime_report.get("physical_claims")
    if (
        not isinstance(physical_claims, Mapping)
        or set(physical_claims) != set(MAX_RUNTIME_PHYSICAL_CLAIM_KEYS)
        or any(value is not False for value in physical_claims.values())
    ):
        errors.append("runtime report physical claims must all remain false")
    expected_runtime_qualification = {
        "maturity": "C1_contracted",
        "authority": "exploratory",
        "registry_match_passed": True,
        "runtime_compatibility_passed": False,
        "physical_execution_status": "not_run",
        "production_supported": False,
    }
    if runtime_report.get("qualification") != expected_runtime_qualification:
        errors.append("runtime report qualification fields are not fail-closed C1")

    max_command = tuple(expected_max_command)
    commands = runtime_report.get("commands")
    if not isinstance(commands, Mapping):
        errors.append("runtime report commands must be an object")
    else:
        version = commands.get("version")
        listing = commands.get("list_json")
        if not isinstance(version, Mapping) or version.get("argv") != [
            *max_command,
            "--version",
        ]:
            errors.append("runtime report MAX version argv does not match launch")
        if not isinstance(listing, Mapping) or listing.get("argv") != [
            *max_command,
            "list",
            "--json",
        ]:
            errors.append("runtime report MAX registry argv does not match launch")
    if runtime_report.get("max_executable") != dict(expected_max_executable):
        errors.append(
            "runtime report MAX executable identity does not match the launch executable"
        )
    fresh_runtime_fields = {
        "schema_version",
        "record_kind",
        "evidence_scope",
        "collection_provenance",
        "ok",
        "errors",
        "warnings",
        "expected",
        "observed",
        "commands",
        "qualification",
        "physical_claims",
        "interpretation",
        "catalog_sha256",
        "model",
        "platform",
        "max_executable",
    }
    if set(fresh_runtime_report) != fresh_runtime_fields:
        errors.append(
            "fresh runtime report fields must exactly match the live report schema"
        )
    _exact_live_provenance(
        fresh_runtime_report.get("collection_provenance"),
        label="fresh runtime report",
        errors=errors,
    )
    if (
        fresh_runtime_report.get("ok") is not True
        or fresh_runtime_report.get("errors") != []
    ):
        errors.append("fresh MAX runtime registry probe did not pass")
    for field in (
        "catalog_sha256",
        "model",
        "platform",
        "expected",
        "observed",
        "qualification",
        "physical_claims",
        "max_executable",
    ):
        if fresh_runtime_report.get(field) != runtime_report.get(field):
            errors.append(
                f"recorded runtime report {field} does not match fresh observation"
            )

    binding = {
        "schema_version": 1,
        "record_kind": NVIDIA_SINGLE_PREFLIGHT_KIND,
        "ok": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": warnings,
        "catalog_sha256": expected_catalog_sha256,
        "model_id": expected_model_id,
        "model_profile_sha256": expected_model_profile_sha256,
        "platform_id": expected_platform_id,
        "platform_profile_sha256": expected_platform_profile_sha256,
        "model_dir": model_dir,
        "device": expected_device,
        "device_binding": {
            "physical_selector": expected_device,
            "nvidia_smi_index": device_index if device_index >= 0 else None,
            "nvidia_gpu_uuid": selected_gpu_uuid,
            "cuda_visible_devices": selected_gpu_uuid,
            "max_launch_device": "gpu:0",
            "verified": selected_gpu_uuid is not None,
        },
        "max_command": list(max_command),
        "max_executable": dict(expected_max_executable),
        "nvidia_smi_executable": dict(expected_nvidia_smi_executable),
        "evidence_files": {
            "model_artifacts": dict(artifact_file),
            "host_identity": dict(host_file),
            "max_runtime": dict(runtime_file),
        },
        "fresh_artifact_manifest_sha256": fresh_artifact_report.get(
            "artifact_manifest_sha256"
        ),
        "fresh_host_observation_sha256": _sha256_bytes(
            json.dumps(
                {
                    "identity": fresh_host_identity,
                    "match": fresh_host_match,
                },
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ),
        "fresh_runtime_report_sha256": _sha256_bytes(
            json.dumps(
                fresh_runtime_report,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ),
        "artifact_validations": artifact_validations,
        "authenticated": False,
    }
    canonical = json.dumps(
        binding,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    binding["binding_sha256"] = _sha256_bytes(canonical)
    return binding


__all__ = [
    "APPLE_SINGLE_PREFLIGHT_KIND",
    "HOST_REPORT_KIND",
    "MAX_EVIDENCE_FILE_BYTES",
    "NVIDIA_SINGLE_PREFLIGHT_KIND",
    "QualificationEvidenceError",
    "executable_identity",
    "load_json_evidence",
    "validate_apple_single_preflights",
    "validate_nvidia_single_preflights",
]
