"""Integrity manifest and verification for rendered qualification packets.

The manifest is content-addressing, not a signature.  It detects incomplete or
modified managed files relative to the published manifest, while authenticity
still depends on how the operator obtains and records the manifest digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping


PACKET_SCHEMA_VERSION = 1
PACKET_MANIFEST_KIND = "fornax_qualification_recipe_packet_manifest"
PACKET_VERIFICATION_KIND = "fornax_qualification_recipe_packet_verification"
PACKET_MANIFEST_FILENAME = "bundle-manifest.json"
MANAGED_PACKET_FILENAMES = (
    "recipe-lock.json",
    "commands.json",
    "RUNBOOK.md",
)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANAGED_FILE_BYTES = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
INTEGRITY_SCOPE = (
    "Managed-file byte integrity and packet self-consistency only; "
    "this unsigned manifest does not establish publisher authenticity."
)
PHYSICAL_CLAIM_KEYS = {
    "distributed_runtime_passed",
    "formal_g2_passed",
    "formal_g3_passed",
    "hardware_detected",
    "model_artifact_verified",
    "production_supported",
    "single_platform_bringup_passed",
    "target_model_parity_passed",
}
_LOCK_KEYS = {
    "schema_version",
    "record_kind",
    "recipe_id",
    "catalog_id",
    "catalog_sha256",
    "inputs",
    "capacity_estimate",
    "precision_contract",
    "qualification",
    "required_operations",
    "physical_claims",
    "blockers",
    "lock_content_sha256",
}
_COMMAND_KEYS = {
    "schema_version",
    "record_kind",
    "recipe_id",
    "substitution_policy",
    "commands",
    "physical_claims",
}
_COMMAND_ROW_KEYS = {
    "step_id",
    "purpose",
    "argv",
    "network_required",
    "execution_status",
    "evidence_scope",
}
_SHELL_EXECUTABLES = {
    "bash",
    "cmd",
    "fish",
    "powershell",
    "pwsh",
    "sh",
    "zsh",
}


class RecipePacketError(ValueError):
    """Raised when packet inputs cannot form a valid integrity manifest."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RecipePacketError(f"value is not canonical JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_canonical(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecipePacketError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RecipePacketError(f"non-finite JSON number is forbidden: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecipePacketError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _load_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except RecipePacketError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RecipePacketError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RecipePacketError(f"{label} top level must be an object")
    return parsed


def build_recipe_packet_manifest(
    recipe_id: str,
    managed_files: Mapping[str, bytes],
) -> dict[str, Any]:
    """Build a deterministic byte-integrity manifest for managed packet files."""

    if not isinstance(recipe_id, str) or not recipe_id:
        raise RecipePacketError("recipe_id must be a non-empty string")
    expected_names = set(MANAGED_PACKET_FILENAMES)
    actual_names = set(managed_files)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unknown = sorted(actual_names - expected_names)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise RecipePacketError(
            "managed packet files must match the contract: " + "; ".join(details)
        )

    file_records: dict[str, dict[str, Any]] = {}
    for name in MANAGED_PACKET_FILENAMES:
        value = managed_files[name]
        if not isinstance(value, bytes):
            raise RecipePacketError(f"managed packet file {name} must be bytes")
        if len(value) > MAX_MANAGED_FILE_BYTES:
            raise RecipePacketError(
                f"managed packet file {name} exceeds {MAX_MANAGED_FILE_BYTES} bytes"
            )
        file_records[name] = {
            "size_bytes": len(value),
            "sha256": _sha256_bytes(value),
        }

    payload = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "record_kind": PACKET_MANIFEST_KIND,
        "recipe_id": recipe_id,
        "files": file_records,
        "integrity_scope": INTEGRITY_SCOPE,
    }
    manifest = dict(payload)
    manifest["bundle_content_sha256"] = _sha256_canonical(payload)
    return manifest


def _manifest_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_keys = {
        "schema_version",
        "record_kind",
        "recipe_id",
        "files",
        "integrity_scope",
        "bundle_content_sha256",
    }
    if set(manifest) != required_keys:
        missing = sorted(required_keys - set(manifest))
        unknown = sorted(set(manifest) - required_keys)
        if missing:
            errors.append("bundle manifest missing keys: " + ", ".join(missing))
        if unknown:
            errors.append("bundle manifest has unknown keys: " + ", ".join(unknown))
        return errors
    if (
        isinstance(manifest["schema_version"], bool)
        or not isinstance(manifest["schema_version"], int)
        or manifest["schema_version"] != PACKET_SCHEMA_VERSION
    ):
        errors.append("bundle manifest schema_version must be 1")
    if manifest["record_kind"] != PACKET_MANIFEST_KIND:
        errors.append("bundle manifest record_kind is invalid")
    if not isinstance(manifest["recipe_id"], str) or not manifest["recipe_id"]:
        errors.append("bundle manifest recipe_id must be a non-empty string")
    if manifest["integrity_scope"] != INTEGRITY_SCOPE:
        errors.append("bundle manifest integrity_scope does not match the contract")
    claimed_content_hash = manifest["bundle_content_sha256"]
    if not isinstance(claimed_content_hash, str) or not _SHA256_RE.fullmatch(
        claimed_content_hash
    ):
        errors.append("bundle manifest content hash must use sha256:<64 lowercase hex>")
    else:
        payload = dict(manifest)
        payload.pop("bundle_content_sha256")
        try:
            observed_content_hash = _sha256_canonical(payload)
        except RecipePacketError as exc:
            errors.append(f"bundle manifest payload is not canonical JSON: {exc}")
        else:
            if observed_content_hash != claimed_content_hash:
                errors.append("bundle manifest content hash does not match its payload")

    files = manifest["files"]
    if not isinstance(files, dict):
        errors.append("bundle manifest files must be an object")
        return errors
    expected_names = set(MANAGED_PACKET_FILENAMES)
    if set(files) != expected_names:
        missing = sorted(expected_names - set(files))
        unknown = sorted(set(files) - expected_names)
        if missing:
            errors.append("bundle manifest is missing managed files: " + ", ".join(missing))
        if unknown:
            errors.append("bundle manifest has unknown managed files: " + ", ".join(unknown))
        return errors
    for name in MANAGED_PACKET_FILENAMES:
        record = files[name]
        if not isinstance(record, dict) or set(record) != {"size_bytes", "sha256"}:
            errors.append(
                f"bundle manifest file record {name} must contain size_bytes and sha256"
            )
            continue
        size = record["size_bytes"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > MAX_MANAGED_FILE_BYTES
        ):
            errors.append(
                f"bundle manifest size for {name} must be between 0 and "
                f"{MAX_MANAGED_FILE_BYTES}"
            )
        digest = record["sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(
                f"bundle manifest digest for {name} must use "
                "sha256:<64 lowercase hex>"
            )
    return errors


def _open_packet_directory(root: Path) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RecipePacketError(
            "packet verification requires O_DIRECTORY and O_NOFOLLOW"
        )
    flags = os.O_RDONLY
    flags |= os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise RecipePacketError(
            f"packet directory cannot be opened without following a symlink: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RecipePacketError("packet path is not a directory")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RecipePacketError(
            "packet file verification requires O_NOFOLLOW and O_NONBLOCK"
        )
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW
    flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RecipePacketError(
            f"{name} cannot be opened as a no-follow packet file: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RecipePacketError(f"{name} must be a regular file")
        if metadata.st_nlink != 1:
            raise RecipePacketError(
                f"{name} must have exactly one hard link; observed {metadata.st_nlink}"
            )
        if metadata.st_size > max_bytes:
            raise RecipePacketError(f"{name} exceeds {max_bytes} bytes")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) > max_bytes:
            raise RecipePacketError(f"{name} exceeds {max_bytes} bytes")
        if len(value) != metadata.st_size:
            raise RecipePacketError(
                f"{name} changed size while it was being verified"
            )
        return value
    finally:
        os.close(descriptor)


def _recipe_lock_catalog_binding(
    managed_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    """Extract catalog-selection fields from the verifier's captured lock bytes."""

    lock_bytes = managed_bytes.get("recipe-lock.json")
    if not isinstance(lock_bytes, bytes):
        raise RecipePacketError(
            "recipe-lock.json bytes are unavailable for catalog binding"
        )
    lock = _load_json_bytes(lock_bytes, "recipe-lock.json")
    inputs = lock.get("inputs")
    model = inputs.get("model") if isinstance(inputs, dict) else None
    platform = inputs.get("platform") if isinstance(inputs, dict) else None
    capacity = lock.get("capacity_estimate")
    binding = {
        "recipe_id": lock.get("recipe_id"),
        "catalog_id": lock.get("catalog_id"),
        "catalog_sha256": lock.get("catalog_sha256"),
        "model_id": model.get("model_id") if isinstance(model, dict) else None,
        "platform_id": (
            platform.get("platform_id")
            if isinstance(platform, dict)
            else None
        ),
        "selected_units": (
            capacity.get("selected_units")
            if isinstance(capacity, dict)
            else None
        ),
    }
    for field in (
        "recipe_id",
        "catalog_id",
        "catalog_sha256",
        "model_id",
        "platform_id",
    ):
        if not isinstance(binding[field], str) or not binding[field]:
            raise RecipePacketError(
                f"recipe-lock.json catalog binding {field} must be a "
                "non-empty string"
            )
    if _SHA256_RE.fullmatch(binding["catalog_sha256"]) is None:
        raise RecipePacketError(
            "recipe-lock.json catalog binding catalog_sha256 is invalid"
        )
    units = binding["selected_units"]
    if isinstance(units, bool) or not isinstance(units, int) or units < 1:
        raise RecipePacketError(
            "recipe-lock.json catalog binding selected_units must be a "
            "positive integer"
        )
    return binding


def _duplicate_directory_fd(directory_fd: int) -> int:
    if (
        isinstance(directory_fd, bool)
        or not isinstance(directory_fd, int)
        or directory_fd < 0
    ):
        raise RecipePacketError("packet directory descriptor must be non-negative")
    try:
        descriptor = os.dup(directory_fd)
    except OSError as exc:
        raise RecipePacketError(
            f"packet directory descriptor cannot be duplicated: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RecipePacketError("packet descriptor does not refer to a directory")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _semantic_errors(
    manifest: dict[str, Any],
    managed_bytes: Mapping[str, bytes],
) -> list[str]:
    errors: list[str] = []
    try:
        lock = _load_json_bytes(managed_bytes["recipe-lock.json"], "recipe-lock.json")
        commands = _load_json_bytes(managed_bytes["commands.json"], "commands.json")
    except RecipePacketError as exc:
        return [str(exc)]
    try:
        runbook = managed_bytes["RUNBOOK.md"].decode("utf-8")
    except UnicodeError as exc:
        return [f"RUNBOOK.md is not valid UTF-8: {exc}"]

    recipe_id = manifest.get("recipe_id")
    for label, record, kind, expected_keys in (
        (
            "recipe-lock.json",
            lock,
            "fornax_qualification_recipe_lock",
            _LOCK_KEYS,
        ),
        (
            "commands.json",
            commands,
            "fornax_qualification_recipe_commands",
            _COMMAND_KEYS,
        ),
    ):
        if set(record) != expected_keys:
            missing = sorted(expected_keys - set(record))
            unknown = sorted(set(record) - expected_keys)
            if missing:
                errors.append(f"{label} missing keys: " + ", ".join(missing))
            if unknown:
                errors.append(f"{label} has unknown keys: " + ", ".join(unknown))
        schema_version = record.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            errors.append(f"{label} schema_version must be 1")
        if record.get("record_kind") != kind:
            errors.append(f"{label} record_kind is invalid")
        if record.get("recipe_id") != recipe_id:
            errors.append(f"{label} recipe_id does not match bundle manifest")

    lock_hash = lock.get("lock_content_sha256")
    if not isinstance(lock_hash, str) or not _SHA256_RE.fullmatch(lock_hash):
        errors.append("recipe-lock.json lock_content_sha256 is invalid")
    else:
        lock_payload = dict(lock)
        lock_payload.pop("lock_content_sha256", None)
        try:
            observed_lock_hash = _sha256_canonical(lock_payload)
        except RecipePacketError as exc:
            errors.append(f"recipe-lock.json payload is not canonical JSON: {exc}")
        else:
            if observed_lock_hash != lock_hash:
                errors.append(
                    "recipe-lock.json content hash does not match its payload"
                )

    lock_claims = lock.get("physical_claims")
    command_claims = commands.get("physical_claims")
    if not isinstance(lock_claims, dict) or set(lock_claims) != PHYSICAL_CLAIM_KEYS:
        errors.append(
            "recipe-lock.json physical_claims must contain the exact C1 claim keys"
        )
    elif any(value is not False for value in lock_claims.values()):
        errors.append("recipe-lock.json contains a non-false physical claim")
    if command_claims != lock_claims:
        errors.append("commands.json physical_claims do not match recipe-lock.json")

    qualification = lock.get("qualification")
    expected_qualification = {
        "maturity": "C1_contracted",
        "support_state": "contract_validated",
        "physical_execution_status": "not_run",
        "authority": "exploratory",
    }
    if qualification != expected_qualification:
        errors.append(
            "recipe-lock.json qualification must remain C1 contracted, "
            "contract validated, exploratory, and not run"
        )
    capacity = lock.get("capacity_estimate")
    if not isinstance(capacity, dict):
        errors.append("recipe-lock.json capacity_estimate must be an object")
    else:
        if capacity.get("capacity_only") is not True:
            errors.append("recipe-lock.json capacity estimate must be capacity-only")
        if capacity.get("performance_feasibility_evaluated") is not False:
            errors.append(
                "recipe-lock.json must not claim performance feasibility was evaluated"
            )
        if capacity.get("capacity_sufficient_by_estimate") is not True:
            errors.append(
                "rendered recipe must meet its capacity-only selected-unit minimum"
            )
        selected_units = capacity.get("selected_units")
        if (
            isinstance(selected_units, bool)
            or not isinstance(selected_units, int)
            or selected_units < 1
        ):
            errors.append(
                "recipe-lock.json selected_units must be a positive integer"
            )
    precision = lock.get("precision_contract")
    if not isinstance(precision, dict):
        errors.append("recipe-lock.json precision_contract must be an object")
    elif precision.get("scope") is None:
        errors.append("recipe-lock.json precision_contract must declare its scope")

    substitution_policy = commands.get("substitution_policy")
    if (
        not isinstance(substitution_policy, str)
        or "Do not concatenate" not in substitution_policy
        or "shell command" not in substitution_policy
    ):
        errors.append(
            "commands.json substitution_policy must preserve argv boundaries "
            "and forbid shell concatenation"
        )

    command_rows = commands.get("commands")
    if not isinstance(command_rows, list) or not command_rows:
        errors.append("commands.json commands must be a non-empty array")
    else:
        step_ids: set[str] = set()
        for index, row in enumerate(command_rows):
            if not isinstance(row, dict):
                errors.append(f"commands.json commands[{index}] must be an object")
                continue
            if set(row) != _COMMAND_ROW_KEYS:
                errors.append(
                    f"commands.json commands[{index}] must contain the exact "
                    "command-contract keys"
                )
            step_id = row.get("step_id")
            if not isinstance(step_id, str) or not step_id:
                errors.append(
                    f"commands.json commands[{index}].step_id must be non-empty"
                )
            elif step_id in step_ids:
                errors.append(f"commands.json contains duplicate step_id {step_id}")
            else:
                step_ids.add(step_id)
            argv = row.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(value, str) or not value for value in argv)
            ):
                errors.append(
                    f"commands.json commands[{index}].argv must be a non-empty "
                    "array of non-empty strings"
                )
            else:
                if Path(argv[0]).name.casefold() in _SHELL_EXECUTABLES:
                    errors.append(
                        f"commands.json commands[{index}] invokes a forbidden shell"
                    )
                if any(
                    value in {"&&", "||", ";", "|"}
                    or "$(" in value
                    or "`" in value
                    for value in argv
                ):
                    errors.append(
                        f"commands.json commands[{index}] contains a shell operator"
                    )
            purpose = row.get("purpose")
            if not isinstance(purpose, str) or not purpose:
                errors.append(
                    f"commands.json commands[{index}].purpose must be non-empty"
                )
            network_required = row.get("network_required")
            if network_required is not True and network_required is not False:
                errors.append(
                    f"commands.json commands[{index}].network_required must be boolean"
                )
            status = row.get("execution_status")
            if not isinstance(status, str) or not status:
                errors.append(
                    f"commands.json commands[{index}].execution_status must be non-empty"
                )
            scope = row.get("evidence_scope")
            if not isinstance(scope, str) or not scope:
                errors.append(
                    f"commands.json commands[{index}].evidence_scope must be non-empty"
                )

        required_steps = {
            "acquire_pinned_model",
            "inspect_local_model",
            "probe_host_identity",
            "collect_local_inventory",
            "probe_max_runtime_registry",
        }
        missing_steps = sorted(required_steps - step_ids)
        if missing_steps:
            errors.append(
                "commands.json is missing required steps: " + ", ".join(missing_steps)
            )
        terminal_steps = step_ids & {
            "single_platform_model_bringup",
            "capacity_spanning_readiness",
        }
        if len(terminal_steps) != 1:
            errors.append(
                "commands.json must contain exactly one single-unit bring-up or "
                "capacity-spanning readiness step"
            )

    if isinstance(recipe_id, str) and recipe_id not in runbook:
        errors.append("RUNBOOK.md does not contain the bundle recipe_id")
    if isinstance(lock_hash, str) and lock_hash not in runbook:
        errors.append("RUNBOOK.md does not contain the recipe lock content hash")
    if "C1 contracted / contract validated" not in runbook:
        errors.append("RUNBOOK.md does not declare the C1 evidence boundary")
    if "not a supported-hardware" not in runbook:
        errors.append("RUNBOOK.md does not disclaim supported-hardware status")
    return errors


def verify_recipe_packet(
    packet_dir: str | Path,
    *,
    expected_bundle_content_sha256: str | None = None,
    allow_unmanaged_entries: bool = False,
    _directory_fd: int | None = None,
) -> dict[str, Any]:
    """Verify packet bytes and semantics without treating them as authenticated.

    ``expected_bundle_content_sha256`` is an optional out-of-band comparison
    anchor.  Matching it still does not create a signature or publisher
    authentication.
    """

    root = Path(packet_dir)
    consistency_errors: list[str] = []
    policy_errors: list[str] = []
    manifest: dict[str, Any] | None = None
    managed_bytes: dict[str, bytes] = {}
    observed_files: dict[str, dict[str, Any]] = {}
    extras: list[str] = []
    directory_fd: int | None = None
    recipe_lock_binding: dict[str, Any] | None = None

    if expected_bundle_content_sha256 is not None and (
        not isinstance(expected_bundle_content_sha256, str)
        or not _SHA256_RE.fullmatch(expected_bundle_content_sha256)
    ):
        policy_errors.append(
            "expected bundle digest must use sha256:<64 lowercase hex>"
        )

    try:
        directory_fd = (
            _open_packet_directory(root)
            if _directory_fd is None
            else _duplicate_directory_fd(_directory_fd)
        )
        try:
            manifest_bytes = _read_regular_file_at(
                directory_fd,
                PACKET_MANIFEST_FILENAME,
                max_bytes=MAX_MANIFEST_BYTES,
            )
            manifest = _load_json_bytes(
                manifest_bytes,
                PACKET_MANIFEST_FILENAME,
            )
            consistency_errors.extend(_manifest_errors(manifest))
        except RecipePacketError as exc:
            consistency_errors.append(str(exc))

        if manifest is not None and isinstance(manifest.get("files"), dict):
            for name in MANAGED_PACKET_FILENAMES:
                try:
                    value = _read_regular_file_at(
                        directory_fd,
                        name,
                        max_bytes=MAX_MANAGED_FILE_BYTES,
                    )
                except RecipePacketError as exc:
                    consistency_errors.append(str(exc))
                    continue
                managed_bytes[name] = value
                observed_files[name] = {
                    "size_bytes": len(value),
                    "sha256": _sha256_bytes(value),
                }
                expected = manifest["files"].get(name)
                if not isinstance(expected, dict):
                    continue
                if expected.get("size_bytes") != len(value):
                    consistency_errors.append(
                        f"managed packet file size mismatch: {name}"
                    )
                if expected.get("sha256") != observed_files[name]["sha256"]:
                    consistency_errors.append(
                        f"managed packet file digest mismatch: {name}"
                    )

        if (
            manifest is not None
            and set(managed_bytes) == set(MANAGED_PACKET_FILENAMES)
        ):
            consistency_errors.extend(_semantic_errors(manifest, managed_bytes))
            try:
                recipe_lock_binding = _recipe_lock_catalog_binding(
                    managed_bytes
                )
            except RecipePacketError as exc:
                consistency_errors.append(str(exc))

        try:
            managed_names = {
                PACKET_MANIFEST_FILENAME,
                *MANAGED_PACKET_FILENAMES,
            }
            extras = sorted(
                name
                for name in os.listdir(directory_fd)
                if name not in managed_names
            )
        except OSError as exc:
            consistency_errors.append(f"cannot list packet directory: {exc}")
    except RecipePacketError as exc:
        consistency_errors.append(str(exc))
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    if extras and not allow_unmanaged_entries:
        policy_errors.append(
            "packet contains unmanaged entries: " + ", ".join(extras)
        )

    observed_bundle_digest = (
        manifest.get("bundle_content_sha256") if manifest is not None else None
    )
    if expected_bundle_content_sha256 is None:
        expected_digest_matched: bool | None = None
    else:
        expected_digest_matched = (
            observed_bundle_digest == expected_bundle_content_sha256
        )
        if expected_digest_matched is False:
            policy_errors.append(
                "bundle content digest does not match the expected out-of-band digest"
            )

    self_consistent = not consistency_errors
    errors = [*consistency_errors, *policy_errors]
    return {
        "schema_version": PACKET_SCHEMA_VERSION,
        "record_kind": PACKET_VERIFICATION_KIND,
        "packet_dir": str(root),
        "recipe_id": manifest.get("recipe_id") if manifest is not None else None,
        "bundle_content_sha256": observed_bundle_digest,
        "expected_bundle_content_sha256": expected_bundle_content_sha256,
        "expected_digest_matched": expected_digest_matched,
        "self_consistent": self_consistent,
        "authenticated": False,
        "recipe_lock_binding": recipe_lock_binding,
        "managed_files": observed_files,
        "unmanaged_entries": extras,
        "allow_unmanaged_entries": allow_unmanaged_entries,
        "ok": self_consistent and not policy_errors,
        "errors": errors,
        "integrity_scope": INTEGRITY_SCOPE,
    }


__all__ = [
    "MANAGED_PACKET_FILENAMES",
    "PACKET_MANIFEST_FILENAME",
    "PACKET_MANIFEST_KIND",
    "PACKET_SCHEMA_VERSION",
    "RecipePacketError",
    "build_recipe_packet_manifest",
    "verify_recipe_packet",
]
