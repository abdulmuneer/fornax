from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "fornax_model_artifact_report"
REPORT_MODE = "offline_local_inspection"
MAX_REPORTED_FILES = 512
MAX_CONFIG_BYTES = 16 * 1024 * 1024
MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_REMOTE_CODE_SCAN_DIRECTORIES = 1024
MAX_REMOTE_CODE_SCAN_ENTRIES = 16 * 1024
MAX_REMOTE_CODE_SCAN_DEPTH = 32
SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
PREFIXED_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
REVISION_MARKERS = (
    ".fornax-revision",
    ".fornax_revision",
    "fornax-revision.txt",
    "revision.txt",
)
TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tekken.json",
    "tiktoken.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
TOKENIZER_PAYLOAD_NAMES = {
    "sentencepiece.bpe.model",
    "spiece.model",
    "tekken.json",
    "tiktoken.model",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.txt",
}
UNRESOLVED_REMOTE_CODE_POLICIES = {
    "blocked",
    "denied",
    "",
    "none",
    "not_reviewed",
    "not_recorded",
    "operator_approval_required",
    "pending_review",
    "review_required",
    "required",
    "unknown",
    "unresolved",
}
_MISSING = object()


class _ModelRootReader:
    """Descriptor-anchored, no-follow access to one model snapshot.

    Every path component below the root is opened relative to an already-open
    directory descriptor.  Artifact reads therefore cannot escape through a
    symlink or switch to a replacement model directory after inspection starts.
    """

    def __init__(self, root: Path) -> None:
        if not hasattr(os, "O_NOFOLLOW"):
            raise OSError("O_NOFOLLOW is unavailable on this platform")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        self._fd = os.open(root, flags)
        metadata = os.fstat(self._fd)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(self._fd)
            raise NotADirectoryError(str(root))

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def _open_directory(self, relative_dir: str | None = None) -> int:
        descriptor = os.dup(self._fd)
        if not relative_dir:
            return descriptor
        safe = _safe_relative_path(relative_dir)
        if safe is None:
            os.close(descriptor)
            raise OSError(f"unsafe relative directory path: {relative_dir!r}")
        try:
            for part in PurePosixPath(safe).parts:
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _open_regular_file(self, relative_path: str) -> tuple[int, os.stat_result]:
        safe = _safe_relative_path(relative_path)
        if safe is None:
            raise OSError(f"unsafe relative file path: {relative_path!r}")
        parts = PurePosixPath(safe).parts
        parent = self._open_directory(
            "/".join(parts[:-1]) if len(parts) > 1 else None
        )
        try:
            entry_metadata = os.stat(
                parts[-1],
                dir_fd=parent,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(entry_metadata.st_mode):
                raise OSError(
                    f"{safe} must be a no-follow regular file"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
            flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(parts[-1], flags, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError(f"{safe} is not a regular file")
            return descriptor, metadata
        except Exception:
            os.close(descriptor)
            raise

    def listdir(
        self,
        relative_dir: str | None = None,
        *,
        maximum: int | None = None,
    ) -> list[str]:
        descriptor = self._open_directory(relative_dir)
        try:
            names: list[str] = []
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if maximum is not None and len(names) > maximum:
                        label = relative_dir or "."
                        raise OSError(
                            f"directory {label} exceeds the bounded "
                            f"{maximum}-entry limit"
                        )
        finally:
            os.close(descriptor)
        return names

    def lstat(self, relative_path: str) -> os.stat_result:
        safe = _safe_relative_path(relative_path)
        if safe is None:
            raise OSError(f"unsafe relative path: {relative_path!r}")
        parts = PurePosixPath(safe).parts
        parent = self._open_directory(
            "/".join(parts[:-1]) if len(parts) > 1 else None
        )
        try:
            return os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        finally:
            os.close(parent)

    def stat_regular_file(self, relative_path: str) -> os.stat_result:
        descriptor, metadata = self._open_regular_file(relative_path)
        os.close(descriptor)
        return metadata

    def read_regular_file(
        self,
        relative_path: str,
        *,
        max_bytes: int | None = None,
        expected_size: int | None = None,
        capture: bool = False,
    ) -> tuple[int, str, bytes | None, os.stat_result]:
        descriptor, before = self._open_regular_file(relative_path)
        try:
            if expected_size is not None and before.st_size != expected_size:
                raise OSError(
                    f"{relative_path} changed size after preflight: expected "
                    f"{expected_size}, observed {before.st_size}"
                )
            if max_bytes is not None and before.st_size > max_bytes:
                raise OSError(
                    f"{relative_path} exceeds the {max_bytes}-byte read limit"
                )
            digest = hashlib.sha256()
            chunks: list[bytes] | None = [] if capture else None
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise OSError(
                        f"{relative_path} changed size while it was being read"
                    )
                remaining -= len(chunk)
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            if os.read(descriptor, 1):
                raise OSError(
                    f"{relative_path} grew while it was being read"
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
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise OSError(
                    f"{relative_path} changed while it was being read"
                )
            value = b"".join(chunks) if chunks is not None else None
            return (
                before.st_size,
                "sha256:" + digest.hexdigest(),
                value,
                before,
            )
        finally:
            os.close(descriptor)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _manifest_sha256(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row.get("path", ""))):
        digest.update(str(record.get("path", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.get("bytes", "")).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record.get("sha256", "")).encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if "\x00" in value:
        return None
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        return None
    if any(not part for part in path.parts):
        return None
    return str(path)


def _inventory_python_files(
    reader: _ModelRootReader,
) -> tuple[list[str], list[str]]:
    """Inventory regular ``*.py`` files without following model-root links."""

    pending: list[tuple[str | None, int]] = [(None, 0)]
    python_files: list[str] = []
    errors: list[str] = []
    scanned_directories = 0
    scanned_entries = 0
    while pending:
        relative_dir, depth = pending.pop()
        scanned_directories += 1
        if scanned_directories > MAX_REMOTE_CODE_SCAN_DIRECTORIES:
            errors.append(
                "remote-code dependency inventory exceeds the bounded "
                f"{MAX_REMOTE_CODE_SCAN_DIRECTORIES}-directory limit"
            )
            break
        try:
            names = reader.listdir(
                relative_dir,
                maximum=MAX_REMOTE_CODE_SCAN_ENTRIES,
            )
        except OSError as exc:
            label = relative_dir or "."
            errors.append(
                f"remote-code dependency inventory cannot list {label}: {exc}"
            )
            continue
        for name in sorted(names):
            scanned_entries += 1
            if scanned_entries > MAX_REMOTE_CODE_SCAN_ENTRIES:
                errors.append(
                    "remote-code dependency inventory exceeds the bounded "
                    f"{MAX_REMOTE_CODE_SCAN_ENTRIES}-entry limit"
                )
                return sorted(python_files), errors
            path = name if relative_dir is None else f"{relative_dir}/{name}"
            safe_path = _safe_relative_path(path)
            if safe_path is None:
                errors.append(
                    "remote-code dependency inventory encountered an unsafe "
                    f"path: {path!r}"
                )
                continue
            try:
                metadata = reader.lstat(safe_path)
            except OSError as exc:
                errors.append(
                    "remote-code dependency inventory cannot inspect "
                    f"{safe_path}: {exc}"
                )
                continue
            if stat.S_ISLNK(metadata.st_mode):
                errors.append(
                    "remote-code dependency inventory refuses symbolic link: "
                    f"{safe_path}"
                )
            elif stat.S_ISDIR(metadata.st_mode):
                if depth >= MAX_REMOTE_CODE_SCAN_DEPTH:
                    errors.append(
                        "remote-code dependency inventory exceeds the bounded "
                        f"{MAX_REMOTE_CODE_SCAN_DEPTH}-level depth at {safe_path}"
                    )
                else:
                    pending.append((safe_path, depth + 1))
            elif stat.S_ISREG(metadata.st_mode):
                if safe_path.endswith(".py"):
                    python_files.append(safe_path)
            elif safe_path.endswith(".py"):
                errors.append(
                    "remote-code dependency must be a no-follow regular file: "
                    f"{safe_path}"
                )
    return sorted(python_files), errors


def _nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _first_nested(value: dict[str, Any], paths: Iterable[str]) -> tuple[Any, str | None]:
    for path in paths:
        candidate = _nested(value, path)
        if candidate is not _MISSING:
            return candidate, path
    return _MISSING, None


def _normalize_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = SHA256_RE.fullmatch(value.strip())
    return "sha256:" + match.group(1) if match else None


def _normalize_revision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    revision = value.strip()
    return revision if REVISION_RE.fullmatch(revision) else None


def _parse_json_object(
    value: bytes,
    *,
    field: str,
    blockers: list[str],
) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        blockers.append(f"{field} is not valid UTF-8 JSON: {exc}")
        return None
    if not isinstance(parsed, dict):
        blockers.append(f"{field} must contain a JSON object")
        return None
    return parsed


def _config_value(
    config: dict[str, Any], aliases: Iterable[str]
) -> tuple[Any, str | None]:
    sections: list[tuple[str, dict[str, Any]]] = [("", config)]
    for name in ("text_config", "language_config", "llm_config"):
        section = config.get(name)
        if isinstance(section, dict):
            sections.append((name + ".", section))
    for prefix, section in sections:
        for alias in aliases:
            value = section.get(alias, _MISSING)
            if value is not _MISSING:
                return value, prefix + alias
    return None, None


def _parsed_config(config: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "decoder_layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "hidden_size": ("hidden_size", "n_embd", "d_model"),
        "total_experts": (
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "moe_num_experts",
        ),
        "active_experts_per_token": (
            "num_experts_per_tok",
            "experts_per_token",
            "num_selected_experts",
            "moe_top_k",
            "top_k",
        ),
        "shared_experts": ("n_shared_experts", "num_shared_experts"),
        "config_max_position_embeddings": (
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        ),
    }
    parsed: dict[str, Any] = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "values": {},
        "config_keys": {},
    }
    for field, field_aliases in aliases.items():
        value, source = _config_value(config, field_aliases)
        parsed["values"][field] = value
        parsed["config_keys"][field] = source
    quantization = config.get("quantization_config")
    if isinstance(quantization, dict):
        parsed["quantization"] = {
            key: quantization.get(key)
            for key in (
                "quant_method",
                "format",
                "bits",
                "group_size",
                "activation_scheme",
                "fmt",
            )
            if key in quantization
        }
    else:
        parsed["quantization"] = None
    return parsed


def _profile_expectations(profile: dict[str, Any]) -> list[tuple[str, Any]]:
    fields = {
        "family": (
            "architecture.family",
            "architecture.model_type",
            "model_type",
        ),
        "decoder_layers": (
            "architecture.decoder_layers",
            "architecture.num_hidden_layers",
            "architecture.num_layers",
            "num_hidden_layers",
            "num_layers",
        ),
        "hidden_size": ("architecture.hidden_size", "hidden_size"),
        "total_experts": (
            "architecture.total_experts",
            "architecture.num_experts",
            "architecture.moe.total_experts",
            "architecture.moe.num_experts",
            "moe.total_experts",
            "moe.num_experts",
            "num_experts",
        ),
        "active_experts_per_token": (
            "architecture.active_experts_per_token",
            "architecture.moe.active_experts_per_token",
            "architecture.moe.top_k",
            "moe.active_experts_per_token",
            "moe.top_k",
            "active_experts_per_token",
            "top_k",
        ),
        "config_max_position_embeddings": (
            "architecture.config_max_position_embeddings",
            "config.max_position_embeddings",
            "max_position_embeddings",
        ),
    }
    result: list[tuple[str, Any]] = []
    for canonical, paths in fields.items():
        expected, _ = _first_nested(profile, paths)
        if expected is not _MISSING:
            result.append((canonical, expected))
    architectures, _ = _first_nested(
        profile,
        (
            "architecture.architectures",
            "architecture.model_classes",
            "architectures",
        ),
    )
    if architectures is not _MISSING:
        result.append(("architectures", architectures))
    return result


def _expected_file_hashes(profile: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    def add_map(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for raw_path, raw_value in value.items():
            path = _safe_relative_path(raw_path)
            if path is None:
                continue
            hash_value = raw_value
            if isinstance(raw_value, dict):
                hash_value = raw_value.get("sha256")
            normalized = _normalize_hash(hash_value)
            if normalized is not None:
                result[path] = normalized

    for path in (
        "file_hashes",
        "artifact.file_hashes",
        "artifact.expected_file_hashes",
        "artifacts.file_hashes",
        "artifacts.expected_file_hashes",
    ):
        add_map(_nested(profile, path))
    for path in ("artifact.files", "artifacts.files", "artifact_manifest.files"):
        value = _nested(profile, path)
        if isinstance(value, list):
            for row in value:
                if not isinstance(row, dict):
                    continue
                file_path = _safe_relative_path(row.get("path"))
                digest = _normalize_hash(row.get("sha256"))
                if file_path is not None and digest is not None:
                    result[file_path] = digest
        else:
            add_map(value)
    config_hash, _ = _first_nested(
        profile,
        ("artifact.config_sha256", "artifacts.config_sha256", "config_sha256"),
    )
    normalized_config_hash = _normalize_hash(config_hash)
    if normalized_config_hash is not None:
        result["config.json"] = normalized_config_hash
    return result


def _required_files(profile: dict[str, Any]) -> list[str]:
    value, _ = _first_nested(
        profile,
        (
            "artifact.required_files",
            "artifacts.required_files",
            "required_files",
        ),
    )
    if value is _MISSING:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _remote_code_spec(
    profile: dict[str, Any],
) -> tuple[bool, str | None, dict[str, str]]:
    required_value, _ = _first_nested(
        profile,
        (
            "artifact.trust_remote_code_required",
            "artifacts.trust_remote_code_required",
            "runtime.trust_remote_code",
            "trust_remote_code",
        ),
    )
    if isinstance(required_value, dict):
        required = required_value.get("required") is True
    else:
        required = required_value is True
    policy, _ = _first_nested(
        profile,
        (
            "artifact.remote_code.policy",
            "artifacts.remote_code.policy",
            "remote_code.policy",
            "trust_remote_code.policy",
        ),
    )
    policy_value = policy.strip() if isinstance(policy, str) else None
    files: dict[str, str] = {}
    for field in (
        "artifact.remote_code.files",
        "artifact.remote_code.expected_files",
        "artifacts.remote_code.files",
        "remote_code.files",
        "remote_code.expected_files",
        "trust_remote_code.files",
    ):
        value = _nested(profile, field)
        if isinstance(value, dict):
            rows = [{"path": key, "sha256": item} for key, item in value.items()]
        elif isinstance(value, list):
            rows = value
        else:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            path = _safe_relative_path(row.get("path"))
            digest_value = row.get("sha256")
            if isinstance(digest_value, dict):
                digest_value = digest_value.get("sha256")
            digest = _normalize_hash(digest_value)
            if path is not None and digest is not None:
                files[path] = digest
    return required, policy_value, files


def _revision_candidates(
    model_dir: Path,
    reader: _ModelRootReader,
    blockers: list[str],
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    candidate_path = Path(os.path.abspath(model_dir))
    parts = candidate_path.parts
    for index, part in enumerate(parts[:-1]):
        if part != "snapshots":
            continue
        revision = _normalize_revision(parts[index + 1])
        if revision is not None:
            candidates.append(
                {"value": revision, "source": "hugging_face_snapshot_path"}
            )
            break

    ref_files: list[str] = []
    try:
        main_metadata = reader.lstat("refs/main")
    except FileNotFoundError:
        main_metadata = None
    except OSError as exc:
        blockers.append(f"Hugging Face ref refs/main cannot be inspected: {exc}")
        main_metadata = None
    if main_metadata is not None:
        if stat.S_ISREG(main_metadata.st_mode):
            ref_files.append("refs/main")
        else:
            blockers.append("Hugging Face ref refs/main must be a no-follow regular file")
    else:
        try:
            refs_metadata = reader.lstat("refs")
        except FileNotFoundError:
            refs_metadata = None
        except OSError as exc:
            blockers.append(f"Hugging Face refs directory cannot be inspected: {exc}")
            refs_metadata = None
        if refs_metadata is not None:
            if not stat.S_ISDIR(refs_metadata.st_mode):
                blockers.append(
                    "Hugging Face refs must be a no-follow directory"
                )
            else:
                pending = ["refs"]
                visited_entries = 0
                while pending and len(ref_files) <= 64:
                    current = pending.pop()
                    try:
                        names = sorted(reader.listdir(current, maximum=128))
                    except OSError as exc:
                        blockers.append(
                            f"Hugging Face refs directory {current} cannot be read: {exc}"
                        )
                        break
                    for name in names:
                        visited_entries += 1
                        if visited_entries > 256:
                            blockers.append(
                                "Hugging Face refs traversal exceeds the "
                                "256-entry inspection limit"
                            )
                            pending.clear()
                            break
                        relative = f"{current}/{name}"
                        try:
                            metadata = reader.lstat(relative)
                        except OSError as exc:
                            blockers.append(
                                f"Hugging Face ref {relative} cannot be inspected: {exc}"
                            )
                            continue
                        if stat.S_ISDIR(metadata.st_mode):
                            pending.append(relative)
                        elif stat.S_ISREG(metadata.st_mode):
                            ref_files.append(relative)
                        else:
                            blockers.append(
                                f"Hugging Face ref {relative} must not be a symbolic "
                                "link or special file"
                            )
                if len(ref_files) > 64:
                    blockers.append(
                        "Hugging Face refs directory exceeds the 64-file inspection limit"
                    )
                    ref_files = sorted(ref_files)[:64]
    for ref_file in sorted(ref_files):
        try:
            _size, _digest, raw_value, _metadata = reader.read_regular_file(
                ref_file,
                max_bytes=256,
                capture=True,
            )
            assert raw_value is not None
            ref_value = raw_value.decode("utf-8").strip()
        except (OSError, UnicodeError) as exc:
            blockers.append(f"Hugging Face ref {ref_file} cannot be read: {exc}")
            continue
        revision = _normalize_revision(ref_value)
        if revision is None:
            blockers.append(
                f"Hugging Face ref {ref_file} is not a resolved commit"
            )
        else:
            candidates.append(
                {
                    "value": revision,
                    "source": f"hugging_face_ref:{ref_file}",
                }
            )

    for marker_name in REVISION_MARKERS:
        try:
            marker_metadata = reader.lstat(marker_name)
        except FileNotFoundError:
            continue
        except OSError as exc:
            blockers.append(f"revision marker {marker_name} cannot be inspected: {exc}")
            continue
        if not stat.S_ISREG(marker_metadata.st_mode):
            blockers.append(
                f"revision marker {marker_name} must be a no-follow regular file"
            )
            continue
        try:
            _size, _digest, raw_value, _metadata = reader.read_regular_file(
                marker_name,
                max_bytes=256,
                capture=True,
            )
            assert raw_value is not None
            marker_value = raw_value.decode("utf-8").strip()
        except (OSError, UnicodeError) as exc:
            blockers.append(f"revision marker {marker_name} cannot be read: {exc}")
            continue
        revision = _normalize_revision(marker_value)
        if revision is None:
            blockers.append(
                f"revision marker {marker_name} must contain a 40- or 64-character lowercase hex commit"
            )
        else:
            candidates.append(
                {"value": revision, "source": f"marker:{marker_name}"}
            )
    deduplicated: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in candidates:
        key = (row["value"], row["source"])
        if key not in seen:
            deduplicated.append(row)
            seen.add(key)
    return deduplicated


def _local_metadata_revision_candidate(
    reader: _ModelRootReader,
    artifact_paths: Iterable[str],
    blockers: list[str],
) -> dict[str, str] | None:
    metadata_root = ".cache/huggingface/download"
    try:
        metadata_root_stat = reader.lstat(metadata_root)
    except FileNotFoundError:
        return None
    except OSError as exc:
        blockers.append(f"Hugging Face local metadata cannot be inspected: {exc}")
        return None
    if not stat.S_ISDIR(metadata_root_stat.st_mode):
        blockers.append(
            "Hugging Face local metadata root must be a no-follow directory"
        )
        return None
    paths = sorted(set(artifact_paths))
    if not paths:
        blockers.append(
            "Hugging Face local metadata cannot resolve a revision without selected artifacts"
        )
        return None
    if len(paths) > MAX_REPORTED_FILES:
        blockers.append(
            "Hugging Face local metadata coverage exceeds the bounded inspection "
            f"limit {MAX_REPORTED_FILES}"
        )
        return None
    missing: list[str] = []
    invalid: list[str] = []
    revisions: set[str] = set()
    now = time.time()
    for path in paths:
        metadata_path = f"{metadata_root}/{path}.metadata"
        try:
            metadata_stat = reader.lstat(metadata_path)
        except FileNotFoundError:
            missing.append(path)
            continue
        except OSError as exc:
            invalid.append(f"{path}: metadata cannot be inspected: {exc}")
            continue
        if not stat.S_ISREG(metadata_stat.st_mode):
            invalid.append(f"{path}: metadata must be a no-follow regular file")
            continue
        try:
            _size, _digest, raw_metadata, _metadata = reader.read_regular_file(
                metadata_path,
                max_bytes=4096,
                capture=True,
            )
            assert raw_metadata is not None
            lines = raw_metadata.decode("utf-8").splitlines()
            if len(lines) < 3:
                invalid.append(f"{path}: metadata must contain commit, etag, timestamp")
                continue
            revision = _normalize_revision(lines[0].strip())
            etag = lines[1].strip()
            timestamp = float(lines[2].strip())
            artifact_mtime = reader.stat_regular_file(path).st_mtime
        except (OSError, UnicodeError, ValueError) as exc:
            invalid.append(f"{path}: {exc}")
            continue
        if revision is None:
            invalid.append(f"{path}: unresolved metadata commit")
            continue
        if not etag:
            invalid.append(f"{path}: empty metadata etag")
            continue
        if timestamp > now + 300:
            invalid.append(f"{path}: metadata timestamp is implausibly in the future")
            continue
        if artifact_mtime - 1 > timestamp:
            invalid.append(f"{path}: metadata is older than the artifact file")
            continue
        revisions.add(revision)
    if missing:
        blockers.append(
            "Hugging Face local metadata is missing for selected artifacts: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    if invalid:
        blockers.append(
            "Hugging Face local metadata is invalid for selected artifacts: "
            + "; ".join(invalid[:10])
            + (" ..." if len(invalid) > 10 else "")
        )
    if len(revisions) > 1:
        blockers.append(
            "Hugging Face local metadata contains mixed commit revisions: "
            + ", ".join(sorted(revisions))
        )
    if missing or invalid or len(revisions) != 1:
        return None
    return {
        "value": next(iter(revisions)),
        "source": "hugging_face_local_dir_metadata",
    }


def inspect_model_artifacts(
    model_dir: str | Path,
    model_profile: dict[str, Any],
    *,
    catalog_sha256: str | None = None,
    profile_sha256: str | None = None,
    require_complete_hash_coverage: bool = False,
) -> dict[str, Any]:
    """Inspect a local Hugging Face model snapshot without network or tensor parsing.

    The function hashes the selected metadata, tokenizer, and safetensors files.
    It never imports model code, downloads artifacts, or reads safetensors headers
    or tensor payload structures.
    """

    root = Path(model_dir).expanduser()
    display_root = os.path.abspath(root)
    blockers: list[str] = []
    file_records: dict[str, dict[str, Any]] = {}
    captured_file_bytes: dict[str, bytes] = {}
    size_blocked_paths: set[str] = set()
    reader: _ModelRootReader | None = None
    top_level_names: list[str] = []
    try:
        reader = _ModelRootReader(root)
        top_level_names = sorted(reader.listdir(maximum=4096))
    except OSError as exc:
        blockers.append(
            "model_dir cannot be opened as a no-follow directory: "
            f"{display_root}: {exc}"
        )
    strict_hash_coverage = require_complete_hash_coverage is True
    if not isinstance(require_complete_hash_coverage, bool):
        blockers.append("require_complete_hash_coverage must be a boolean")
    lineage_values = {
        "catalog_sha256": catalog_sha256,
        "profile_sha256": profile_sha256,
    }
    normalized_lineage: dict[str, str | None] = {}
    for field, value in lineage_values.items():
        if value is None:
            normalized_lineage[field] = None
        elif isinstance(value, str) and PREFIXED_SHA256_RE.fullmatch(value):
            normalized_lineage[field] = value
        else:
            normalized_lineage[field] = None
            blockers.append(
                f"{field} must be sha256:<64 lowercase hex> when supplied"
            )
    if (catalog_sha256 is None) != (profile_sha256 is None):
        blockers.append("catalog_sha256 and profile_sha256 must be supplied together")
    if strict_hash_coverage and (
        normalized_lineage["catalog_sha256"] is None
        or normalized_lineage["profile_sha256"] is None
    ):
        blockers.append(
            "catalog_sha256 and profile_sha256 are required for strict inspection"
        )
    if not isinstance(model_profile, dict):
        blockers.append("model_profile must be a dictionary")
        profile: dict[str, Any] = {}
    else:
        profile = model_profile

    def add_file(
        raw_path: str,
        role: str,
        *,
        expected_size: int | None = None,
    ) -> dict[str, Any] | None:
        path = _safe_relative_path(raw_path)
        if path is None:
            blockers.append(f"artifact path is not safe and relative: {raw_path!r}")
            return None
        if path in size_blocked_paths:
            return None
        existing = file_records.get(path)
        if existing is not None:
            if role not in existing["roles"]:
                existing["roles"].append(role)
                existing["roles"].sort()
            return existing
        if len(file_records) >= MAX_REPORTED_FILES:
            blockers.append(
                f"selected artifact file count exceeds bounded report limit {MAX_REPORTED_FILES}"
            )
            return None
        if reader is None:
            blockers.append(f"missing required artifact file: {path}")
            return None
        capture = path == "config.json" or path.endswith(
            ".safetensors.index.json"
        )
        max_bytes = (
            MAX_CONFIG_BYTES
            if path == "config.json"
            else MAX_INDEX_BYTES
            if path.endswith(".safetensors.index.json")
            else None
        )
        try:
            size, digest, captured, _metadata = reader.read_regular_file(
                path,
                max_bytes=max_bytes,
                expected_size=expected_size,
                capture=capture,
            )
            record = {
                "path": path,
                "bytes": size,
                "sha256": digest,
                "roles": [role],
            }
            if captured is not None:
                captured_file_bytes[path] = captured
        except (OSError, ValueError) as exc:
            blockers.append(f"cannot read artifact file {path}: {exc}")
            return None
        file_records[path] = record
        return record

    expected_revision_value, expected_revision_path = _first_nested(
        profile,
        (
            "artifact.revision",
            "artifacts.revision",
            "hf_revision",
            "revision",
        ),
    )
    expected_revision: str | None = None
    expected_revision_valid = True
    if expected_revision_value is not _MISSING:
        expected_revision = _normalize_revision(expected_revision_value)
        if expected_revision is None:
            expected_revision_valid = False
            blockers.append(
                f"profile {expected_revision_path} must be a resolved 40- or "
                "64-character lowercase hex commit"
            )
    revision_candidates = (
        _revision_candidates(root, reader, blockers)
        if reader is not None
        else []
    )

    required_raw = _required_files(profile)
    required_safe: list[str] = []
    for raw_path in required_raw:
        safe_path = _safe_relative_path(raw_path)
        if safe_path is None:
            blockers.append(f"profile required file is not a safe relative path: {raw_path!r}")
            continue
        required_safe.append(safe_path)
        add_file(safe_path, "profile_required")

    config_record = add_file("config.json", "config")
    config_bytes = captured_file_bytes.get("config.json")
    config_data = (
        _parse_json_object(
            config_bytes,
            field="config.json",
            blockers=blockers,
        )
        if config_record is not None and config_bytes is not None
        else None
    )
    parsed_config = _parsed_config(config_data) if config_data is not None else {}
    config_report = {
        "present": config_record is not None and config_data is not None,
        "path": "config.json",
        "bytes": config_record.get("bytes") if config_record else None,
        "sha256": config_record.get("sha256") if config_record else None,
        "parsed": parsed_config,
    }

    profile_checks: list[dict[str, Any]] = []
    values = parsed_config.get("values", {}) if isinstance(parsed_config, dict) else {}
    config_keys = (
        parsed_config.get("config_keys", {}) if isinstance(parsed_config, dict) else {}
    )
    for field, expected in _profile_expectations(profile):
        if field == "family":
            actual = parsed_config.get("model_type")
            source = "model_type"
        elif field == "architectures":
            actual = parsed_config.get("architectures")
            source = "architectures"
        else:
            actual = values.get(field)
            source = config_keys.get(field)
        passed = actual == expected
        profile_checks.append(
            {
                "field": f"architecture.{field}",
                "expected": expected,
                "actual": actual,
                "config_key": source,
                "passed": passed,
            }
        )
        if not passed:
            blockers.append(
                f"profile/config mismatch for architecture.{field}: expected {expected!r}, got {actual!r}"
            )

    tokenizer_paths: list[str] = []
    if reader is not None:
        for name in top_level_names:
            try:
                metadata = reader.lstat(name)
            except OSError as exc:
                blockers.append(f"top-level artifact {name} cannot be inspected: {exc}")
                continue
            if not stat.S_ISREG(metadata.st_mode):
                if stat.S_ISLNK(metadata.st_mode) and (
                    name in TOKENIZER_NAMES
                    or name.startswith("tokenizer")
                    or name.startswith("chat_template")
                ):
                    blockers.append(
                        f"tokenizer artifact {name} must be a no-follow regular file"
                    )
                continue
            if (
                name in TOKENIZER_NAMES
                or name.startswith("tokenizer") and name.endswith((".json", ".model"))
                or name.startswith("chat_template") and name.endswith(".jinja")
            ):
                tokenizer_paths.append(name)
    tokenizer_paths = tokenizer_paths[:64]
    tokenizer_records: list[dict[str, Any]] = []
    for path in tokenizer_paths:
        record = add_file(path, "tokenizer")
        if record is not None:
            tokenizer_records.append(record)
    tokenizer_config_present = "tokenizer_config.json" in tokenizer_paths
    tokenizer_payload_present = bool(TOKENIZER_PAYLOAD_NAMES.intersection(tokenizer_paths))
    if "vocab.json" in tokenizer_paths and "merges.txt" in tokenizer_paths:
        tokenizer_payload_present = True
    if not tokenizer_config_present:
        blockers.append("missing tokenizer_config.json")
    if not tokenizer_payload_present:
        blockers.append(
            "missing tokenizer payload (tokenizer.json/model, sentencepiece, tiktoken/tekken, vocab.txt, or vocab.json+merges.txt)"
        )
    tokenizer_report = {
        "ready": tokenizer_config_present and tokenizer_payload_present,
        "tokenizer_config_present": tokenizer_config_present,
        "payload_present": tokenizer_payload_present,
        "files": sorted(record["path"] for record in tokenizer_records),
        "file_count": len(tokenizer_records),
        "exact_file_bytes": sum(record["bytes"] for record in tokenizer_records),
        "manifest_sha256": (
            _manifest_sha256(tokenizer_records) if tokenizer_records else None
        ),
    }

    index_paths: list[str] = []
    if reader is not None:
        for name in top_level_names:
            if not name.endswith(".safetensors.index.json"):
                continue
            try:
                metadata = reader.lstat(name)
            except OSError as exc:
                blockers.append(f"safetensors index {name} cannot be inspected: {exc}")
                continue
            if stat.S_ISREG(metadata.st_mode):
                index_paths.append(name)
            else:
                blockers.append(
                    f"safetensors index {name} must be a no-follow regular file"
                )
    index_paths.sort()
    if len(index_paths) > 1:
        blockers.append(
            "multiple top-level safetensors index files make the selected representation ambiguous"
        )
    index_path = index_paths[0] if index_paths else None
    index_record: dict[str, Any] | None = None
    index_data: dict[str, Any] | None = None
    shard_paths: list[str] = []
    tensor_entry_count = 0
    declared_tensor_bytes: int | None = None
    if index_path is not None:
        index_record = add_file(index_path, "safetensors_index")
        index_bytes = captured_file_bytes.get(index_path)
        if index_record is not None and index_bytes is not None:
            index_data = _parse_json_object(
                index_bytes,
                field=index_path,
                blockers=blockers,
            )
        if index_data is not None:
            weight_map = index_data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                blockers.append(f"{index_path}.weight_map must be a non-empty object")
            else:
                tensor_entry_count = len(weight_map)
                unsafe_values = [
                    value
                    for value in weight_map.values()
                    if _safe_relative_path(value) is None
                ]
                if unsafe_values:
                    blockers.append(
                        f"{index_path}.weight_map contains unsafe or non-string shard paths"
                    )
                shard_paths = sorted(
                    {
                        safe
                        for value in weight_map.values()
                        if (safe := _safe_relative_path(value)) is not None
                    }
                )
            metadata = index_data.get("metadata")
            if isinstance(metadata, dict) and _is_int(metadata.get("total_size")):
                declared_tensor_bytes = metadata["total_size"]
    else:
        shard_paths = []
        if reader is not None:
            for name in top_level_names:
                if not name.endswith(".safetensors"):
                    continue
                try:
                    metadata = reader.lstat(name)
                except OSError as exc:
                    blockers.append(
                        f"safetensors artifact {name} cannot be inspected: {exc}"
                    )
                    continue
                if stat.S_ISREG(metadata.st_mode):
                    shard_paths.append(name)
                else:
                    blockers.append(
                        f"safetensors artifact {name} must be a no-follow regular file"
                    )
            shard_paths.sort()
        if len(shard_paths) > 1:
            blockers.append(
                "multiple safetensors files require a safetensors index to define the shard set"
            )
    if not shard_paths:
        blockers.append("no safetensors weight file or indexed shard set found")
    if len(shard_paths) > MAX_REPORTED_FILES:
        blockers.append(
            f"safetensors shard count exceeds bounded report limit {MAX_REPORTED_FILES}"
        )
        shard_paths = shard_paths[:MAX_REPORTED_FILES]
    estimated_weight_bytes, _ = _first_nested(
        profile,
        (
            "artifact.estimated_weight_bytes",
            "artifacts.estimated_weight_bytes",
            "estimated_weight_bytes",
        ),
    )
    shard_preflight_sizes: dict[str, int] = {}
    if reader is not None:
        for path in shard_paths:
            if not path.endswith(".safetensors"):
                continue
            try:
                shard_preflight_sizes[path] = reader.stat_regular_file(path).st_size
            except OSError as exc:
                blockers.append(f"cannot preflight artifact file {path}: {exc}")
    preflight_weight_bytes = sum(shard_preflight_sizes.values())
    preflight_weight_size_matches = (
        estimated_weight_bytes is _MISSING
        or (
            _is_int(estimated_weight_bytes)
            and estimated_weight_bytes > 0
            and len(shard_preflight_sizes) == len(shard_paths)
            and preflight_weight_bytes == estimated_weight_bytes
        )
    )
    if estimated_weight_bytes is not _MISSING and not preflight_weight_size_matches:
        blockers.append(
            "profile/artifact mismatch for artifact.estimated_weight_bytes: "
            f"expected {estimated_weight_bytes!r}, got {preflight_weight_bytes!r} "
            "during no-follow size preflight"
        )
        size_blocked_paths.update(shard_paths)
    shard_records: list[dict[str, Any]] = []
    if preflight_weight_size_matches:
        for path in shard_paths:
            if not path.endswith(".safetensors"):
                blockers.append(f"index references a non-safetensors weight file: {path}")
                continue
            record = add_file(
                path,
                "safetensors_shard",
                expected_size=shard_preflight_sizes.get(path),
            )
            if record is not None:
                shard_records.append(record)
    top_level_shards: set[str] = set()
    if reader is not None:
        for name in top_level_names:
            if not name.endswith(".safetensors"):
                continue
            try:
                metadata = reader.lstat(name)
            except OSError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                top_level_shards.add(name)
            else:
                blockers.append(
                    f"safetensors artifact {name} must be a no-follow regular file"
                )
    indexed_top_level = {path for path in shard_paths if "/" not in path}
    unindexed = sorted(top_level_shards - indexed_top_level) if index_path else []
    if unindexed:
        blockers.append(
            "unindexed top-level safetensors files make the representation ambiguous: "
            + ", ".join(unindexed[:10])
        )
    weight_records = ([index_record] if index_record is not None else []) + shard_records
    weights_report = {
        "ready": bool(shard_records)
        and len(shard_records) == len(shard_paths)
        and not unindexed,
        "format": "safetensors",
        "index": index_path,
        "index_sha256": index_record.get("sha256") if index_record else None,
        "shards": [record["path"] for record in shard_records],
        "shard_count": len(shard_records),
        "tensor_entry_count": tensor_entry_count,
        "exact_shard_bytes": sum(record["bytes"] for record in shard_records),
        "exact_index_bytes": index_record.get("bytes", 0) if index_record else 0,
        "index_declared_tensor_bytes": declared_tensor_bytes,
        "manifest_sha256": (
            _manifest_sha256(weight_records) if weight_records else None
        ),
    }
    if estimated_weight_bytes is not _MISSING:
        actual_weight_bytes = weights_report["exact_shard_bytes"]
        weight_bytes_match = (
            _is_int(estimated_weight_bytes)
            and estimated_weight_bytes > 0
            and actual_weight_bytes == estimated_weight_bytes
        )
        profile_checks.append(
            {
                "field": "artifact.estimated_weight_bytes",
                "expected": estimated_weight_bytes,
                "actual": actual_weight_bytes,
                "config_key": None,
                "passed": weight_bytes_match,
            }
        )
        if not weight_bytes_match and preflight_weight_size_matches:
            blockers.append(
                "profile/artifact mismatch for artifact.estimated_weight_bytes: "
                f"expected {estimated_weight_bytes!r}, got {actual_weight_bytes!r}"
            )
    weight_format, _ = _first_nested(
        profile,
        ("artifact.weight_format", "artifacts.weight_format", "weight_format"),
    )
    if weight_format is not _MISSING and str(weight_format).lower() != "safetensors":
        blockers.append(
            f"profile weight_format {weight_format!r} is unsupported by this safetensors verifier"
        )

    expected_hashes = _expected_file_hashes(profile)
    for path, expected_hash in sorted(expected_hashes.items()):
        record = add_file(path, "profile_hash")
        if record is not None and record["sha256"] != expected_hash:
            blockers.append(
                f"sha256 mismatch for {path}: expected {expected_hash}, got {record['sha256']}"
            )

    remote_required, remote_policy, remote_hashes = _remote_code_spec(profile)
    remote_records: list[dict[str, Any]] = []
    discovered_code_files: list[str] = []
    remote_inventory_complete = not remote_required
    if remote_required:
        if (
            remote_policy is None
            or remote_policy.strip().lower() in UNRESOLVED_REMOTE_CODE_POLICIES
        ):
            blockers.append(
                "trust_remote_code is required but no explicit resolved review/allowlist policy is recorded"
            )
        if not remote_hashes:
            blockers.append(
                "trust_remote_code is required but no expected code files with sha256 hashes are recorded"
            )
        if reader is not None:
            discovered_code_files, inventory_errors = _inventory_python_files(
                reader
            )
            blockers.extend(inventory_errors)
            remote_inventory_complete = not inventory_errors
        required_code_files = {
            path for path in required_safe if path.endswith(".py")
        }
        missing_code_hashes = sorted(required_code_files - set(remote_hashes))
        if missing_code_hashes:
            blockers.append(
                "trust_remote_code required files lack pinned sha256 entries: "
                + ", ".join(missing_code_hashes)
            )
        non_python_hashes = sorted(
            path for path in remote_hashes if not path.endswith(".py")
        )
        for path in non_python_hashes:
            blockers.append(
                f"remote-code allowlist entry must name a Python file: {path}"
            )
        discovered_code_set = set(discovered_code_files)
        pinned_code_set = {
            path for path in remote_hashes if path.endswith(".py")
        }
        missing_discovered_hashes = sorted(
            discovered_code_set - pinned_code_set
        )
        if missing_discovered_hashes:
            blockers.append(
                "discovered remote-code Python files lack pinned sha256 "
                "entries: "
                + ", ".join(missing_discovered_hashes)
            )
        undiscovered_hashes = sorted(pinned_code_set - discovered_code_set)
        if undiscovered_hashes:
            blockers.append(
                "remote-code pinned Python files were not discovered as "
                "no-follow regular files: "
                + ", ".join(undiscovered_hashes)
            )
        for path in discovered_code_files:
            expected_hash = remote_hashes.get(path)
            record = add_file(path, "remote_code")
            if record is None:
                continue
            remote_records.append(record)
            if (
                expected_hash is not None
                and record["sha256"] != expected_hash
            ):
                blockers.append(
                    f"remote-code sha256 mismatch for {path}: "
                    f"expected {expected_hash}, got {record['sha256']}"
                )
    remote_code_report = {
        "required": remote_required,
        "ready": (
            not remote_required
            or (
                bool(remote_policy)
                and remote_policy.strip().lower()
                not in UNRESOLVED_REMOTE_CODE_POLICIES
                and bool(remote_hashes)
                and remote_inventory_complete
                and set(remote_hashes) == set(discovered_code_files)
                and len(remote_records) == len(discovered_code_files)
                and all(
                    record["sha256"] == remote_hashes.get(record["path"])
                    for record in remote_records
                )
            )
        ),
        "policy": remote_policy,
        "expected_files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(remote_hashes.items())
        ],
    }

    declared_file_hashes_value, _ = _first_nested(
        profile,
        ("artifact.file_hashes", "artifacts.file_hashes", "file_hashes"),
    )
    declared_hash_paths: set[str] = set()
    if declared_file_hashes_value is not _MISSING:
        if not isinstance(declared_file_hashes_value, dict) or not declared_file_hashes_value:
            blockers.append("profile artifact.file_hashes must be a non-empty object")
        else:
            for raw_path, raw_digest in declared_file_hashes_value.items():
                safe_path = _safe_relative_path(raw_path)
                if safe_path is None:
                    blockers.append(
                        f"profile artifact.file_hashes contains an unsafe path: {raw_path!r}"
                    )
                    continue
                if _normalize_hash(raw_digest) is None:
                    blockers.append(
                        "profile artifact.file_hashes contains an invalid SHA-256 "
                        f"for {safe_path}"
                    )
                    continue
                declared_hash_paths.add(safe_path)
    selected_file_paths = set(file_records)
    unpinned_selected_files = sorted(selected_file_paths - declared_hash_paths)
    unselected_expected_files = sorted(declared_hash_paths - selected_file_paths)
    if declared_file_hashes_value is not _MISSING:
        if unpinned_selected_files:
            blockers.append(
                "selected artifacts lack pinned profile SHA-256 values: "
                + ", ".join(unpinned_selected_files[:10])
                + (" ..." if len(unpinned_selected_files) > 10 else "")
            )
        if unselected_expected_files:
            blockers.append(
                "pinned profile SHA-256 entries were not selected or found: "
                + ", ".join(unselected_expected_files[:10])
                + (" ..." if len(unselected_expected_files) > 10 else "")
            )
    hash_coverage_complete = (
        declared_file_hashes_value is not _MISSING
        and bool(declared_hash_paths)
        and not unpinned_selected_files
        and not unselected_expected_files
    )
    if strict_hash_coverage and not hash_coverage_complete:
        blockers.append(
            "strict inspection requires complete pinned SHA-256 coverage for "
            "every selected artifact"
        )
    hash_coverage_report = {
        "required_by_profile": declared_file_hashes_value is not _MISSING,
        "required_by_inspection": strict_hash_coverage,
        "expected_file_count": len(declared_hash_paths),
        "selected_file_count": len(selected_file_paths),
        "unpinned_selected_files": unpinned_selected_files,
        "unselected_expected_files": unselected_expected_files,
        "complete": hash_coverage_complete,
    }

    files = sorted(file_records.values(), key=lambda row: row["path"])
    missing_required = sorted(set(required_safe) - set(file_records))
    for path in missing_required:
        message = f"missing required artifact file: {path}"
        if message not in blockers:
            blockers.append(message)
    if not revision_candidates and reader is not None:
        revision_artifact_paths = (
            set(file_records)
            | set(required_safe)
            | set(tokenizer_paths)
            | set(shard_paths)
        )
        if index_path is not None:
            revision_artifact_paths.add(index_path)
        metadata_candidate = _local_metadata_revision_candidate(
            reader,
            revision_artifact_paths,
            blockers,
        )
        if metadata_candidate is not None:
            revision_candidates.append(metadata_candidate)
    revision_values = sorted({row["value"] for row in revision_candidates})
    if not revision_values:
        blockers.append(
            "no locally observed Hugging Face revision found in snapshot path, "
            "revision marker, refs, or complete local-download metadata"
        )
    elif len(revision_values) > 1:
        blockers.append(
            "locally observed revision mismatch across snapshot path, markers, "
            "refs, or local-download metadata: "
            + ", ".join(revision_values)
        )
    observed_revision = revision_values[0] if len(revision_values) == 1 else None
    if (
        observed_revision is not None
        and expected_revision is not None
        and observed_revision != expected_revision
    ):
        blockers.append(
            "resolved revision mismatch between local artifact and model profile: "
            f"local {observed_revision}, expected {expected_revision}"
        )
    revision_resolved = (
        observed_revision is not None
        and expected_revision_valid
        and (
            expected_revision is None
            or observed_revision == expected_revision
        )
    )
    revision = {
        "resolved": revision_resolved,
        "value": observed_revision,
        "expected": expected_revision,
        "source": (
            revision_candidates[0]["source"]
            if observed_revision is not None and revision_candidates
            else None
        ),
        "candidates": revision_candidates,
    }
    profile_model_id, _ = _first_nested(profile, ("model_id", "artifact.model_id"))
    repository, _ = _first_nested(
        profile, ("artifact.repository", "artifacts.repository", "repository")
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "mode": REPORT_MODE,
        "model_dir": display_root,
        "catalog_sha256": normalized_lineage["catalog_sha256"],
        "profile_sha256": normalized_lineage["profile_sha256"],
        "profile_identity": {
            "model_id": None if profile_model_id is _MISSING else profile_model_id,
            "repository": None if repository is _MISSING else repository,
            "expected_revision": expected_revision,
        },
        "revision": revision,
        "config": config_report,
        "tokenizer": tokenizer_report,
        "weights": weights_report,
        "remote_code": remote_code_report,
        "hash_coverage": hash_coverage_report,
        "required_files": {
            "expected": sorted(set(required_safe)),
            "missing": missing_required,
        },
        "profile_checks": profile_checks,
        "files": files,
        "file_list_limit": MAX_REPORTED_FILES,
        "file_list_truncated": len(file_records) >= MAX_REPORTED_FILES,
        "artifact_manifest_sha256": (
            _manifest_sha256(files) if files else None
        ),
        "errors": list(dict.fromkeys(blockers)),
        "warnings": [],
    }
    report["ok"] = not report["errors"]
    report["summary"] = {
        "resolved_revision": revision["value"],
        "file_count": len(files),
        "exact_selected_file_bytes": sum(row["bytes"] for row in files),
        "tokenizer_file_count": tokenizer_report["file_count"],
        "safetensors_shard_count": weights_report["shard_count"],
        "exact_safetensors_shard_bytes": weights_report["exact_shard_bytes"],
        "profile_check_count": len(profile_checks),
        "profile_check_pass_count": sum(
            check["passed"] is True for check in profile_checks
        ),
        "expected_file_hash_count": hash_coverage_report["expected_file_count"],
        "file_hash_coverage_complete": hash_coverage_report["complete"],
        "blocking_issue_count": len(report["errors"]),
        "artifact_ready": report["ok"],
    }
    if reader is not None:
        reader.close()
    return report


def validate_model_artifact_report(
    report: dict[str, Any],
    *,
    expected_catalog_sha256: str | None = None,
    expected_profile_sha256: str | None = None,
    require_complete_hash_coverage: bool = False,
) -> dict[str, Any]:
    """Validate a serialized report without touching the model directory."""

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(report, dict):
        return {
            "ok": False,
            "errors": ["report must be a dictionary"],
            "warnings": [],
            "summary": {},
        }
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {REPORT_SCHEMA_VERSION}")
    if report.get("kind") != REPORT_KIND:
        errors.append(f"kind must be {REPORT_KIND}")
    if report.get("mode") != REPORT_MODE:
        errors.append(f"mode must be {REPORT_MODE}")
    strict_hash_coverage = require_complete_hash_coverage is True
    if not isinstance(require_complete_hash_coverage, bool):
        errors.append("require_complete_hash_coverage must be a boolean")
    expected_lineage = {
        "catalog_sha256": expected_catalog_sha256,
        "profile_sha256": expected_profile_sha256,
    }
    if (expected_catalog_sha256 is None) != (expected_profile_sha256 is None):
        errors.append(
            "expected_catalog_sha256 and expected_profile_sha256 must be supplied together"
        )
    for field, expected in expected_lineage.items():
        actual = report.get(field)
        if actual is not None and (
            not isinstance(actual, str)
            or PREFIXED_SHA256_RE.fullmatch(actual) is None
        ):
            errors.append(f"{field} must be sha256:<64 lowercase hex> or null")
        if expected is not None:
            if (
                not isinstance(expected, str)
                or PREFIXED_SHA256_RE.fullmatch(expected) is None
            ):
                errors.append(
                    f"expected_{field} must be sha256:<64 lowercase hex>"
                )
            elif actual != expected:
                errors.append(f"{field} does not match expected_{field}")
    if (report.get("catalog_sha256") is None) != (
        report.get("profile_sha256") is None
    ):
        errors.append(
            "catalog_sha256 and profile_sha256 must both be present or both be null"
        )
    if strict_hash_coverage and (
        report.get("catalog_sha256") is None
        or report.get("profile_sha256") is None
    ):
        errors.append(
            "catalog_sha256 and profile_sha256 are required for strict validation"
        )

    report_errors = report.get("errors")
    if not isinstance(report_errors, list) or not all(
        isinstance(item, str) and item for item in report_errors
    ):
        errors.append("errors must be a list of non-empty strings")
        report_errors = []
    elif report_errors:
        errors.extend(f"inspection blocker: {item}" for item in report_errors)
    report_warnings = report.get("warnings")
    if not isinstance(report_warnings, list) or not all(
        isinstance(item, str) and item for item in report_warnings
    ):
        errors.append("warnings must be a list of non-empty strings")
    else:
        warnings.extend(report_warnings)

    revision = report.get("revision")
    if not isinstance(revision, dict):
        errors.append("revision must be an object")
        revision = {}
    if revision.get("resolved") is not True:
        errors.append("revision.resolved must be true")
    if _normalize_revision(revision.get("value")) is None:
        errors.append("revision.value must be a resolved 40- or 64-character commit")
    profile_identity = report.get("profile_identity")
    if not isinstance(profile_identity, dict):
        errors.append("profile_identity must be an object")
        profile_identity = {}
    model_id = profile_identity.get("model_id")
    if model_id is not None and (not isinstance(model_id, str) or not model_id):
        errors.append("profile_identity.model_id must be a non-empty string or null")
    repository = profile_identity.get("repository")
    if repository is not None and (
        not isinstance(repository, str)
        or repository.count("/") != 1
        or any(not part for part in repository.split("/"))
    ):
        errors.append("profile_identity.repository must be owner/name or null")
    if profile_identity.get("expected_revision") != revision.get("expected"):
        errors.append(
            "profile_identity.expected_revision does not match revision.expected"
        )

    files = report.get("files")
    if not isinstance(files, list):
        errors.append("files must be a list")
        files = []
    if len(files) > MAX_REPORTED_FILES:
        errors.append(f"files exceeds bounded limit {MAX_REPORTED_FILES}")
    seen_paths: set[str] = set()
    file_bytes = 0
    valid_file_rows: list[dict[str, Any]] = []
    for index, row in enumerate(files):
        field = f"files[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{field} must be an object")
            continue
        path = _safe_relative_path(row.get("path"))
        if path is None:
            errors.append(f"{field}.path must be safe and relative")
        elif path in seen_paths:
            errors.append(f"{field}.path is duplicated: {path}")
        else:
            seen_paths.add(path)
        size = row.get("bytes")
        if not _is_int(size) or size < 0:
            errors.append(f"{field}.bytes must be a non-negative integer")
        else:
            file_bytes += size
        if _normalize_hash(row.get("sha256")) is None:
            errors.append(f"{field}.sha256 must be sha256:<64 lowercase hex>")
        roles = row.get("roles")
        if not isinstance(roles, list) or not roles or not all(
            isinstance(role, str) and role for role in roles
        ):
            errors.append(f"{field}.roles must be a non-empty string list")
        if (
            path is not None
            and _is_int(size)
            and size >= 0
            and _normalize_hash(row.get("sha256")) is not None
            and isinstance(roles, list)
            and bool(roles)
        ):
            valid_file_rows.append(row)
    file_by_path = {
        row["path"]: row
        for row in valid_file_rows
        if isinstance(row.get("path"), str)
    }
    artifact_manifest = _normalize_hash(report.get("artifact_manifest_sha256"))
    if artifact_manifest is None:
        errors.append(
            "artifact_manifest_sha256 must be sha256:<64 lowercase hex>"
        )
    elif artifact_manifest != _manifest_sha256(valid_file_rows):
        errors.append("artifact_manifest_sha256 does not match files")

    config = report.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
        config = {}
    if config.get("present") is not True:
        errors.append("config.present must be true")
    if config.get("path") != "config.json":
        errors.append("config.path must be config.json")
    if _normalize_hash(config.get("sha256")) is None:
        errors.append("config.sha256 must be sha256:<64 lowercase hex>")
    if not isinstance(config.get("parsed"), dict):
        errors.append("config.parsed must be an object")
    config_file = file_by_path.get("config.json")
    if config_file is None:
        errors.append("files must contain config.json")
    else:
        if config.get("bytes") != config_file.get("bytes"):
            errors.append("config.bytes does not match files")
        if config.get("sha256") != config_file.get("sha256"):
            errors.append("config.sha256 does not match files")

    tokenizer = report.get("tokenizer")
    if not isinstance(tokenizer, dict):
        errors.append("tokenizer must be an object")
        tokenizer = {}
    if tokenizer.get("ready") is not True:
        errors.append("tokenizer.ready must be true")
    if tokenizer.get("tokenizer_config_present") is not True:
        errors.append("tokenizer.tokenizer_config_present must be true")
    if tokenizer.get("payload_present") is not True:
        errors.append("tokenizer.payload_present must be true")
    if _normalize_hash(tokenizer.get("manifest_sha256")) is None:
        errors.append("tokenizer.manifest_sha256 must be sha256:<64 lowercase hex>")
    tokenizer_rows = [
        row
        for row in valid_file_rows
        if "tokenizer" in row.get("roles", [])
    ]
    tokenizer_paths = sorted(row["path"] for row in tokenizer_rows)
    if tokenizer.get("files") != tokenizer_paths:
        errors.append("tokenizer.files does not match files with tokenizer role")
    if tokenizer.get("file_count") != len(tokenizer_rows):
        errors.append("tokenizer.file_count does not match files")
    if tokenizer.get("exact_file_bytes") != sum(
        row["bytes"] for row in tokenizer_rows
    ):
        errors.append("tokenizer.exact_file_bytes does not match files")
    if tokenizer_rows and tokenizer.get("manifest_sha256") != _manifest_sha256(
        tokenizer_rows
    ):
        errors.append("tokenizer.manifest_sha256 does not match files")

    weights = report.get("weights")
    if not isinstance(weights, dict):
        errors.append("weights must be an object")
        weights = {}
    if weights.get("ready") is not True:
        errors.append("weights.ready must be true")
    if weights.get("format") != "safetensors":
        errors.append("weights.format must be safetensors")
    if not _is_int(weights.get("shard_count")) or weights.get("shard_count", 0) <= 0:
        errors.append("weights.shard_count must be a positive integer")
    if (
        not _is_int(weights.get("exact_shard_bytes"))
        or weights.get("exact_shard_bytes", 0) <= 0
    ):
        errors.append("weights.exact_shard_bytes must be a positive integer")
    if _normalize_hash(weights.get("manifest_sha256")) is None:
        errors.append("weights.manifest_sha256 must be sha256:<64 lowercase hex>")
    shard_rows = [
        row
        for row in valid_file_rows
        if "safetensors_shard" in row.get("roles", [])
    ]
    index_rows = [
        row
        for row in valid_file_rows
        if "safetensors_index" in row.get("roles", [])
    ]
    if weights.get("shards") != [row["path"] for row in shard_rows]:
        errors.append("weights.shards does not match files with safetensors_shard role")
    if weights.get("shard_count") != len(shard_rows):
        errors.append("weights.shard_count does not match files")
    if weights.get("exact_shard_bytes") != sum(row["bytes"] for row in shard_rows):
        errors.append("weights.exact_shard_bytes does not match files")
    if weights.get("exact_index_bytes") != sum(row["bytes"] for row in index_rows):
        errors.append("weights.exact_index_bytes does not match files")
    weight_rows = index_rows + shard_rows
    if weight_rows and weights.get("manifest_sha256") != _manifest_sha256(weight_rows):
        errors.append("weights.manifest_sha256 does not match files")
    if weights.get("index") is None and index_rows:
        errors.append("weights.index is missing despite a safetensors index file")
    if weights.get("index") is not None:
        if len(index_rows) != 1 or index_rows[0]["path"] != weights.get("index"):
            errors.append("weights.index does not match files")
        elif weights.get("index_sha256") != index_rows[0]["sha256"]:
            errors.append("weights.index_sha256 does not match files")

    required = report.get("required_files")
    if not isinstance(required, dict):
        errors.append("required_files must be an object")
    elif required.get("missing") != []:
        errors.append("required_files.missing must be empty")
    else:
        expected_required = required.get("expected")
        if not isinstance(expected_required, list) or not all(
            _safe_relative_path(path) is not None for path in expected_required
        ):
            errors.append("required_files.expected must be a safe relative path list")
        elif not set(expected_required).issubset(file_by_path):
            errors.append("required_files.expected contains files absent from files")

    checks = report.get("profile_checks")
    if not isinstance(checks, list):
        errors.append("profile_checks must be a list")
        checks = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"profile_checks[{index}] must be an object")
        elif check.get("passed") is not True:
            errors.append(f"profile_checks[{index}].passed must be true")

    remote_code = report.get("remote_code")
    if not isinstance(remote_code, dict):
        errors.append("remote_code must be an object")
    elif remote_code.get("required") is True and remote_code.get("ready") is not True:
        errors.append("remote_code.ready must be true when remote code is required")
    elif remote_code.get("required") is True:
        policy = remote_code.get("policy")
        if (
            not isinstance(policy, str)
            or policy.strip().lower() in UNRESOLVED_REMOTE_CODE_POLICIES
        ):
            errors.append("remote_code.policy must be explicitly resolved")
        expected_code = remote_code.get("expected_files")
        if not isinstance(expected_code, list) or not expected_code:
            errors.append("remote_code.expected_files must be non-empty")
        else:
            for index, row in enumerate(expected_code):
                if not isinstance(row, dict):
                    errors.append(f"remote_code.expected_files[{index}] must be an object")
                    continue
                path = _safe_relative_path(row.get("path"))
                digest = _normalize_hash(row.get("sha256"))
                file_row = file_by_path.get(path) if path is not None else None
                if path is None or not path.endswith(".py"):
                    errors.append(
                        f"remote_code.expected_files[{index}].path must name a safe Python file"
                    )
                if digest is None:
                    errors.append(
                        f"remote_code.expected_files[{index}].sha256 must be a sha256 hash"
                    )
                if file_row is None or "remote_code" not in file_row.get("roles", []):
                    errors.append(
                        f"remote_code.expected_files[{index}] is absent from remote-code files"
                    )
                elif digest != file_row.get("sha256"):
                    errors.append(
                        f"remote_code.expected_files[{index}].sha256 does not match files"
                    )

    hash_coverage = report.get("hash_coverage")
    if not isinstance(hash_coverage, dict):
        errors.append("hash_coverage must be an object")
        hash_coverage = {}
    hash_coverage_required = hash_coverage.get("required_by_profile")
    if not isinstance(hash_coverage_required, bool):
        errors.append("hash_coverage.required_by_profile must be a boolean")
    hash_coverage_required_by_inspection = hash_coverage.get(
        "required_by_inspection"
    )
    if not isinstance(hash_coverage_required_by_inspection, bool):
        errors.append("hash_coverage.required_by_inspection must be a boolean")
    if strict_hash_coverage and hash_coverage_required_by_inspection is not True:
        errors.append(
            "hash_coverage.required_by_inspection must be true for strict validation"
        )
    expected_file_count = hash_coverage.get("expected_file_count")
    if not _is_int(expected_file_count) or expected_file_count < 0:
        errors.append(
            "hash_coverage.expected_file_count must be a non-negative integer"
        )
    if hash_coverage.get("selected_file_count") != len(files):
        errors.append("hash_coverage.selected_file_count does not match files")
    for field in ("unpinned_selected_files", "unselected_expected_files"):
        value = hash_coverage.get(field)
        if not isinstance(value, list) or not all(
            _safe_relative_path(path) is not None for path in value
        ):
            errors.append(f"hash_coverage.{field} must be a safe relative path list")
    coverage_is_required = (
        hash_coverage_required is True
        or hash_coverage_required_by_inspection is True
        or strict_hash_coverage
    )
    if coverage_is_required:
        if hash_coverage.get("complete") is not True:
            errors.append("hash_coverage.complete must be true when required")
        if hash_coverage.get("expected_file_count") != len(files):
            errors.append("hash_coverage.expected_file_count does not match files")
        if hash_coverage.get("unpinned_selected_files") != []:
            errors.append("hash_coverage.unpinned_selected_files must be empty")
        if hash_coverage.get("unselected_expected_files") != []:
            errors.append("hash_coverage.unselected_expected_files must be empty")
    elif hash_coverage.get("complete") is not False:
        errors.append("hash_coverage.complete must be false when not required")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("file_count") != len(files):
        errors.append("summary.file_count does not match files")
    if summary.get("exact_selected_file_bytes") != file_bytes:
        errors.append("summary.exact_selected_file_bytes does not match files")
    if summary.get("blocking_issue_count") != len(report_errors):
        errors.append("summary.blocking_issue_count does not match errors")
    if summary.get("expected_file_hash_count") != hash_coverage.get(
        "expected_file_count"
    ):
        errors.append("summary.expected_file_hash_count does not match hash_coverage")
    if summary.get("file_hash_coverage_complete") is not hash_coverage.get(
        "complete"
    ):
        errors.append(
            "summary.file_hash_coverage_complete does not match hash_coverage"
        )
    expected_ok = not report_errors
    if report.get("ok") is not expected_ok:
        errors.append("ok does not match the presence of inspection blockers")
    if summary.get("artifact_ready") is not expected_ok:
        errors.append("summary.artifact_ready does not match inspection blockers")
    return {
        "ok": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "summary": {
            "report_ok": report.get("ok") is True,
            "file_count": len(files),
            "exact_selected_file_bytes": file_bytes,
            "profile_check_count": len(checks),
            "inspection_blocker_count": len(report_errors),
        },
    }
