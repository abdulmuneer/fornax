"""Offline qualification catalog and deterministic recipe composition.

The catalog is deliberately a C1 contract artifact.  Vendor specifications and
capacity arithmetic are inputs to future qualification; they are not observed
performance, physical backend evidence, or a hardware support claim.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .recipe_packet import (
    MANAGED_PACKET_FILENAMES,
    PACKET_MANIFEST_FILENAME,
    build_recipe_packet_manifest,
    verify_recipe_packet,
)


CATALOG_SCHEMA_VERSION = 1
CATALOG_KIND = "fornax_qualification_catalog"
MODEL_KIND = "fornax_model_qualification_profile"
PLATFORM_KIND = "fornax_platform_qualification_profile"
LOCK_KIND = "fornax_qualification_recipe_lock"
COMMANDS_KIND = "fornax_qualification_recipe_commands"

REQUIRED_OPERATIONS = (
    "attention",
    "dense_mlp",
    "router_topk",
    "expert_gemm_mlp",
    "collect_scatter_gather",
    "kv_operations",
    "sampling_logits",
    "serialization_pack_gather",
    "transport",
)
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

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class QualificationCatalogError(ValueError):
    """Raised when a catalog, profile, or recipe violates its contract."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for qualification hashes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise QualificationCatalogError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Return a prefixed SHA-256 over :func:`canonical_json` UTF-8 bytes."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def qualification_catalog_root() -> Path:
    """Return the package-owned default catalog directory."""

    return Path(__file__).with_name("qualification_catalog")


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationCatalogError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise QualificationCatalogError(f"non-finite JSON number is forbidden: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_constant,
        )
    except QualificationCatalogError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationCatalogError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationCatalogError(f"{path}: top level must be an object")
    return value


def _object(
    value: Any,
    field: str,
    keys: Iterable[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualificationCatalogError(f"{field} must be an object")
    expected = set(keys)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise QualificationCatalogError(f"{field} missing keys: {', '.join(missing)}")
    if unknown:
        raise QualificationCatalogError(f"{field} has unknown keys: {', '.join(unknown)}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationCatalogError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualificationCatalogError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise QualificationCatalogError(f"{field} must be a boolean")
    return value


def _strings(value: Any, field: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = "non-empty " if nonempty else ""
        raise QualificationCatalogError(f"{field} must be a {suffix}list")
    result = [_text(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise QualificationCatalogError(f"{field} must not contain duplicates")
    return result


def _artifact_file_hashes(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise QualificationCatalogError(f"{field} must be a non-empty object")
    if len(value) > 512:
        raise QualificationCatalogError(f"{field} must contain at most 512 files")
    result: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        path = _text(raw_path, f"{field} key")
        parts = path.split("/")
        if (
            "\\" in path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise QualificationCatalogError(
                f"{field} paths must be safe relative POSIX paths"
            )
        digest = _text(raw_digest, f"{field}.{path}")
        if not _SHA256_RE.fullmatch(digest):
            raise QualificationCatalogError(
                f"{field}.{path} must be sha256:<64 lowercase hex>"
            )
        result[path] = digest
    return result


def _slug(value: Any, field: str) -> str:
    text = _text(value, field)
    if not _SLUG_RE.fullmatch(text):
        raise QualificationCatalogError(f"{field} must be a lowercase kebab-case identifier")
    return text


def _sources(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise QualificationCatalogError(f"{field} must be a non-empty list")
    source_ids: set[str] = set()
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        item = _object(
            raw,
            item_field,
            {"source_id", "url", "retrieved_date", "supports"},
        )
        source_id = _slug(item["source_id"], f"{item_field}.source_id")
        if source_id in source_ids:
            raise QualificationCatalogError(f"{field} has duplicate source_id {source_id}")
        source_ids.add(source_id)
        url = _text(item["url"], f"{item_field}.url")
        if not url.startswith("https://"):
            raise QualificationCatalogError(f"{item_field}.url must use https")
        date = _text(item["retrieved_date"], f"{item_field}.retrieved_date")
        if not _DATE_RE.fullmatch(date):
            raise QualificationCatalogError(
                f"{item_field}.retrieved_date must be YYYY-MM-DD"
            )
        _strings(item["supports"], f"{item_field}.supports")
    return source_ids


def _provenance(value: Any, field: str, source_ids: set[str]) -> None:
    if not isinstance(value, list) or not value:
        raise QualificationCatalogError(f"{field} must be a non-empty list")
    seen_fields: set[str] = set()
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        item = _object(
            raw,
            item_field,
            {"field", "status", "calibration_status", "source_ids"},
        )
        path = _text(item["field"], f"{item_field}.field")
        if path in seen_fields:
            raise QualificationCatalogError(f"{field} has duplicate field {path}")
        seen_fields.add(path)
        _text(item["status"], f"{item_field}.status")
        calibration = _text(
            item["calibration_status"], f"{item_field}.calibration_status"
        )
        if calibration not in {"uncalibrated", "not_applicable"}:
            raise QualificationCatalogError(
                f"{item_field}.calibration_status must be uncalibrated or not_applicable"
            )
        refs = _strings(item["source_ids"], f"{item_field}.source_ids")
        unknown = sorted(set(refs) - source_ids)
        if unknown:
            raise QualificationCatalogError(
                f"{item_field}.source_ids reference unknown sources: {', '.join(unknown)}"
            )


def _validate_qualification(value: Any, field: str, *, model: bool) -> None:
    if model:
        item = _object(
            value,
            field,
            {
                "maturity",
                "support_state",
                "artifact_hash_status",
                "tokenizer_hash_status",
                "chat_template_hash_status",
                "parity_status",
                "unresolved_evidence",
            },
        )
        expected = {
            "artifact_hash_status": "pinned_expected_sha256_unobserved",
            "tokenizer_hash_status": "pinned_expected_sha256_unobserved",
            "chat_template_hash_status": "pinned_expected_sha256_unobserved",
            "parity_status": "not_run",
        }
    else:
        item = _object(
            value,
            field,
            {
                "maturity",
                "support_state",
                "hardware_validation_status",
                "runtime_validation_status",
                "unresolved_evidence",
            },
        )
        expected = {
            "hardware_validation_status": "not_run",
            "runtime_validation_status": "not_run",
        }
    if item["maturity"] != "C1_contracted":
        raise QualificationCatalogError(f"{field}.maturity must be C1_contracted")
    if item["support_state"] != "contract_validated":
        raise QualificationCatalogError(
            f"{field}.support_state must be contract_validated"
        )
    for key, expected_value in expected.items():
        if item[key] != expected_value:
            raise QualificationCatalogError(
                f"{field}.{key} must be {expected_value}"
            )
    _strings(item["unresolved_evidence"], f"{field}.unresolved_evidence")


def _validate_model(data: dict[str, Any], path: Path) -> str:
    item = _object(
        data,
        str(path),
        {
            "schema_version",
            "kind",
            "model_id",
            "display_name",
            "selection",
            "artifact",
            "architecture",
            "runtime",
            "qualification",
            "sources",
            "provenance",
        },
    )
    if item["schema_version"] != CATALOG_SCHEMA_VERSION or item["kind"] != MODEL_KIND:
        raise QualificationCatalogError(f"{path}: unsupported model profile schema")
    model_id = _slug(item["model_id"], f"{path}.model_id")
    if path.stem != model_id:
        raise QualificationCatalogError(f"{path}: filename must match model_id")
    _text(item["display_name"], f"{path}.display_name")
    selection = _object(
        item["selection"],
        f"{path}.selection",
        {
            "snapshot_date",
            "popularity_proxy",
            "downloads_last_month",
            "likes",
            "cohort_policy",
            "selection_reason",
        },
    )
    if not _DATE_RE.fullmatch(
        _text(selection["snapshot_date"], f"{path}.selection.snapshot_date")
    ):
        raise QualificationCatalogError(
            f"{path}.selection.snapshot_date must be YYYY-MM-DD"
        )
    if selection["popularity_proxy"] != "hugging_face_downloads_last_month":
        raise QualificationCatalogError(
            f"{path}.selection.popularity_proxy must be "
            "hugging_face_downloads_last_month"
        )
    _integer(
        selection["downloads_last_month"],
        f"{path}.selection.downloads_last_month",
        minimum=0,
    )
    _integer(selection["likes"], f"{path}.selection.likes", minimum=0)
    _text(selection["cohort_policy"], f"{path}.selection.cohort_policy")
    _text(selection["selection_reason"], f"{path}.selection.selection_reason")

    artifact = _object(
        item["artifact"],
        f"{path}.artifact",
        {
            "provider",
            "repository",
            "revision",
            "revision_kind",
            "weight_format",
            "weight_dtype",
            "quantization",
            "estimated_weight_bytes",
            "estimate_basis",
            "license_spdx",
            "trust_remote_code_required",
            "required_files",
            "download_exclude_patterns",
            "file_hashes",
        },
    )
    if artifact["provider"] != "hugging_face":
        raise QualificationCatalogError(f"{path}.artifact.provider must be hugging_face")
    repository = _text(artifact["repository"], f"{path}.artifact.repository")
    if repository.count("/") != 1:
        raise QualificationCatalogError(
            f"{path}.artifact.repository must be owner/name"
        )
    revision = _text(artifact["revision"], f"{path}.artifact.revision")
    if not _REVISION_RE.fullmatch(revision):
        raise QualificationCatalogError(
            f"{path}.artifact.revision must be a 40-character lowercase commit SHA"
        )
    if artifact["revision_kind"] != "git_commit":
        raise QualificationCatalogError(
            f"{path}.artifact.revision_kind must be git_commit"
        )
    if artifact["weight_format"] != "safetensors":
        raise QualificationCatalogError(
            f"{path}.artifact.weight_format must be safetensors"
        )
    dtype = _text(artifact["weight_dtype"], f"{path}.artifact.weight_dtype")
    quantization = _text(artifact["quantization"], f"{path}.artifact.quantization")
    dtype_quantization = {
        "bf16": "none",
        "mxfp4": "mxfp4",
        "fp8_e4m3": "fp8",
    }
    if dtype not in dtype_quantization or dtype_quantization[dtype] != quantization:
        raise QualificationCatalogError(
            f"{path}.artifact weight dtype/quantization pair is unsupported"
        )
    _integer(artifact["estimated_weight_bytes"], f"{path}.artifact.estimated_weight_bytes")
    _text(artifact["estimate_basis"], f"{path}.artifact.estimate_basis")
    _text(artifact["license_spdx"], f"{path}.artifact.license_spdx")
    _boolean(
        artifact["trust_remote_code_required"],
        f"{path}.artifact.trust_remote_code_required",
    )
    required_files = _strings(
        artifact["required_files"], f"{path}.artifact.required_files"
    )
    for required_file in required_files:
        parts = required_file.split("/")
        if (
            "\\" in required_file
            or required_file.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise QualificationCatalogError(
                f"{path}.artifact.required_files must contain safe relative POSIX paths"
            )
    for required in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        if required not in required_files:
            raise QualificationCatalogError(
                f"{path}.artifact.required_files must include {required}"
            )
    file_hashes = _artifact_file_hashes(
        artifact["file_hashes"],
        f"{path}.artifact.file_hashes",
    )
    missing_required_hashes = sorted(set(required_files) - set(file_hashes))
    if missing_required_hashes:
        raise QualificationCatalogError(
            f"{path}.artifact.file_hashes missing required files: "
            + ", ".join(missing_required_hashes)
        )
    if not any(name.endswith(".safetensors") for name in file_hashes):
        raise QualificationCatalogError(
            f"{path}.artifact.file_hashes must include safetensors shards"
        )
    exclude_patterns = _strings(
        artifact["download_exclude_patterns"],
        f"{path}.artifact.download_exclude_patterns",
        nonempty=False,
    )
    if any(pattern.startswith("/") or ".." in pattern.split("/") for pattern in exclude_patterns):
        raise QualificationCatalogError(
            f"{path}.artifact.download_exclude_patterns must be relative glob patterns"
        )
    excluded_hashes = sorted(
        file_path
        for file_path in file_hashes
        if any(fnmatch.fnmatchcase(file_path, pattern) for pattern in exclude_patterns)
    )
    if excluded_hashes:
        raise QualificationCatalogError(
            f"{path}.artifact.file_hashes include excluded files: "
            + ", ".join(excluded_hashes)
        )

    architecture = _object(
        item["architecture"],
        f"{path}.architecture",
        {
            "family",
            "decoder_layers",
            "hidden_size",
            "total_experts",
            "active_experts_per_token",
            "config_max_position_embeddings",
            "publisher_supported_context_tokens",
            "context_note",
            "total_parameters",
            "active_parameters",
        },
    )
    _text(architecture["family"], f"{path}.architecture.family")
    _integer(architecture["decoder_layers"], f"{path}.architecture.decoder_layers")
    _integer(architecture["hidden_size"], f"{path}.architecture.hidden_size")
    experts = _integer(architecture["total_experts"], f"{path}.architecture.total_experts")
    active_experts = _integer(
        architecture["active_experts_per_token"],
        f"{path}.architecture.active_experts_per_token",
    )
    if active_experts > experts:
        raise QualificationCatalogError(
            f"{path}.architecture.active_experts_per_token exceeds total_experts"
        )
    config_context = _integer(
        architecture["config_max_position_embeddings"],
        f"{path}.architecture.config_max_position_embeddings",
    )
    publisher_context = _optional_integer(
        architecture["publisher_supported_context_tokens"],
        f"{path}.architecture.publisher_supported_context_tokens",
    )
    _text(architecture["context_note"], f"{path}.architecture.context_note")
    _optional_integer(architecture["total_parameters"], f"{path}.architecture.total_parameters")
    _optional_integer(
        architecture["active_parameters"], f"{path}.architecture.active_parameters"
    )

    runtime = _object(
        item["runtime"],
        f"{path}.runtime",
        {
            "candidate_backend",
            "max_architecture",
            "max_weight_encoding",
            "activation_dtype_candidates",
            "kv_dtype_candidates",
            "required_operations",
            "prompt_format",
            "qualification_context_tokens",
            "model_path_placeholder",
        },
    )
    if runtime["candidate_backend"] != "max":
        raise QualificationCatalogError(f"{path}.runtime.candidate_backend must be max")
    expected_max_contracts = {
        "deepseek-r1": ("DeepseekV3ForCausalLM", "float8_e4m3fn"),
        "gpt-oss-120b": ("GptOssForCausalLM", "float4_e2m1fnx2"),
        "qwen3-30b-a3b": ("Qwen3MoeForCausalLM", "bfloat16"),
    }
    max_architecture = _text(
        runtime["max_architecture"],
        f"{path}.runtime.max_architecture",
    )
    max_weight_encoding = _text(
        runtime["max_weight_encoding"],
        f"{path}.runtime.max_weight_encoding",
    )
    if (max_architecture, max_weight_encoding) != expected_max_contracts[model_id]:
        expected_architecture, expected_encoding = expected_max_contracts[model_id]
        raise QualificationCatalogError(
            f"{path}.runtime MAX contract must be "
            f"{expected_architecture}/{expected_encoding}"
        )
    _strings(runtime["activation_dtype_candidates"], f"{path}.runtime.activation_dtype_candidates")
    _strings(runtime["kv_dtype_candidates"], f"{path}.runtime.kv_dtype_candidates")
    operations = _strings(runtime["required_operations"], f"{path}.runtime.required_operations")
    if tuple(operations) != REQUIRED_OPERATIONS:
        raise QualificationCatalogError(
            f"{path}.runtime.required_operations must equal the qualification operation contract"
        )
    _text(runtime["prompt_format"], f"{path}.runtime.prompt_format")
    qualification_context = _integer(
        runtime["qualification_context_tokens"],
        f"{path}.runtime.qualification_context_tokens",
    )
    if qualification_context > config_context or (
        publisher_context is not None and qualification_context > publisher_context
    ):
        raise QualificationCatalogError(
            f"{path}.runtime.qualification_context_tokens exceeds a declared model limit"
        )
    if runtime["model_path_placeholder"] != "<MODEL_DIR>":
        raise QualificationCatalogError(
            f"{path}.runtime.model_path_placeholder must be <MODEL_DIR>"
        )
    _validate_qualification(item["qualification"], f"{path}.qualification", model=True)
    source_ids = _sources(item["sources"], f"{path}.sources")
    _provenance(item["provenance"], f"{path}.provenance", source_ids)
    return model_id


def _validate_platform(data: dict[str, Any], path: Path) -> str:
    item = _object(
        data,
        str(path),
        {
            "schema_version",
            "kind",
            "platform_id",
            "display_name",
            "vendor",
            "device_class",
            "identity",
            "static_facts",
            "capacity_policy",
            "runtime",
            "topology",
            "qualification",
            "sources",
            "provenance",
        },
    )
    if item["schema_version"] != CATALOG_SCHEMA_VERSION or item["kind"] != PLATFORM_KIND:
        raise QualificationCatalogError(f"{path}: unsupported platform profile schema")
    platform_id = _slug(item["platform_id"], f"{path}.platform_id")
    if path.stem != platform_id:
        raise QualificationCatalogError(f"{path}: filename must match platform_id")
    _text(item["display_name"], f"{path}.display_name")
    vendor = _text(item["vendor"], f"{path}.vendor")
    if vendor not in {"apple", "nvidia"}:
        raise QualificationCatalogError(f"{path}.vendor must be apple or nvidia")
    expected_class = "apple_silicon" if vendor == "apple" else "cuda_gpu"
    if item["device_class"] != expected_class:
        raise QualificationCatalogError(
            f"{path}.device_class must be {expected_class} for {vendor}"
        )

    identity_keys = (
        {
            "chip",
            "memory_gb",
            "marketed_memory_gb",
            "configuration_note",
        }
        if vendor == "apple"
        else {
            "gpu_name",
            "vram_gb",
            "marketed_memory_gb",
            "configuration_note",
        }
    )
    identity = _object(item["identity"], f"{path}.identity", identity_keys)
    if vendor == "apple":
        _text(identity["chip"], f"{path}.identity.chip")
        matcher_memory = _integer(identity["memory_gb"], f"{path}.identity.memory_gb")
    else:
        _text(identity["gpu_name"], f"{path}.identity.gpu_name")
        matcher_memory = _integer(identity["vram_gb"], f"{path}.identity.vram_gb")
    marketed_memory = _integer(
        identity["marketed_memory_gb"], f"{path}.identity.marketed_memory_gb"
    )
    if matcher_memory != marketed_memory:
        raise QualificationCatalogError(
            f"{path}: matcher memory and marketed memory disagree"
        )
    _text(identity["configuration_note"], f"{path}.identity.configuration_note")

    facts = _object(
        item["static_facts"],
        f"{path}.static_facts",
        {"accelerator", "memory", "memory_bandwidth_bytes_per_second", "interfaces", "power"},
    )
    accelerator = _object(
        facts["accelerator"],
        f"{path}.static_facts.accelerator",
        {"model", "accelerator_count", "gpu_cores", "compute_capability"},
    )
    _text(accelerator["model"], f"{path}.static_facts.accelerator.model")
    _integer(
        accelerator["accelerator_count"],
        f"{path}.static_facts.accelerator.accelerator_count",
    )
    _optional_integer(accelerator["gpu_cores"], f"{path}.static_facts.accelerator.gpu_cores")
    if accelerator["compute_capability"] is not None:
        _text(
            accelerator["compute_capability"],
            f"{path}.static_facts.accelerator.compute_capability",
        )
    memory = _object(
        facts["memory"],
        f"{path}.static_facts.memory",
        {"kind", "marketed_gb", "unified_with_system"},
    )
    _text(memory["kind"], f"{path}.static_facts.memory.kind")
    if _integer(memory["marketed_gb"], f"{path}.static_facts.memory.marketed_gb") != marketed_memory:
        raise QualificationCatalogError(f"{path}: identity and static memory disagree")
    unified = _boolean(
        memory["unified_with_system"], f"{path}.static_facts.memory.unified_with_system"
    )
    if unified != (vendor == "apple"):
        raise QualificationCatalogError(f"{path}: unified memory flag disagrees with vendor")
    _integer(
        facts["memory_bandwidth_bytes_per_second"],
        f"{path}.static_facts.memory_bandwidth_bytes_per_second",
    )
    interfaces = facts["interfaces"]
    if not isinstance(interfaces, list) or not interfaces:
        raise QualificationCatalogError(f"{path}.static_facts.interfaces must be non-empty")
    for index, raw in enumerate(interfaces):
        field = f"{path}.static_facts.interfaces[{index}]"
        interface = _object(
            raw,
            field,
            {
                "kind",
                "count",
                "advertised_bandwidth_bytes_per_second",
                "bandwidth_scope",
                "memory_pooling",
                "note",
            },
        )
        _text(interface["kind"], f"{field}.kind")
        _integer(interface["count"], f"{field}.count")
        _optional_integer(
            interface["advertised_bandwidth_bytes_per_second"],
            f"{field}.advertised_bandwidth_bytes_per_second",
        )
        _text(interface["bandwidth_scope"], f"{field}.bandwidth_scope")
        if _boolean(interface["memory_pooling"], f"{field}.memory_pooling"):
            raise QualificationCatalogError(
                f"{field}.memory_pooling must be false until physically proven"
            )
        _text(interface["note"], f"{field}.note")
    power = _object(
        facts["power"],
        f"{path}.static_facts.power",
        {"value_watts", "rating_kind", "not_chip_tdp"},
    )
    _integer(power["value_watts"], f"{path}.static_facts.power.value_watts")
    _text(power["rating_kind"], f"{path}.static_facts.power.rating_kind")
    _boolean(power["not_chip_tdp"], f"{path}.static_facts.power.not_chip_tdp")

    capacity = _object(
        item["capacity_policy"],
        f"{path}.capacity_policy",
        {"sizing_memory_bytes", "usable_memory_basis_points", "policy_scope", "excludes"},
    )
    expected_sizing_bytes = marketed_memory * 1024**3
    if capacity["sizing_memory_bytes"] != expected_sizing_bytes:
        raise QualificationCatalogError(
            f"{path}.capacity_policy.sizing_memory_bytes must equal nominal marketed GiB"
        )
    usable_bps = _integer(
        capacity["usable_memory_basis_points"],
        f"{path}.capacity_policy.usable_memory_basis_points",
    )
    if usable_bps >= 10000:
        raise QualificationCatalogError(
            f"{path}.capacity_policy.usable_memory_basis_points must be below 10000"
        )
    if capacity["policy_scope"] != "capacity_estimate_only":
        raise QualificationCatalogError(
            f"{path}.capacity_policy.policy_scope must be capacity_estimate_only"
        )
    _strings(capacity["excludes"], f"{path}.capacity_policy.excludes")

    runtime = _object(
        item["runtime"],
        f"{path}.runtime",
        {
            "candidate_backend",
            "os_family",
            "architecture",
            "minimum_os",
            "minimum_driver",
            "hardware_precision",
            "runtime_precision_candidates",
            "verification_status",
        },
    )
    if runtime["candidate_backend"] != "max":
        raise QualificationCatalogError(f"{path}.runtime.candidate_backend must be max")
    _text(runtime["os_family"], f"{path}.runtime.os_family")
    _text(runtime["architecture"], f"{path}.runtime.architecture")
    _text(runtime["minimum_os"], f"{path}.runtime.minimum_os")
    if runtime["minimum_driver"] is not None:
        _text(runtime["minimum_driver"], f"{path}.runtime.minimum_driver")
    _strings(runtime["hardware_precision"], f"{path}.runtime.hardware_precision")
    _strings(
        runtime["runtime_precision_candidates"],
        f"{path}.runtime.runtime_precision_candidates",
    )
    if runtime["verification_status"] != "unverified":
        raise QualificationCatalogError(
            f"{path}.runtime.verification_status must be unverified"
        )

    topology = _object(
        item["topology"],
        f"{path}.topology",
        {
            "unit_semantics",
            "multi_unit_memory_pooling",
            "internal_fabric",
            "default_cross_unit_transport",
            "notes",
        },
    )
    _text(topology["unit_semantics"], f"{path}.topology.unit_semantics")
    if _boolean(
        topology["multi_unit_memory_pooling"],
        f"{path}.topology.multi_unit_memory_pooling",
    ):
        raise QualificationCatalogError(
            f"{path}.topology.multi_unit_memory_pooling must be false"
        )
    _text(topology["internal_fabric"], f"{path}.topology.internal_fabric")
    if topology["default_cross_unit_transport"] != "unmeasured":
        raise QualificationCatalogError(
            f"{path}.topology.default_cross_unit_transport must be unmeasured"
        )
    _strings(topology["notes"], f"{path}.topology.notes")
    _validate_qualification(item["qualification"], f"{path}.qualification", model=False)
    source_ids = _sources(item["sources"], f"{path}.sources")
    _provenance(item["provenance"], f"{path}.provenance", source_ids)
    return platform_id


@dataclass(frozen=True)
class ModelQualificationProfile:
    model_id: str
    display_name: str
    profile_sha256: str
    source_path: Path
    _canonical_data: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_data)

    @property
    def artifact(self) -> dict[str, Any]:
        return self.to_dict()["artifact"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.to_dict()["selection"]

    @property
    def architecture(self) -> dict[str, Any]:
        return self.to_dict()["architecture"]


@dataclass(frozen=True)
class PlatformQualificationProfile:
    platform_id: str
    display_name: str
    profile_sha256: str
    source_path: Path
    _canonical_data: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_data)

    @property
    def capacity_policy(self) -> dict[str, Any]:
        return self.to_dict()["capacity_policy"]


@dataclass(frozen=True)
class QualificationCatalog:
    root: Path
    catalog_sha256: str
    models: tuple[ModelQualificationProfile, ...]
    platforms: tuple[PlatformQualificationProfile, ...]
    _canonical_manifest: str

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._canonical_manifest)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(profile.model_id for profile in self.models)

    @property
    def platform_ids(self) -> tuple[str, ...]:
        return tuple(profile.platform_id for profile in self.platforms)

    def model(self, model_id: str) -> ModelQualificationProfile:
        for profile in self.models:
            if profile.model_id == model_id:
                return profile
        raise QualificationCatalogError(f"unknown model profile: {model_id}")

    def platform(self, platform_id: str) -> PlatformQualificationProfile:
        for rejected in self.manifest["rejected_platform_identifiers"]:
            if rejected["identifier"] == platform_id:
                raise QualificationCatalogError(
                    f"unsupported marketed configuration {platform_id}: "
                    f"{rejected['reason']} Use {rejected['replacement_platform_id']}."
                )
        for profile in self.platforms:
            if profile.platform_id == platform_id:
                return profile
        raise QualificationCatalogError(f"unknown platform profile: {platform_id}")

    def recipe_ids(self) -> tuple[str, ...]:
        return tuple(
            qualification_recipe_id(model_id, platform_id)
            for model_id in self.model_ids
            for platform_id in self.platform_ids
        )


def load_qualification_catalog(
    root: str | Path | None = None,
) -> QualificationCatalog:
    """Load and strictly validate the packaged or supplied qualification catalog."""

    catalog_root = Path(root) if root is not None else qualification_catalog_root()
    manifest_path = catalog_root / "catalog.json"
    manifest = _read_json(manifest_path)
    manifest = _object(
        manifest,
        str(manifest_path),
        {
            "schema_version",
            "kind",
            "catalog_id",
            "model_ids",
            "platform_ids",
            "expected_recipe_count",
            "defaults",
            "rejected_platform_identifiers",
        },
    )
    if manifest["schema_version"] != CATALOG_SCHEMA_VERSION or manifest["kind"] != CATALOG_KIND:
        raise QualificationCatalogError(f"{manifest_path}: unsupported catalog schema")
    _slug(manifest["catalog_id"], f"{manifest_path}.catalog_id")
    model_ids = _strings(manifest["model_ids"], f"{manifest_path}.model_ids")
    platform_ids = _strings(manifest["platform_ids"], f"{manifest_path}.platform_ids")
    if model_ids != sorted(model_ids) or platform_ids != sorted(platform_ids):
        raise QualificationCatalogError(f"{manifest_path}: IDs must be sorted")
    expected_count = _integer(
        manifest["expected_recipe_count"], f"{manifest_path}.expected_recipe_count"
    )
    if expected_count != len(model_ids) * len(platform_ids):
        raise QualificationCatalogError(
            f"{manifest_path}.expected_recipe_count does not match the cross-product"
        )
    defaults = _object(
        manifest["defaults"],
        f"{manifest_path}.defaults",
        {"runtime_headroom_basis_points", "maturity", "support_state", "physical_claims"},
    )
    headroom = _integer(
        defaults["runtime_headroom_basis_points"],
        f"{manifest_path}.defaults.runtime_headroom_basis_points",
        minimum=0,
    )
    if headroom >= 10000:
        raise QualificationCatalogError(
            f"{manifest_path}: runtime headroom must be below 10000 basis points"
        )
    if defaults["maturity"] != "C1_contracted" or defaults["support_state"] != "contract_validated":
        raise QualificationCatalogError(f"{manifest_path}: catalog defaults overstate maturity")
    claims = _object(
        defaults["physical_claims"],
        f"{manifest_path}.defaults.physical_claims",
        PHYSICAL_CLAIM_KEYS,
    )
    if any(_boolean(claims[key], f"{manifest_path}.defaults.physical_claims.{key}") for key in claims):
        raise QualificationCatalogError(f"{manifest_path}: physical claims must all be false")

    rejected = manifest["rejected_platform_identifiers"]
    if not isinstance(rejected, list):
        raise QualificationCatalogError(
            f"{manifest_path}.rejected_platform_identifiers must be a list"
        )
    rejected_ids: set[str] = set()
    for index, raw in enumerate(rejected):
        field = f"{manifest_path}.rejected_platform_identifiers[{index}]"
        item = _object(
            raw,
            field,
            {"identifier", "reason", "replacement_platform_id", "source_url"},
        )
        identifier = _slug(item["identifier"], f"{field}.identifier")
        if identifier in rejected_ids or identifier in platform_ids:
            raise QualificationCatalogError(f"{field}.identifier collides with another ID")
        rejected_ids.add(identifier)
        _text(item["reason"], f"{field}.reason")
        replacement = _slug(item["replacement_platform_id"], f"{field}.replacement_platform_id")
        if replacement not in platform_ids:
            raise QualificationCatalogError(f"{field}.replacement_platform_id is unknown")
        if not _text(item["source_url"], f"{field}.source_url").startswith("https://"):
            raise QualificationCatalogError(f"{field}.source_url must use https")

    model_paths = sorted((catalog_root / "models").glob("*.json"))
    platform_paths = sorted((catalog_root / "platforms").glob("*.json"))
    if [path.stem for path in model_paths] != model_ids:
        raise QualificationCatalogError(
            f"{catalog_root}: model files do not exactly match catalog model_ids"
        )
    if [path.stem for path in platform_paths] != platform_ids:
        raise QualificationCatalogError(
            f"{catalog_root}: platform files do not exactly match catalog platform_ids"
        )

    models: list[ModelQualificationProfile] = []
    normalized_models: list[dict[str, Any]] = []
    for path in model_paths:
        data = _read_json(path)
        model_id = _validate_model(data, path)
        normalized_models.append(data)
        models.append(
            ModelQualificationProfile(
                model_id=model_id,
                display_name=data["display_name"],
                profile_sha256=canonical_sha256(data),
                source_path=path,
                _canonical_data=canonical_json(data),
            )
        )

    platforms: list[PlatformQualificationProfile] = []
    normalized_platforms: list[dict[str, Any]] = []
    for path in platform_paths:
        data = _read_json(path)
        platform_id = _validate_platform(data, path)
        normalized_platforms.append(data)
        platforms.append(
            PlatformQualificationProfile(
                platform_id=platform_id,
                display_name=data["display_name"],
                profile_sha256=canonical_sha256(data),
                source_path=path,
                _canonical_data=canonical_json(data),
            )
        )

    digest_payload = {
        "manifest": manifest,
        "models": normalized_models,
        "platforms": normalized_platforms,
    }
    return QualificationCatalog(
        root=catalog_root,
        catalog_sha256=canonical_sha256(digest_payload),
        models=tuple(models),
        platforms=tuple(platforms),
        _canonical_manifest=canonical_json(manifest),
    )


def qualification_recipe_id(model_id: str, platform_id: str) -> str:
    """Return the stable recipe ID for one model/platform pair."""

    return f"{_slug(model_id, 'model_id')}--{_slug(platform_id, 'platform_id')}"


def _capacity_estimate(
    model: ModelQualificationProfile,
    platform: PlatformQualificationProfile,
    *,
    runtime_headroom_basis_points: int,
    units: int | None,
) -> dict[str, Any]:
    weight_bytes = int(model.artifact["estimated_weight_bytes"])
    headroom_bytes = (
        weight_bytes * runtime_headroom_basis_points + 9999
    ) // 10000
    required_bytes = weight_bytes + headroom_bytes
    policy = platform.capacity_policy
    sizing_bytes = int(policy["sizing_memory_bytes"])
    usable_bps = int(policy["usable_memory_basis_points"])
    usable_bytes = sizing_bytes * usable_bps // 10000
    minimum_units = (required_bytes + usable_bytes - 1) // usable_bytes
    selected_units = minimum_units if units is None else _integer(units, "units")
    return {
        "scope": "weights_plus_static_runtime_headroom_only",
        "capacity_only": True,
        "performance_feasibility_evaluated": False,
        "checkpoint_weight_bytes": weight_bytes,
        "runtime_headroom_basis_points": runtime_headroom_basis_points,
        "runtime_headroom_bytes": headroom_bytes,
        "estimated_required_bytes": required_bytes,
        "nominal_memory_bytes_per_unit": sizing_bytes,
        "usable_memory_basis_points": usable_bps,
        "estimated_usable_memory_bytes_per_unit": usable_bytes,
        "minimum_units": minimum_units,
        "selected_units": selected_units,
        "capacity_sufficient_by_estimate": selected_units >= minimum_units,
        "excludes": list(policy["excludes"]),
    }


def _precision_contract(
    model: dict[str, Any],
    platform: dict[str, Any],
) -> dict[str, Any]:
    model_candidates = list(model["runtime"]["activation_dtype_candidates"])
    platform_candidates = list(
        platform["runtime"]["runtime_precision_candidates"]
    )
    direct_overlap = sorted(set(model_candidates) & set(platform_candidates))
    return {
        "checkpoint_weight_encoding": model["runtime"]["max_weight_encoding"],
        "model_activation_dtype_candidates": model_candidates,
        "platform_runtime_precision_candidates": platform_candidates,
        "direct_activation_precision_overlap": direct_overlap,
        "conversion_or_custom_kernel_required": not direct_overlap,
        "scope": (
            "Static candidate intersection only; device execution, decode, "
            "conversion accuracy, kernel coverage, and performance are unverified."
        ),
    }


def _command_contract(
    recipe_id: str,
    model: dict[str, Any],
    platform: dict[str, Any],
    claims: dict[str, bool],
    capacity: dict[str, Any],
    precision: dict[str, Any],
) -> dict[str, Any]:
    repository = model["artifact"]["repository"]
    revision = model["artifact"]["revision"]
    prompt = "Reply with exactly FORNAX_MOE_SMOKE_OK and nothing else."
    max_command_placeholder = "<DIRECT_FORNAX_MAX_EXECUTABLE_PATH>"
    acquisition_argv = [
        "hf",
        "download",
        repository,
        "--revision",
        revision,
    ]
    for pattern in model["artifact"]["download_exclude_patterns"]:
        acquisition_argv.extend(["--exclude", pattern])
    acquisition_argv.extend(["--local-dir", "<MODEL_DIR>"])
    inspect_model_argv = [
        "python3",
        "-m",
        "fornax",
        "recipe",
        "inspect-model",
        "--model",
        model["model_id"],
        "--model-dir",
        "<MODEL_DIR>",
        "--out",
        "<EVIDENCE_DIR>/model-artifacts.json",
    ]
    if model["artifact"]["trust_remote_code_required"]:
        inspect_model_argv.extend(
            [
                "--remote-code-review",
                "<REMOTE_CODE_REVIEW_JSON>",
                "--expected-remote-code-review-sha256",
                "<REMOTE_CODE_REVIEW_SHA256>",
            ]
        )
    commands: list[dict[str, Any]] = [
        {
            "step_id": "acquire_pinned_model",
            "purpose": "Download the exact pinned artifact without executing model code.",
            "argv": acquisition_argv,
            "network_required": True,
            "execution_status": "operator_review_required",
            "evidence_scope": "artifact_acquisition_only",
        },
        {
            "step_id": "inspect_local_model",
            "purpose": (
                "Hash and verify the pinned local representation, tokenizer, "
                "config, revision metadata, index, shards, and exact bytes."
            ),
            "argv": inspect_model_argv,
            "network_required": False,
            "execution_status": "blocked_until_acquisition_completes",
            "evidence_scope": "local_model_artifact_identity_only",
        },
        {
            "step_id": "probe_host_identity",
            "purpose": (
                "Match the local chip or GPU name and nominal memory threshold; "
                "repeat once per host and qualify fleet count/topology separately."
            ),
            "argv": [
                "python3",
                "-m",
                "fornax",
                "recipe",
                "probe-host",
                "--platform",
                platform["platform_id"],
                "--out",
                "<EVIDENCE_DIR>/hosts/<HOST_ID>/host-identity.json",
            ],
            "network_required": False,
            "execution_status": "operator_review_required",
            "evidence_scope": "observed_host_identity_only",
        },
        {
            "step_id": "collect_local_inventory",
            "purpose": "Capture local identity inputs; exact recipe matching remains a separate preflight.",
            "argv": [
                "python3",
                "-m",
                "fornax",
                "inventory",
                "collect",
                "--out",
                "<EVIDENCE_DIR>/hosts/<HOST_ID>/inventory.json",
            ],
            "network_required": False,
            "execution_status": "operator_review_required",
            "evidence_scope": "inventory_only",
        },
        {
            "step_id": "probe_max_runtime_registry",
            "purpose": (
                "Record MAX version and fail closed unless `max list --json` "
                "advertises the exact model architecture and weight encoding. "
                "This is registry evidence, not device-kernel compatibility."
            ),
            "argv": [
                "python3",
                "-m",
                "fornax",
                "recipe",
                "probe-runtime",
                "--model",
                model["model_id"],
                "--platform",
                platform["platform_id"],
                "--max-command",
                max_command_placeholder,
                "--out",
                "<EVIDENCE_DIR>/hosts/<HOST_ID>/max-runtime.json",
            ],
            "network_required": False,
            "execution_status": "operator_review_required",
            "evidence_scope": "max_registry_identity_only",
        },
    ]
    if capacity["selected_units"] == 1:
        if platform["vendor"] == "apple":
            argv = [
                "python3",
                "-m",
                "fornax",
                "recipe",
                "run-apple-single",
                "--model",
                model["model_id"],
                "--platform",
                platform["platform_id"],
                "--model-dir",
                "<MODEL_DIR>",
                "--model-artifact-report",
                "<EVIDENCE_DIR>/model-artifacts.json",
                "--host-report",
                "<EVIDENCE_DIR>/hosts/<HOST_ID>/host-identity.json",
                "--runtime-report",
                "<EVIDENCE_DIR>/hosts/<HOST_ID>/max-runtime.json",
                "--max-command",
                max_command_placeholder,
                "--out",
                "<EVIDENCE_DIR>/single-platform-bringup.json",
                "--prompt",
                prompt,
                "--max-new-tokens",
                "16",
                "--top-k",
                "1",
            ]
            purpose = (
                "Candidate single-Mac MAX generation after binding fresh artifact, "
                "exact Apple host, collector-executable, and MAX-registry evidence "
                "to the selected catalog profiles. This requires an explicit "
                "future/custom Fornax-capable MAX command because upstream stock "
                "MAX currently does not provide large GenAI inference on Apple "
                "silicon. The wrapper reruns the local checkpoint and exact "
                "sentinel; success remains bring-up, not publisher support or "
                "numerical parity."
            )
        else:
            argv = [
                "python3",
                "-m",
                "fornax",
                "recipe",
                "run-nvidia-single",
                "--model",
                model["model_id"],
                "--platform",
                platform["platform_id"],
                "--model-dir",
                "<MODEL_DIR>",
                "--model-artifact-report",
                "<EVIDENCE_DIR>/model-artifacts.json",
                "--host-report",
                "<EVIDENCE_DIR>/hosts/<HOST_ID>/host-identity.json",
                "--runtime-report",
                "<EVIDENCE_DIR>/hosts/<HOST_ID>/max-runtime.json",
                "--out",
                "<EVIDENCE_DIR>/single-platform-bringup.json",
                "--device",
                "<NVIDIA_SMI_GPU_DEVICE>",
                "--max-command",
                max_command_placeholder,
                "--max-new-tokens",
                "16",
                "--top-k",
                "1",
                "--prompt",
                prompt,
            ]
            purpose = (
                "Candidate single-GPU MAX generation after manual review of "
                "artifact, host, and runtime-registry evidence. The selected "
                "physical nvidia-smi gpu:N is freshly resolved to its GPU UUID, "
                "mapped through CUDA_VISIBLE_DEVICES, and launched as MAX gpu:0. "
                "The wrapper binds one direct MAX executable plus the model "
                "directory and artifact bytes immediately before and after "
                "generation, and records bounded argv/output evidence; success "
                "remains bring-up, not numerical parity or support."
            )
        execution_status = (
            "blocked_until_precision_conversion_and_all_preflights_pass"
            if precision["conversion_or_custom_kernel_required"]
            else "blocked_until_all_preflights_pass"
        )
        commands.append(
            {
                "step_id": "single_platform_model_bringup",
                "purpose": purpose,
                "argv": argv,
                "network_required": False,
                "execution_status": execution_status,
                "evidence_scope": "single_platform_bringup_only",
            }
        )
    else:
        commands.append(
            {
                "step_id": "capacity_spanning_readiness",
                "purpose": (
                    "Generate the fail-closed G2 readiness packet only. No "
                    "full-model execution argv is emitted because the capacity "
                    f"estimate spans {capacity['selected_units']} distinct units "
                    "and no physical topology or StageBackend is yet bound."
                ),
                "argv": [
                    "python3",
                    "-m",
                    "fornax",
                    "program",
                    "g2-validate",
                    "--out-dir",
                    "<EVIDENCE_DIR>/g2-readiness",
                ],
                "network_required": False,
                "execution_status": (
                    "blocked_until_physical_backend_topology_and_per_host_"
                    "preflight"
                ),
                "evidence_scope": "readiness_only_no_model_execution",
            }
        )
    return {
        "schema_version": 1,
        "record_kind": COMMANDS_KIND,
        "recipe_id": recipe_id,
        "substitution_policy": (
            "Replace angle-bracket placeholders inside each existing argv "
            "element while preserving element boundaries. Do not concatenate "
            "these arrays into a shell command."
        ),
        "commands": commands,
        "physical_claims": dict(claims),
    }


def _runbook(
    lock: dict[str, Any],
    commands: dict[str, Any],
    model: dict[str, Any],
    platform: dict[str, Any],
) -> str:
    capacity = lock["capacity_estimate"]
    precision = lock["precision_contract"]
    lines = [
        f"# Qualification recipe: {lock['recipe_id']}",
        "",
        "Status: **C1 contracted / contract validated**. This is an offline",
        "qualification contract, not a supported-hardware, performance, parity,",
        "distributed-runtime, G2, G3, or production claim.",
        "",
        "## Pinned inputs",
        "",
        f"- Model: `{model['artifact']['repository']}`",
        f"- Revision: `{model['artifact']['revision']}`",
        f"- Weight encoding: `{model['artifact']['weight_dtype']}`",
        f"- MAX architecture: `{model['runtime']['max_architecture']}`",
        f"- MAX encoding contract: `{model['runtime']['max_weight_encoding']}`",
        f"- Platform: `{platform['display_name']}`",
        f"- Recipe lock content hash: `{lock['lock_content_sha256']}`",
        "",
        "## Capacity-only estimate",
        "",
        f"- Checkpoint bytes: {capacity['checkpoint_weight_bytes']}",
        f"- Static runtime headroom: {capacity['runtime_headroom_basis_points']} basis points",
        f"- Estimated usable bytes per unit: {capacity['estimated_usable_memory_bytes_per_unit']}",
        f"- Minimum units by this arithmetic: {capacity['minimum_units']}",
        f"- Selected units: {capacity['selected_units']}",
        "",
        "This arithmetic excludes KV-cache sizing, measured runtime workspaces,",
        "transport buffers, topology, performance, thermals, and power. Multiple",
        "units remain distinct memory pools; the count is not proof that the model",
        "can execute with acceptable correctness or throughput.",
        "",
        "## Precision contract",
        "",
        "- Model activation candidates: "
        + ", ".join(
            f"`{value}`"
            for value in precision["model_activation_dtype_candidates"]
        ),
        "- Platform runtime candidates: "
        + ", ".join(
            f"`{value}`"
            for value in precision["platform_runtime_precision_candidates"]
        ),
        "- Direct overlap: "
        + (
            ", ".join(
                f"`{value}`"
                for value in precision["direct_activation_precision_overlap"]
            )
            or "none"
        ),
        "- Conversion or custom kernel required: "
        + str(precision["conversion_or_custom_kernel_required"]).lower(),
        "",
        "The checkpoint encoding is a storage/decode contract, not proof that",
        "the target device can execute that dtype. `max list --json` establishes",
        "registry advertisement only; physical decode, conversion, kernels, and",
        "numerical parity remain separate gates.",
        "",
        "## Operator commands",
        "",
        "Commands are recorded as argv arrays. Substitute placeholders element by",
        "element and do not evaluate them through a shell.",
        "",
    ]
    if capacity["selected_units"] > 1:
        lines.extend(
            [
                "This recipe spans multiple distinct memory pools. It intentionally",
                "contains no full-model execution command until a physical topology",
                "and Fornax StageBackend bind all selected units. Run host/runtime",
                "preflight once per host, using a distinct `<HOST_ID>`.",
                "",
            ]
        )
    for command in commands["commands"]:
        lines.extend(
            [
                f"### {command['step_id']}",
                "",
                command["purpose"],
                "",
                "```json",
                json.dumps(command["argv"], ensure_ascii=False),
                "```",
                "",
                f"Execution status: `{command['execution_status']}`.",
                "",
            ]
        )
    lines.extend(["## Blocking evidence", ""])
    lines.extend(f"- {blocker}" for blocker in lock["blockers"])
    lines.extend(
        [
            "",
            "A generated-text smoke can only establish bounded bring-up. Promote",
            "maturity only through the repository's physical parity, backend,",
            "transport, and formal gate contracts.",
            "",
        ]
    )
    return "\n".join(lines)


def compose_qualification_recipe(
    model_id: str,
    platform_id: str,
    *,
    units: int | None = None,
    catalog: QualificationCatalog | None = None,
) -> dict[str, Any]:
    """Compose one deterministic C1 qualification recipe bundle."""

    active_catalog = catalog or load_qualification_catalog()
    model_profile = active_catalog.model(model_id)
    platform_profile = active_catalog.platform(platform_id)
    model = model_profile.to_dict()
    platform = platform_profile.to_dict()
    recipe_id = qualification_recipe_id(model_id, platform_id)
    defaults = active_catalog.manifest["defaults"]
    claims = dict(defaults["physical_claims"])
    capacity = _capacity_estimate(
        model_profile,
        platform_profile,
        runtime_headroom_basis_points=defaults["runtime_headroom_basis_points"],
        units=units,
    )
    precision = _precision_contract(model, platform)
    blockers = [
        *(f"Model: {message}" for message in model["qualification"]["unresolved_evidence"]),
        *(
            f"Platform: {message}"
            for message in platform["qualification"]["unresolved_evidence"]
        ),
        "Resolve exact multi-unit topology and measure every cross-unit route.",
        "Provide a physical Fornax stage backend and pass backend conformance.",
        "Pass formal parity and G2/G3 gates before any distributed support claim.",
    ]
    if precision["conversion_or_custom_kernel_required"]:
        blockers.insert(
            0,
            "Model activation precision candidates and platform runtime precision "
            "candidates have no direct overlap; prove an explicit conversion or "
            "custom-kernel path before model execution.",
        )
    if not capacity["capacity_sufficient_by_estimate"]:
        blockers.insert(
            0,
            "Selected unit count is below the capacity-only minimum; physical run is forbidden.",
        )
    lock_payload = {
        "schema_version": 1,
        "record_kind": LOCK_KIND,
        "recipe_id": recipe_id,
        "catalog_id": active_catalog.manifest["catalog_id"],
        "catalog_sha256": active_catalog.catalog_sha256,
        "inputs": {
            "model": {
                "model_id": model_id,
                "profile_sha256": model_profile.profile_sha256,
                "repository": model["artifact"]["repository"],
                "revision": model["artifact"]["revision"],
                "weight_dtype": model["artifact"]["weight_dtype"],
                "quantization": model["artifact"]["quantization"],
            },
            "platform": {
                "platform_id": platform_id,
                "profile_sha256": platform_profile.profile_sha256,
                "vendor": platform["vendor"],
                "marketed_memory_gb": platform["identity"]["marketed_memory_gb"],
            },
        },
        "capacity_estimate": capacity,
        "precision_contract": precision,
        "qualification": {
            "maturity": defaults["maturity"],
            "support_state": defaults["support_state"],
            "physical_execution_status": "not_run",
            "authority": "exploratory",
        },
        "required_operations": list(REQUIRED_OPERATIONS),
        "physical_claims": claims,
        "blockers": blockers,
    }
    lock = dict(lock_payload)
    lock["lock_content_sha256"] = canonical_sha256(lock_payload)
    commands = _command_contract(
        recipe_id,
        model,
        platform,
        claims,
        capacity,
        precision,
    )
    return {
        "recipe_id": recipe_id,
        "lock": lock,
        "commands": commands,
        "runbook_markdown": _runbook(lock, commands, model, platform),
    }


def compose_all_qualification_recipes(
    catalog: QualificationCatalog | None = None,
) -> tuple[dict[str, Any], ...]:
    """Compose the catalog's complete, sorted model/platform cross-product."""

    active_catalog = catalog or load_qualification_catalog()
    return tuple(
        compose_qualification_recipe(model_id, platform_id, catalog=active_catalog)
        for model_id in active_catalog.model_ids
        for platform_id in active_catalog.platform_ids
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _managed_recipe_bytes(bundle: dict[str, Any]) -> dict[str, bytes]:
    return {
        "recipe-lock.json": _pretty_json(bundle["lock"]).encode("utf-8"),
        "commands.json": _pretty_json(bundle["commands"]).encode("utf-8"),
        "RUNBOOK.md": bundle["runbook_markdown"].encode("utf-8"),
    }


def _directory_flags(*, nofollow: bool) -> int:
    if not hasattr(os, "O_DIRECTORY"):
        raise QualificationCatalogError(
            "directory-descriptor operations require O_DIRECTORY"
        )
    if nofollow and not hasattr(os, "O_NOFOLLOW"):
        raise QualificationCatalogError(
            "no-follow directory operations require O_NOFOLLOW"
        )
    flags = os.O_RDONLY
    flags |= os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0)
    if nofollow:
        flags |= os.O_NOFOLLOW
    return flags


def _open_materialization_directory(
    output: Path,
) -> tuple[int | None, int, str | None, Path]:
    """Create/open ``output`` and retain descriptors anchoring its identity."""

    absolute_output = Path(os.path.abspath(os.fspath(output)))
    if absolute_output.parent == absolute_output:
        try:
            output_fd = os.open(
                absolute_output,
                _directory_flags(nofollow=True),
            )
        except OSError as exc:
            raise QualificationCatalogError(
                "recipe output directory cannot be opened without following "
                f"a symbolic link: {absolute_output}: {exc}"
            ) from exc
        return None, output_fd, None, absolute_output

    parent = absolute_output.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_fd = os.open(parent, _directory_flags(nofollow=False))
    except OSError as exc:
        raise QualificationCatalogError(
            f"recipe output parent directory cannot be opened: {parent}: {exc}"
        ) from exc

    entry_name = absolute_output.name
    output_fd: int | None = None
    try:
        try:
            os.mkdir(entry_name, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        entry_before = os.stat(
            entry_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(entry_before.st_mode):
            raise QualificationCatalogError(
                "recipe output path must be a real directory and not a "
                f"symbolic link: {absolute_output}"
            )
        output_fd = os.open(
            entry_name,
            _directory_flags(nofollow=True),
            dir_fd=parent_fd,
        )
        metadata = os.fstat(output_fd)
        entry_after = os.stat(
            entry_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        expected_identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (entry_before.st_dev, entry_before.st_ino)
            != expected_identity
            or (entry_after.st_dev, entry_after.st_ino)
            != expected_identity
        ):
            raise QualificationCatalogError(
                "recipe output directory changed while it was being opened: "
                f"{absolute_output}"
            )
    except Exception as exc:
        if output_fd is not None:
            os.close(output_fd)
        os.close(parent_fd)
        if isinstance(exc, QualificationCatalogError):
            raise
        raise QualificationCatalogError(
            "recipe output directory must be a real directory and not a "
            f"symbolic link: {absolute_output}: {exc}"
        ) from exc
    assert output_fd is not None
    return parent_fd, output_fd, entry_name, absolute_output


def _assert_materialization_directory_stable(
    *,
    parent_fd: int | None,
    output_fd: int,
    entry_name: str | None,
    absolute_output: Path,
) -> None:
    """Fail if the lexical output entry stopped naming the retained directory."""

    expected = os.fstat(output_fd)
    try:
        lexical = os.stat(absolute_output, follow_symlinks=False)
        if (
            not stat.S_ISDIR(lexical.st_mode)
            or (lexical.st_dev, lexical.st_ino)
            != (expected.st_dev, expected.st_ino)
        ):
            raise QualificationCatalogError(
                "recipe output directory changed identity during materialization"
            )
        if parent_fd is not None and entry_name is not None:
            anchored = os.stat(
                entry_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(anchored.st_mode)
                or (anchored.st_dev, anchored.st_ino)
                != (expected.st_dev, expected.st_ino)
            ):
                raise QualificationCatalogError(
                    "recipe output directory entry changed during materialization"
                )
    except QualificationCatalogError:
        raise
    except OSError as exc:
        raise QualificationCatalogError(
            "recipe output directory became unavailable during "
            f"materialization: {absolute_output}: {exc}"
        ) from exc


def _entry_metadata_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise QualificationCatalogError(
            f"cannot inspect recipe output entry {name}: {exc}"
        ) from exc


def _write_fsynced_at(directory_fd: int, name: str, value: bytes) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        raise QualificationCatalogError(
            "no-follow file creation requires O_NOFOLLOW"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o644, dir_fd=directory_fd)
    try:
        remaining = memoryview(value)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"short write while creating {name}")
            remaining = remaining[written:]
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError(
                f"temporary recipe file {name} is not one private regular file"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_private_recipe_directory(output_fd: int) -> tuple[str, int]:
    for _attempt in range(128):
        name = ".fornax-recipe-" + secrets.token_hex(16)
        try:
            os.mkdir(name, mode=0o700, dir_fd=output_fd)
        except FileExistsError:
            continue
        descriptor: int | None = None
        try:
            created = os.stat(
                name,
                dir_fd=output_fd,
                follow_symlinks=False,
            )
            descriptor = os.open(
                name,
                _directory_flags(nofollow=True),
                dir_fd=output_fd,
            )
            opened = os.fstat(descriptor)
            anchored = os.stat(
                name,
                dir_fd=output_fd,
                follow_symlinks=False,
            )
            expected_identity = (opened.st_dev, opened.st_ino)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (created.st_dev, created.st_ino) != expected_identity
                or (anchored.st_dev, anchored.st_ino) != expected_identity
            ):
                raise QualificationCatalogError(
                    "private recipe directory changed while it was opened"
                )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=output_fd)
            except OSError:
                pass
            raise
        assert descriptor is not None
        return name, descriptor
    raise QualificationCatalogError(
        "cannot allocate a collision-free private recipe directory"
    )


def _cleanup_private_recipe_directory(
    output_fd: int,
    temporary_name: str,
    temporary_fd: int,
) -> list[str]:
    errors: list[str] = []
    for name in (*MANAGED_PACKET_FILENAMES, PACKET_MANIFEST_FILENAME):
        try:
            os.unlink(name, dir_fd=temporary_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"cannot remove temporary {name}: {exc}")
    os.close(temporary_fd)
    try:
        os.rmdir(temporary_name, dir_fd=output_fd)
    except FileNotFoundError:
        pass
    except OSError as exc:
        errors.append(
            f"cannot remove private recipe directory {temporary_name}: {exc}"
        )
    return errors


def _publish_recipe_file_at(
    *,
    temporary_fd: int,
    output_fd: int,
    name: str,
    overwrite: bool,
) -> None:
    if overwrite:
        os.replace(
            name,
            name,
            src_dir_fd=temporary_fd,
            dst_dir_fd=output_fd,
        )
        return
    try:
        os.link(
            name,
            name,
            src_dir_fd=temporary_fd,
            dst_dir_fd=output_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise QualificationCatalogError(
            "recipe output file appeared during no-overwrite publication: "
            f"{name}"
        ) from exc
    except (OSError, NotImplementedError) as exc:
        raise QualificationCatalogError(
            "atomic no-overwrite publication is unavailable for recipe "
            f"output file {name}: {exc}"
        ) from exc
    os.unlink(name, dir_fd=temporary_fd)


def materialize_qualification_recipe(
    model_id: str,
    platform_id: str,
    out_dir: str | Path,
    *,
    units: int | None = None,
    catalog: QualificationCatalog | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Materialize a manifest-last, content-addressed qualification packet.

    Existing files require ``overwrite=True``; observed symbolic-link or
    non-regular targets are rejected and concurrent targets are never followed.
    Managed files are written and synced in a private directory anchored under
    a retained no-follow output descriptor. They are then atomically published,
    with an output-directory durability barrier before the manifest is
    published last. A crash can therefore leave a missing or mismatched
    manifest, never a silently self-consistent mixed packet.
    """

    bundle = compose_qualification_recipe(
        model_id,
        platform_id,
        units=units,
        catalog=catalog,
    )
    output = Path(out_dir)
    paths = {
        "recipe_lock": output / "recipe-lock.json",
        "commands": output / "commands.json",
        "runbook": output / "RUNBOOK.md",
        "bundle_manifest": output / PACKET_MANIFEST_FILENAME,
    }
    managed_bytes = _managed_recipe_bytes(bundle)
    manifest = build_recipe_packet_manifest(bundle["recipe_id"], managed_bytes)
    manifest_bytes = _pretty_json(manifest).encode("utf-8")

    parent_fd: int | None = None
    output_fd: int | None = None
    entry_name: str | None = None
    absolute_output: Path | None = None
    try:
        (
            parent_fd,
            output_fd,
            entry_name,
            absolute_output,
        ) = _open_materialization_directory(output)
        _assert_materialization_directory_stable(
            parent_fd=parent_fd,
            output_fd=output_fd,
            entry_name=entry_name,
            absolute_output=absolute_output,
        )

        target_names = (*MANAGED_PACKET_FILENAMES, PACKET_MANIFEST_FILENAME)
        existing: list[str] = []
        symbolic_links: list[str] = []
        non_regular: list[str] = []
        for name in target_names:
            metadata = _entry_metadata_at(output_fd, name)
            if metadata is None:
                continue
            existing.append(str(output / name))
            if stat.S_ISLNK(metadata.st_mode):
                symbolic_links.append(str(output / name))
            elif not stat.S_ISREG(metadata.st_mode):
                non_regular.append(str(output / name))
        if symbolic_links:
            raise QualificationCatalogError(
                "recipe output files must not be symbolic links: "
                + ", ".join(symbolic_links)
            )
        if non_regular:
            raise QualificationCatalogError(
                "recipe output files must be regular files: "
                + ", ".join(non_regular)
            )
        if existing and not overwrite:
            raise QualificationCatalogError(
                "recipe output files already exist: " + ", ".join(existing)
            )

        temporary_name, temporary_fd = _create_private_recipe_directory(
            output_fd
        )
        try:
            try:
                for name in MANAGED_PACKET_FILENAMES:
                    _write_fsynced_at(
                        temporary_fd,
                        name,
                        managed_bytes[name],
                    )
                _write_fsynced_at(
                    temporary_fd,
                    PACKET_MANIFEST_FILENAME,
                    manifest_bytes,
                )
                os.fsync(temporary_fd)
                _assert_materialization_directory_stable(
                    parent_fd=parent_fd,
                    output_fd=output_fd,
                    entry_name=entry_name,
                    absolute_output=absolute_output,
                )
                for name in MANAGED_PACKET_FILENAMES:
                    _publish_recipe_file_at(
                        temporary_fd=temporary_fd,
                        output_fd=output_fd,
                        name=name,
                        overwrite=overwrite,
                    )
                os.fsync(output_fd)
                _publish_recipe_file_at(
                    temporary_fd=temporary_fd,
                    output_fd=output_fd,
                    name=PACKET_MANIFEST_FILENAME,
                    overwrite=overwrite,
                )
                os.fsync(output_fd)
            except QualificationCatalogError:
                raise
            except OSError as exc:
                raise QualificationCatalogError(
                    f"cannot publish qualification recipe packet: {exc}"
                ) from exc
        finally:
            cleanup_errors = _cleanup_private_recipe_directory(
                output_fd,
                temporary_name,
                temporary_fd,
            )
            if cleanup_errors:
                raise QualificationCatalogError("; ".join(cleanup_errors))

        os.fsync(output_fd)
        _assert_materialization_directory_stable(
            parent_fd=parent_fd,
            output_fd=output_fd,
            entry_name=entry_name,
            absolute_output=absolute_output,
        )
        verification = verify_recipe_packet(
            output,
            expected_bundle_content_sha256=manifest[
                "bundle_content_sha256"
            ],
            allow_unmanaged_entries=True,
            _directory_fd=output_fd,
        )
        if not verification["ok"]:
            raise QualificationCatalogError(
                "rendered recipe packet failed post-write verification: "
                + "; ".join(verification["errors"])
            )
        _assert_materialization_directory_stable(
            parent_fd=parent_fd,
            output_fd=output_fd,
            entry_name=entry_name,
            absolute_output=absolute_output,
        )
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if parent_fd is not None:
            os.close(parent_fd)
    return paths


def verify_materialized_qualification_recipe(
    packet_dir: str | Path,
    *,
    expected_bundle_content_sha256: str | None = None,
    allow_unmanaged_entries: bool = False,
    catalog: QualificationCatalog | None = None,
) -> dict[str, Any]:
    """Verify packet integrity and exact reproducibility from the current catalog."""

    report = verify_recipe_packet(
        packet_dir,
        expected_bundle_content_sha256=expected_bundle_content_sha256,
        allow_unmanaged_entries=allow_unmanaged_entries,
    )
    errors = list(report["errors"])
    current_catalog_match = False
    expected_current_bundle_sha256: str | None = None

    if report["self_consistent"]:
        try:
            binding = report.get("recipe_lock_binding")
            if not isinstance(binding, dict):
                raise QualificationCatalogError(
                    "verifier did not return a captured recipe-lock catalog "
                    "binding"
                )
            model_id = binding.get("model_id")
            platform_id = binding.get("platform_id")
            units = binding.get("selected_units")
            if not isinstance(model_id, str) or not isinstance(platform_id, str):
                raise QualificationCatalogError(
                    "packet lock model_id and platform_id must be strings"
                )
            if isinstance(units, bool) or not isinstance(units, int) or units < 1:
                raise QualificationCatalogError(
                    "packet lock selected_units must be a positive integer"
                )
            expected_bundle = compose_qualification_recipe(
                model_id,
                platform_id,
                units=units,
                catalog=catalog,
            )
            expected_managed = _managed_recipe_bytes(expected_bundle)
            expected_manifest = build_recipe_packet_manifest(
                expected_bundle["recipe_id"],
                expected_managed,
            )
            expected_current_bundle_sha256 = expected_manifest[
                "bundle_content_sha256"
            ]
            observed_files = report["managed_files"]
            mismatches = [
                name
                for name in MANAGED_PACKET_FILENAMES
                if observed_files.get(name) != expected_manifest["files"][name]
            ]
            if report["bundle_content_sha256"] != expected_current_bundle_sha256:
                mismatches.append(PACKET_MANIFEST_FILENAME)
            if mismatches:
                errors.append(
                    "packet does not reproduce from the current qualification "
                    "catalog: " + ", ".join(sorted(set(mismatches)))
                )
            else:
                current_catalog_match = True
        except (KeyError, QualificationCatalogError) as exc:
            errors.append(f"cannot bind packet to the current catalog: {exc}")

    report["current_catalog_match"] = current_catalog_match
    report["expected_current_bundle_sha256"] = expected_current_bundle_sha256
    report["errors"] = errors
    report["ok"] = bool(report["ok"] and current_catalog_match)
    return report


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "PHYSICAL_CLAIM_KEYS",
    "REQUIRED_OPERATIONS",
    "ModelQualificationProfile",
    "PlatformQualificationProfile",
    "QualificationCatalog",
    "QualificationCatalogError",
    "canonical_json",
    "canonical_sha256",
    "compose_all_qualification_recipes",
    "compose_qualification_recipe",
    "load_qualification_catalog",
    "materialize_qualification_recipe",
    "qualification_catalog_root",
    "qualification_recipe_id",
    "verify_materialized_qualification_recipe",
]
