from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .io import read_json

SUPPORTED_DTYPES = {"fp8", "fp16", "bf16", "fp32"}


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _shape(value: Any, field: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty integer list")
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            errors.append(f"{field} entries must be positive integers")
            return None
        result.append(item)
    return result


def _dtype(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or value not in SUPPORTED_DTYPES:
        errors.append(f"{field} must be one of {sorted(SUPPORTED_DTYPES)}")
        return None
    return value


def _number_list(value: Any, field: str, errors: list[str]) -> list[float] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{field} entries must be numeric")
            return None
        result.append(float(item))
    return result


def _int_list(value: Any, field: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            errors.append(f"{field} entries must be integers")
            return None
        result.append(item)
    return result


def _check_activation(data: dict[str, Any], errors: list[str]) -> None:
    shape = _shape(data.get("shape"), "activation.shape", errors)
    _dtype(data.get("dtype"), "activation.dtype", errors)
    if data.get("layout") != "contiguous_row_major":
        errors.append("activation.layout must be contiguous_row_major")
    values = data.get("values")
    if shape is not None and values is not None:
        nums = _number_list(values, "activation.values", errors)
        if nums is not None and len(nums) != _product(shape):
            errors.append(
                f"activation.values length {len(nums)} does not match "
                f"shape product {_product(shape)}"
            )


def _check_kv_page(data: dict[str, Any], errors: list[str]) -> None:
    shape = _shape(data.get("shape"), "kv_page.shape", errors)
    _dtype(data.get("dtype"), "kv_page.dtype", errors)
    page_size = data.get("page_size")
    token_count = data.get("token_count")
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
        errors.append("kv_page.page_size must be a positive integer")
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count < 0
    ):
        errors.append("kv_page.token_count must be a non-negative integer")
    if (
        isinstance(page_size, int)
        and isinstance(token_count, int)
        and token_count > page_size
    ):
        errors.append("kv_page.token_count cannot exceed page_size")
    owner_stage = data.get("owner_stage")
    if (
        not isinstance(owner_stage, int)
        or isinstance(owner_stage, bool)
        or owner_stage < 0
    ):
        errors.append("kv_page.owner_stage must be a non-negative integer")
    if shape is not None and isinstance(page_size, int) and shape[0] != page_size:
        errors.append("kv_page.shape first dimension must equal page_size")
    if shape is not None and isinstance(token_count, int) and shape[0] < max(token_count, 1):
        errors.append("kv_page.shape first dimension must cover token_count")


def _check_expert_batch(data: dict[str, Any], errors: list[str]) -> None:
    layer_id = data.get("layer_id")
    if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
        errors.append("expert_batch.layer_id must be a non-negative integer")
    hidden_shape = _shape(data.get("hidden_shape"), "expert_batch.hidden_shape", errors)
    expert_ids = _int_list(data.get("expert_ids"), "expert_batch.expert_ids", errors)
    token_indices = _int_list(
        data.get("token_indices"), "expert_batch.token_indices", errors
    )
    gather_order = _int_list(
        data.get("gather_order"), "expert_batch.gather_order", errors
    )
    weights = _number_list(data.get("topk_weights"), "expert_batch.topk_weights", errors)
    if expert_ids is not None and not expert_ids:
        errors.append("expert_batch.expert_ids cannot be empty")
    if (
        token_indices is not None
        and weights is not None
        and len(token_indices) != len(weights)
    ):
        errors.append("expert_batch.token_indices and topk_weights lengths must match")
    if token_indices is not None and gather_order is not None:
        if sorted(gather_order) != list(range(len(token_indices))):
            errors.append(
                "expert_batch.gather_order must be a permutation of packed token positions"
            )
    if (
        hidden_shape is not None
        and token_indices is not None
        and hidden_shape[0] != len(token_indices)
    ):
        errors.append("expert_batch.hidden_shape first dimension must match token_indices length")
    if weights is not None and any(
        (not math.isfinite(weight)) or weight < 0 for weight in weights
    ):
        errors.append("expert_batch.topk_weights must be finite non-negative numbers")


def validate_runtime_format_manifest(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    activation = data.get("activation")
    kv_page = data.get("kv_page")
    expert_batch = data.get("expert_batch")
    if not isinstance(activation, dict):
        errors.append("activation must be an object")
    else:
        _check_activation(activation, errors)
    if not isinstance(kv_page, dict):
        errors.append("kv_page must be an object")
    else:
        _check_kv_page(kv_page, errors)
    if not isinstance(expert_batch, dict):
        errors.append("expert_batch must be an object")
    else:
        _check_expert_batch(expert_batch, errors)
    tolerances = data.get("tolerances", {})
    if not isinstance(tolerances, dict) or not tolerances:
        warnings.append(
            "tolerances missing; golden vectors should declare per-dtype tolerances"
        )
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def validate_runtime_format_golden(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if not manifest_path.exists():
        return {
            "ok": False,
            "errors": [f"missing runtime-format manifest: {manifest_path}"],
            "warnings": [],
            "manifest": str(manifest_path),
        }
    try:
        data = read_json(manifest_path)
    except Exception as exc:  # noqa: BLE001 - validator should report bad fixtures.
        return {
            "ok": False,
            "errors": [f"invalid runtime-format manifest JSON: {exc}"],
            "warnings": [],
            "manifest": str(manifest_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["runtime-format manifest must be a JSON object"],
            "warnings": [],
            "manifest": str(manifest_path),
        }
    result = validate_runtime_format_manifest(data)
    result["manifest"] = str(manifest_path)
    return result
