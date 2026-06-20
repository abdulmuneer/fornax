from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .planner import ModelSpec, Target

_FENCE_RE = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)\n```", re.DOTALL)


class TargetContractError(ValueError):
    """Raised when a Phase-0 target contract cannot be parsed or validated."""


def _loads_json_object(text: str, source: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetContractError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TargetContractError(f"{source}: expected a JSON object")
    return data


def _is_target_bundle(data: dict[str, Any]) -> bool:
    return isinstance(data.get("model"), dict) and isinstance(data.get("target"), dict)


def _extract_markdown_bundle(text: str, source: str) -> dict[str, Any]:
    candidates: list[tuple[str, str]] = []
    for match in _FENCE_RE.finditer(text):
        info = " ".join(match.group("info").strip().lower().split())
        body = match.group("body").strip()
        if info in {"json", "json fornax-target", "fornax-target", "fornax target"}:
            candidates.append((info or "json", body))

    errors: list[str] = []
    for info, body in candidates:
        try:
            data = _loads_json_object(body, f"{source} fenced block {info!r}")
        except TargetContractError as exc:
            errors.append(str(exc))
            continue
        if _is_target_bundle(data):
            return data
        errors.append(f"{source} fenced block {info!r}: missing top-level model/target")

    hint = "expected a fenced ```json fornax-target block with top-level model and target"
    if errors:
        raise TargetContractError(f"{source}: {hint}; checked blocks: " + "; ".join(errors))
    raise TargetContractError(f"{source}: {hint}")


def load_target_contract(path: str | Path) -> tuple[ModelSpec, Target, dict[str, Any]]:
    """Load a Phase-0 target contract from JSON or markdown.

    JSON files may be the bare machine-readable bundle. Markdown files must carry
    a fenced JSON block with top-level `model` and `target` objects so the human
    contract remains executable by `fornax target validate`.
    """

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        bundle = _loads_json_object(text, str(p))
    else:
        bundle = _extract_markdown_bundle(text, str(p))

    if not _is_target_bundle(bundle):
        raise TargetContractError(f"{p}: missing top-level model/target")
    try:
        model = ModelSpec.from_dict(bundle["model"])
        target = Target.from_dict(bundle["target"])
    except ValueError as exc:
        raise TargetContractError(f"{p}: invalid target contract fields: {exc}") from exc
    return model, target, bundle
