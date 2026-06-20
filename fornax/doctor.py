from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_JSON = ["inventory.json", "links.json", "placement.json"]
RECOMMENDED_ANY = ["target.json", "v0-target-contract.md"]
RECOMMENDED_JSON = ["validate.json", "simulate.json", "benchmark.json"]


def _read_json_in_bundle(bundle: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    path = bundle / name
    if not path.exists():
        return None, f"missing {name}"
    try:
        data = read_json(path)
    except Exception as exc:  # noqa: BLE001 - diagnostics should report parse failures.
        return None, f"{name} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{name} must contain a JSON object"
    return data, None


def inspect_phase0_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path)
    errors: list[str] = []
    warnings: list[str] = []
    artifacts: dict[str, Any] = {}

    if not bundle.exists() or not bundle.is_dir():
        return {
            "bundle": str(bundle),
            "ok": False,
            "errors": ["bundle directory does not exist"],
            "warnings": [],
            "artifacts": {},
        }

    for name in REQUIRED_JSON:
        data, error = _read_json_in_bundle(bundle, name)
        if error:
            errors.append(error)
            continue
        artifacts[name] = {"present": True}
        if name == "inventory.json":
            nodes = data.get("nodes")
            artifacts[name]["node_count"] = len(nodes) if isinstance(nodes, list) else None
            if not isinstance(nodes, list) or not nodes:
                errors.append("inventory.json must contain at least one node")
        if name == "links.json":
            links = data.get("links")
            artifacts[name]["link_count"] = len(links) if isinstance(links, list) else None
            if not isinstance(links, list):
                errors.append("links.json must contain a links list")
        if name == "placement.json":
            feasible = bool(data.get("feasible"))
            artifacts[name]["feasible"] = feasible
            if not feasible:
                errors.append(f"placement.json is infeasible: {data.get('infeasible_reason')}")

    if not any((bundle / name).exists() for name in RECOMMENDED_ANY):
        warnings.append("missing target contract artifact: target.json or v0-target-contract.md")
    else:
        artifacts["target_contract"] = {
            "present": True,
            "files": [name for name in RECOMMENDED_ANY if (bundle / name).exists()],
        }

    for name in RECOMMENDED_JSON:
        path = bundle / name
        if not path.exists():
            warnings.append(f"missing recommended {name}")
            continue
        data, error = _read_json_in_bundle(bundle, name)
        if error:
            errors.append(error)
            continue
        artifacts[name] = {"present": True}
        if name == "validate.json":
            valid = bool(data.get("valid"))
            artifacts[name]["valid"] = valid
            if not valid:
                errors.append("validate.json reports invalid target contract")
        if name == "benchmark.json":
            measured = bool(data.get("measured"))
            artifacts[name]["measured"] = measured
            if not measured:
                warnings.append("benchmark.json is a dry-run prediction, not measured evidence")
        if name == "simulate.json":
            artifacts[name]["has_predicted"] = isinstance(data.get("predicted"), dict)
            if not artifacts[name]["has_predicted"]:
                errors.append("simulate.json missing predicted block")

    return {
        "bundle": str(bundle),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "artifacts": artifacts,
    }
