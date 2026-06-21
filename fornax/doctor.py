from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark_ledger import validate_benchmark_ledger
from .io import read_json


REQUIRED_JSON = ["inventory.json", "links.json", "placement.json"]
RECOMMENDED_ANY = ["target.json", "v0-target-contract.md"]
RECOMMENDED_JSON = ["validate.json", "simulate.json", "benchmark.json", "calibration.json"]
RECOMMENDED_LEDGER = "benchmark-ledger.jsonl"
SIMULATED_INVENTORY_WARNING = (
    "inventory.json is simulated logical-cluster evidence, not real multi-host "
    "hardware evidence"
)


G1_GATE_ARTIFACT_GROUPS = [
    ("runtime_format_spec", ["runtime-format-and-invariants.md"]),
    ("network_security_spec", ["networking-security-and-backpressure.md"]),
    (
        "substrate_adr",
        [
            "adr/0001-max-mojo-substrate.md",
            "0001-max-mojo-substrate.md",
            "adr-0001-max-mojo-substrate.md",
        ],
    ),
    ("apple_probe", ["apple-expert-mlp-probe.json", "apple-probe.json"]),
    ("apple_probe_validation", ["apple-probe-validation.json"]),
    ("apple_role_decision", ["apple-role-decision.md"]),
    (
        "program_rebaseline",
        ["roadmap-staffing-rebaseline.md", "roadmap-rebaseline.md"],
    ),
]


def _present_files(bundle: Path, names: list[str]) -> list[str]:
    return [name for name in names if (bundle / name).exists()]


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
            simulation = data.get("simulation")
            if isinstance(simulation, dict):
                artifacts[name]["simulation_mode"] = simulation.get("mode")
                artifacts[name]["simulation_profile"] = simulation.get("profile")
                warnings.append(SIMULATED_INVENTORY_WARNING)
            if not isinstance(nodes, list) or not nodes:
                errors.append("inventory.json must contain at least one node")
        if name == "links.json":
            links = data.get("links")
            link_count = len(links) if isinstance(links, list) else None
            artifacts[name]["link_count"] = link_count
            artifacts[name]["measured"] = bool(data.get("measured"))
            artifacts[name]["active_measurement_count"] = data.get(
                "active_measurement_count"
            )
            if not isinstance(links, list):
                errors.append("links.json must contain a links list")
            artifact_warnings = data.get("warnings")
            if isinstance(artifact_warnings, list):
                warnings.extend(f"links.json: {warning}" for warning in artifact_warnings)
            elif link_count and data.get("measured") is False:
                warnings.append("links.json has links but no active fabric measurements")
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
        if name == "calibration.json":
            measured = bool(data.get("measured"))
            artifacts[name]["measured"] = measured
            cuda = data.get("cuda_microprobe")
            if isinstance(cuda, dict):
                artifacts[name]["cuda_measured"] = bool(cuda.get("measured"))
            if not measured:
                warnings.append("calibration.json does not report measured calibration")
            calibration_warnings = data.get("warnings")
            if isinstance(calibration_warnings, list):
                warnings.extend(
                    f"calibration.json: {warning}" for warning in calibration_warnings
                )

    ledger_path = bundle / RECOMMENDED_LEDGER
    if ledger_path.exists():
        ledger = validate_benchmark_ledger(ledger_path)
        artifacts["benchmark_ledger"] = {
            "present": True,
            "valid": bool(ledger.get("ok")),
            "summary": ledger.get("summary", {}),
            "ledger": ledger.get("ledger"),
        }
        if not ledger.get("ok"):
            errors.extend(f"{RECOMMENDED_LEDGER}: {error}" for error in ledger.get("errors", []))
        warnings.extend(f"{RECOMMENDED_LEDGER}: {warning}" for warning in ledger.get("warnings", []))
    else:
        artifacts["benchmark_ledger"] = {"present": False}
        warnings.append(f"missing recommended {RECOMMENDED_LEDGER}")

    for key, names in G1_GATE_ARTIFACT_GROUPS:
        present = _present_files(bundle, names)
        artifacts[key] = {"present": bool(present), "files": present}
        if not present:
            warnings.append(f"missing G1 gate artifact: {key}")

    apple_validation = bundle / "apple-probe-validation.json"
    if apple_validation.exists():
        data, error = _read_json_in_bundle(bundle, "apple-probe-validation.json")
        if error:
            errors.append(error)
        elif not bool(data.get("valid")):
            warnings.append("apple-probe-validation.json is not G1-closable")
        elif isinstance(data.get("recommended_role"), str):
            artifacts["apple_probe_validation"]["recommended_role"] = data.get(
                "recommended_role"
            )

    return {
        "bundle": str(bundle),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "artifacts": artifacts,
    }
