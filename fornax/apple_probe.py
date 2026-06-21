from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


APPLE_PROBE_KIND = "apple-expert-mlp"
SIMULATED_APPLE_PROBE_KIND = "apple-expert-mlp-simulation"
SIMULATED_APPLE_ROLE_VALUES = ("capacity-only", "expert-worker")


def apple_probe_template(
    *,
    target_model: str = "target-model",
    pinned_build: str = "unset",
    threshold_tokens_s: float = 1.0,
) -> dict[str, Any]:
    """Return the Phase-0 Apple expert-MLP probe artifact template."""

    return {
        "version": 1,
        "probe_kind": APPLE_PROBE_KIND,
        "target_model": {
            "name": target_model,
            "expert_mlp_shape": "fill with gate/up/down projection dimensions",
            "quantization": "fill with target quantization",
            "activation_dtype": "fill with in-flight activation dtype",
        },
        "environment": {
            "hardware": "target Apple Silicon Mac SKU and memory",
            "os": "macOS build",
            "max_build": pinned_build,
            "mojo_build": "mojo --version output",
            "command": "exact probe command",
            "log_path": "artifact log path",
            "profiler": "Instruments, Metal, MAX, Mojo, or timing harness",
        },
        "probe": {
            "rank": 1,
            "measured": False,
            "local_to_target_mac": True,
            "input_fixture": "path or hash of deterministic input fixture",
            "output_checksum": "fill after run",
        },
        "correctness": {
            "passed": False,
            "max_abs_error": None,
            "max_rel_error": None,
            "tolerance_source": "runtime-format-and-invariants.md",
        },
        "throughput": {
            "measured": False,
            "tokens_s": None,
            "threshold_tokens_s": threshold_tokens_s,
            "warmup_iterations": 0,
            "measurement_iterations": 0,
            "thermal_notes": "record power mode, throttling, and steady-state notes",
        },
        "decision": {
            "requested_role": "expert-worker",
            "demote_role": "capacity-only",
        },
        "notes": [
            "This template is not evidence until measured on the pinned build.",
            "A measured correctness or throughput miss is still usable evidence: it demotes Apple to capacity-only for G1.",
        ],
    }


def simulated_apple_probe_artifact(
    *,
    target_model: str = "target-model",
    pinned_build: str = "unset",
    recommended_role: str = "capacity-only",
    reason: str = "Simulated development evidence until rank-1 Apple probe is available.",
) -> dict[str, Any]:
    """Return a development-only Apple role simulation artifact.

    This artifact exercises milestone/status plumbing. It is deliberately not
    accepted by validate_apple_probe_artifact and cannot close G1.
    """

    if recommended_role not in SIMULATED_APPLE_ROLE_VALUES:
        raise ValueError(
            "recommended_role must be one of: "
            + ", ".join(SIMULATED_APPLE_ROLE_VALUES)
        )
    return {
        "version": 1,
        "probe_kind": SIMULATED_APPLE_PROBE_KIND,
        "simulation": {
            "mode": "development_only",
            "source": "fornax.apple_probe.simulated_apple_probe_artifact",
            "warning": (
                "Simulation evidence only; this does not replace the rank-1 "
                "local Apple probe required for G1."
            ),
        },
        "target_model": {"name": target_model},
        "environment": {
            "hardware": "simulated Apple Silicon target",
            "max_build": pinned_build,
        },
        "decision": {
            "recommended_role": recommended_role,
            "gate_closable": False,
            "rationale": reason,
        },
        "notes": [
            "Development-only artifact for simulated milestone validation.",
            "Do not write this data to apple-probe-validation.json.",
        ],
    }


def render_simulated_apple_role_decision(
    artifact: dict[str, Any], *, source: str = "apple-probe-simulation.json"
) -> str:
    decision = artifact.get("decision") if isinstance(artifact.get("decision"), dict) else {}
    simulation = artifact.get("simulation") if isinstance(artifact.get("simulation"), dict) else {}
    target = artifact.get("target_model") if isinstance(artifact.get("target_model"), dict) else {}
    return "\n".join(
        [
            "# Simulated Apple Role Decision",
            "",
            "Status: DEVELOPMENT SIMULATION - not G1 closure evidence.",
            "",
            "## Source",
            "",
            f"- Artifact: `{source}`",
            f"- Probe kind: {artifact.get('probe_kind', 'missing')}",
            f"- Target model: {target.get('name', 'missing')}",
            f"- Simulation mode: {simulation.get('mode', 'missing')}",
            "",
            "## Simulated Decision",
            "",
            f"- Recommended role: `{decision.get('recommended_role', 'missing')}`",
            f"- Gate closable: {decision.get('gate_closable', False)}",
            f"- Rationale: {decision.get('rationale', 'missing')}",
            "",
            "## Gate Interpretation",
            "",
            (
                "This artifact can validate program/status plumbing in a "
                "simulated development bundle, but the real G1 Apple "
                "criterion still requires a measured rank-1 local probe "
                "and role decision on the pinned target Mac build."
            ),
            "",
        ]
    )


def validate_apple_probe_file(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    return validate_apple_probe_artifact(data, source=str(path))


def validate_apple_probe_artifact(
    data: Any, *, source: str = "in-memory"
) -> dict[str, Any]:
    """Validate whether the Apple expert-MLP probe can decide Apple's role."""

    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return {
            "valid": False,
            "gate_closable": False,
            "source": source,
            "recommended_role": "undecided",
            "outcome": "incomplete",
            "performance_passed": False,
            "correctness_passed": False,
            "throughput_passed": False,
            "checks": [
                {"name": "artifact_object", "passed": False, "detail": "artifact must be a JSON object"}
            ],
            "errors": ["artifact must be a JSON object"],
            "warnings": [],
        }

    _check_equal(checks, errors, "version", data.get("version"), 1)
    _check_equal(checks, errors, "probe_kind", data.get("probe_kind"), APPLE_PROBE_KIND)

    target = _object(data, "target_model", checks, errors)
    _required_string(target, "target_model.name", checks, errors)
    _required_present(target, "target_model.expert_mlp_shape", checks, errors)
    _required_string(target, "target_model.quantization", checks, errors)
    _required_string(target, "target_model.activation_dtype", checks, errors)

    env = _object(data, "environment", checks, errors)
    _required_string(env, "environment.hardware", checks, errors)
    _required_string(env, "environment.os", checks, errors)
    max_build = _required_string(env, "environment.max_build", checks, errors)
    _required_string(env, "environment.command", checks, errors)
    _required_string(env, "environment.log_path", checks, errors)
    _required_string(env, "environment.mojo_build", checks, errors)
    _required_string(env, "environment.profiler", checks, errors)
    if isinstance(max_build, str) and max_build.strip().lower() in {"", "unset", "unknown"}:
        _add_check(checks, errors, "environment.max_build_pinned", False, "max_build must name a pinned build")
    else:
        _add_check(checks, errors, "environment.max_build_pinned", True, "pinned build is named")

    probe = _object(data, "probe", checks, errors)
    _check_equal(checks, errors, "probe.rank", probe.get("rank"), 1)
    _check_bool(checks, errors, "probe.measured", probe.get("measured"), expected=True)
    _check_bool(
        checks,
        errors,
        "probe.local_to_target_mac",
        probe.get("local_to_target_mac"),
        expected=True,
    )
    _required_string(probe, "probe.input_fixture", checks, errors)
    _required_string(probe, "probe.output_checksum", checks, errors)

    correctness = _object(data, "correctness", checks, errors)
    correctness_passed = correctness.get("passed") is True
    _check_is_bool(checks, errors, "correctness.passed", correctness.get("passed"))
    _required_string(correctness, "correctness.tolerance_source", checks, errors)
    if correctness.get("max_abs_error") is None:
        warnings.append("correctness.max_abs_error is not recorded")
    if correctness.get("max_rel_error") is None:
        warnings.append("correctness.max_rel_error is not recorded")

    throughput = _object(data, "throughput", checks, errors)
    _check_bool(checks, errors, "throughput.measured", throughput.get("measured"), expected=True)
    tokens_s = _required_number(throughput, "throughput.tokens_s", checks, errors)
    threshold = _required_number(throughput, "throughput.threshold_tokens_s", checks, errors)
    measurement_iterations = _required_number(
        throughput,
        "throughput.measurement_iterations",
        checks,
        errors,
    )
    _required_string(throughput, "throughput.thermal_notes", checks, errors)
    if isinstance(measurement_iterations, (int, float)) and measurement_iterations <= 0:
        _add_check(
            checks,
            errors,
            "throughput.measurement_iterations_positive",
            False,
            "measurement_iterations must be > 0",
        )
    elif isinstance(measurement_iterations, (int, float)):
        _add_check(
            checks,
            errors,
            "throughput.measurement_iterations_positive",
            True,
            "measurement_iterations is positive",
        )
    if isinstance(threshold, (int, float)) and threshold <= 0:
        _add_check(
            checks,
            errors,
            "throughput.threshold_tokens_s_positive",
            False,
            "threshold_tokens_s must be > 0",
        )
    elif isinstance(threshold, (int, float)):
        _add_check(
            checks,
            errors,
            "throughput.threshold_tokens_s_positive",
            True,
            "threshold_tokens_s is positive",
        )
    throughput_passed = (
        isinstance(tokens_s, (int, float))
        and isinstance(threshold, (int, float))
        and threshold > 0
        and tokens_s >= threshold
    )

    evidence_complete = not errors
    if evidence_complete and correctness_passed and throughput_passed:
        recommended_role = "expert-worker"
        outcome = "expert-worker-pass"
    elif evidence_complete:
        recommended_role = "capacity-only"
        outcome = "capacity-only-demotion"
    else:
        recommended_role = "undecided"
        outcome = "incomplete"

    _add_check(
        checks,
        errors,
        "role_decision_closable",
        evidence_complete,
        "measured rank-1 evidence can decide role" if evidence_complete else "rank-1 evidence is incomplete",
    )
    return {
        "valid": evidence_complete,
        "gate_closable": evidence_complete,
        "source": source,
        "recommended_role": recommended_role,
        "outcome": outcome,
        "performance_passed": bool(correctness_passed and throughput_passed),
        "correctness_passed": bool(correctness_passed),
        "throughput_passed": bool(throughput_passed),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def render_apple_role_decision_draft(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    validation = validate_apple_probe_artifact(data, source=str(path))
    markdown = _render_role_markdown(path=Path(path), data=data, validation=validation)
    return {"markdown": markdown, "validation": validation}


def _render_role_markdown(
    *, path: Path, data: dict[str, Any], validation: dict[str, Any]
) -> str:
    target = data.get("target_model") if isinstance(data.get("target_model"), dict) else {}
    env = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    throughput = data.get("throughput") if isinstance(data.get("throughput"), dict) else {}
    correctness = data.get("correctness") if isinstance(data.get("correctness"), dict) else {}
    if validation.get("valid"):
        closure = "G1 role decision evidence is complete."
    else:
        closure = "Not a G1 closure claim; rank-1 evidence is incomplete."
    lines = [
        "# Apple Expert-MLP Role Decision Draft",
        "",
        "Status: DRAFT - generated by `fornax apple role-decision`; " + closure,
        "",
        "## Probe Source",
        "",
        f"- Probe artifact: `{path}`",
        f"- Probe kind: {data.get('probe_kind', 'missing')}",
        f"- Target model: {target.get('name', 'missing')}",
        f"- Quantization: {target.get('quantization', 'missing')}",
        f"- Activation dtype: {target.get('activation_dtype', 'missing')}",
        f"- Hardware: {env.get('hardware', 'missing')}",
        f"- OS: {env.get('os', 'missing')}",
        f"- MAX build: {env.get('max_build', 'missing')}",
        f"- Mojo build: {env.get('mojo_build', 'missing')}",
        f"- Command: `{env.get('command', 'missing')}`",
        f"- Log path: `{env.get('log_path', 'missing')}`",
        "",
        "## Decision",
        "",
        f"- Recommended Apple role: `{validation.get('recommended_role')}`",
        f"- Outcome: `{validation.get('outcome')}`",
        f"- Gate closable from this artifact: {validation.get('gate_closable')}",
        f"- Correctness passed: {validation.get('correctness_passed')}",
        f"- Throughput passed: {validation.get('throughput_passed')}",
        f"- Measured tokens/s: {throughput.get('tokens_s', 'missing')}",
        f"- Threshold tokens/s: {throughput.get('threshold_tokens_s', 'missing')}",
        f"- Max abs error: {correctness.get('max_abs_error', 'missing')}",
        f"- Max rel error: {correctness.get('max_rel_error', 'missing')}",
        "",
        "## Checks",
        "",
        *_checks_table(validation),
        "",
        "## Source Precedence Reminder",
        "",
        "Rank 1 local probe on the pinned build decides Apple's v0 role. Upstream docs, supported-model pages, blogs, and nightlies can motivate this probe but cannot replace it.",
        "",
        "## Gate Interpretation",
        "",
        "- `expert-worker` means the target expert MLP passed correctness and throughput on the pinned target Mac build.",
        "- `capacity-only` means measured evidence exists but correctness or throughput missed the contract; Apple is demoted for v0.",
        "- `undecided` means this artifact cannot close G1 and the probe must be rerun or Sponsor must explicitly narrow scope at G1.",
        "",
    ]
    return "\n".join(lines)


def _checks_table(validation: dict[str, Any]) -> list[str]:
    rows = ["| Check | Passed | Detail |", "|---|---:|---|"]
    checks = validation.get("checks") if isinstance(validation.get("checks"), list) else []
    if not checks:
        rows.append("| n/a | no | no checks recorded |")
        return rows
    for check in checks:
        if not isinstance(check, dict):
            continue
        rows.append(
            "| {name} | {passed} | {detail} |".format(
                name=_escape_table(check.get("name", "missing")),
                passed="yes" if check.get("passed") else "no",
                detail=_escape_table(check.get("detail", "")),
            )
        )
    return rows


def _object(
    data: dict[str, Any], key: str, checks: list[dict[str, Any]], errors: list[str]
) -> dict[str, Any]:
    value = data.get(key)
    passed = isinstance(value, dict)
    _add_check(checks, errors, key, passed, "object present" if passed else "missing object")
    return value if isinstance(value, dict) else {}


def _required_present(
    data: dict[str, Any], path: str, checks: list[dict[str, Any]], errors: list[str]
) -> Any:
    key = path.split(".")[-1]
    value = data.get(key)
    passed = value is not None and value != ""
    _add_check(checks, errors, path, passed, "present" if passed else "missing")
    return value


def _required_string(
    data: dict[str, Any],
    path: str,
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    required: bool = True,
) -> str | None:
    key = path.split(".")[-1]
    value = data.get(key)
    passed = isinstance(value, str) and value.strip() != ""
    if required or value not in (None, ""):
        _add_check(checks, errors, path, passed, "string present" if passed else "missing string")
    return value if isinstance(value, str) else None


def _required_number(
    data: dict[str, Any], path: str, checks: list[dict[str, Any]], errors: list[str]
) -> float | None:
    key = path.split(".")[-1]
    value = data.get(key)
    passed = isinstance(value, (int, float)) and not isinstance(value, bool)
    _add_check(checks, errors, path, passed, "number present" if passed else "missing number")
    return float(value) if passed else None


def _check_equal(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    passed = actual == expected
    _add_check(checks, errors, name, passed, f"expected {expected!r}, got {actual!r}")


def _check_bool(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    actual: Any,
    *,
    expected: bool,
) -> None:
    passed = actual is expected
    _add_check(checks, errors, name, passed, f"expected {expected!r}, got {actual!r}")


def _check_is_bool(
    checks: list[dict[str, Any]], errors: list[str], name: str, actual: Any
) -> None:
    passed = isinstance(actual, bool)
    _add_check(checks, errors, name, passed, "boolean present" if passed else "missing boolean")


def _add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|")
