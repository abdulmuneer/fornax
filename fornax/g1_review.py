from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .doctor import inspect_phase0_bundle
from .io import read_json


def render_g1_gate_review_draft(
    bundle_path: str | Path,
    *,
    review_date: str | None = None,
    plan_version: str = "v3",
) -> dict[str, Any]:
    """Render a G1 gate-review draft from a Phase-0 evidence bundle."""

    bundle = Path(bundle_path)
    doctor = inspect_phase0_bundle(bundle)
    review_date = review_date or date.today().isoformat()
    evidence_rows = _evidence_rows(bundle, doctor)
    machine_missing = [row for row in evidence_rows if not row["machine_ok"]]
    warnings = list(doctor.get("warnings", []))
    machine_complete = not doctor.get("errors") and not machine_missing
    closure_blockers = _closure_blockers(evidence_rows, doctor)
    recommended_outcome = "SPONSOR_DECISION_REQUIRED" if machine_complete else "ITERATE"
    markdown = _render_markdown(
        bundle=bundle,
        review_date=review_date,
        plan_version=plan_version,
        doctor=doctor,
        evidence_rows=evidence_rows,
        closure_blockers=closure_blockers,
        recommended_outcome=recommended_outcome,
    )
    return {
        "markdown": markdown,
        "bundle": str(bundle),
        "review_date": review_date,
        "plan_version": plan_version,
        "machine_complete": machine_complete,
        "gate_ready": machine_complete and not closure_blockers,
        "recommended_outcome": recommended_outcome,
        "machine_missing_criteria": [row["criterion"] for row in machine_missing],
        "closure_blockers": closure_blockers,
        "doctor_errors": list(doctor.get("errors", [])),
        "doctor_warnings": warnings,
    }


def _evidence_rows(bundle: Path, doctor: dict[str, Any]) -> list[dict[str, Any]]:
    validate = _read_json_object(bundle / "validate.json")
    apple_validation = _read_json_object(bundle / "apple-probe-validation.json")

    artifacts = doctor.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}

    target_artifact = artifacts.get("target_contract")
    target_present = isinstance(target_artifact, dict) and bool(
        target_artifact.get("present")
    )
    validation_ok = isinstance(validate, dict) and bool(validate.get("valid"))
    target_signoff = _present_any(
        bundle,
        ["target-contract-signoff.md", "tl-sp-target-signoff.md", "g1-target-signoff.md"],
    )
    persona_ok = _check_passed(validate, "contract.persona_concurrency")
    runtime_present = _artifact_present(artifacts, "runtime_format_spec")
    network_present = _artifact_present(artifacts, "network_security_spec")
    substrate_present = _artifact_present(artifacts, "substrate_adr")
    apple_probe_present = _artifact_present(artifacts, "apple_probe")
    apple_validation_ok = isinstance(apple_validation, dict) and bool(
        apple_validation.get("valid")
    ) and bool(apple_validation.get("gate_closable"))
    apple_role_decision = _artifact_present(artifacts, "apple_role_decision")
    program_rebaseline = _artifact_present(artifacts, "program_rebaseline")
    staffing_signoff = _present_any(
        bundle,
        ["staffing-signoff.md", "g1-staffing-signoff.md", "owner-assignments.md"],
    )
    preflight_machine_ok = bool(doctor.get("ok")) and _has_required_preflight_artifacts(
        artifacts
    )
    golden_artifact = _read_json_object(bundle / "golden-plans.json")
    t0_ok = _golden_plans_passed(golden_artifact)

    memory_and_throughput_ok = _check_passed(validate, "planner.memory_headroom_met") and _check_passed(
        validate, "planner.throughput_threshold_met"
    )

    return [
        {
            "criterion": "v0 target contract reviewed and signed off",
            "evidence": _target_contract_evidence(validate, target_present),
            "machine_ok": target_present and validation_ok,
            "closure_ok": target_present and validation_ok and bool(target_signoff),
            "closure_gap": "missing TL/SP target-contract sign-off artifact",
        },
        {
            "criterion": "memory budget closes with headroom and predicted throughput meets bar",
            "evidence": _target_gate_numbers(validate),
            "machine_ok": memory_and_throughput_ok,
            "closure_ok": memory_and_throughput_ok,
            "closure_gap": "planner memory or throughput gate is not proven",
        },
        {
            "criterion": "concurrency-market evidence supplied or scope narrowed",
            "evidence": _check_detail(validate, "contract.persona_concurrency")
            or "persona concurrency evidence missing from validate.json",
            "machine_ok": persona_ok,
            "closure_ok": persona_ok,
            "closure_gap": "persona saturation concurrency is not proven in the contract",
        },
        {
            "criterion": "Apple reversal trigger evaluated from rank-1 local probe",
            "evidence": _apple_evidence(
                apple_probe_present, apple_validation, apple_role_decision
            ),
            "machine_ok": apple_probe_present and apple_validation_ok and apple_role_decision,
            "closure_ok": apple_probe_present and apple_validation_ok and apple_role_decision,
            "closure_gap": "missing G1-closable Apple probe validation or role decision",
        },
        {
            "criterion": "runtime, networking/security, and substrate ADR reviewed",
            "evidence": _spec_evidence(runtime_present, network_present, substrate_present),
            "machine_ok": runtime_present and network_present and substrate_present,
            "closure_ok": False,
            "closure_gap": "review sign-off for generated specs is not represented in the bundle",
        },
        {
            "criterion": "Phase-0 preflight workflow runnable without oral context",
            "evidence": _doctor_evidence(doctor),
            "machine_ok": preflight_machine_ok,
            "closure_ok": preflight_machine_ok,
            "closure_gap": "doctor reports errors or required preflight artifacts are absent",
        },
        {
            "criterion": "golden-plan tests T0 green",
            "evidence": _golden_evidence(golden_artifact),
            "machine_ok": t0_ok,
            "closure_ok": t0_ok,
            "closure_gap": "missing golden-plans.json evidence generated from `fornax test golden-plans`",
        },
        {
            "criterion": "owners and staffing closed or Sponsor accepts narrowed scope",
            "evidence": _staffing_evidence(program_rebaseline, staffing_signoff),
            "machine_ok": program_rebaseline,
            "closure_ok": program_rebaseline and bool(staffing_signoff),
            "closure_gap": "missing named owner/staffing sign-off artifact",
        },
    ]


def _closure_blockers(
    evidence_rows: list[dict[str, Any]], doctor: dict[str, Any]
) -> list[str]:
    blockers = [f"doctor: {error}" for error in doctor.get("errors", [])]
    for row in evidence_rows:
        if not row["closure_ok"]:
            blockers.append(f"{row['criterion']}: {row['closure_gap']}")
    return blockers


def _render_markdown(
    *,
    bundle: Path,
    review_date: str,
    plan_version: str,
    doctor: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    closure_blockers: list[str],
    recommended_outcome: str,
) -> str:
    lines: list[str] = [
        f"# Gate Review - G1 Evidence / Go-No-Go - {review_date}",
        "",
        f"**Decision authority:** Sponsor - **Advised by:** PM, TL - **Plan version:** {plan_version}",
        "",
        (
            "Status: DRAFT - generated by `fornax program g1-review`; "
            "not Sponsor approval, not DEC-005, and not a G1 closure claim."
        ),
        "",
        "## Entry Criteria",
        "",
        f"- [{'x' if not doctor.get('errors') else ' '}] Phase-0 evidence bundle is doctorable: `{bundle}`",
        "- [ ] Phase-0 evidence sprint complete per S0-1 through S0-9.",
        "- [ ] Human review/sign-off artifacts are attached for items that require TL/SP/Sponsor closure.",
        "",
        "## Evidence Presented",
        "",
        "| Exit criterion | Evidence | Machine evidence | G1 closure |",
        "|---|---|---|---|",
        *[
            "| {criterion} | {evidence} | {machine} | {closure} |".format(
                criterion=_escape_table(row["criterion"]),
                evidence=_escape_table(row["evidence"]),
                machine="yes" if row["machine_ok"] else "no",
                closure="yes" if row["closure_ok"] else f"no - {row['closure_gap']}",
            )
            for row in evidence_rows
        ],
        "",
        "## Risk And Assumption Check",
        "",
        "- R-8 concurrency-market fit: covered only if persona concurrency evidence is machine-complete.",
        "- R-4 / B5 Apple role: covered only if Apple probe validation and role decision are G1-closable.",
        "- I-5 staffing: covered only if owner/staffing sign-off is attached or Sponsor narrows scope.",
        "- I-6 preflight workflow: covered by doctorable bundle and command reproducibility.",
        "",
        "## Doctor Summary",
        "",
        *_bullet_list("Errors", doctor.get("errors", []), empty="none"),
        *_bullet_list("Warnings", doctor.get("warnings", []), empty="none"),
        "",
        "## Closure Blockers",
        "",
        *_bullet_list("Blockers", closure_blockers, empty="none"),
        "",
        "## Decision",
        "",
        f"**Recommended disposition:** {recommended_outcome}",
        "",
        (
            "**Outcome:** PROCEED / ITERATE / NARROW / KILL "
            "(Sponsor must select and record DEC-005)."
        ),
        "",
        "**Rationale:** Fill during gate review.",
        "",
        "**If NARROW:** new scope = <capacity-only / homogeneous-island / other>",
        "**If ITERATE:** unmet criteria + re-review date = <...>",
        "**If KILL:** learning captured in = <...>",
        "",
        "## Actions",
        "",
        "- [ ] Record outcome as DEC-005 in `docs/fornax/program_management/08-decision-log.md`.",
        "- [ ] Re-baseline schedule if Sponsor selects PROCEED or NARROW.",
        "- [ ] Update RAID, sprint board, and `fornax_development_journal.md`.",
        "",
    ]
    return "\n".join(lines)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except Exception:  # noqa: BLE001 - gate draft should degrade to missing evidence.
        return None
    return data if isinstance(data, dict) else None


def _artifact_present(artifacts: dict[str, Any], key: str) -> bool:
    value = artifacts.get(key)
    return isinstance(value, dict) and bool(value.get("present"))


def _present_any(bundle: Path, names: list[str]) -> bool:
    return any((bundle / name).exists() for name in names)


def _has_required_preflight_artifacts(artifacts: dict[str, Any]) -> bool:
    required = (
        "inventory.json",
        "links.json",
        "placement.json",
        "validate.json",
        "simulate.json",
        "benchmark.json",
    )
    return all(
        isinstance(artifacts.get(name), dict) and artifacts[name].get("present")
        for name in required
    )


def _check_passed(validate: dict[str, Any] | None, name: str) -> bool:
    if not isinstance(validate, dict):
        return False
    checks = validate.get("checks")
    if not isinstance(checks, list):
        return False
    return any(
        isinstance(check, dict)
        and check.get("name") == name
        and bool(check.get("passed"))
        for check in checks
    )


def _check_detail(validate: dict[str, Any] | None, name: str) -> str | None:
    if not isinstance(validate, dict):
        return None
    checks = validate.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            detail = check.get("detail")
            return str(detail) if detail is not None else None
    return None


def _target_contract_evidence(validate: dict[str, Any] | None, present: bool) -> str:
    validation = (
        "validate.json valid"
        if isinstance(validate, dict) and validate.get("valid")
        else "validate.json not valid or missing"
    )
    contract = "target contract present" if present else "target contract missing"
    return f"{contract}; {validation}"


def _target_gate_numbers(validate: dict[str, Any] | None) -> str:
    memory = _check_detail(validate, "planner.memory_headroom_met") or "memory headroom missing"
    throughput = _check_detail(validate, "planner.throughput_threshold_met") or "throughput threshold missing"
    return f"{memory}; {throughput}"


def _apple_evidence(
    apple_probe_present: bool,
    apple_validation: dict[str, Any] | None,
    apple_role_decision: bool,
) -> str:
    parts = ["probe present" if apple_probe_present else "probe missing"]
    if isinstance(apple_validation, dict):
        parts.append(
            "validation valid={valid}, gate_closable={closable}, recommended_role={role}".format(
                valid=bool(apple_validation.get("valid")),
                closable=bool(apple_validation.get("gate_closable")),
                role=apple_validation.get("recommended_role", "unset"),
            )
        )
    else:
        parts.append("validation missing")
    parts.append("role decision present" if apple_role_decision else "role decision missing")
    return "; ".join(parts)


def _spec_evidence(
    runtime_present: bool, network_present: bool, substrate_present: bool
) -> str:
    return (
        f"runtime={'present' if runtime_present else 'missing'}; "
        f"network={'present' if network_present else 'missing'}; "
        f"substrate_adr={'present' if substrate_present else 'missing'}"
    )


def _doctor_evidence(doctor: dict[str, Any]) -> str:
    errors = doctor.get("errors", [])
    warnings = doctor.get("warnings", [])
    return (
        f"doctor ok={bool(doctor.get('ok'))}; "
        f"errors={len(errors) if isinstance(errors, list) else 'unknown'}; "
        f"warnings={len(warnings) if isinstance(warnings, list) else 'unknown'}"
    )


def _golden_plans_passed(golden_artifact: dict[str, Any] | None) -> bool:
    if not isinstance(golden_artifact, dict):
        return False
    if isinstance(golden_artifact.get("passed"), bool):
        return bool(golden_artifact["passed"])
    results = golden_artifact.get("results")
    if not isinstance(results, list) or not results:
        return False
    return all(isinstance(item, dict) and bool(item.get("passed")) for item in results)


def _golden_evidence(golden_artifact: dict[str, Any] | None) -> str:
    if _golden_plans_passed(golden_artifact):
        return "golden-plans.json reports all T0 golden plans passed"
    if golden_artifact is None:
        return "golden-plans.json missing from bundle"
    return "golden-plans.json present but not passing"


def _staffing_evidence(program_rebaseline: bool, staffing_signoff: bool) -> str:
    parts = [
        "program rebaseline present" if program_rebaseline else "program rebaseline missing",
        "staffing sign-off present" if staffing_signoff else "staffing sign-off missing",
    ]
    return "; ".join(parts)


def _bullet_list(header: str, values: Any, *, empty: str) -> list[str]:
    rows = [f"### {header}", ""]
    if not isinstance(values, list) or not values:
        rows.append(f"- {empty}")
        rows.append("")
        return rows
    rows.extend(f"- {value}" for value in values)
    rows.append("")
    return rows


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
