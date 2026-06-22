from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .doctor import inspect_phase0_bundle
from .g1_review import render_g1_gate_review_draft
from .phase0_status import render_phase0_status_report


RECORD_KIND = "g1-evidence-packet"
MODE = "phase0-g1-review"
SIMULATION_METHOD = "preflight-bundle-evidence-packet"

REQUIRED_EVIDENCE_IDS = {
    "target_contract",
    "target_validation",
    "memory_throughput",
    "persona_concurrency",
    "runtime_format_spec",
    "network_security_spec",
    "substrate_adr",
    "apple_probe",
    "apple_probe_validation",
    "apple_role_decision",
    "program_rebaseline",
    "golden_plans",
    "preflight_workflow",
    "benchmark_ledger",
}

REQUIRED_SIGNOFF_IDS = {
    "target_contract_signoff",
    "spec_review_signoff",
    "apple_role_decision_signoff",
    "staffing_signoff",
    "sponsor_dec005",
}


def _present(doctor: dict[str, Any], key: str) -> bool:
    artifacts = doctor.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    value = artifacts.get(key)
    return isinstance(value, dict) and bool(value.get("present"))


def _artifact_files(doctor: dict[str, Any], key: str) -> list[str]:
    artifacts = doctor.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    value = artifacts.get(key)
    if not isinstance(value, dict):
        return []
    files = value.get("files")
    return [str(item) for item in files] if isinstance(files, list) else []


def _present_any(bundle: Path, names: list[str]) -> tuple[bool, list[str]]:
    present = [name for name in names if (bundle / name).exists()]
    return bool(present), present


def _evidence_item(
    *,
    item_id: str,
    title: str,
    workstream: str,
    status: str,
    evidence: str,
    gate_closable: bool,
    caveat: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": title,
        "workstream": workstream,
        "status": status,
        "evidence": evidence,
        "gate_closable": gate_closable,
        "caveat": caveat,
    }


def _status_from_present(present: bool) -> str:
    return "present" if present else "missing"


def _signoff_requirement(
    *,
    req_id: str,
    title: str,
    expected_files: list[str],
    present_files: list[str],
    required_for_g1: bool = True,
) -> dict[str, Any]:
    return {
        "id": req_id,
        "title": title,
        "required_for_g1": required_for_g1,
        "present": bool(present_files),
        "expected_files": expected_files,
        "present_files": present_files,
    }


def build_g1_evidence_packet(
    bundle_path: str | Path,
    *,
    packet_date: str | None = None,
    plan_version: str = "v3",
) -> dict[str, Any]:
    """Build a machine-checkable G1 evidence packet from a Phase-0 bundle."""

    bundle = Path(bundle_path)
    packet_date = packet_date or date.today().isoformat()
    doctor = inspect_phase0_bundle(bundle)
    g1 = render_g1_gate_review_draft(
        bundle,
        review_date=packet_date,
        plan_version=plan_version,
    )
    phase0_status = render_phase0_status_report(
        bundle,
        report_date=packet_date,
        plan_version=plan_version,
    )

    validate_present = _present(doctor, "validate.json")
    validate_valid = bool(
        isinstance(doctor.get("artifacts"), dict)
        and isinstance(doctor["artifacts"].get("validate.json"), dict)
        and doctor["artifacts"]["validate.json"].get("valid")
    )
    benchmark_ledger_present = _present(doctor, "benchmark_ledger")
    benchmark_ledger_valid = bool(
        isinstance(doctor.get("artifacts"), dict)
        and isinstance(doctor["artifacts"].get("benchmark_ledger"), dict)
        and doctor["artifacts"]["benchmark_ledger"].get("valid")
    )
    machine_missing = list(g1.get("machine_missing_criteria", []))
    closure_blockers = list(g1.get("closure_blockers", []))

    evidence_items = [
        _evidence_item(
            item_id="target_contract",
            title="v0 target contract artifact",
            workstream="S0-2/A5",
            status=_status_from_present(_present(doctor, "target_contract")),
            evidence=", ".join(_artifact_files(doctor, "target_contract")) or "missing",
            gate_closable=False,
            caveat="requires TL/SP sign-off before G1 closure",
        ),
        _evidence_item(
            item_id="target_validation",
            title="target contract machine validation",
            workstream="S0-2/A5",
            status="valid" if validate_present and validate_valid else "missing_or_invalid",
            evidence="validate.json valid" if validate_valid else "validate.json missing or invalid",
            gate_closable=validate_valid,
            caveat="machine validation is not human sign-off",
        ),
        _evidence_item(
            item_id="memory_throughput",
            title="memory budget and throughput bar",
            workstream="S0-2/A2",
            status="machine_checked"
            if "memory budget closes with headroom and predicted throughput meets bar"
            not in machine_missing
            else "missing_or_failed",
            evidence="derived from validate.json and planner output",
            gate_closable=(
                "memory budget closes with headroom and predicted throughput meets bar"
                not in machine_missing
            ),
            caveat="simulation/planner numbers still need Sponsor acceptance",
        ),
        _evidence_item(
            item_id="persona_concurrency",
            title="concurrency-market evidence",
            workstream="S0-3/A1",
            status="machine_checked"
            if "concurrency-market evidence supplied or scope narrowed" not in machine_missing
            else "missing_or_failed",
            evidence="contract.persona_concurrency check in validate.json",
            gate_closable=(
                "concurrency-market evidence supplied or scope narrowed" not in machine_missing
            ),
            caveat="persona evidence remains a product/market assumption until accepted",
        ),
        _evidence_item(
            item_id="runtime_format_spec",
            title="runtime-format-and-invariants.md",
            workstream="S0-4/B1",
            status=_status_from_present(_present(doctor, "runtime_format_spec")),
            evidence=", ".join(_artifact_files(doctor, "runtime_format_spec")) or "missing",
            gate_closable=False,
            caveat="generated draft requires review sign-off",
        ),
        _evidence_item(
            item_id="network_security_spec",
            title="networking-security-and-backpressure.md",
            workstream="S0-5/E1",
            status=_status_from_present(_present(doctor, "network_security_spec")),
            evidence=", ".join(_artifact_files(doctor, "network_security_spec")) or "missing",
            gate_closable=False,
            caveat="generated draft requires review sign-off",
        ),
        _evidence_item(
            item_id="substrate_adr",
            title="ADR-0001 MAX/Mojo substrate",
            workstream="S0-6/B5",
            status=_status_from_present(_present(doctor, "substrate_adr")),
            evidence=", ".join(_artifact_files(doctor, "substrate_adr")) or "missing",
            gate_closable=False,
            caveat="generated draft requires accepted ADR decision",
        ),
        _evidence_item(
            item_id="apple_probe",
            title="Apple expert-MLP probe template or artifact",
            workstream="S0-7/D2-D4",
            status=_status_from_present(_present(doctor, "apple_probe")),
            evidence=", ".join(_artifact_files(doctor, "apple_probe")) or "missing",
            gate_closable=False,
            caveat="template/simulation is not rank-1 Apple probe evidence",
        ),
        _evidence_item(
            item_id="apple_probe_validation",
            title="Apple probe validation",
            workstream="S0-7/D2-D4",
            status=_status_from_present(_present(doctor, "apple_probe_validation")),
            evidence=", ".join(_artifact_files(doctor, "apple_probe_validation")) or "missing",
            gate_closable=(
                "Apple reversal trigger evaluated from rank-1 local probe"
                not in machine_missing
            ),
            caveat="must be valid and gate-closable, not simulated",
        ),
        _evidence_item(
            item_id="apple_role_decision",
            title="Apple role decision",
            workstream="S0-7/D4",
            status=_status_from_present(_present(doctor, "apple_role_decision")),
            evidence=", ".join(_artifact_files(doctor, "apple_role_decision")) or "missing",
            gate_closable=(
                "Apple reversal trigger evaluated from rank-1 local probe"
                not in machine_missing
            ),
            caveat="Sponsor/TL must accept Apple role before G1 closure",
        ),
        _evidence_item(
            item_id="program_rebaseline",
            title="roadmap and staffing rebaseline",
            workstream="S0-8/X1-X3",
            status=_status_from_present(_present(doctor, "program_rebaseline")),
            evidence=", ".join(_artifact_files(doctor, "program_rebaseline")) or "missing",
            gate_closable=False,
            caveat="requires named owner/staffing sign-off or Sponsor narrowing",
        ),
        _evidence_item(
            item_id="golden_plans",
            title="T0 golden plans",
            workstream="S0-1/A4",
            status="machine_checked"
            if "golden-plan tests T0 green" not in machine_missing
            else "missing_or_failed",
            evidence="golden-plans.json" if (bundle / "golden-plans.json").exists() else "missing",
            gate_closable="golden-plan tests T0 green" not in machine_missing,
            caveat="T0 only; not distributed runtime evidence",
        ),
        _evidence_item(
            item_id="preflight_workflow",
            title="Phase-0 preflight workflow",
            workstream="S0-9/A6",
            status="doctorable" if doctor.get("ok") else "doctor_errors",
            evidence=f"doctor errors={len(doctor.get('errors', []))}",
            gate_closable=bool(doctor.get("ok")),
            caveat="doctorable bundle still needs human closure artifacts",
        ),
        _evidence_item(
            item_id="benchmark_ledger",
            title="benchmark ledger reproducibility",
            workstream="S0-2/QA",
            status="valid" if benchmark_ledger_present and benchmark_ledger_valid else "missing_or_invalid",
            evidence="benchmark-ledger.jsonl valid"
            if benchmark_ledger_valid
            else "benchmark-ledger.jsonl missing or invalid",
            gate_closable=benchmark_ledger_valid,
            caveat="simulation/local benchmark evidence is not T3/T4 hardware evidence",
        ),
    ]

    target_signoff, target_signoff_files = _present_any(
        bundle,
        ["target-contract-signoff.md", "tl-sp-target-signoff.md", "g1-target-signoff.md"],
    )
    spec_signoff, spec_signoff_files = _present_any(
        bundle,
        [
            "spec-review-signoff.md",
            "runtime-format-signoff.md",
            "network-security-signoff.md",
            "substrate-adr-signoff.md",
        ],
    )
    apple_signoff, apple_signoff_files = _present_any(
        bundle,
        ["apple-role-decision.md", "apple-role-signoff.md", "g1-apple-role-signoff.md"],
    )
    staffing_signoff, staffing_signoff_files = _present_any(
        bundle,
        ["staffing-signoff.md", "g1-staffing-signoff.md", "owner-assignments.md"],
    )
    dec005, dec005_files = _present_any(
        bundle,
        ["dec-005.md", "DEC-005.md", "g1-sponsor-decision.md"],
    )
    signoff_requirements = [
        _signoff_requirement(
            req_id="target_contract_signoff",
            title="TL/SP target-contract sign-off",
            expected_files=[
                "target-contract-signoff.md",
                "tl-sp-target-signoff.md",
                "g1-target-signoff.md",
            ],
            present_files=target_signoff_files,
        ),
        _signoff_requirement(
            req_id="spec_review_signoff",
            title="runtime/network/substrate review sign-off",
            expected_files=[
                "spec-review-signoff.md",
                "runtime-format-signoff.md",
                "network-security-signoff.md",
                "substrate-adr-signoff.md",
            ],
            present_files=spec_signoff_files,
        ),
        _signoff_requirement(
            req_id="apple_role_decision_signoff",
            title="Apple role decision accepted for G1",
            expected_files=[
                "apple-role-decision.md",
                "apple-role-signoff.md",
                "g1-apple-role-signoff.md",
            ],
            present_files=apple_signoff_files,
        ),
        _signoff_requirement(
            req_id="staffing_signoff",
            title="staffing or narrowed-scope sign-off",
            expected_files=[
                "staffing-signoff.md",
                "g1-staffing-signoff.md",
                "owner-assignments.md",
            ],
            present_files=staffing_signoff_files,
        ),
        _signoff_requirement(
            req_id="sponsor_dec005",
            title="Sponsor DEC-005 outcome",
            expected_files=["dec-005.md", "DEC-005.md", "g1-sponsor-decision.md"],
            present_files=dec005_files,
        ),
    ]
    human_signoff_complete = all(
        item["present"] for item in signoff_requirements if item["required_for_g1"]
    )
    simulation_present = bool(phase0_status.get("simulation", {}).get("present"))
    summary = {
        "machine_complete": bool(g1.get("machine_complete")),
        "g1_gate_ready": bool(g1.get("gate_ready")),
        "recommended_outcome": g1.get("recommended_outcome"),
        "machine_missing_count": len(machine_missing),
        "closure_blocker_count": len(closure_blockers),
        "human_signoff_complete": human_signoff_complete,
        "target_signoff_present": target_signoff,
        "spec_signoff_present": spec_signoff,
        "apple_signoff_present": apple_signoff,
        "staffing_signoff_present": staffing_signoff,
        "sponsor_dec005_present": dec005,
        "simulation_present": simulation_present,
        "simulation_only": simulation_present and not bool(g1.get("gate_ready")),
        "g2_g3_gate_evidence": False,
    }
    packet: dict[str, Any] = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "simulation_method": SIMULATION_METHOD,
        "bundle": str(bundle),
        "packet_date": packet_date,
        "plan_version": plan_version,
        "summary": summary,
        "machine_missing_criteria": machine_missing,
        "closure_blockers": closure_blockers,
        "evidence_items": evidence_items,
        "signoff_requirements": signoff_requirements,
        "doctor_errors": list(doctor.get("errors", [])),
        "doctor_warnings": list(doctor.get("warnings", [])),
        "phase0_status_summary": phase0_status.get("summary", {}),
        "markdown": "",
    }
    packet["markdown"] = render_g1_evidence_packet_markdown(packet)
    return packet


def render_g1_evidence_packet_markdown(packet: dict[str, Any]) -> str:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    lines = [
        f"# G1 Evidence Packet - {packet.get('packet_date', 'unknown')}",
        "",
        (
            "Status: DRAFT - generated by `fornax program g1-evidence-packet`; "
            "not Sponsor approval, not DEC-005, and not a G1 closure claim."
        ),
        "",
        f"- Bundle: `{packet.get('bundle', 'unknown')}`",
        f"- Plan version: `{packet.get('plan_version', 'unknown')}`",
        f"- Machine complete: `{summary.get('machine_complete')}`",
        f"- G1 gate ready: `{summary.get('g1_gate_ready')}`",
        f"- Recommended outcome: `{summary.get('recommended_outcome')}`",
        f"- Human sign-off complete: `{summary.get('human_signoff_complete')}`",
        f"- Simulation-only scope: `{summary.get('simulation_only')}`",
        "",
        "## Evidence Items",
        "",
        "| ID | Workstream | Status | G1 closable | Caveat |",
        "|---|---|---|---|---|",
    ]
    for item in packet.get("evidence_items", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "| {id} | {workstream} | {status} | {closable} | {caveat} |".format(
                id=_escape_table(item.get("id")),
                workstream=_escape_table(item.get("workstream")),
                status=_escape_table(item.get("status")),
                closable="yes" if item.get("gate_closable") else "no",
                caveat=_escape_table(item.get("caveat")),
            )
        )
    lines.extend(
        [
            "",
            "## Sign-off Requirements",
            "",
            "| ID | Present | Expected files | Present files |",
            "|---|---|---|---|",
        ]
    )
    for item in packet.get("signoff_requirements", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "| {id} | {present} | {expected} | {present_files} |".format(
                id=_escape_table(item.get("id")),
                present="yes" if item.get("present") else "no",
                expected=_escape_table(", ".join(item.get("expected_files", []))),
                present_files=_escape_table(", ".join(item.get("present_files", [])) or "missing"),
            )
        )
    lines.extend(["", "## Machine Missing Criteria", ""])
    lines.extend(_bullet_list(packet.get("machine_missing_criteria", []), empty="none"))
    lines.extend(["", "## Closure Blockers", ""])
    lines.extend(_bullet_list(packet.get("closure_blockers", []), empty="none"))
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "- The packet can make G1 review reproducible.",
            "- The packet cannot replace TL/SP/Sponsor sign-off.",
            "- Simulation/local two-GPU evidence must not be reported as real G2/G3 evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_g1_evidence_packet_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "G1 evidence packet is review preparation, not Sponsor approval or DEC-005"
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if data.get("simulation_method") != SIMULATION_METHOD:
        errors.append(f"simulation_method must be {SIMULATION_METHOD}")
    for field in ["bundle", "packet_date", "plan_version", "markdown"]:
        if not isinstance(data.get(field), str) or not data.get(field):
            errors.append(f"{field} must be a non-empty string")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    for field in [
        "machine_complete",
        "g1_gate_ready",
        "human_signoff_complete",
        "simulation_present",
        "simulation_only",
        "g2_g3_gate_evidence",
    ]:
        if not isinstance(summary.get(field), bool):
            errors.append(f"summary.{field} must be a boolean")
    if summary.get("g2_g3_gate_evidence") is not False:
        errors.append("summary.g2_g3_gate_evidence must be false")

    machine_missing = data.get("machine_missing_criteria")
    if not isinstance(machine_missing, list):
        errors.append("machine_missing_criteria must be a list")
        machine_missing = []
    closure_blockers = data.get("closure_blockers")
    if not isinstance(closure_blockers, list):
        errors.append("closure_blockers must be a list")
        closure_blockers = []
    if summary.get("machine_missing_count") != len(machine_missing):
        errors.append("summary.machine_missing_count must match machine_missing_criteria")
    if summary.get("closure_blocker_count") != len(closure_blockers):
        errors.append("summary.closure_blocker_count must match closure_blockers")
    if summary.get("machine_complete") is True and machine_missing:
        errors.append("summary.machine_complete cannot be true with missing machine criteria")
    if summary.get("g1_gate_ready") is True and closure_blockers:
        errors.append("summary.g1_gate_ready cannot be true with closure blockers")
    if summary.get("g1_gate_ready") is True and summary.get("human_signoff_complete") is not True:
        errors.append("summary.g1_gate_ready requires human_signoff_complete")

    evidence_items = data.get("evidence_items")
    if not isinstance(evidence_items, list) or not evidence_items:
        errors.append("evidence_items must be a non-empty list")
        evidence_items = []
    evidence_ids: set[str] = set()
    for index, item in enumerate(evidence_items):
        field = f"evidence_items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{field} must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{field}.id must be a non-empty string")
        else:
            evidence_ids.add(item_id)
        for key in ["title", "workstream", "status", "evidence", "caveat"]:
            if not isinstance(item.get(key), str) or not item.get(key):
                errors.append(f"{field}.{key} must be a non-empty string")
        if not isinstance(item.get("gate_closable"), bool):
            errors.append(f"{field}.gate_closable must be a boolean")
    missing_evidence = REQUIRED_EVIDENCE_IDS - evidence_ids
    if missing_evidence:
        errors.append(f"missing evidence items: {sorted(missing_evidence)}")

    signoffs = data.get("signoff_requirements")
    if not isinstance(signoffs, list) or not signoffs:
        errors.append("signoff_requirements must be a non-empty list")
        signoffs = []
    signoff_ids: set[str] = set()
    required_signoffs = []
    for index, item in enumerate(signoffs):
        field = f"signoff_requirements[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{field} must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{field}.id must be a non-empty string")
        else:
            signoff_ids.add(item_id)
        if not isinstance(item.get("present"), bool):
            errors.append(f"{field}.present must be a boolean")
        if not isinstance(item.get("required_for_g1"), bool):
            errors.append(f"{field}.required_for_g1 must be a boolean")
        if item.get("required_for_g1") is True:
            required_signoffs.append(item)
        for key in ["expected_files", "present_files"]:
            if not isinstance(item.get(key), list):
                errors.append(f"{field}.{key} must be a list")
    missing_signoffs = REQUIRED_SIGNOFF_IDS - signoff_ids
    if missing_signoffs:
        errors.append(f"missing signoff requirements: {sorted(missing_signoffs)}")
    expected_human_complete = all(bool(item.get("present")) for item in required_signoffs)
    if summary.get("human_signoff_complete") != expected_human_complete:
        errors.append("summary.human_signoff_complete must match required signoffs")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "evidence_item_count": len(evidence_ids),
            "signoff_requirement_count": len(signoff_ids),
            "machine_missing_count": len(machine_missing),
            "closure_blocker_count": len(closure_blockers),
            "machine_complete": summary.get("machine_complete") is True,
            "g1_gate_ready": summary.get("g1_gate_ready") is True,
            "human_signoff_complete": summary.get("human_signoff_complete") is True,
        },
    }


def _bullet_list(values: Any, *, empty: str) -> list[str]:
    if not isinstance(values, list) or not values:
        return [f"- {empty}"]
    return [f"- {value}" for value in values]


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
