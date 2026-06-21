from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .doctor import inspect_phase0_bundle
from .g1_review import render_g1_gate_review_draft


REQUIRED_PREFLIGHT_ARTIFACTS = (
    "inventory.json",
    "links.json",
    "placement.json",
    "validate.json",
    "simulate.json",
    "benchmark.json",
)


def render_phase0_status_report(
    bundle_path: str | Path,
    *,
    report_date: str | None = None,
    plan_version: str = "v3",
) -> dict[str, Any]:
    """Render a Phase-0 sprint status report from a preflight evidence bundle."""

    bundle = Path(bundle_path)
    report_date = report_date or date.today().isoformat()
    doctor = inspect_phase0_bundle(bundle)
    g1 = render_g1_gate_review_draft(
        bundle,
        review_date=report_date,
        plan_version=plan_version,
    )
    evidence_rows = g1.get("evidence_rows", [])
    artifacts = doctor.get("artifacts") if isinstance(doctor.get("artifacts"), dict) else {}
    simulation = _simulation_summary(artifacts)
    deliverables = _deliverables(artifacts, evidence_rows, simulation)
    summary = _summary(deliverables)
    markdown = _render_markdown(
        bundle=bundle,
        report_date=report_date,
        plan_version=plan_version,
        simulation=simulation,
        deliverables=deliverables,
        summary=summary,
        doctor=doctor,
        g1=g1,
    )
    return {
        "bundle": str(bundle),
        "report_date": report_date,
        "plan_version": plan_version,
        "current_gate": "G1",
        "simulation": simulation,
        "summary": summary,
        "deliverables": deliverables,
        "g1": {
            "machine_complete": bool(g1.get("machine_complete")),
            "gate_ready": bool(g1.get("gate_ready")),
            "recommended_outcome": g1.get("recommended_outcome"),
            "machine_missing_criteria": list(g1.get("machine_missing_criteria", [])),
            "closure_blockers": list(g1.get("closure_blockers", [])),
        },
        "doctor_errors": list(doctor.get("errors", [])),
        "doctor_warnings": list(doctor.get("warnings", [])),
        "markdown": markdown,
    }


def _deliverables(
    artifacts: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    simulation: dict[str, Any],
) -> list[dict[str, Any]]:
    row = _row_lookup(evidence_rows)
    target = row.get("v0 target contract reviewed and signed off", {})
    budget = row.get("memory budget closes with headroom and predicted throughput meets bar", {})
    concurrency = row.get("concurrency-market evidence supplied or scope narrowed", {})
    apple = row.get("Apple reversal trigger evaluated from rank-1 local probe", {})
    preflight = row.get("Phase-0 preflight workflow runnable without oral context", {})
    golden = row.get("golden-plan tests T0 green", {})
    staffing = row.get("owners and staffing closed or Sponsor accepts narrowed scope", {})

    return [
        _from_row(
            "S0-1",
            "Partitioner + cost model + golden plans (T0)",
            "DIST",
            "A1-A4",
            golden,
            simulation_sensitive=False,
            simulation=simulation,
        ),
        _combined(
            "S0-2",
            "v0-target-contract.md memory/throughput contract",
            "DIST+PM",
            "B1/P1",
            [target, budget],
            simulation_sensitive=True,
            simulation=simulation,
            closure_gap="target sign-off or planner budget/throughput proof is not closed",
        ),
        _from_row(
            "S0-3",
            "Concurrency sweep and persona supply evidence",
            "DIST",
            "B2",
            concurrency,
            simulation_sensitive=False,
            simulation=simulation,
        ),
        _artifact(
            "S0-4",
            "runtime-format-and-invariants.md",
            "RT",
            "B3",
            artifacts,
            "runtime_format_spec",
            closure_gap="runtime format review sign-off is not represented in the bundle",
        ),
        _artifact(
            "S0-5",
            "networking-security-and-backpressure.md",
            "NET",
            "B4",
            artifacts,
            "network_security_spec",
            closure_gap="network/security review sign-off is not represented in the bundle",
        ),
        _artifact(
            "S0-6",
            "adr/0001-max-mojo-substrate.md",
            "TL",
            "B5",
            artifacts,
            "substrate_adr",
            closure_gap="substrate ADR review sign-off is not represented in the bundle",
        ),
        _from_row(
            "S0-7",
            "Apple expert-MLP probe and role decision",
            "KER",
            "A-2/R-4",
            apple,
            simulation_sensitive=False,
            simulation=simulation,
        ),
        _from_row(
            "S0-8",
            "Roadmap rebaseline and staffing answer",
            "PM",
            "A-5/I-5",
            staffing,
            simulation_sensitive=False,
            simulation=simulation,
        ),
        _preflight_deliverable(preflight, artifacts, simulation),
    ]


def _row_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in rows:
        if isinstance(item, dict):
            criterion = item.get("criterion")
            if isinstance(criterion, str):
                lookup[criterion] = item
    return lookup


def _from_row(
    deliverable_id: str,
    name: str,
    owner: str,
    closes: str,
    row: dict[str, Any],
    *,
    simulation_sensitive: bool,
    simulation: dict[str, Any],
) -> dict[str, Any]:
    machine_ok = bool(row.get("machine_ok"))
    closure_ok = bool(row.get("closure_ok"))
    status = _status(machine_ok, closure_ok, simulation_sensitive, simulation)
    gap = None if closure_ok else str(row.get("closure_gap", "not closed"))
    if status == "simulation_complete" and gap is None:
        gap = "simulation evidence only; not real multi-host hardware evidence"
    return {
        "id": deliverable_id,
        "deliverable": name,
        "owner": owner,
        "closes": closes,
        "status": status,
        "machine_ok": machine_ok,
        "closure_ok": closure_ok,
        "simulation_sensitive": simulation_sensitive,
        "evidence": str(row.get("evidence", "missing evidence")),
        "closure_gap": gap,
    }


def _combined(
    deliverable_id: str,
    name: str,
    owner: str,
    closes: str,
    rows: list[dict[str, Any]],
    *,
    simulation_sensitive: bool,
    simulation: dict[str, Any],
    closure_gap: str,
) -> dict[str, Any]:
    machine_ok = bool(rows) and all(bool(row.get("machine_ok")) for row in rows)
    closure_ok = bool(rows) and all(bool(row.get("closure_ok")) for row in rows)
    status = _status(machine_ok, closure_ok, simulation_sensitive, simulation)
    evidence = "; ".join(str(row.get("evidence", "missing evidence")) for row in rows)
    gaps = [str(row.get("closure_gap", "not closed")) for row in rows if not row.get("closure_ok")]
    gap = None if closure_ok else ("; ".join(gaps) or closure_gap)
    if status == "simulation_complete" and gap is None:
        gap = "simulation evidence only; not real multi-host hardware evidence"
    return {
        "id": deliverable_id,
        "deliverable": name,
        "owner": owner,
        "closes": closes,
        "status": status,
        "machine_ok": machine_ok,
        "closure_ok": closure_ok,
        "simulation_sensitive": simulation_sensitive,
        "evidence": evidence or "missing evidence",
        "closure_gap": gap,
    }


def _artifact(
    deliverable_id: str,
    name: str,
    owner: str,
    closes: str,
    artifacts: dict[str, Any],
    key: str,
    *,
    closure_gap: str,
) -> dict[str, Any]:
    artifact = artifacts.get(key)
    machine_ok = isinstance(artifact, dict) and bool(artifact.get("present"))
    files = artifact.get("files") if isinstance(artifact, dict) else None
    evidence = "present: " + ", ".join(str(item) for item in files) if files else "missing artifact"
    return {
        "id": deliverable_id,
        "deliverable": name,
        "owner": owner,
        "closes": closes,
        "status": "machine_complete" if machine_ok else "incomplete",
        "machine_ok": machine_ok,
        "closure_ok": False,
        "simulation_sensitive": False,
        "evidence": evidence,
        "closure_gap": closure_gap if machine_ok else "artifact is missing",
    }


def _preflight_deliverable(
    row: dict[str, Any], artifacts: dict[str, Any], simulation: dict[str, Any]
) -> dict[str, Any]:
    required_present = all(
        isinstance(artifacts.get(name), dict) and bool(artifacts[name].get("present"))
        for name in REQUIRED_PREFLIGHT_ARTIFACTS
    )
    machine_ok = bool(row.get("machine_ok")) and required_present
    closure_ok = bool(row.get("closure_ok")) and required_present
    deliverable = _from_row(
        "S0-9",
        "Phase-0 preflight workflow runnable without oral context",
        "DIST+SRE",
        "I-6",
        row,
        simulation_sensitive=True,
        simulation=simulation,
    )
    deliverable["machine_ok"] = machine_ok
    deliverable["closure_ok"] = closure_ok
    deliverable["status"] = _status(machine_ok, closure_ok, True, simulation)
    deliverable["evidence"] = (
        deliverable["evidence"]
        + f"; required_artifacts_present={required_present}"
    )
    if not required_present:
        deliverable["closure_gap"] = "required preflight artifacts are missing"
    return deliverable


def _status(
    machine_ok: bool,
    closure_ok: bool,
    simulation_sensitive: bool,
    simulation: dict[str, Any],
) -> str:
    if machine_ok and simulation_sensitive and simulation.get("present"):
        return "simulation_complete"
    if closure_ok:
        return "closed"
    if machine_ok:
        return "machine_complete"
    return "incomplete"


def _simulation_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    inventory = artifacts.get("inventory.json")
    if not isinstance(inventory, dict) or not inventory.get("simulation_mode"):
        return {"present": False}
    return {
        "present": True,
        "mode": inventory.get("simulation_mode"),
        "profile": inventory.get("simulation_profile"),
        "warning": "simulation evidence is not real multi-host hardware evidence",
    }


def _summary(deliverables: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"closed": 0, "machine_complete": 0, "simulation_complete": 0, "incomplete": 0}
    for item in deliverables:
        status = str(item.get("status"))
        counts[status] = counts.get(status, 0) + 1
    counts["total"] = len(deliverables)
    counts["machine_or_better"] = sum(
        1
        for item in deliverables
        if item.get("status") in {"closed", "machine_complete", "simulation_complete"}
    )
    return counts


def _render_markdown(
    *,
    bundle: Path,
    report_date: str,
    plan_version: str,
    simulation: dict[str, Any],
    deliverables: list[dict[str, Any]],
    summary: dict[str, Any],
    doctor: dict[str, Any],
    g1: dict[str, Any],
) -> str:
    missing = g1.get("machine_missing_criteria", [])
    headline = (
        "At risk: Phase-0 machine evidence is incomplete"
        if missing
        else "Machine evidence complete; human gate closure still required"
    )
    simulation_note = "none"
    if simulation.get("present"):
        simulation_note = (
            f"{simulation.get('mode')} / {simulation.get('profile')} - "
            "simulation evidence only, not physical hardware closure"
        )
    lines = [
        f"# Weekly Status Report - Fornax - Week of {report_date}",
        "",
        f"**Reporter:** Fornax CLI - **Plan version:** {plan_version} - **Current gate:** G1",
        "",
        "## Headline",
        headline,
        "",
        "## Gate & Milestone Posture",
        f"- Current gate: G1 - recommended disposition: {g1.get('recommended_outcome')}",
        (
            "- Phase-0 S0 deliverables: "
            f"{summary.get('machine_or_better', 0)}/{summary.get('total', 0)} "
            "machine/simulation complete or closed"
        ),
        f"- Simulation evidence: {simulation_note}",
        "",
        "## Deliverable Status",
        "| ID | Deliverable | Status | Evidence | Gap |",
        "|---|---|---|---|---|",
    ]
    for item in deliverables:
        lines.append(
            "| {id} | {deliverable} | {status} | {evidence} | {gap} |".format(
                id=_escape(item["id"]),
                deliverable=_escape(item["deliverable"]),
                status=_escape(item["status"]),
                evidence=_escape(item["evidence"]),
                gap=_escape(item.get("closure_gap") or "none"),
            )
        )
    lines.extend(
        [
            "",
            "## Top Risks",
            "| ID | Risk | Trend | Action this week |",
            "|---|---|---|---|",
            "| R-10 | Status drift: planned artifacts look proven | -> | Use this report's status column; do not count simulation_complete as physical closure. |",
            "| R-4 | Apple/MAX capability unknown | -> | Attach rank-1 Apple probe validation and role decision, or record Sponsor narrowing. |",
            "| I-5 | Staffing/owner closure | -> | Attach named owner/staffing sign-off or Sponsor narrowed-scope decision. |",
            "",
            "## Decisions Needed",
            "- DEC-005 G1 disposition remains pending Sponsor decision.",
            "- Apple v0 role remains pending until S0-7 is complete.",
            "- Staffing/sign-off remains pending until S0-8 is closed.",
            "",
            "## Doctor Summary",
            f"- ok: {bool(doctor.get('ok'))}",
            f"- errors: {len(doctor.get('errors', [])) if isinstance(doctor.get('errors'), list) else 'unknown'}",
            f"- warnings: {len(doctor.get('warnings', [])) if isinstance(doctor.get('warnings'), list) else 'unknown'}",
            "",
            "## Next Week",
            "- Complete S0-7 Apple rank-1 probe/role-decision evidence or prepare Sponsor narrowing input.",
            "- Attach target/spec/staffing review sign-offs if the Sponsor is preparing for G1.",
            "",
        ]
    )
    return "\n".join(lines)


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
