from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "program-governance-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "program-governance-x1-x3-controls"

REQUIRED_WORKSTREAMS = {"X1", "X2", "X3"}
REQUIRED_DECISIONS = {"DEC-001", "DEC-002", "DEC-003", "DEC-004", "DEC-005", "DEC-006"}
REQUIRED_RISK_IDS = {"R-4", "R-8", "R-10"}
REQUIRED_ASSUMPTION_IDS = {"A-1", "A-2", "A-5"}
REQUIRED_ISSUE_IDS = {"I-1", "I-2", "I-3", "I-4", "I-5", "I-6"}
REQUIRED_DEPENDENCY_IDS = {"D-1", "D-2", "D-3", "D-4"}
REQUIRED_CADENCE_ARTIFACTS = {
    "weekly-status",
    "gate-review",
    "decision-log",
    "raid-log",
    "external-watch",
    "phase0-status",
}
VALID_GATE_OUTCOMES = {"PROCEED", "ITERATE", "NARROW", "KILL"}
VALID_EXTERNAL_STATUSES = {"probing", "partial", "sufficient", "regressed"}


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _string_list(value: Any, field: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return []
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
        else:
            result.append(item)
    return result


def _decision(
    decision_id: str,
    status: str,
    decision: str,
    authority: str,
    reverses: str,
    ref: str,
) -> dict[str, Any]:
    return {
        "id": decision_id,
        "status": status,
        "decision": decision,
        "authority": authority,
        "reverses": reverses,
        "ref": ref,
    }


def _raid_item(
    item_id: str,
    *,
    kind: str,
    owner: str,
    status: str,
    summary: str,
    mitigation: str,
    rank: int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "kind": kind,
        "owner": owner,
        "status": status,
        "summary": summary,
        "mitigation": mitigation,
    }
    if rank is not None:
        item["rank"] = rank
    return item


def _control(
    control_id: str,
    *,
    workstream: str,
    status: str,
    evidence: list[str],
    validator: str,
    failure_mode: str,
) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "workstream": workstream,
        "status": status,
        "evidence": evidence,
        "validator": validator,
        "failure_mode": failure_mode,
    }


def simulate_program_governance(
    *,
    plan_id: str = "program-governance-plan",
    report_date: str = "2026-06-22",
    plan_version: str = "v3",
    current_gate: str = "G1",
) -> dict[str, Any]:
    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(report_date, "report_date", errors)
    _non_empty_string(plan_version, "plan_version", errors)
    _non_empty_string(current_gate, "current_gate", errors)
    if errors:
        raise ValueError("; ".join(errors))

    decision_log = {
        "path": "docs/fornax/program_management/08-decision-log.md",
        "decision_authority": "Sponsor",
        "allowed_gate_outcomes": sorted(VALID_GATE_OUTCOMES),
        "silent_proceed_forbidden": True,
        "entries": [
            _decision(
                "DEC-001",
                "Accepted",
                "Fornax is an engine, not a llama.cpp harness",
                "Sponsor",
                "one-way-ish",
                "plan v3 section 5.4",
            ),
            _decision(
                "DEC-002",
                "Accepted",
                "Pipeline-parallel spine with bounded remote experts",
                "Sponsor",
                "reversible per deployment",
                "plan v3 section 5.1",
            ),
            _decision(
                "DEC-003",
                "Accepted",
                "Apple participation is staged and gated",
                "Sponsor",
                "reversible",
                "plan v3 section 5.5",
            ),
            _decision(
                "DEC-004",
                "Accepted",
                "Plan changes only by version bump",
                "Sponsor",
                "no",
                "program governance",
            ),
            _decision(
                "DEC-005",
                "Pending",
                "G1 go/no-go outcome plus re-baselined schedule",
                "Sponsor",
                "no",
                "stage gate G1",
            ),
            _decision(
                "DEC-006",
                "Accepted",
                "Speculative decoding is out of v0 unless the target contract opts in",
                "Sponsor",
                "contract opt-in",
                "plan v3 section 3.5",
            ),
        ],
    }

    raid_register = {
        "path": "docs/fornax/program_management/05-raid-log.md",
        "ids_stable": True,
        "review_cadence": "weekly and every gate",
        "risks": [
            _raid_item(
                "R-4",
                kind="risk",
                owner="KER",
                status="Open",
                rank=1,
                summary="Apple/MAX readiness misses target role",
                mitigation="source precedence ladder; pinned local probe; staged role demotion",
            ),
            _raid_item(
                "R-8",
                kind="risk",
                owner="PM",
                status="Open - resolved at G1",
                rank=2,
                summary="Concurrency-market fit",
                mitigation="persona concurrency evidence or NARROW at G1",
            ),
            _raid_item(
                "R-10",
                kind="risk",
                owner="PM",
                status="Open",
                rank=10,
                summary="Status drift: planned artifacts look proven",
                mitigation="owner checklist per artifact and explicit evidence tier",
            ),
        ],
        "assumptions": [
            _raid_item(
                "A-1",
                kind="assumption",
                owner="PM",
                status="Unvalidated -> G1",
                summary="Persona supplies pipeline-filling concurrency",
                mitigation="contract concurrency sweep or narrowed scope",
            ),
            _raid_item(
                "A-2",
                kind="assumption",
                owner="KER",
                status="Unvalidated -> G1",
                summary="MAX can run target expert-MLP on the target Mac acceptably",
                mitigation="rank-1 local probe decides Apple role",
            ),
            _raid_item(
                "A-5",
                kind="assumption",
                owner="PM",
                status="Unvalidated",
                summary="Required skills are staffable",
                mitigation="named owner or Sponsor accepts narrowed scope",
            ),
        ],
        "issues": [
            _raid_item("I-1", kind="issue", owner="DIST/PM", status="Open", summary="v0-target-contract.md not signed off", mitigation="target contract review before G1"),
            _raid_item("I-2", kind="issue", owner="RT", status="Open", summary="runtime-format spec review remains required", mitigation="reviewed spec before G1"),
            _raid_item("I-3", kind="issue", owner="NET", status="Open", summary="networking/security spec review remains required", mitigation="reviewed spec before G1"),
            _raid_item("I-4", kind="issue", owner="TL", status="Open", summary="substrate ADR review remains required", mitigation="reviewed ADR before G1"),
            _raid_item("I-5", kind="issue", owner="PM", status="Open", summary="KER/Apple staffing unresolved", mitigation="name owner or narrow scope"),
            _raid_item("I-6", kind="issue", owner="DIST/SRE", status="Open", summary="Phase-0 preflight workflow needs G1 closure review", mitigation="runnable bundle plus review"),
        ],
        "dependencies": [
            _raid_item("D-1", kind="dependency", owner="KER", status="Open", summary="Modular/MAX Apple plus MoE capability", mitigation="external watch and local probe"),
            _raid_item("D-2", kind="dependency", owner="PM", status="Open", summary="Hardware procurement for lab tiers", mitigation="order lead-time hardware early"),
            _raid_item("D-3", kind="dependency", owner="TL", status="Open", summary="Ignis Engine seam stability", mitigation="keep generate boundary stable"),
            _raid_item("D-4", kind="dependency", owner="DIST", status="Open", summary="Planner blocks downstream phases", mitigation="prioritize WS-A/G1 evidence"),
        ],
    }

    external_watch = {
        "path": "docs/fornax/program_management/06-dependencies-and-external-watch.md",
        "dependency_id": "D-1",
        "status": "probing",
        "last_checked": report_date,
        "pinned_build": "max-26.4.0",
        "local_probe_required": True,
        "reversal_trigger_armed": True,
        "capability_needed": "target-model expert-MLP on target Mac within tolerance and throughput bound",
        "source_precedence": [
            {"rank": 1, "source": "local probe on pinned build", "gate_of_record": True},
            {"rank": 2, "source": "package docs and changelog for pinned build", "gate_of_record": False},
            {"rank": 3, "source": "supported-model catalog and model docs", "gate_of_record": False},
            {"rank": 4, "source": "blog posts and launch announcements", "gate_of_record": False},
            {"rank": 5, "source": "nightly behavior", "gate_of_record": False},
        ],
        "policy": "future promises never unblock a gate; capability is unproven until local probe passes",
    }

    cadence = {
        "report_date": report_date,
        "current_gate": current_gate,
        "artifacts": [
            {
                "artifact_id": "weekly-status",
                "path": "docs/fornax/program_management/templates/status-report.md",
                "cadence": "weekly",
                "owner": "PM",
                "required_sections": [
                    "headline",
                    "gate posture",
                    "top risks",
                    "decisions needed",
                    "workstream progress",
                    "external watch",
                    "next week",
                ],
            },
            {
                "artifact_id": "gate-review",
                "path": "docs/fornax/program_management/templates/gate-review.md",
                "cadence": "every gate",
                "owner": "Sponsor/PM/TL",
                "required_sections": [
                    "entry criteria",
                    "evidence presented",
                    "risk and assumption check",
                    "decision",
                    "actions",
                ],
            },
            {
                "artifact_id": "decision-log",
                "path": "docs/fornax/program_management/08-decision-log.md",
                "cadence": "on irreversible or expensive decision",
                "owner": "PM",
                "required_sections": ["decision log", "ADR index", "rules"],
            },
            {
                "artifact_id": "raid-log",
                "path": "docs/fornax/program_management/05-raid-log.md",
                "cadence": "weekly and every gate",
                "owner": "PM",
                "required_sections": ["risks", "assumptions", "issues", "dependencies"],
            },
            {
                "artifact_id": "external-watch",
                "path": "docs/fornax/program_management/06-dependencies-and-external-watch.md",
                "cadence": "each MAX nightly or release",
                "owner": "KER",
                "required_sections": ["source precedence", "watch register", "dependency-driven sequencing"],
            },
            {
                "artifact_id": "phase0-status",
                "path": "phase0-status.json",
                "cadence": "per Phase-0 validation bundle",
                "owner": "PM/SRE",
                "required_sections": ["deliverables", "g1", "simulation", "markdown"],
            },
        ],
    }

    controls = [
        _control(
            "X1-gate-operation",
            workstream="X1",
            status="active",
            evidence=["gate-review template", "DEC-005 pending", "allowed outcomes"],
            validator="validate_program_governance_fixture",
            failure_mode="gate outcome is implied without Sponsor decision",
        ),
        _control(
            "X1-decision-log",
            workstream="X1",
            status="active",
            evidence=["DEC-001..DEC-006 present", "DEC-005 pending"],
            validator="validate_program_governance_fixture",
            failure_mode="expensive decision lacks DEC entry",
        ),
        _control(
            "X2-raid-upkeep",
            workstream="X2",
            status="active",
            evidence=["R/A/I/D IDs stable", "R-10 status drift tracked"],
            validator="validate_program_governance_fixture",
            failure_mode="risk or issue is renumbered or silently closed",
        ),
        _control(
            "X2-external-watch",
            workstream="X2",
            status="active",
            evidence=["D-1 source precedence", "rank-1 local probe gate"],
            validator="validate_program_governance_fixture",
            failure_mode="blog/nightly evidence is treated as a gate pass",
        ),
        _control(
            "X3-cadence-reporting",
            workstream="X3",
            status="active",
            evidence=["weekly status template", "phase0-status artifact"],
            validator="validate_program_governance_fixture",
            failure_mode="status cannot be produced without oral context",
        ),
        _control(
            "R10-status-drift",
            workstream="X3",
            status="active",
            evidence=["simulation-only warnings", "G1 not gate-ready"],
            validator="validate_program_governance_fixture",
            failure_mode="planned or simulated work is reported as proven closure",
        ),
    ]

    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "plan_version": plan_version,
        "current_gate": current_gate,
        "simulation_method": SIMULATION_METHOD,
        "governance_scope": {
            "workstreams": sorted(REQUIRED_WORKSTREAMS),
            "gate_of_record": current_gate,
            "decision_authority": "Sponsor",
            "advised_by": ["PM", "TL"],
            "phase1_authorized": False,
        },
        "decision_log": decision_log,
        "raid_register": raid_register,
        "external_watch": external_watch,
        "cadence": cadence,
        "controls": controls,
        "summary": {
            "decision_count": len(decision_log["entries"]),
            "control_count": len(controls),
            "cadence_artifact_count": len(cadence["artifacts"]),
            "risk_count": len(raid_register["risks"]),
            "issue_count": len(raid_register["issues"]),
            "dependency_count": len(raid_register["dependencies"]),
            "dec005_pending": True,
            "g1_gate_ready": False,
            "silent_proceed_forbidden": True,
            "status_drift_controlled": True,
            "external_watch_rank1_required": True,
            "simulation_only": True,
        },
        "note": (
            "T1 program governance simulation: validates WS-X gate, decision-log, "
            "RAID, external-watch, and cadence controls. Not G1/G5 closure evidence."
        ),
    }


def _ids(rows: list[Any], field: str, errors: list[str]) -> set[str]:
    ids: set[str] = set()
    for index, row in enumerate(rows):
        item_field = f"{field}[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{item_field} must be an object")
            continue
        item_id = _non_empty_string(row.get("id"), f"{item_field}.id", errors)
        if item_id is not None:
            if item_id in ids:
                errors.append(f"{item_field}.id duplicates {item_id}")
            ids.add(item_id)
    return ids


def _validate_decision_log(data: dict[str, Any], errors: list[str]) -> set[str]:
    decision_log = data.get("decision_log")
    if not isinstance(decision_log, dict):
        errors.append("decision_log must be an object")
        return set()
    _non_empty_string(decision_log.get("path"), "decision_log.path", errors)
    if decision_log.get("decision_authority") != "Sponsor":
        errors.append("decision_log.decision_authority must be Sponsor")
    if decision_log.get("silent_proceed_forbidden") is not True:
        errors.append("decision_log.silent_proceed_forbidden must be true")
    outcomes = set(_string_list(decision_log.get("allowed_gate_outcomes"), "decision_log.allowed_gate_outcomes", errors))
    if outcomes != VALID_GATE_OUTCOMES:
        errors.append(f"decision_log.allowed_gate_outcomes must be {sorted(VALID_GATE_OUTCOMES)}")
    entries = decision_log.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("decision_log.entries must be a non-empty list")
        return set()
    decision_ids = _ids(entries, "decision_log.entries", errors)
    missing = REQUIRED_DECISIONS - decision_ids
    if missing:
        errors.append(f"decision_log.entries missing required DEC IDs: {sorted(missing)}")
    by_id = {entry.get("id"): entry for entry in entries if isinstance(entry, dict)}
    dec005 = by_id.get("DEC-005")
    if not isinstance(dec005, dict):
        errors.append("DEC-005 must be present")
    else:
        if dec005.get("status") != "Pending":
            errors.append("DEC-005.status must remain Pending in simulated governance evidence")
        if dec005.get("authority") != "Sponsor":
            errors.append("DEC-005.authority must be Sponsor")
        if "G1" not in str(dec005.get("decision", "")):
            errors.append("DEC-005.decision must describe G1 outcome")
    return decision_ids


def _validate_raid_section(
    raid: dict[str, Any],
    section: str,
    required_ids: set[str],
    errors: list[str],
) -> set[str]:
    rows = raid.get(section)
    if not isinstance(rows, list) or not rows:
        errors.append(f"raid_register.{section} must be a non-empty list")
        return set()
    row_ids = _ids(rows, f"raid_register.{section}", errors)
    missing = required_ids - row_ids
    if missing:
        errors.append(f"raid_register.{section} missing required IDs: {sorted(missing)}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        field = f"raid_register.{section}[{index}]"
        _non_empty_string(row.get("kind"), f"{field}.kind", errors)
        _non_empty_string(row.get("owner"), f"{field}.owner", errors)
        _non_empty_string(row.get("status"), f"{field}.status", errors)
        _non_empty_string(row.get("summary"), f"{field}.summary", errors)
        _non_empty_string(row.get("mitigation"), f"{field}.mitigation", errors)
    return row_ids


def _validate_raid(data: dict[str, Any], errors: list[str]) -> dict[str, set[str]]:
    raid = data.get("raid_register")
    if not isinstance(raid, dict):
        errors.append("raid_register must be an object")
        return {"risks": set(), "assumptions": set(), "issues": set(), "dependencies": set()}
    _non_empty_string(raid.get("path"), "raid_register.path", errors)
    if raid.get("ids_stable") is not True:
        errors.append("raid_register.ids_stable must be true")
    if "weekly" not in str(raid.get("review_cadence", "")):
        errors.append("raid_register.review_cadence must mention weekly review")
    risk_ids = _validate_raid_section(raid, "risks", REQUIRED_RISK_IDS, errors)
    assumption_ids = _validate_raid_section(raid, "assumptions", REQUIRED_ASSUMPTION_IDS, errors)
    issue_ids = _validate_raid_section(raid, "issues", REQUIRED_ISSUE_IDS, errors)
    dependency_ids = _validate_raid_section(raid, "dependencies", REQUIRED_DEPENDENCY_IDS, errors)
    risks = raid.get("risks") if isinstance(raid.get("risks"), list) else []
    risk_by_id = {risk.get("id"): risk for risk in risks if isinstance(risk, dict)}
    if isinstance(risk_by_id.get("R-4"), dict) and risk_by_id["R-4"].get("rank") != 1:
        errors.append("R-4.rank must be 1")
    if isinstance(risk_by_id.get("R-8"), dict) and risk_by_id["R-8"].get("rank") != 2:
        errors.append("R-8.rank must be 2")
    if isinstance(risk_by_id.get("R-10"), dict) and "status drift" not in risk_by_id["R-10"].get("summary", "").lower():
        errors.append("R-10.summary must mention status drift")
    return {
        "risks": risk_ids,
        "assumptions": assumption_ids,
        "issues": issue_ids,
        "dependencies": dependency_ids,
    }


def _validate_external_watch(data: dict[str, Any], errors: list[str]) -> None:
    watch = data.get("external_watch")
    if not isinstance(watch, dict):
        errors.append("external_watch must be an object")
        return
    if watch.get("dependency_id") != "D-1":
        errors.append("external_watch.dependency_id must be D-1")
    if watch.get("status") not in VALID_EXTERNAL_STATUSES:
        errors.append(f"external_watch.status must be one of {sorted(VALID_EXTERNAL_STATUSES)}")
    _non_empty_string(watch.get("path"), "external_watch.path", errors)
    _non_empty_string(watch.get("last_checked"), "external_watch.last_checked", errors)
    _non_empty_string(watch.get("pinned_build"), "external_watch.pinned_build", errors)
    if watch.get("local_probe_required") is not True:
        errors.append("external_watch.local_probe_required must be true")
    if watch.get("reversal_trigger_armed") is not True:
        errors.append("external_watch.reversal_trigger_armed must be true")
    precedence = watch.get("source_precedence")
    if not isinstance(precedence, list) or len(precedence) != 5:
        errors.append("external_watch.source_precedence must contain exactly five ranks")
        return
    ranks = [row.get("rank") for row in precedence if isinstance(row, dict)]
    if ranks != [1, 2, 3, 4, 5]:
        errors.append("external_watch.source_precedence ranks must be [1, 2, 3, 4, 5]")
    for index, row in enumerate(precedence):
        field = f"external_watch.source_precedence[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(row.get("source"), f"{field}.source", errors)
        if not isinstance(row.get("gate_of_record"), bool):
            errors.append(f"{field}.gate_of_record must be boolean")
    rank1 = precedence[0] if isinstance(precedence[0], dict) else {}
    if rank1.get("rank") != 1 or rank1.get("gate_of_record") is not True:
        errors.append("external_watch rank 1 must be the gate of record")
    for row in precedence[1:]:
        if isinstance(row, dict) and row.get("gate_of_record") is True:
            errors.append("only external_watch rank 1 may be gate_of_record")
    if "future promises never unblock" not in str(watch.get("policy", "")):
        errors.append("external_watch.policy must reject future promises as gate evidence")


def _validate_cadence(data: dict[str, Any], errors: list[str]) -> set[str]:
    cadence = data.get("cadence")
    if not isinstance(cadence, dict):
        errors.append("cadence must be an object")
        return set()
    _non_empty_string(cadence.get("report_date"), "cadence.report_date", errors)
    _non_empty_string(cadence.get("current_gate"), "cadence.current_gate", errors)
    artifacts = cadence.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("cadence.artifacts must be a non-empty list")
        return set()
    artifact_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        field = f"cadence.artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{field} must be an object")
            continue
        artifact_id = _non_empty_string(artifact.get("artifact_id"), f"{field}.artifact_id", errors)
        if artifact_id is not None:
            artifact_ids.add(artifact_id)
        _non_empty_string(artifact.get("path"), f"{field}.path", errors)
        _non_empty_string(artifact.get("cadence"), f"{field}.cadence", errors)
        _non_empty_string(artifact.get("owner"), f"{field}.owner", errors)
        sections = _string_list(artifact.get("required_sections"), f"{field}.required_sections", errors)
        if len(sections) < 3:
            errors.append(f"{field}.required_sections must contain at least three sections")
    missing = REQUIRED_CADENCE_ARTIFACTS - artifact_ids
    if missing:
        errors.append(f"cadence.artifacts missing required IDs: {sorted(missing)}")
    return artifact_ids


def _validate_controls(data: dict[str, Any], errors: list[str]) -> set[str]:
    controls = data.get("controls")
    if not isinstance(controls, list) or not controls:
        errors.append("controls must be a non-empty list")
        return set()
    workstreams: set[str] = set()
    control_ids: set[str] = set()
    for index, control in enumerate(controls):
        field = f"controls[{index}]"
        if not isinstance(control, dict):
            errors.append(f"{field} must be an object")
            continue
        control_id = _non_empty_string(control.get("control_id"), f"{field}.control_id", errors)
        if control_id is not None:
            if control_id in control_ids:
                errors.append(f"{field}.control_id duplicates {control_id}")
            control_ids.add(control_id)
        workstream = _non_empty_string(control.get("workstream"), f"{field}.workstream", errors)
        if workstream is not None:
            workstreams.add(workstream)
        if control.get("status") != "active":
            errors.append(f"{field}.status must be active")
        _string_list(control.get("evidence"), f"{field}.evidence", errors)
        _non_empty_string(control.get("validator"), f"{field}.validator", errors)
        _non_empty_string(control.get("failure_mode"), f"{field}.failure_mode", errors)
    missing = REQUIRED_WORKSTREAMS - workstreams
    if missing:
        errors.append(f"controls missing required workstreams: {sorted(missing)}")
    if not any(control_id == "R10-status-drift" for control_id in control_ids):
        errors.append("controls must include R10-status-drift")
    return workstreams


def validate_program_governance_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if data.get("simulation_method") != SIMULATION_METHOD:
        errors.append(f"simulation_method must be {SIMULATION_METHOD}")
    _non_empty_string(data.get("plan_id"), "plan_id", errors)
    _non_empty_string(data.get("plan_version"), "plan_version", errors)
    _non_empty_string(data.get("current_gate"), "current_gate", errors)

    scope = data.get("governance_scope")
    if not isinstance(scope, dict):
        errors.append("governance_scope must be an object")
        scope = {}
    if set(scope.get("workstreams", [])) != REQUIRED_WORKSTREAMS:
        errors.append(f"governance_scope.workstreams must be {sorted(REQUIRED_WORKSTREAMS)}")
    if scope.get("decision_authority") != "Sponsor":
        errors.append("governance_scope.decision_authority must be Sponsor")
    if scope.get("phase1_authorized") is not False:
        errors.append("governance_scope.phase1_authorized must be false for simulated governance evidence")

    decision_ids = _validate_decision_log(data, errors)
    raid_ids = _validate_raid(data, errors)
    _validate_external_watch(data, errors)
    cadence_ids = _validate_cadence(data, errors)
    workstreams = _validate_controls(data, errors)

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    decisions = data.get("decision_log", {}).get("entries", []) if isinstance(data.get("decision_log"), dict) else []
    controls = data.get("controls") if isinstance(data.get("controls"), list) else []
    cadence_artifacts = data.get("cadence", {}).get("artifacts", []) if isinstance(data.get("cadence"), dict) else []
    raid = data.get("raid_register") if isinstance(data.get("raid_register"), dict) else {}
    risks = raid.get("risks") if isinstance(raid.get("risks"), list) else []
    issues = raid.get("issues") if isinstance(raid.get("issues"), list) else []
    dependencies = raid.get("dependencies") if isinstance(raid.get("dependencies"), list) else []
    if summary.get("decision_count") != len(decisions):
        errors.append("summary.decision_count must equal decision log entry count")
    if summary.get("control_count") != len(controls):
        errors.append("summary.control_count must equal controls count")
    if summary.get("cadence_artifact_count") != len(cadence_artifacts):
        errors.append("summary.cadence_artifact_count must equal cadence artifact count")
    if summary.get("risk_count") != len(risks):
        errors.append("summary.risk_count must equal risk count")
    if summary.get("issue_count") != len(issues):
        errors.append("summary.issue_count must equal issue count")
    if summary.get("dependency_count") != len(dependencies):
        errors.append("summary.dependency_count must equal dependency count")
    if summary.get("dec005_pending") is not ("DEC-005" in decision_ids):
        errors.append("summary.dec005_pending must reflect DEC-005 presence")
    if summary.get("dec005_pending") is not True:
        errors.append("summary.dec005_pending must be true")
    if summary.get("g1_gate_ready") is not False:
        errors.append("summary.g1_gate_ready must be false for this simulated governance artifact")
    if summary.get("silent_proceed_forbidden") is not True:
        errors.append("summary.silent_proceed_forbidden must be true")
    if summary.get("status_drift_controlled") is not ("R-10" in raid_ids["risks"]):
        errors.append("summary.status_drift_controlled must reflect R-10 coverage")
    if summary.get("external_watch_rank1_required") is not True:
        errors.append("summary.external_watch_rank1_required must be true")
    if summary.get("simulation_only") is not True:
        errors.append("summary.simulation_only must be true")
    if not REQUIRED_CADENCE_ARTIFACTS <= cadence_ids:
        errors.append("summary cannot pass without all cadence artifacts")
    if not REQUIRED_WORKSTREAMS <= workstreams:
        errors.append("summary cannot pass without X1-X3 controls")
    warnings.append("program governance is simulation evidence, not G1/G5 closure evidence")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "decision_count": summary.get("decision_count"),
            "control_count": summary.get("control_count"),
            "cadence_artifact_count": summary.get("cadence_artifact_count"),
            "risk_count": summary.get("risk_count"),
            "issue_count": summary.get("issue_count"),
            "dependency_count": summary.get("dependency_count"),
            "dec005_pending": summary.get("dec005_pending") is True,
            "g1_gate_ready": summary.get("g1_gate_ready") is True,
            "silent_proceed_forbidden": summary.get("silent_proceed_forbidden") is True,
            "status_drift_controlled": summary.get("status_drift_controlled") is True,
            "external_watch_rank1_required": summary.get("external_watch_rank1_required") is True,
            "simulation_only": summary.get("simulation_only") is True,
        },
    }


def validate_program_governance(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid program governance artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["program governance artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_program_governance_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
