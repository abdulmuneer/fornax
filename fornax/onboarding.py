from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "onboarding-methodology-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "operator-onboarding-and-benchmark-methodology"

REQUIRED_TRACKS = {
    "operator",
    "developer",
    "benchmark-owner",
    "reviewer",
}
REQUIRED_DOCUMENTS = {
    "onboarding/quickstart.md",
    "onboarding/operator-runbook.md",
    "onboarding/developer-workflow.md",
    "onboarding/benchmark-methodology.md",
    "glossary.md",
}
REQUIRED_GLOSSARY_TERMS = {
    "plan_id",
    "logical_host",
    "placement_plan",
    "t1_simulation",
    "benchmark_of_record",
    "lab_reference",
    "drain",
    "rollback",
    "node_replacement",
    "gate",
}
REQUIRED_BENCHMARK_INPUTS = {
    "commands",
    "prompts_or_traces",
    "versions",
    "correctness_artifacts",
    "environment",
    "raw_logs",
    "ledger_record",
}


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _require_string_list(value: Any, field: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return []
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
        else:
            strings.append(item)
    return strings


def _document(
    *,
    path: str,
    title: str,
    audience: str,
    sections: list[str],
    markdown: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "title": title,
        "audience": audience,
        "sections": sections,
        "markdown": markdown,
    }


def _track(
    *,
    track_id: str,
    audience: str,
    documents: list[str],
    prerequisites: list[str],
    first_run_commands: list[str],
    success_evidence: list[str],
    escalation: list[str],
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "audience": audience,
        "documents": documents,
        "prerequisites": prerequisites,
        "first_run_commands": first_run_commands,
        "success_evidence": success_evidence,
        "escalation": escalation,
    }


def _glossary_term(
    term_id: str,
    term: str,
    definition: str,
    referenced_by: list[str],
) -> dict[str, Any]:
    return {
        "term_id": term_id,
        "term": term,
        "definition": definition,
        "referenced_by": referenced_by,
    }


def simulate_onboarding_methodology(
    *,
    plan_id: str = "onboarding-methodology-plan",
    package_id: str = "fornax-operator-onboarding",
    benchmark_id: str = "fornax-benchmark-of-record-methodology",
) -> dict[str, Any]:
    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(package_id, "package_id", errors)
    _non_empty_string(benchmark_id, "benchmark_id", errors)
    if errors:
        raise ValueError("; ".join(errors))

    documents = {
        "onboarding/quickstart.md": _document(
            path="onboarding/quickstart.md",
            title="Fornax Operator Quickstart",
            audience="firm operator",
            sections=[
                "Prerequisites",
                "Create cluster.yaml",
                "Run simulated validation",
                "Interpret pass/fail evidence",
            ],
            markdown=(
                "# Fornax Operator Quickstart\n\n"
                "Use the two-logical-host simulation before real cluster work. "
                "Run `python3 -m fornax program simulate-t1`, then inspect the "
                "validation bundle and warnings before moving to lab hardware."
            ),
        ),
        "onboarding/operator-runbook.md": _document(
            path="onboarding/operator-runbook.md",
            title="Fornax Operator Runbook",
            audience="SRE / operator",
            sections=[
                "Deploy",
                "Drain",
                "Upgrade",
                "Restart",
                "Rollback",
                "Node replacement",
                "Escalation",
            ],
            markdown=(
                "# Fornax Operator Runbook\n\n"
                "Operator actions must drain before mutation, verify health after "
                "each mutation, preserve plan-integrity tags, and keep dropped "
                "in-flight request counts at zero."
            ),
        ),
        "onboarding/developer-workflow.md": _document(
            path="onboarding/developer-workflow.md",
            title="Fornax Developer Workflow",
            audience="developer",
            sections=[
                "Local checks",
                "Golden vectors",
                "T1 simulated bundle",
                "Journal and review lenses",
            ],
            markdown=(
                "# Fornax Developer Workflow\n\n"
                "Run `make fornax-golden`, `make fornax-test`, and a T1 bundle "
                "before committing milestone work. Record review-lens findings in "
                "`fornax_development_journal.md`."
            ),
        ),
        "onboarding/benchmark-methodology.md": _document(
            path="onboarding/benchmark-methodology.md",
            title="Benchmark Methodology Of Record",
            audience="benchmark owner",
            sections=[
                "Benchmark of record",
                "Required inputs",
                "Correctness before throughput",
                "Reproducibility",
                "Gate mapping",
            ],
            markdown=(
                "# Benchmark Methodology Of Record\n\n"
                "The benchmark of record runs on `lab-reference` with commands, "
                "prompts or traces, versions, correctness artifacts, environment, "
                "raw logs, and a ledger record. Simulation evidence cannot replace "
                "T2-T4 lab evidence."
            ),
        ),
        "glossary.md": _document(
            path="glossary.md",
            title="Fornax Glossary",
            audience="all roles",
            sections=[
                "Planning terms",
                "Runtime terms",
                "Operations terms",
                "Benchmark terms",
                "Gate terms",
            ],
            markdown=(
                "# Fornax Glossary\n\n"
                "Defines plan IDs, logical hosts, placement plans, T1 simulation, "
                "benchmark of record, lab-reference, drain, rollback, node "
                "replacement, and gate terminology."
            ),
        ),
    }

    tracks = [
        _track(
            track_id="operator",
            audience="firm operator / SRE",
            documents=[
                "onboarding/quickstart.md",
                "onboarding/operator-runbook.md",
                "glossary.md",
            ],
            prerequisites=[
                "cluster.yaml",
                "model.yaml",
                "placement.json",
                "access to simulated validation bundle",
            ],
            first_run_commands=[
                "python3 -m fornax ops lifecycle-simulate --out ops-lifecycle.json",
                "python3 -m fornax test ops-lifecycle",
                "python3 -m fornax program simulate-t1 --out-dir t1-bundle",
            ],
            success_evidence=[
                "ops-lifecycle validator passes",
                "T1 simulated bundle passes",
                "simulation-only warnings are preserved",
            ],
            escalation=[
                "missing config artifact",
                "drain-before-mutation violation",
                "dropped in-flight request",
                "health check failure",
            ],
        ),
        _track(
            track_id="developer",
            audience="Fornax developer",
            documents=[
                "onboarding/developer-workflow.md",
                "onboarding/benchmark-methodology.md",
                "glossary.md",
            ],
            prerequisites=[
                "branch fornax",
                "local Python environment",
                "review lenses",
                "development journal",
            ],
            first_run_commands=[
                "make fornax-golden",
                "make fornax-test",
                "python3 -m fornax program simulate-t1 --out-dir t1-bundle",
            ],
            success_evidence=[
                "golden vectors pass",
                "unit tests pass",
                "journal records review-lens status",
            ],
            escalation=[
                "new warning lacks program-management note",
                "fixture cannot be regenerated from CLI",
                "review-lens issue blocks next milestone",
            ],
        ),
        _track(
            track_id="benchmark-owner",
            audience="benchmark owner",
            documents=[
                "onboarding/benchmark-methodology.md",
                "glossary.md",
            ],
            prerequisites=[
                "lab-reference hardware reservation",
                "prompt or trace corpus",
                "version manifest",
                "correctness reference artifacts",
            ],
            first_run_commands=[
                "python3 -m fornax benchmark --plan placement.json --mode tiny-moe-or-expert-mlp --out benchmark.json",
                "python3 -m fornax test benchmark-ledger",
            ],
            success_evidence=[
                "commands captured",
                "raw logs retained",
                "ledger record validates",
                "correctness evidence attached before throughput claim",
            ],
            escalation=[
                "lab-reference unavailable",
                "version manifest incomplete",
                "correctness parity missing",
                "benchmark number cannot be reproduced",
            ],
        ),
        _track(
            track_id="reviewer",
            audience="PM / TL / review-lens owner",
            documents=[
                "onboarding/developer-workflow.md",
                "onboarding/benchmark-methodology.md",
                "glossary.md",
            ],
            prerequisites=[
                "program-management WBS",
                "stage-gate definitions",
                "review lens assignment",
            ],
            first_run_commands=[
                "python3 -m fornax test onboarding-methodology",
                "python3 -m fornax program simulate-t1 --out-dir t1-bundle",
            ],
            success_evidence=[
                "required docs are present",
                "glossary covers gate and benchmark vocabulary",
                "benchmark methodology preserves lab-reference boundary",
            ],
            escalation=[
                "gate claim exceeds evidence tier",
                "simulation warning removed",
                "DoD cannot be mapped to a validator",
            ],
        ),
    ]

    glossary_terms = [
        _glossary_term("plan_id", "Plan ID", "Stable identifier propagated through placement, transport, serving, and evidence artifacts.", ["glossary.md", "onboarding/operator-runbook.md"]),
        _glossary_term("logical_host", "Logical host", "A simulated host used to exercise multi-host semantics on local hardware without claiming real cluster evidence.", ["glossary.md", "onboarding/quickstart.md"]),
        _glossary_term("placement_plan", "Placement plan", "The planner output that maps layers, stages, replicas, and explanations to nodes.", ["glossary.md", "onboarding/quickstart.md"]),
        _glossary_term("t1_simulation", "T1 simulation", "CI-safe simulated-worker evidence used for development before T2-T4 lab validation.", ["glossary.md", "onboarding/developer-workflow.md"]),
        _glossary_term("benchmark_of_record", "Benchmark of record", "The reproducible lab-reference benchmark used for gate-grade performance claims.", ["glossary.md", "onboarding/benchmark-methodology.md"]),
        _glossary_term("lab_reference", "lab-reference", "Controlled heterogeneous lab target for benchmark-of-record runs and T2-T4 evidence.", ["glossary.md", "onboarding/benchmark-methodology.md"]),
        _glossary_term("drain", "Drain", "Stop admitting new work to a node and let in-flight work complete before mutation.", ["glossary.md", "onboarding/operator-runbook.md"]),
        _glossary_term("rollback", "Rollback", "Return a node or deployment to the previous known-good version after a failed or rejected change.", ["glossary.md", "onboarding/operator-runbook.md"]),
        _glossary_term("node_replacement", "Node replacement", "Remove a drained node, admit a replacement, verify health, and restore traffic.", ["glossary.md", "onboarding/operator-runbook.md"]),
        _glossary_term("gate", "Gate", "Sponsor decision point with PROCEED, ITERATE, NARROW, or KILL outcomes.", ["glossary.md", "onboarding/developer-workflow.md"]),
    ]

    benchmark_methodology = {
        "benchmark_id": benchmark_id,
        "status": "methodology-stub",
        "lab_reference_required": True,
        "benchmark_harness_is_production_code": True,
        "correctness_first": True,
        "no_fabricated_numbers": True,
        "required_inputs": sorted(REQUIRED_BENCHMARK_INPUTS),
        "reproducibility_controls": [
            "commands captured exactly",
            "prompt or trace corpus checksummed",
            "version manifest includes code, model, runtime, driver, and hardware",
            "environment and thermal posture recorded for lab-reference",
            "raw logs retained with ledger record",
        ],
        "correctness_artifacts": [
            "golden vectors",
            "reference-path parity",
            "per-dtype tolerance",
            "failure-mode evidence",
        ],
        "publication_artifacts": [
            "benchmark.json",
            "ledger.jsonl",
            "raw logs",
            "version manifest",
            "correctness report",
        ],
        "gate_mapping": {
            "G1": "T0 green and calibrated cost-model numbers only",
            "G2": "T3 pipeline correctness plus MoE logit parity",
            "G3": "T4 heterogeneous serve at predicted throughput plus correctness",
            "G4": "T4 elasticity with zero dropped in-flight requests",
            "G5": "productized benchmark methodology and operator handoff",
        },
        "development_path": {
            "t0_t1_commands": [
                "make fornax-golden",
                "make fornax-test",
                "python3 -m fornax program simulate-t1 --out-dir t1-bundle",
            ],
            "simulation_warning_required": True,
            "not_lab_evidence": True,
        },
    }

    operator_handoff = {
        "checklist": [
            "install prerequisites",
            "prepare cluster.yaml, model.yaml, and placement.json",
            "run doctor and T1 simulation before lab deployment",
            "drain before upgrade, restart, rollback, or node replacement",
            "attach benchmark-of-record evidence before performance claims",
            "escalate when evidence tier does not match gate claim",
        ],
        "product_ga_complete": False,
    }

    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "package_id": package_id,
        "simulation_method": SIMULATION_METHOD,
        "tracks": tracks,
        "documents": documents,
        "glossary_terms": glossary_terms,
        "benchmark_methodology": benchmark_methodology,
        "operator_handoff": operator_handoff,
        "summary": {
            "track_count": len(tracks),
            "document_count": len(documents),
            "glossary_term_count": len(glossary_terms),
            "required_tracks_present": True,
            "required_documents_present": True,
            "required_glossary_terms_present": True,
            "benchmark_methodology_present": True,
            "correctness_first": True,
            "lab_reference_required": True,
            "reproducibility_fields_count": len(benchmark_methodology["required_inputs"]),
            "onboarding_complete_for_simulation": True,
            "product_ga_complete": False,
        },
        "note": (
            "T1 onboarding methodology simulation: validates I3 documentation, "
            "glossary, and benchmark-methodology structure. Not G5 product-GA "
            "closure evidence."
        ),
    }


def _validate_documents(data: dict[str, Any], errors: list[str]) -> set[str]:
    documents = data.get("documents")
    if not isinstance(documents, dict):
        errors.append("documents must be an object keyed by path")
        return set()
    document_paths = set(documents)
    missing = REQUIRED_DOCUMENTS - document_paths
    if missing:
        errors.append(f"documents missing required paths: {sorted(missing)}")
    for path, document in documents.items():
        field = f"documents[{path!r}]"
        if not isinstance(document, dict):
            errors.append(f"{field} must be an object")
            continue
        if document.get("path") != path:
            errors.append(f"{field}.path must match document key")
        _non_empty_string(document.get("title"), f"{field}.title", errors)
        _non_empty_string(document.get("audience"), f"{field}.audience", errors)
        sections = _require_string_list(document.get("sections"), f"{field}.sections", errors)
        markdown = _non_empty_string(document.get("markdown"), f"{field}.markdown", errors)
        if len(sections) < 3:
            errors.append(f"{field}.sections must contain at least three sections")
        if markdown is not None and not markdown.startswith("# "):
            errors.append(f"{field}.markdown must start with a markdown H1")
    return document_paths


def _validate_tracks(
    data: dict[str, Any],
    document_paths: set[str],
    errors: list[str],
) -> set[str]:
    tracks = data.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        errors.append("tracks must be a non-empty list")
        return set()
    track_ids: set[str] = set()
    for index, track in enumerate(tracks):
        field = f"tracks[{index}]"
        if not isinstance(track, dict):
            errors.append(f"{field} must be an object")
            continue
        track_id = _non_empty_string(track.get("track_id"), f"{field}.track_id", errors)
        if track_id is not None:
            track_ids.add(track_id)
        _non_empty_string(track.get("audience"), f"{field}.audience", errors)
        docs = _require_string_list(track.get("documents"), f"{field}.documents", errors)
        _require_string_list(track.get("prerequisites"), f"{field}.prerequisites", errors)
        _require_string_list(track.get("first_run_commands"), f"{field}.first_run_commands", errors)
        _require_string_list(track.get("success_evidence"), f"{field}.success_evidence", errors)
        _require_string_list(track.get("escalation"), f"{field}.escalation", errors)
        for document_path in docs:
            if document_path not in document_paths:
                errors.append(f"{field}.documents references missing document {document_path!r}")
    missing = REQUIRED_TRACKS - track_ids
    if missing:
        errors.append(f"tracks missing required track_ids: {sorted(missing)}")
    return track_ids


def _validate_glossary(data: dict[str, Any], errors: list[str]) -> set[str]:
    terms = data.get("glossary_terms")
    if not isinstance(terms, list) or not terms:
        errors.append("glossary_terms must be a non-empty list")
        return set()
    term_ids: set[str] = set()
    for index, term in enumerate(terms):
        field = f"glossary_terms[{index}]"
        if not isinstance(term, dict):
            errors.append(f"{field} must be an object")
            continue
        term_id = _non_empty_string(term.get("term_id"), f"{field}.term_id", errors)
        if term_id is not None:
            term_ids.add(term_id)
        _non_empty_string(term.get("term"), f"{field}.term", errors)
        _non_empty_string(term.get("definition"), f"{field}.definition", errors)
        _require_string_list(term.get("referenced_by"), f"{field}.referenced_by", errors)
    missing = REQUIRED_GLOSSARY_TERMS - term_ids
    if missing:
        errors.append(f"glossary_terms missing required term_ids: {sorted(missing)}")
    return term_ids


def _validate_benchmark_methodology(data: dict[str, Any], errors: list[str]) -> None:
    methodology = data.get("benchmark_methodology")
    if not isinstance(methodology, dict):
        errors.append("benchmark_methodology must be an object")
        return
    _non_empty_string(methodology.get("benchmark_id"), "benchmark_methodology.benchmark_id", errors)
    if methodology.get("status") != "methodology-stub":
        errors.append("benchmark_methodology.status must be methodology-stub")
    if methodology.get("lab_reference_required") is not True:
        errors.append("benchmark_methodology.lab_reference_required must be true")
    if methodology.get("benchmark_harness_is_production_code") is not True:
        errors.append("benchmark_methodology.benchmark_harness_is_production_code must be true")
    if methodology.get("correctness_first") is not True:
        errors.append("benchmark_methodology.correctness_first must be true")
    if methodology.get("no_fabricated_numbers") is not True:
        errors.append("benchmark_methodology.no_fabricated_numbers must be true")
    required_inputs = set(
        _require_string_list(methodology.get("required_inputs"), "benchmark_methodology.required_inputs", errors)
    )
    missing_inputs = REQUIRED_BENCHMARK_INPUTS - required_inputs
    if missing_inputs:
        errors.append(f"benchmark_methodology.required_inputs missing: {sorted(missing_inputs)}")
    for field in (
        "reproducibility_controls",
        "correctness_artifacts",
        "publication_artifacts",
    ):
        values = _require_string_list(methodology.get(field), f"benchmark_methodology.{field}", errors)
        if len(values) < 3:
            errors.append(f"benchmark_methodology.{field} must contain at least three entries")
    gate_mapping = methodology.get("gate_mapping")
    if not isinstance(gate_mapping, dict):
        errors.append("benchmark_methodology.gate_mapping must be an object")
    else:
        for gate in ("G1", "G2", "G3", "G4", "G5"):
            _non_empty_string(gate_mapping.get(gate), f"benchmark_methodology.gate_mapping.{gate}", errors)
    development_path = methodology.get("development_path")
    if not isinstance(development_path, dict):
        errors.append("benchmark_methodology.development_path must be an object")
    else:
        _require_string_list(
            development_path.get("t0_t1_commands"),
            "benchmark_methodology.development_path.t0_t1_commands",
            errors,
        )
        if development_path.get("simulation_warning_required") is not True:
            errors.append("benchmark_methodology.development_path.simulation_warning_required must be true")
        if development_path.get("not_lab_evidence") is not True:
            errors.append("benchmark_methodology.development_path.not_lab_evidence must be true")


def _validate_operator_handoff(data: dict[str, Any], errors: list[str]) -> None:
    handoff = data.get("operator_handoff")
    if not isinstance(handoff, dict):
        errors.append("operator_handoff must be an object")
        return
    checklist = _require_string_list(handoff.get("checklist"), "operator_handoff.checklist", errors)
    if len(checklist) < 5:
        errors.append("operator_handoff.checklist must contain at least five items")
    if handoff.get("product_ga_complete") is not False:
        errors.append("operator_handoff.product_ga_complete must be false for T1 simulation")


def validate_onboarding_methodology_fixture(data: dict[str, Any]) -> dict[str, Any]:
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
    _non_empty_string(data.get("package_id"), "package_id", errors)

    document_paths = _validate_documents(data, errors)
    track_ids = _validate_tracks(data, document_paths, errors)
    term_ids = _validate_glossary(data, errors)
    _validate_benchmark_methodology(data, errors)
    _validate_operator_handoff(data, errors)

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    tracks = data.get("tracks") if isinstance(data.get("tracks"), list) else []
    documents = data.get("documents") if isinstance(data.get("documents"), dict) else {}
    glossary_terms = data.get("glossary_terms") if isinstance(data.get("glossary_terms"), list) else []
    if summary.get("track_count") != len(tracks):
        errors.append("summary.track_count must equal len(tracks)")
    if summary.get("document_count") != len(documents):
        errors.append("summary.document_count must equal len(documents)")
    if summary.get("glossary_term_count") != len(glossary_terms):
        errors.append("summary.glossary_term_count must equal len(glossary_terms)")
    if summary.get("required_tracks_present") is not (REQUIRED_TRACKS <= track_ids):
        errors.append("summary.required_tracks_present must reflect required track coverage")
    if summary.get("required_documents_present") is not (REQUIRED_DOCUMENTS <= document_paths):
        errors.append("summary.required_documents_present must reflect required document coverage")
    if summary.get("required_glossary_terms_present") is not (REQUIRED_GLOSSARY_TERMS <= term_ids):
        errors.append("summary.required_glossary_terms_present must reflect required glossary coverage")
    if summary.get("benchmark_methodology_present") is not True:
        errors.append("summary.benchmark_methodology_present must be true")
    if summary.get("correctness_first") is not True:
        errors.append("summary.correctness_first must be true")
    if summary.get("lab_reference_required") is not True:
        errors.append("summary.lab_reference_required must be true")
    if summary.get("reproducibility_fields_count") != len(REQUIRED_BENCHMARK_INPUTS):
        errors.append("summary.reproducibility_fields_count must equal required benchmark input count")
    if summary.get("onboarding_complete_for_simulation") is not True:
        errors.append("summary.onboarding_complete_for_simulation must be true")
    if summary.get("product_ga_complete") is not False:
        errors.append("summary.product_ga_complete must be false for T1 simulation")

    warnings.append("onboarding methodology is simulation evidence, not G5 product-GA closure evidence")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "track_count": summary.get("track_count"),
            "document_count": summary.get("document_count"),
            "glossary_term_count": summary.get("glossary_term_count"),
            "required_tracks_present": summary.get("required_tracks_present") is True,
            "required_documents_present": summary.get("required_documents_present") is True,
            "required_glossary_terms_present": summary.get("required_glossary_terms_present") is True,
            "benchmark_methodology_present": summary.get("benchmark_methodology_present") is True,
            "lab_reference_required": summary.get("lab_reference_required") is True,
            "correctness_first": summary.get("correctness_first") is True,
            "onboarding_complete_for_simulation": summary.get("onboarding_complete_for_simulation") is True,
            "product_ga_complete": summary.get("product_ga_complete") is True,
        },
    }


def validate_onboarding_methodology(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid onboarding methodology artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["onboarding methodology artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_onboarding_methodology_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
