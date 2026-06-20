from __future__ import annotations

import platform
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any


STATUS_VALUES = ("probing", "partial", "sufficient", "regressed")
APPLE_ROLE_VALUES = (
    "undecided",
    "capacity-only",
    "expert-worker",
    "kv-stage",
    "arbitrary-stage",
)


def probe_local_substrate_environment() -> dict[str, Any]:
    """Record local MAX/Mojo tool presence without treating it as capability proof."""

    tools: list[dict[str, Any]] = []
    for tool in ("max", "mojo", "modular"):
        path = shutil.which(tool)
        entry: dict[str, Any] = {
            "tool": tool,
            "available": path is not None,
            "path": path or "",
        }
        if path is not None:
            entry.update(_version_probe(path))
        tools.append(entry)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "tools": tools,
        "note": (
            "Local tool discovery is provenance only. Fornax Apple role assignment "
            "still requires the rank-1 target expert-MLP probe on the pinned build."
        ),
    }


def render_substrate_adr_draft(
    *,
    pinned_build: str = "unset",
    last_checked: str | None = None,
    status: str = "probing",
    apple_role: str = "undecided",
    local_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the Phase-0 MAX/Mojo substrate ADR draft."""

    if status not in STATUS_VALUES:
        raise ValueError(f"status must be one of {', '.join(STATUS_VALUES)}")
    if apple_role not in APPLE_ROLE_VALUES:
        raise ValueError(
            f"apple_role must be one of {', '.join(APPLE_ROLE_VALUES)}"
        )
    checked = last_checked or datetime.now(timezone.utc).date().isoformat()
    probe = (
        local_probe if local_probe is not None else probe_local_substrate_environment()
    )
    warnings = _warnings(pinned_build=pinned_build, status=status, local_probe=probe)
    markdown = _render_markdown(
        pinned_build=pinned_build,
        last_checked=checked,
        status=status,
        apple_role=apple_role,
        local_probe=probe,
        warnings=warnings,
    )
    return {
        "markdown": markdown,
        "ok": True,
        "warnings": warnings,
        "pinned_build": pinned_build,
        "last_checked": checked,
        "status": status,
        "apple_role": apple_role,
        "local_probe": probe,
    }


def _version_probe(path: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"version_error": str(exc)}
    output = (result.stdout or result.stderr or "").strip().splitlines()
    return {
        "returncode": result.returncode,
        "version": output[0] if output else "",
    }


def _warnings(
    *, pinned_build: str, status: str, local_probe: dict[str, Any]
) -> list[str]:
    warnings: list[str] = []
    if pinned_build.strip().lower() in {"", "unset", "unknown"}:
        warnings.append("pinned MAX/Mojo build is not set")
    tools = local_probe.get("tools") if isinstance(local_probe, dict) else None
    if isinstance(tools, list):
        available = {
            item.get("tool")
            for item in tools
            if isinstance(item, dict) and item.get("available")
        }
        if "max" not in available:
            warnings.append("local `max` tool was not discovered")
        if "mojo" not in available:
            warnings.append("local `mojo` tool was not discovered")
    if status != "sufficient":
        warnings.append("Apple expert-MLP capability remains unproven")
    return warnings


def _tool_table(local_probe: dict[str, Any]) -> list[str]:
    rows = ["| Tool | Available | Version | Path |", "|---|---:|---|---|"]
    tools = local_probe.get("tools") if isinstance(local_probe, dict) else None
    if not isinstance(tools, list) or not tools:
        rows.append("| n/a | n/a | n/a | n/a |")
        return rows
    for item in tools:
        if not isinstance(item, dict):
            rows.append("| invalid | n/a | n/a | n/a |")
            continue
        rows.append(
            "| {tool} | {available} | {version} | `{path}` |".format(
                tool=_escape_table(item.get("tool", "missing")),
                available="yes" if item.get("available") else "no",
                version=_escape_table(
                    item.get("version")
                    or item.get("version_error")
                    or "not discovered"
                ),
                path=_escape_table(item.get("path", "")),
            )
        )
    return rows


def _warning_table(warnings: list[str]) -> list[str]:
    rows = ["| Category | Message |", "|---|---|"]
    if not warnings:
        rows.append("| ok | no generator warnings |")
        return rows
    rows.extend(f"| warning | {_escape_table(warning)} |" for warning in warnings)
    return rows


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _render_markdown(
    *,
    pinned_build: str,
    last_checked: str,
    status: str,
    apple_role: str,
    local_probe: dict[str, Any],
    warnings: list[str],
) -> str:
    platform_name = (
        local_probe.get("platform", "missing")
        if isinstance(local_probe, dict)
        else "missing"
    )
    machine = (
        local_probe.get("machine", "missing")
        if isinstance(local_probe, dict)
        else "missing"
    )
    probe_note = (
        local_probe.get("note", "")
        if isinstance(local_probe, dict)
        else ""
    )
    lines: list[str] = [
        "# ADR 0001 - MAX/Mojo Substrate Draft",
        "",
        (
            "Status: DRAFT - generated by `fornax spec substrate-adr`; "
            "not TL/SP sign-off and not a G1 closure claim."
        ),
        "",
        "## Decision",
        "",
        (
            "Fornax uses MAX/Mojo as the preferred substrate for Phase-0 "
            "planning and later runtime surgery because the program needs one "
            "portable path across Linux NVIDIA, Linux AMD, and macOS Apple "
            "Silicon. This is a strategic bet, not proof that every target "
            "operation is supported today."
        ),
        "",
        "## Watch Register",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| ADR path | `adr/0001-max-mojo-substrate.md` |",
        f"| Pinned MAX/Mojo build | `{_escape_table(pinned_build)}` |",
        "| Capability needed for v0 | target-model expert MLP on the target Mac within contract correctness and throughput bounds |",
        "| Adjudicated by | rank-1 local probe on the pinned build in the target environment |",
        f"| Last checked | {last_checked} |",
        f"| Status | {status} |",
        f"| Current Apple role | {apple_role} |",
        "| Reversal trigger armed? | yes |",
        "| Owner | KER for Apple probe; TL/SP for gate decision |",
        "",
        "## Generator Warnings",
        "",
        *_warning_table(warnings),
        "",
        "## Source Precedence",
        "",
        (
            "When sources disagree, capability is unproven until the local "
            "probe passes. Higher rank wins."
        ),
        "",
        "| Rank | Source | Authority |",
        "|---:|---|---|",
        "| 1 | Local probe on the pinned build in the target environment | gate of record for Fornax role assignment |",
        "| 2 | Package docs and changelog for the pinned build | official support status |",
        "| 3 | Supported-model catalog and model docs | model-level availability |",
        "| 4 | Blog posts and launch announcements | directional signal only; never a release gate |",
        "| 5 | Nightly behavior | useful only after pinned, probed, and recorded |",
        "",
        "## Upstream Anchors To Recheck",
        "",
        (
            "- Modular docs and FAQ are rank-2 sources for install, platform, "
            "and serving support on the pinned build: `https://docs.modular.com/` "
            "and `https://docs.modular.com/max/faq/`."
        ),
        (
            "- Supported-model pages are rank-3 sources. They can narrow what "
            "model or operation is plausible, but they do not replace the "
            "target expert-MLP probe."
        ),
        (
            "- Release blogs and launch notes are rank-4 direction only. They "
            "may justify a probe, not a gate pass."
        ),
        "",
        "## Local Environment Snapshot",
        "",
        f"- Platform: `{platform_name}`",
        f"- Machine: `{machine}`",
        f"- Probe note: {probe_note or 'none'}",
        "",
        *_tool_table(local_probe),
        "",
        "## Pinned Build Policy",
        "",
        (
            "- Every target-contract, Apple probe, benchmark, and gate artifact "
            "must record MAX/Mojo build, OS, driver/runtime versions, hardware, "
            "quantization, command, and log path."
        ),
        (
            "- Changing the pinned build invalidates prior Apple role evidence "
            "until the probe is rerun and recorded."
        ),
        (
            "- Nightlies can unblock exploration only after they are explicitly "
            "pinned, probed, and copied into the evidence ledger."
        ),
        (
            "- If an upstream release changes the decision, record the change "
            "as a dated DEC entry before treating it as program state."
        ),
        "",
        "## Apple Plan B And Reversal Trigger",
        "",
        (
            "Apple participation is assigned to the highest role proven by G1, "
            "in this order: capacity/store, expert worker, KV-heavy decode "
            "stage, arbitrary pipeline stage."
        ),
        "",
        (
            "If the target model's expert MLP cannot run on the target Mac on "
            "the pinned build within the contract's correctness and throughput "
            "bounds by G1, Apple is demoted to `capacity-only` and the v0 thesis "
            "narrows. This is an accepted gate outcome, not a late failure."
        ),
        "",
        "## Required Rank-1 Probe",
        "",
        (
            "- Run the exact target-model expert MLP or a TL-approved isolated "
            "equivalent with the target quantization and in-flight dtype."
        ),
        (
            "- Compare outputs against the slow reference path within the "
            "runtime-format per-dtype tolerances."
        ),
        (
            "- Measure throughput against the target-contract bound under "
            "recorded thermals and power mode."
        ),
        (
            "- Record command, build, hardware, OS, model hash, tokenizer or "
            "template version if relevant, input fixture, output checksum, and "
            "log path."
        ),
        "",
        "## Rejected Alternatives",
        "",
        "| Alternative | Decision | Rationale |",
        "|---|---|---|",
        "| llama.cpp / ggml as primary substrate | reject as primary | strong local baseline, but it does not provide the intended MAX/Mojo cross-vendor surgery layer |",
        "| MLX as primary substrate | reject as primary | useful Apple baseline, but it does not cover Linux NVIDIA/AMD target fleets |",
        "| vLLM or SGLang as primary substrate | reject as primary | strong homogeneous-serving baselines, but less aligned with cross-vendor MAX/Mojo graph surgery |",
        "| black-box `max serve` only | reject as sufficient | useful baseline and compatibility check, but Fornax needs planner-owned placement, transport, and expert-worker seams |",
        "| custom runtime from scratch | reject for v0 | too much scope before G1; use MAX/Mojo where it is real and isolate custom seams |",
        "",
        "## Implications",
        "",
        (
            "- Phase 1 distributed runtime engineering remains blocked until "
            "G1 accepts this ADR together with the target contract, runtime "
            "format spec, networking/backpressure spec, preflight workflow, and "
            "Apple role decision."
        ),
        (
            "- MAX/Mojo is the preferred path only where the pinned build and "
            "local probes prove it. Unsupported operations remain explicit "
            "backend gaps, not assumptions."
        ),
        (
            "- Existing engines remain baselines and fallback references for "
            "single-node or homogeneous cases."
        ),
        "",
        "## Review Checklist",
        "",
        "- MAX/Mojo rationale is explicit and scoped as a strategic bet.",
        "- Rejected alternatives are named with concrete reasons.",
        "- Source precedence is explicit, with local probe as gate of record.",
        "- Pinned build policy is explicit.",
        "- Apple Plan B and reversal trigger are explicit.",
        "- Probe owner, command evidence, and status-update path are explicit.",
        "- TL/SP review signs the final ADR before G1 closure is claimed.",
        "",
    ]
    return "\n".join(lines)
