from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .planner.evidence import EvidenceRegistry


G2_EVIDENCE_SCHEMA_VERSION = 1
MAX_LINEAGE_SCHEMA_VERSION = 1
G2_RUN_MANIFEST_SCHEMA_VERSION = 1

PHYSICAL_STEPS: tuple[dict[str, Any], ...] = (
    {
        "step_id": "V6_NVIDIA",
        "tier": "T2-physical-single-node",
        "title": "NVIDIA operator and stage parity",
        "roles": ("nvidia",),
        "dependencies": (),
    },
    {
        "step_id": "V6_APPLE",
        "tier": "T2-physical-single-node",
        "title": "Apple operator and stage parity plus role decision",
        "roles": ("apple",),
        "dependencies": (),
    },
    {
        "step_id": "V7_PIPELINE",
        "tier": "T3-physical-multinode",
        "title": "Linux/NVIDIA to macOS/Apple prefill and decode",
        "roles": ("nvidia", "apple"),
        "dependencies": ("V6_NVIDIA", "V6_APPLE"),
    },
    {
        "step_id": "V8_LOAD_CALIBRATION",
        "tier": "T3-physical-multinode",
        "title": "Concurrency, attribution, and planner calibration",
        "roles": ("nvidia", "apple"),
        "dependencies": ("V7_PIPELINE",),
    },
    {
        "step_id": "V9_STABILITY",
        "tier": "T3-physical-multinode",
        "title": "Thirty-minute stability and bounded lifecycle",
        "roles": ("nvidia", "apple"),
        "dependencies": ("V8_LOAD_CALIBRATION",),
    },
    {
        "step_id": "V10_FAILURES",
        "tier": "T3-physical-multinode",
        "title": "Physical failure and cleanup matrix",
        "roles": ("nvidia", "apple"),
        "dependencies": ("V7_PIPELINE",),
    },
)

_PHYSICAL_BY_ID = {row["step_id"]: row for row in PHYSICAL_STEPS}
_HASH_REPLACEMENTS = {
    "",
    "unset",
    "unknown",
    "todo",
    "tbd",
    "replace-me",
    "placeholder",
    "n/a",
    "none",
}
_NODE_IDENTITY_FIELDS = (
    "role",
    "physical_host_id",
    "hostname",
    "os_build",
    "architecture",
    "device_identity",
    "driver_runtime",
    "max_cli_version",
    "mojo_version",
    "bazel_version",
    "bazelisk_version",
    "python_version",
    "compiler_version",
    "toolchain_version",
    "build_target",
    "build_flags_sha256",
    "build_environment_sha256",
    "max_patch_commit",
    "max_binary_sha256",
    "memory_bytes",
)
_REQUIRED_CONTEXT_TOKENS = (16, 128, 512, 4096)
_REQUIRED_CONCURRENCY = (1, 4, 8)
_CANONICAL_FAULTS: dict[str, dict[str, Any]] = {
    "cancel": {
        "outcome_code": "CANCELLED",
        "replay_disposition": "terminal_rejected",
        "mutation_count": 0,
    },
    "timeout": {
        "outcome_code": "DEADLINE_EXCEEDED",
        "replay_disposition": "terminal_rejected",
        "mutation_count": 0,
    },
    "stale_plan": {
        "outcome_code": "STALE_PLAN",
        "replay_disposition": "terminal_rejected",
        "mutation_count": 0,
    },
    "crc": {
        "outcome_code": "CRC_MISMATCH",
        "replay_disposition": "frame_rejected",
        "mutation_count": 0,
    },
    "link_loss": {
        "outcome_code": "LINK_LOSS_RECOVERED",
        "replay_disposition": "replay_deduplicated",
        "mutation_count": 1,
    },
}
_EXECUTION_SOURCE_PATHS = (
    "Makefile",
    "pyproject.toml",
    "fornax",
    "tests",
    "dependencies",
)


def _default_prerequisites(python_executable: str) -> list[dict[str, Any]]:
    return [
        {
            "step_id": "V1_PLANNER_REGRESSIONS",
            "tier": "T0-contract",
            "title": "Planner memory and greater-than-six-node regressions",
            "argv": [
                python_executable,
                "-m",
                "unittest",
                "tests.test_fornax_planner.FornaxPlannerTest.test_remote_expert_concurrent_assignments_share_host_capacity",
                "tests.test_fornax_planner.FornaxPlannerTest.test_more_than_six_nodes_retains_only_feasible_high_memory_node",
            ],
        },
        {
            "step_id": "V2_STAGE_ABI",
            "tier": "T0/T1-contract",
            "title": "Stage ABI valid and malformed-frame corpus",
            "argv": [python_executable, "-m", "fornax", "test", "stage-abi-v1"],
        },
        {
            "step_id": "V2_RUNTIME_FORMAT",
            "tier": "T0-contract",
            "title": "Runtime-format golden vectors",
            "argv": [python_executable, "-m", "fornax", "test", "runtime-format"],
        },
        {
            "step_id": "V2_STAGE_ABI_V2",
            "tier": "T0/T1-contract",
            "title": "FNX2 ragged multi-sequence two-worker golden",
            "argv": [python_executable, "-m", "fornax", "test", "stage-abi-v2"],
        },
        {
            "step_id": "V3_REFERENCE_STAGE",
            "tier": "T1-simulation",
            "title": "Reference stage boundary and lifecycle",
            "argv": [python_executable, "-m", "fornax", "test", "stage-host"],
        },
        {
            "step_id": "V4_SIMULATED_MAX",
            "tier": "T1-simulation",
            "title": "Reference/simulated MAX parity and injected failures",
            "argv": [
                python_executable,
                "-m",
                "unittest",
                "tests.test_fornax_phase05.Phase05StageRuntimeTest.test_reference_and_simulated_backends_share_contract",
                "tests.test_fornax_phase05.Phase05StageRuntimeTest.test_cancel_stale_plan_deadline_and_faults_fail_closed",
            ],
        },
        {
            "step_id": "V5_NETWORK_TRANSPORT",
            "tier": "T1-simulation",
            "title": "Network contract",
            "argv": [python_executable, "-m", "fornax", "test", "network-contract"],
        },
        {
            "step_id": "V5_WORKER_CONTRACT",
            "tier": "T1-simulation",
            "title": "Worker contract",
            "argv": [python_executable, "-m", "fornax", "test", "worker-contract"],
        },
        {
            "step_id": "V5_TRANSPORT_CONTRACT",
            "tier": "T1-simulation",
            "title": "Transport contract",
            "argv": [python_executable, "-m", "fornax", "test", "transport-contract"],
        },
        {
            "step_id": "V5_LOOPBACK_WORKERS",
            "tier": "T1-simulation",
            "title": "Two independent loopback workers, credit, reconnect, and sustained runner",
            "argv": [
                python_executable,
                "-m",
                "unittest",
                "tests.test_fornax_phase05.Phase05EngineV0Test.test_two_independent_worker_processes_execute_prefill_and_decode",
                "tests.test_fornax_phase05.Phase05EngineV0Test.test_channel_credit_cancel_and_reconnect",
                "tests.test_fornax_phase05.Phase05EngineV0Test.test_sustained_runner_uses_wall_clock_and_real_concurrency",
            ],
        },
    ]


def _now_iso(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started_at = _now_iso()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=dict(environment) if environment is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
        launch_error = None
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = _decode_subprocess_output(exc.stdout)
        stderr = _decode_subprocess_output(exc.stderr)
        timed_out = True
        launch_error = f"timeout after {timeout_seconds:g} seconds"
    except OSError as exc:
        returncode = None
        stdout = ""
        stderr = ""
        timed_out = False
        launch_error = f"{type(exc).__name__}: {exc}"
    return {
        "argv": list(argv),
        "command": shlex.join(argv),
        "cwd": str(cwd),
        "started_at": started_at,
        "completed_at": _now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "timeout_seconds": timeout_seconds,
        "returncode": returncode,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "stdout": stdout,
        "stderr": stderr,
    }


def _decode_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _git_text(repo: Path, *args: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    # Porcelain status uses its leading two columns as data (for example
    # `` M path``).  Strip only line terminators so the first record cannot lose
    # a status column and evade execution-source filtering.
    return (
        completed.returncode,
        completed.stdout.rstrip("\r\n"),
        completed.stderr.strip(),
    )


def _git_bytes(repo: Path, *args: str) -> tuple[int, bytes, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr.decode("utf-8", errors="replace").strip(),
    )


def _execution_relevant_status(line: str) -> bool:
    path = line[3:] if len(line) > 3 else line
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return (
        path == "Makefile"
        or path == "pyproject.toml"
        or path.startswith("fornax/")
        or path.startswith("tests/")
        or path.startswith("dependencies/")
    )


def capture_fornax_source(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    head_code, head, head_error = _git_text(root, "rev-parse", "HEAD")
    tree_code, tree, tree_error = _git_text(root, "rev-parse", "HEAD^{tree}")
    branch_code, branch, _ = _git_text(root, "branch", "--show-current")
    status_code, status, status_error = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_EXECUTION_SOURCE_PATHS,
    )
    diff_code, diff, diff_error = _git_bytes(
        root,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *_EXECUTION_SOURCE_PATHS,
    )
    status_lines = status.splitlines() if status else []
    relevant_status = [
        line for line in status_lines if _execution_relevant_status(line)
    ]
    errors = [
        value
        for code, value in (
            (head_code, head_error),
            (tree_code, tree_error),
            (status_code, status_error),
            (diff_code, diff_error),
        )
        if code != 0 and value
    ]
    return {
        "repository_path": str(root),
        "git_available": not errors and head_code == 0,
        "head_commit": head if head_code == 0 else None,
        "head_tree": tree if tree_code == 0 else None,
        "branch": branch if branch_code == 0 else None,
        "dirty": bool(relevant_status),
        "status_scope": list(_EXECUTION_SOURCE_PATHS),
        "status": relevant_status,
        "execution_relevant_status": relevant_status,
        "execution_source_clean": not relevant_status and not errors,
        "tracked_diff_sha256": _sha256_bytes(diff) if diff_code == 0 else None,
        "errors": errors,
    }


def _normalise_remote(url: str) -> str:
    value = url.strip().rstrip("/")
    return value[:-4] if value.endswith(".git") else value


def verify_max_lineage(
    repo_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    supplied_path = Path(manifest_path)
    pin_path = supplied_path if supplied_path.is_absolute() else root / supplied_path
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, expected: Any, observed: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASSED" if passed else "FAILED",
                "expected": expected,
                "observed": observed,
            }
        )
        if not passed:
            errors.append(f"{name}: expected {expected!r}, observed {observed!r}")

    if not pin_path.is_file():
        return {
            "ok": False,
            "manifest_path": str(pin_path),
            "manifest_sha256": None,
            "checkout_path": None,
            "checks": [],
            "errors": ["MAX lineage manifest is missing"],
        }
    try:
        data = _read_json_object(pin_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "ok": False,
            "manifest_path": str(pin_path),
            "manifest_sha256": _sha256_file(pin_path),
            "checkout_path": None,
            "checks": [],
            "errors": [f"invalid MAX lineage manifest: {exc}"],
        }

    check(
        "manifest-schema",
        data.get("schema_version") == MAX_LINEAGE_SCHEMA_VERSION,
        MAX_LINEAGE_SCHEMA_VERSION,
        data.get("schema_version"),
    )
    check(
        "dependency-name",
        data.get("dependency") == "modular-max",
        "modular-max",
        data.get("dependency"),
    )
    repository = data.get("repository")
    lineage = data.get("lineage")
    build = data.get("build")
    if not isinstance(repository, dict):
        errors.append("repository must be an object")
        repository = {}
    if not isinstance(lineage, dict):
        errors.append("lineage must be an object")
        lineage = {}
    if not isinstance(build, dict):
        errors.append("build must be an object")
        build = {}

    checkout_value = repository.get("checkout_path")
    checkout_path: Path | None = None
    if isinstance(checkout_value, str) and checkout_value:
        candidate = (root / checkout_value).resolve()
        try:
            candidate.relative_to(root)
            checkout_path = candidate
        except ValueError:
            errors.append("repository.checkout_path escapes the Fornax repository")
    else:
        errors.append("repository.checkout_path must be a non-empty relative path")

    base = lineage.get("upstream_base_commit")
    patch = lineage.get("patch_commit")
    patch_tree = lineage.get("patch_tree")
    aggregate_hash = lineage.get("patch_diff_sha256")
    for field, value in (
        ("upstream_base_commit", base),
        ("patch_commit", patch),
        ("patch_tree", patch_tree),
    ):
        check(
            f"lineage-{field}",
            isinstance(value, str) and len(value) == 40 and _is_hex(value),
            "40 lowercase hexadecimal characters",
            value,
        )
    check(
        "lineage-patch-diff-sha256",
        _is_sha256(aggregate_hash),
        "sha256:<64 lowercase hexadecimal characters>",
        aggregate_hash,
    )
    for path_field, hash_field in (
        ("patch_file", "patch_diff_sha256"),
        ("reconstruction_script", "reconstruction_script_sha256"),
    ):
        relative_value = lineage.get(path_field)
        expected_hash = lineage.get(hash_field)
        valid_relative = _safe_relative_artifact(relative_value)
        check(
            f"lineage-{path_field}-path",
            valid_relative,
            "a safe repository-relative path",
            relative_value,
        )
        check(
            f"lineage-{hash_field}",
            _is_sha256(expected_hash),
            "sha256:<64 lowercase hexadecimal characters>",
            expected_hash,
        )
        if valid_relative:
            tracked_path = root / str(relative_value)
            observed_hash = (
                _sha256_file(tracked_path) if tracked_path.is_file() else None
            )
            check(
                f"lineage-{path_field}-content",
                observed_hash == expected_hash,
                expected_hash,
                observed_hash,
            )
    reconstruction = lineage.get("reconstruction")
    reconstruction_ok = isinstance(reconstruction, dict) and all(
        _is_concrete(reconstruction.get(field))
        for field in (
            "author_name",
            "author_email",
            "author_date",
            "committer_name",
            "committer_email",
            "committer_date",
            "message",
        )
    )
    check(
        "lineage-reconstruction-metadata",
        reconstruction_ok,
        "complete deterministic commit metadata",
        reconstruction,
    )
    check(
        "accepted-cli-version-recorded",
        _is_concrete(build.get("accepted_cli_version")),
        "a concrete MAX CLI version",
        build.get("accepted_cli_version"),
    )
    check(
        "primary-build-target-recorded",
        _is_concrete(build.get("primary_target")),
        "a concrete Bazel target",
        build.get("primary_target"),
    )

    series = lineage.get("patch_series")
    if not isinstance(series, list) or not series:
        errors.append("lineage.patch_series must be a non-empty list")
        series = []
    if series:
        last_commit = series[-1].get("commit") if isinstance(series[-1], dict) else None
        first_parent = series[0].get("parent") if isinstance(series[0], dict) else None
        check("patch-series-head", last_commit == patch, patch, last_commit)
        check("patch-series-base", first_parent == base, base, first_parent)
        for index, row in enumerate(series):
            if not isinstance(row, dict):
                errors.append(f"patch_series[{index}] must be an object")
                continue
            check(
                f"patch-series-{index}-diff-hash",
                _is_sha256(row.get("diff_sha256")),
                "sha256:<64 lowercase hexadecimal characters>",
                row.get("diff_sha256"),
            )

    instructions = data.get("fetch_instructions")
    valid_instructions = isinstance(instructions, list) and bool(instructions)
    if valid_instructions:
        for row in instructions:
            argv = row.get("argv") if isinstance(row, dict) else None
            if not _valid_argv(argv):
                valid_instructions = False
                break
    check(
        "fetch-instructions", valid_instructions, "non-empty argv arrays", instructions
    )

    if checkout_path is None or not checkout_path.is_dir():
        errors.append("pinned MAX checkout is not present")
    elif all(isinstance(value, str) and len(value) == 40 for value in (base, patch)):
        remote_name = str(repository.get("remote", "origin"))
        code, observed_remote, remote_error = _git_text(
            checkout_path, "remote", "get-url", remote_name
        )
        check(
            "repository-remote",
            code == 0
            and _normalise_remote(observed_remote)
            == _normalise_remote(str(repository.get("url", ""))),
            repository.get("url"),
            observed_remote if code == 0 else remote_error,
        )
        code, observed_head, git_error = _git_text(checkout_path, "rev-parse", "HEAD")
        check(
            "checkout-head",
            code == 0 and observed_head == patch,
            patch,
            observed_head if code == 0 else git_error,
        )
        code, observed_tree, git_error = _git_text(
            checkout_path, "rev-parse", "HEAD^{tree}"
        )
        check(
            "checkout-tree",
            code == 0 and observed_tree == patch_tree,
            patch_tree,
            observed_tree if code == 0 else git_error,
        )
        code, _, git_error = _git_text(
            checkout_path, "merge-base", "--is-ancestor", str(base), str(patch)
        )
        check("base-is-ancestor", code == 0, True, code if not git_error else git_error)
        code, observed_parent, git_error = _git_text(
            checkout_path, "rev-parse", f"{patch}^"
        )
        expected_parent = (
            series[0].get("parent")
            if len(series) == 1 and isinstance(series[0], dict)
            else None
        )
        if expected_parent is not None:
            check(
                "patch-parent",
                code == 0 and observed_parent == expected_parent,
                expected_parent,
                observed_parent if code == 0 else git_error,
            )
        code, commit_object, git_error = _git_text(
            checkout_path, "cat-file", "-p", str(patch)
        )
        if code == 0 and isinstance(reconstruction, dict):
            headers, separator, message = commit_object.partition("\n\n")
            header_lines = headers.splitlines()
            observed_author = next(
                (line for line in header_lines if line.startswith("author ")), None
            )
            observed_committer = next(
                (line for line in header_lines if line.startswith("committer ")), None
            )
            expected_author = (
                f"author {reconstruction.get('author_name')} "
                f"<{reconstruction.get('author_email')}> "
                f"{reconstruction.get('author_date')}"
            )
            expected_committer = (
                f"committer {reconstruction.get('committer_name')} "
                f"<{reconstruction.get('committer_email')}> "
                f"{reconstruction.get('committer_date')}"
            )
            check(
                "commit-author-metadata",
                observed_author == expected_author,
                expected_author,
                observed_author,
            )
            check(
                "commit-committer-metadata",
                observed_committer == expected_committer,
                expected_committer,
                observed_committer,
            )
            check(
                "commit-message",
                separator == "\n\n" and message == reconstruction.get("message"),
                reconstruction.get("message"),
                message if separator else git_error,
            )
        else:
            check("commit-object-readable", False, "readable Git commit", git_error)
        code, diff, git_error = _git_bytes(
            checkout_path, "diff", "--binary", str(base), str(patch)
        )
        observed_hash = _sha256_bytes(diff) if code == 0 else git_error
        check(
            "patch-diff-hash",
            code == 0 and observed_hash == aggregate_hash,
            aggregate_hash,
            observed_hash,
        )
        code, status, git_error = _git_text(
            checkout_path, "status", "--porcelain=v1", "--untracked-files=all"
        )
        check(
            "checkout-clean",
            code == 0 and status == "",
            "clean worktree",
            status if code == 0 else git_error,
        )

    return {
        "ok": not errors,
        "manifest_path": str(pin_path),
        "manifest_sha256": _sha256_file(pin_path),
        "checkout_path": str(checkout_path) if checkout_path else None,
        "pinned": data,
        "checks": checks,
        "errors": errors,
    }


def _is_hex(value: str) -> bool:
    return value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and _is_hex(value[7:])
    )


def _is_concrete(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in _HASH_REPLACEMENTS


def _valid_argv(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item and "\x00" not in item for item in value)
    )


def _strict_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _strict_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], label: str, errors: list[str]
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        errors.append(f"{label} contains unknown fields: {', '.join(unexpected)}")


def _read_bound_json(
    manifest_base: Path | None,
    relative: Any,
    label: str,
    errors: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    if not _safe_relative_artifact(relative):
        errors.append(f"{label} must be a safe path relative to the run manifest")
        return None, None
    if manifest_base is None:
        errors.append(f"{label} cannot be verified without the run-manifest directory")
        return None, None
    if not _contained_regular_file(manifest_base, str(relative)):
        errors.append(f"{label} is missing or is not a contained regular file")
        return None, None
    path = manifest_base / str(relative)
    try:
        return _read_json_object(path), _sha256_file(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{label} is not valid JSON: {exc}")
        return None, None


def _validate_correctness_policy(
    data: Mapping[str, Any], errors: list[str]
) -> dict[str, Any]:
    correctness = data.get("correctness")
    if not isinstance(correctness, dict):
        errors.append("correctness must be an object")
        return {}
    _reject_unknown_keys(
        correctness,
        {"reference", "tolerance_policy", "corpus"},
        "correctness",
        errors,
    )
    reference = correctness.get("reference")
    if not isinstance(reference, dict):
        errors.append("correctness.reference must be an object")
        reference = {}
    else:
        _reject_unknown_keys(
            reference,
            {"reference_id", "implementation", "artifact_sha256"},
            "correctness.reference",
            errors,
        )
    for field in ("reference_id", "implementation"):
        if not _is_concrete(reference.get(field)):
            errors.append(f"correctness.reference.{field} must be concrete")
    if not _is_sha256(reference.get("artifact_sha256")):
        errors.append("correctness.reference.artifact_sha256 must be SHA-256")

    policy = correctness.get("tolerance_policy")
    if not isinstance(policy, dict):
        errors.append("correctness.tolerance_policy must be an object")
        policy = {}
    else:
        _reject_unknown_keys(
            policy,
            {
                "approval_id",
                "dtype_tolerances",
                "nonfinite",
                "top1",
                "routing",
            },
            "correctness.tolerance_policy",
            errors,
        )
    if not _is_concrete(policy.get("approval_id")):
        errors.append("correctness.tolerance_policy.approval_id must be concrete")
    for field, expected in (
        ("nonfinite", "reject"),
        ("top1", "exact"),
        ("routing", "exact"),
    ):
        if policy.get(field) != expected:
            errors.append(
                f"correctness.tolerance_policy.{field} must equal {expected!r}"
            )
    tolerances = policy.get("dtype_tolerances")
    if not isinstance(tolerances, dict) or not tolerances:
        errors.append(
            "correctness.tolerance_policy.dtype_tolerances must be non-empty"
        )
        tolerances = {}
    elif not set(tolerances).issubset({"bf16", "fp16", "fp32"}):
        errors.append("dtype_tolerances contains an unsupported dtype")
    for dtype, tolerance in tolerances.items():
        if not isinstance(tolerance, dict):
            errors.append(f"dtype_tolerances.{dtype} must be an object")
            continue
        _reject_unknown_keys(
            tolerance, {"atol", "rtol"}, f"dtype_tolerances.{dtype}", errors
        )
        for field in ("atol", "rtol"):
            if not _numeric(tolerance.get(field)):
                errors.append(
                    f"dtype_tolerances.{dtype}.{field} must be finite and non-negative"
                )

    corpus = correctness.get("corpus")
    if not isinstance(corpus, dict):
        errors.append("correctness.corpus must be an object")
        corpus = {}
    else:
        _reject_unknown_keys(
            corpus,
            {"prompt_count", "context_tokens", "generated_tokens_per_prompt"},
            "correctness.corpus",
            errors,
        )
    if not _strict_positive_int(corpus.get("prompt_count")) or corpus.get(
        "prompt_count", 0
    ) < 20:
        errors.append("correctness.corpus.prompt_count must be an integer >= 20")
    contexts = corpus.get("context_tokens")
    if not (
        isinstance(contexts, list)
        and all(_strict_positive_int(value) for value in contexts)
        and set(_REQUIRED_CONTEXT_TOKENS).issubset(contexts)
    ):
        errors.append(
            "correctness.corpus.context_tokens must include 16, 128, 512, and 4096"
        )
    if not _strict_positive_int(corpus.get("generated_tokens_per_prompt")) or corpus.get(
        "generated_tokens_per_prompt", 0
    ) < 128:
        errors.append(
            "correctness.corpus.generated_tokens_per_prompt must be an integer >= 128"
        )
    return correctness


def _validate_stability_policy(
    data: Mapping[str, Any], errors: list[str]
) -> dict[str, Any]:
    policy = data.get("stability")
    if not isinstance(policy, dict):
        errors.append("stability must be an object")
        return {}
    _reject_unknown_keys(
        policy,
        {
            "duration_seconds",
            "sample_interval_seconds",
            "target_inflight",
            "minimum_completed_requests",
            "post_drain_timeout_seconds",
        },
        "stability",
        errors,
    )
    minimums = {
        "duration_seconds": 1800,
        "sample_interval_seconds": 1,
        "target_inflight": 1,
        "minimum_completed_requests": 20,
        "post_drain_timeout_seconds": 1,
    }
    for field, minimum in minimums.items():
        value = policy.get(field)
        if not _strict_positive_int(value) or value < minimum:
            errors.append(f"stability.{field} must be an integer >= {minimum}")
    duration = policy.get("duration_seconds")
    cadence = policy.get("sample_interval_seconds")
    if _strict_positive_int(duration) and _strict_positive_int(cadence):
        if cadence > duration:
            errors.append("stability.sample_interval_seconds exceeds duration")
    return policy


def _validate_bound_manifests(
    data: Mapping[str, Any],
    *,
    manifest_base: Path | None,
    nodes_by_role: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
    plan_document, observed_plan_hash = _read_bound_json(
        manifest_base, plan.get("plan_artifact"), "plan.plan_artifact", errors
    )
    expected_plan_hash = plan.get("plan_hash")
    if observed_plan_hash is not None and observed_plan_hash != expected_plan_hash:
        errors.append("plan.plan_hash does not match the actual plan artifact bytes")

    registry_document, observed_registry_hash = _read_bound_json(
        manifest_base,
        plan.get("evidence_registry_artifact"),
        "plan.evidence_registry_artifact",
        errors,
    )
    registry: EvidenceRegistry | None = None
    evidence_artifact_paths: list[str] = []
    registry_relative = plan.get("evidence_registry_artifact")
    if (
        manifest_base is not None
        and isinstance(registry_relative, str)
        and _safe_relative_artifact(registry_relative)
        and _contained_regular_file(manifest_base, registry_relative)
    ):
        try:
            registry = EvidenceRegistry.from_file(manifest_base / registry_relative)
        except ValueError as exc:
            errors.append(f"planner evidence registry is invalid: {exc}")
    if registry_document is not None:
        records = registry_document.get("records")
        registry_base = (
            (manifest_base / str(registry_relative)).parent
            if manifest_base is not None and isinstance(registry_relative, str)
            else None
        )
        if isinstance(records, list):
            for index, record in enumerate(records):
                artifact_path = (
                    record.get("artifact_path") if isinstance(record, dict) else None
                )
                if not _safe_relative_artifact(artifact_path):
                    errors.append(
                        f"planner evidence registry record {index} artifact_path is unsafe"
                    )
                    continue
                if registry_base is None or not _contained_regular_file(
                    registry_base, str(artifact_path)
                ):
                    errors.append(
                        f"planner evidence registry record {index} artifact is missing"
                    )
                    continue
                resolved_artifact = (registry_base / str(artifact_path)).resolve()
                try:
                    relative_to_manifest = resolved_artifact.relative_to(
                        manifest_base.resolve() if manifest_base is not None else registry_base
                    )
                except ValueError:
                    errors.append(
                        f"planner evidence registry record {index} escapes the run-manifest directory"
                    )
                    continue
                evidence_artifact_paths.append(relative_to_manifest.as_posix())
    if registry is not None:
        for record in registry.records:
            errors.extend(
                registry.resolution_issues(
                    record.source_id,
                    evidence_type=record.evidence_type,
                    label=f"G2 planner evidence {record.source_id}",
                )
            )

    stage_paths = plan.get("stage_manifest_artifacts")
    stage_hashes = plan.get("stage_manifest_sha256")
    stage_documents: list[dict[str, Any]] = []
    if not (
        isinstance(stage_paths, list)
        and len(stage_paths) >= 2
        and all(_safe_relative_artifact(value) for value in stage_paths)
    ):
        errors.append("plan.stage_manifest_artifacts must contain at least two safe paths")
        stage_paths = []
    if isinstance(stage_hashes, list) and len(stage_hashes) != len(stage_paths):
        errors.append("plan stage artifact/hash lists must have the same length")
    for index, relative in enumerate(stage_paths):
        document, observed_hash = _read_bound_json(
            manifest_base,
            relative,
            f"plan.stage_manifest_artifacts[{index}]",
            errors,
        )
        expected_hash = (
            stage_hashes[index]
            if isinstance(stage_hashes, list) and index < len(stage_hashes)
            else None
        )
        if observed_hash is not None and observed_hash != expected_hash:
            errors.append(
                f"plan.stage_manifest_sha256[{index}] does not match artifact bytes"
            )
        if document is not None:
            stage_documents.append(document)

    if plan_document is None:
        return {
            "plan": None,
            "stages": stage_documents,
            "frozen_predictions": {},
            "evidence_registry": registry_document,
            "evidence_registry_sha256": observed_registry_hash,
            "evidence_artifacts": evidence_artifact_paths,
        }
    _reject_unknown_keys(
        plan_document,
        {
            "schema_version",
            "plan_id",
            "model",
            "feasible",
            "authority",
            "stages",
            "frozen_predictions",
        },
        "plan artifact",
        errors,
    )
    if plan_document.get("schema_version") != 1:
        errors.append("plan artifact schema_version must equal 1")
    if plan_document.get("plan_id") != plan.get("plan_id"):
        errors.append("plan artifact plan_id does not match the run manifest")
    if plan_document.get("model") != model:
        errors.append("plan artifact model identity does not match the run manifest")
    if plan_document.get("feasible") is not True:
        errors.append("plan artifact feasible must be true")
    authority = plan_document.get("authority")
    if isinstance(authority, dict):
        required_authority_fields = {
            "requested_mode",
            "status",
            "deployment_authorized",
            "confidence",
            "prediction_expected_relative_error",
            "input_max_expected_relative_error",
            "source_ids",
            "evidence_registry_sha256",
            "reasons",
        }
        _reject_unknown_keys(
            authority,
            required_authority_fields,
            "plan artifact authority",
            errors,
        )
        missing_authority_fields = sorted(required_authority_fields - set(authority))
        if missing_authority_fields:
            errors.append(
                "plan artifact authority misses fields: "
                + ", ".join(missing_authority_fields)
            )
    if not isinstance(authority, dict) or not (
        authority.get("requested_mode") == "deployment"
        and authority.get("status") == "deployment_authoritative"
        and authority.get("deployment_authorized") is True
    ):
        errors.append(
            "plan artifact authority must be deployment-authoritative"
        )
    if isinstance(authority, dict):
        if authority.get("confidence") not in {"high", "medium"}:
            errors.append("deployment-authoritative plan confidence must be high or medium")
        for field in (
            "prediction_expected_relative_error",
            "input_max_expected_relative_error",
        ):
            value = authority.get(field)
            if not _numeric(value) or float(value) > 0.20:
                errors.append(f"plan artifact authority.{field} must be within 0.20")
        source_ids = authority.get("source_ids")
        if not (
            isinstance(source_ids, list)
            and bool(source_ids)
            and len(source_ids) == len(set(source_ids))
            and all(_is_concrete(value) for value in source_ids)
        ):
            errors.append("plan artifact authority.source_ids must be unique and concrete")
            source_ids = []
        if authority.get("reasons") != ["all deployment authority checks passed"]:
            errors.append("plan artifact authority reasons do not record a clean admission")
        if registry is None:
            errors.append("deployment-authoritative plan has no valid evidence registry")
        else:
            registry_ids = {record.source_id for record in registry.records}
            if set(source_ids) != registry_ids:
                errors.append(
                    "plan authority source_ids do not equal the resolved evidence registry"
                )
            required_types = {
                "model",
                "quantization",
                "expert_trace",
                "capability",
                "measurement",
                "calibration",
                "route",
            }
            observed_types = {record.evidence_type for record in registry.records}
            if not required_types.issubset(observed_types):
                errors.append(
                    "planner evidence registry lacks a required deployment evidence type"
                )
        if authority.get("evidence_registry_sha256") != observed_registry_hash:
            errors.append(
                "plan authority evidence_registry_sha256 does not match registry bytes"
            )

    plan_stages = plan_document.get("stages")
    if not isinstance(plan_stages, list) or len(plan_stages) < 2:
        errors.append("plan artifact stages must contain at least two stages")
        plan_stages = []
    seen_stage_ids: set[str] = set()
    seen_roles: set[str] = set()
    previous_end: int | None = None
    for index, stage in enumerate(plan_stages):
        if not isinstance(stage, dict):
            errors.append(f"plan artifact stages[{index}] must be an object")
            continue
        _reject_unknown_keys(
            stage,
            {
                "stage_id",
                "stage_index",
                "layer_start",
                "layer_end",
                "node_role",
                "physical_host_id",
            },
            f"plan artifact stages[{index}]",
            errors,
        )
        stage_id = stage.get("stage_id")
        if not _is_concrete(stage_id) or stage_id in seen_stage_ids:
            errors.append(f"plan artifact stages[{index}].stage_id is invalid/duplicate")
        elif isinstance(stage_id, str):
            seen_stage_ids.add(stage_id)
        if stage.get("stage_index") != index:
            errors.append("plan artifact stage indices must be contiguous from zero")
        start = stage.get("layer_start")
        end = stage.get("layer_end")
        if not (
            _strict_nonnegative_int(start)
            and _strict_nonnegative_int(end)
            and end >= start
        ):
            errors.append(f"plan artifact stages[{index}] has an invalid layer range")
        elif previous_end is not None and start != previous_end + 1:
            errors.append("plan artifact stage cut has a gap or overlap")
        if _strict_nonnegative_int(end):
            previous_end = end
        role = stage.get("node_role")
        if isinstance(role, str):
            seen_roles.add(role)
        bound_node = nodes_by_role.get(str(role))
        if bound_node is None or stage.get("physical_host_id") != bound_node.get(
            "physical_host_id"
        ):
            errors.append(f"plan artifact stages[{index}] node binding is not admitted")
    if plan_stages and plan_stages[0].get("layer_start") != 0:
        errors.append("plan artifact stage cut must begin at layer zero")
    if seen_roles != {"nvidia", "apple"}:
        errors.append("plan artifact stages must bind both NVIDIA and Apple roles")

    if len(stage_documents) == len(plan_stages):
        for index, (stage, document) in enumerate(zip(plan_stages, stage_documents)):
            if not isinstance(stage, dict):
                continue
            _reject_unknown_keys(
                document,
                {
                    "manifest_version",
                    "model_id",
                    "model_snapshot",
                    "model_config_hash",
                    "tokenizer_hash",
                    "template_hash",
                    "max_build_id",
                    "fornax_abi_major",
                    "fornax_abi_minor",
                    "plan_id",
                    "plan_hash",
                    "stage_id",
                    "stage_index",
                    "layer_start",
                    "layer_end",
                    "input_contract",
                    "output_contract",
                    "kv_policy",
                    "weight_artifacts",
                    "device_requirement",
                    "node_binding",
                    "max_patch_commit",
                    "max_binary_sha256",
                },
                f"stage manifest {index}",
                errors,
            )
            if document.get("manifest_version") != 1:
                errors.append(f"stage manifest {index} manifest_version must equal 1")
            node_binding = document.get("node_binding")
            if not isinstance(node_binding, dict) or set(node_binding) != {
                "role",
                "physical_host_id",
            }:
                errors.append(f"stage manifest {index} node_binding has an invalid schema")
            expected = {
                "model_id": model.get("model_id"),
                "model_snapshot": model.get("snapshot_id"),
                "model_config_hash": model.get("model_config_sha256"),
                "tokenizer_hash": model.get("tokenizer_sha256"),
                "template_hash": model.get("template_sha256"),
                "plan_id": plan.get("plan_id"),
                "plan_hash": plan.get("plan_hash"),
                "stage_id": stage.get("stage_id"),
                "stage_index": stage.get("stage_index"),
                "layer_start": stage.get("layer_start"),
                "layer_end": stage.get("layer_end"),
                "node_binding": {
                    "role": stage.get("node_role"),
                    "physical_host_id": stage.get("physical_host_id"),
                },
            }
            for field, expected_value in expected.items():
                if document.get(field) != expected_value:
                    errors.append(
                        f"stage manifest {index} {field} does not match the bound plan"
                    )
            bound_node = nodes_by_role.get(str(stage.get("node_role")), {})
            if document.get("max_patch_commit") != bound_node.get("max_patch_commit"):
                errors.append(f"stage manifest {index} MAX commit is not node-bound")
            if document.get("max_binary_sha256") != bound_node.get(
                "max_binary_sha256"
            ):
                errors.append(f"stage manifest {index} MAX binary is not node-bound")

    predictions = plan_document.get("frozen_predictions")
    frozen: dict[int, float] = {}
    if not isinstance(predictions, list):
        errors.append("plan artifact frozen_predictions must be a list")
    else:
        for index, row in enumerate(predictions):
            if not isinstance(row, dict) or set(row) != {
                "inflight",
                "aggregate_tokens_per_second",
            }:
                errors.append(f"frozen_predictions[{index}] has an invalid schema")
                continue
            inflight = row.get("inflight")
            predicted = row.get("aggregate_tokens_per_second")
            if (
                inflight not in _REQUIRED_CONCURRENCY
                or inflight in frozen
                or not _numeric(predicted, minimum=0.000001)
            ):
                errors.append(f"frozen_predictions[{index}] is invalid or duplicate")
                continue
            frozen[int(inflight)] = float(predicted)
        if set(frozen) != set(_REQUIRED_CONCURRENCY):
            errors.append("frozen_predictions must contain exactly inflight 1, 4, and 8")
    return {
        "plan": plan_document,
        "stages": stage_documents,
        "frozen_predictions": frozen,
        "evidence_registry": registry_document,
        "evidence_registry_sha256": observed_registry_hash,
        "evidence_artifacts": evidence_artifact_paths,
    }


def validate_g2_run_manifest(
    data: Mapping[str, Any], *, pinned_max_commit: str, manifest_base: Path | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    _reject_unknown_keys(
        data,
        {
            "schema_version",
            "model",
            "plan",
            "nodes",
            "network",
            "correctness",
            "stability",
            "steps",
        },
        "run manifest",
        errors,
    )
    if data.get("schema_version") != G2_RUN_MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {G2_RUN_MANIFEST_SCHEMA_VERSION}")

    model = data.get("model")
    if not isinstance(model, dict):
        errors.append("model must be an object")
        model = {}
    else:
        _reject_unknown_keys(
            model,
            {
                "model_id",
                "snapshot_id",
                "model_config_sha256",
                "weights_manifest_sha256",
                "tokenizer_sha256",
                "template_sha256",
                "prompt_corpus_sha256",
            },
            "model",
            errors,
        )
    for field in ("model_id", "snapshot_id"):
        if not _is_concrete(model.get(field)):
            errors.append(f"model.{field} must be concrete")
    for field in (
        "model_config_sha256",
        "weights_manifest_sha256",
        "tokenizer_sha256",
        "template_sha256",
        "prompt_corpus_sha256",
    ):
        if not _is_sha256(model.get(field)):
            errors.append(f"model.{field} must be sha256:<64 lowercase hex>")

    plan = data.get("plan")
    if not isinstance(plan, dict):
        errors.append("plan must be an object")
        plan = {}
    else:
        _reject_unknown_keys(
            plan,
            {
                "plan_id",
                "plan_hash",
                "plan_artifact",
                "evidence_registry_artifact",
                "stage_manifest_sha256",
                "stage_manifest_artifacts",
            },
            "plan",
            errors,
        )
    try:
        parsed_plan_id = uuid.UUID(str(plan.get("plan_id")))
        if str(parsed_plan_id) != plan.get("plan_id"):
            raise ValueError
    except (TypeError, ValueError, AttributeError):
        errors.append("plan.plan_id must be a canonical UUID")
    if not _is_sha256(plan.get("plan_hash")):
        errors.append("plan.plan_hash must be sha256:<64 lowercase hex>")
    if not _safe_relative_artifact(plan.get("plan_artifact")):
        errors.append("plan.plan_artifact must be a safe run-manifest-relative path")
    if not _safe_relative_artifact(plan.get("evidence_registry_artifact")):
        errors.append(
            "plan.evidence_registry_artifact must be a safe run-manifest-relative path"
        )
    stage_hashes = plan.get("stage_manifest_sha256")
    if not (
        isinstance(stage_hashes, list)
        and len(stage_hashes) >= 2
        and all(_is_sha256(value) for value in stage_hashes)
    ):
        errors.append("plan.stage_manifest_sha256 must contain at least two hashes")
    stage_paths = plan.get("stage_manifest_artifacts")
    if not (
        isinstance(stage_paths, list)
        and len(stage_paths) >= 2
        and all(_safe_relative_artifact(value) for value in stage_paths)
    ):
        errors.append("plan.stage_manifest_artifacts must contain at least two paths")

    nodes = data.get("nodes")
    nodes_by_role: dict[str, dict[str, Any]] = {}
    if not isinstance(nodes, list):
        errors.append("nodes must be a list")
        nodes = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue
        _reject_unknown_keys(
            node, set(_NODE_IDENTITY_FIELDS), f"nodes[{index}]", errors
        )
        role = node.get("role")
        if role not in {"nvidia", "apple"}:
            errors.append(f"nodes[{index}].role must be nvidia or apple")
            continue
        if role in nodes_by_role:
            errors.append(f"nodes contains duplicate role {role}")
        nodes_by_role[str(role)] = node
        for field in _NODE_IDENTITY_FIELDS:
            if field == "memory_bytes":
                continue
            if field in {
                "max_binary_sha256",
                "build_flags_sha256",
                "build_environment_sha256",
            }:
                if not _is_sha256(node.get(field)):
                    errors.append(
                        f"nodes[{index}].{field} must be sha256:<64 lowercase hex>"
                    )
            elif field == "max_patch_commit":
                if node.get(field) != pinned_max_commit:
                    errors.append(
                        f"nodes[{index}].max_patch_commit does not match the root pin"
                    )
            elif not _is_concrete(node.get(field)):
                errors.append(f"nodes[{index}].{field} must be concrete")
        memory_bytes = node.get("memory_bytes")
        if (
            not isinstance(memory_bytes, int)
            or isinstance(memory_bytes, bool)
            or memory_bytes <= 0
        ):
            errors.append(f"nodes[{index}].memory_bytes must be a positive integer")
    if set(nodes_by_role) != {"nvidia", "apple"}:
        errors.append("nodes must contain exactly one nvidia and one apple role")
    host_ids = [node.get("physical_host_id") for node in nodes_by_role.values()]
    if len(host_ids) == 2 and host_ids[0] == host_ids[1]:
        errors.append("nvidia and apple must have different physical_host_id values")

    network = data.get("network")
    if not isinstance(network, dict):
        errors.append("network must be an object")
        network = {}
    else:
        _reject_unknown_keys(
            network,
            {
                "source_host_id",
                "destination_host_id",
                "route",
                "interface",
                "mtu_bytes",
                "declared_link_bits_per_second",
            },
            "network",
            errors,
        )
    for field in ("source_host_id", "destination_host_id", "route", "interface"):
        if not _is_concrete(network.get(field)):
            errors.append(f"network.{field} must be concrete")
    if nodes_by_role:
        expected_source = nodes_by_role.get("nvidia", {}).get("physical_host_id")
        expected_destination = nodes_by_role.get("apple", {}).get("physical_host_id")
        if network.get("source_host_id") != expected_source:
            errors.append("network.source_host_id must identify the NVIDIA host")
        if network.get("destination_host_id") != expected_destination:
            errors.append("network.destination_host_id must identify the Apple host")
    for field in ("mtu_bytes", "declared_link_bits_per_second"):
        value = network.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"network.{field} must be a positive integer")

    _validate_correctness_policy(data, errors)
    _validate_stability_policy(data, errors)
    bound_inputs = _validate_bound_manifests(
        data,
        manifest_base=manifest_base,
        nodes_by_role=nodes_by_role,
        errors=errors,
    )

    steps = data.get("steps")
    if not isinstance(steps, dict):
        errors.append("steps must be an object")
        steps = {}
    normalised_steps: dict[str, dict[str, Any]] = {}
    for definition in PHYSICAL_STEPS:
        step_id = definition["step_id"]
        spec = steps.get(step_id)
        if not isinstance(spec, dict):
            blockers.append(f"{step_id}: no physical command or availability record")
            normalised_steps[step_id] = {
                "status": "BLOCKED",
                "reason": "missing step configuration",
            }
            continue
        status = spec.get("status")
        if status == "BLOCKED":
            _reject_unknown_keys(
                spec, {"status", "reason"}, f"steps.{step_id}", errors
            )
            reason = spec.get("reason")
            if not _is_concrete(reason):
                errors.append(f"steps.{step_id}.reason must explain the blocker")
                reason = "blocked without a concrete reason"
            blockers.append(f"{step_id}: {reason}")
            normalised_steps[step_id] = {"status": "BLOCKED", "reason": reason}
            continue
        if status != "READY":
            errors.append(f"steps.{step_id}.status must be READY or BLOCKED")
            normalised_steps[step_id] = {
                "status": "BLOCKED",
                "reason": "invalid step status",
            }
            continue
        _reject_unknown_keys(
            spec,
            {"status", "argv", "cwd", "timeout_seconds", "result_artifact"},
            f"steps.{step_id}",
            errors,
        )
        argv = spec.get("argv")
        if not _valid_argv(argv):
            errors.append(f"steps.{step_id}.argv must be a non-empty string array")
        timeout = spec.get("timeout_seconds", 3600)
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            errors.append(f"steps.{step_id}.timeout_seconds must be positive")
            timeout = 3600
        if step_id == "V9_STABILITY" and float(timeout) <= 1800:
            errors.append(
                "steps.V9_STABILITY.timeout_seconds must exceed the 1800-second soak"
            )
        result_artifact = spec.get("result_artifact", "result.json")
        if not _safe_relative_artifact(result_artifact):
            errors.append(
                f"steps.{step_id}.result_artifact must be a safe relative path"
            )
            result_artifact = "result.json"
        cwd = spec.get("cwd", "{repo_root}")
        if not isinstance(cwd, str) or not cwd or "\x00" in cwd:
            errors.append(f"steps.{step_id}.cwd must be a non-empty string")
            cwd = "{repo_root}"
        normalised_steps[step_id] = {
            "status": "READY",
            "argv": list(argv) if isinstance(argv, list) else [],
            "cwd": cwd,
            "timeout_seconds": float(timeout),
            "result_artifact": result_artifact,
        }

    unexpected_steps = sorted(set(steps) - set(_PHYSICAL_BY_ID))
    if unexpected_steps:
        errors.append("steps contains unknown IDs: " + ", ".join(unexpected_steps))
    return {
        "ok": not errors,
        "errors": errors,
        "blockers": blockers,
        "normalised_steps": normalised_steps,
        "nodes_by_role": nodes_by_role,
        "bound_inputs": bound_inputs,
    }


def _safe_relative_artifact(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _contained_regular_file(base: Path, relative: str) -> bool:
    candidate = base / relative
    if candidate.is_symlink() or not candidate.is_file():
        return False
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _numeric(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _validate_timing_summary(result: Mapping[str, Any], errors: list[str]) -> None:
    timings = result.get("timings_ms")
    required = (
        "stage_0",
        "pack",
        "transfer",
        "unpack",
        "stage_1",
        "exposed_wait",
        "end_to_end",
    )
    if not isinstance(timings, dict):
        errors.append("timings_ms must be an object")
        return
    for field in required:
        value = timings.get(field)
        if (
            not isinstance(value, dict)
            or not _numeric(value.get("median"))
            or not _numeric(value.get("p95"))
        ):
            errors.append(
                f"timings_ms.{field} must contain non-negative median and p95"
            )


def _load_raw_measurements(
    result: Mapping[str, Any],
    *,
    artifact_base: Path | None,
    expected_nonce: str | None,
    errors: list[str],
) -> dict[str, Any] | None:
    relative = result.get("raw_measurements_artifact")
    raw_artifacts = result.get("raw_artifacts")
    if not _safe_relative_artifact(relative):
        errors.append("raw_measurements_artifact must be a safe relative path")
        return None
    if not isinstance(raw_artifacts, list) or relative not in raw_artifacts:
        errors.append("raw_measurements_artifact must be listed in raw_artifacts")
        return None
    if artifact_base is None:
        errors.append("artifact_base is required to verify raw measurements")
        return None
    if not _contained_regular_file(artifact_base, relative):
        errors.append(f"raw measurement artifact is missing: {relative}")
        return None
    try:
        raw = _read_json_object(artifact_base / relative)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid raw measurement artifact: {exc}")
        return None
    if raw.get("schema_version") != 1:
        errors.append("raw measurements schema_version must equal 1")
    _reject_unknown_keys(
        raw,
        {
            "schema_version",
            "step_id",
            "runner_nonce",
            "model",
            "plan",
            "max_patch_commit",
            "node_observations",
            "network_observation",
            "observations",
            "limits",
        },
        "raw measurements",
        errors,
    )
    if raw.get("step_id") != result.get("step_id"):
        errors.append("raw measurements step_id does not match result")
    if raw.get("model") != result.get("model"):
        errors.append("raw measurements model does not match result")
    if raw.get("plan") != result.get("plan"):
        errors.append("raw measurements plan does not match result")
    if raw.get("max_patch_commit") != result.get("max_patch_commit"):
        errors.append("raw measurements MAX commit does not match result")
    if not _is_concrete(expected_nonce) or raw.get("runner_nonce") != expected_nonce:
        errors.append("raw measurements do not contain the runner-issued nonce")
    observations = raw.get("observations")
    if not isinstance(observations, list) or not observations:
        errors.append("raw measurements observations must be a non-empty list")
    return raw


def _raw_numeric_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(_numeric(item, minimum=-math.inf) for item in value):
        return None
    return [float(item) for item in value]


def _derive_numerical(
    observations: Sequence[Any],
    required_scopes: set[str],
    correctness: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any] | None, set[str]]:
    maximum = 0.0
    seen: set[str] = set()
    all_within_tolerance = True
    top1_mismatch_count = 0
    routing_mismatch_count = 0
    reference = correctness.get("reference")
    policy = correctness.get("tolerance_policy")
    if not isinstance(reference, Mapping) or not isinstance(policy, Mapping):
        errors.append("validated correctness policy is unavailable")
        return None, seen
    reference_id = reference.get("reference_id")
    approval_id = policy.get("approval_id")
    tolerances = policy.get("dtype_tolerances")
    if not isinstance(tolerances, Mapping):
        errors.append("validated dtype tolerance policy is unavailable")
        return None, seen
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or row.get("kind") != "numerical":
            continue
        scope = row.get("scope")
        if scope not in required_scopes:
            continue
        dtype = row.get("dtype")
        tolerance = tolerances.get(dtype)
        if not isinstance(tolerance, Mapping):
            errors.append(
                f"raw numerical observation {index} uses an unapproved dtype"
            )
            continue
        reference_values = _raw_numeric_vector(row.get("reference"))
        observed_values = _raw_numeric_vector(row.get("observed"))
        if set(row) != {
            "kind",
            "scope",
            "dtype",
            "reference_id",
            "reference",
            "observed",
            "reference_top1",
            "observed_top1",
            "reference_routes",
            "observed_routes",
        }:
            errors.append(f"raw numerical observation {index} has an invalid schema")
            continue
        if (
            reference_values is None
            or observed_values is None
            or len(reference_values) != len(observed_values)
        ):
            errors.append(
                f"raw numerical observation {index} needs equal non-empty finite vectors"
            )
            continue
        if row.get("reference_id") != reference_id:
            errors.append(
                f"raw numerical observation {index} has the wrong reference identity"
            )
            continue
        reference_top1 = row.get("reference_top1")
        observed_top1 = row.get("observed_top1")
        if not _strict_nonnegative_int(reference_top1) or not _strict_nonnegative_int(
            observed_top1
        ):
            errors.append(
                f"raw numerical observation {index} needs integer top-1 identities"
            )
            continue
        reference_routes = row.get("reference_routes")
        observed_routes = row.get("observed_routes")
        if not (
            isinstance(reference_routes, list)
            and reference_routes
            and all(_strict_nonnegative_int(value) for value in reference_routes)
            and isinstance(observed_routes, list)
            and all(_strict_nonnegative_int(value) for value in observed_routes)
        ):
            errors.append(
                f"raw numerical observation {index} needs integer routing identities"
            )
            continue
        seen.add(str(scope))
        atol = float(tolerance["atol"])
        rtol = float(tolerance["rtol"])
        for left, right in zip(reference_values, observed_values):
            error = abs(left - right)
            maximum = max(maximum, error)
            if error > atol + rtol * abs(left):
                all_within_tolerance = False
        if reference_top1 != observed_top1:
            top1_mismatch_count += 1
        if reference_routes != observed_routes:
            routing_mismatch_count += 1
    missing = sorted(required_scopes - seen)
    if missing:
        errors.append("raw numerical observations miss scopes: " + ", ".join(missing))
        return None, seen
    summary = {
        "reference_id": reference_id,
        "tolerance_approval_id": approval_id,
        "max_abs_error": maximum,
        "nonfinite_count": 0,
        "top1_mismatch_count": top1_mismatch_count,
        "routing_mismatch_count": routing_mismatch_count,
        "all_within_tolerance": all_within_tolerance,
    }
    return summary, seen


def _derive_correctness_corpus(
    observations: Sequence[Any],
    *,
    kind: str,
    corpus: Mapping[str, Any],
    reference_id: Any,
    errors: list[str],
    inflight: int | None = None,
) -> bool:
    rows = [
        row
        for row in observations
        if isinstance(row, dict)
        and row.get("kind") == kind
        and (inflight is None or row.get("inflight") == inflight)
    ]
    prompt_ids: set[str] = set()
    contexts: set[int] = set()
    minimum_tokens = corpus.get("generated_tokens_per_prompt")
    valid = True
    for index, row in enumerate(rows):
        expected_fields = {
            "kind",
            "prompt_id",
            "context_tokens",
            "reference_id",
            "expected_token_ids",
            "observed_token_ids",
            "reference_top1",
            "observed_top1",
            "reference_routes",
            "observed_routes",
        }
        if inflight is not None:
            expected_fields.add("inflight")
        if set(row) != expected_fields:
            errors.append(f"raw {kind} observation {index} has an invalid schema")
            valid = False
            continue
        prompt_id = row.get("prompt_id")
        context_tokens = row.get("context_tokens")
        expected = row.get("expected_token_ids")
        observed = row.get("observed_token_ids")
        reference_top1 = row.get("reference_top1")
        observed_top1 = row.get("observed_top1")
        reference_routes = row.get("reference_routes")
        observed_routes = row.get("observed_routes")
        if not _is_concrete(prompt_id) or prompt_id in prompt_ids:
            errors.append(f"raw {kind} observation {index} has an invalid prompt_id")
            valid = False
        else:
            prompt_ids.add(str(prompt_id))
        if not _strict_positive_int(context_tokens):
            errors.append(f"raw {kind} observation {index} has invalid context_tokens")
            valid = False
        else:
            contexts.add(int(context_tokens))
        if not (
            isinstance(expected, list)
            and _strict_positive_int(minimum_tokens)
            and len(expected) >= int(minimum_tokens)
            and all(_strict_nonnegative_int(value) for value in expected)
            and observed == expected
        ):
            errors.append(
                f"raw {kind} observation {index} lacks the required exact generated tokens"
            )
            valid = False
        if (
            row.get("reference_id") != reference_id
            or not _strict_nonnegative_int(reference_top1)
            or observed_top1 != reference_top1
        ):
            errors.append(f"raw {kind} observation {index} fails exact top-1")
            valid = False
        if not (
            isinstance(reference_routes, list)
            and reference_routes
            and all(_strict_nonnegative_int(value) for value in reference_routes)
            and observed_routes == reference_routes
        ):
            errors.append(f"raw {kind} observation {index} fails exact routing")
            valid = False
    prompt_count = corpus.get("prompt_count")
    if not _strict_positive_int(prompt_count) or len(prompt_ids) < int(prompt_count):
        label = f" at inflight {inflight}" if inflight is not None else ""
        errors.append(f"raw {kind} corpus{label} contains fewer than 20 accepted prompts")
        valid = False
    if not set(_REQUIRED_CONTEXT_TOKENS).issubset(contexts):
        label = f" at inflight {inflight}" if inflight is not None else ""
        errors.append(
            f"raw {kind} corpus{label} misses context 16, 128, 512, or 4096"
        )
        valid = False
    return bool(rows and valid)


def _derive_apple_role(
    observations: Sequence[Any],
    *,
    node: Mapping[str, Any],
    numerical_parity: bool,
    errors: list[str],
) -> dict[str, str] | None:
    rows = [
        row
        for row in observations
        if isinstance(row, dict) and row.get("kind") == "apple_role_criteria"
    ]
    if len(rows) != 1:
        errors.append("raw Apple evidence requires exactly one apple_role_criteria row")
        return None
    row = rows[0]
    allowed = {
        "kind",
        "operator_cases_total",
        "operator_cases_passed",
        "stage_cases_total",
        "stage_cases_passed",
        "expert_cases_total",
        "expert_cases_passed",
        "decode_context_tokens",
        "memory_high_water_bytes",
        "runtime_error_count",
    }
    if set(row) != allowed:
        errors.append("raw apple_role_criteria has an invalid schema")
        return None
    for field in (
        "operator_cases_total",
        "operator_cases_passed",
        "stage_cases_total",
        "stage_cases_passed",
        "expert_cases_total",
        "expert_cases_passed",
        "memory_high_water_bytes",
        "runtime_error_count",
    ):
        if not _strict_nonnegative_int(row.get(field)):
            errors.append(f"raw apple_role_criteria.{field} must be an integer")
            return None
    for prefix in ("operator", "stage", "expert"):
        if row[f"{prefix}_cases_passed"] > row[f"{prefix}_cases_total"]:
            errors.append(f"raw Apple {prefix} passed count exceeds total")
            return None
    contexts = row.get("decode_context_tokens")
    if not (
        isinstance(contexts, list)
        and all(_strict_positive_int(value) for value in contexts)
    ):
        errors.append("raw Apple decode_context_tokens must be positive integers")
        return None
    within_memory = row["memory_high_water_bytes"] <= node.get("memory_bytes", -1)
    operator_ok = bool(
        numerical_parity
        and row["runtime_error_count"] == 0
        and row["operator_cases_total"] > 0
        and row["operator_cases_passed"] == row["operator_cases_total"]
    )
    stage_ok = bool(
        operator_ok
        and row["stage_cases_total"] >= 20
        and row["stage_cases_passed"] == row["stage_cases_total"]
        and set(_REQUIRED_CONTEXT_TOKENS).issubset(contexts)
    )
    expert_ok = bool(
        operator_ok
        and row["expert_cases_total"] > 0
        and row["expert_cases_passed"] == row["expert_cases_total"]
    )
    if not operator_ok:
        role = "unsupported"
    elif not within_memory:
        role = "capacity-only"
    elif stage_ok and expert_ok:
        role = "pipeline-stage"
    elif stage_ok:
        role = "kv-decode-stage"
    elif expert_ok:
        role = "expert-worker"
    else:
        role = "capacity-only"
    return {
        "role": role,
        "evidence": "derived-from-raw",
        "criteria_id": "g2-apple-role-v1",
    }


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _derive_timings(
    observations: Sequence[Any], errors: list[str]
) -> dict[str, dict[str, float]] | None:
    required = (
        "stage_0",
        "pack",
        "transfer",
        "unpack",
        "stage_1",
        "exposed_wait",
        "end_to_end",
    )
    samples: dict[str, list[float]] = {name: [] for name in required}
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or row.get("kind") != "timing":
            continue
        component = row.get("component")
        if component not in samples:
            continue
        if set(row) != {"kind", "component", "milliseconds"}:
            errors.append(f"raw timing observation {index} has an invalid schema")
            continue
        value = row.get("milliseconds")
        if not _numeric(value):
            errors.append(f"raw timing observation {index} is not finite/non-negative")
            continue
        samples[str(component)].append(float(value))
    missing = [name for name, values in samples.items() if not values]
    if missing:
        errors.append("raw timing observations miss components: " + ", ".join(missing))
        return None
    return {
        name: {
            "median": _nearest_rank(values, 0.5),
            "p95": _nearest_rank(values, 0.95),
        }
        for name, values in samples.items()
    }


def _validate_derived_timings(
    result: Mapping[str, Any],
    observations: Sequence[Any],
    errors: list[str],
) -> None:
    _validate_timing_summary(result, errors)
    derived = _derive_timings(observations, errors)
    summary = result.get("timings_ms")
    if derived is not None and isinstance(summary, dict) and summary != derived:
        errors.append("timings_ms does not equal the summary derived from raw samples")


def _validate_raw_identity(
    raw: Mapping[str, Any],
    expected_nodes: Sequence[Mapping[str, Any] | None],
    run_manifest: Mapping[str, Any],
    errors: list[str],
) -> None:
    expected = [
        {field: node.get(field) for field in _NODE_IDENTITY_FIELDS}
        for node in expected_nodes
        if isinstance(node, Mapping)
    ]
    if raw.get("node_observations") != expected:
        errors.append("raw node observations do not match required manifest identities")
    if len(expected) > 1:
        network = raw.get("network_observation")
        configured = run_manifest.get("network")
        if not isinstance(network, dict) or not isinstance(configured, dict):
            errors.append("multi-node raw measurements require network_observation")
            return
        if set(network) != {
            "source_host_id",
            "destination_host_id",
            "route",
            "interface",
            "mtu_bytes",
            "declared_link_bits_per_second",
            "transfer_samples",
        }:
            errors.append("raw network_observation has an invalid schema")
        for field in (
            "source_host_id",
            "destination_host_id",
            "route",
            "interface",
            "mtu_bytes",
            "declared_link_bits_per_second",
        ):
            if network.get(field) != configured.get(field):
                errors.append(f"raw network_observation.{field} does not match manifest")
        if network.get("source_host_id") == network.get("destination_host_id"):
            errors.append("raw network observation uses the same physical host")
        samples = network.get("transfer_samples")
        if not (
            isinstance(samples, list)
            and samples
            and all(
                isinstance(row, dict)
                and set(row) == {"bytes", "duration_seconds"}
                and _strict_positive_int(row.get("bytes"))
                and _numeric(row.get("duration_seconds"), minimum=0.000000001)
                for row in samples
            )
        ):
            errors.append("raw network observation needs positive transfer samples")
    elif "network_observation" in raw:
        errors.append("single-node raw measurements must not contain network_observation")


def validate_physical_step_result(
    step_id: str,
    result: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    *,
    pinned_max_commit: str,
    artifact_base: Path | None = None,
    observed_process_elapsed_seconds: float | None = None,
    expected_nonce: str | None = None,
    bound_inputs: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    definition = _PHYSICAL_BY_ID[step_id]
    common_result_fields = {
        "schema_version",
        "step_id",
        "evidence_class",
        "measured",
        "physical",
        "same_host_proxy",
        "passed",
        "model",
        "plan",
        "max_patch_commit",
        "observed_nodes",
        "physical_host_ids",
        "raw_artifacts",
        "raw_measurements_artifact",
        "checks",
    }
    per_step_result_fields = {
        "V6_NVIDIA": {"numerical"},
        "V6_APPLE": {"numerical", "apple_role_decision"},
        "V7_PIPELINE": {"numerical", "timings_ms"},
        "V8_LOAD_CALIBRATION": {
            "concurrency",
            "planner_max_relative_error",
            "timings_ms",
        },
        "V9_STABILITY": {"duration_seconds", "bounds", "high_water"},
        "V10_FAILURES": {"faults"},
    }
    _reject_unknown_keys(
        result,
        common_result_fields | per_step_result_fields[step_id],
        f"{step_id} result",
        errors,
    )
    if result.get("schema_version") != G2_RUN_MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {G2_RUN_MANIFEST_SCHEMA_VERSION}")
    if result.get("step_id") != step_id:
        errors.append(f"step_id must equal {step_id}")
    if result.get("evidence_class") != definition["tier"]:
        errors.append(f"evidence_class must equal {definition['tier']}")
    if result.get("measured") is not True:
        errors.append("measured must be true")
    if result.get("physical") is not True:
        errors.append("physical must be true")
    if result.get("same_host_proxy") is not False:
        errors.append("same_host_proxy must be false")
    if result.get("passed") is not True:
        errors.append("passed must be true")
    if result.get("model") != run_manifest.get("model"):
        errors.append("model identity does not exactly match the run manifest")
    if result.get("plan") != run_manifest.get("plan"):
        errors.append("plan identity does not exactly match the run manifest")
    if result.get("max_patch_commit") != pinned_max_commit:
        errors.append("max_patch_commit does not match the root pin")
    raw_artifacts = result.get("raw_artifacts")
    if not (
        isinstance(raw_artifacts, list)
        and bool(raw_artifacts)
        and all(_safe_relative_artifact(value) for value in raw_artifacts)
    ):
        errors.append("raw_artifacts must contain safe relative artifact paths")

    configured_nodes = {
        node.get("role"): node
        for node in run_manifest.get("nodes", [])
        if isinstance(node, dict)
    }
    expected_nodes = [configured_nodes.get(role) for role in definition["roles"]]
    observed_nodes = result.get("observed_nodes")
    if not isinstance(observed_nodes, list):
        errors.append("observed_nodes must be a list")
    elif (
        any(node is None for node in expected_nodes) or observed_nodes != expected_nodes
    ):
        errors.append("observed_nodes do not exactly match the required physical roles")
    physical_host_ids = result.get("physical_host_ids")
    expected_host_ids = [
        node.get("physical_host_id")
        for node in expected_nodes
        if isinstance(node, dict)
    ]
    if physical_host_ids != expected_host_ids:
        errors.append("physical_host_ids do not exactly match the required nodes")
    if len(set(expected_host_ids)) != len(expected_host_ids):
        errors.append("physical_host_ids are not distinct")

    raw = _load_raw_measurements(
        result,
        artifact_base=artifact_base,
        expected_nonce=expected_nonce,
        errors=errors,
    )
    observations: list[Any] = []
    if raw is not None:
        _validate_raw_identity(raw, expected_nodes, run_manifest, errors)
        raw_observations = raw.get("observations")
        if isinstance(raw_observations, list):
            observations = raw_observations
        allowed_observation_kinds = {
            "V6_NVIDIA": {"numerical"},
            "V6_APPLE": {"numerical", "apple_role_criteria"},
            "V7_PIPELINE": {"numerical", "generation", "correctness", "timing"},
            "V8_LOAD_CALIBRATION": {"concurrency", "load_correctness", "timing"},
            "V9_STABILITY": {"stability", "post_drain"},
            "V10_FAILURES": {"fault"},
        }[step_id]
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict) or observation.get(
                "kind"
            ) not in allowed_observation_kinds:
                errors.append(f"raw observation {index} has an unknown schema kind")
        if step_id != "V9_STABILITY" and "limits" in raw:
            errors.append("raw limits are only valid for V9 stability evidence")

    checks = result.get("checks")
    if not isinstance(checks, dict):
        errors.append("checks must be an object")
        checks = {}
    correctness = run_manifest.get("correctness")
    if not isinstance(correctness, Mapping):
        errors.append("validated correctness policy is unavailable")
        correctness = {}
    corpus = correctness.get("corpus")
    reference = correctness.get("reference")
    corpus = corpus if isinstance(corpus, Mapping) else {}
    reference_id = reference.get("reference_id") if isinstance(reference, Mapping) else None

    if step_id in {"V6_NVIDIA", "V6_APPLE"}:
        numerical_summary, seen = _derive_numerical(
            observations, {"operator", "stage"}, correctness, errors
        )
        derived_parity = bool(
            numerical_summary is not None
            and numerical_summary["all_within_tolerance"]
            and numerical_summary["nonfinite_count"] == 0
            and numerical_summary["top1_mismatch_count"] == 0
            and numerical_summary["routing_mismatch_count"] == 0
            and seen == {"operator", "stage"}
        )
        if result.get("numerical") != numerical_summary:
            errors.append("numerical summary is not derived from the approved raw policy")
        for field in ("operator_stage_parity", "numerical_parity"):
            if checks.get(field) is not derived_parity or not derived_parity:
                errors.append(f"checks.{field} does not equal derived passing parity")
        if step_id == "V6_APPLE":
            apple_node = configured_nodes.get("apple")
            decision = _derive_apple_role(
                observations,
                node=apple_node if isinstance(apple_node, Mapping) else {},
                numerical_parity=derived_parity,
                errors=errors,
            )
            if result.get("apple_role_decision") != decision:
                errors.append("apple_role_decision is not derived from raw criteria")
    elif step_id == "V7_PIPELINE":
        numerical_summary, seen = _derive_numerical(
            observations, {"boundary", "final_logits"}, correctness, errors
        )
        parity = bool(
            numerical_summary is not None
            and numerical_summary["all_within_tolerance"]
            and numerical_summary["nonfinite_count"] == 0
            and numerical_summary["top1_mismatch_count"] == 0
            and numerical_summary["routing_mismatch_count"] == 0
            and seen == {"boundary", "final_logits"}
        )
        if result.get("numerical") != numerical_summary:
            errors.append("numerical summary is not derived from the approved raw policy")
        phases: set[str] = set()
        greedy_matches = True
        for index, row in enumerate(observations):
            if not isinstance(row, dict) or row.get("kind") != "generation":
                continue
            if set(row) != {
                "kind",
                "phase",
                "expected_token_ids",
                "observed_token_ids",
            }:
                errors.append(f"raw generation observation {index} has an invalid schema")
                continue
            phase = row.get("phase")
            if phase not in {"prefill", "decode"}:
                errors.append(f"raw generation observation {index} has invalid phase")
                continue
            expected_tokens = row.get("expected_token_ids")
            observed_tokens = row.get("observed_token_ids")
            if not (
                isinstance(expected_tokens, list)
                and expected_tokens
                and all(type(item) is int and item >= 0 for item in expected_tokens)
                and observed_tokens == expected_tokens
            ):
                greedy_matches = False
            phases.add(str(phase))
        corpus_passed = _derive_correctness_corpus(
            observations,
            kind="correctness",
            corpus=corpus,
            reference_id=reference_id,
            errors=errors,
        )
        derived_checks = {
            "prefill": "prefill" in phases,
            "decode": "decode" in phases,
            "greedy_output": bool(phases == {"prefill", "decode"} and greedy_matches),
            "boundary_parity": parity,
            "final_logits_parity": parity,
            "correctness_corpus": corpus_passed,
        }
        for field, derived in derived_checks.items():
            if checks.get(field) is not derived or not derived:
                errors.append(f"checks.{field} does not equal derived passing result")
        _validate_derived_timings(result, observations, errors)
    elif step_id == "V8_LOAD_CALIBRATION":
        derived_rows: list[dict[str, float | int]] = []
        relative_errors: list[float] = []
        frozen = (
            bound_inputs.get("frozen_predictions", {})
            if isinstance(bound_inputs, Mapping)
            else {}
        )
        seen_inflight: set[int] = set()
        for row in observations:
            if not isinstance(row, dict) or row.get("kind") != "concurrency":
                continue
            if set(row) != {
                "kind",
                "inflight",
                "aggregate_tokens_per_second",
                "predicted_tokens_per_second",
            }:
                errors.append("raw concurrency observation has an invalid schema")
                continue
            inflight = row.get("inflight")
            measured = row.get("aggregate_tokens_per_second")
            predicted = row.get("predicted_tokens_per_second")
            if not (
                type(inflight) is int
                and inflight > 0
                and _numeric(measured, minimum=0.000001)
                and _numeric(predicted, minimum=0.000001)
            ):
                errors.append("raw concurrency observations must be finite and positive")
                continue
            if inflight in seen_inflight:
                errors.append("raw concurrency observations contain duplicate inflight")
                continue
            seen_inflight.add(inflight)
            frozen_prediction = frozen.get(inflight) if isinstance(frozen, Mapping) else None
            if not _numeric(frozen_prediction, minimum=0.000001) or not math.isclose(
                float(predicted),
                float(frozen_prediction) if _numeric(frozen_prediction) else math.inf,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"raw concurrency prediction at inflight {inflight} is not the frozen plan input"
                )
            derived_rows.append(
                {
                    "inflight": inflight,
                    "aggregate_tokens_per_second": float(measured),
                    "predicted_tokens_per_second": float(predicted),
                }
            )
            relative_errors.append(abs(float(predicted) - float(measured)) / float(measured))
        derived_rows.sort(key=lambda row: int(row["inflight"]))
        points = {row["inflight"] for row in derived_rows}
        if points != set(_REQUIRED_CONCURRENCY):
            errors.append("raw concurrency observations must contain exactly 1, 4, and 8")
        if result.get("concurrency") != derived_rows:
            errors.append("concurrency summary is not derived from raw observations")
        derived_error = max(relative_errors, default=math.inf)
        reported_error = result.get("planner_max_relative_error")
        if (
            not _numeric(reported_error)
            or not math.isclose(float(reported_error), derived_error, rel_tol=1e-12, abs_tol=1e-12)
            or derived_error > 0.20
        ):
            errors.append("planner_max_relative_error is not a derived value within 0.20")
        throughput = {
            int(row["inflight"]): float(row["aggregate_tokens_per_second"])
            for row in derived_rows
        }
        scales = bool(
            {1, 4, 8}.issubset(throughput)
            and throughput[1] <= throughput[4] <= throughput[8]
            and throughput[8] > throughput[1]
        )
        if checks.get("throughput_scales") is not scales or not scales:
            errors.append("checks.throughput_scales is not derived from raw observations")
        correctness_results = [
            _derive_correctness_corpus(
                observations,
                kind="load_correctness",
                corpus=corpus,
                reference_id=reference_id,
                errors=errors,
                inflight=inflight,
            )
            for inflight in _REQUIRED_CONCURRENCY
        ]
        correctness_at_load = all(correctness_results)
        if (
            checks.get("correctness_at_1_4_8") is not correctness_at_load
            or not correctness_at_load
        ):
            errors.append("checks.correctness_at_1_4_8 is not derived from raw corpora")
        derived_timings = _derive_timings(observations, errors)
        timing_recorded = derived_timings is not None
        if checks.get("timing_attribution_recorded") is not timing_recorded or not timing_recorded:
            errors.append("checks.timing_attribution_recorded is not derived from raw observations")
        _validate_timing_summary(result, errors)
        if derived_timings is not None and result.get("timings_ms") != derived_timings:
            errors.append("timings_ms does not equal the raw timing summary")
    elif step_id == "V9_STABILITY":
        stability_policy = run_manifest.get("stability")
        stability_policy = (
            stability_policy if isinstance(stability_policy, Mapping) else {}
        )
        required_duration = stability_policy.get("duration_seconds")
        if (
            not _numeric(observed_process_elapsed_seconds)
            or not _strict_positive_int(required_duration)
            or float(observed_process_elapsed_seconds) < required_duration
        ):
            errors.append(
                "runner-observed V9 process elapsed time is shorter than the approved duration"
            )
        if not _strict_positive_int(result.get("duration_seconds")):
            errors.append("duration_seconds must be a strict positive integer")
        elif result.get("duration_seconds") != required_duration:
            errors.append("duration_seconds must equal the approved stability duration")
        elif _numeric(observed_process_elapsed_seconds) and result[
            "duration_seconds"
        ] > float(observed_process_elapsed_seconds) + 1.0:
            errors.append("duration_seconds exceeds runner-observed process elapsed time")
        stability_rows = [
            row
            for row in observations
            if isinstance(row, dict) and row.get("kind") == "stability"
        ]
        exact_stability_fields = {
            "kind",
            "sample_index",
            "elapsed_seconds",
            "phase",
            "target_inflight",
            "active_inflight",
            "requests_started",
            "requests_completed",
            "requests_failed",
            "explicit_releases",
            "expiry_releases",
            "live_requests",
            "queue_depth",
            "credit_bytes_inflight",
            "retained_state_bytes",
            "native_buffer_bytes",
            "kv_bytes",
            "rss_bytes",
            "divergence_count",
        }
        integer_fields = exact_stability_fields - {"kind", "phase"}
        rows_valid = bool(stability_rows)
        for index, row in enumerate(stability_rows):
            if set(row) != exact_stability_fields:
                errors.append(f"raw stability sample {index} has an invalid schema")
                rows_valid = False
                continue
            if row.get("phase") != "load":
                errors.append(f"raw stability sample {index} phase must be load")
                rows_valid = False
            for field in integer_fields:
                if not _strict_nonnegative_int(row.get(field)):
                    errors.append(
                        f"raw stability sample {index}.{field} must be an integer"
                    )
                    rows_valid = False
        sample_indices = [row.get("sample_index") for row in stability_rows]
        elapsed_values = [row.get("elapsed_seconds") for row in stability_rows]
        cadence = stability_policy.get("sample_interval_seconds")
        cadence_ok = bool(
            rows_valid
            and _strict_positive_int(cadence)
            and _strict_positive_int(required_duration)
            and sample_indices == list(range(len(stability_rows)))
            and elapsed_values
            and elapsed_values[0] == 0
            and all(
                0 < right - left <= cadence
                for left, right in zip(elapsed_values, elapsed_values[1:])
            )
            and elapsed_values[-1] - elapsed_values[0] == required_duration
        )
        if not cadence_ok:
            errors.append("raw stability samples violate sequence/span/cadence")
        target_inflight = stability_policy.get("target_inflight")
        load_held = bool(
            rows_valid
            and all(
                row.get("target_inflight") == target_inflight
                and row.get("active_inflight") == target_inflight
                for row in stability_rows
            )
        )
        if not load_held:
            errors.append("raw stability samples do not hold the approved load")

        counter_fields = (
            "requests_started",
            "requests_completed",
            "requests_failed",
            "explicit_releases",
            "expiry_releases",
        )
        monotonic = all(
            all(
                right.get(field, -1) >= left.get(field, 0)
                for field in counter_fields
            )
            for left, right in zip(stability_rows, stability_rows[1:])
        )
        request_accounting = bool(
            rows_valid
            and monotonic
            and all(
                row["requests_started"]
                == row["requests_completed"]
                + row["requests_failed"]
                + row["live_requests"]
                for row in stability_rows
            )
            and all(row["requests_failed"] == 0 for row in stability_rows)
            and stability_rows[-1]["requests_completed"]
            >= stability_policy.get("minimum_completed_requests", math.inf)
        )
        if not request_accounting:
            errors.append("raw stability request counters do not reconcile")
        lifecycle_accounting = bool(
            rows_valid
            and all(
                row["explicit_releases"] + row["expiry_releases"]
                == row["requests_completed"]
                for row in stability_rows
            )
        )
        if not lifecycle_accounting:
            errors.append("raw stability lifecycle counters do not reconcile")

        post_drain_rows = [
            row
            for row in observations
            if isinstance(row, dict) and row.get("kind") == "post_drain"
        ]
        post_drain_fields = {
            "kind",
            "elapsed_seconds",
            "drain_wait_seconds",
            "requests_started",
            "requests_completed",
            "requests_failed",
            "explicit_releases",
            "expiry_releases",
            "live_requests",
            "queue_depth",
            "credit_bytes_inflight",
            "retained_state_bytes",
            "native_buffer_bytes",
            "kv_bytes",
            "inflight",
        }
        post_drain = post_drain_rows[0] if len(post_drain_rows) == 1 else None
        post_drain_cleanup = bool(
            post_drain is not None
            and bool(elapsed_values)
            and set(post_drain) == post_drain_fields
            and all(
                _strict_nonnegative_int(post_drain.get(field))
                for field in post_drain_fields - {"kind"}
            )
            and post_drain["elapsed_seconds"] >= elapsed_values[-1]
            and post_drain["elapsed_seconds"]
            == elapsed_values[-1] + post_drain["drain_wait_seconds"]
            and (
                not _numeric(observed_process_elapsed_seconds)
                or post_drain["elapsed_seconds"]
                <= float(observed_process_elapsed_seconds) + 1.0
            )
            and post_drain["drain_wait_seconds"]
            <= stability_policy.get("post_drain_timeout_seconds", -1)
            and all(
                post_drain[field] == 0
                for field in (
                    "requests_failed",
                    "live_requests",
                    "queue_depth",
                    "credit_bytes_inflight",
                    "retained_state_bytes",
                    "native_buffer_bytes",
                    "kv_bytes",
                    "inflight",
                )
            )
            and post_drain["requests_started"] == post_drain["requests_completed"]
            and post_drain["requests_started"]
            == stability_rows[-1]["requests_started"]
            and post_drain["requests_completed"]
            == stability_rows[-1]["requests_completed"]
            + stability_rows[-1]["live_requests"]
            and post_drain["explicit_releases"] + post_drain["expiry_releases"]
            == post_drain["requests_completed"]
        )
        if not post_drain_cleanup:
            errors.append("raw post-drain evidence is absent, non-integer, or not clean")
        bounds = result.get("bounds")
        limits = raw.get("limits") if isinstance(raw, dict) else None
        high_water = result.get("high_water")
        fields = {
            "queue": "queue_depth",
            "credit": "credit_bytes_inflight",
            "retained_state": "retained_state_bytes",
            "native_buffer": "native_buffer_bytes",
            "kv": "kv_bytes",
            "rss": "rss_bytes",
        }
        if not isinstance(bounds, dict) or set(bounds) != set(fields):
            errors.append("bounds must contain exactly the six configured resources")
        if not isinstance(limits, dict) or set(limits) != set(fields.values()):
            errors.append("raw limits must contain exactly the six resource limits")
        derived_high_water: dict[str, float] = {}
        for bound_name, field in fields.items():
            values = [
                row[field]
                for row in stability_rows
                if _strict_nonnegative_int(row.get(field))
            ]
            if len(values) != len(stability_rows) or not values:
                errors.append(f"raw stability samples require {field}")
                continue
            derived_high_water[field] = max(values)
            limit = limits.get(field) if isinstance(limits, dict) else None
            derived_bound = bool(
                _strict_positive_int(limit) and max(values) <= int(limit)
            )
            if not isinstance(bounds, dict) or bounds.get(bound_name) is not derived_bound or not derived_bound:
                errors.append(f"bounds.{bound_name} is not derived from raw samples and limit")
        if high_water != derived_high_water:
            errors.append("high_water does not equal values derived from raw stability samples")
        no_divergence = bool(
            stability_rows
            and all(row.get("divergence_count") == 0 for row in stability_rows)
        )
        derived_checks = {
            "sampling_cadence": cadence_ok,
            "load_held": load_held,
            "request_accounting": request_accounting,
            "lifecycle_accounting": lifecycle_accounting,
            "post_drain_cleanup": post_drain_cleanup,
            "no_silent_divergence": no_divergence,
        }
        if checks != derived_checks or not all(derived_checks.values()):
            errors.append("V9 checks are not a complete passing raw derivation")
    elif step_id == "V10_FAILURES":
        faults = result.get("faults")
        derived_faults: dict[str, dict[str, Any]] = {}
        for row in observations:
            if not isinstance(row, dict) or row.get("kind") != "fault":
                continue
            name = row.get("name")
            if name not in _CANONICAL_FAULTS:
                continue
            if name in derived_faults:
                errors.append(f"raw fault observations contain duplicate {name}")
                continue
            policy = _CANONICAL_FAULTS[str(name)]
            expected_fault_fields = {
                "kind",
                "name",
                "expected_outcome_code",
                "observed_outcome_code",
                "state_before_sha256",
                "state_after_sha256",
                "mutation_count",
                "replay_attempt_count",
                "replay_execution_count",
                "replay_disposition",
                "cleanup",
            }
            if set(row) != expected_fault_fields:
                errors.append(f"raw fault observation {name} has an invalid schema")
            cleanup = row.get("cleanup")
            cleanup_passed = bool(
                isinstance(cleanup, dict)
                and set(cleanup)
                == {
                    "live_requests",
                    "kv_bytes",
                    "retained_state_bytes",
                    "native_buffer_bytes",
                    "inflight",
                    "queue_depth",
                    "credit_bytes_inflight",
                }
                and all(type(cleanup.get(field)) is int and cleanup[field] == 0 for field in cleanup)
            )
            before = row.get("state_before_sha256")
            after = row.get("state_after_sha256")
            expected_mutations = policy["mutation_count"]
            mutation_passed = bool(
                _is_sha256(before)
                and _is_sha256(after)
                and type(row.get("mutation_count")) is int
                and row.get("mutation_count") == expected_mutations
                and ((before == after) if expected_mutations == 0 else (before != after))
            )
            replay_passed = bool(
                type(row.get("replay_attempt_count")) is int
                and row.get("replay_attempt_count") == 1
                and type(row.get("replay_execution_count")) is int
                and row.get("replay_execution_count") == 0
                and row.get("replay_disposition") == policy["replay_disposition"]
            )
            outcome_passed = bool(
                row.get("expected_outcome_code") == policy["outcome_code"]
                and row.get("observed_outcome_code") == policy["outcome_code"]
            )
            derived_faults[str(name)] = {
                "outcome_code": policy["outcome_code"],
                "passed": outcome_passed,
                "cleanup_passed": cleanup_passed,
                "replay_passed": replay_passed,
                "mutation_passed": mutation_passed,
            }
        required_faults = set(_CANONICAL_FAULTS)
        if set(derived_faults) != required_faults:
            errors.append("raw fault observations must cover the complete failure matrix")
        if faults != derived_faults or not all(
            row["passed"]
            and row["cleanup_passed"]
            and row["replay_passed"]
            and row["mutation_passed"]
            for row in derived_faults.values()
        ):
            errors.append("faults summary is not a complete passing derivation from raw samples")
    return errors


def _expand_token(value: str, *, repo_root: Path, step_dir: Path) -> str:
    return value.replace("{repo_root}", str(repo_root)).replace(
        "{step_dir}", str(step_dir)
    )


def _hash_directory(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    ):
        rows.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _run_prerequisites(
    repo_root: Path,
    logs_dir: Path,
    commands: Iterable[Mapping[str, Any]],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in commands:
        step_id = str(command["step_id"])
        process = _run_process(
            command["argv"], cwd=repo_root, timeout_seconds=timeout_seconds
        )
        stdout_path = logs_dir / f"{step_id}.stdout.log"
        stderr_path = logs_dir / f"{step_id}.stderr.log"
        stdout_path.write_text(process.pop("stdout"), encoding="utf-8")
        stderr_path.write_text(process.pop("stderr"), encoding="utf-8")
        passed = process["returncode"] == 0 and not process["timed_out"]
        rows.append(
            {
                "step_id": step_id,
                "tier": command["tier"],
                "title": command["title"],
                "status": "PASSED" if passed else "FAILED",
                "process": process,
                "stdout": {
                    "path": stdout_path.name,
                    "sha256": _sha256_file(stdout_path),
                },
                "stderr": {
                    "path": stderr_path.name,
                    "sha256": _sha256_file(stderr_path),
                },
            }
        )
    return rows


def _run_physical_steps(
    *,
    repo_root: Path,
    bundle_root: Path,
    run_manifest: Mapping[str, Any] | None,
    run_validation: Mapping[str, Any] | None,
    pinned_max_commit: str,
    run_physical: bool,
    prerequisites_passed: bool,
    lineage_passed: bool,
    source_clean: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_by_id: dict[str, str] = {}
    step_specs = run_validation.get("normalised_steps", {}) if run_validation else {}
    manifest_ok = bool(run_validation and run_validation.get("ok"))
    for definition in PHYSICAL_STEPS:
        step_id = definition["step_id"]
        row: dict[str, Any] = {
            "step_id": step_id,
            "tier": definition["tier"],
            "title": definition["title"],
            "dependencies": list(definition["dependencies"]),
            "status": "BLOCKED",
            "reason": None,
            "process": None,
            "result_validation_errors": [],
            "artifacts": [],
        }
        spec = step_specs.get(
            step_id, {"status": "BLOCKED", "reason": "run manifest is absent"}
        )
        if run_manifest is None:
            row["reason"] = "no physical run manifest was supplied"
        elif not manifest_ok:
            row["reason"] = "physical run manifest is invalid"
        elif spec.get("status") == "BLOCKED":
            row["reason"] = str(spec.get("reason", "physical resource unavailable"))
        elif not run_physical:
            row["status"] = "NOT_RUN"
            row["reason"] = (
                "physical execution requires the explicit --run-physical flag"
            )
        elif not lineage_passed:
            row["reason"] = "root MAX lineage verification failed"
        elif not source_clean:
            row["reason"] = "Fornax execution source has uncommitted changes"
        elif not prerequisites_passed:
            row["reason"] = "one or more T0/T1 prerequisites failed"
        else:
            failed_dependencies = [
                dependency
                for dependency in definition["dependencies"]
                if status_by_id.get(dependency) != "PASSED"
            ]
            if failed_dependencies:
                row["reason"] = "failed or unrun dependencies: " + ", ".join(
                    failed_dependencies
                )
            else:
                step_dir = bundle_root / "physical" / step_id
                step_dir.mkdir(parents=True, exist_ok=False)
                argv = [
                    _expand_token(value, repo_root=repo_root, step_dir=step_dir)
                    for value in spec["argv"]
                ]
                cwd = Path(
                    _expand_token(
                        spec.get("cwd", "{repo_root}"),
                        repo_root=repo_root,
                        step_dir=step_dir,
                    )
                )
                if not cwd.is_absolute():
                    cwd = (repo_root / cwd).resolve()
                environment = dict(os.environ)
                runner_nonce = str(uuid.uuid4())
                environment.update(
                    {
                        "FORNAX_G2_BUNDLE_DIR": str(bundle_root),
                        "FORNAX_G2_STEP_DIR": str(step_dir),
                        "FORNAX_G2_STEP_ID": step_id,
                        "FORNAX_G2_RUN_NONCE": runner_nonce,
                    }
                )
                process = _run_process(
                    argv,
                    cwd=cwd,
                    timeout_seconds=float(spec["timeout_seconds"]),
                    environment=environment,
                )
                stdout_path = bundle_root / "logs" / f"{step_id}.stdout.log"
                stderr_path = bundle_root / "logs" / f"{step_id}.stderr.log"
                stdout_path.write_text(process.pop("stdout"), encoding="utf-8")
                stderr_path.write_text(process.pop("stderr"), encoding="utf-8")
                row["process"] = process
                row["stdout"] = {
                    "path": f"logs/{stdout_path.name}",
                    "sha256": _sha256_file(stdout_path),
                }
                row["stderr"] = {
                    "path": f"logs/{stderr_path.name}",
                    "sha256": _sha256_file(stderr_path),
                }
                result_path = step_dir / spec["result_artifact"]
                result_errors: list[str] = []
                if process["returncode"] != 0 or process["timed_out"]:
                    result_errors.append("physical command did not exit successfully")
                if not _contained_regular_file(step_dir, spec["result_artifact"]):
                    result_errors.append(
                        f"required result artifact is missing: {spec['result_artifact']}"
                    )
                else:
                    try:
                        result_data = _read_json_object(result_path)
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        result_errors.append(f"invalid result artifact: {exc}")
                    else:
                        result_errors.extend(
                            validate_physical_step_result(
                                step_id,
                                result_data,
                                run_manifest,
                                pinned_max_commit=pinned_max_commit,
                                artifact_base=step_dir,
                                observed_process_elapsed_seconds=process.get(
                                    "elapsed_seconds"
                                ),
                                expected_nonce=runner_nonce,
                                bound_inputs=(
                                    run_validation.get("bound_inputs")
                                    if run_validation
                                    else None
                                ),
                            )
                        )
                        raw_artifacts = result_data.get("raw_artifacts", [])
                        if isinstance(raw_artifacts, list):
                            for raw_path in raw_artifacts:
                                if _safe_relative_artifact(
                                    raw_path
                                ) and not _contained_regular_file(step_dir, raw_path):
                                    result_errors.append(
                                        f"declared raw artifact is missing: {raw_path}"
                                    )
                row["result_validation_errors"] = result_errors
                row["artifacts"] = _hash_directory(step_dir)
                if result_errors:
                    row["status"] = "FAILED"
                    row["reason"] = "; ".join(result_errors)
                else:
                    row["status"] = "PASSED"
                    row["reason"] = None
        status_by_id[step_id] = row["status"]
        rows.append(row)
    return rows


def _render_summary(evidence: Mapping[str, Any]) -> str:
    summary = evidence["summary"]
    lines = [
        "# Fornax G2 readiness and validation summary",
        "",
        f"Run: `{evidence['run_id']}`  ",
        f"Generated: `{evidence['generated_at']}`  ",
        f"Status: **{summary['status']}**  ",
        f"Technical G2 packet passed: **{str(summary['technical_g2_packet_passed']).lower()}**  ",
        "Formal gate decision authority: **false** (Sponsor/TL review remains required)",
        "",
        "## Readiness",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Root MAX lineage | {'PASS' if summary['max_lineage_passed'] else 'FAIL'} |",
        f"| Fornax execution source clean | {'PASS' if summary['fornax_execution_source_clean'] else 'FAIL'} |",
        f"| T0/T1 prerequisites | {'PASS' if summary['t0_t1_prerequisites_passed'] else 'FAIL'} |",
        f"| Physical V6-V10 | {'PASS' if summary['physical_steps_passed'] else 'OPEN'} |",
        "",
        "## T0/T1 commands",
        "",
        "| Step | Tier | Status | Command |",
        "|---|---|---|---|",
    ]
    for row in evidence["prerequisites"]:
        lines.append(
            f"| {row['step_id']} | {row['tier']} | {row['status']} | `{row['process']['command']}` |"
        )
    lines.extend(
        [
            "",
            "## Physical validation",
            "",
            "| Step | Tier | Status | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row in evidence["physical_steps"]:
        reason = str(row.get("reason") or "validated").replace("|", "\\|")
        lines.append(
            f"| {row['step_id']} | {row['tier']} | {row['status']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Identity",
            "",
            f"- Fornax commit: `{evidence['fornax_source'].get('head_commit')}`",
            f"- MAX pin manifest: `{evidence['max_lineage'].get('manifest_sha256')}`",
        ]
    )
    run_manifest = evidence.get("run_manifest")
    if run_manifest and run_manifest.get("data"):
        model = run_manifest["data"].get("model", {})
        lines.append(
            f"- Model: `{model.get('model_id')}` snapshot `{model.get('snapshot_id')}`"
        )
        for node in run_manifest["data"].get("nodes", []):
            lines.append(
                f"- {node.get('role')}: `{node.get('physical_host_id')}` / `{node.get('device_identity')}` / `{node.get('max_cli_version')}`"
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This runner validates a technical evidence packet. It never converts T0/T1, a same-host proxy, a zero exit code, or a hand-written result into physical G2 evidence. Formal gate closure requires Sponsor/TL review of this durable bundle.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_bundle_manifest(bundle_root: Path) -> dict[str, Any]:
    rows = [
        row
        for row in _hash_directory(bundle_root)
        if row["path"] not in {"artifact-manifest.json", "artifact-manifest.sha256"}
    ]
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "files": rows,
    }
    manifest_path = bundle_root / "artifact-manifest.json"
    _write_json(manifest_path, manifest)
    (bundle_root / "artifact-manifest.sha256").write_text(
        f"{_sha256_file(manifest_path)[7:]}  artifact-manifest.json\n",
        encoding="utf-8",
    )
    return manifest


def run_g2_validation(
    *,
    repo_root: str | Path,
    out_dir: str | Path,
    max_lineage_manifest: str | Path = "dependencies/max-lineage.json",
    run_manifest_path: str | Path | None = None,
    run_physical: bool = False,
    prerequisite_timeout_seconds: float = 300.0,
    prerequisite_commands: Iterable[Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repo root does not exist: {root}")
    bundle_root = Path(out_dir).resolve()
    if bundle_root.exists():
        raise ValueError(f"output directory already exists: {bundle_root}")
    if prerequisite_timeout_seconds <= 0:
        raise ValueError("prerequisite timeout must be positive")

    source = capture_fornax_source(root)
    lineage = verify_max_lineage(root, max_lineage_manifest)
    bundle_root.mkdir(parents=True, exist_ok=False)
    (bundle_root / "inputs").mkdir()
    (bundle_root / "logs").mkdir()
    if lineage.get("manifest_sha256"):
        shutil.copyfile(
            lineage["manifest_path"], bundle_root / "inputs" / "max-lineage.json"
        )

    run_manifest: dict[str, Any] | None = None
    run_validation: dict[str, Any] | None = None
    run_manifest_record: dict[str, Any] | None = None
    pinned_commit = str(
        lineage.get("pinned", {}).get("lineage", {}).get("patch_commit", "")
    )
    if run_manifest_path is not None:
        supplied = Path(run_manifest_path)
        path = supplied if supplied.is_absolute() else root / supplied
        if not path.is_file():
            raise ValueError(f"physical run manifest does not exist: {path}")
        run_manifest = _read_json_object(path)
        run_validation = validate_g2_run_manifest(
            run_manifest,
            pinned_max_commit=pinned_commit,
            manifest_base=path.resolve().parent,
        )
        copy_path = bundle_root / "inputs" / "g2-run-manifest.json"
        shutil.copyfile(path, copy_path)
        bound_artifact_records: list[dict[str, Any]] = []
        plan_section = run_manifest.get("plan")
        if isinstance(plan_section, dict):
            artifacts: list[tuple[str, str]] = []
            plan_artifact = plan_section.get("plan_artifact")
            if isinstance(plan_artifact, str):
                artifacts.append((plan_artifact, "physical-plan.json"))
            registry_artifact = plan_section.get("evidence_registry_artifact")
            if isinstance(registry_artifact, str):
                artifacts.append(
                    (registry_artifact, "planner-evidence-registry.json")
                )
            stage_artifacts = plan_section.get("stage_manifest_artifacts")
            if isinstance(stage_artifacts, list):
                artifacts.extend(
                    (relative, f"stage-manifest-{index}.json")
                    for index, relative in enumerate(stage_artifacts)
                    if isinstance(relative, str)
                )
            evidence_artifacts = run_validation.get("bound_inputs", {}).get(
                "evidence_artifacts", []
            )
            if isinstance(evidence_artifacts, list):
                artifacts.extend(
                    (relative, f"planner-evidence-{index}.json")
                    for index, relative in enumerate(evidence_artifacts)
                    if isinstance(relative, str)
                )
            for relative, bundle_name in artifacts:
                if _safe_relative_artifact(relative) and _contained_regular_file(
                    path.resolve().parent, relative
                ):
                    destination = bundle_root / "inputs" / bundle_name
                    shutil.copyfile(path.resolve().parent / relative, destination)
                    bound_artifact_records.append(
                        {
                            "source_path": str((path.resolve().parent / relative).resolve()),
                            "bundle_path": f"inputs/{bundle_name}",
                            "sha256": _sha256_file(destination),
                        }
                    )
        run_manifest_record = {
            "source_path": str(path.resolve()),
            "bundle_path": "inputs/g2-run-manifest.json",
            "sha256": _sha256_file(copy_path),
            "data": run_manifest,
            "validation": run_validation,
            "bound_artifacts": bound_artifact_records,
        }

    commands = list(
        prerequisite_commands
        if prerequisite_commands is not None
        else _default_prerequisites(sys.executable)
    )
    prerequisites = _run_prerequisites(
        root,
        bundle_root / "logs",
        commands,
        timeout_seconds=prerequisite_timeout_seconds,
    )
    prerequisites_passed = bool(prerequisites) and all(
        row["status"] == "PASSED" for row in prerequisites
    )
    physical_steps = _run_physical_steps(
        repo_root=root,
        bundle_root=bundle_root,
        run_manifest=run_manifest,
        run_validation=run_validation,
        pinned_max_commit=pinned_commit,
        run_physical=run_physical,
        prerequisites_passed=prerequisites_passed,
        lineage_passed=bool(lineage.get("ok")),
        source_clean=bool(source.get("execution_source_clean")),
    )
    post_run_source = capture_fornax_source(root)
    post_run_lineage = verify_max_lineage(root, max_lineage_manifest)
    source_unchanged = all(
        source.get(field) == post_run_source.get(field)
        for field in (
            "head_commit",
            "head_tree",
            "execution_relevant_status",
            "tracked_diff_sha256",
        )
    )
    lineage_unchanged = bool(
        post_run_lineage.get("ok")
        and lineage.get("manifest_sha256") == post_run_lineage.get("manifest_sha256")
        and lineage.get("pinned", {}).get("lineage", {}).get("patch_commit")
        == post_run_lineage.get("pinned", {}).get("lineage", {}).get("patch_commit")
    )
    physical_passed = all(row["status"] == "PASSED" for row in physical_steps)
    any_failed = bool(
        any(row["status"] == "FAILED" for row in prerequisites + physical_steps)
        or not lineage.get("ok")
        or not lineage_unchanged
        or not source.get("git_available")
        or not post_run_source.get("git_available")
        or not source_unchanged
        or (run_validation is not None and not run_validation.get("ok"))
    )
    technical_passed = bool(
        lineage.get("ok")
        and lineage_unchanged
        and source.get("execution_source_clean")
        and post_run_source.get("execution_source_clean")
        and source_unchanged
        and prerequisites_passed
        and physical_passed
    )
    if technical_passed:
        status = "PASS_PENDING_GATE_REVIEW"
    elif any_failed:
        status = "FAILED"
    else:
        status = "BLOCKED"
    generated_at = _now_iso(now)
    evidence: dict[str, Any] = {
        "schema_version": G2_EVIDENCE_SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "generated_at": generated_at,
        "runner": {
            "name": "fornax-g2-validation",
            "invocation_argv": list(sys.argv),
            "invocation_command": shlex.join(sys.argv),
            "invocation_working_directory": os.getcwd(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "run_physical_authorized": run_physical,
            "prerequisite_timeout_seconds": prerequisite_timeout_seconds,
        },
        "fornax_source": source,
        "max_lineage": lineage,
        "post_run_source_verification": {
            "unchanged": source_unchanged,
            "source": post_run_source,
        },
        "post_run_max_lineage_verification": {
            "unchanged": lineage_unchanged,
            "lineage": post_run_lineage,
        },
        "run_manifest": run_manifest_record,
        "prerequisites": prerequisites,
        "physical_steps": physical_steps,
        "summary": {
            "status": status,
            "max_lineage_passed": bool(lineage.get("ok") and lineage_unchanged),
            "fornax_execution_source_clean": bool(
                source.get("execution_source_clean")
                and post_run_source.get("execution_source_clean")
                and source_unchanged
            ),
            "t0_t1_prerequisites_passed": prerequisites_passed,
            "physical_steps_passed": physical_passed,
            "technical_g2_packet_passed": technical_passed,
            "formal_gate_decision": "PENDING_SPONSOR_TL_REVIEW"
            if technical_passed
            else "NOT_READY",
            "gate_decision_authority": False,
        },
        "claim_boundary": {
            "simulation_closes_physical_gate": False,
            "same_host_proxy_closes_physical_gate": False,
            "command_exit_zero_alone_is_evidence": False,
            "formal_g2_closed": False,
        },
        "bundle": {
            "root": str(bundle_root),
            "machine_evidence": "g2-evidence.json",
            "human_summary": "g2-summary.md",
            "artifact_manifest": "artifact-manifest.json",
            "artifact_manifest_hash": "artifact-manifest.sha256",
        },
    }
    evidence_path = bundle_root / "g2-evidence.json"
    summary_path = bundle_root / "g2-summary.md"
    _write_json(evidence_path, evidence)
    summary_path.write_text(_render_summary(evidence), encoding="utf-8")
    _write_bundle_manifest(bundle_root)
    return evidence


__all__ = [
    "G2_EVIDENCE_SCHEMA_VERSION",
    "G2_RUN_MANIFEST_SCHEMA_VERSION",
    "MAX_LINEAGE_SCHEMA_VERSION",
    "PHYSICAL_STEPS",
    "capture_fornax_source",
    "run_g2_validation",
    "validate_g2_run_manifest",
    "validate_physical_step_result",
    "verify_max_lineage",
]
