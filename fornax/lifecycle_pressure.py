"""Hardware-free unique-request lifecycle pressure evidence.

This is a T0/T1 contract stressor, not a physical memory or performance
benchmark.  Lifecycle deadlines use a deterministic logical clock so automatic
expiry is covered without sleeps; wall duration is measured independently for a
runnable sustained mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .stage_runtime import (
    ReferenceStageBackend,
    StageManifest,
    StageRequest,
    Tensor,
)


class _LogicalClock:
    def __init__(self, now_ns: int = 1_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, delta_ns: int) -> None:
        self.now_ns += delta_ns


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "fornax/lifecycle_pressure.py",
        "fornax/stage_runtime.py",
    )
    files = {
        relative: _sha256_file(root / relative) for relative in relative_paths
    }
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--",
                *relative_paths,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        git_head = head.stdout.strip() if head.returncode == 0 else None
        source_status = status.stdout.splitlines() if status.returncode == 0 else []
        git_available = head.returncode == 0 and status.returncode == 0
    except OSError:
        git_head = None
        source_status = []
        git_available = False
    return {
        "repository_root": str(root),
        "git_available": git_available,
        "git_head": git_head,
        "source_status": source_status,
        "source_committed": bool(git_available and not source_status),
        "files": files,
    }


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _manifest() -> StageManifest:
    digest = "sha256:" + "1" * 64
    return StageManifest(
        manifest_version=1,
        model_id="fornax/lifecycle-pressure-reference",
        model_snapshot="t0-logical-fixture-v1",
        model_config_hash=digest,
        tokenizer_hash="sha256:" + "2" * 64,
        template_hash="sha256:" + "3" * 64,
        max_build_id="fornax-reference-python-v1",
        fornax_abi_major=1,
        fornax_abi_minor=0,
        plan_id="51515151-5151-4151-8151-515151515151",
        plan_hash="sha256:" + "4" * 64,
        stage_id="lifecycle-pressure-stage-0",
        stage_index=0,
        layer_start=0,
        layer_end=1,
        input_contract={
            "kind": "activation",
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": 4,
        },
        output_contract={
            "kind": "activation",
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "hidden_size": 4,
        },
        kv_policy="stage_local",
    )


def _tensor(iteration: int) -> Tensor:
    offset = (iteration % 17) / 128.0
    return Tensor.from_values(
        [
            -0.5 + offset,
            -0.25 + offset,
            0.25 + offset,
            0.5 + offset,
            0.75 - offset,
            -0.75 + offset,
            1.0 - offset,
            -1.0 + offset,
        ],
        kind="activation",
        dtype="bf16",
        shape=(2, 4),
    )


def _request(
    manifest: StageManifest,
    tensor: Tensor,
    *,
    request_id: str,
    deadline_ns: int,
) -> StageRequest:
    return StageRequest(
        plan_id=manifest.plan_id,
        plan_hash=manifest.plan_hash,
        request_id=request_id,
        microbatch_id="lifecycle-pressure",
        sequence_no=0,
        phase="prefill",
        token_start=0,
        token_count=2,
        input_activation=tensor,
        kv_epoch=0,
        deadline_ns=deadline_ns,
        trace_context={
            "trace_id": f"trace-{request_id}",
            "span_id": "lifecycle-pressure-stage",
        },
    )


def run_lifecycle_pressure(
    *,
    min_iterations: int = 1_000,
    wall_seconds: float = 0.0,
    defer_every: int = 7,
    pace_ms: float = 0.0,
    max_pause_seconds: float = 5.0,
    track_allocations: bool = True,
) -> dict[str, Any]:
    """Run bounded unique-request churn and return an evidence dictionary."""

    integer_options = {
        "min_iterations": min_iterations,
        "defer_every": defer_every,
    }
    for name, value in integer_options.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value, allow_zero in (
        ("wall_seconds", wall_seconds, True),
        ("pace_ms", pace_ms, True),
        ("max_pause_seconds", max_pause_seconds, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            or (not allow_zero and float(value) == 0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a finite {qualifier} number")

    logical_step_ns = 1_000_000
    request_idle_timeout_ns = 3 * logical_step_ns
    execution_lease_timeout_ns = 10 * logical_step_ns
    tombstone_retention_ns = 64 * logical_step_ns
    # Sustained churn must provision enough fences for the entire retention
    # window.  The backend now fails closed instead of evicting a live fence.
    max_release_tombstones = 128
    clock = _LogicalClock()
    manifest = _manifest()
    backend = ReferenceStageBackend(
        clock_ns=clock,
        max_live_requests=8,
        max_completed_results_per_request=2,
        max_transform_cache_entries=8,
        max_completed_result_bytes=8 * 1024,
        max_transform_cache_bytes=8 * 1024,
        max_release_tombstones=max_release_tombstones,
        request_idle_timeout_ns=request_idle_timeout_ns,
        execution_lease_timeout_ns=execution_lease_timeout_ns,
        release_tombstone_ttl_ns=tombstone_retention_ns,
        max_native_buffer_bytes=4 * 1024,
    )
    handle = backend.load(manifest)

    source_before = _source_identity()
    started_at = datetime.now(timezone.utc).isoformat()

    tracing_was_active = tracemalloc.is_tracing()
    if track_allocations and not tracing_was_active:
        tracemalloc.start()
    allocation_before = tracemalloc.get_traced_memory()[0] if track_allocations else 0
    rss_before = _rss_bytes()
    wall_started_ns = time.monotonic_ns()
    calendar_started_ns = time.time_ns()
    previous_progress_ns = calendar_started_ns
    max_inter_iteration_gap_ns = 0
    continuity_limit_ns = int(max_pause_seconds * 1_000_000_000)
    iteration = 0
    explicit_releases = 0
    deferred_for_expiry = 0
    errors: list[str] = []
    maxima = {
        "live_requests": 0,
        "release_tombstones": 0,
        "completed_result_bytes": 0,
        "transform_cache_bytes": 0,
        "native_buffer_bytes": 0,
    }

    while True:
        progress_ns = time.time_ns()
        max_inter_iteration_gap_ns = max(
            max_inter_iteration_gap_ns, progress_ns - previous_progress_ns
        )
        previous_progress_ns = progress_ns
        wall_elapsed_s = (time.monotonic_ns() - wall_started_ns) / 1_000_000_000
        if iteration >= min_iterations and wall_elapsed_s >= wall_seconds:
            break
        request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"fornax:lifecycle-pressure:{iteration}",
            )
        )
        result = backend.execute(
            handle,
            _request(
                manifest,
                _tensor(iteration),
                request_id=request_id,
                deadline_ns=clock.now_ns + execution_lease_timeout_ns,
            ),
        )
        if result.status != "ok":
            errors.append(
                f"iteration {iteration}: status={result.status} error={result.error}"
            )
            break
        if iteration % defer_every:
            released = backend.release(handle, request_id)
            if not released.released or not released.tombstoned:
                errors.append(f"iteration {iteration}: explicit release did not fence")
                break
            explicit_releases += 1
        else:
            deferred_for_expiry += 1

        health = backend.health(handle)
        maxima["live_requests"] = max(maxima["live_requests"], health.live_requests)
        maxima["release_tombstones"] = max(
            maxima["release_tombstones"], health.release_tombstones
        )
        maxima["completed_result_bytes"] = max(
            maxima["completed_result_bytes"], health.completed_result_bytes
        )
        maxima["transform_cache_bytes"] = max(
            maxima["transform_cache_bytes"], health.transform_cache_bytes
        )
        maxima["native_buffer_bytes"] = max(
            maxima["native_buffer_bytes"], health.native_buffer_bytes
        )
        clock.advance(logical_step_ns)
        iteration += 1
        if pace_ms:
            time.sleep(pace_ms / 1_000.0)

    clock.advance(request_idle_timeout_ns + 1)
    final_sweep = backend.sweep_expired(handle)
    final_health = backend.health(handle)
    wall_elapsed_ns = time.monotonic_ns() - wall_started_ns
    calendar_elapsed_ns = time.time_ns() - calendar_started_ns
    rss_after = _rss_bytes()
    if track_allocations:
        allocation_after, allocation_peak = tracemalloc.get_traced_memory()
    else:
        allocation_after = 0
        allocation_peak = 0
    if track_allocations and not tracing_was_active:
        tracemalloc.stop()

    source_after = _source_identity()
    source_unchanged = source_before["files"] == source_after["files"]
    current_contract_authority = bool(
        source_before["source_committed"]
        and source_after["source_committed"]
        and source_unchanged
    )

    bounds_ok = (
        final_health.live_requests == 0
        and final_health.completed_results == 0
        and final_health.release_tombstones
        <= final_health.max_release_tombstones
        and final_health.completed_result_bytes
        <= final_health.max_completed_result_bytes
        and final_health.transform_cache_bytes
        <= final_health.max_transform_cache_bytes
        and final_health.native_buffer_bytes == 0
        and final_health.native_buffer_high_water_bytes
        <= final_health.max_native_buffer_bytes
        and final_health.expired_requests == deferred_for_expiry
    )
    continuity_ok = wall_seconds == 0 or (
        max_inter_iteration_gap_ns <= continuity_limit_ns
    )
    ok = not errors and iteration >= min_iterations and bounds_ok and continuity_ok
    return {
        "schema_version": 1,
        "record_kind": "fornax-lifecycle-pressure",
        "evidence_class": "t0_t1_contract_stress",
        "measurement_kind": "reference",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "physical_evidence": False,
        "current_contract_authority": current_contract_authority,
        "source_identity": {
            "before": source_before,
            "after": source_after,
            "unchanged_during_run": source_unchanged,
        },
        "runner": {
            "argv": list(sys.argv),
            "working_directory": os.getcwd(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "configuration": {
            "minimum_iterations": min_iterations,
            "configured_wall_seconds": wall_seconds,
            "defer_every": defer_every,
            "pace_ms": pace_ms,
            "max_pause_seconds": max_pause_seconds,
            "lifecycle_clock": "deterministic_logical",
            "logical_step_ns": logical_step_ns,
            "request_idle_timeout_ns": request_idle_timeout_ns,
            "execution_lease_timeout_ns": execution_lease_timeout_ns,
            "release_tombstone_ttl_ns": tombstone_retention_ns,
            "max_release_tombstones": max_release_tombstones,
        },
        "summary": {
            "unique_requests_completed": iteration,
            "explicit_releases": explicit_releases,
            "deferred_for_automatic_expiry": deferred_for_expiry,
            "automatic_expiries": final_health.expired_requests,
            "execution_lease_expiries": final_health.expired_execution_leases,
            "wall_elapsed_ns": wall_elapsed_ns,
            "monotonic_elapsed_ns": wall_elapsed_ns,
            "calendar_elapsed_ns": calendar_elapsed_ns,
            "max_inter_iteration_gap_ns": max_inter_iteration_gap_ns,
            "continuity_limit_ns": continuity_limit_ns,
            "continuity_ok": continuity_ok,
            "bounds_ok": bounds_ok,
            "errors": errors,
        },
        "backend_high_water": maxima,
        "final_health": {
            "live_requests": final_health.live_requests,
            "completed_results": final_health.completed_results,
            "release_tombstones": final_health.release_tombstones,
            "max_release_tombstones": final_health.max_release_tombstones,
            "completed_result_bytes": final_health.completed_result_bytes,
            "max_completed_result_bytes": final_health.max_completed_result_bytes,
            "transform_cache_bytes": final_health.transform_cache_bytes,
            "max_transform_cache_bytes": final_health.max_transform_cache_bytes,
            "native_buffer_bytes": final_health.native_buffer_bytes,
            "native_buffer_high_water_bytes": (
                final_health.native_buffer_high_water_bytes
            ),
            "max_native_buffer_bytes": final_health.max_native_buffer_bytes,
            "native_buffer_copy_operations": (
                final_health.native_buffer_copy_operations
            ),
            "expired_requests": final_health.expired_requests,
            "expired_execution_leases": final_health.expired_execution_leases,
        },
        "final_sweep": {
            "expired_requests": final_sweep.expired_requests,
            "expired_execution_leases": final_sweep.expired_execution_leases,
            "pruned_tombstones": final_sweep.pruned_tombstones,
        },
        "process_diagnostics": {
            "rss_high_water_before_bytes": rss_before,
            "rss_high_water_after_bytes": rss_after,
            "tracemalloc_enabled": track_allocations,
            "traced_current_delta_bytes": allocation_after - allocation_before,
            "traced_peak_bytes": allocation_peak,
        },
        "limitations": [
            "Reference Python execution only; no MAX, accelerator, or network path.",
            "Backend count/byte caps are contract evidence, not a physical RSS bound.",
            "Lifecycle expiry uses a deterministic logical clock; wall time only establishes sustained churn duration.",
            "Sustained continuity requires the recorded maximum civil-clock progress gap to remain within the configured limit.",
            "This artifact does not close G2 or an indefinite-service claim.",
            "current_contract_authority is false unless the exact executed source files were committed and unchanged for the full run.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hardware-free Fornax unique-request lifecycle pressure"
    )
    parser.add_argument("--min-iterations", type=int, default=1_000)
    parser.add_argument("--wall-seconds", type=float, default=0.0)
    parser.add_argument("--defer-every", type=int, default=7)
    parser.add_argument("--pace-ms", type=float, default=0.0)
    parser.add_argument(
        "--max-pause-seconds",
        type=float,
        default=5.0,
        help="fail a sustained run if the civil-clock gap between progress samples exceeds this value",
    )
    parser.add_argument("--no-tracemalloc", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_lifecycle_pressure(
            min_iterations=args.min_iterations,
            wall_seconds=args.wall_seconds,
            defer_every=args.defer_every,
            pace_ms=args.pace_ms,
            max_pause_seconds=args.max_pause_seconds,
            track_allocations=not args.no_tracemalloc,
        )
    except ValueError as exc:
        _parser().error(str(exc))
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(serialized)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
