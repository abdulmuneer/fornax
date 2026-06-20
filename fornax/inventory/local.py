from __future__ import annotations

import csv
import os
import platform
import subprocess
from io import StringIO
from typing import Any

_MIB = 1024 * 1024


NVIDIA_QUERY_ARGS = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.free,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]


def _memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return 8 * 1024**3


def _cpu_node() -> dict[str, Any]:
    mem = int(_memory_bytes() * 0.70)
    return {
        "id": platform.node() or "localhost",
        "vendor": "cpu",
        "runtime": "custom",
        "mem_free_bytes": mem,
        "compute_class": 2.0e11,
        "mem_bandwidth_bytes_s": 5.0e10,
        "reliability": 1.0,
        "supports_stage": True,
        "supports_expert_worker": True,
        "supports_kv": True,
        "supported_dtypes": ["fp16", "bf16", "fp8"],
        "measurement": {
            "mem_free_bytes": "os.sysconf physical memory with 30% reserve",
            "compute_class": "static estimate, not measured",
            "mem_bandwidth_bytes_s": "static estimate, not measured",
        },
    }


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    """Parse the CSV form emitted by NVIDIA_QUERY_ARGS."""

    rows: list[dict[str, Any]] = []
    for row in csv.reader(StringIO(text.strip())):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError(f"expected 5 nvidia-smi columns, got {len(row)}: {row}")
        index, name, mem_free_mib, mem_total_mib, driver_version = [x.strip() for x in row]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_free_mib": int(float(mem_free_mib)),
                "memory_total_mib": int(float(mem_total_mib)),
                "driver_version": driver_version,
            }
        )
    return rows


def _nvidia_perf_estimate(name: str) -> tuple[float, float, str]:
    """Return conservative planning estimates until profiler probes replace them."""

    normalized = name.lower()
    if "h100" in normalized:
        return 4.0e14, 2.5e12, "static_estimate:h100_conservative"
    if "a100" in normalized:
        return 1.5e14, 1.5e12, "static_estimate:a100_conservative"
    if "l40" in normalized:
        return 9.0e13, 7.0e11, "static_estimate:l40_conservative"
    if "4090" in normalized:
        return 8.0e13, 8.0e11, "static_estimate:rtx4090_conservative"
    return 5.0e13, 5.0e11, "static_estimate:unknown_nvidia_conservative"


def nvidia_nodes_from_smi_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for row in rows:
        compute_class, mem_bandwidth, estimate_source = _nvidia_perf_estimate(row["name"])
        mem_free_bytes_measured = row["memory_free_mib"] * _MIB
        mem_total_bytes_measured = row["memory_total_mib"] * _MIB
        nodes.append(
            {
                "id": f"gpu{row['index']}",
                "vendor": "nvidia",
                "runtime": "max",
                "device": f"cuda:{row['index']}",
                "name": row["name"],
                "driver_version": row["driver_version"],
                "mem_free_bytes": int(mem_free_bytes_measured * 0.90),
                "mem_total_bytes": mem_total_bytes_measured,
                "compute_class": compute_class,
                "mem_bandwidth_bytes_s": mem_bandwidth,
                "reliability": 1.0,
                "supports_stage": True,
                "supports_expert_worker": True,
                "supports_kv": True,
                "supported_dtypes": ["fp16", "bf16", "fp8"],
                "measurement": {
                    "mem_free_bytes": "nvidia-smi memory.free with 10% planner reserve",
                    "mem_total_bytes": "nvidia-smi memory.total",
                    "compute_class": estimate_source,
                    "mem_bandwidth_bytes_s": estimate_source,
                },
            }
        )
    return nodes


def _query_nvidia_smi() -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            NVIDIA_QUERY_ARGS,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    return result.stdout, None


def collect_local_inventory(nvidia_smi_csv: str | None = None) -> dict[str, Any]:
    """Return a local Phase-0 inventory with measured memory and honest estimates.

    GPU memory comes from `nvidia-smi` when available. Effective FLOP/s and memory
    bandwidth remain static estimates until the Phase-0 benchmark/profiler probes
    calibrate them, so they are explicitly labeled as estimates in the output.
    """

    nodes = [_cpu_node()]
    errors: list[str] = []
    if nvidia_smi_csv is None:
        nvidia_smi_csv, error = _query_nvidia_smi()
        if error:
            errors.append(error)
    nvidia_rows: list[dict[str, Any]] = []
    if nvidia_smi_csv:
        try:
            nvidia_rows = parse_nvidia_smi_csv(nvidia_smi_csv)
            nodes.extend(nvidia_nodes_from_smi_rows(nvidia_rows))
        except ValueError as exc:
            errors.append(f"nvidia-smi parse error: {exc}")

    measured_fields = ["cpu.physical_memory"]
    if nvidia_rows:
        measured_fields.extend(["nvidia.memory_free_mib", "nvidia.memory_total_mib"])

    note = (
        "GPU memory is measured when nvidia-smi is available; compute and bandwidth "
        "are planner estimates until calibrated by Phase-0 probes."
        if len(nodes) > 1
        else "CPU-only conservative placeholder; replace with measured probes for G1 evidence."
    )
    return {
        "nodes": nodes,
        "links": [],
        "source": "fornax.inventory.collect_local_inventory",
        "measured_fields": measured_fields,
        "estimated_fields": ["compute_class", "mem_bandwidth_bytes_s"],
        "collection_errors": errors,
        "note": note,
    }


def probe_declared_links(inventory: dict[str, Any]) -> dict[str, Any]:
    """Echo declared links with a provenance marker.

    Phase 0 may run without a multi-node lab. This helper keeps the workflow
    runnable while making it explicit that declared links are not measurements.
    """

    return {
        "links": list(inventory.get("links", [])),
        "source": "fornax.inventory.probe_declared_links",
        "measured": False,
        "note": "No active network probe yet; links are copied from inventory.",
    }
