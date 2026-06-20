from __future__ import annotations

import csv
import itertools
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


def _host_id() -> str:
    return platform.node() or "localhost"


def _memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return 8 * 1024**3


def _cpu_node(host_id: str) -> dict[str, Any]:
    mem = int(_memory_bytes() * 0.70)
    return {
        "id": host_id,
        "host_id": host_id,
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


def nvidia_nodes_from_smi_rows(
    rows: list[dict[str, Any]], host_id: str | None = None
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    host = host_id or _host_id()
    for row in rows:
        compute_class, mem_bandwidth, estimate_source = _nvidia_perf_estimate(row["name"])
        mem_free_bytes_measured = row["memory_free_mib"] * _MIB
        mem_total_bytes_measured = row["memory_total_mib"] * _MIB
        nodes.append(
            {
                "id": f"gpu{row['index']}",
                "host_id": host,
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

    host = _host_id()
    nodes = [_cpu_node(host)]
    errors: list[str] = []
    if nvidia_smi_csv is None:
        nvidia_smi_csv, error = _query_nvidia_smi()
        if error:
            errors.append(error)
    nvidia_rows: list[dict[str, Any]] = []
    if nvidia_smi_csv:
        try:
            nvidia_rows = parse_nvidia_smi_csv(nvidia_smi_csv)
            nodes.extend(nvidia_nodes_from_smi_rows(nvidia_rows, host_id=host))
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
        "host_id": host,
        "source": "fornax.inventory.collect_local_inventory",
        "measured_fields": measured_fields,
        "estimated_fields": ["compute_class", "mem_bandwidth_bytes_s"],
        "collection_errors": errors,
        "note": note,
    }


def _node_host(inventory: dict[str, Any], node: dict[str, Any]) -> str | None:
    value = node.get("host_id", inventory.get("host_id"))
    return str(value) if value else None


def _link_key(link: dict[str, Any]) -> tuple[str, str]:
    a = str(link["a"])
    b = str(link["b"])
    return tuple(sorted((a, b)))


def _local_link_estimate(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float, str]:
    vendors = {str(a.get("vendor")), str(b.get("vendor"))}
    if vendors == {"nvidia"}:
        return 50.0e9, 0.000005, "same_host_gpu_peer_conservative_estimate"
    if "nvidia" in vendors:
        return 24.0e9, 0.000020, "same_host_cpu_gpu_pcie_conservative_estimate"
    return 40.0e9, 0.000010, "same_host_memory_conservative_estimate"


def _estimated_local_links(
    inventory: dict[str, Any], existing: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    nodes = [node for node in inventory.get("nodes", []) if isinstance(node, dict)]
    links: list[dict[str, Any]] = []
    for a, b in itertools.combinations(nodes, 2):
        host_a = _node_host(inventory, a)
        host_b = _node_host(inventory, b)
        if host_a is None or host_a != host_b:
            continue
        node_a = str(a.get("id", ""))
        node_b = str(b.get("id", ""))
        if not node_a or not node_b:
            continue
        key = tuple(sorted((node_a, node_b)))
        if key in existing:
            continue
        bandwidth, latency, source = _local_link_estimate(a, b)
        links.append(
            {
                "a": key[0],
                "b": key[1],
                "bandwidth_bytes_s": bandwidth,
                "latency_s": latency,
                "measurement": {
                    "measured": False,
                    "source": source,
                    "host_id": host_a,
                    "note": (
                        "Topology-derived local estimate for Phase-0 preflight "
                        "runnability; replace with active fabric probe evidence "
                        "before making G1 throughput claims."
                    ),
                },
            }
        )
    return links


def probe_declared_links(inventory: dict[str, Any]) -> dict[str, Any]:
    """Return declared links plus same-host local estimates with provenance.

    Phase 0 may run without a multi-node lab. This helper keeps the workflow
    runnable while making it explicit which links are not active measurements.
    """

    links: list[dict[str, Any]] = []
    for link in inventory.get("links", []):
        if not isinstance(link, dict):
            continue
        copied = dict(link)
        copied.setdefault(
            "measurement",
            {
                "measured": False,
                "source": "declared_inventory_link",
                "note": "Copied from inventory; not actively probed by Fornax.",
            },
        )
        links.append(copied)
    existing = {_link_key(link) for link in links}
    local_links = _estimated_local_links(inventory, existing)
    links.extend(local_links)
    active_measurements = [
        link
        for link in links
        if isinstance(link.get("measurement"), dict)
        and bool(link["measurement"].get("measured"))
    ]
    warnings = []
    if local_links:
        warnings.append("same-host local links are topology-derived estimates")
    if links and not active_measurements:
        warnings.append("no active fabric measurements recorded")
    return {
        "links": links,
        "source": "fornax.inventory.probe_declared_links",
        "measured": bool(active_measurements) and len(active_measurements) == len(links),
        "active_measurement_count": len(active_measurements),
        "estimated_link_count": len(links) - len(active_measurements),
        "warnings": warnings,
        "note": (
            "Declared links are preserved and same-host local links are synthesized "
            "as conservative estimates. Use active fabric measurements before G1 "
            "throughput sign-off."
        ),
    }
