from __future__ import annotations

import csv
import itertools
import json
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


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _external_torch_active_local_links(
    *,
    inventory: dict[str, Any],
    links: list[dict[str, Any]],
    torch_python: str,
    size_bytes: int,
    iterations: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    request_links: list[dict[str, Any]] = []
    nodes = {
        str(node.get("id")): node
        for node in inventory.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    for link in links:
        a_id = str(link.get("a", ""))
        b_id = str(link.get("b", ""))
        a = nodes.get(a_id)
        b = nodes.get(b_id)
        if not a or not b:
            continue
        if _node_host(inventory, a) != _node_host(inventory, b):
            continue
        vendors = {str(a.get("vendor")), str(b.get("vendor"))}
        if "nvidia" not in vendors:
            continue
        request_links.append(
            {
                "a": a_id,
                "b": b_id,
                "a_vendor": a.get("vendor"),
                "b_vendor": b.get("vendor"),
                "a_device": a.get("device"),
                "b_device": b.get("device"),
            }
        )
    if not request_links:
        return {}, ["active local probe had no torch-measurable same-host links"]

    script = """
import json
import sys
import time

request = json.load(sys.stdin)
size_bytes = int(request["size_bytes"])
iterations = int(request["iterations"])
links = request["links"]
try:
    import torch
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": f"torch import failed: {type(exc).__name__}: {exc}",
    }))
    raise SystemExit(0)
if not torch.cuda.is_available():
    print(json.dumps({
        "ok": False,
        "error": "torch.cuda.is_available() is false",
        "torch_version": getattr(torch, "__version__", "unknown"),
    }))
    raise SystemExit(0)

def _sync(*devices):
    seen = set()
    for device in devices:
        if device and device not in seen:
            torch.cuda.synchronize(torch.device(device))
            seen.add(device)

def _elapsed(copy_fn, *devices, loops=None):
    count = loops or iterations
    for _ in range(2):
        copy_fn()
    _sync(*devices)
    started = time.perf_counter_ns()
    for _ in range(count):
        copy_fn()
    _sync(*devices)
    return (time.perf_counter_ns() - started) / 1_000_000_000.0, count

elements = max(size_bytes // 2, 1)
out = []
for link in links:
    try:
        a_vendor = str(link.get("a_vendor"))
        b_vendor = str(link.get("b_vendor"))
        a_device = link.get("a_device")
        b_device = link.get("b_device")
        vendors = {a_vendor, b_vendor}
        if vendors == {"nvidia"}:
            if not a_device or not b_device:
                raise RuntimeError("missing cuda device for GPU-GPU link")
            src = torch.empty((elements,), device=torch.device(str(a_device)), dtype=torch.float16)
            dst = torch.empty((elements,), device=torch.device(str(b_device)), dtype=torch.float16)
            elapsed_s, count = _elapsed(lambda: dst.copy_(src, non_blocking=False), str(a_device), str(b_device))
            tiny_src = torch.empty((1,), device=torch.device(str(a_device)), dtype=torch.float16)
            tiny_dst = torch.empty((1,), device=torch.device(str(b_device)), dtype=torch.float16)
            latency_elapsed, latency_count = _elapsed(lambda: tiny_dst.copy_(tiny_src, non_blocking=False), str(a_device), str(b_device), loops=max(iterations, 20))
            latency_s = latency_elapsed / latency_count
            details = {"direction": f"{a_device}->{b_device}"}
        elif "nvidia" in vendors:
            gpu_device = a_device if a_vendor == "nvidia" else b_device
            if not gpu_device:
                raise RuntimeError("missing cuda device for CPU-GPU link")
            cpu = torch.empty((elements,), device="cpu", dtype=torch.float16)
            gpu = torch.empty((elements,), device=torch.device(str(gpu_device)), dtype=torch.float16)
            h2d_elapsed, h2d_count = _elapsed(lambda: gpu.copy_(cpu, non_blocking=False), str(gpu_device))
            d2h_elapsed, d2h_count = _elapsed(lambda: cpu.copy_(gpu, non_blocking=False), str(gpu_device))
            tiny_cpu = torch.empty((1,), device="cpu", dtype=torch.float16)
            tiny_gpu = torch.empty((1,), device=torch.device(str(gpu_device)), dtype=torch.float16)
            h2d_latency_elapsed, h2d_latency_count = _elapsed(lambda: tiny_gpu.copy_(tiny_cpu, non_blocking=False), str(gpu_device), loops=max(iterations, 20))
            d2h_latency_elapsed, d2h_latency_count = _elapsed(lambda: tiny_cpu.copy_(tiny_gpu, non_blocking=False), str(gpu_device), loops=max(iterations, 20))
            h2d_bw = size_bytes * h2d_count / h2d_elapsed if h2d_elapsed > 0 else None
            d2h_bw = size_bytes * d2h_count / d2h_elapsed if d2h_elapsed > 0 else None
            bandwidths = [v for v in (h2d_bw, d2h_bw) if v is not None]
            if not bandwidths:
                raise RuntimeError("CPU-GPU bandwidth timing was zero")
            elapsed_s = size_bytes * iterations / min(bandwidths)
            count = iterations
            latency_s = max(h2d_latency_elapsed / h2d_latency_count, d2h_latency_elapsed / d2h_latency_count)
            details = {
                "h2d_bandwidth_bytes_s": h2d_bw,
                "d2h_bandwidth_bytes_s": d2h_bw,
                "gpu_device": gpu_device,
            }
        else:
            raise RuntimeError("link is not torch-measurable")
        bandwidth = size_bytes * count / elapsed_s if elapsed_s > 0 else None
        if bandwidth is None:
            raise RuntimeError("bandwidth timing was zero")
        out.append({
            "a": link["a"],
            "b": link["b"],
            "bandwidth_bytes_s": bandwidth,
            "latency_s": latency_s,
            "measurement": {
                "measured": True,
                "source": "fornax.inventory.active_local_torch_copy",
                "backend": "torch",
                "backend_mode": "external_python",
                "python_executable": sys.executable,
                "torch_version": getattr(torch, "__version__", "unknown"),
                "size_bytes": size_bytes,
                "iterations": iterations,
                "details": details,
                "note": "Same-host torch copy microprobe for Phase-0 fabric evidence; not a distributed network benchmark.",
            },
        })
    except Exception as exc:
        out.append({
            "a": link.get("a"),
            "b": link.get("b"),
            "measurement": {
                "measured": False,
                "source": "fornax.inventory.active_local_torch_copy",
                "error": f"{type(exc).__name__}: {exc}",
            },
        })
print(json.dumps({"ok": True, "links": out}))
"""
    request = {
        "links": request_links,
        "size_bytes": size_bytes,
        "iterations": iterations,
    }
    try:
        result = subprocess.run(
            [torch_python, "-c", script],
            input=json.dumps(request),
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, [f"active local probe failed to launch: {type(exc).__name__}: {exc}"]
    stdout = result.stdout.strip()
    if result.returncode != 0:
        return {}, [
            "active local probe exited nonzero: "
            f"returncode={result.returncode} stderr={result.stderr.strip()[-500:]}"
        ]
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return {}, [f"active local probe did not emit JSON: {exc}"]
    if not isinstance(payload, dict):
        return {}, ["active local probe JSON was not an object"]
    if not payload.get("ok"):
        return {}, [f"active local probe unavailable: {payload.get('error', 'unknown error')}"]

    measured: dict[tuple[str, str], dict[str, Any]] = {}
    warnings: list[str] = []
    for item in payload.get("links", []):
        if not isinstance(item, dict):
            continue
        a = str(item.get("a", ""))
        b = str(item.get("b", ""))
        if not a or not b:
            continue
        key = tuple(sorted((a, b)))
        measurement = item.get("measurement")
        if isinstance(measurement, dict) and measurement.get("measured"):
            measured[key] = item
        elif isinstance(measurement, dict) and measurement.get("error"):
            warnings.append(f"active local probe failed for {key[0]}-{key[1]}: {measurement['error']}")
    return measured, warnings


def probe_declared_links(
    inventory: dict[str, Any],
    *,
    active_local: bool = False,
    torch_python: str | None = None,
    active_local_bytes: int = 16 * 1024 * 1024,
    active_local_iterations: int = 4,
) -> dict[str, Any]:
    """Return declared links plus same-host local estimates with provenance.

    Phase 0 may run without a multi-node lab. This helper keeps the workflow
    runnable while making it explicit which links are not active measurements.
    When requested, same-host CPU/GPU and GPU/GPU links can be replaced by
    external-torch copy microprobe measurements.
    """

    _positive_int("active_local_bytes", active_local_bytes)
    _positive_int("active_local_iterations", active_local_iterations)
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
    active_warnings: list[str] = []
    if active_local:
        if torch_python:
            measured_links, active_warnings = _external_torch_active_local_links(
                inventory=inventory,
                links=[*links, *local_links],
                torch_python=torch_python,
                size_bytes=active_local_bytes,
                iterations=active_local_iterations,
            )
            for link in [*links, *local_links]:
                measured = measured_links.get(_link_key(link))
                if measured:
                    link.update(
                        {
                            "bandwidth_bytes_s": measured["bandwidth_bytes_s"],
                            "latency_s": measured["latency_s"],
                            "measurement": measured["measurement"],
                        }
                    )
        else:
            active_warnings.append("active local probe requested without --torch-python")
    links.extend(local_links)
    active_measurements = [
        link
        for link in links
        if isinstance(link.get("measurement"), dict)
        and bool(link["measurement"].get("measured"))
    ]
    estimated_links = len(links) - len(active_measurements)
    warnings = []
    if local_links and estimated_links:
        if active_local and active_measurements:
            warnings.append("links include unmeasured same-host estimates or declarations")
        else:
            warnings.append("same-host local links are topology-derived estimates")
    elif estimated_links:
        warnings.append("links include unmeasured declarations")
    if links and not active_measurements:
        warnings.append("no active fabric measurements recorded")
    warnings.extend(active_warnings)
    return {
        "links": links,
        "source": "fornax.inventory.probe_declared_links",
        "measured": bool(active_measurements) and len(active_measurements) == len(links),
        "active_measurement_count": len(active_measurements),
        "estimated_link_count": estimated_links,
        "warnings": warnings,
        "active_probe": {
            "requested": active_local,
            "backend": "torch" if active_local and torch_python else None,
            "backend_mode": "external_python" if active_local and torch_python else None,
            "torch_python": torch_python,
            "size_bytes": active_local_bytes,
            "iterations": active_local_iterations,
        },
        "note": (
            "Declared links are preserved and same-host local links are synthesized "
            "as conservative estimates unless active local measurement is requested. "
            "Same-host torch copy probes are Phase-0 fabric evidence, not a "
            "distributed network benchmark."
        ),
    }
