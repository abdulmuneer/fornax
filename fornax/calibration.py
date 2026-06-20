from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any

from .inventory import collect_local_inventory


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def run_cpu_memory_copy_probe(*, size_bytes: int = 16 * 1024 * 1024, iterations: int = 8) -> dict[str, Any]:
    """Measure a simple CPU memory copy path with stdlib bytearrays."""

    _positive_int("size_bytes", size_bytes)
    _positive_int("iterations", iterations)
    src = bytearray((i * 17 + 3) % 251 for i in range(size_bytes))
    dst = bytearray(size_bytes)
    started = time.perf_counter_ns()
    checksum = 0
    for _ in range(iterations):
        dst[:] = src
        checksum = (checksum + dst[0] + dst[len(dst) // 2] + dst[-1]) % 1_000_000_007
    elapsed_ns = time.perf_counter_ns() - started
    elapsed_s = elapsed_ns / 1_000_000_000.0
    copied_bytes = size_bytes * iterations
    return {
        "measured": True,
        "source": "fornax.calibration.cpu_memory_copy.stdlib_bytearray",
        "size_bytes": size_bytes,
        "iterations": iterations,
        "copied_bytes": copied_bytes,
        "elapsed_s": elapsed_s,
        "elapsed_ns": elapsed_ns,
        "bandwidth_bytes_s": copied_bytes / elapsed_s if elapsed_s > 0 else None,
        "checksum": checksum,
        "note": "CPU bytearray copy probe; not a GPU, NIC, or model benchmark.",
    }


def run_cpu_scalar_compute_probe(*, iterations: int = 200_000) -> dict[str, Any]:
    """Measure a deterministic scalar floating-point loop for calibration plumbing."""

    _positive_int("iterations", iterations)
    acc = 0.125
    started = time.perf_counter_ns()
    for i in range(iterations):
        x = ((i % 997) - 498) / 997.0
        acc = acc * 1.0000001 + x * x - x * 0.125
    elapsed_ns = time.perf_counter_ns() - started
    elapsed_s = elapsed_ns / 1_000_000_000.0
    # Four scalar float operations per loop is a lower-bound plumbing metric, not CPU peak FLOP/s.
    scalar_ops = iterations * 4
    return {
        "measured": True,
        "source": "fornax.calibration.cpu_scalar_compute.python_loop",
        "iterations": iterations,
        "scalar_ops": scalar_ops,
        "elapsed_s": elapsed_s,
        "elapsed_ns": elapsed_ns,
        "scalar_ops_s": scalar_ops / elapsed_s if elapsed_s > 0 else None,
        "checksum": acc,
        "note": "Python scalar loop for calibration artifact plumbing; not peak CPU FLOP/s.",
    }


def _torch_cuda_probe(*, matrix_dim: int, iterations: int) -> dict[str, Any]:
    """Optionally measure a tiny CUDA matmul if torch is already installed."""

    _positive_int("matrix_dim", matrix_dim)
    _positive_int("iterations", iterations)
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - calibration should record unavailable backends.
        return {
            "measured": False,
            "backend": "torch",
            "available": False,
            "error": f"torch import failed: {type(exc).__name__}: {exc}",
        }
    if not torch.cuda.is_available():
        return {
            "measured": False,
            "backend": "torch",
            "available": False,
            "error": "torch.cuda.is_available() is false",
            "torch_version": getattr(torch, "__version__", "unknown"),
        }
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        a = torch.ones((matrix_dim, matrix_dim), device=device, dtype=torch.float16)
        b = torch.eye(matrix_dim, device=device, dtype=torch.float16)
        # Warmup and synchronization are explicit so elapsed timing is meaningful.
        for _ in range(2):
            _ = a @ b
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = None
        for _ in range(iterations):
            out = a @ b
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
        elapsed_s = elapsed_ms / 1000.0
        # Dense matmul convention: 2 * n^3 ops per multiply.
        flops = 2 * (matrix_dim ** 3) * iterations
        checksum = float(out[0, 0].item()) if out is not None else None
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "matrix_dim": matrix_dim,
                "iterations": iterations,
                "elapsed_s": elapsed_s,
                "flops": flops,
                "flops_s": flops / elapsed_s if elapsed_s > 0 else None,
                "checksum": checksum,
            }
        )
    return {
        "measured": bool(devices),
        "backend": "torch",
        "available": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "devices": devices,
        "note": "Optional torch CUDA microprobe; useful calibration evidence only for the named installed torch build.",
    }


def run_local_calibration(
    *,
    cpu_memory_bytes: int = 16 * 1024 * 1024,
    cpu_memory_iterations: int = 8,
    cpu_compute_iterations: int = 200_000,
    try_torch_cuda: bool = True,
    cuda_matrix_dim: int = 512,
    cuda_iterations: int = 10,
) -> dict[str, Any]:
    """Return a Phase-0 local calibration artifact with measured/provenance labels."""

    inventory = collect_local_inventory()
    cpu_memory = run_cpu_memory_copy_probe(
        size_bytes=cpu_memory_bytes, iterations=cpu_memory_iterations
    )
    cpu_compute = run_cpu_scalar_compute_probe(iterations=cpu_compute_iterations)
    cuda = (
        _torch_cuda_probe(matrix_dim=cuda_matrix_dim, iterations=cuda_iterations)
        if try_torch_cuda
        else {"measured": False, "backend": "torch", "available": False, "error": "disabled by caller"}
    )
    nvidia_nodes = [
        node for node in inventory.get("nodes", [])
        if isinstance(node, dict) and node.get("vendor") == "nvidia"
    ]
    warnings: list[str] = []
    if nvidia_nodes and not cuda.get("measured"):
        warnings.append("NVIDIA GPUs discovered but no CUDA microprobe was measured")
    warnings.append("CPU probes are plumbing/calibration evidence, not target-model throughput")
    return {
        "version": 1,
        "source": "fornax.calibration.run_local_calibration",
        "measured": True,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "inventory_summary": {
            "node_count": len(inventory.get("nodes", [])) if isinstance(inventory.get("nodes"), list) else None,
            "nvidia_gpu_count": len(nvidia_nodes),
            "nvidia_gpus": [
                {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "device": node.get("device"),
                    "driver_version": node.get("driver_version"),
                    "mem_total_bytes": node.get("mem_total_bytes"),
                    "mem_free_bytes": node.get("mem_free_bytes"),
                }
                for node in nvidia_nodes
            ],
            "collection_errors": inventory.get("collection_errors", []),
        },
        "cpu_memory_copy": cpu_memory,
        "cpu_scalar_compute": cpu_compute,
        "cuda_microprobe": cuda,
        "warnings": warnings,
        "note": (
            "Calibration artifact for planner evidence plumbing. G1 throughput claims still require "
            "target-model/device/fabric probes and reviewed thresholds."
        ),
    }
