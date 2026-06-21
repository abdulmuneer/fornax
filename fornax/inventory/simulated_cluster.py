from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any


SIMULATED_CLUSTER_PROFILES = ("two-gpu-heterogeneous", "two-gpu-balanced")
SIMULATED_CLUSTER_WARNING = (
    "Simulation evidence only; this logical cluster does not close real multi-host "
    "hardware gates."
)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _non_negative_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


def _cuda_visible_device(node: dict[str, Any], fallback: int) -> str:
    device = str(node.get("device", ""))
    if device.startswith("cuda:"):
        value = device.split(":", 1)[1]
        if value:
            return value
    node_id = str(node.get("id", ""))
    if node_id.startswith("gpu") and node_id[3:].isdigit():
        return node_id[3:]
    return str(fallback)


def _scaled_number(value: Any, factor: float, *, minimum: float = 1.0) -> Any:
    scaled = max(minimum, float(value) * factor)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(scaled)
    return scaled


def _gpu_nodes(source_inventory: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = source_inventory.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("source inventory must contain a nodes list")
    return [
        node
        for node in nodes
        if isinstance(node, dict) and str(node.get("vendor")) == "nvidia"
    ]


def build_logical_cluster_inventory(
    source_inventory: dict[str, Any],
    *,
    gpu_count: int = 2,
    profile: str = "two-gpu-heterogeneous",
    link_bandwidth_bytes_s: float = 25.0e9,
    link_latency_s: float = 0.00025,
    slow_node_factor: float = 0.65,
) -> dict[str, Any]:
    """Split local GPUs into logical hosts for simulated distributed validation.

    Each selected physical GPU becomes a logical single-GPU host. Runtime workers
    can bind the physical GPU through CUDA_VISIBLE_DEVICES while Fornax sees a
    normal multi-host inventory and declared cross-host links.
    """

    if not isinstance(source_inventory, dict):
        raise ValueError("source inventory must contain a JSON object")
    _positive_int("gpu_count", gpu_count)
    if gpu_count < 2:
        raise ValueError("gpu_count must be at least 2 for a logical cluster")
    if profile not in SIMULATED_CLUSTER_PROFILES:
        raise ValueError(
            "profile must be one of: " + ", ".join(SIMULATED_CLUSTER_PROFILES)
        )
    _positive_float("link_bandwidth_bytes_s", link_bandwidth_bytes_s)
    _non_negative_float("link_latency_s", link_latency_s)
    _positive_float("slow_node_factor", slow_node_factor)

    gpus = _gpu_nodes(source_inventory)
    if len(gpus) < gpu_count:
        raise ValueError(
            f"source inventory has {len(gpus)} NVIDIA GPU node(s), "
            f"but {gpu_count} are required"
        )

    source_host = str(source_inventory.get("host_id") or "localhost")
    simulated_nodes: list[dict[str, Any]] = []
    for index, source_node in enumerate(gpus[:gpu_count]):
        node = deepcopy(source_node)
        original_id = str(node.get("id") or f"gpu{index}")
        original_host = str(node.get("host_id") or source_host)
        original_device = str(node.get("device") or f"cuda:{index}")
        logical_host = f"sim-host-{index}"
        role = "primary_stage_node" if index == 0 else "capacity_worker_node"

        node["id"] = f"sim-gpu{index}"
        node["host_id"] = logical_host
        node["physical_host_id"] = original_host
        node["physical_device"] = original_device
        node["device"] = "cuda:0"
        node["logical_host"] = True
        node["worker_environment"] = {
            "CUDA_VISIBLE_DEVICES": _cuda_visible_device(source_node, index)
        }
        node["simulation"] = {
            "mode": "logical_multi_host",
            "profile": profile,
            "source_node_id": original_id,
            "source_host_id": original_host,
            "source_device": original_device,
            "role": role,
            "warning": SIMULATED_CLUSTER_WARNING,
        }

        if profile == "two-gpu-heterogeneous" and index > 0:
            node["compute_class"] = _scaled_number(
                node.get("compute_class", 1.0), slow_node_factor
            )
            node["mem_bandwidth_bytes_s"] = _scaled_number(
                node.get("mem_bandwidth_bytes_s", 1.0), slow_node_factor
            )
            node["mem_free_bytes"] = _scaled_number(
                node.get("mem_free_bytes", 1), slow_node_factor
            )
            node["simulation"]["role"] = "slower_capacity_worker_node"
            node["simulation"]["slow_node_factor"] = slow_node_factor
            measurement = dict(node.get("measurement", {}))
            measurement["simulation"] = (
                "compute, bandwidth, and available memory scaled by logical "
                "heterogeneity profile"
            )
            node["measurement"] = measurement

        simulated_nodes.append(node)

    links: list[dict[str, Any]] = []
    for a, b in itertools.combinations(simulated_nodes, 2):
        links.append(
            {
                "a": a["id"],
                "b": b["id"],
                "bandwidth_bytes_s": float(link_bandwidth_bytes_s),
                "latency_s": float(link_latency_s),
                "measurement": {
                    "measured": False,
                    "simulated": True,
                    "source": "fornax.inventory.simulated_cluster.logical_link",
                    "profile": profile,
                    "note": (
                        "Synthetic cross-host link for local milestone development; "
                        "replace with real fabric evidence on a physical cluster."
                    ),
                },
            }
        )

    estimated_fields = set(source_inventory.get("estimated_fields", []))
    estimated_fields.update(
        [
            "logical_host_assignment",
            "simulated_link_bandwidth_bytes_s",
            "simulated_link_latency_s",
        ]
    )
    if profile == "two-gpu-heterogeneous":
        estimated_fields.update(
            [
                "simulated_compute_class_scale",
                "simulated_mem_bandwidth_scale",
                "simulated_mem_free_scale",
            ]
        )

    return {
        "nodes": simulated_nodes,
        "links": links,
        "host_id": "simulated-logical-cluster",
        "source": "fornax.inventory.simulated_cluster.build_logical_cluster_inventory",
        "source_inventory": {
            "host_id": source_host,
            "source": source_inventory.get("source"),
            "node_count": len(source_inventory.get("nodes", [])),
            "selected_gpu_count": gpu_count,
        },
        "simulation": {
            "mode": "logical_multi_host",
            "profile": profile,
            "source_host_id": source_host,
            "logical_host_count": len(simulated_nodes),
            "physical_gpu_count": gpu_count,
            "warning": SIMULATED_CLUSTER_WARNING,
        },
        "measured_fields": list(source_inventory.get("measured_fields", [])),
        "estimated_fields": sorted(estimated_fields),
        "collection_errors": list(source_inventory.get("collection_errors", [])),
        "note": (
            "Local GPUs are exposed as separate logical hosts for distributed "
            "development. This validates scheduler, planner, transport-contract, "
            "and observability paths without claiming real multi-host hardware "
            "evidence."
        ),
    }
