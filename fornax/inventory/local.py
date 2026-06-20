from __future__ import annotations

import os
import platform
from typing import Any


def _memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return 8 * 1024**3


def collect_local_inventory() -> dict[str, Any]:
    """Return a conservative CPU-only inventory for Phase-0 dry runs."""

    mem = int(_memory_bytes() * 0.70)
    return {
        "nodes": [
            {
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
            }
        ],
        "links": [],
        "source": "fornax.inventory.collect_local_inventory",
        "note": "CPU-only conservative placeholder; replace with measured probes for G1 evidence.",
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
