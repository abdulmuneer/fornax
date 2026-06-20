from __future__ import annotations

from dataclasses import dataclass

from .model import Inventory, Link, ModelSpec, Node, Target, activation_nbytes


@dataclass(frozen=True)
class StageCost:
    mode: str
    memory_bytes: float
    local_decode_s: float
    prefill_s: float
    remote_wait_exposed_s: float
    remote_hit_rate_decode: float
    expert_hosts: tuple[str, ...]

    @property
    def decode_compute_s(self) -> float:
        return self.local_decode_s


def activation_buffer_bytes(model: ModelSpec, target: Target) -> float:
    return 2 * target.concurrency * model.hidden_dim * activation_nbytes(
        model.dtype_activation
    )


def boundary_transfer_s(model: ModelSpec, target: Target, link: Link) -> float:
    payload = target.concurrency * model.hidden_dim * activation_nbytes(
        model.dtype_activation
    )
    return payload / link.bandwidth_bytes_s + link.latency_s


def stage_memory_bytes(
    model: ModelSpec, layers: tuple[int, ...], target: Target, mode: str
) -> float:
    selected = [model.layers[i] for i in layers]
    if mode == "resident":
        weight_bytes = sum(layer.resident_weight_bytes for layer in selected)
    elif mode in {"remote_experts", "weight_lru"}:
        weight_bytes = sum(layer.base_weight_bytes for layer in selected)
    else:
        raise ValueError(f"unsupported stage mode: {mode}")
    kv_bytes = target.concurrency * sum(layer.kv_bytes_per_token for layer in selected)
    return weight_bytes + kv_bytes + activation_buffer_bytes(model, target)


def _local_flops_per_token(layers: list, mode: str) -> int:
    if mode == "resident":
        return sum(layer.resident_flops_per_token for layer in layers)
    return sum(layer.base_flops_per_token for layer in layers)


def _best_expert_host(
    inventory: Inventory, node: Node, model: ModelSpec, target: Target
) -> tuple[Node | None, float]:
    candidates = [x for x in inventory.nodes if x.supports_expert_worker]
    best: tuple[Node | None, float] = (None, float("inf"))
    payload = target.concurrency * model.hidden_dim * activation_nbytes(
        model.dtype_activation
    )
    for candidate in candidates:
        if candidate.id == node.id:
            transfer = 0.0
        else:
            link = inventory.best_link(node.id, candidate.id)
            if link is None:
                continue
            transfer = 2 * (payload / link.bandwidth_bytes_s + link.latency_s)
        if transfer < best[1]:
            best = (candidate, transfer)
    return best


def _remote_expert_wait_s(
    inventory: Inventory,
    node: Node,
    model: ModelSpec,
    target: Target,
    layers: list,
) -> tuple[float, float, tuple[str, ...]]:
    remote_layers = [layer for layer in layers if layer.kind == "moe"]
    if not remote_layers:
        return 0.0, 0.0, ()
    host, transfer_per_expert = _best_expert_host(inventory, node, model, target)
    if host is None:
        return float("inf"), 1.0, ()

    remote_wait = 0.0
    total_active = 0
    for layer in remote_layers:
        total_active += layer.experts_active
        expert_compute = (
            target.concurrency * layer.expert_flops_per_token / host.compute_class
        )
        remote_wait += layer.experts_active * (transfer_per_expert + expert_compute)
    return remote_wait, 1.0 if total_active else 0.0, (host.id,)


def estimate_stage_cost(
    model: ModelSpec,
    inventory: Inventory,
    node: Node,
    layers: tuple[int, ...],
    target: Target,
) -> StageCost | None:
    if not layers:
        return None
    if not node.supports_stage:
        return None
    if model.dtype_activation not in node.supported_dtypes:
        return None

    selected = [model.layers[i] for i in layers]
    resident_mem = stage_memory_bytes(model, layers, target, "resident")
    if resident_mem <= node.mem_free_bytes:
        mode = "resident"
        memory = resident_mem
        remote_wait = 0.0
        remote_hit_rate = 0.0
        expert_hosts: tuple[str, ...] = ()
    else:
        remote_mem = stage_memory_bytes(model, layers, target, "remote_experts")
        remote_wait, remote_hit_rate, expert_hosts = _remote_expert_wait_s(
            inventory, node, model, target, selected
        )
        if (
            remote_mem > node.mem_free_bytes
            or remote_wait == float("inf")
            or (
                target.remote_expert_wait_slo_s is not None
                and remote_wait / target.concurrency > target.remote_expert_wait_slo_s
            )
        ):
            return None
        mode = "remote_experts"
        memory = remote_mem

    weight_bytes = (
        sum(layer.resident_weight_bytes for layer in selected)
        if mode == "resident"
        else sum(layer.base_weight_bytes for layer in selected)
    )
    kv_bytes = target.concurrency * sum(layer.kv_bytes_per_token for layer in selected)
    local_flops = target.concurrency * _local_flops_per_token(selected, mode)
    local_decode = (
        weight_bytes / node.mem_bandwidth_bytes_s
        + kv_bytes / node.mem_bandwidth_bytes_s
        + local_flops / node.compute_class
    )
    prefill_flops = (
        target.concurrency
        * target.prompt_len
        * sum(layer.resident_flops_per_token for layer in selected)
    )
    prefill = prefill_flops / node.compute_class
    return StageCost(
        mode=mode,
        memory_bytes=memory,
        local_decode_s=local_decode,
        prefill_s=prefill,
        remote_wait_exposed_s=remote_wait,
        remote_hit_rate_decode=remote_hit_rate,
        expert_hosts=expert_hosts,
    )
