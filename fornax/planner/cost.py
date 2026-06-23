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


def _context_tokens(target: Target) -> int:
    return target.prompt_len + target.gen_len


def kv_cache_bytes(model: ModelSpec, layers: tuple[int, ...], target: Target) -> float:
    selected = [model.layers[i] for i in layers]
    return (
        target.concurrency
        * _context_tokens(target)
        * sum(layer.kv_bytes_per_token for layer in selected)
    )


def routing_metadata_bytes(
    model: ModelSpec, layers: tuple[int, ...], target: Target
) -> float:
    selected = [model.layers[i] for i in layers]
    active_experts = sum(
        layer.experts_active for layer in selected if layer.kind == "moe"
    )
    return (
        target.concurrency
        * _context_tokens(target)
        * active_experts
        * target.routing_metadata_bytes_per_token
    )


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
    working_bytes = (
        weight_bytes
        + kv_cache_bytes(model, layers, target)
        + activation_buffer_bytes(model, target)
        + routing_metadata_bytes(model, layers, target)
    )
    temp_bytes = working_bytes * target.temp_buffer_fraction
    reserved_bytes = target.runtime_reserve_bytes + (
        working_bytes * target.memory_reserve_fraction
    )
    fragmented_bytes = (
        working_bytes + temp_bytes + reserved_bytes
    ) * target.fragmentation_margin_fraction
    return working_bytes + temp_bytes + reserved_bytes + fragmented_bytes


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


def _expected_decode_experts(model: ModelSpec, layer_id: int) -> tuple[float, float]:
    layer = model.layers[layer_id]
    if layer.kind != "moe":
        return 0.0, 0.0
    traces = [trace for trace in model.expert_traces if trace.layer_id == layer_id]
    if traces:
        expected = min(
            sum(trace.hit_rate_decode for trace in traces),
            float(layer.experts_active),
        )
    else:
        expected = float(layer.experts_active)
    hit_rate = min(expected / max(layer.num_experts, 1), 1.0)
    return expected, hit_rate


def _remote_expert_wait_s(
    inventory: Inventory,
    node: Node,
    model: ModelSpec,
    target: Target,
    layer_ids: tuple[int, ...],
) -> tuple[float, float, tuple[str, ...]]:
    remote_layer_ids = [
        layer_id for layer_id in layer_ids if model.layers[layer_id].kind == "moe"
    ]
    if not remote_layer_ids:
        return 0.0, 0.0, ()
    host, transfer_per_expert = _best_expert_host(inventory, node, model, target)
    if host is None:
        return float("inf"), 1.0, ()

    remote_wait = 0.0
    remote_hit_rate = 0.0
    for layer_id in remote_layer_ids:
        layer = model.layers[layer_id]
        expected_experts, layer_hit_rate = _expected_decode_experts(model, layer_id)
        expert_compute = (
            target.concurrency * layer.expert_flops_per_token / host.compute_class
        )
        remote_wait += expected_experts * (transfer_per_expert + expert_compute)
        remote_hit_rate += layer_hit_rate
    return remote_wait, remote_hit_rate / len(remote_layer_ids), (host.id,)


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
            inventory, node, model, target, layers
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
    kv_bytes = kv_cache_bytes(model, layers, target)
    local_flops = target.concurrency * _local_flops_per_token(selected, mode)
    # Conservative serial roofline: do not credit memory/compute overlap yet.
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
