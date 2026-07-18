from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .authority import (
    assess_plan_authority,
    evidence_registry_deployment_issues,
    global_deployment_issues,
    link_admission_issues,
    node_admission_issues,
)
from .cost import StageCost, boundary_transfer_s, estimate_stage_cost
from .evidence import EvidenceRegistry
from .model import (
    BoundaryLink,
    ExpertPlacement,
    Inventory,
    ModelSpec,
    Node,
    PlacementExplanation,
    PlacementPlan,
    Predicted,
    Stage,
    Target,
    normalize_authority_mode,
)


@dataclass(frozen=True)
class _StageCandidate:
    layers: tuple[int, ...]
    node: Node
    cost: StageCost
    transfer_in_s: float
    primary_time_s: float
    replica_times_s: tuple[tuple[str, float], ...]
    replica_costs: tuple[StageCost, ...]

    @property
    def replica_ids(self) -> tuple[str, ...]:
        return tuple(node_id for node_id, _ in self.replica_times_s)

    @property
    def effective_time_s(self) -> float:
        capacity = sum(1.0 / max(time_s, 1e-12) for _, time_s in self.replica_times_s)
        return 1.0 / capacity


@dataclass(frozen=True)
class _CandidatePlan:
    stages: tuple[_StageCandidate, ...]
    boundary_links: tuple[BoundaryLink, ...]


def _json_safe_metric(value: object) -> object:
    """Keep rejection diagnostics serializable when defending tampered objects."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _candidate_node_orders(nodes: list[Node], depth: int) -> list[tuple[Node, ...]]:
    """Return deterministic bounded node orders without a fixed top-N cutoff.

    Exact enumeration is retained for ordinary local fleets. Beyond the bound,
    multiple resource orderings plus forced coverage ensure that memory-heavy,
    bandwidth-heavy, reliable, and compute-heavy nodes all remain candidates.
    """

    max_orders = 4096
    usable = sorted(
        nodes,
        key=lambda n: (
            n.compute_class,
            n.mem_bandwidth_bytes_s,
            n.mem_free_bytes,
            n.reliability,
        ),
        reverse=True,
    )
    permutation_count = math.perm(len(usable), depth)
    if permutation_count <= max_orders:
        combos = itertools.combinations(usable, depth)
        return [perm for combo in combos for perm in itertools.permutations(combo)]

    orderings = [
        usable,
        sorted(usable, key=lambda n: (n.mem_free_bytes, n.compute_class, n.id), reverse=True),
        sorted(
            usable,
            key=lambda n: (n.mem_bandwidth_bytes_s, n.compute_class, n.id),
            reverse=True,
        ),
        sorted(usable, key=lambda n: (n.reliability, n.compute_class, n.id), reverse=True),
    ]
    results: list[tuple[Node, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def add(order: tuple[Node, ...]) -> None:
        key = tuple(node.id for node in order)
        if len(order) == depth and len(set(key)) == depth and key not in seen:
            seen.add(key)
            results.append(order)

    # Preserve every node in every pipeline position before spending the rest of
    # the bound on locally rich permutations.
    for ranking in orderings:
        for forced in ranking:
            remaining = [node for node in ranking if node.id != forced.id]
            for position in range(depth):
                order = remaining[: depth - 1]
                order.insert(position, forced)
                add(tuple(order))

    pool_size = min(len(usable), max(depth + 3, 8))
    for ranking in orderings:
        pool = ranking[:pool_size]
        for order in itertools.permutations(pool, depth):
            add(order)
            if len(results) >= max_orders:
                return results
    return results[:max_orders]


def _resource_capacity_ok(inventory: Inventory, stages: tuple[_StageCandidate, ...]) -> bool:
    allocations: dict[str, float] = {}
    for stage in stages:
        for (node_id, _), cost in zip(stage.replica_times_s, stage.replica_costs):
            allocations[node_id] = allocations.get(node_id, 0.0) + cost.memory_bytes
            for host_id, memory_bytes in cost.expert_host_memory_bytes:
                allocations[host_id] = allocations.get(host_id, 0.0) + memory_bytes
    return all(
        allocated <= inventory.node(node_id).mem_free_bytes
        for node_id, allocated in allocations.items()
    )


def _partition_for_order(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    order: tuple[Node, ...],
    authority_mode: str,
) -> tuple[tuple[int, int, StageCost], ...] | None:
    n_layers = len(model.layers)
    depth = len(order)
    dp: list[list[tuple[float, tuple[tuple[int, int, StageCost], ...]] | None]] = [
        [None for _ in range(n_layers + 1)] for _ in range(depth + 1)
    ]
    dp[0][0] = (0.0, ())

    for stage_idx in range(1, depth + 1):
        node = order[stage_idx - 1]
        for end in range(stage_idx, n_layers + 1):
            best: tuple[float, tuple[tuple[int, int, StageCost], ...]] | None = None
            for start in range(stage_idx - 1, end):
                previous = dp[stage_idx - 1][start]
                if previous is None:
                    continue
                layers = tuple(range(start, end))
                cost = estimate_stage_cost(
                    model,
                    inventory,
                    node,
                    layers,
                    target,
                    authority_mode=authority_mode,
                )
                if cost is None:
                    continue
                transfer_in = 0.0
                if stage_idx > 1:
                    link = inventory.best_link(order[stage_idx - 2].id, node.id)
                    if link is None:
                        continue
                    if link_admission_issues(
                        link, target, deployment=authority_mode == "deployment"
                    ):
                        continue
                    transfer_in = boundary_transfer_s(model, target, link)
                stage_load = (
                    max(cost.decode_compute_s, transfer_in)
                    + cost.remote_wait_exposed_s
                )
                score = max(previous[0], stage_load)
                cuts = previous[1] + ((start, end, cost),)
                if best is None or score < best[0]:
                    best = (score, cuts)
            dp[stage_idx][end] = best
    result = dp[depth][n_layers]
    return result[1] if result is not None else None


def _build_candidate(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    order: tuple[Node, ...],
    cuts: tuple[tuple[int, int, StageCost], ...],
    authority_mode: str,
) -> _CandidatePlan | None:
    stages: list[_StageCandidate] = []
    boundary_links: list[BoundaryLink] = []
    for idx, (start, end, cost) in enumerate(cuts):
        transfer_in = 0.0
        if idx > 0:
            link = inventory.best_link(order[idx - 1].id, order[idx].id)
            if link is None:
                return None
            if link_admission_issues(
                link, target, deployment=authority_mode == "deployment"
            ):
                return None
            boundary_links.append(BoundaryLink(idx - 1, idx, link))
            transfer_in = boundary_transfer_s(model, target, link)
        primary_time = max(cost.decode_compute_s, transfer_in) + cost.remote_wait_exposed_s
        stages.append(
            _StageCandidate(
                layers=tuple(range(start, end)),
                node=order[idx],
                cost=cost,
                transfer_in_s=transfer_in,
                primary_time_s=primary_time,
                replica_times_s=((order[idx].id, primary_time),),
                replica_costs=(cost,),
            )
        )
    candidate = _CandidatePlan(tuple(stages), tuple(boundary_links))
    return candidate if _resource_capacity_ok(inventory, candidate.stages) else None


def _with_replicas(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    plan: _CandidatePlan,
    authority_mode: str,
) -> _CandidatePlan:
    # Replica paths are still modeled with the primary's inbound route.  Keep
    # that useful exploratory optimization out of deployment-authoritative
    # planning until independent replica-route calibration is implemented.
    if authority_mode == "deployment":
        return plan
    used = {stage.node.id for stage in plan.stages}
    spare = [
        node
        for node in inventory.nodes
        if node.supports_stage and node.id not in used
    ]
    stages = list(plan.stages)

    while spare:
        bottleneck_idx = max(
            range(len(stages)), key=lambda idx: stages[idx].effective_time_s
        )
        bottleneck = stages[bottleneck_idx]
        best: tuple[int, float, StageCost, float] | None = None
        for node_idx, node in enumerate(spare):
            cost = estimate_stage_cost(
                model,
                inventory,
                node,
                bottleneck.layers,
                target,
                authority_mode=authority_mode,
            )
            if cost is None:
                continue
            replica_time = (
                max(cost.decode_compute_s, bottleneck.transfer_in_s)
                + cost.remote_wait_exposed_s
            )
            effective_before = bottleneck.effective_time_s
            capacity_after = sum(
                1.0 / max(time_s, 1e-12)
                for _, time_s in bottleneck.replica_times_s + ((node.id, replica_time),)
            )
            improvement = effective_before - 1.0 / capacity_after
            tentative_stage = _StageCandidate(
                layers=bottleneck.layers,
                node=bottleneck.node,
                cost=bottleneck.cost,
                transfer_in_s=bottleneck.transfer_in_s,
                primary_time_s=bottleneck.primary_time_s,
                replica_times_s=bottleneck.replica_times_s + ((node.id, replica_time),),
                replica_costs=bottleneck.replica_costs + (cost,),
            )
            tentative_stages = tuple(
                tentative_stage if index == bottleneck_idx else stage
                for index, stage in enumerate(stages)
            )
            if not _resource_capacity_ok(inventory, tentative_stages):
                continue
            if best is None or improvement > best[1]:
                best = (node_idx, improvement, cost, replica_time)
        if best is None or best[1] <= 0:
            break
        node = spare.pop(best[0])
        _, _, cost, replica_time = best
        stages[bottleneck_idx] = _StageCandidate(
            layers=bottleneck.layers,
            node=bottleneck.node,
            cost=bottleneck.cost,
            transfer_in_s=bottleneck.transfer_in_s,
            primary_time_s=bottleneck.primary_time_s,
            replica_times_s=bottleneck.replica_times_s + ((node.id, replica_time),),
            replica_costs=bottleneck.replica_costs + (cost,),
        )
    return _CandidatePlan(tuple(stages), plan.boundary_links)


def _expert_placements(model: ModelSpec, stages: tuple[_StageCandidate, ...]) -> tuple[ExpertPlacement, ...]:
    placements: list[ExpertPlacement] = []
    for stage in stages:
        primary = stage.node.id
        remote_hosts = stage.cost.expert_hosts or (primary,)
        for layer_id in stage.layers:
            layer = model.layers[layer_id]
            if layer.kind != "moe":
                continue
            for expert_id in range(layer.num_experts):
                if stage.cost.mode == "resident":
                    placements.append(
                        ExpertPlacement(layer_id, expert_id, primary, "hot_resident")
                    )
                else:
                    host = remote_hosts[expert_id % len(remote_hosts)]
                    placements.append(
                        ExpertPlacement(layer_id, expert_id, host, "warm_remote")
                    )
    return tuple(placements)


def _predict(
    target: Target, stages: tuple[_StageCandidate, ...]
) -> Predicted:
    effective = tuple(stage.effective_time_s for stage in stages)
    bottleneck_idx = max(range(len(effective)), key=lambda idx: effective[idx])
    bottleneck = effective[bottleneck_idx]
    throughput = target.concurrency / bottleneck
    mean_stage = sum(effective) / len(effective)
    bubble = max(0.0, 1.0 - mean_stage / bottleneck) if bottleneck else 0.0
    ttft = sum(stage.cost.prefill_s for stage in stages)
    ttft += sum(stage.transfer_in_s for stage in stages[1:])
    latency = sum(stage.primary_time_s for stage in stages) * target.gen_len
    remote_wait = sum(stage.cost.remote_wait_exposed_s for stage in stages)
    remote_hit = max((stage.cost.remote_hit_rate_decode for stage in stages), default=0.0)
    return Predicted(
        throughput_tok_s=throughput,
        ttft_s=ttft,
        per_request_latency_s=latency,
        remote_expert_wait_s_per_token=remote_wait / target.concurrency,
        remote_expert_hit_rate_decode=remote_hit,
        bottleneck_stage=bottleneck_idx,
        bubble_fraction=bubble,
        stage_effective_times_s=effective,
        prediction_provenance=target.prediction_calibration,
    )


def _score(plan: PlacementPlan, objective: str) -> tuple[float, float, float]:
    assert plan.predicted is not None
    p = plan.predicted
    if objective == "min_latency":
        return (-p.per_request_latency_s, p.throughput_tok_s, -p.bubble_fraction)
    if objective == "balanced":
        return (
            p.throughput_tok_s / max(p.per_request_latency_s, 1e-12),
            p.throughput_tok_s,
            -p.bubble_fraction,
        )
    return (p.throughput_tok_s, -p.bubble_fraction, -p.per_request_latency_s)


def _single_layer_stage_possible(
    model: ModelSpec,
    inventory: Inventory,
    node: Node,
    target: Target,
    authority_mode: str,
) -> bool:
    return any(
        estimate_stage_cost(
            model,
            inventory,
            node,
            (layer_id,),
            target,
            authority_mode=authority_mode,
        )
        is not None
        for layer_id in range(len(model.layers))
    )


def _node_exclusion_reason(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    node: Node,
    authority_mode: str,
) -> str:
    if not node.supports_stage:
        return "excluded: node is not stage-capable"
    if model.dtype_activation not in node.supported_dtypes:
        return (
            "excluded: node does not support activation dtype "
            f"{model.dtype_activation}"
        )
    authority_issues = node_admission_issues(
        node,
        model,
        target,
        deployment=authority_mode == "deployment",
    )
    if authority_issues:
        return f"excluded: {authority_issues[0]}"
    if not node.supports_kv and all(
        layer.kind == "attention" or layer.kv_bytes_per_token > 0
        for layer in model.layers
    ):
        return "excluded: node does not support KV-bearing attention stages"
    if not _single_layer_stage_possible(
        model, inventory, node, target, authority_mode
    ):
        return (
            "excluded: insufficient memory, missing expert host/link, or remote "
            "expert wait SLO for even one layer"
        )
    return "excluded: candidate placement scored lower than selected plan"


def _placement_explanations(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    candidate: _CandidatePlan,
    authority_mode: str,
) -> tuple[PlacementExplanation, ...]:
    explanations: list[PlacementExplanation] = []
    selected_ids = {node_id for stage in candidate.stages for node_id, _ in stage.replica_times_s}
    stage_nodes = [
        node
        for node in inventory.nodes
        if node.supports_stage and model.dtype_activation in node.supported_dtypes
    ]
    fastest_compute = max((node.compute_class for node in stage_nodes), default=0.0)

    for index, stage in enumerate(candidate.stages):
        explanations.append(
            PlacementExplanation(
                node_id=stage.node.id,
                decision="selected",
                stage_index=index,
                layers=stage.layers,
                reason=(
                    f"selected as primary for stage {index}; mode={stage.cost.mode}; "
                    f"effective_time_s={stage.effective_time_s:.9f}"
                ),
                metrics={
                    "primary_time_s": stage.primary_time_s,
                    "effective_time_s": stage.effective_time_s,
                    "transfer_in_s": stage.transfer_in_s,
                    "stage_memory_bytes": stage.cost.memory_bytes,
                    "remote_wait_exposed_s": stage.cost.remote_wait_exposed_s,
                    "replicas": list(stage.replica_ids),
                },
            )
        )
        for replica_id, replica_time_s in stage.replica_times_s:
            if replica_id == stage.node.id:
                continue
            explanations.append(
                PlacementExplanation(
                    node_id=replica_id,
                    decision="selected",
                    stage_index=index,
                    layers=stage.layers,
                    reason=f"selected as data-parallel replica for stage {index}",
                    metrics={
                        "replica_time_s": replica_time_s,
                        "primary_node_id": stage.node.id,
                    },
                )
            )

    for node in inventory.nodes:
        if node.id not in selected_ids:
            explanations.append(
                PlacementExplanation(
                    node_id=node.id,
                    decision="excluded",
                    reason=_node_exclusion_reason(
                        model, inventory, target, node, authority_mode
                    ),
                    metrics={
                        "compute_class": _json_safe_metric(node.compute_class),
                        "mem_free_bytes": _json_safe_metric(node.mem_free_bytes),
                        "supports_stage": node.supports_stage,
                        "supported_dtypes": list(node.supported_dtypes),
                    },
                )
            )
            continue
        if fastest_compute > 0 and node.compute_class < fastest_compute:
            assigned_layers = sum(len(stage.layers) for stage in candidate.stages if stage.node.id == node.id)
            explanations.append(
                PlacementExplanation(
                    node_id=node.id,
                    decision="demoted",
                    reason=(
                        "slower than fastest stage-capable node; planner limits "
                        "primary layer ownership or uses replica role"
                    ),
                    metrics={
                        "compute_class": _json_safe_metric(node.compute_class),
                        "fastest_compute_class": _json_safe_metric(fastest_compute),
                        "relative_compute": _json_safe_metric(
                            node.compute_class / fastest_compute
                        ),
                        "primary_layer_count": assigned_layers,
                    },
                )
            )
    return tuple(explanations)


def _infeasible_explanations(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    authority_mode: str,
) -> tuple[PlacementExplanation, ...]:
    return tuple(
        PlacementExplanation(
            node_id=node.id,
            decision="excluded",
            reason=_node_exclusion_reason(
                model, inventory, target, node, authority_mode
            ),
            metrics={
                "compute_class": _json_safe_metric(node.compute_class),
                "mem_free_bytes": _json_safe_metric(node.mem_free_bytes),
                "supports_stage": node.supports_stage,
                "supported_dtypes": list(node.supported_dtypes),
            },
        )
        for node in inventory.nodes
    )


def _materialize(
    model: ModelSpec,
    candidate: _CandidatePlan,
    target: Target,
    inventory: Inventory,
    authority_mode: str,
    evidence_registry: EvidenceRegistry | None,
) -> PlacementPlan:
    predicted = _predict(target, candidate.stages)
    stages = tuple(
        Stage(
            index=idx,
            layers=stage.layers,
            replicas=stage.replica_ids,
            mode=stage.cost.mode,
            expert_hosts=stage.cost.expert_hosts,
        )
        for idx, stage in enumerate(candidate.stages)
    )
    authority = assess_plan_authority(
        model,
        inventory,
        target,
        stages,
        candidate.boundary_links,
        requested_mode=authority_mode,
        evidence_registry=evidence_registry,
    )
    return PlacementPlan(
        stages=stages,
        boundary_links=candidate.boundary_links,
        expert_placement=_expert_placements(model, candidate.stages),
        predicted=predicted,
        feasible=True,
        infeasible_reason=None,
        explanations=_placement_explanations(
            model, inventory, target, candidate, authority_mode
        ),
        authority=authority,
    )


def _infeasible(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    authority_mode: str,
    *,
    evidence_registry: EvidenceRegistry | None = None,
    reason: str | None = None,
) -> PlacementPlan:
    total_mem = sum(node.mem_free_bytes for node in inventory.nodes if node.supports_stage)
    infeasible_reason = reason or (
        f"no feasible contiguous stage placement; model resident bytes "
        f"{model.resident_weight_bytes} vs total stage memory {total_mem}"
    )
    authority = assess_plan_authority(
        model,
        inventory,
        target,
        (),
        (),
        requested_mode=authority_mode,
        evidence_registry=evidence_registry,
        additional_reasons=(infeasible_reason,),
    )
    return PlacementPlan(
        stages=(),
        boundary_links=(),
        expert_placement=(),
        predicted=None,
        feasible=False,
        infeasible_reason=infeasible_reason,
        explanations=_infeasible_explanations(
            model, inventory, target, authority_mode
        ),
        authority=authority,
    )


def plan_placement(
    model: ModelSpec,
    inventory: Inventory,
    target: Target,
    *,
    min_stages: int | None = None,
    max_stages: int | None = None,
    authority_mode: str | None = None,
    evidence_registry: EvidenceRegistry | None = None,
) -> PlacementPlan:
    resolved_authority_mode = normalize_authority_mode(
        authority_mode or target.authority_mode
    )
    if resolved_authority_mode == "deployment":
        input_issues = (
            *global_deployment_issues(model, target),
            *evidence_registry_deployment_issues(
                model,
                inventory,
                target,
                evidence_registry,
            ),
        )
        if input_issues:
            return _infeasible(
                model,
                inventory,
                target,
                resolved_authority_mode,
                evidence_registry=evidence_registry,
                reason="deployment authority rejected: " + "; ".join(input_issues),
            )
    stage_nodes = [
        node
        for node in inventory.nodes
        if node.supports_stage and model.dtype_activation in node.supported_dtypes
    ]
    if not stage_nodes:
        return _infeasible(
            model,
            inventory,
            target,
            resolved_authority_mode,
            evidence_registry=evidence_registry,
            reason=(
                f"no stage-capable nodes support activation dtype "
                f"{model.dtype_activation}"
            ),
        )

    lower = max(1, min_stages or 1)
    upper = max_stages or min(len(stage_nodes), len(model.layers))
    upper = min(max(1, upper), len(stage_nodes), len(model.layers))
    if lower > upper:
        return _infeasible(
            model,
            inventory,
            target,
            resolved_authority_mode,
            evidence_registry=evidence_registry,
        )

    best: PlacementPlan | None = None
    best_score: tuple[float, float, float] | None = None
    for depth in range(lower, upper + 1):
        for order in _candidate_node_orders(stage_nodes, depth):
            cuts = _partition_for_order(
                model, inventory, target, order, resolved_authority_mode
            )
            if cuts is None:
                continue
            candidate = _build_candidate(
                model,
                inventory,
                target,
                order,
                cuts,
                resolved_authority_mode,
            )
            if candidate is None:
                continue
            candidate = _with_replicas(
                model,
                inventory,
                target,
                candidate,
                resolved_authority_mode,
            )
            plan = _materialize(
                model,
                candidate,
                target,
                inventory,
                resolved_authority_mode,
                evidence_registry,
            )
            if (
                resolved_authority_mode == "deployment"
                and not plan.authority.deployment_authorized
            ):
                continue
            score = _score(plan, target.objective)
            if best is None or best_score is None or score > best_score:
                best = plan
                best_score = score
    if best is not None:
        return best
    reason = None
    if resolved_authority_mode == "deployment":
        reason = (
            "deployment authority rejected: no placement satisfies exact "
            "capability, provenance, and calibration admission"
        )
    return _infeasible(
        model,
        inventory,
        target,
        resolved_authority_mode,
        evidence_registry=evidence_registry,
        reason=reason,
    )
