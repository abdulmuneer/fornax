from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DTYPE_BYTES: dict[str, float] = {
    "q4": 0.5,
    "q8": 1.0,
    "fp8": 1.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
}


def activation_nbytes(dtype: str) -> float:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported activation dtype: {dtype}")
    return DTYPE_BYTES[dtype]


def _required(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required field: {key}")
    return d[key]


def _as_bool(d: dict[str, Any], key: str, default: bool) -> bool:
    return bool(d[key]) if key in d else default


@dataclass(frozen=True)
class LayerSpec:
    kind: str
    weight_bytes: int
    active_flops_per_token: int
    kv_bytes_per_token: int = 0
    num_experts: int = 0
    experts_active: int = 0
    expert_bytes: int = 0
    expert_flops_per_token: int = 0
    shared_expert_bytes: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"dense", "attention", "moe"}:
            raise ValueError(f"unsupported layer kind: {self.kind}")
        for name in (
            "weight_bytes",
            "active_flops_per_token",
            "kv_bytes_per_token",
            "num_experts",
            "experts_active",
            "expert_bytes",
            "expert_flops_per_token",
            "shared_expert_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.kind != "moe" and (self.num_experts or self.experts_active):
            raise ValueError("non-moe layers cannot declare experts")
        if self.kind == "moe":
            if self.num_experts <= 0 or self.experts_active <= 0:
                raise ValueError("moe layers require num_experts and experts_active")
            if self.experts_active > self.num_experts:
                raise ValueError("experts_active cannot exceed num_experts")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerSpec":
        return cls(
            kind=str(_required(d, "kind")),
            weight_bytes=int(_required(d, "weight_bytes")),
            active_flops_per_token=int(_required(d, "active_flops_per_token")),
            kv_bytes_per_token=int(d.get("kv_bytes_per_token", 0)),
            num_experts=int(d.get("num_experts", 0)),
            experts_active=int(d.get("experts_active", 0)),
            expert_bytes=int(d.get("expert_bytes", 0)),
            expert_flops_per_token=int(d.get("expert_flops_per_token", 0)),
            shared_expert_bytes=int(d.get("shared_expert_bytes", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "weight_bytes": self.weight_bytes,
            "active_flops_per_token": self.active_flops_per_token,
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }
        if self.kind == "moe":
            data.update(
                {
                    "num_experts": self.num_experts,
                    "experts_active": self.experts_active,
                    "expert_bytes": self.expert_bytes,
                    "expert_flops_per_token": self.expert_flops_per_token,
                    "shared_expert_bytes": self.shared_expert_bytes,
                }
            )
        return data

    @property
    def resident_weight_bytes(self) -> int:
        return (
            self.weight_bytes
            + self.shared_expert_bytes
            + self.num_experts * self.expert_bytes
        )

    @property
    def base_weight_bytes(self) -> int:
        return self.weight_bytes + self.shared_expert_bytes

    @property
    def resident_flops_per_token(self) -> int:
        return (
            self.active_flops_per_token
            + self.experts_active * self.expert_flops_per_token
        )

    @property
    def base_flops_per_token(self) -> int:
        return self.active_flops_per_token


@dataclass(frozen=True)
class ModelSpec:
    hidden_dim: int
    num_layers: int
    layers: tuple[LayerSpec, ...]
    dtype_weight: str
    dtype_activation: str
    expert_traces: tuple["ExpertTrace", ...] = ()

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_layers != len(self.layers):
            raise ValueError("num_layers must match layers length")
        activation_nbytes(self.dtype_activation)
        if self.dtype_weight not in DTYPE_BYTES:
            raise ValueError(f"unsupported weight dtype: {self.dtype_weight}")
        for trace in self.expert_traces:
            if trace.layer_id < 0 or trace.layer_id >= self.num_layers:
                raise ValueError("expert trace layer_id out of range")
            layer = self.layers[trace.layer_id]
            if layer.kind != "moe":
                raise ValueError("expert traces can only reference moe layers")
            if trace.expert_id < 0 or trace.expert_id >= layer.num_experts:
                raise ValueError("expert trace expert_id out of range")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        layers = tuple(LayerSpec.from_dict(x) for x in _required(d, "layers"))
        return cls(
            hidden_dim=int(_required(d, "hidden_dim")),
            num_layers=int(d.get("num_layers", len(layers))),
            layers=layers,
            dtype_weight=str(_required(d, "dtype_weight")),
            dtype_activation=str(_required(d, "dtype_activation")),
            expert_traces=tuple(
                ExpertTrace.from_dict(x) for x in d.get("expert_traces", [])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "layers": [x.to_dict() for x in self.layers],
            "dtype_weight": self.dtype_weight,
            "dtype_activation": self.dtype_activation,
        }
        if self.expert_traces:
            data["expert_traces"] = [x.to_dict() for x in self.expert_traces]
        return data

    @property
    def resident_weight_bytes(self) -> int:
        return sum(layer.resident_weight_bytes for layer in self.layers)


@dataclass(frozen=True)
class ExpertTrace:
    layer_id: int
    expert_id: int
    hit_rate_prefill: float
    hit_rate_decode: float
    coactivation: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0")
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")
        for name in ("hit_rate_prefill", "hit_rate_decode"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be 0..1")
        for expert_id, rate in self.coactivation:
            if expert_id < 0:
                raise ValueError("coactivation expert_id must be >= 0")
            if not (0.0 <= rate <= 1.0):
                raise ValueError("coactivation rate must be 0..1")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExpertTrace":
        return cls(
            layer_id=int(_required(d, "layer_id")),
            expert_id=int(_required(d, "expert_id")),
            hit_rate_prefill=float(_required(d, "hit_rate_prefill")),
            hit_rate_decode=float(_required(d, "hit_rate_decode")),
            coactivation=tuple((int(a), float(b)) for a, b in d.get("coactivation", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "hit_rate_prefill": self.hit_rate_prefill,
            "hit_rate_decode": self.hit_rate_decode,
        }
        if self.coactivation:
            data["coactivation"] = [list(x) for x in self.coactivation]
        return data


@dataclass(frozen=True)
class Node:
    id: str
    vendor: str
    runtime: str
    mem_free_bytes: int
    compute_class: float
    mem_bandwidth_bytes_s: float
    reliability: float = 1.0
    supports_stage: bool = True
    supports_expert_worker: bool = False
    supports_kv: bool = True
    supported_dtypes: tuple[str, ...] = ("fp16",)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("node id cannot be empty")
        if self.vendor not in {"nvidia", "apple", "amd", "cpu"}:
            raise ValueError(f"unsupported vendor: {self.vendor}")
        if self.mem_free_bytes <= 0:
            raise ValueError("mem_free_bytes must be > 0")
        if self.compute_class <= 0:
            raise ValueError("compute_class must be > 0")
        if self.mem_bandwidth_bytes_s <= 0:
            raise ValueError("mem_bandwidth_bytes_s must be > 0")
        if not (0 <= self.reliability <= 1):
            raise ValueError("reliability must be 0..1")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        return cls(
            id=str(_required(d, "id")),
            vendor=str(_required(d, "vendor")),
            runtime=str(_required(d, "runtime")),
            mem_free_bytes=int(_required(d, "mem_free_bytes")),
            compute_class=float(_required(d, "compute_class")),
            mem_bandwidth_bytes_s=float(_required(d, "mem_bandwidth_bytes_s")),
            reliability=float(d.get("reliability", 1.0)),
            supports_stage=_as_bool(d, "supports_stage", True),
            supports_expert_worker=_as_bool(d, "supports_expert_worker", False),
            supports_kv=_as_bool(d, "supports_kv", True),
            supported_dtypes=tuple(str(x) for x in d.get("supported_dtypes", ["fp16"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vendor": self.vendor,
            "runtime": self.runtime,
            "mem_free_bytes": self.mem_free_bytes,
            "compute_class": self.compute_class,
            "mem_bandwidth_bytes_s": self.mem_bandwidth_bytes_s,
            "reliability": self.reliability,
            "supports_stage": self.supports_stage,
            "supports_expert_worker": self.supports_expert_worker,
            "supports_kv": self.supports_kv,
            "supported_dtypes": list(self.supported_dtypes),
        }


@dataclass(frozen=True)
class Link:
    a: str
    b: str
    bandwidth_bytes_s: float
    latency_s: float

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("link endpoints must be different")
        if self.bandwidth_bytes_s <= 0:
            raise ValueError("bandwidth_bytes_s must be > 0")
        if self.latency_s < 0:
            raise ValueError("latency_s must be >= 0")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Link":
        return cls(
            a=str(_required(d, "a")),
            b=str(_required(d, "b")),
            bandwidth_bytes_s=float(_required(d, "bandwidth_bytes_s")),
            latency_s=float(_required(d, "latency_s")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "bandwidth_bytes_s": self.bandwidth_bytes_s,
            "latency_s": self.latency_s,
        }

    def connects(self, a: str, b: str) -> bool:
        return (self.a == a and self.b == b) or (self.a == b and self.b == a)


@dataclass(frozen=True)
class Inventory:
    nodes: tuple[Node, ...]
    links: tuple[Link, ...] = ()

    def __post_init__(self) -> None:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        known = set(ids)
        for link in self.links:
            if link.a not in known or link.b not in known:
                raise ValueError(f"link references unknown node: {link.a}-{link.b}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Inventory":
        return cls(
            nodes=tuple(Node.from_dict(x) for x in _required(d, "nodes")),
            links=tuple(Link.from_dict(x) for x in d.get("links", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [x.to_dict() for x in self.nodes],
            "links": [x.to_dict() for x in self.links],
        }

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def best_link(self, a: str, b: str) -> Link | None:
        matches = [link for link in self.links if link.connects(a, b)]
        if not matches:
            return None
        return max(matches, key=lambda link: (link.bandwidth_bytes_s, -link.latency_s))


@dataclass(frozen=True)
class Target:
    concurrency: int
    prompt_len: int
    gen_len: int
    objective: str = "max_throughput"
    remote_expert_wait_slo_s: float | None = None
    memory_reserve_fraction: float = 0.05
    fragmentation_margin_fraction: float = 0.05
    routing_metadata_bytes_per_token: float = 16.0
    temp_buffer_fraction: float = 0.05
    runtime_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if self.prompt_len <= 0:
            raise ValueError("prompt_len must be > 0")
        if self.gen_len <= 0:
            raise ValueError("gen_len must be > 0")
        if self.objective not in {"max_throughput", "min_latency", "balanced"}:
            raise ValueError(f"unsupported objective: {self.objective}")
        for name in (
            "memory_reserve_fraction",
            "fragmentation_margin_fraction",
            "routing_metadata_bytes_per_token",
            "temp_buffer_fraction",
            "runtime_reserve_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Target":
        return cls(
            concurrency=int(_required(d, "concurrency")),
            prompt_len=int(_required(d, "prompt_len")),
            gen_len=int(_required(d, "gen_len")),
            objective=str(d.get("objective", "max_throughput")),
            remote_expert_wait_slo_s=(
                float(d["remote_expert_wait_slo_s"])
                if d.get("remote_expert_wait_slo_s") is not None
                else None
            ),
            memory_reserve_fraction=float(d.get("memory_reserve_fraction", 0.05)),
            fragmentation_margin_fraction=float(
                d.get("fragmentation_margin_fraction", 0.05)
            ),
            routing_metadata_bytes_per_token=float(
                d.get("routing_metadata_bytes_per_token", 16.0)
            ),
            temp_buffer_fraction=float(d.get("temp_buffer_fraction", 0.05)),
            runtime_reserve_bytes=int(d.get("runtime_reserve_bytes", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "concurrency": self.concurrency,
            "prompt_len": self.prompt_len,
            "gen_len": self.gen_len,
            "objective": self.objective,
        }
        if self.remote_expert_wait_slo_s is not None:
            data["remote_expert_wait_slo_s"] = self.remote_expert_wait_slo_s
        if self.memory_reserve_fraction != 0.05:
            data["memory_reserve_fraction"] = self.memory_reserve_fraction
        if self.fragmentation_margin_fraction != 0.05:
            data["fragmentation_margin_fraction"] = self.fragmentation_margin_fraction
        if self.routing_metadata_bytes_per_token != 16.0:
            data["routing_metadata_bytes_per_token"] = (
                self.routing_metadata_bytes_per_token
            )
        if self.temp_buffer_fraction != 0.05:
            data["temp_buffer_fraction"] = self.temp_buffer_fraction
        if self.runtime_reserve_bytes:
            data["runtime_reserve_bytes"] = self.runtime_reserve_bytes
        return data


@dataclass(frozen=True)
class Stage:
    index: int
    layers: tuple[int, ...]
    replicas: tuple[str, ...]
    mode: str
    expert_hosts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "layers": list(self.layers),
            "replicas": list(self.replicas),
            "mode": self.mode,
            "expert_hosts": list(self.expert_hosts),
        }


@dataclass(frozen=True)
class BoundaryLink:
    stage_i: int
    stage_j: int
    link: Link

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_i": self.stage_i,
            "stage_j": self.stage_j,
            "link": self.link.to_dict(),
        }


@dataclass(frozen=True)
class ExpertPlacement:
    layer_id: int
    expert_id: int
    node_id: str
    role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "node_id": self.node_id,
            "role": self.role,
        }


@dataclass(frozen=True)
class PlacementExplanation:
    node_id: str
    decision: str
    reason: str
    stage_index: int | None = None
    layers: tuple[int, ...] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "node_id": self.node_id,
            "decision": self.decision,
            "reason": self.reason,
            "metrics": self.metrics,
        }
        if self.stage_index is not None:
            data["stage_index"] = self.stage_index
        if self.layers:
            data["layers"] = list(self.layers)
        return data


@dataclass(frozen=True)
class Predicted:
    throughput_tok_s: float
    ttft_s: float
    per_request_latency_s: float
    remote_expert_wait_s_per_token: float
    remote_expert_hit_rate_decode: float
    bottleneck_stage: int
    bubble_fraction: float
    stage_effective_times_s: tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "throughput_tok_s": self.throughput_tok_s,
            "ttft_s": self.ttft_s,
            "per_request_latency_s": self.per_request_latency_s,
            "remote_expert_wait_s_per_token": self.remote_expert_wait_s_per_token,
            "remote_expert_hit_rate_decode": self.remote_expert_hit_rate_decode,
            "bottleneck_stage": self.bottleneck_stage,
            "bubble_fraction": self.bubble_fraction,
            "stage_effective_times_s": list(self.stage_effective_times_s),
        }


@dataclass(frozen=True)
class PlacementPlan:
    stages: tuple[Stage, ...]
    boundary_links: tuple[BoundaryLink, ...]
    expert_placement: tuple[ExpertPlacement, ...]
    predicted: Predicted | None
    feasible: bool
    infeasible_reason: str | None = None
    explanations: tuple[PlacementExplanation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "boundary_links": [link.to_dict() for link in self.boundary_links],
            "expert_placement": [x.to_dict() for x in self.expert_placement],
            "predicted": self.predicted.to_dict() if self.predicted else None,
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "explanations": [x.to_dict() for x in self.explanations],
        }
