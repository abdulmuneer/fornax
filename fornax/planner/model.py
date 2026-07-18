from __future__ import annotations

import math
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

MEASUREMENT_STATUSES = {"measured", "estimated", "uncalibrated", "unsupported"}
CONFIDENCE_LEVELS = {"high", "medium", "low", "unknown"}
AUTHORITY_MODES = {"exploratory", "deployment"}


def activation_nbytes(dtype: str) -> float:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported activation dtype: {dtype}")
    return DTYPE_BYTES[dtype]


def _required(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required field: {key}")
    return d[key]


def _finite_float(value: Any, field_name: str) -> float:
    """Parse a JSON numeric field without accepting booleans or NaN/Infinity."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _integer(value: Any, field_name: str) -> int:
    """Parse an integral JSON field without bool coercion or truncation."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not boolean")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        if not value.is_integer():
            raise ValueError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return parsed


def _validate_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, not boolean")


def _validate_finite_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _validate_json_finite(value: Any, path: str = "plan") -> None:
    """Fail instead of producing Python's non-standard NaN/Infinity JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite numeric value")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_json_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_finite(child, f"{path}[{index}]")


def _as_bool(d: dict[str, Any], key: str, default: bool) -> bool:
    return bool(d[key]) if key in d else default


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field_name} values must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} values must be unique")
    return result


def normalize_authority_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "authoritative":
        normalized = "deployment"
    if normalized not in AUTHORITY_MODES:
        raise ValueError(f"unsupported authority mode: {value}")
    return normalized


@dataclass(frozen=True)
class MeasurementProvenance:
    """Attribution and uncertainty for one planner input or calibration.

    ``expected_relative_error`` is a fraction (``0.20`` means +/-20%).  It is
    deliberately optional: a measured value without an error study remains
    useful for exploration, but cannot authorize deployment.
    """

    status: str = "uncalibrated"
    source_id: str | None = None
    confidence: str = "unknown"
    expected_relative_error: float | None = None

    def __post_init__(self) -> None:
        if self.status not in MEASUREMENT_STATUSES:
            raise ValueError(f"unsupported measurement status: {self.status}")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"unsupported confidence level: {self.confidence}")
        if self.source_id is not None and not self.source_id.strip():
            raise ValueError("measurement source_id must be non-empty when provided")
        if self.status != "uncalibrated" and self.source_id is None:
            raise ValueError(f"{self.status} measurement requires source_id")
        error = self.expected_relative_error
        if error is not None:
            _validate_finite_number(error, "expected_relative_error")
            if error < 0:
                raise ValueError("expected_relative_error must be finite and >= 0")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "MeasurementProvenance":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("measurement provenance must be an object")
        raw_error = value.get("expected_relative_error")
        return cls(
            status=str(value.get("status", "uncalibrated")),
            source_id=(
                str(value["source_id"]) if value.get("source_id") is not None else None
            ),
            confidence=str(value.get("confidence", "unknown")),
            expected_relative_error=(
                _finite_float(raw_error, "expected_relative_error")
                if raw_error is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "expected_relative_error": self.expected_relative_error,
        }


def _measurement_provenance_map(
    value: Any, field_name: str = "measurement_provenance"
) -> tuple[tuple[str, MeasurementProvenance], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    rows: list[tuple[str, MeasurementProvenance]] = []
    for raw_name, raw_provenance in value.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        rows.append((name, MeasurementProvenance.from_dict(raw_provenance)))
    return tuple(sorted(rows, key=lambda row: row[0]))


def _measurement_provenance_dict(
    value: tuple[tuple[str, MeasurementProvenance], ...]
) -> dict[str, Any]:
    return {name: provenance.to_dict() for name, provenance in value}


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
            _validate_integer(getattr(self, name), name)
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
            weight_bytes=_integer(_required(d, "weight_bytes"), "weight_bytes"),
            active_flops_per_token=_integer(
                _required(d, "active_flops_per_token"),
                "active_flops_per_token",
            ),
            kv_bytes_per_token=_integer(
                d.get("kv_bytes_per_token", 0), "kv_bytes_per_token"
            ),
            num_experts=_integer(d.get("num_experts", 0), "num_experts"),
            experts_active=_integer(
                d.get("experts_active", 0), "experts_active"
            ),
            expert_bytes=_integer(d.get("expert_bytes", 0), "expert_bytes"),
            expert_flops_per_token=_integer(
                d.get("expert_flops_per_token", 0), "expert_flops_per_token"
            ),
            shared_expert_bytes=_integer(
                d.get("shared_expert_bytes", 0), "shared_expert_bytes"
            ),
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
    source_id: str | None = None
    quantization_source_id: str | None = None
    expert_trace_source_id: str | None = None

    def __post_init__(self) -> None:
        _validate_integer(self.hidden_dim, "hidden_dim")
        _validate_integer(self.num_layers, "num_layers")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_layers != len(self.layers):
            raise ValueError("num_layers must match layers length")
        activation_nbytes(self.dtype_activation)
        if self.dtype_weight not in DTYPE_BYTES:
            raise ValueError(f"unsupported weight dtype: {self.dtype_weight}")
        for field_name in (
            "source_id",
            "quantization_source_id",
            "expert_trace_source_id",
        ):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when provided")
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
            hidden_dim=_integer(_required(d, "hidden_dim"), "hidden_dim"),
            num_layers=_integer(d.get("num_layers", len(layers)), "num_layers"),
            layers=layers,
            dtype_weight=str(_required(d, "dtype_weight")),
            dtype_activation=str(_required(d, "dtype_activation")),
            expert_traces=tuple(
                ExpertTrace.from_dict(x) for x in d.get("expert_traces", [])
            ),
            source_id=(str(d["source_id"]) if d.get("source_id") is not None else None),
            quantization_source_id=(
                str(d["quantization_source_id"])
                if d.get("quantization_source_id") is not None
                else None
            ),
            expert_trace_source_id=(
                str(d["expert_trace_source_id"])
                if d.get("expert_trace_source_id") is not None
                else None
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
        if self.source_id is not None:
            data["source_id"] = self.source_id
        if self.quantization_source_id is not None:
            data["quantization_source_id"] = self.quantization_source_id
        if self.expert_trace_source_id is not None:
            data["expert_trace_source_id"] = self.expert_trace_source_id
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
        _validate_integer(self.layer_id, "layer_id")
        _validate_integer(self.expert_id, "expert_id")
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0")
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")
        for name in ("hit_rate_prefill", "hit_rate_decode"):
            value = getattr(self, name)
            _validate_finite_number(value, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be 0..1")
        for expert_id, rate in self.coactivation:
            _validate_integer(expert_id, "coactivation expert_id")
            _validate_finite_number(rate, "coactivation rate")
            if expert_id < 0:
                raise ValueError("coactivation expert_id must be >= 0")
            if not (0.0 <= rate <= 1.0):
                raise ValueError("coactivation rate must be 0..1")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExpertTrace":
        return cls(
            layer_id=_integer(_required(d, "layer_id"), "layer_id"),
            expert_id=_integer(_required(d, "expert_id"), "expert_id"),
            hit_rate_prefill=_finite_float(
                _required(d, "hit_rate_prefill"), "hit_rate_prefill"
            ),
            hit_rate_decode=_finite_float(
                _required(d, "hit_rate_decode"), "hit_rate_decode"
            ),
            coactivation=tuple(
                (
                    _integer(a, "coactivation expert_id"),
                    _finite_float(b, "coactivation rate"),
                )
                for a, b in d.get("coactivation", [])
            ),
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
    build_id: str | None = None
    capabilities_complete: bool = False
    supported_operations: tuple[str, ...] = ()
    supported_quantizations: tuple[str, ...] = ()
    capability_source_id: str | None = None
    measurement_provenance: tuple[tuple[str, MeasurementProvenance], ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("node id cannot be empty")
        if not self.runtime:
            raise ValueError("node runtime cannot be empty")
        if self.vendor not in {"nvidia", "apple", "amd", "cpu"}:
            raise ValueError(f"unsupported vendor: {self.vendor}")
        _validate_integer(self.mem_free_bytes, "mem_free_bytes")
        _validate_finite_number(self.compute_class, "compute_class")
        _validate_finite_number(
            self.mem_bandwidth_bytes_s, "mem_bandwidth_bytes_s"
        )
        _validate_finite_number(self.reliability, "reliability")
        if self.mem_free_bytes <= 0:
            raise ValueError("mem_free_bytes must be > 0")
        if self.compute_class <= 0:
            raise ValueError("compute_class must be > 0")
        if self.mem_bandwidth_bytes_s <= 0:
            raise ValueError("mem_bandwidth_bytes_s must be > 0")
        if not (0 <= self.reliability <= 1):
            raise ValueError("reliability must be 0..1")
        for field_name, values in (
            ("supported_dtypes", self.supported_dtypes),
            ("supported_operations", self.supported_operations),
            ("supported_quantizations", self.supported_quantizations),
        ):
            if any(not value for value in values):
                raise ValueError(f"{field_name} values must be non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} values must be unique")
        if self.build_id is not None and not self.build_id.strip():
            raise ValueError("build_id must be non-empty when provided")
        if self.capability_source_id is not None and not self.capability_source_id.strip():
            raise ValueError("capability_source_id must be non-empty when provided")
        provenance_fields = [name for name, _ in self.measurement_provenance]
        if len(provenance_fields) != len(set(provenance_fields)):
            raise ValueError("measurement_provenance fields must be unique")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        return cls(
            id=str(_required(d, "id")),
            vendor=str(_required(d, "vendor")),
            runtime=str(_required(d, "runtime")),
            mem_free_bytes=_integer(
                _required(d, "mem_free_bytes"), "mem_free_bytes"
            ),
            compute_class=_finite_float(
                _required(d, "compute_class"), "compute_class"
            ),
            mem_bandwidth_bytes_s=_finite_float(
                _required(d, "mem_bandwidth_bytes_s"),
                "mem_bandwidth_bytes_s",
            ),
            reliability=_finite_float(d.get("reliability", 1.0), "reliability"),
            supports_stage=_as_bool(d, "supports_stage", True),
            supports_expert_worker=_as_bool(d, "supports_expert_worker", False),
            supports_kv=_as_bool(d, "supports_kv", True),
            supported_dtypes=tuple(str(x) for x in d.get("supported_dtypes", ["fp16"])),
            build_id=(str(d["build_id"]) if d.get("build_id") is not None else None),
            capabilities_complete=_as_bool(d, "capabilities_complete", False),
            supported_operations=_string_tuple(
                d.get("supported_operations"), "supported_operations"
            ),
            supported_quantizations=_string_tuple(
                d.get("supported_quantizations"), "supported_quantizations"
            ),
            capability_source_id=(
                str(d["capability_source_id"])
                if d.get("capability_source_id") is not None
                else None
            ),
            measurement_provenance=_measurement_provenance_map(
                d.get("measurement_provenance")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        if self.build_id is not None:
            data["build_id"] = self.build_id
        if self.capabilities_complete:
            data["capabilities_complete"] = True
        if self.supported_operations:
            data["supported_operations"] = list(self.supported_operations)
        if self.supported_quantizations:
            data["supported_quantizations"] = list(self.supported_quantizations)
        if self.capability_source_id is not None:
            data["capability_source_id"] = self.capability_source_id
        if self.measurement_provenance:
            data["measurement_provenance"] = _measurement_provenance_dict(
                self.measurement_provenance
            )
        return data

    def provenance_for(self, field_name: str) -> MeasurementProvenance:
        for name, provenance in self.measurement_provenance:
            if name == field_name:
                return provenance
        return MeasurementProvenance()


@dataclass(frozen=True)
class Link:
    a: str
    b: str
    bandwidth_bytes_s: float
    latency_s: float
    measurement_provenance: tuple[tuple[str, MeasurementProvenance], ...] = ()

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("link endpoints must be different")
        _validate_finite_number(self.bandwidth_bytes_s, "bandwidth_bytes_s")
        _validate_finite_number(self.latency_s, "latency_s")
        if self.bandwidth_bytes_s <= 0:
            raise ValueError("bandwidth_bytes_s must be > 0")
        if self.latency_s < 0:
            raise ValueError("latency_s must be >= 0")
        provenance_fields = [name for name, _ in self.measurement_provenance]
        if len(provenance_fields) != len(set(provenance_fields)):
            raise ValueError("measurement_provenance fields must be unique")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Link":
        return cls(
            a=str(_required(d, "a")),
            b=str(_required(d, "b")),
            bandwidth_bytes_s=_finite_float(
                _required(d, "bandwidth_bytes_s"), "bandwidth_bytes_s"
            ),
            latency_s=_finite_float(_required(d, "latency_s"), "latency_s"),
            measurement_provenance=_measurement_provenance_map(
                d.get("measurement_provenance")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "a": self.a,
            "b": self.b,
            "bandwidth_bytes_s": self.bandwidth_bytes_s,
            "latency_s": self.latency_s,
        }
        if self.measurement_provenance:
            data["measurement_provenance"] = _measurement_provenance_dict(
                self.measurement_provenance
            )
        return data

    def provenance_for(self, field_name: str) -> MeasurementProvenance:
        for name, provenance in self.measurement_provenance:
            if name == field_name:
                return provenance
        return MeasurementProvenance()

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
    authority_mode: str = "exploratory"
    required_runtime: str | None = None
    accepted_build_ids: tuple[str, ...] = ()
    required_operations: tuple[str, ...] = ()
    prediction_calibration: MeasurementProvenance = field(
        default_factory=MeasurementProvenance
    )
    max_expected_relative_error: float = 0.20

    def __post_init__(self) -> None:
        for name in ("concurrency", "prompt_len", "gen_len", "runtime_reserve_bytes"):
            _validate_integer(getattr(self, name), name)
        for name in (
            "memory_reserve_fraction",
            "fragmentation_margin_fraction",
            "routing_metadata_bytes_per_token",
            "temp_buffer_fraction",
            "max_expected_relative_error",
        ):
            _validate_finite_number(getattr(self, name), name)
        if self.remote_expert_wait_slo_s is not None:
            _validate_finite_number(
                self.remote_expert_wait_slo_s, "remote_expert_wait_slo_s"
            )
            if self.remote_expert_wait_slo_s < 0:
                raise ValueError("remote_expert_wait_slo_s must be >= 0")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if self.prompt_len <= 0:
            raise ValueError("prompt_len must be > 0")
        if self.gen_len <= 0:
            raise ValueError("gen_len must be > 0")
        if self.objective not in {"max_throughput", "min_latency", "balanced"}:
            raise ValueError(f"unsupported objective: {self.objective}")
        object.__setattr__(self, "authority_mode", normalize_authority_mode(self.authority_mode))
        if self.required_runtime is not None and not self.required_runtime.strip():
            raise ValueError("required_runtime must be non-empty when provided")
        for field_name, values in (
            ("accepted_build_ids", self.accepted_build_ids),
            ("required_operations", self.required_operations),
        ):
            if any(not value for value in values):
                raise ValueError(f"{field_name} values must be non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} values must be unique")
        if self.max_expected_relative_error < 0:
            raise ValueError("max_expected_relative_error must be finite and >= 0")
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
        raw_max_error = d.get("max_expected_relative_error", 0.20)
        return cls(
            concurrency=_integer(_required(d, "concurrency"), "concurrency"),
            prompt_len=_integer(_required(d, "prompt_len"), "prompt_len"),
            gen_len=_integer(_required(d, "gen_len"), "gen_len"),
            objective=str(d.get("objective", "max_throughput")),
            remote_expert_wait_slo_s=(
                _finite_float(
                    d["remote_expert_wait_slo_s"], "remote_expert_wait_slo_s"
                )
                if d.get("remote_expert_wait_slo_s") is not None
                else None
            ),
            memory_reserve_fraction=_finite_float(
                d.get("memory_reserve_fraction", 0.05), "memory_reserve_fraction"
            ),
            fragmentation_margin_fraction=_finite_float(
                d.get("fragmentation_margin_fraction", 0.05),
                "fragmentation_margin_fraction",
            ),
            routing_metadata_bytes_per_token=_finite_float(
                d.get("routing_metadata_bytes_per_token", 16.0),
                "routing_metadata_bytes_per_token",
            ),
            temp_buffer_fraction=_finite_float(
                d.get("temp_buffer_fraction", 0.05), "temp_buffer_fraction"
            ),
            runtime_reserve_bytes=_integer(
                d.get("runtime_reserve_bytes", 0), "runtime_reserve_bytes"
            ),
            authority_mode=str(d.get("authority_mode", "exploratory")),
            required_runtime=(
                str(d["required_runtime"])
                if d.get("required_runtime") is not None
                else None
            ),
            accepted_build_ids=_string_tuple(
                d.get("accepted_build_ids"), "accepted_build_ids"
            ),
            required_operations=_string_tuple(
                d.get("required_operations"), "required_operations"
            ),
            prediction_calibration=MeasurementProvenance.from_dict(
                d.get("prediction_calibration")
            ),
            max_expected_relative_error=_finite_float(
                raw_max_error, "max_expected_relative_error"
            ),
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
        if self.authority_mode != "exploratory":
            data["authority_mode"] = self.authority_mode
        if self.required_runtime is not None:
            data["required_runtime"] = self.required_runtime
        if self.accepted_build_ids:
            data["accepted_build_ids"] = list(self.accepted_build_ids)
        if self.required_operations:
            data["required_operations"] = list(self.required_operations)
        if self.prediction_calibration != MeasurementProvenance():
            data["prediction_calibration"] = self.prediction_calibration.to_dict()
        if self.max_expected_relative_error != 0.20:
            data["max_expected_relative_error"] = self.max_expected_relative_error
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
    prediction_provenance: MeasurementProvenance = field(
        default_factory=MeasurementProvenance
    )

    def __post_init__(self) -> None:
        for name in (
            "throughput_tok_s",
            "ttft_s",
            "per_request_latency_s",
            "remote_expert_wait_s_per_token",
            "remote_expert_hit_rate_decode",
            "bubble_fraction",
        ):
            _validate_finite_number(getattr(self, name), name)
        _validate_integer(self.bottleneck_stage, "bottleneck_stage")
        for index, value in enumerate(self.stage_effective_times_s):
            _validate_finite_number(value, f"stage_effective_times_s[{index}]")

    def _interval(self, value: float) -> list[float] | None:
        error = self.prediction_provenance.expected_relative_error
        if error is None:
            return None
        return [max(0.0, value * (1.0 - error)), value * (1.0 + error)]

    def to_dict(self) -> dict[str, Any]:
        intervals = {
            "throughput_tok_s": self._interval(self.throughput_tok_s),
        }
        return {
            "throughput_tok_s": self.throughput_tok_s,
            "ttft_s": self.ttft_s,
            "per_request_latency_s": self.per_request_latency_s,
            "remote_expert_wait_s_per_token": self.remote_expert_wait_s_per_token,
            "remote_expert_hit_rate_decode": self.remote_expert_hit_rate_decode,
            "bottleneck_stage": self.bottleneck_stage,
            "bubble_fraction": self.bubble_fraction,
            "stage_effective_times_s": list(self.stage_effective_times_s),
            "prediction_provenance": self.prediction_provenance.to_dict(),
            "prediction_intervals": (
                intervals
                if self.prediction_provenance.expected_relative_error is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PlanAuthority:
    requested_mode: str = "exploratory"
    status: str = "exploratory"
    deployment_authorized: bool = False
    confidence: str = "unknown"
    prediction_expected_relative_error: float | None = None
    input_max_expected_relative_error: float | None = None
    source_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ("exploratory mode requested",)
    evidence_registry_sha256: str | None = None

    def __post_init__(self) -> None:
        if normalize_authority_mode(self.requested_mode) != self.requested_mode:
            raise ValueError("requested_mode must be canonical")
        if self.status not in {"exploratory", "deployment_authoritative", "rejected"}:
            raise ValueError(f"unsupported plan authority status: {self.status}")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"unsupported confidence level: {self.confidence}")
        if self.deployment_authorized != (self.status == "deployment_authoritative"):
            raise ValueError("deployment_authorized must match authority status")
        for name in (
            "prediction_expected_relative_error",
            "input_max_expected_relative_error",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_finite_number(value, name)
                if value < 0:
                    raise ValueError(f"{name} must be >= 0")
        if self.evidence_registry_sha256 is not None:
            prefix = "sha256:"
            digest = self.evidence_registry_sha256.removeprefix(prefix)
            if (
                not self.evidence_registry_sha256.startswith(prefix)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    "evidence_registry_sha256 must be sha256:<64 lowercase hex>"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "status": self.status,
            "deployment_authorized": self.deployment_authorized,
            "confidence": self.confidence,
            "prediction_expected_relative_error": (
                self.prediction_expected_relative_error
            ),
            "input_max_expected_relative_error": (
                self.input_max_expected_relative_error
            ),
            "source_ids": list(self.source_ids),
            "evidence_registry_sha256": self.evidence_registry_sha256,
            "reasons": list(self.reasons),
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
    authority: PlanAuthority = field(default_factory=PlanAuthority)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "stages": [stage.to_dict() for stage in self.stages],
            "boundary_links": [link.to_dict() for link in self.boundary_links],
            "expert_placement": [x.to_dict() for x in self.expert_placement],
            "predicted": self.predicted.to_dict() if self.predicted else None,
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "explanations": [x.to_dict() for x in self.explanations],
            "authority": self.authority.to_dict(),
        }
        _validate_json_finite(data)
        return data
