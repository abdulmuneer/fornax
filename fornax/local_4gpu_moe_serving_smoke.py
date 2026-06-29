from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .io import read_json, write_json
from .serving import simulate_serving_adapter, validate_serving_adapter_fixture


RECORD_KIND = "local-4gpu-moe-serving-smoke-bundle"
PROBE_KIND = "four-gpu-moe-serving-probe"
EVIDENCE_SCOPE = "same-host-4gpu-moe-serving-proxy"
TIER = "T3-same-host-4gpu-moe-serving-proxy"
DEFAULT_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
DTYPES = {"float32", "float16", "bfloat16"}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


def _positive_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def parse_cuda_devices(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        devices = [item.strip() for item in value.split(",") if item.strip()]
    else:
        devices = [str(item).strip() for item in value if str(item).strip()]
    if len(devices) != 4:
        raise ValueError("devices must contain exactly four CUDA devices")
    if len(set(devices)) != 4:
        raise ValueError("devices must be distinct")
    for device in devices:
        if not re.fullmatch(r"cuda:\d+", device):
            raise ValueError(f"device must be cuda:<index>: {device}")
    return devices


def _validate_config(
    *,
    iterations: int,
    warmup: int,
    token_count: int,
    hidden_dim: int,
    intermediate_dim: int,
    vocab_size: int,
    expert_count: int,
    top_k: int,
    tolerance: float,
    timeout_s: float,
) -> None:
    _positive_int("iterations", iterations)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    _positive_int("token_count", token_count)
    _positive_int("hidden_dim", hidden_dim)
    _positive_int("intermediate_dim", intermediate_dim)
    _positive_int("vocab_size", vocab_size)
    _positive_int("expert_count", expert_count)
    _positive_int("top_k", top_k)
    if top_k > expert_count:
        raise ValueError("top_k must be <= expert_count")
    if expert_count < 3:
        raise ValueError("expert_count must be at least 3 to cover three expert GPUs")
    _non_negative_number("tolerance", tolerance)
    _positive_number("timeout_s", timeout_s)


def _validation_entry(name: str, result: dict[str, Any], artifact: str) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(result.get("ok")),
        "artifact": artifact,
        "errors": list(result.get("errors", [])),
        "warnings": list(result.get("warnings", [])),
        "summary": result.get("summary", {}),
    }


def _external_torch_script() -> str:
    return r'''
import json
import os
import platform
import re
import sys
import time

iterations = int(sys.argv[1])
warmup = int(sys.argv[2])
token_count = int(sys.argv[3])
hidden_dim = int(sys.argv[4])
intermediate_dim = int(sys.argv[5])
vocab_size = int(sys.argv[6])
expert_count = int(sys.argv[7])
top_k = int(sys.argv[8])
devices = [item for item in sys.argv[9].split(",") if item]
dtype_name = sys.argv[10]
tolerance = float(sys.argv[11])
model_id = sys.argv[12]
request_id = sys.argv[13]
prompt_text = sys.argv[14]
probe_kind = "four-gpu-moe-serving-probe"
evidence_scope = "same-host-4gpu-moe-serving-proxy"
tier = "T3-same-host-4gpu-moe-serving-proxy"


def emit_unavailable(error):
    print(json.dumps({
        "version": 1,
        "probe_kind": probe_kind,
        "evidence_scope": evidence_scope,
        "tier": tier,
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.local_4gpu_moe_serving_smoke.torch_probe",
        "error": error,
        "environment": {"python_executable": sys.executable},
    }))

try:
    import torch
except Exception as exc:
    emit_unavailable(f"torch import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

if dtype_name == "float32":
    dtype = torch.float32
elif dtype_name == "float16":
    dtype = torch.float16
elif dtype_name == "bfloat16":
    dtype = torch.bfloat16
else:
    emit_unavailable(f"unsupported dtype: {dtype_name}")
    raise SystemExit(0)

if not torch.cuda.is_available():
    emit_unavailable("torch.cuda.is_available() is false")
    raise SystemExit(0)


def cuda_index(name):
    match = re.fullmatch(r"cuda:(\d+)", name)
    if not match:
        raise ValueError(f"device must be cuda:<index>: {name}")
    return int(match.group(1))

try:
    if len(devices) != 4:
        raise ValueError("exactly four CUDA devices are required")
    if len(set(devices)) != 4:
        raise ValueError("CUDA devices must be distinct")
    device_indices = [cuda_index(device) for device in devices]
    cuda_device_count = int(torch.cuda.device_count())
    for index in device_indices:
        if index >= cuda_device_count:
            raise ValueError(f"device index out of range for cuda_device_count={cuda_device_count}: cuda:{index}")
    gateway_device_name = devices[0]
    expert_device_names = devices[1:]
    gateway_device = torch.device(gateway_device_name)
    expert_devices = {name: torch.device(name) for name in expert_device_names}
except Exception as exc:
    emit_unavailable(f"device setup failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)


def route_trace():
    routes = []
    for token in range(token_count):
        expert_ids = []
        candidate = token % expert_count
        while len(expert_ids) < top_k:
            if candidate not in expert_ids:
                expert_ids.append(candidate)
            candidate = (candidate + 1) % expert_count
        raw = [float(top_k - rank) + 0.25 * ((token + rank) % 3) for rank in range(top_k)]
        total = sum(raw)
        routes.append({"token_index": token, "expert_ids": expert_ids, "topk_weights": [value / total for value in raw]})
    return routes

routes = route_trace()
expert_device_by_id = {expert_id: expert_device_names[expert_id % len(expert_device_names)] for expert_id in range(expert_count)}
expert_device_calls = {name: 0 for name in expert_device_names}
expert_calls_per_iteration = 0
for route in routes:
    for expert_id in route["expert_ids"]:
        expert_device_calls[expert_device_by_id[int(expert_id)]] += 1
        expert_calls_per_iteration += 1
expert_devices_used = [name for name, count in expert_device_calls.items() if count > 0]
all_expert_devices_used = set(expert_devices_used) == set(expert_device_names)
all_devices_used = all_expert_devices_used and gateway_device_name in devices
remote_bucket_count = len({int(expert_id) for route in routes for expert_id in route["expert_ids"]})


def make_inputs(device, target_dtype):
    tokens = torch.arange(token_count, device=device, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((tokens + 2) * (dims + 5), 31.0)
    return ((values - 15.0) / 31.0).to(dtype=target_dtype)


def up_weights(expert_id, device, target_dtype):
    hidden = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(1)
    intermediate = torch.arange(intermediate_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((expert_id + 3) * (hidden + 7) * (intermediate + 11), 37.0)
    return ((values - 18.0) / 37.0).to(dtype=target_dtype)


def down_weights(expert_id, device, target_dtype):
    intermediate = torch.arange(intermediate_dim, device=device, dtype=torch.float32).unsqueeze(1)
    hidden = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((expert_id + 5) * (intermediate + 13) * (hidden + 17), 41.0)
    return ((values - 20.0) / 41.0).to(dtype=target_dtype)


def logit_weights(device, target_dtype):
    hidden = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(1)
    vocab = torch.arange(vocab_size, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((hidden + 11) * (vocab + 13), 43.0)
    return ((values - 21.0) / 43.0).to(dtype=target_dtype)


def expert_outputs(x, expert_id, device, target_dtype):
    up = up_weights(expert_id, device, target_dtype)
    down = down_weights(expert_id, device, target_dtype)
    return torch.relu(x @ up) @ down


def run_reference(x):
    gathered = torch.zeros((token_count, hidden_dim), device=gateway_device, dtype=torch.float32)
    x_float = x.float()
    for token_index, route in enumerate(routes):
        for expert_id, weight in zip(route["expert_ids"], route["topk_weights"]):
            out = expert_outputs(x_float[token_index:token_index + 1], int(expert_id), gateway_device, torch.float32)
            gathered[token_index:token_index + 1] += float(weight) * out
    layer = x_float + gathered
    logits = layer @ logit_weights(gateway_device, torch.float32)
    return layer, logits, torch.argmax(logits, dim=1)


def run_split(x):
    gathered = torch.zeros((token_count, hidden_dim), device=gateway_device, dtype=torch.float32)
    for token_index, route in enumerate(routes):
        for expert_id, weight in zip(route["expert_ids"], route["topk_weights"]):
            expert_id = int(expert_id)
            expert_device_name = expert_device_by_id[expert_id]
            expert_device = expert_devices[expert_device_name]
            payload = x[token_index:token_index + 1].to(expert_device, non_blocking=False)
            out = expert_outputs(payload, expert_id, expert_device, dtype).to(gateway_device, non_blocking=False).float()
            gathered[token_index:token_index + 1] += float(weight) * out
    layer = x.float() + gathered
    logits = layer @ logit_weights(gateway_device, torch.float32)
    return layer, logits, torch.argmax(logits, dim=1)


def synchronize_all():
    for name in devices:
        torch.cuda.synchronize(torch.device(name))


def memory_snapshot(index):
    with torch.cuda.device(index):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            free_bytes, total_bytes = None, None
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "free_bytes": int(free_bytes) if free_bytes is not None else None,
            "total_bytes": int(total_bytes) if total_bytes is not None else int(torch.cuda.get_device_properties(index).total_memory),
        }

try:
    for index in device_indices:
        try:
            torch.cuda.reset_peak_memory_stats(index)
        except Exception:
            pass
    memory_before = {name: memory_snapshot(index) for name, index in zip(devices, device_indices)}
    with torch.no_grad():
        x = make_inputs(gateway_device, dtype)
        reference_layer, reference_logits, reference_tokens = run_reference(x)
        layer, logits, next_tokens = run_split(x)
        synchronize_all()
        max_layer_abs_error = float((layer.detach().cpu().float() - reference_layer.detach().cpu().float()).abs().max().item())
        max_logit_abs_error = float((logits.detach().cpu().float() - reference_logits.detach().cpu().float()).abs().max().item())
        next_tokens_match = bool(torch.equal(next_tokens.detach().cpu(), reference_tokens.detach().cpu()))
        correctness_passed = bool(next_tokens_match and max_layer_abs_error <= tolerance and max_logit_abs_error <= tolerance)
        for _ in range(warmup):
            run_split(x)
        synchronize_all()
        started = time.perf_counter()
        timed_layer = None
        timed_logits = None
        timed_tokens = None
        for _ in range(iterations):
            timed_layer, timed_logits, timed_tokens = run_split(x)
        synchronize_all()
        elapsed_s = time.perf_counter() - started
        timed_layer_cpu = timed_layer.detach().cpu().float() if timed_layer is not None else layer.detach().cpu().float()
        timed_logits_cpu = timed_logits.detach().cpu().float() if timed_logits is not None else logits.detach().cpu().float()
        timed_tokens_cpu = timed_tokens.detach().cpu().tolist() if timed_tokens is not None else next_tokens.detach().cpu().tolist()
        reference_layer_cpu = reference_layer.detach().cpu().float()
        reference_logits_cpu = reference_logits.detach().cpu().float()
        reference_tokens_cpu = reference_tokens.detach().cpu().tolist()
    memory_after = {name: memory_snapshot(index) for name, index in zip(devices, device_indices)}
except Exception as exc:
    emit_unavailable(f"4-GPU MoE serving probe failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

try:
    p2p = []
    for source_name, source_index in zip(devices, device_indices):
        for destination_name, destination_index in zip(devices, device_indices):
            if source_name == destination_name:
                continue
            try:
                access = bool(torch.cuda.can_device_access_peer(source_index, destination_index))
            except Exception:
                access = None
            p2p.append({"source": source_name, "destination": destination_name, "can_access_peer": access})
except Exception:
    p2p = []

element_size = int(torch.tensor([], dtype=dtype).element_size())
transfer_payload_bytes_per_iteration = int(expert_calls_per_iteration * hidden_dim * element_size * 2)
expert_calls = int(iterations * expert_calls_per_iteration)
remote_batches = int(iterations * remote_bucket_count)
layer_weights = (1.0 + torch.arange(token_count, dtype=torch.float32).unsqueeze(1) * 0.01) * (1.0 + torch.arange(hidden_dim, dtype=torch.float32).unsqueeze(0) * 0.0001)
logit_weights_for_checksum = (1.0 + torch.arange(token_count, dtype=torch.float32).unsqueeze(1) * 0.01) * (1.0 + torch.arange(vocab_size, dtype=torch.float32).unsqueeze(0) * 0.0001)
generated_text = " ".join(f"tok{int(token)}" for token in timed_tokens_cpu)
choice = {
    "index": 0,
    "message": {"role": "assistant", "content": generated_text},
    "finish_reason": "length",
}
print(json.dumps({
    "version": 1,
    "probe_kind": probe_kind,
    "evidence_scope": evidence_scope,
    "tier": tier,
    "measured": True,
    "accelerator_measured": True,
    "backend": "torch",
    "available": True,
    "source": "fornax.local_4gpu_moe_serving_smoke.torch_probe",
    "config": {
        "iterations": iterations,
        "warmup": warmup,
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "vocab_size": vocab_size,
        "expert_count": expert_count,
        "top_k": top_k,
        "devices": devices,
        "gateway_device": gateway_device_name,
        "expert_devices": expert_device_names,
        "logical_hosts": ["logical-host-0", "logical-host-1", "logical-host-2", "logical-host-3"],
        "dtype": dtype_name,
        "tolerance": tolerance,
        "model_id": model_id,
        "request_id": request_id,
        "prompt_text": prompt_text,
        "expert_device_calls_per_iteration": expert_device_calls,
        "expert_devices_used": expert_devices_used,
        "all_expert_devices_used": all_expert_devices_used,
        "all_devices_used": all_devices_used,
        "transfer_payload_bytes_per_iteration": transfer_payload_bytes_per_iteration,
    },
    "routing": {
        "routes": routes,
        "expert_placement": [
            {
                "expert_id": expert_id,
                "device": expert_device_by_id[expert_id],
                "logical_host": f"logical-host-{1 + (expert_id % len(expert_device_names))}",
                "local_to_gateway": False,
            }
            for expert_id in range(expert_count)
        ],
    },
    "serving": {
        "request": {
            "id": request_id,
            "model": model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": token_count,
            "stream": False,
        },
        "response": {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "model": model_id,
            "choices": [choice],
            "usage": {
                "prompt_tokens": token_count,
                "completion_tokens": len(timed_tokens_cpu),
                "total_tokens": token_count + len(timed_tokens_cpu),
            },
        },
        "generated_text": generated_text,
        "openai_compatible_shape": True,
        "live_http_endpoint": False,
    },
    "result": {
        "elapsed_s": elapsed_s,
        "elapsed_ns": int(elapsed_s * 1000000000),
        "tokens_processed": int(iterations * token_count),
        "expert_calls": expert_calls,
        "remote_expert_calls": expert_calls,
        "remote_batches": remote_batches,
        "transfer_payload_bytes": int(iterations * transfer_payload_bytes_per_iteration),
        "tokens_s": iterations * token_count / elapsed_s if elapsed_s > 0 else None,
        "expert_calls_s": expert_calls / elapsed_s if elapsed_s > 0 else None,
        "layer_checksum": float((timed_layer_cpu * layer_weights).sum().item()),
        "reference_layer_checksum": float((reference_layer_cpu * layer_weights).sum().item()),
        "logit_checksum": float((timed_logits_cpu * logit_weights_for_checksum).sum().item()),
        "reference_logit_checksum": float((reference_logits_cpu * logit_weights_for_checksum).sum().item()),
        "max_layer_abs_error": max_layer_abs_error,
        "max_logit_abs_error": max_logit_abs_error,
        "routing_match": True,
        "next_tokens": timed_tokens_cpu,
        "reference_next_tokens": reference_tokens_cpu,
        "next_tokens_match": next_tokens_match,
        "correctness_passed": correctness_passed,
        "all_expert_devices_used": all_expert_devices_used,
        "all_devices_used": all_devices_used,
        "timing_method": "perf_counter_cuda_synchronize_4gpu_moe_serving",
    },
    "environment": {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": getattr(torch.version, "cuda", None),
    },
    "hardware": {
        "device_type": "cuda-4gpu-moe-serving",
        "same_physical_host": True,
        "devices": [
            {
                "device": name,
                "index": index,
                "role": "gateway" if position == 0 else "expert",
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                "memory_before": memory_before[name],
                "memory_after": memory_after[name],
            }
            for position, (name, index) in enumerate(zip(devices, device_indices))
        ],
        "peer_access": p2p,
        "logical_hosts": ["logical-host-0", "logical-host-1", "logical-host-2", "logical-host-3"],
    },
    "claims": {
        "real_frontier_model": False,
        "target_model_parity": False,
        "live_http_endpoint": False,
        "formal_g2_passed": False,
        "formal_g3_passed": False,
        "g2_g3_gate_evidence": False,
        "production_distributed_serving": False,
    },
    "note": "Measured same-host four-CUDA-device tiny MoE serving probe. It proves gateway/expert GPU placement and split-vs-reference parity for the fixture, but not frontier-model parity, live HTTP serving, production distributed transport, or formal G2/G3 closure.",
}))
'''


def _unavailable(error: str, python: str) -> dict[str, Any]:
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "tier": TIER,
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.local_4gpu_moe_serving_smoke.torch_probe",
        "error": error,
        "environment": {"python_executable": python},
    }


def run_torch_4gpu_moe_serving_probe(
    *,
    torch_python: str | None = None,
    devices: str | Sequence[str] = DEFAULT_DEVICES,
    dtype: str = "float32",
    iterations: int = 5,
    warmup: int = 1,
    token_count: int = 6,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    vocab_size: int = 17,
    expert_count: int = 6,
    top_k: int = 2,
    tolerance: float = 1e-4,
    model_id: str = "fornax-tiny-4gpu-moe-fixture",
    request_id: str = "local-4gpu-moe-serving-request",
    prompt_text: str = "route a tiny MoE request across four GPUs",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    parsed_devices = parse_cuda_devices(devices)
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    _validate_config(
        iterations=iterations,
        warmup=warmup,
        token_count=token_count,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
        expert_count=expert_count,
        top_k=top_k,
        tolerance=tolerance,
        timeout_s=timeout_s,
    )
    if not model_id:
        raise ValueError("model_id must be non-empty")
    if not request_id:
        raise ValueError("request_id must be non-empty")
    python = torch_python or sys.executable
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                _external_torch_script(),
                str(iterations),
                str(warmup),
                str(token_count),
                str(hidden_dim),
                str(intermediate_dim),
                str(vocab_size),
                str(expert_count),
                str(top_k),
                ",".join(parsed_devices),
                dtype,
                str(tolerance),
                model_id,
                request_id,
                prompt_text,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(f"4-GPU MoE serving probe failed to launch: {type(exc).__name__}: {exc}", python)
    stdout = result.stdout.strip()
    if result.returncode != 0:
        data = _unavailable("4-GPU MoE serving probe exited nonzero", python)
        data.update({"returncode": result.returncode, "stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    try:
        data = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        data = _unavailable(f"4-GPU MoE serving probe did not emit JSON: {exc}", python)
        data.update({"stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    if not isinstance(data, dict):
        return _unavailable("4-GPU MoE serving probe JSON was not an object", python)
    data.setdefault("source", "fornax.local_4gpu_moe_serving_smoke.torch_probe")
    data.setdefault("backend", "torch")
    data.setdefault("environment", {})
    if isinstance(data["environment"], dict):
        data["environment"].setdefault("python_executable", python)
    return data


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _positive_int_field(value: Any, field: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return None
    return value


def _non_negative_number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{field} must be a non-negative number")
        return None
    return float(value)


def _positive_number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


def _int_list(value: Any, field: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    out: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            errors.append(f"{field}[{index}] must be an integer")
            return None
        out.append(item)
    return out


def _string_list(value: Any, field: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
            return None
        out.append(item)
    return out


def _cuda_device_name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"cuda:\d+", value) is not None


def validate_4gpu_moe_serving_probe_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "4-GPU MoE serving probe is same-host fixture evidence, not formal G2/G3 or frontier-model evidence"
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("probe_kind") != PROBE_KIND:
        errors.append(f"probe_kind must be {PROBE_KIND}")
    if data.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must be {EVIDENCE_SCOPE}")
    if data.get("tier") != TIER:
        errors.append(f"tier must be {TIER}")
    if data.get("backend") != "torch":
        errors.append("backend must be torch")
    if data.get("available") is not True:
        errors.append("available must be true")
    if data.get("measured") is not True:
        errors.append("measured must be true")
    if data.get("accelerator_measured") is not True:
        errors.append("accelerator_measured must be true")
    _non_empty_string(data.get("source"), "source", errors)

    config = data.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
        config = {}
    iterations = _positive_int_field(config.get("iterations"), "config.iterations", errors)
    token_count = _positive_int_field(config.get("token_count"), "config.token_count", errors)
    _positive_int_field(config.get("hidden_dim"), "config.hidden_dim", errors)
    _positive_int_field(config.get("intermediate_dim"), "config.intermediate_dim", errors)
    _positive_int_field(config.get("vocab_size"), "config.vocab_size", errors)
    expert_count = _positive_int_field(config.get("expert_count"), "config.expert_count", errors)
    top_k = _positive_int_field(config.get("top_k"), "config.top_k", errors)
    if top_k is not None and expert_count is not None and top_k > expert_count:
        errors.append("config.top_k must be <= config.expert_count")
    tolerance = _non_negative_number_field(config.get("tolerance"), "config.tolerance", errors)
    dtype = _non_empty_string(config.get("dtype"), "config.dtype", errors)
    if dtype is not None and dtype not in DTYPES:
        errors.append(f"config.dtype must be one of {sorted(DTYPES)}")
    devices = _string_list(config.get("devices"), "config.devices", errors)
    gateway_device = _non_empty_string(config.get("gateway_device"), "config.gateway_device", errors)
    expert_devices = _string_list(config.get("expert_devices"), "config.expert_devices", errors)
    logical_hosts = _string_list(config.get("logical_hosts"), "config.logical_hosts", errors)
    expert_devices_used = _string_list(config.get("expert_devices_used"), "config.expert_devices_used", errors)
    expert_device_calls = config.get("expert_device_calls_per_iteration")
    if devices is not None:
        if len(devices) != 4:
            errors.append("config.devices must contain exactly four devices")
        if len(set(devices)) != len(devices):
            errors.append("config.devices must be distinct")
        for device in devices:
            if not _cuda_device_name(device):
                errors.append("config.devices must contain cuda:<index> names")
                break
    if gateway_device is not None and devices is not None and gateway_device != devices[0]:
        errors.append("config.gateway_device must be config.devices[0]")
    if expert_devices is not None:
        if len(expert_devices) != 3:
            errors.append("config.expert_devices must contain three expert devices")
        if len(set(expert_devices)) != len(expert_devices):
            errors.append("config.expert_devices must be distinct")
        if devices is not None and expert_devices != devices[1:]:
            errors.append("config.expert_devices must match config.devices[1:]")
    if logical_hosts is not None and len(logical_hosts) != 4:
        errors.append("config.logical_hosts must contain four logical hosts")
    if expert_devices is not None and expert_devices_used is not None:
        if set(expert_devices_used) != set(expert_devices):
            errors.append("config.expert_devices_used must cover every expert device")
    if not isinstance(expert_device_calls, dict):
        errors.append("config.expert_device_calls_per_iteration must be an object")
        expert_device_calls = {}
    if expert_devices is not None and isinstance(expert_device_calls, dict):
        for device in expert_devices:
            calls = expert_device_calls.get(device)
            if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
                errors.append(f"config.expert_device_calls_per_iteration[{device}] must be a positive integer")
    if config.get("all_expert_devices_used") is not True:
        errors.append("config.all_expert_devices_used must be true")
    if config.get("all_devices_used") is not True:
        errors.append("config.all_devices_used must be true")
    _positive_int_field(config.get("transfer_payload_bytes_per_iteration"), "config.transfer_payload_bytes_per_iteration", errors)
    _non_empty_string(config.get("model_id"), "config.model_id", errors)
    _non_empty_string(config.get("request_id"), "config.request_id", errors)

    routing = data.get("routing")
    if not isinstance(routing, dict):
        errors.append("routing must be an object")
        routing = {}
    routes = routing.get("routes")
    if not isinstance(routes, list) or not routes:
        errors.append("routing.routes must be a non-empty list")
    elif token_count is not None and len(routes) != token_count:
        errors.append("routing.routes length must equal config.token_count")
    if isinstance(routes, list):
        for row_index, route in enumerate(routes):
            if not isinstance(route, dict):
                errors.append(f"routing.routes[{row_index}] must be an object")
                continue
            if route.get("token_index") != row_index:
                errors.append(f"routing.routes[{row_index}].token_index must equal its index")
            expert_ids = _int_list(route.get("expert_ids"), f"routing.routes[{row_index}].expert_ids", errors)
            weights = route.get("topk_weights")
            if not isinstance(weights, list):
                errors.append(f"routing.routes[{row_index}].topk_weights must be a list")
            else:
                if top_k is not None and len(weights) != top_k:
                    errors.append(f"routing.routes[{row_index}].topk_weights length must equal config.top_k")
                numeric = [float(item) for item in weights if isinstance(item, (int, float)) and not isinstance(item, bool)]
                if len(numeric) != len(weights) or any(item <= 0.0 for item in numeric):
                    errors.append(f"routing.routes[{row_index}].topk_weights must be positive numbers")
                elif abs(sum(numeric) - 1.0) > 1e-9:
                    errors.append(f"routing.routes[{row_index}].topk_weights must sum to 1")
            if expert_ids is not None:
                if top_k is not None and len(expert_ids) != top_k:
                    errors.append(f"routing.routes[{row_index}].expert_ids length must equal config.top_k")
                if len(set(expert_ids)) != len(expert_ids):
                    errors.append(f"routing.routes[{row_index}].expert_ids must be unique")
                if expert_count is not None and any(item < 0 or item >= expert_count for item in expert_ids):
                    errors.append(f"routing.routes[{row_index}].expert_ids must be in [0, config.expert_count)")
    placements = routing.get("expert_placement")
    if not isinstance(placements, list) or not placements:
        errors.append("routing.expert_placement must be a non-empty list")
    elif expert_count is not None and len(placements) != expert_count:
        errors.append("routing.expert_placement length must equal config.expert_count")
    if isinstance(placements, list):
        placed_devices = []
        for index, placement in enumerate(placements):
            if not isinstance(placement, dict):
                errors.append(f"routing.expert_placement[{index}] must be an object")
                continue
            if placement.get("expert_id") != index:
                errors.append(f"routing.expert_placement[{index}].expert_id must equal its index")
            device = placement.get("device")
            if not isinstance(device, str) or not device:
                errors.append(f"routing.expert_placement[{index}].device must be a non-empty string")
            else:
                placed_devices.append(device)
                if expert_devices is not None and device not in expert_devices:
                    errors.append(f"routing.expert_placement[{index}].device must be one of config.expert_devices")
            if placement.get("local_to_gateway") is not False:
                errors.append(f"routing.expert_placement[{index}].local_to_gateway must be false")
        if expert_devices is not None and set(placed_devices) != set(expert_devices):
            errors.append("routing.expert_placement must assign at least one expert to every expert device")

    serving = data.get("serving")
    if not isinstance(serving, dict):
        errors.append("serving must be an object")
        serving = {}
    response = serving.get("response")
    if not isinstance(response, dict):
        errors.append("serving.response must be an object")
        response = {}
    if serving.get("openai_compatible_shape") is not True:
        errors.append("serving.openai_compatible_shape must be true")
    if serving.get("live_http_endpoint") is not False:
        errors.append("serving.live_http_endpoint must be false")
    if response.get("object") != "chat.completion":
        errors.append("serving.response.object must be chat.completion")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        errors.append("serving.response.choices must be a non-empty list")
    _non_empty_string(serving.get("generated_text"), "serving.generated_text", errors)

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    tokens_processed = _positive_int_field(result.get("tokens_processed"), "result.tokens_processed", errors)
    expert_calls = _positive_int_field(result.get("expert_calls"), "result.expert_calls", errors)
    remote_calls = _positive_int_field(result.get("remote_expert_calls"), "result.remote_expert_calls", errors)
    _positive_int_field(result.get("remote_batches"), "result.remote_batches", errors)
    _positive_int_field(result.get("transfer_payload_bytes"), "result.transfer_payload_bytes", errors)
    _positive_number_field(result.get("elapsed_s"), "result.elapsed_s", errors)
    _positive_number_field(result.get("tokens_s"), "result.tokens_s", errors)
    _positive_number_field(result.get("expert_calls_s"), "result.expert_calls_s", errors)
    max_layer_abs_error = _non_negative_number_field(result.get("max_layer_abs_error"), "result.max_layer_abs_error", errors)
    max_logit_abs_error = _non_negative_number_field(result.get("max_logit_abs_error"), "result.max_logit_abs_error", errors)
    _non_empty_string(result.get("timing_method"), "result.timing_method", errors)
    if result.get("routing_match") is not True:
        errors.append("result.routing_match must be true")
    if result.get("next_tokens_match") is not True:
        errors.append("result.next_tokens_match must be true")
    if result.get("correctness_passed") is not True:
        errors.append("result.correctness_passed must be true")
    if result.get("all_expert_devices_used") is not True:
        errors.append("result.all_expert_devices_used must be true")
    if result.get("all_devices_used") is not True:
        errors.append("result.all_devices_used must be true")
    if iterations is not None and token_count is not None and tokens_processed is not None and tokens_processed != iterations * token_count:
        errors.append("result.tokens_processed must equal iterations * token_count")
    if iterations is not None and token_count is not None and top_k is not None and expert_calls is not None and expert_calls != iterations * token_count * top_k:
        errors.append("result.expert_calls must equal iterations * token_count * top_k")
    if remote_calls is not None and expert_calls is not None and remote_calls != expert_calls:
        errors.append("result.remote_expert_calls must equal result.expert_calls")
    if tolerance is not None and max_layer_abs_error is not None and max_layer_abs_error > tolerance:
        errors.append("result.max_layer_abs_error must be <= config.tolerance")
    if tolerance is not None and max_logit_abs_error is not None and max_logit_abs_error > tolerance:
        errors.append("result.max_logit_abs_error must be <= config.tolerance")
    next_tokens = result.get("next_tokens")
    reference_next_tokens = result.get("reference_next_tokens")
    if token_count is not None and (not isinstance(next_tokens, list) or len(next_tokens) != token_count):
        errors.append("result.next_tokens must contain config.token_count values")
    if token_count is not None and (not isinstance(reference_next_tokens, list) or len(reference_next_tokens) != token_count):
        errors.append("result.reference_next_tokens must contain config.token_count values")

    environment = data.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment must be an object")
        environment = {}
    _non_empty_string(environment.get("python_executable"), "environment.python_executable", errors)
    _non_empty_string(environment.get("torch_version"), "environment.torch_version", errors)
    cuda_count = _positive_int_field(environment.get("cuda_device_count"), "environment.cuda_device_count", errors)
    if cuda_count is not None and cuda_count < 4:
        errors.append("environment.cuda_device_count must be at least 4")

    hardware = data.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("hardware must be an object")
        hardware = {}
    if hardware.get("device_type") != "cuda-4gpu-moe-serving":
        errors.append("hardware.device_type must be cuda-4gpu-moe-serving")
    if hardware.get("same_physical_host") is not True:
        errors.append("hardware.same_physical_host must be true")
    hardware_devices = hardware.get("devices")
    if not isinstance(hardware_devices, list) or len(hardware_devices) != 4:
        errors.append("hardware.devices must contain four device records")
    elif devices is not None:
        hardware_device_names = [item.get("device") for item in hardware_devices if isinstance(item, dict)]
        if hardware_device_names != devices:
            errors.append("hardware.devices must match config.devices order")
        for index, item in enumerate(hardware_devices):
            if not isinstance(item, dict):
                errors.append(f"hardware.devices[{index}] must be an object")
                continue
            _non_empty_string(item.get("name"), f"hardware.devices[{index}].name", errors)
            _positive_int_field(item.get("total_memory_bytes"), f"hardware.devices[{index}].total_memory_bytes", errors)
            expected_role = "gateway" if index == 0 else "expert"
            if item.get("role") != expected_role:
                errors.append(f"hardware.devices[{index}].role must be {expected_role}")
    peer_access = hardware.get("peer_access")
    if not isinstance(peer_access, list):
        errors.append("hardware.peer_access must be a list")

    claims = data.get("claims")
    if not isinstance(claims, dict):
        errors.append("claims must be an object")
        claims = {}
    for field in [
        "real_frontier_model",
        "target_model_parity",
        "live_http_endpoint",
        "formal_g2_passed",
        "formal_g3_passed",
        "g2_g3_gate_evidence",
        "production_distributed_serving",
    ]:
        if claims.get(field) is not False:
            errors.append(f"claims.{field} must be false")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "tier": data.get("tier"),
            "backend": data.get("backend"),
            "accelerator_measured": data.get("accelerator_measured") is True,
            "gpu_count": len(devices) if devices is not None else None,
            "gateway_device": gateway_device,
            "expert_devices": expert_devices,
            "expert_devices_used": expert_devices_used,
            "all_expert_devices_used": config.get("all_expert_devices_used") is True,
            "all_devices_used": config.get("all_devices_used") is True,
            "tokens_s": result.get("tokens_s") if isinstance(result, dict) else None,
            "expert_calls_s": result.get("expert_calls_s") if isinstance(result, dict) else None,
            "max_layer_abs_error": result.get("max_layer_abs_error") if isinstance(result, dict) else None,
            "max_logit_abs_error": result.get("max_logit_abs_error") if isinstance(result, dict) else None,
            "generated_text": serving.get("generated_text") if isinstance(serving, dict) else None,
            "correctness_passed": result.get("correctness_passed") is True,
            "live_http_endpoint": False,
            "real_frontier_model": False,
            "g2_g3_gate_evidence": False,
        },
    }


def run_local_4gpu_moe_serving_smoke(
    *,
    out_dir: str | Path,
    torch_python: str | None = None,
    devices: str | Sequence[str] = DEFAULT_DEVICES,
    plan_id: str = "local-4gpu-moe-serving-plan",
    request_id: str = "local-4gpu-moe-serving-request",
    model: str = "fornax-tiny-4gpu-moe-fixture",
    prompt_text: str = "route a tiny MoE request across four GPUs",
    dtype: str = "float32",
    iterations: int = 5,
    warmup: int = 1,
    token_count: int = 6,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    vocab_size: int = 17,
    expert_count: int = 6,
    top_k: int = 2,
    tolerance: float = 1e-4,
    require_accelerator: bool = True,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    parsed_devices = parse_cuda_devices(devices)
    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    serving_path = bundle / "serving-adapter.json"
    probe_path = bundle / "four-gpu-moe-serving-probe.json"
    result_path = bundle / "local-4gpu-moe-serving-smoke.json"

    serving = simulate_serving_adapter(
        plan_id=plan_id,
        request_id=request_id,
        model=model,
        stream=False,
        max_tokens=token_count,
    )
    write_json(serving_path, serving)
    serving_validation = validate_serving_adapter_fixture(serving)
    checks = [_validation_entry("serving-adapter", serving_validation, str(serving_path))]

    probe = run_torch_4gpu_moe_serving_probe(
        torch_python=torch_python,
        devices=parsed_devices,
        dtype=dtype,
        iterations=iterations,
        warmup=warmup,
        token_count=token_count,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
        expert_count=expert_count,
        top_k=top_k,
        tolerance=tolerance,
        model_id=model,
        request_id=request_id,
        prompt_text=prompt_text,
        timeout_s=timeout_s,
    )
    write_json(probe_path, probe)
    probe_validation = validate_4gpu_moe_serving_probe_fixture(probe)
    checks.append(_validation_entry("four-gpu-moe-serving-probe", probe_validation, str(probe_path)))

    probe_summary = probe_validation.get("summary", {})
    bundle_policy_errors: list[str] = []
    if require_accelerator and probe_summary.get("accelerator_measured") is not True:
        bundle_policy_errors.append("four-gpu-moe-serving-probe must be measured accelerator evidence")
    if probe_summary.get("gpu_count") != 4:
        bundle_policy_errors.append("four-gpu-moe-serving-probe must use exactly four CUDA devices")
    if probe_summary.get("all_expert_devices_used") is not True:
        bundle_policy_errors.append("four-gpu-moe-serving-probe must execute work on all three expert GPUs")
    if probe_summary.get("all_devices_used") is not True:
        bundle_policy_errors.append("four-gpu-moe-serving-probe must use the gateway GPU plus all expert GPUs")
    if probe_summary.get("correctness_passed") is not True:
        bundle_policy_errors.append("four-gpu-moe-serving-probe must pass split-vs-reference correctness")
    checks.append(
        {
            "name": "bundle-policy",
            "ok": not bundle_policy_errors,
            "artifact": str(result_path),
            "errors": bundle_policy_errors,
            "warnings": [
                "same-host 4-GPU MoE serving smoke is not formal G2/G3 closure evidence",
                "tiny fixture MoE response is not real frontier target-model parity",
                "serving adapter fixture is not live HTTP endpoint evidence",
            ],
            "summary": {
                "require_accelerator": require_accelerator,
                "devices": parsed_devices,
            },
        }
    )

    passed_count = sum(1 for check in checks if check["ok"])
    smoke_passed = passed_count == len(checks)
    summary = {
        "accelerator_required": require_accelerator,
        "check_count": len(checks),
        "passed_count": passed_count,
        "serving_adapter_valid": bool(serving_validation.get("ok")),
        "moe_serving_probe_valid": bool(probe_validation.get("ok")),
        "accelerator_measured": probe_summary.get("accelerator_measured") is True,
        "gpu_count": probe_summary.get("gpu_count"),
        "gateway_device": probe_summary.get("gateway_device"),
        "expert_devices": probe_summary.get("expert_devices"),
        "expert_devices_used": probe_summary.get("expert_devices_used"),
        "all_expert_devices_used": probe_summary.get("all_expert_devices_used") is True,
        "all_devices_used": probe_summary.get("all_devices_used") is True,
        "tokens_s": probe_summary.get("tokens_s"),
        "expert_calls_s": probe_summary.get("expert_calls_s"),
        "max_layer_abs_error": probe_summary.get("max_layer_abs_error"),
        "max_logit_abs_error": probe_summary.get("max_logit_abs_error"),
        "generated_text": probe_summary.get("generated_text"),
        "correctness_passed": probe_summary.get("correctness_passed") is True,
        "local_4gpu_moe_serving_smoke_passed": smoke_passed,
        "live_http_endpoint": False,
        "real_frontier_model": False,
        "target_model_parity": False,
        "formal_g2_passed": False,
        "formal_g3_passed": False,
        "g2_g3_gate_evidence": False,
        "production_distributed_serving": False,
    }
    result = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "bundle": str(bundle),
        "artifacts": {
            "serving_adapter": str(serving_path),
            "four_gpu_moe_serving_probe": str(probe_path),
            "validation": str(result_path),
        },
        "summary": summary,
        "checks": checks,
        "ok": smoke_passed,
        "note": (
            "Local four-GPU MoE serving smoke evidence for a deterministic tiny MoE fixture. "
            "It proves same-host gateway/expert GPU placement and split-vs-reference serving response parity, "
            "but it is not live HTTP, frontier target-model parity, production distributed serving, or formal G2/G3 gate evidence."
        ),
    }
    write_json(result_path, result)
    return result


def validate_local_4gpu_moe_serving_smoke_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "local 4-GPU MoE serving smoke is same-host proxy evidence, not formal G2/G3 or frontier-model evidence"
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must be {EVIDENCE_SCOPE}")
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("checks must be a non-empty list")
        checks = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"checks[{index}] must be an object")
            continue
        if not check.get("name"):
            errors.append(f"checks[{index}].name must be set")
        if check.get("ok") is not True:
            errors.append(f"checks[{index}] {check.get('name', '<unknown>')} must pass")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    passed_count = sum(1 for check in checks if isinstance(check, dict) and check.get("ok") is True)
    if summary.get("check_count") != len(checks):
        errors.append("summary.check_count must match checks")
    if summary.get("passed_count") != passed_count:
        errors.append("summary.passed_count must match checks")
    if summary.get("serving_adapter_valid") is not True:
        errors.append("summary.serving_adapter_valid must be true")
    if summary.get("moe_serving_probe_valid") is not True:
        errors.append("summary.moe_serving_probe_valid must be true")
    if summary.get("accelerator_required") is True and summary.get("accelerator_measured") is not True:
        errors.append("summary.accelerator_measured must be true when accelerator is required")
    if summary.get("gpu_count") != 4:
        errors.append("summary.gpu_count must be 4")
    expert_devices = summary.get("expert_devices")
    expert_devices_used = summary.get("expert_devices_used")
    if not isinstance(expert_devices, list) or len(expert_devices) != 3:
        errors.append("summary.expert_devices must contain three expert devices")
    if not isinstance(expert_devices_used, list) or not isinstance(expert_devices, list) or set(expert_devices_used) != set(expert_devices):
        errors.append("summary.expert_devices_used must cover summary.expert_devices")
    if summary.get("all_expert_devices_used") is not True:
        errors.append("summary.all_expert_devices_used must be true")
    if summary.get("all_devices_used") is not True:
        errors.append("summary.all_devices_used must be true")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")
    if summary.get("local_4gpu_moe_serving_smoke_passed") is not True:
        errors.append("summary.local_4gpu_moe_serving_smoke_passed must be true")
    for field in [
        "live_http_endpoint",
        "real_frontier_model",
        "target_model_parity",
        "formal_g2_passed",
        "formal_g3_passed",
        "g2_g3_gate_evidence",
        "production_distributed_serving",
    ]:
        if summary.get(field) is not False:
            errors.append(f"summary.{field} must be false")
    if data.get("ok") is not True:
        errors.append("ok must be true")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "check_count": summary.get("check_count"),
            "passed_count": passed_count,
            "serving_adapter_valid": summary.get("serving_adapter_valid") is True,
            "moe_serving_probe_valid": summary.get("moe_serving_probe_valid") is True,
            "accelerator_measured": summary.get("accelerator_measured") is True,
            "gpu_count": summary.get("gpu_count"),
            "gateway_device": summary.get("gateway_device"),
            "expert_devices": summary.get("expert_devices"),
            "expert_devices_used": summary.get("expert_devices_used"),
            "all_expert_devices_used": summary.get("all_expert_devices_used") is True,
            "all_devices_used": summary.get("all_devices_used") is True,
            "tokens_s": summary.get("tokens_s"),
            "expert_calls_s": summary.get("expert_calls_s"),
            "generated_text": summary.get("generated_text"),
            "correctness_passed": summary.get("correctness_passed") is True,
            "local_4gpu_moe_serving_smoke_passed": summary.get("local_4gpu_moe_serving_smoke_passed") is True,
            "live_http_endpoint": False,
            "real_frontier_model": False,
            "g2_g3_gate_evidence": False,
        },
    }


def validate_local_4gpu_moe_serving_smoke(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "local-4gpu-moe-serving-smoke.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid local 4-GPU MoE serving smoke artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["local 4-GPU MoE serving smoke artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_local_4gpu_moe_serving_smoke_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
