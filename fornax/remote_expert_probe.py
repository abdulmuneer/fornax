from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .io import read_json

PROBE_KIND = "remote-expert-batch-probe"
BACKENDS = {"cpu-stdlib", "torch"}
DTYPES = {"float32", "float16", "bfloat16"}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _non_negative_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


def _validate_config(iterations: int, warmup: int, token_count: int, hidden_dim: int, intermediate_dim: int, expert_id: int, tolerance: float) -> None:
    _positive_int("iterations", iterations)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    _positive_int("token_count", token_count)
    _positive_int("hidden_dim", hidden_dim)
    _positive_int("intermediate_dim", intermediate_dim)
    if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 0:
        raise ValueError("expert_id must be a non-negative integer")
    _non_negative_number("tolerance", tolerance)


def _input_value(token: int, dim: int) -> float:
    return (((token + 2) * (dim + 5)) % 31 - 15) / 31.0


def _up_weight(expert_id: int, hidden: int, intermediate: int) -> float:
    return (((expert_id + 3) * (hidden + 7) * (intermediate + 11)) % 37 - 18) / 37.0


def _down_weight(expert_id: int, intermediate: int, hidden: int) -> float:
    return (((expert_id + 5) * (intermediate + 13) * (hidden + 17)) % 41 - 20) / 41.0


def _route_weight(token: int) -> float:
    return 0.5 + 0.1 * (token % 3)


def _inputs(token_count: int, hidden_dim: int) -> list[list[float]]:
    return [[_input_value(token, dim) for dim in range(hidden_dim)] for token in range(token_count)]


def _expert_outputs(values: list[list[float]], *, expert_id: int, intermediate_dim: int) -> list[list[float]]:
    hidden_dim = len(values[0]) if values else 0
    outputs: list[list[float]] = []
    for row in values:
        hidden_values: list[float] = []
        for intermediate in range(intermediate_dim):
            acc = 0.0
            for hidden, value in enumerate(row):
                acc += value * _up_weight(expert_id, hidden, intermediate)
            hidden_values.append(acc if acc > 0.0 else 0.0)
        output: list[float] = []
        for hidden in range(hidden_dim):
            acc = 0.0
            for intermediate, value in enumerate(hidden_values):
                acc += value * _down_weight(expert_id, intermediate, hidden)
            output.append(acc)
        outputs.append(output)
    return outputs


def _weighted(outputs: list[list[float]]) -> list[list[float]]:
    return [[_route_weight(token) * value for value in row] for token, row in enumerate(outputs)]


def _checksum(outputs: list[list[float]]) -> float:
    total = 0.0
    for row_index, row in enumerate(outputs):
        for dim, value in enumerate(row):
            total += value * (1.0 + row_index * 0.01 + dim * 0.0001)
    return total


def _max_abs_error(a: list[list[float]], b: list[list[float]]) -> float:
    max_error = 0.0
    for row_a, row_b in zip(a, b):
        for left, right in zip(row_a, row_b):
            max_error = max(max_error, abs(left - right))
    return max_error


def run_cpu_remote_expert_batch_probe(
    *,
    iterations: int = 5,
    warmup: int = 1,
    token_count: int = 4,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    expert_id: int = 5,
    tolerance: float = 0.0,
    logical_source_host: str = "logical-host-0",
    logical_expert_host: str = "logical-host-1",
) -> dict[str, Any]:
    _validate_config(iterations, warmup, token_count, hidden_dim, intermediate_dim, expert_id, tolerance)
    if not logical_source_host or not logical_expert_host:
        raise ValueError("logical host names must be non-empty")
    source_inputs = _inputs(token_count, hidden_dim)
    reference = _weighted(_expert_outputs(source_inputs, expert_id=expert_id, intermediate_dim=intermediate_dim))
    for _ in range(warmup):
        payload = [list(row) for row in source_inputs]
        remote = _weighted(_expert_outputs(payload, expert_id=expert_id, intermediate_dim=intermediate_dim))
        _ = [list(row) for row in remote]
    started_ns = time.perf_counter_ns()
    remote_result: list[list[float]] = []
    for _ in range(iterations):
        payload = [list(row) for row in source_inputs]
        remote = _weighted(_expert_outputs(payload, expert_id=expert_id, intermediate_dim=intermediate_dim))
        remote_result = [list(row) for row in remote]
    elapsed_ns = time.perf_counter_ns() - started_ns
    elapsed_s = elapsed_ns / 1_000_000_000.0
    max_abs_error = _max_abs_error(reference, remote_result)
    expert_calls = iterations * token_count
    transfer_payload_bytes_per_batch = token_count * hidden_dim * 8 * 2
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T0/T1-reference",
        "measured": True,
        "accelerator_measured": False,
        "backend": "cpu-stdlib",
        "available": True,
        "source": "fornax.remote_expert_probe.cpu_remote_expert_batch.stdlib",
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "token_count": token_count,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "expert_id": expert_id,
            "source_device": "cpu",
            "expert_device": "cpu",
            "logical_source_host": logical_source_host,
            "logical_expert_host": logical_expert_host,
            "dtype": "float64-reference",
            "tolerance": tolerance,
            "transfer_payload_bytes_per_batch": transfer_payload_bytes_per_batch,
        },
        "result": {
            "elapsed_s": elapsed_s,
            "elapsed_ns": elapsed_ns,
            "remote_batches": iterations,
            "tokens_processed": expert_calls,
            "expert_calls": expert_calls,
            "transfer_payload_bytes": iterations * transfer_payload_bytes_per_batch,
            "batches_s": iterations / elapsed_s if elapsed_s > 0 else None,
            "expert_calls_s": expert_calls / elapsed_s if elapsed_s > 0 else None,
            "checksum": _checksum(remote_result),
            "reference_checksum": _checksum(reference),
            "max_abs_error": max_abs_error,
            "correctness_passed": max_abs_error <= tolerance,
            "timing_method": "perf_counter_ns_cpu_remote_expert_batch",
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "hardware": {
            "device_type": "cpu-remote-expert",
            "source_device": "cpu",
            "expert_device": "cpu",
            "source_name": platform.processor() or platform.machine(),
            "expert_name": platform.processor() or platform.machine(),
            "same_physical_host": True,
            "logical_hosts": [logical_source_host, logical_expert_host],
        },
        "note": "CPU reference remote expert batch probe; not accelerator or T3 hardware evidence.",
    }


def _external_torch_script() -> str:
    return r"""
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
expert_id = int(sys.argv[6])
source_device_name = sys.argv[7]
expert_device_name = sys.argv[8]
dtype_name = sys.argv[9]
tolerance = float(sys.argv[10])
logical_source_host = sys.argv[11]
logical_expert_host = sys.argv[12]
probe_kind = "remote-expert-batch-probe"
tier = "T3-same-host-remote-expert-simulation"


def emit_unavailable(error):
    print(json.dumps({
        "version": 1,
        "probe_kind": probe_kind,
        "tier": tier,
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.remote_expert_probe.torch_remote_expert_batch",
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
    source_index = cuda_index(source_device_name)
    expert_index = cuda_index(expert_device_name)
    if source_index == expert_index:
        raise ValueError("source and expert CUDA devices must differ")
    device_count = int(torch.cuda.device_count())
    if source_index >= device_count or expert_index >= device_count:
        raise ValueError(f"device index out of range for cuda_device_count={device_count}: {source_index}->{expert_index}")
    source_device = torch.device(source_device_name)
    expert_device = torch.device(expert_device_name)
except Exception as exc:
    emit_unavailable(f"device setup failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)


def make_inputs(device, target_dtype):
    tokens = torch.arange(token_count, device=device, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((tokens + 2) * (dims + 5), 31.0)
    return ((values - 15.0) / 31.0).to(dtype=target_dtype)


def up_weights(device, target_dtype):
    hidden = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(1)
    intermediate = torch.arange(intermediate_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((expert_id + 3) * (hidden + 7) * (intermediate + 11), 37.0)
    return ((values - 18.0) / 37.0).to(dtype=target_dtype)


def down_weights(device, target_dtype):
    intermediate = torch.arange(intermediate_dim, device=device, dtype=torch.float32).unsqueeze(1)
    hidden = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((expert_id + 5) * (intermediate + 13) * (hidden + 17), 41.0)
    return ((values - 20.0) / 41.0).to(dtype=target_dtype)


def route_weights(device, target_dtype):
    tokens = torch.arange(token_count, device=device, dtype=torch.float32).unsqueeze(1)
    return (0.5 + 0.1 * torch.remainder(tokens, 3.0)).to(dtype=target_dtype)


def run_local_reference(x):
    up = up_weights(source_device, torch.float32)
    down = down_weights(source_device, torch.float32)
    weights = route_weights(source_device, torch.float32)
    return (torch.relu(x.float() @ up) @ down) * weights


def run_remote(x):
    payload = x.to(expert_device, non_blocking=False)
    up = up_weights(expert_device, dtype)
    down = down_weights(expert_device, dtype)
    weights = route_weights(expert_device, dtype)
    remote = (torch.relu(payload @ up) @ down) * weights
    return remote.to(source_device, non_blocking=False)

try:
    try:
        p2p_source_to_expert = bool(torch.cuda.can_device_access_peer(source_index, expert_index))
        p2p_expert_to_source = bool(torch.cuda.can_device_access_peer(expert_index, source_index))
    except Exception:
        p2p_source_to_expert = None
        p2p_expert_to_source = None
    with torch.no_grad():
        x = make_inputs(source_device, dtype)
        reference = run_local_reference(x)
        remote = run_remote(x)
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(expert_device)
        max_abs_error = float((remote.detach().cpu().float() - reference.detach().cpu().float()).abs().max().item())
        correctness_passed = bool(max_abs_error <= tolerance)
        for _ in range(warmup):
            run_remote(x)
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(expert_device)
        started = time.perf_counter()
        timed = None
        for _ in range(iterations):
            timed = run_remote(x)
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(expert_device)
        elapsed_s = time.perf_counter() - started
        timed_cpu = timed.detach().cpu().float() if timed is not None else remote.detach().cpu().float()
        reference_cpu = reference.detach().cpu().float()
except Exception as exc:
    emit_unavailable(f"remote expert batch failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

element_size = int(torch.tensor([], dtype=dtype).element_size())
transfer_payload_bytes_per_batch = int(token_count * hidden_dim * element_size * 2)
expert_calls = int(iterations * token_count)
source_props = torch.cuda.get_device_properties(source_device)
expert_props = torch.cuda.get_device_properties(expert_device)
col_weights = 1.0 + torch.arange(hidden_dim, dtype=torch.float32).unsqueeze(0) * 0.0001
row_weights = 1.0 + torch.arange(token_count, dtype=torch.float32).unsqueeze(1) * 0.01
weights = row_weights * col_weights
print(json.dumps({
    "version": 1,
    "probe_kind": probe_kind,
    "tier": tier,
    "measured": True,
    "accelerator_measured": True,
    "backend": "torch",
    "available": True,
    "source": "fornax.remote_expert_probe.torch_remote_expert_batch",
    "config": {
        "iterations": iterations,
        "warmup": warmup,
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "expert_id": expert_id,
        "source_device": source_device_name,
        "expert_device": expert_device_name,
        "logical_source_host": logical_source_host,
        "logical_expert_host": logical_expert_host,
        "dtype": dtype_name,
        "tolerance": tolerance,
        "transfer_payload_bytes_per_batch": transfer_payload_bytes_per_batch,
    },
    "result": {
        "elapsed_s": elapsed_s,
        "elapsed_ns": int(elapsed_s * 1000000000),
        "remote_batches": iterations,
        "tokens_processed": expert_calls,
        "expert_calls": expert_calls,
        "transfer_payload_bytes": int(iterations * transfer_payload_bytes_per_batch),
        "batches_s": iterations / elapsed_s if elapsed_s > 0 else None,
        "expert_calls_s": expert_calls / elapsed_s if elapsed_s > 0 else None,
        "checksum": float((timed_cpu * weights).sum().item()),
        "reference_checksum": float((reference_cpu * weights).sum().item()),
        "max_abs_error": max_abs_error,
        "correctness_passed": correctness_passed,
        "timing_method": "perf_counter_cuda_synchronize_remote_expert_batch",
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
        "device_type": "cuda-remote-expert",
        "source_device": source_device_name,
        "expert_device": expert_device_name,
        "source_index": source_index,
        "expert_index": expert_index,
        "source_name": torch.cuda.get_device_name(source_device),
        "expert_name": torch.cuda.get_device_name(expert_device),
        "source_total_memory_bytes": int(source_props.total_memory),
        "expert_total_memory_bytes": int(expert_props.total_memory),
        "peer_access": {"source_to_expert": p2p_source_to_expert, "expert_to_source": p2p_expert_to_source},
        "same_physical_host": True,
        "logical_hosts": [logical_source_host, logical_expert_host],
    },
    "note": "Measured same-host two-GPU remote expert batch; not real multi-host T3 closure.",
}))
"""


def _unavailable(error: str, python: str) -> dict[str, Any]:
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T3-same-host-remote-expert-simulation",
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.remote_expert_probe.torch_remote_expert_batch",
        "error": error,
        "environment": {"python_executable": python},
    }


def run_torch_remote_expert_batch_probe(
    *,
    torch_python: str | None = None,
    source_device: str = "cuda:0",
    expert_device: str = "cuda:1",
    dtype: str = "float32",
    iterations: int = 20,
    warmup: int = 3,
    token_count: int = 8,
    hidden_dim: int = 64,
    intermediate_dim: int = 128,
    expert_id: int = 5,
    tolerance: float = 1e-4,
    logical_source_host: str = "logical-host-0",
    logical_expert_host: str = "logical-host-1",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    _validate_config(iterations, warmup, token_count, hidden_dim, intermediate_dim, expert_id, tolerance)
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if not logical_source_host or not logical_expert_host:
        raise ValueError("logical host names must be non-empty")
    _positive_number("timeout_s", timeout_s)
    python = torch_python or sys.executable
    try:
        result = subprocess.run(
            [python, "-c", _external_torch_script(), str(iterations), str(warmup), str(token_count), str(hidden_dim), str(intermediate_dim), str(expert_id), source_device, expert_device, dtype, str(tolerance), logical_source_host, logical_expert_host],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(f"torch remote expert batch failed to launch: {type(exc).__name__}: {exc}", python)
    stdout = result.stdout.strip()
    if result.returncode != 0:
        data = _unavailable("torch remote expert batch exited nonzero", python)
        data.update({"returncode": result.returncode, "stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    try:
        data = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        data = _unavailable(f"torch remote expert batch did not emit JSON: {exc}", python)
        data.update({"stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    if not isinstance(data, dict):
        return _unavailable("torch remote expert batch JSON was not an object", python)
    data.setdefault("source", "fornax.remote_expert_probe.torch_remote_expert_batch")
    data.setdefault("backend", "torch")
    data.setdefault("environment", {})
    if isinstance(data["environment"], dict):
        data["environment"].setdefault("python_executable", python)
    return data


def run_remote_expert_batch_probe(
    *,
    backend: str = "cpu-stdlib",
    torch_python: str | None = None,
    source_device: str = "cuda:0",
    expert_device: str = "cuda:1",
    dtype: str = "float32",
    iterations: int = 5,
    warmup: int = 1,
    token_count: int = 4,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    expert_id: int = 5,
    tolerance: float = 0.0,
    logical_source_host: str = "logical-host-0",
    logical_expert_host: str = "logical-host-1",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    if backend == "cpu-stdlib":
        return run_cpu_remote_expert_batch_probe(iterations=iterations, warmup=warmup, token_count=token_count, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, expert_id=expert_id, tolerance=tolerance, logical_source_host=logical_source_host, logical_expert_host=logical_expert_host)
    return run_torch_remote_expert_batch_probe(torch_python=torch_python, source_device=source_device, expert_device=expert_device, dtype=dtype, iterations=iterations, warmup=warmup, token_count=token_count, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, expert_id=expert_id, tolerance=tolerance, logical_source_host=logical_source_host, logical_expert_host=logical_expert_host, timeout_s=timeout_s)


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


def _positive_number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


def _number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field} must be numeric")
        return None
    return float(value)


def _non_negative_number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{field} must be a non-negative number")
        return None
    return float(value)


def _cuda_device_name(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith("cuda:") and value[5:].isdigit()


def validate_remote_expert_batch_probe_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("probe_kind") != PROBE_KIND:
        errors.append(f"probe_kind must be {PROBE_KIND}")
    tier = _non_empty_string(data.get("tier"), "tier", errors)
    backend = _non_empty_string(data.get("backend"), "backend", errors)
    if backend is not None and backend not in BACKENDS:
        errors.append(f"backend must be one of {sorted(BACKENDS)}")
    measured = data.get("measured")
    if not isinstance(measured, bool):
        errors.append("measured must be a boolean")
    accelerator_measured = data.get("accelerator_measured")
    if not isinstance(accelerator_measured, bool):
        errors.append("accelerator_measured must be a boolean")
    _non_empty_string(data.get("source"), "source", errors)
    config = data.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
        config = {}
    iterations = _positive_int_field(config.get("iterations"), "config.iterations", errors)
    token_count = _positive_int_field(config.get("token_count"), "config.token_count", errors)
    _positive_int_field(config.get("hidden_dim"), "config.hidden_dim", errors)
    _positive_int_field(config.get("intermediate_dim"), "config.intermediate_dim", errors)
    _positive_int_field(config.get("transfer_payload_bytes_per_batch"), "config.transfer_payload_bytes_per_batch", errors)
    expert_id = config.get("expert_id")
    if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 0:
        errors.append("config.expert_id must be a non-negative integer")
    _non_negative_number_field(config.get("tolerance"), "config.tolerance", errors)
    dtype = _non_empty_string(config.get("dtype"), "config.dtype", errors)
    if backend == "torch" and dtype is not None and dtype not in DTYPES:
        errors.append(f"config.dtype must be one of {sorted(DTYPES)}")
    source_device = _non_empty_string(config.get("source_device"), "config.source_device", errors)
    expert_device = _non_empty_string(config.get("expert_device"), "config.expert_device", errors)
    logical_source_host = _non_empty_string(config.get("logical_source_host"), "config.logical_source_host", errors)
    logical_expert_host = _non_empty_string(config.get("logical_expert_host"), "config.logical_expert_host", errors)
    if logical_source_host == logical_expert_host and logical_source_host is not None:
        errors.append("config.logical_source_host must differ from logical_expert_host")
    result = data.get("result")
    if not isinstance(result, dict):
        if measured:
            errors.append("result must be an object for measured probes")
        result = {}
    if measured:
        remote_batches = _positive_int_field(result.get("remote_batches"), "result.remote_batches", errors)
        tokens_processed = _positive_int_field(result.get("tokens_processed"), "result.tokens_processed", errors)
        expert_calls = _positive_int_field(result.get("expert_calls"), "result.expert_calls", errors)
        transfer_payload_bytes = _positive_int_field(result.get("transfer_payload_bytes"), "result.transfer_payload_bytes", errors)
        _positive_number_field(result.get("elapsed_s"), "result.elapsed_s", errors)
        _positive_number_field(result.get("batches_s"), "result.batches_s", errors)
        _positive_number_field(result.get("expert_calls_s"), "result.expert_calls_s", errors)
        _number_field(result.get("checksum"), "result.checksum", errors)
        _number_field(result.get("reference_checksum"), "result.reference_checksum", errors)
        _non_negative_number_field(result.get("max_abs_error"), "result.max_abs_error", errors)
        _non_empty_string(result.get("timing_method"), "result.timing_method", errors)
        if result.get("correctness_passed") is not True:
            errors.append("result.correctness_passed must be true for measured probe evidence")
        if iterations is not None and remote_batches is not None and remote_batches != iterations:
            errors.append("result.remote_batches must equal config.iterations")
        if iterations is not None and token_count is not None and tokens_processed is not None and tokens_processed != iterations * token_count:
            errors.append("result.tokens_processed must equal iterations * token_count")
        if tokens_processed is not None and expert_calls is not None and expert_calls != tokens_processed:
            errors.append("result.expert_calls must equal result.tokens_processed")
        per_batch = config.get("transfer_payload_bytes_per_batch")
        if isinstance(per_batch, int) and iterations is not None and transfer_payload_bytes is not None and transfer_payload_bytes != iterations * per_batch:
            errors.append("result.transfer_payload_bytes must equal iterations * transfer_payload_bytes_per_batch")
    else:
        _non_empty_string(data.get("error"), "error", errors)
    environment = data.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment must be an object")
        environment = {}
    _non_empty_string(environment.get("python_executable"), "environment.python_executable", errors)
    if backend == "torch" and measured:
        _non_empty_string(environment.get("torch_version"), "environment.torch_version", errors)
    hardware = data.get("hardware")
    if measured:
        if not isinstance(hardware, dict):
            errors.append("hardware must be an object for measured probes")
            hardware = {}
        device_type = _non_empty_string(hardware.get("device_type"), "hardware.device_type", errors)
        hardware_source_device = _non_empty_string(hardware.get("source_device"), "hardware.source_device", errors)
        hardware_expert_device = _non_empty_string(hardware.get("expert_device"), "hardware.expert_device", errors)
        _non_empty_string(hardware.get("source_name"), "hardware.source_name", errors)
        _non_empty_string(hardware.get("expert_name"), "hardware.expert_name", errors)
        if hardware.get("same_physical_host") is not True:
            errors.append("hardware.same_physical_host must be true for this simulation probe")
        logical_hosts = hardware.get("logical_hosts")
        if not isinstance(logical_hosts, list) or len(logical_hosts) != 2:
            errors.append("hardware.logical_hosts must contain two logical host names")
        if source_device is not None and hardware_source_device != source_device:
            errors.append("hardware.source_device must match config.source_device")
        if expert_device is not None and hardware_expert_device != expert_device:
            errors.append("hardware.expert_device must match config.expert_device")
        if accelerator_measured:
            if tier != "T3-same-host-remote-expert-simulation":
                errors.append("accelerator remote expert probes must use T3-same-host-remote-expert-simulation tier")
            if device_type != "cuda-remote-expert":
                errors.append("hardware.device_type must be cuda-remote-expert when accelerator_measured is true")
            if not _cuda_device_name(source_device):
                errors.append("config.source_device must be cuda:<index> for accelerator evidence")
            if not _cuda_device_name(expert_device):
                errors.append("config.expert_device must be cuda:<index> for accelerator evidence")
            if source_device == expert_device and source_device is not None:
                errors.append("config.source_device and config.expert_device must differ")
            _positive_int_field(hardware.get("source_total_memory_bytes"), "hardware.source_total_memory_bytes", errors)
            _positive_int_field(hardware.get("expert_total_memory_bytes"), "hardware.expert_total_memory_bytes", errors)
        elif tier == "T3-same-host-remote-expert-simulation":
            errors.append("T3-same-host-remote-expert-simulation probes must set accelerator_measured true")
    if backend == "cpu-stdlib" and accelerator_measured:
        errors.append("cpu-stdlib backend cannot be accelerator_measured")
    if measured and not accelerator_measured:
        warnings.append("probe is measured but not accelerator evidence")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "summary": {"tier": tier, "backend": backend, "measured": bool(measured), "accelerator_measured": bool(accelerator_measured), "source_device": source_device, "expert_device": expert_device, "remote_batches": result.get("remote_batches") if isinstance(result, dict) else None, "expert_calls": result.get("expert_calls") if isinstance(result, dict) else None, "batches_s": result.get("batches_s") if isinstance(result, dict) else None, "expert_calls_s": result.get("expert_calls_s") if isinstance(result, dict) else None, "max_abs_error": result.get("max_abs_error") if isinstance(result, dict) else None}}


def validate_remote_expert_batch_probe(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {"ok": False, "errors": [f"invalid remote expert batch probe artifact: {exc}"], "warnings": [], "summary": {}, "fixture": str(fixture_path)}
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["remote expert batch probe artifact must be a JSON object"], "warnings": [], "summary": {}, "fixture": str(fixture_path)}
    result = validate_remote_expert_batch_probe_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
