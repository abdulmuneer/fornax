
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .io import read_json


PROBE_KIND = "expert-mlp-accelerator-probe"
BACKENDS = {"cpu-stdlib", "torch"}
DTYPES = {"float32", "float16", "bfloat16"}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _validate_config(
    *,
    iterations: int,
    warmup: int,
    batch_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts: int,
    top_k: int,
    tolerance: float,
) -> None:
    _positive_int("iterations", iterations)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    for name, value in {
        "batch_tokens": batch_tokens,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "experts": experts,
        "top_k": top_k,
    }.items():
        _positive_int(name, value)
    if top_k > experts:
        raise ValueError("top_k cannot exceed experts")
    _positive_number("tolerance", tolerance)


def _input_value(token: int, dim: int) -> float:
    return (((token + 1) * (dim + 5)) % 23 - 11) / 23.0


def _up_weight(expert: int, hidden: int, intermediate: int) -> float:
    return (((expert + 1) * (hidden + 3) * (intermediate + 5)) % 29 - 14) / 29.0


def _down_weight(expert: int, intermediate: int, hidden: int) -> float:
    return (((expert + 7) * (intermediate + 2) * (hidden + 11)) % 31 - 15) / 31.0


def _routes(token: int, *, experts: int, top_k: int) -> list[tuple[int, float]]:
    weights = [1.0 / float(rank + 1) for rank in range(top_k)]
    total = sum(weights)
    return [((token + rank) % experts, weights[rank] / total) for rank in range(top_k)]


def _cpu_reference_outputs(
    *,
    batch_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts: int,
    top_k: int,
) -> list[list[float]]:
    outputs: list[list[float]] = []
    for token in range(batch_tokens):
        vector = [_input_value(token, hidden) for hidden in range(hidden_dim)]
        output = [0.0 for _ in range(hidden_dim)]
        for expert_id, route_weight in _routes(token, experts=experts, top_k=top_k):
            hidden_values: list[float] = []
            for intermediate in range(intermediate_dim):
                acc = 0.0
                for hidden in range(hidden_dim):
                    acc += vector[hidden] * _up_weight(expert_id, hidden, intermediate)
                hidden_values.append(acc if acc > 0.0 else 0.0)
            for hidden in range(hidden_dim):
                acc = 0.0
                for intermediate in range(intermediate_dim):
                    acc += hidden_values[intermediate] * _down_weight(
                        expert_id, intermediate, hidden
                    )
                output[hidden] += route_weight * acc
        outputs.append(output)
    return outputs


def _checksum(outputs: list[list[float]]) -> float:
    total = 0.0
    for token, row in enumerate(outputs):
        for dim, value in enumerate(row):
            total += value * (1.0 + token * 0.01 + dim * 0.0001)
    return total


def _max_abs_error(a: list[list[float]], b: list[list[float]]) -> float:
    max_error = 0.0
    for row_a, row_b in zip(a, b):
        for value_a, value_b in zip(row_a, row_b):
            max_error = max(max_error, abs(value_a - value_b))
    return max_error


def run_cpu_expert_mlp_probe(
    *,
    iterations: int = 10,
    warmup: int = 1,
    batch_tokens: int = 4,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    experts: int = 4,
    top_k: int = 2,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Run a deterministic CPU reference probe for validator and fallback coverage."""

    _validate_config(
        iterations=iterations,
        warmup=warmup,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        experts=experts,
        top_k=top_k,
        tolerance=tolerance,
    )
    reference = _cpu_reference_outputs(
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        experts=experts,
        top_k=top_k,
    )
    for _ in range(warmup):
        _cpu_reference_outputs(
            batch_tokens=batch_tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            experts=experts,
            top_k=top_k,
        )
    started = time.perf_counter_ns()
    outputs = reference
    for _ in range(iterations):
        outputs = _cpu_reference_outputs(
            batch_tokens=batch_tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            experts=experts,
            top_k=top_k,
        )
    elapsed_ns = time.perf_counter_ns() - started
    elapsed_s = elapsed_ns / 1_000_000_000.0
    tokens_processed = iterations * batch_tokens
    expert_calls = tokens_processed * top_k
    max_abs_error = _max_abs_error(reference, outputs)
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T0/T1-reference",
        "measured": True,
        "accelerator_measured": False,
        "backend": "cpu-stdlib",
        "source": "fornax.accelerator_probe.cpu_expert_mlp.stdlib",
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "batch_tokens": batch_tokens,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "experts": experts,
            "top_k": top_k,
            "dtype": "float64-reference",
            "tolerance": tolerance,
        },
        "result": {
            "elapsed_s": elapsed_s,
            "elapsed_ns": elapsed_ns,
            "tokens_processed": tokens_processed,
            "expert_calls": expert_calls,
            "tokens_s": tokens_processed / elapsed_s if elapsed_s > 0 else None,
            "expert_calls_s": expert_calls / elapsed_s if elapsed_s > 0 else None,
            "checksum": _checksum(outputs),
            "reference_checksum": _checksum(reference),
            "max_abs_error": max_abs_error,
            "correctness_passed": max_abs_error <= tolerance,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "hardware": {
            "device_type": "cpu",
            "device": "cpu",
            "name": platform.processor() or platform.machine(),
        },
        "note": (
            "CPU reference expert-MLP probe for plumbing and correctness checks; "
            "not T2 accelerator evidence."
        ),
    }


def _external_torch_script() -> str:
    return """
import json
import os
import platform
import sys
import time

iterations = int(sys.argv[1])
warmup = int(sys.argv[2])
batch_tokens = int(sys.argv[3])
hidden_dim = int(sys.argv[4])
intermediate_dim = int(sys.argv[5])
experts = int(sys.argv[6])
top_k = int(sys.argv[7])
device_name = sys.argv[8]
dtype_name = sys.argv[9]
tolerance = float(sys.argv[10])

try:
    import torch
except Exception as exc:
    print(json.dumps({
        \"version\": 1,
        \"probe_kind\": \"expert-mlp-accelerator-probe\",
        \"tier\": \"T2-single-node-accelerator\",
        \"measured\": False,
        \"accelerator_measured\": False,
        \"backend\": \"torch\",
        \"available\": False,
        \"error\": f\"torch import failed: {type(exc).__name__}: {exc}\",
        \"environment\": {\"python_executable\": sys.executable},
    }))
    raise SystemExit(0)

if dtype_name == \"float32\":
    dtype = torch.float32
elif dtype_name == \"float16\":
    dtype = torch.float16
elif dtype_name == \"bfloat16\":
    dtype = torch.bfloat16
else:
    print(json.dumps({
        \"version\": 1,
        \"probe_kind\": \"expert-mlp-accelerator-probe\",
        \"tier\": \"T2-single-node-accelerator\",
        \"measured\": False,
        \"accelerator_measured\": False,
        \"backend\": \"torch\",
        \"available\": False,
        \"error\": f\"unsupported dtype: {dtype_name}\",
        \"torch_version\": getattr(torch, \"__version__\", \"unknown\"),
        \"environment\": {\"python_executable\": sys.executable},
    }))
    raise SystemExit(0)

if device_name.startswith(\"cuda\") and not torch.cuda.is_available():
    print(json.dumps({
        \"version\": 1,
        \"probe_kind\": \"expert-mlp-accelerator-probe\",
        \"tier\": \"T2-single-node-accelerator\",
        \"measured\": False,
        \"accelerator_measured\": False,
        \"backend\": \"torch\",
        \"available\": False,
        \"error\": \"torch.cuda.is_available() is false\",
        \"torch_version\": getattr(torch, \"__version__\", \"unknown\"),
        \"environment\": {\"python_executable\": sys.executable},
    }))
    raise SystemExit(0)

try:
    device = torch.device(device_name)
    if device.type == \"cuda\":
        torch.cuda.set_device(device)
except Exception as exc:
    print(json.dumps({
        \"version\": 1,
        \"probe_kind\": \"expert-mlp-accelerator-probe\",
        \"tier\": \"T2-single-node-accelerator\",
        \"measured\": False,
        \"accelerator_measured\": False,
        \"backend\": \"torch\",
        \"available\": False,
        \"error\": f\"device setup failed: {type(exc).__name__}: {exc}\",
        \"torch_version\": getattr(torch, \"__version__\", \"unknown\"),
        \"environment\": {\"python_executable\": sys.executable},
    }))
    raise SystemExit(0)


def input_tensor(target_device, target_dtype):
    rows = []
    for token in range(batch_tokens):
        rows.append([(((token + 1) * (dim + 5)) % 23 - 11) / 23.0 for dim in range(hidden_dim)])
    return torch.tensor(rows, device=target_device, dtype=target_dtype)


def up_weights(target_device, target_dtype):
    rows = []
    for expert in range(experts):
        expert_rows = []
        for hidden in range(hidden_dim):
            expert_rows.append([(((expert + 1) * (hidden + 3) * (intermediate + 5)) % 29 - 14) / 29.0 for intermediate in range(intermediate_dim)])
        rows.append(expert_rows)
    return torch.tensor(rows, device=target_device, dtype=target_dtype)


def down_weights(target_device, target_dtype):
    rows = []
    for expert in range(experts):
        expert_rows = []
        for intermediate in range(intermediate_dim):
            expert_rows.append([(((expert + 7) * (intermediate + 2) * (hidden + 11)) % 31 - 15) / 31.0 for hidden in range(hidden_dim)])
        rows.append(expert_rows)
    return torch.tensor(rows, device=target_device, dtype=target_dtype)


def route_pairs(token):
    weights = [1.0 / float(rank + 1) for rank in range(top_k)]
    total = sum(weights)
    return [((token + rank) % experts, weights[rank] / total) for rank in range(top_k)]


def run_once(target_device, target_dtype):
    x = input_tensor(target_device, target_dtype)
    up = up_weights(target_device, target_dtype)
    down = down_weights(target_device, target_dtype)
    out = torch.zeros((batch_tokens, hidden_dim), device=target_device, dtype=target_dtype)
    for token in range(batch_tokens):
        token_x = x[token:token + 1]
        for expert_id, route_weight in route_pairs(token):
            hidden = torch.relu(token_x @ up[expert_id])
            expert_out = hidden @ down[expert_id]
            out[token:token + 1] += float(route_weight) * expert_out
    return out

with torch.no_grad():
    reference = run_once(torch.device(\"cpu\"), torch.float32)
    device_out = run_once(device, dtype)
    if device.type == \"cuda\":
        torch.cuda.synchronize(device)
    max_abs_error = float((device_out.detach().cpu().float() - reference).abs().max().item())
    correctness_passed = bool(max_abs_error <= tolerance)
    for _ in range(warmup):
        _ = run_once(device, dtype)
    if device.type == \"cuda\":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_out = None
        for _ in range(iterations):
            timed_out = run_once(device, dtype)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_s = float(start.elapsed_time(end)) / 1000.0
    else:
        started = time.perf_counter()
        timed_out = None
        for _ in range(iterations):
            timed_out = run_once(device, dtype)
        elapsed_s = time.perf_counter() - started
    timed_cpu = timed_out.detach().cpu().float() if timed_out is not None else device_out.detach().cpu().float()
    checksum_weights = torch.tensor(
        [[1.0 + token * 0.01 + dim * 0.0001 for dim in range(hidden_dim)] for token in range(batch_tokens)],
        dtype=torch.float32,
    )
    checksum = float((timed_cpu * checksum_weights).sum().item())
    reference_checksum = float((reference * checksum_weights).sum().item())

tokens_processed = iterations * batch_tokens
expert_calls = tokens_processed * top_k
hardware = {
    \"device_type\": device.type,
    \"device\": str(device),
    \"name\": \"cpu\",
}
if device.type == \"cuda\":
    props = torch.cuda.get_device_properties(device)
    hardware = {
        \"device_type\": \"cuda\",
        \"device\": str(device),
        \"index\": int(device.index or 0),
        \"name\": torch.cuda.get_device_name(device),
        \"capability\": list(torch.cuda.get_device_capability(device)),
        \"total_memory_bytes\": int(props.total_memory),
    }

print(json.dumps({
    \"version\": 1,
    \"probe_kind\": \"expert-mlp-accelerator-probe\",
    \"tier\": \"T2-single-node-accelerator\" if device.type == \"cuda\" else \"T0/T1-reference\",
    \"measured\": True,
    \"accelerator_measured\": device.type == \"cuda\",
    \"backend\": \"torch\",
    \"available\": True,
    \"source\": \"fornax.accelerator_probe.torch_expert_mlp\",
    \"config\": {
        \"iterations\": iterations,
        \"warmup\": warmup,
        \"batch_tokens\": batch_tokens,
        \"hidden_dim\": hidden_dim,
        \"intermediate_dim\": intermediate_dim,
        \"experts\": experts,
        \"top_k\": top_k,
        \"dtype\": dtype_name,
        \"tolerance\": tolerance,
    },
    \"result\": {
        \"elapsed_s\": elapsed_s,
        \"elapsed_ns\": int(elapsed_s * 1000000000),
        \"tokens_processed\": tokens_processed,
        \"expert_calls\": expert_calls,
        \"tokens_s\": tokens_processed / elapsed_s if elapsed_s > 0 else None,
        \"expert_calls_s\": expert_calls / elapsed_s if elapsed_s > 0 else None,
        \"checksum\": checksum,
        \"reference_checksum\": reference_checksum,
        \"max_abs_error\": max_abs_error,
        \"correctness_passed\": correctness_passed,
    },
    \"environment\": {
        \"python_executable\": sys.executable,
        \"python_version\": sys.version.split()[0],
        \"platform\": platform.platform(),
        \"machine\": platform.machine(),
        \"cpu_count\": os.cpu_count(),
        \"torch_version\": getattr(torch, \"__version__\", \"unknown\"),
        \"cuda_available\": bool(torch.cuda.is_available()),
        \"cuda_device_count\": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        \"cuda_version\": getattr(torch.version, \"cuda\", None),
    },
    \"hardware\": hardware,
    \"note\": \"Measured expert-MLP microprobe. CUDA output is T2 single-node accelerator evidence for this tiny operation only, not target-model parity.\",
}))
"""


def run_torch_expert_mlp_probe(
    *,
    torch_python: str | None = None,
    device: str = "cuda:0",
    dtype: str = "float16",
    iterations: int = 25,
    warmup: int = 3,
    batch_tokens: int = 8,
    hidden_dim: int = 64,
    intermediate_dim: int = 128,
    experts: int = 4,
    top_k: int = 2,
    tolerance: float = 1e-1,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run the expert-MLP probe in a torch-capable Python process."""

    _validate_config(
        iterations=iterations,
        warmup=warmup,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        experts=experts,
        top_k=top_k,
        tolerance=tolerance,
    )
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    _positive_number("timeout_s", timeout_s)
    python = torch_python or sys.executable
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                _external_torch_script(),
                str(iterations),
                str(warmup),
                str(batch_tokens),
                str(hidden_dim),
                str(intermediate_dim),
                str(experts),
                str(top_k),
                device,
                dtype,
                str(tolerance),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "version": 1,
            "probe_kind": PROBE_KIND,
            "tier": "T2-single-node-accelerator" if device.startswith("cuda") else "T0/T1-reference",
            "measured": False,
            "accelerator_measured": False,
            "backend": "torch",
            "available": False,
            "source": "fornax.accelerator_probe.torch_expert_mlp",
            "error": f"torch expert-MLP probe failed to launch: {type(exc).__name__}: {exc}",
            "environment": {"python_executable": python},
        }
    stdout = result.stdout.strip()
    if result.returncode != 0:
        return {
            "version": 1,
            "probe_kind": PROBE_KIND,
            "tier": "T2-single-node-accelerator" if device.startswith("cuda") else "T0/T1-reference",
            "measured": False,
            "accelerator_measured": False,
            "backend": "torch",
            "available": False,
            "source": "fornax.accelerator_probe.torch_expert_mlp",
            "returncode": result.returncode,
            "stdout": stdout[-2000:],
            "stderr": result.stderr.strip()[-2000:],
            "error": "torch expert-MLP probe exited nonzero",
            "environment": {"python_executable": python},
        }
    try:
        data = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return {
            "version": 1,
            "probe_kind": PROBE_KIND,
            "tier": "T2-single-node-accelerator" if device.startswith("cuda") else "T0/T1-reference",
            "measured": False,
            "accelerator_measured": False,
            "backend": "torch",
            "available": False,
            "source": "fornax.accelerator_probe.torch_expert_mlp",
            "stdout": stdout[-2000:],
            "stderr": result.stderr.strip()[-2000:],
            "error": f"torch expert-MLP probe did not emit JSON: {exc}",
            "environment": {"python_executable": python},
        }
    if not isinstance(data, dict):
        return {
            "version": 1,
            "probe_kind": PROBE_KIND,
            "tier": "T2-single-node-accelerator" if device.startswith("cuda") else "T0/T1-reference",
            "measured": False,
            "accelerator_measured": False,
            "backend": "torch",
            "available": False,
            "source": "fornax.accelerator_probe.torch_expert_mlp",
            "error": "torch expert-MLP probe JSON was not an object",
            "environment": {"python_executable": python},
        }
    data.setdefault("source", "fornax.accelerator_probe.torch_expert_mlp")
    data.setdefault("backend", "torch")
    data.setdefault("environment", {})
    if isinstance(data["environment"], dict):
        data["environment"].setdefault("python_executable", python)
    return data


def run_expert_mlp_probe(
    *,
    backend: str = "torch",
    torch_python: str | None = None,
    device: str = "cuda:0",
    dtype: str = "float16",
    iterations: int = 25,
    warmup: int = 3,
    batch_tokens: int = 8,
    hidden_dim: int = 64,
    intermediate_dim: int = 128,
    experts: int = 4,
    top_k: int = 2,
    tolerance: float = 1e-1,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    if backend == "cpu-stdlib":
        return run_cpu_expert_mlp_probe(
            iterations=iterations,
            warmup=warmup,
            batch_tokens=batch_tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            experts=experts,
            top_k=top_k,
            tolerance=tolerance,
        )
    return run_torch_expert_mlp_probe(
        torch_python=torch_python,
        device=device,
        dtype=dtype,
        iterations=iterations,
        warmup=warmup,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        experts=experts,
        top_k=top_k,
        tolerance=tolerance,
        timeout_s=timeout_s,
    )


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


def validate_expert_mlp_probe_fixture(data: dict[str, Any]) -> dict[str, Any]:
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
    batch_tokens = _positive_int_field(config.get("batch_tokens"), "config.batch_tokens", errors)
    top_k = _positive_int_field(config.get("top_k"), "config.top_k", errors)
    experts = _positive_int_field(config.get("experts"), "config.experts", errors)
    _positive_int_field(config.get("hidden_dim"), "config.hidden_dim", errors)
    _positive_int_field(config.get("intermediate_dim"), "config.intermediate_dim", errors)
    _non_negative_number_field(config.get("tolerance"), "config.tolerance", errors)
    dtype = _non_empty_string(config.get("dtype"), "config.dtype", errors)
    if backend == "torch" and dtype is not None and dtype not in DTYPES:
        errors.append(f"config.dtype must be one of {sorted(DTYPES)}")
    if top_k is not None and experts is not None and top_k > experts:
        errors.append("config.top_k cannot exceed config.experts")

    result = data.get("result")
    if not isinstance(result, dict):
        if measured:
            errors.append("result must be an object for measured probes")
        result = {}
    if measured:
        tokens_processed = _positive_int_field(
            result.get("tokens_processed"), "result.tokens_processed", errors
        )
        expert_calls = _positive_int_field(result.get("expert_calls"), "result.expert_calls", errors)
        _positive_number_field(result.get("elapsed_s"), "result.elapsed_s", errors)
        _positive_number_field(result.get("tokens_s"), "result.tokens_s", errors)
        _positive_number_field(result.get("expert_calls_s"), "result.expert_calls_s", errors)
        _non_negative_number_field(result.get("max_abs_error"), "result.max_abs_error", errors)
        _number_field(result.get("checksum"), "result.checksum", errors)
        _number_field(
            result.get("reference_checksum"), "result.reference_checksum", errors
        )
        if result.get("correctness_passed") is not True:
            errors.append("result.correctness_passed must be true for measured probe evidence")
        if iterations is not None and batch_tokens is not None and tokens_processed is not None:
            if tokens_processed != iterations * batch_tokens:
                errors.append("result.tokens_processed must equal iterations * batch_tokens")
        if tokens_processed is not None and top_k is not None and expert_calls is not None:
            if expert_calls != tokens_processed * top_k:
                errors.append("result.expert_calls must equal tokens_processed * top_k")
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
        _non_empty_string(hardware.get("device"), "hardware.device", errors)
        _non_empty_string(hardware.get("name"), "hardware.name", errors)
        if accelerator_measured:
            if device_type != "cuda":
                errors.append("hardware.device_type must be cuda when accelerator_measured is true")
            _positive_int_field(hardware.get("total_memory_bytes"), "hardware.total_memory_bytes", errors)
        elif tier == "T2-single-node-accelerator":
            errors.append("T2-single-node-accelerator probes must set accelerator_measured true")
    if backend == "cpu-stdlib" and accelerator_measured:
        errors.append("cpu-stdlib backend cannot be accelerator_measured")
    if measured and not accelerator_measured:
        warnings.append("probe is measured but not accelerator evidence")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "tier": tier,
            "backend": backend,
            "measured": bool(measured),
            "accelerator_measured": bool(accelerator_measured),
            "device": hardware.get("device") if isinstance(hardware, dict) else None,
            "device_name": hardware.get("name") if isinstance(hardware, dict) else None,
            "tokens_s": result.get("tokens_s") if isinstance(result, dict) else None,
            "expert_calls_s": result.get("expert_calls_s") if isinstance(result, dict) else None,
            "max_abs_error": result.get("max_abs_error") if isinstance(result, dict) else None,
        },
    }


def validate_expert_mlp_probe(path: str | Path) -> dict[str, Any]:
    try:
        data = read_json(path)
    except Exception as exc:  # noqa: BLE001 - validator reports fixture parse failures.
        return {
            "ok": False,
            "errors": [f"invalid expert-MLP probe artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["expert-MLP probe artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(path),
        }
    result = validate_expert_mlp_probe_fixture(data)
    result["fixture"] = str(path)
    return result
