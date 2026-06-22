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

PROBE_KIND = "moe-layer-parity-probe"
BACKENDS = {"cpu-stdlib", "torch"}
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
    _non_negative_number("tolerance", tolerance)


def _input_value(token: int, dim: int) -> float:
    return (((token + 2) * (dim + 5)) % 31 - 15) / 31.0


def _up_weight(expert_id: int, hidden: int, intermediate: int) -> float:
    return (((expert_id + 3) * (hidden + 7) * (intermediate + 11)) % 37 - 18) / 37.0


def _down_weight(expert_id: int, intermediate: int, hidden: int) -> float:
    return (((expert_id + 5) * (intermediate + 13) * (hidden + 17)) % 41 - 20) / 41.0


def _logit_weight(hidden: int, vocab: int) -> float:
    return (((hidden + 11) * (vocab + 13)) % 43 - 21) / 43.0


def _inputs(token_count: int, hidden_dim: int) -> list[list[float]]:
    return [[_input_value(token, dim) for dim in range(hidden_dim)] for token in range(token_count)]


def _route_trace(token_count: int, expert_count: int, top_k: int) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for token in range(token_count):
        expert_ids: list[int] = []
        candidate = token % expert_count
        while len(expert_ids) < top_k:
            if candidate not in expert_ids:
                expert_ids.append(candidate)
            candidate = (candidate + 1) % expert_count
        raw = [float(top_k - rank) + 0.25 * ((token + rank) % 3) for rank in range(top_k)]
        total = sum(raw)
        routes.append(
            {
                "token_index": token,
                "expert_ids": expert_ids,
                "topk_weights": [value / total for value in raw],
            }
        )
    return routes


def _local_expert_ids(expert_count: int) -> list[int]:
    return [expert_id for expert_id in range(expert_count) if expert_id % 2 == 0]


def _remote_expert_ids(expert_count: int) -> list[int]:
    return [expert_id for expert_id in range(expert_count) if expert_id % 2 == 1]


def _expert_output(row: list[float], *, expert_id: int, intermediate_dim: int) -> list[float]:
    hidden_dim = len(row)
    intermediate_values: list[float] = []
    for intermediate in range(intermediate_dim):
        acc = 0.0
        for hidden, value in enumerate(row):
            acc += value * _up_weight(expert_id, hidden, intermediate)
        intermediate_values.append(acc if acc > 0.0 else 0.0)
    output: list[float] = []
    for hidden in range(hidden_dim):
        acc = 0.0
        for intermediate, value in enumerate(intermediate_values):
            acc += value * _down_weight(expert_id, intermediate, hidden)
        output.append(acc)
    return output


def _project_logits(values: list[list[float]], vocab_size: int) -> list[list[float]]:
    logits: list[list[float]] = []
    for row in values:
        out_row: list[float] = []
        for vocab in range(vocab_size):
            acc = 0.0
            for hidden, value in enumerate(row):
                acc += value * _logit_weight(hidden, vocab)
            out_row.append(acc)
        logits.append(out_row)
    return logits


def _argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _run_reference_layer(
    inputs: list[list[float]],
    routes: list[dict[str, Any]],
    *,
    intermediate_dim: int,
    vocab_size: int,
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    layer_output: list[list[float]] = []
    for token_index, row in enumerate(inputs):
        gathered = [0.0 for _ in row]
        route = routes[token_index]
        for expert_id, weight in zip(route["expert_ids"], route["topk_weights"]):
            expert = _expert_output(row, expert_id=int(expert_id), intermediate_dim=intermediate_dim)
            for hidden, value in enumerate(expert):
                gathered[hidden] += float(weight) * value
        layer_output.append([value + gathered[hidden] for hidden, value in enumerate(row)])
    logits = _project_logits(layer_output, vocab_size)
    return layer_output, logits, [_argmax(row) for row in logits]


def _run_split_layer(
    inputs: list[list[float]],
    routes: list[dict[str, Any]],
    *,
    intermediate_dim: int,
    vocab_size: int,
    remote_expert_ids: set[int],
) -> tuple[list[list[float]], list[list[float]], list[int], dict[int, list[int]]]:
    buckets: dict[int, list[int]] = {}
    for route in routes:
        token_index = int(route["token_index"])
        for expert_id in route["expert_ids"]:
            buckets.setdefault(int(expert_id), []).append(token_index)
    expert_results: dict[tuple[int, int], list[float]] = {}
    for expert_id in sorted(buckets):
        token_indices = buckets[expert_id]
        payload = [list(inputs[token_index]) for token_index in token_indices]
        for token_index, row in zip(token_indices, payload):
            expert_results[(token_index, expert_id)] = _expert_output(
                row,
                expert_id=expert_id,
                intermediate_dim=intermediate_dim,
            )
    layer_output: list[list[float]] = []
    for token_index, row in enumerate(inputs):
        gathered = [0.0 for _ in row]
        route = routes[token_index]
        for expert_id, weight in zip(route["expert_ids"], route["topk_weights"]):
            expert = expert_results[(token_index, int(expert_id))]
            for hidden, value in enumerate(expert):
                gathered[hidden] += float(weight) * value
        layer_output.append([value + gathered[hidden] for hidden, value in enumerate(row)])
    logits = _project_logits(layer_output, vocab_size)
    remote_buckets = {
        expert_id: token_indices
        for expert_id, token_indices in buckets.items()
        if expert_id in remote_expert_ids
    }
    return layer_output, logits, [_argmax(row) for row in logits], remote_buckets


def _checksum(matrix: list[list[float]]) -> float:
    total = 0.0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            total += value * (1.0 + row_index * 0.01 + col_index * 0.0001)
    return total


def _max_abs_error(left: list[list[float]], right: list[list[float]]) -> float:
    max_error = 0.0
    for row_left, row_right in zip(left, right):
        for value_left, value_right in zip(row_left, row_right):
            max_error = max(max_error, abs(value_left - value_right))
    return max_error


def _remote_token_copies(routes: list[dict[str, Any]], remote_expert_ids: set[int]) -> int:
    return sum(1 for route in routes for expert_id in route["expert_ids"] if int(expert_id) in remote_expert_ids)


def _local_token_copies(routes: list[dict[str, Any]], remote_expert_ids: set[int]) -> int:
    return sum(1 for route in routes for expert_id in route["expert_ids"] if int(expert_id) not in remote_expert_ids)


def run_cpu_moe_layer_parity_probe(
    *,
    iterations: int = 5,
    warmup: int = 1,
    token_count: int = 4,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    vocab_size: int = 17,
    expert_count: int = 4,
    top_k: int = 2,
    tolerance: float = 0.0,
    logical_source_host: str = "logical-host-0",
    logical_expert_host: str = "logical-host-1",
) -> dict[str, Any]:
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
    )
    if not logical_source_host or not logical_expert_host:
        raise ValueError("logical host names must be non-empty")
    if logical_source_host == logical_expert_host:
        raise ValueError("logical host names must differ")
    inputs = _inputs(token_count, hidden_dim)
    routes = _route_trace(token_count, expert_count, top_k)
    local_ids = _local_expert_ids(expert_count)
    remote_ids = _remote_expert_ids(expert_count)
    remote_id_set = set(remote_ids)
    reference_layer, reference_logits, reference_tokens = _run_reference_layer(
        inputs,
        routes,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
    )
    for _ in range(warmup):
        _run_split_layer(
            inputs,
            routes,
            intermediate_dim=intermediate_dim,
            vocab_size=vocab_size,
            remote_expert_ids=remote_id_set,
        )
    started_ns = time.perf_counter_ns()
    layer_output: list[list[float]] = []
    logits: list[list[float]] = []
    next_tokens: list[int] = []
    remote_buckets: dict[int, list[int]] = {}
    for _ in range(iterations):
        layer_output, logits, next_tokens, remote_buckets = _run_split_layer(
            inputs,
            routes,
            intermediate_dim=intermediate_dim,
            vocab_size=vocab_size,
            remote_expert_ids=remote_id_set,
        )
    elapsed_ns = time.perf_counter_ns() - started_ns
    elapsed_s = elapsed_ns / 1_000_000_000.0
    max_layer_abs_error = _max_abs_error(reference_layer, layer_output)
    max_logit_abs_error = _max_abs_error(reference_logits, logits)
    remote_copies = _remote_token_copies(routes, remote_id_set)
    local_copies = _local_token_copies(routes, remote_id_set)
    transfer_payload_bytes_per_iteration = remote_copies * hidden_dim * 8 * 2
    expert_calls = iterations * token_count * top_k
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T0/T1-reference",
        "measured": True,
        "accelerator_measured": False,
        "backend": "cpu-stdlib",
        "available": True,
        "source": "fornax.moe_parity.cpu_moe_layer_parity.stdlib",
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "token_count": token_count,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "vocab_size": vocab_size,
            "expert_count": expert_count,
            "top_k": top_k,
            "local_expert_ids": local_ids,
            "remote_expert_ids": remote_ids,
            "source_device": "cpu",
            "expert_device": "cpu",
            "logical_source_host": logical_source_host,
            "logical_expert_host": logical_expert_host,
            "dtype": "float64-reference",
            "tolerance": tolerance,
            "remote_token_copies_per_iteration": remote_copies,
            "local_token_copies_per_iteration": local_copies,
            "remote_bucket_count_per_iteration": len(remote_buckets),
            "transfer_payload_bytes_per_iteration": transfer_payload_bytes_per_iteration,
        },
        "routing": {
            "routes": routes,
            "expert_placement": [
                {
                    "expert_id": expert_id,
                    "logical_host": logical_source_host if expert_id in local_ids else logical_expert_host,
                    "local_to_source": expert_id in local_ids,
                }
                for expert_id in range(expert_count)
            ],
        },
        "result": {
            "elapsed_s": elapsed_s,
            "elapsed_ns": elapsed_ns,
            "tokens_processed": iterations * token_count,
            "expert_calls": expert_calls,
            "local_expert_calls": iterations * local_copies,
            "remote_expert_calls": iterations * remote_copies,
            "remote_batches": iterations * len(remote_buckets),
            "transfer_payload_bytes": iterations * transfer_payload_bytes_per_iteration,
            "tokens_s": (iterations * token_count) / elapsed_s if elapsed_s > 0 else None,
            "expert_calls_s": expert_calls / elapsed_s if elapsed_s > 0 else None,
            "layer_checksum": _checksum(layer_output),
            "reference_layer_checksum": _checksum(reference_layer),
            "logit_checksum": _checksum(logits),
            "reference_logit_checksum": _checksum(reference_logits),
            "max_layer_abs_error": max_layer_abs_error,
            "max_logit_abs_error": max_logit_abs_error,
            "routing_match": True,
            "next_tokens": next_tokens,
            "reference_next_tokens": reference_tokens,
            "next_tokens_match": next_tokens == reference_tokens,
            "correctness_passed": (
                next_tokens == reference_tokens
                and max_layer_abs_error <= tolerance
                and max_logit_abs_error <= tolerance
            ),
            "timing_method": "perf_counter_ns_cpu_moe_layer_parity",
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "hardware": {
            "device_type": "cpu-moe-layer-parity",
            "source_device": "cpu",
            "expert_device": "cpu",
            "source_name": platform.processor() or platform.machine(),
            "expert_name": platform.processor() or platform.machine(),
            "same_physical_host": True,
            "logical_hosts": [logical_source_host, logical_expert_host],
        },
        "note": (
            "CPU reference MoE layer/logit parity probe. Validates router top-k, "
            "local and remote expert bucketing, weighted gather, and logit parity, "
            "but is not T3 accelerator evidence."
        ),
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
source_device_name = sys.argv[9]
expert_device_name = sys.argv[10]
dtype_name = sys.argv[11]
tolerance = float(sys.argv[12])
logical_source_host = sys.argv[13]
logical_expert_host = sys.argv[14]
probe_kind = "moe-layer-parity-probe"
tier = "T3-same-host-moe-parity-simulation"


def emit_unavailable(error):
    print(json.dumps({
        "version": 1,
        "probe_kind": probe_kind,
        "tier": tier,
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.moe_parity.torch_moe_layer_parity",
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
local_expert_ids = [expert_id for expert_id in range(expert_count) if expert_id % 2 == 0]
remote_expert_ids = [expert_id for expert_id in range(expert_count) if expert_id % 2 == 1]
remote_expert_set = set(remote_expert_ids)
remote_token_copies = sum(1 for route in routes for expert_id in route["expert_ids"] if expert_id in remote_expert_set)
local_token_copies = sum(1 for route in routes for expert_id in route["expert_ids"] if expert_id not in remote_expert_set)
remote_bucket_count = len({expert_id for route in routes for expert_id in route["expert_ids"] if expert_id in remote_expert_set})


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
    gathered = torch.zeros((token_count, hidden_dim), device=source_device, dtype=torch.float32)
    x_float = x.float()
    for token_index, route in enumerate(routes):
        for expert_id, weight in zip(route["expert_ids"], route["topk_weights"]):
            out = expert_outputs(x_float[token_index:token_index + 1], expert_id, source_device, torch.float32)
            gathered[token_index:token_index + 1] += float(weight) * out
    layer = x_float + gathered
    logits = layer @ logit_weights(source_device, torch.float32)
    return layer, logits, torch.argmax(logits, dim=1)


def run_split(x):
    gathered = torch.zeros((token_count, hidden_dim), device=source_device, dtype=torch.float32)
    for token_index, route in enumerate(routes):
        for expert_id, weight in zip(route["expert_ids"], route["topk_weights"]):
            if expert_id in remote_expert_set:
                payload = x[token_index:token_index + 1].to(expert_device, non_blocking=False)
                out = expert_outputs(payload, expert_id, expert_device, dtype).to(source_device, non_blocking=False).float()
            else:
                out = expert_outputs(x[token_index:token_index + 1], expert_id, source_device, dtype).float()
            gathered[token_index:token_index + 1] += float(weight) * out
    layer = x.float() + gathered
    logits = layer @ logit_weights(source_device, torch.float32)
    return layer, logits, torch.argmax(logits, dim=1)

try:
    try:
        p2p_source_to_expert = bool(torch.cuda.can_device_access_peer(source_index, expert_index))
        p2p_expert_to_source = bool(torch.cuda.can_device_access_peer(expert_index, source_index))
    except Exception:
        p2p_source_to_expert = None
        p2p_expert_to_source = None
    with torch.no_grad():
        x = make_inputs(source_device, dtype)
        reference_layer, reference_logits, reference_tokens = run_reference(x)
        layer, logits, next_tokens = run_split(x)
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(expert_device)
        max_layer_abs_error = float((layer.detach().cpu().float() - reference_layer.detach().cpu().float()).abs().max().item())
        max_logit_abs_error = float((logits.detach().cpu().float() - reference_logits.detach().cpu().float()).abs().max().item())
        next_tokens_match = bool(torch.equal(next_tokens.detach().cpu(), reference_tokens.detach().cpu()))
        correctness_passed = bool(next_tokens_match and max_layer_abs_error <= tolerance and max_logit_abs_error <= tolerance)
        for _ in range(warmup):
            run_split(x)
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(expert_device)
        started = time.perf_counter()
        timed_layer = None
        timed_logits = None
        timed_tokens = None
        for _ in range(iterations):
            timed_layer, timed_logits, timed_tokens = run_split(x)
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(expert_device)
        elapsed_s = time.perf_counter() - started
        timed_layer_cpu = timed_layer.detach().cpu().float() if timed_layer is not None else layer.detach().cpu().float()
        timed_logits_cpu = timed_logits.detach().cpu().float() if timed_logits is not None else logits.detach().cpu().float()
        timed_tokens_cpu = timed_tokens.detach().cpu().tolist() if timed_tokens is not None else next_tokens.detach().cpu().tolist()
        reference_layer_cpu = reference_layer.detach().cpu().float()
        reference_logits_cpu = reference_logits.detach().cpu().float()
        reference_tokens_cpu = reference_tokens.detach().cpu().tolist()
except Exception as exc:
    emit_unavailable(f"MoE parity failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

element_size = int(torch.tensor([], dtype=dtype).element_size())
transfer_payload_bytes_per_iteration = int(remote_token_copies * hidden_dim * element_size * 2)
expert_calls = int(iterations * token_count * top_k)
source_props = torch.cuda.get_device_properties(source_device)
expert_props = torch.cuda.get_device_properties(expert_device)
layer_weights = (1.0 + torch.arange(token_count, dtype=torch.float32).unsqueeze(1) * 0.01) * (1.0 + torch.arange(hidden_dim, dtype=torch.float32).unsqueeze(0) * 0.0001)
logit_weights_for_checksum = (1.0 + torch.arange(token_count, dtype=torch.float32).unsqueeze(1) * 0.01) * (1.0 + torch.arange(vocab_size, dtype=torch.float32).unsqueeze(0) * 0.0001)
print(json.dumps({
    "version": 1,
    "probe_kind": probe_kind,
    "tier": tier,
    "measured": True,
    "accelerator_measured": True,
    "backend": "torch",
    "available": True,
    "source": "fornax.moe_parity.torch_moe_layer_parity",
    "config": {
        "iterations": iterations,
        "warmup": warmup,
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "vocab_size": vocab_size,
        "expert_count": expert_count,
        "top_k": top_k,
        "local_expert_ids": local_expert_ids,
        "remote_expert_ids": remote_expert_ids,
        "source_device": source_device_name,
        "expert_device": expert_device_name,
        "logical_source_host": logical_source_host,
        "logical_expert_host": logical_expert_host,
        "dtype": dtype_name,
        "tolerance": tolerance,
        "remote_token_copies_per_iteration": remote_token_copies,
        "local_token_copies_per_iteration": local_token_copies,
        "remote_bucket_count_per_iteration": remote_bucket_count,
        "transfer_payload_bytes_per_iteration": transfer_payload_bytes_per_iteration,
    },
    "routing": {
        "routes": routes,
        "expert_placement": [
            {"expert_id": expert_id, "logical_host": logical_source_host if expert_id in local_expert_ids else logical_expert_host, "local_to_source": expert_id in local_expert_ids}
            for expert_id in range(expert_count)
        ],
    },
    "result": {
        "elapsed_s": elapsed_s,
        "elapsed_ns": int(elapsed_s * 1000000000),
        "tokens_processed": int(iterations * token_count),
        "expert_calls": expert_calls,
        "local_expert_calls": int(iterations * local_token_copies),
        "remote_expert_calls": int(iterations * remote_token_copies),
        "remote_batches": int(iterations * remote_bucket_count),
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
        "timing_method": "perf_counter_cuda_synchronize_moe_layer_parity",
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
        "device_type": "cuda-moe-layer-parity",
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
    "note": "Measured same-host two-GPU MoE layer/logit parity simulation; not real multi-host T3 closure.",
}))
'''


def _unavailable(error: str, python: str) -> dict[str, Any]:
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T3-same-host-moe-parity-simulation",
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.moe_parity.torch_moe_layer_parity",
        "error": error,
        "environment": {"python_executable": python},
    }


def run_torch_moe_layer_parity_probe(
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
    vocab_size: int = 37,
    expert_count: int = 4,
    top_k: int = 2,
    tolerance: float = 1e-4,
    logical_source_host: str = "logical-host-0",
    logical_expert_host: str = "logical-host-1",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
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
    )
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if not logical_source_host or not logical_expert_host:
        raise ValueError("logical host names must be non-empty")
    if logical_source_host == logical_expert_host:
        raise ValueError("logical host names must differ")
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
                str(token_count),
                str(hidden_dim),
                str(intermediate_dim),
                str(vocab_size),
                str(expert_count),
                str(top_k),
                source_device,
                expert_device,
                dtype,
                str(tolerance),
                logical_source_host,
                logical_expert_host,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(f"torch MoE parity failed to launch: {type(exc).__name__}: {exc}", python)
    stdout = result.stdout.strip()
    if result.returncode != 0:
        data = _unavailable("torch MoE parity exited nonzero", python)
        data.update({"returncode": result.returncode, "stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    try:
        data = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        data = _unavailable(f"torch MoE parity did not emit JSON: {exc}", python)
        data.update({"stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    if not isinstance(data, dict):
        return _unavailable("torch MoE parity JSON was not an object", python)
    data.setdefault("source", "fornax.moe_parity.torch_moe_layer_parity")
    data.setdefault("backend", "torch")
    data.setdefault("environment", {})
    if isinstance(data["environment"], dict):
        data["environment"].setdefault("python_executable", python)
    return data


def run_moe_layer_parity_probe(
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
    vocab_size: int = 17,
    expert_count: int = 4,
    top_k: int = 2,
    tolerance: float = 0.0,
    logical_source_host: str = "logical-host-0",
    logical_expert_host: str = "logical-host-1",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    if backend == "cpu-stdlib":
        return run_cpu_moe_layer_parity_probe(
            iterations=iterations,
            warmup=warmup,
            token_count=token_count,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            vocab_size=vocab_size,
            expert_count=expert_count,
            top_k=top_k,
            tolerance=tolerance,
            logical_source_host=logical_source_host,
            logical_expert_host=logical_expert_host,
        )
    return run_torch_moe_layer_parity_probe(
        torch_python=torch_python,
        source_device=source_device,
        expert_device=expert_device,
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
        logical_source_host=logical_source_host,
        logical_expert_host=logical_expert_host,
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


def _number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field} must be numeric")
        return None
    return float(value)


def _int_list(value: Any, field: str, errors: list[str]) -> list[int] | None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            errors.append(f"{field}[{index}] must be an integer")
            return None
        result.append(item)
    return result


def _cuda_device_name(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith("cuda:") and value[5:].isdigit()


def _validate_routing(
    routing: Any,
    *,
    token_count: int | None,
    top_k: int | None,
    expert_count: int | None,
    errors: list[str],
) -> None:
    if not isinstance(routing, dict):
        errors.append("routing must be an object")
        return
    routes = routing.get("routes")
    if not isinstance(routes, list) or not routes:
        errors.append("routing.routes must be a non-empty list")
        return
    if token_count is not None and len(routes) != token_count:
        errors.append("routing.routes length must equal config.token_count")
    for row_index, route in enumerate(routes):
        if not isinstance(route, dict):
            errors.append(f"routing.routes[{row_index}] must be an object")
            continue
        token_index = route.get("token_index")
        if token_index != row_index:
            errors.append(f"routing.routes[{row_index}].token_index must equal its index")
        expert_ids = _int_list(route.get("expert_ids"), f"routing.routes[{row_index}].expert_ids", errors)
        weights = route.get("topk_weights")
        if not isinstance(weights, list):
            errors.append(f"routing.routes[{row_index}].topk_weights must be a list")
            continue
        if top_k is not None:
            if expert_ids is not None and len(expert_ids) != top_k:
                errors.append(f"routing.routes[{row_index}].expert_ids length must equal config.top_k")
            if len(weights) != top_k:
                errors.append(f"routing.routes[{row_index}].topk_weights length must equal config.top_k")
        if expert_ids is not None:
            if len(set(expert_ids)) != len(expert_ids):
                errors.append(f"routing.routes[{row_index}].expert_ids must be unique")
            if expert_count is not None and any(expert_id < 0 or expert_id >= expert_count for expert_id in expert_ids):
                errors.append(f"routing.routes[{row_index}].expert_ids must be in [0, config.expert_count)")
        numeric_weights: list[float] = []
        for weight_index, weight in enumerate(weights):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                errors.append(f"routing.routes[{row_index}].topk_weights[{weight_index}] must be a positive number")
            else:
                numeric_weights.append(float(weight))
        if len(numeric_weights) == len(weights) and abs(sum(numeric_weights) - 1.0) > 1e-9:
            errors.append(f"routing.routes[{row_index}].topk_weights must sum to 1")
    placements = routing.get("expert_placement")
    if not isinstance(placements, list) or not placements:
        errors.append("routing.expert_placement must be a non-empty list")


def validate_moe_layer_parity_probe_fixture(data: dict[str, Any]) -> dict[str, Any]:
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
    _positive_int_field(config.get("vocab_size"), "config.vocab_size", errors)
    expert_count = _positive_int_field(config.get("expert_count"), "config.expert_count", errors)
    top_k = _positive_int_field(config.get("top_k"), "config.top_k", errors)
    if top_k is not None and expert_count is not None and top_k > expert_count:
        errors.append("config.top_k must be <= config.expert_count")
    local_ids = _int_list(config.get("local_expert_ids"), "config.local_expert_ids", errors)
    remote_ids = _int_list(config.get("remote_expert_ids"), "config.remote_expert_ids", errors)
    if local_ids is not None and remote_ids is not None:
        if set(local_ids) & set(remote_ids):
            errors.append("config.local_expert_ids and config.remote_expert_ids must not overlap")
        if expert_count is not None and sorted(local_ids + remote_ids) != list(range(expert_count)):
            errors.append("config.local_expert_ids plus remote_expert_ids must cover all experts")
    remote_copies = _positive_int_field(config.get("remote_token_copies_per_iteration"), "config.remote_token_copies_per_iteration", errors)
    local_copies = _positive_int_field(config.get("local_token_copies_per_iteration"), "config.local_token_copies_per_iteration", errors)
    remote_bucket_count = _positive_int_field(config.get("remote_bucket_count_per_iteration"), "config.remote_bucket_count_per_iteration", errors)
    transfer_per_iteration = _positive_int_field(config.get("transfer_payload_bytes_per_iteration"), "config.transfer_payload_bytes_per_iteration", errors)
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
    _validate_routing(data.get("routing"), token_count=token_count, top_k=top_k, expert_count=expert_count, errors=errors)
    result = data.get("result")
    if not isinstance(result, dict):
        if measured:
            errors.append("result must be an object for measured probes")
        result = {}
    if measured:
        tokens_processed = _positive_int_field(result.get("tokens_processed"), "result.tokens_processed", errors)
        expert_calls = _positive_int_field(result.get("expert_calls"), "result.expert_calls", errors)
        local_calls = _positive_int_field(result.get("local_expert_calls"), "result.local_expert_calls", errors)
        remote_calls = _positive_int_field(result.get("remote_expert_calls"), "result.remote_expert_calls", errors)
        remote_batches = _positive_int_field(result.get("remote_batches"), "result.remote_batches", errors)
        transfer_payload_bytes = _positive_int_field(result.get("transfer_payload_bytes"), "result.transfer_payload_bytes", errors)
        _positive_number_field(result.get("elapsed_s"), "result.elapsed_s", errors)
        _positive_number_field(result.get("tokens_s"), "result.tokens_s", errors)
        _positive_number_field(result.get("expert_calls_s"), "result.expert_calls_s", errors)
        _number_field(result.get("layer_checksum"), "result.layer_checksum", errors)
        _number_field(result.get("reference_layer_checksum"), "result.reference_layer_checksum", errors)
        _number_field(result.get("logit_checksum"), "result.logit_checksum", errors)
        _number_field(result.get("reference_logit_checksum"), "result.reference_logit_checksum", errors)
        _non_negative_number_field(result.get("max_layer_abs_error"), "result.max_layer_abs_error", errors)
        _non_negative_number_field(result.get("max_logit_abs_error"), "result.max_logit_abs_error", errors)
        _non_empty_string(result.get("timing_method"), "result.timing_method", errors)
        if result.get("routing_match") is not True:
            errors.append("result.routing_match must be true")
        if result.get("next_tokens_match") is not True:
            errors.append("result.next_tokens_match must be true")
        if result.get("correctness_passed") is not True:
            errors.append("result.correctness_passed must be true for measured probe evidence")
        if iterations is not None and token_count is not None and tokens_processed is not None and tokens_processed != iterations * token_count:
            errors.append("result.tokens_processed must equal iterations * token_count")
        if iterations is not None and token_count is not None and top_k is not None and expert_calls is not None and expert_calls != iterations * token_count * top_k:
            errors.append("result.expert_calls must equal iterations * token_count * top_k")
        if local_calls is not None and remote_calls is not None and expert_calls is not None and local_calls + remote_calls != expert_calls:
            errors.append("result.local_expert_calls plus remote_expert_calls must equal result.expert_calls")
        if iterations is not None and remote_copies is not None and remote_calls is not None and remote_calls != iterations * remote_copies:
            errors.append("result.remote_expert_calls must equal iterations * remote_token_copies_per_iteration")
        if iterations is not None and local_copies is not None and local_calls is not None and local_calls != iterations * local_copies:
            errors.append("result.local_expert_calls must equal iterations * local_token_copies_per_iteration")
        if iterations is not None and remote_bucket_count is not None and remote_batches is not None and remote_batches != iterations * remote_bucket_count:
            errors.append("result.remote_batches must equal iterations * remote_bucket_count_per_iteration")
        if iterations is not None and transfer_per_iteration is not None and transfer_payload_bytes is not None and transfer_payload_bytes != iterations * transfer_per_iteration:
            errors.append("result.transfer_payload_bytes must equal iterations * transfer_payload_bytes_per_iteration")
        if token_count is not None:
            next_tokens = result.get("next_tokens")
            reference_next_tokens = result.get("reference_next_tokens")
            if not isinstance(next_tokens, list) or len(next_tokens) != token_count:
                errors.append("result.next_tokens must contain config.token_count integers")
            if not isinstance(reference_next_tokens, list) or len(reference_next_tokens) != token_count:
                errors.append("result.reference_next_tokens must contain config.token_count integers")
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
            if tier != "T3-same-host-moe-parity-simulation":
                errors.append("accelerator MoE parity probes must use T3-same-host-moe-parity-simulation tier")
            if device_type != "cuda-moe-layer-parity":
                errors.append("hardware.device_type must be cuda-moe-layer-parity when accelerator_measured is true")
            if not _cuda_device_name(source_device):
                errors.append("config.source_device must be cuda:<index> for accelerator evidence")
            if not _cuda_device_name(expert_device):
                errors.append("config.expert_device must be cuda:<index> for accelerator evidence")
            if source_device == expert_device and source_device is not None:
                errors.append("config.source_device and config.expert_device must differ")
            _positive_int_field(hardware.get("source_total_memory_bytes"), "hardware.source_total_memory_bytes", errors)
            _positive_int_field(hardware.get("expert_total_memory_bytes"), "hardware.expert_total_memory_bytes", errors)
        elif tier == "T3-same-host-moe-parity-simulation":
            errors.append("T3-same-host-moe-parity-simulation probes must set accelerator_measured true")
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
            "source_device": source_device,
            "expert_device": expert_device,
            "token_count": token_count,
            "expert_count": expert_count,
            "top_k": top_k,
            "remote_batches": result.get("remote_batches") if isinstance(result, dict) else None,
            "expert_calls": result.get("expert_calls") if isinstance(result, dict) else None,
            "tokens_s": result.get("tokens_s") if isinstance(result, dict) else None,
            "expert_calls_s": result.get("expert_calls_s") if isinstance(result, dict) else None,
            "max_layer_abs_error": result.get("max_layer_abs_error") if isinstance(result, dict) else None,
            "max_logit_abs_error": result.get("max_logit_abs_error") if isinstance(result, dict) else None,
        },
    }


def validate_moe_layer_parity_probe(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {"ok": False, "errors": [f"invalid MoE layer parity probe artifact: {exc}"], "warnings": [], "summary": {}, "fixture": str(fixture_path)}
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["MoE layer parity probe artifact must be a JSON object"], "warnings": [], "summary": {}, "fixture": str(fixture_path)}
    result = validate_moe_layer_parity_probe_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
