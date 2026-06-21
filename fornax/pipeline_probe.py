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

PROBE_KIND = "pipeline-correctness-probe"
BACKENDS = {"cpu-stdlib", "torch"}
DTYPES = {"float32", "float16", "bfloat16"}
DEFAULT_PROMPTS = [[1, 2, 3], [4, 5, 6]]


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


def _positive_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _normalize_prompts(prompts: list[list[int]] | None, vocab_size: int) -> list[list[int]]:
    raw = DEFAULT_PROMPTS if prompts is None else prompts
    if not isinstance(raw, list) or not raw:
        raise ValueError("prompts must be a non-empty list")
    normalized: list[list[int]] = []
    for prompt_index, prompt in enumerate(raw):
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"prompts[{prompt_index}] must be a non-empty list")
        row: list[int] = []
        for token_index, token in enumerate(prompt):
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError(f"prompts[{prompt_index}][{token_index}] must be an integer")
            if token < 0 or token >= vocab_size:
                raise ValueError(
                    f"prompts[{prompt_index}][{token_index}] must be in [0, vocab_size)"
                )
            row.append(token)
        normalized.append(row)
    return normalized


def parse_prompts_json(value: str | None, vocab_size: int) -> list[list[int]]:
    if value is None:
        return _normalize_prompts(None, vocab_size)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompts_json must be valid JSON: {exc}") from exc
    return _normalize_prompts(parsed, vocab_size)


def _validate_config(
    *,
    iterations: int,
    warmup: int,
    vocab_size: int,
    hidden_dim: int,
    new_tokens: int,
    tolerance: float,
    prompts: list[list[int]] | None,
) -> list[list[int]]:
    _positive_int("iterations", iterations)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    _positive_int("vocab_size", vocab_size)
    _positive_int("hidden_dim", hidden_dim)
    _positive_int("new_tokens", new_tokens)
    _non_negative_number("tolerance", tolerance)
    return _normalize_prompts(prompts, vocab_size)


def _embedding_value(token: int, dim: int) -> float:
    return (((token + 1) * (dim + 3)) % 37 - 18) / 37.0


def _stage0_weight(input_dim: int, output_dim: int) -> float:
    return (((input_dim + 5) * (output_dim + 7)) % 41 - 20) / 41.0


def _stage1_weight(hidden_dim: int, vocab: int) -> float:
    return (((hidden_dim + 11) * (vocab + 13)) % 43 - 21) / 43.0


def _stage0(token: int, hidden_dim: int) -> list[float]:
    embedding = [_embedding_value(token, dim) for dim in range(hidden_dim)]
    hidden: list[float] = []
    for output_dim in range(hidden_dim):
        acc = 0.0
        for input_dim, value in enumerate(embedding):
            acc += value * _stage0_weight(input_dim, output_dim)
        hidden.append(acc if acc > 0.0 else 0.0)
    return hidden


def _stage1_logits(
    hidden: list[float],
    *,
    current_token: int,
    step: int,
    vocab_size: int,
) -> list[float]:
    logits: list[float] = []
    forced_next = (current_token + step + 1) % vocab_size
    for vocab in range(vocab_size):
        acc = 0.0
        for dim, value in enumerate(hidden):
            acc += value * _stage1_weight(dim, vocab)
        if vocab == forced_next:
            acc += 100.0
        logits.append(acc)
    return logits


def _argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _checksum(matrix: list[list[float]]) -> float:
    total = 0.0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            total += value * (1.0 + row_index * 0.01 + col_index * 0.0001)
    return total


def _max_abs_error(a: list[list[float]], b: list[list[float]]) -> float:
    max_error = 0.0
    for row_a, row_b in zip(a, b):
        for value_a, value_b in zip(row_a, row_b):
            max_error = max(max_error, abs(value_a - value_b))
    return max_error


def _run_reference(
    prompts: list[list[int]],
    *,
    vocab_size: int,
    hidden_dim: int,
    new_tokens: int,
) -> tuple[list[list[int]], list[list[float]]]:
    sequences = [list(prompt) for prompt in prompts]
    final_logits: list[list[float]] = []
    for sequence in sequences:
        current = sequence[-1]
        logits: list[float] = []
        for step in range(new_tokens):
            hidden = _stage0(current, hidden_dim)
            logits = _stage1_logits(
                hidden,
                current_token=current,
                step=step,
                vocab_size=vocab_size,
            )
            current = _argmax(logits)
            sequence.append(current)
        final_logits.append(logits)
    return sequences, final_logits


def _run_cpu_pipeline(
    prompts: list[list[int]],
    *,
    vocab_size: int,
    hidden_dim: int,
    new_tokens: int,
) -> tuple[list[list[int]], list[list[float]]]:
    sequences = [list(prompt) for prompt in prompts]
    final_logits: list[list[float]] = []
    for sequence in sequences:
        current = sequence[-1]
        logits: list[float] = []
        for step in range(new_tokens):
            transferred_activation = list(_stage0(current, hidden_dim))
            logits = _stage1_logits(
                transferred_activation,
                current_token=current,
                step=step,
                vocab_size=vocab_size,
            )
            current = _argmax(logits)
            sequence.append(current)
        final_logits.append(logits)
    return sequences, final_logits


def run_cpu_pipeline_correctness_probe(
    *,
    iterations: int = 5,
    warmup: int = 1,
    vocab_size: int = 17,
    hidden_dim: int = 16,
    new_tokens: int = 4,
    prompts: list[list[int]] | None = None,
    tolerance: float = 0.0,
    logical_source_host: str = "logical-host-0",
    logical_destination_host: str = "logical-host-1",
) -> dict[str, Any]:
    """Run a deterministic CPU split-pipeline reference probe."""

    normalized = _validate_config(
        iterations=iterations,
        warmup=warmup,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        new_tokens=new_tokens,
        tolerance=tolerance,
        prompts=prompts,
    )
    if not logical_source_host or not logical_destination_host:
        raise ValueError("logical host names must be non-empty")
    reference_sequences, reference_logits = _run_reference(
        normalized,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        new_tokens=new_tokens,
    )
    for _ in range(warmup):
        _run_cpu_pipeline(
            normalized,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            new_tokens=new_tokens,
        )
    started_ns = time.perf_counter_ns()
    pipeline_sequences: list[list[int]] = []
    pipeline_logits: list[list[float]] = []
    for _ in range(iterations):
        pipeline_sequences, pipeline_logits = _run_cpu_pipeline(
            normalized,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            new_tokens=new_tokens,
        )
    elapsed_ns = time.perf_counter_ns() - started_ns
    elapsed_s = elapsed_ns / 1_000_000_000.0
    max_abs_error = _max_abs_error(reference_logits, pipeline_logits)
    sequences_match = pipeline_sequences == reference_sequences
    prompt_count = len(normalized)
    tokens_generated = iterations * prompt_count * new_tokens
    activation_payload_bytes = prompt_count * hidden_dim * 8
    activation_transfers = iterations * new_tokens
    activation_bytes = activation_transfers * activation_payload_bytes
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T0/T1-reference",
        "measured": True,
        "accelerator_measured": False,
        "backend": "cpu-stdlib",
        "available": True,
        "source": "fornax.pipeline_probe.cpu_pipeline_correctness.stdlib",
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "new_tokens": new_tokens,
            "prompts": normalized,
            "prompt_count": prompt_count,
            "source_device": "cpu",
            "destination_device": "cpu",
            "logical_source_host": logical_source_host,
            "logical_destination_host": logical_destination_host,
            "dtype": "float64-reference",
            "tolerance": tolerance,
            "activation_payload_bytes": activation_payload_bytes,
        },
        "result": {
            "elapsed_s": elapsed_s,
            "elapsed_ns": elapsed_ns,
            "tokens_generated": tokens_generated,
            "tokens_s": tokens_generated / elapsed_s if elapsed_s > 0 else None,
            "activation_transfers": activation_transfers,
            "activation_bytes_transferred": activation_bytes,
            "generated_sequences": pipeline_sequences,
            "reference_generated_sequences": reference_sequences,
            "sequences_match": sequences_match,
            "logit_checksum": _checksum(pipeline_logits),
            "reference_logit_checksum": _checksum(reference_logits),
            "max_abs_error": max_abs_error,
            "correctness_passed": sequences_match and max_abs_error <= tolerance,
            "timing_method": "perf_counter_ns_cpu_split_pipeline",
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "hardware": {
            "device_type": "cpu-pipeline",
            "source_device": "cpu",
            "destination_device": "cpu",
            "source_name": platform.processor() or platform.machine(),
            "destination_name": platform.processor() or platform.machine(),
            "same_physical_host": True,
            "logical_hosts": [logical_source_host, logical_destination_host],
        },
        "note": (
            "CPU reference split-pipeline correctness probe. Validates generated-token "
            "and final-logit parity for a deterministic small model, but is not T3 "
            "accelerator evidence."
        ),
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
vocab_size = int(sys.argv[3])
hidden_dim = int(sys.argv[4])
new_tokens = int(sys.argv[5])
prompts = json.loads(sys.argv[6])
source_device_name = sys.argv[7]
destination_device_name = sys.argv[8]
dtype_name = sys.argv[9]
tolerance = float(sys.argv[10])
logical_source_host = sys.argv[11]
logical_destination_host = sys.argv[12]
probe_kind = "pipeline-correctness-probe"
tier = "T3-same-host-two-gpu-simulation"


def emit_unavailable(error):
    print(json.dumps({
        "version": 1,
        "probe_kind": probe_kind,
        "tier": tier,
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.pipeline_probe.torch_pipeline_correctness",
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
    destination_index = cuda_index(destination_device_name)
    if source_index == destination_index:
        raise ValueError("source and destination CUDA devices must differ")
    device_count = int(torch.cuda.device_count())
    if source_index >= device_count or destination_index >= device_count:
        raise ValueError(
            f"device index out of range for cuda_device_count={device_count}: "
            f"{source_index}->{destination_index}"
        )
    source_device = torch.device(source_device_name)
    destination_device = torch.device(destination_device_name)
except Exception as exc:
    emit_unavailable(f"device setup failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

try:
    normalized_prompts = []
    for prompt in prompts:
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("prompts must contain non-empty token lists")
        row = []
        for token in prompt:
            if not isinstance(token, int) or token < 0 or token >= vocab_size:
                raise ValueError("prompt tokens must be integers in [0, vocab_size)")
            row.append(token)
        normalized_prompts.append(row)
    if not normalized_prompts:
        raise ValueError("prompts must be non-empty")
except Exception as exc:
    emit_unavailable(f"prompt validation failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)


def embedding_table(device, target_dtype):
    tokens = torch.arange(vocab_size, device=device, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((tokens + 1) * (dims + 3), 37.0)
    return ((values - 18.0) / 37.0).to(dtype=target_dtype)


def stage0_weights(device, target_dtype):
    left = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(1)
    right = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((left + 5) * (right + 7), 41.0)
    return ((values - 20.0) / 41.0).to(dtype=target_dtype)


def stage1_weights(device, target_dtype):
    left = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(1)
    right = torch.arange(vocab_size, device=device, dtype=torch.float32).unsqueeze(0)
    values = torch.remainder((left + 11) * (right + 13), 43.0)
    return ((values - 21.0) / 43.0).to(dtype=target_dtype)


def checksum(matrix):
    rows = torch.arange(matrix.shape[0], dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(matrix.shape[1], dtype=torch.float32).unsqueeze(0)
    weights = 1.0 + rows * 0.01 + cols * 0.0001
    return float((matrix.detach().cpu().float() * weights).sum().item())


def run_reference():
    emb = embedding_table(torch.device("cpu"), torch.float32)
    w0 = stage0_weights(torch.device("cpu"), torch.float32)
    w1 = stage1_weights(torch.device("cpu"), torch.float32)
    sequences = [list(prompt) for prompt in normalized_prompts]
    final_logits = []
    for sequence in sequences:
        current = torch.tensor([sequence[-1]], dtype=torch.long)
        logits = torch.empty((1, vocab_size), dtype=torch.float32)
        for step in range(new_tokens):
            hidden = torch.relu(emb[current] @ w0)
            logits = hidden @ w1
            forced = int((int(current.item()) + step + 1) % vocab_size)
            logits[0, forced] += 100.0
            current = torch.argmax(logits, dim=-1)
            sequence.append(int(current.item()))
        final_logits.append(logits.squeeze(0))
    return sequences, torch.stack(final_logits)


def run_pipeline():
    emb = embedding_table(source_device, dtype)
    w0 = stage0_weights(source_device, dtype)
    w1 = stage1_weights(destination_device, dtype)
    sequences = [list(prompt) for prompt in normalized_prompts]
    final_logits = []
    for sequence in sequences:
        current = torch.tensor([sequence[-1]], device=source_device, dtype=torch.long)
        logits = torch.empty((1, vocab_size), device=destination_device, dtype=dtype)
        for step in range(new_tokens):
            hidden = torch.relu(emb[current] @ w0)
            transferred = hidden.to(destination_device, non_blocking=False)
            logits = transferred @ w1
            forced = int((int(current.detach().cpu().item()) + step + 1) % vocab_size)
            logits[:, forced] += 100.0
            next_token = torch.argmax(logits.float(), dim=-1)
            sequence.append(int(next_token.detach().cpu().item()))
            current = next_token.to(source_device)
        final_logits.append(logits.squeeze(0).detach().cpu().float())
    return sequences, torch.stack(final_logits)

try:
    try:
        p2p_source_to_destination = bool(
            torch.cuda.can_device_access_peer(source_index, destination_index)
        )
        p2p_destination_to_source = bool(
            torch.cuda.can_device_access_peer(destination_index, source_index)
        )
    except Exception:
        p2p_source_to_destination = None
        p2p_destination_to_source = None
    with torch.no_grad():
        reference_sequences, reference_logits = run_reference()
        pipeline_sequences, pipeline_logits = run_pipeline()
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(destination_device)
        max_abs_error = float((pipeline_logits - reference_logits).abs().max().item())
        sequences_match = bool(pipeline_sequences == reference_sequences)
        for _ in range(warmup):
            run_pipeline()
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(destination_device)
        started = time.perf_counter()
        timed_sequences = None
        timed_logits = None
        for _ in range(iterations):
            timed_sequences, timed_logits = run_pipeline()
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(destination_device)
        elapsed_s = time.perf_counter() - started
except Exception as exc:
    emit_unavailable(f"pipeline probe failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

prompt_count = len(normalized_prompts)
tokens_generated = int(iterations * prompt_count * new_tokens)
element_size = int(torch.tensor([], dtype=dtype).element_size())
activation_payload_bytes = int(prompt_count * hidden_dim * element_size)
activation_transfers = int(iterations * new_tokens)
activation_bytes = int(activation_transfers * activation_payload_bytes)
source_props = torch.cuda.get_device_properties(source_device)
destination_props = torch.cuda.get_device_properties(destination_device)
print(json.dumps({
    "version": 1,
    "probe_kind": probe_kind,
    "tier": tier,
    "measured": True,
    "accelerator_measured": True,
    "backend": "torch",
    "available": True,
    "source": "fornax.pipeline_probe.torch_pipeline_correctness",
    "config": {
        "iterations": iterations,
        "warmup": warmup,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "new_tokens": new_tokens,
        "prompts": normalized_prompts,
        "prompt_count": prompt_count,
        "source_device": source_device_name,
        "destination_device": destination_device_name,
        "logical_source_host": logical_source_host,
        "logical_destination_host": logical_destination_host,
        "dtype": dtype_name,
        "tolerance": tolerance,
        "activation_payload_bytes": activation_payload_bytes,
    },
    "result": {
        "elapsed_s": elapsed_s,
        "elapsed_ns": int(elapsed_s * 1000000000),
        "tokens_generated": tokens_generated,
        "tokens_s": tokens_generated / elapsed_s if elapsed_s > 0 else None,
        "activation_transfers": activation_transfers,
        "activation_bytes_transferred": activation_bytes,
        "generated_sequences": timed_sequences if timed_sequences is not None else pipeline_sequences,
        "reference_generated_sequences": reference_sequences,
        "sequences_match": sequences_match,
        "logit_checksum": checksum(timed_logits if timed_logits is not None else pipeline_logits),
        "reference_logit_checksum": checksum(reference_logits),
        "max_abs_error": max_abs_error,
        "correctness_passed": bool(sequences_match and max_abs_error <= tolerance),
        "timing_method": "perf_counter_cuda_synchronize_split_pipeline",
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
        "device_type": "cuda-pipeline",
        "source_device": source_device_name,
        "destination_device": destination_device_name,
        "source_index": source_index,
        "destination_index": destination_index,
        "source_name": torch.cuda.get_device_name(source_device),
        "destination_name": torch.cuda.get_device_name(destination_device),
        "source_total_memory_bytes": int(source_props.total_memory),
        "destination_total_memory_bytes": int(destination_props.total_memory),
        "peer_access": {
            "source_to_destination": p2p_source_to_destination,
            "destination_to_source": p2p_destination_to_source,
        },
        "same_physical_host": True,
        "logical_hosts": [logical_source_host, logical_destination_host],
    },
    "note": (
        "Measured same-host two-GPU split-pipeline correctness probe. Treats local "
        "GPUs as logical hosts for development simulation; not proof of real "
        "multi-host T3 cluster closure."
    ),
}))
"""


def run_torch_pipeline_correctness_probe(
    *,
    torch_python: str | None = None,
    source_device: str = "cuda:0",
    destination_device: str = "cuda:1",
    dtype: str = "float32",
    iterations: int = 20,
    warmup: int = 3,
    vocab_size: int = 17,
    hidden_dim: int = 32,
    new_tokens: int = 4,
    prompts: list[list[int]] | None = None,
    tolerance: float = 1e-4,
    logical_source_host: str = "logical-host-0",
    logical_destination_host: str = "logical-host-1",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    normalized = _validate_config(
        iterations=iterations,
        warmup=warmup,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        new_tokens=new_tokens,
        tolerance=tolerance,
        prompts=prompts,
    )
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if not logical_source_host or not logical_destination_host:
        raise ValueError("logical host names must be non-empty")
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
                str(vocab_size),
                str(hidden_dim),
                str(new_tokens),
                json.dumps(normalized),
                source_device,
                destination_device,
                dtype,
                str(tolerance),
                logical_source_host,
                logical_destination_host,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable_artifact(f"torch pipeline-correctness probe failed to launch: {type(exc).__name__}: {exc}", python)
    stdout = result.stdout.strip()
    if result.returncode != 0:
        data = _unavailable_artifact("torch pipeline-correctness probe exited nonzero", python)
        data.update({"returncode": result.returncode, "stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    try:
        data = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        data = _unavailable_artifact(f"torch pipeline-correctness probe did not emit JSON: {exc}", python)
        data.update({"stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    if not isinstance(data, dict):
        return _unavailable_artifact("torch pipeline-correctness probe JSON was not an object", python)
    data.setdefault("source", "fornax.pipeline_probe.torch_pipeline_correctness")
    data.setdefault("backend", "torch")
    data.setdefault("environment", {})
    if isinstance(data["environment"], dict):
        data["environment"].setdefault("python_executable", python)
    return data


def _unavailable_artifact(error: str, python: str) -> dict[str, Any]:
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T3-same-host-two-gpu-simulation",
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.pipeline_probe.torch_pipeline_correctness",
        "error": error,
        "environment": {"python_executable": python},
    }


def run_pipeline_correctness_probe(
    *,
    backend: str = "cpu-stdlib",
    torch_python: str | None = None,
    source_device: str = "cuda:0",
    destination_device: str = "cuda:1",
    dtype: str = "float32",
    iterations: int = 5,
    warmup: int = 1,
    vocab_size: int = 17,
    hidden_dim: int = 16,
    new_tokens: int = 4,
    prompts: list[list[int]] | None = None,
    tolerance: float = 0.0,
    logical_source_host: str = "logical-host-0",
    logical_destination_host: str = "logical-host-1",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    if backend == "cpu-stdlib":
        return run_cpu_pipeline_correctness_probe(
            iterations=iterations,
            warmup=warmup,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            new_tokens=new_tokens,
            prompts=prompts,
            tolerance=tolerance,
            logical_source_host=logical_source_host,
            logical_destination_host=logical_destination_host,
        )
    return run_torch_pipeline_correctness_probe(
        torch_python=torch_python,
        source_device=source_device,
        destination_device=destination_device,
        dtype=dtype,
        iterations=iterations,
        warmup=warmup,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        new_tokens=new_tokens,
        prompts=prompts,
        tolerance=tolerance,
        logical_source_host=logical_source_host,
        logical_destination_host=logical_destination_host,
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


def _cuda_device_name(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("cuda:") and value[5:].isdigit()


def _prompts_field(value: Any, vocab_size: int | None, errors: list[str]) -> list[list[int]] | None:
    if not isinstance(value, list) or not value:
        errors.append("config.prompts must be a non-empty list")
        return None
    normalized: list[list[int]] = []
    for prompt_index, prompt in enumerate(value):
        if not isinstance(prompt, list) or not prompt:
            errors.append(f"config.prompts[{prompt_index}] must be a non-empty list")
            return None
        row: list[int] = []
        for token_index, token in enumerate(prompt):
            if isinstance(token, bool) or not isinstance(token, int):
                errors.append(f"config.prompts[{prompt_index}][{token_index}] must be an integer")
                return None
            if token < 0 or (vocab_size is not None and token >= vocab_size):
                errors.append(f"config.prompts[{prompt_index}][{token_index}] must be in [0, vocab_size)")
                return None
            row.append(token)
        normalized.append(row)
    return normalized


def validate_pipeline_correctness_probe_fixture(data: dict[str, Any]) -> dict[str, Any]:
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
    vocab_size = _positive_int_field(config.get("vocab_size"), "config.vocab_size", errors)
    hidden_dim = _positive_int_field(config.get("hidden_dim"), "config.hidden_dim", errors)
    new_tokens = _positive_int_field(config.get("new_tokens"), "config.new_tokens", errors)
    prompt_count = _positive_int_field(config.get("prompt_count"), "config.prompt_count", errors)
    activation_payload_bytes = _positive_int_field(config.get("activation_payload_bytes"), "config.activation_payload_bytes", errors)
    tolerance = _non_negative_number_field(config.get("tolerance"), "config.tolerance", errors)
    dtype = _non_empty_string(config.get("dtype"), "config.dtype", errors)
    if backend == "torch" and dtype is not None and dtype not in DTYPES:
        errors.append(f"config.dtype must be one of {sorted(DTYPES)}")
    source_device = _non_empty_string(config.get("source_device"), "config.source_device", errors)
    destination_device = _non_empty_string(config.get("destination_device"), "config.destination_device", errors)
    logical_source_host = _non_empty_string(config.get("logical_source_host"), "config.logical_source_host", errors)
    logical_destination_host = _non_empty_string(config.get("logical_destination_host"), "config.logical_destination_host", errors)
    if logical_source_host == logical_destination_host and logical_source_host is not None:
        errors.append("config.logical_source_host must differ from logical_destination_host")
    prompts = _prompts_field(config.get("prompts"), vocab_size, errors)
    if prompts is not None and prompt_count is not None and len(prompts) != prompt_count:
        errors.append("config.prompt_count must equal len(config.prompts)")

    result = data.get("result")
    if not isinstance(result, dict):
        if measured:
            errors.append("result must be an object for measured probes")
        result = {}
    if measured:
        tokens_generated = _positive_int_field(result.get("tokens_generated"), "result.tokens_generated", errors)
        activation_transfers = _positive_int_field(result.get("activation_transfers"), "result.activation_transfers", errors)
        activation_bytes = _positive_int_field(result.get("activation_bytes_transferred"), "result.activation_bytes_transferred", errors)
        _positive_number_field(result.get("elapsed_s"), "result.elapsed_s", errors)
        _positive_number_field(result.get("tokens_s"), "result.tokens_s", errors)
        _number_field(result.get("logit_checksum"), "result.logit_checksum", errors)
        _number_field(result.get("reference_logit_checksum"), "result.reference_logit_checksum", errors)
        max_abs_error = _non_negative_number_field(result.get("max_abs_error"), "result.max_abs_error", errors)
        _non_empty_string(result.get("timing_method"), "result.timing_method", errors)
        if result.get("sequences_match") is not True:
            errors.append("result.sequences_match must be true for pipeline correctness")
        if result.get("correctness_passed") is not True:
            errors.append("result.correctness_passed must be true for measured probe evidence")
        if tolerance is not None and max_abs_error is not None and max_abs_error > tolerance:
            errors.append("result.max_abs_error exceeds config.tolerance")
        generated_sequences = result.get("generated_sequences")
        reference_sequences = result.get("reference_generated_sequences")
        if not isinstance(generated_sequences, list) or not generated_sequences:
            errors.append("result.generated_sequences must be a non-empty list")
        if generated_sequences != reference_sequences:
            errors.append("result.generated_sequences must match reference_generated_sequences")
        if iterations is not None and prompt_count is not None and new_tokens is not None and tokens_generated is not None:
            if tokens_generated != iterations * prompt_count * new_tokens:
                errors.append("result.tokens_generated must equal iterations * prompt_count * new_tokens")
        if iterations is not None and new_tokens is not None and activation_transfers is not None:
            if activation_transfers != iterations * new_tokens:
                errors.append("result.activation_transfers must equal iterations * new_tokens")
        if activation_transfers is not None and activation_payload_bytes is not None and activation_bytes is not None:
            if activation_bytes != activation_transfers * activation_payload_bytes:
                errors.append("result.activation_bytes_transferred must equal activation_transfers * activation_payload_bytes")
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
        hardware_destination_device = _non_empty_string(hardware.get("destination_device"), "hardware.destination_device", errors)
        _non_empty_string(hardware.get("source_name"), "hardware.source_name", errors)
        _non_empty_string(hardware.get("destination_name"), "hardware.destination_name", errors)
        if hardware.get("same_physical_host") is not True:
            errors.append("hardware.same_physical_host must be true for this simulation probe")
        logical_hosts = hardware.get("logical_hosts")
        if not isinstance(logical_hosts, list) or len(logical_hosts) != 2:
            errors.append("hardware.logical_hosts must contain two logical host names")
        if source_device is not None and hardware_source_device != source_device:
            errors.append("hardware.source_device must match config.source_device")
        if destination_device is not None and hardware_destination_device != destination_device:
            errors.append("hardware.destination_device must match config.destination_device")
        if accelerator_measured:
            if tier != "T3-same-host-two-gpu-simulation":
                errors.append("accelerator pipeline probes must use T3-same-host-two-gpu-simulation tier")
            if device_type != "cuda-pipeline":
                errors.append("hardware.device_type must be cuda-pipeline when accelerator_measured is true")
            if not _cuda_device_name(source_device):
                errors.append("config.source_device must be cuda:<index> for accelerator evidence")
            if not _cuda_device_name(destination_device):
                errors.append("config.destination_device must be cuda:<index> for accelerator evidence")
            if source_device == destination_device and source_device is not None:
                errors.append("config.source_device and config.destination_device must differ")
            _positive_int_field(hardware.get("source_total_memory_bytes"), "hardware.source_total_memory_bytes", errors)
            _positive_int_field(hardware.get("destination_total_memory_bytes"), "hardware.destination_total_memory_bytes", errors)
        elif tier == "T3-same-host-two-gpu-simulation":
            errors.append("T3-same-host-two-gpu-simulation probes must set accelerator_measured true")
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
            "destination_device": destination_device,
            "prompt_count": prompt_count,
            "new_tokens": new_tokens,
            "tokens_generated": result.get("tokens_generated") if isinstance(result, dict) else None,
            "tokens_s": result.get("tokens_s") if isinstance(result, dict) else None,
            "activation_bytes_transferred": result.get("activation_bytes_transferred") if isinstance(result, dict) else None,
            "max_abs_error": result.get("max_abs_error") if isinstance(result, dict) else None,
            "generated_sequences": result.get("generated_sequences") if isinstance(result, dict) else None,
        },
    }


def validate_pipeline_correctness_probe(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {"ok": False, "errors": [f"invalid pipeline-correctness probe artifact: {exc}"], "warnings": [], "summary": {}, "fixture": str(fixture_path)}
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["pipeline-correctness probe artifact must be a JSON object"], "warnings": [], "summary": {}, "fixture": str(fixture_path)}
    result = validate_pipeline_correctness_probe_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
