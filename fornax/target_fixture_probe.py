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

PROBE_KIND = "target-fixture-execution-probe"
BACKENDS = {"cpu-stdlib", "torch"}
DTYPES = {"float32", "float16", "bfloat16"}
TARGET_FIXTURE_MODEL_ID = "fornax-local-target-fixture-v1"
TARGET_FIXTURE_TEMPLATE_HASH = "sha256:" + "c" * 64
TARGET_FIXTURE_TOKENIZER_HASH = "sha256:" + "d" * 64
DEFAULT_PROMPT_TOKENS = [1, 2, 3]
DEFAULT_STOP_TOKEN_ID = 9
TOKEN_TEXT = {
    0: "<pad>",
    1: "user",
    2: "asks",
    3: "start",
    4: "fixture",
    5: "target",
    6: "h100",
    7: "parity",
    8: "done",
    9: "</final>",
    10: "ignored",
}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


def _positive_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def _normalize_prompt(prompt_tokens: list[int] | None, vocab_size: int) -> list[int]:
    raw = DEFAULT_PROMPT_TOKENS if prompt_tokens is None else prompt_tokens
    if not isinstance(raw, list) or not raw:
        raise ValueError("prompt_tokens must be a non-empty list")
    normalized: list[int] = []
    for index, token in enumerate(raw):
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"prompt_tokens[{index}] must be an integer")
        if token < 0 or token >= vocab_size:
            raise ValueError(f"prompt_tokens[{index}] must be in [0, vocab_size)")
        normalized.append(token)
    return normalized


def parse_prompt_tokens_json(value: str | None, vocab_size: int) -> list[int]:
    if value is None:
        return _normalize_prompt(None, vocab_size)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompt_tokens_json must be valid JSON: {exc}") from exc
    return _normalize_prompt(parsed, vocab_size)


def _validate_config(
    *,
    iterations: int,
    warmup: int,
    vocab_size: int,
    new_tokens: int,
    stop_token_id: int,
    prompt_tokens: list[int] | None,
    tolerance: float,
) -> list[int]:
    _positive_int("iterations", iterations)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    _positive_int("vocab_size", vocab_size)
    _positive_int("new_tokens", new_tokens)
    if isinstance(stop_token_id, bool) or not isinstance(stop_token_id, int) or stop_token_id < 0 or stop_token_id >= vocab_size:
        raise ValueError("stop_token_id must be in [0, vocab_size)")
    _non_negative_number("tolerance", tolerance)
    return _normalize_prompt(prompt_tokens, vocab_size)


def _logits(current_token: int, step: int, vocab_size: int) -> list[float]:
    forced = (current_token + step + 1) % vocab_size
    values: list[float] = []
    for vocab in range(vocab_size):
        value = (((current_token + 1) * (vocab + 3)) % 29 - 14) / 29.0
        if vocab == forced:
            value += 100.0
        values.append(value)
    return values


def _argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _text_for_tokens(tokens: list[int]) -> str:
    return " ".join(TOKEN_TEXT.get(token, f"tok{token}") for token in tokens)


def _checksum(values: list[float]) -> float:
    return sum(value * (1.0 + index * 0.0001) for index, value in enumerate(values))


def _max_abs_error(left: list[float], right: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def _run_reference(
    prompt_tokens: list[int],
    *,
    vocab_size: int,
    new_tokens: int,
    stop_token_id: int,
) -> tuple[list[int], list[float], str, int | None]:
    current = prompt_tokens[-1]
    generated: list[int] = []
    finish_reason = "length"
    stop_observed: int | None = None
    final_logits = _logits(current, 0, vocab_size)
    for step in range(new_tokens):
        final_logits = _logits(current, step, vocab_size)
        next_token = _argmax(final_logits)
        if next_token == stop_token_id:
            finish_reason = "stop"
            stop_observed = stop_token_id
            break
        generated.append(next_token)
        current = next_token
    return generated, final_logits, finish_reason, stop_observed


def run_cpu_target_fixture_execution_probe(
    *,
    iterations: int = 5,
    warmup: int = 1,
    vocab_size: int = 17,
    new_tokens: int = 4,
    prompt_tokens: list[int] | None = None,
    stop_token_id: int = DEFAULT_STOP_TOKEN_ID,
    tolerance: float = 0.0,
    logical_host: str = "logical-host-0",
) -> dict[str, Any]:
    normalized = _validate_config(
        iterations=iterations,
        warmup=warmup,
        vocab_size=vocab_size,
        new_tokens=new_tokens,
        stop_token_id=stop_token_id,
        prompt_tokens=prompt_tokens,
        tolerance=tolerance,
    )
    if not logical_host:
        raise ValueError("logical_host must be non-empty")
    reference_tokens, reference_logits, finish_reason, stop_observed = _run_reference(
        normalized,
        vocab_size=vocab_size,
        new_tokens=new_tokens,
        stop_token_id=stop_token_id,
    )
    for _ in range(warmup):
        _run_reference(normalized, vocab_size=vocab_size, new_tokens=new_tokens, stop_token_id=stop_token_id)
    started_ns = time.perf_counter_ns()
    generated_tokens: list[int] = []
    final_logits: list[float] = []
    for _ in range(iterations):
        generated_tokens, final_logits, finish_reason, stop_observed = _run_reference(
            normalized,
            vocab_size=vocab_size,
            new_tokens=new_tokens,
            stop_token_id=stop_token_id,
        )
    elapsed_ns = time.perf_counter_ns() - started_ns
    elapsed_s = elapsed_ns / 1_000_000_000.0
    max_abs_error = _max_abs_error(final_logits, reference_logits)
    sequence_match = generated_tokens == reference_tokens
    token_count = len(generated_tokens)
    tokens_generated = iterations * token_count
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T1-target-fixture-reference",
        "measured": True,
        "accelerator_measured": False,
        "backend": "cpu-stdlib",
        "available": True,
        "source": "fornax.target_fixture_probe.cpu_target_fixture_execution",
        "target_fixture": {
            "model_id": TARGET_FIXTURE_MODEL_ID,
            "scope": "local-target-fixture",
            "template_hash": TARGET_FIXTURE_TEMPLATE_HASH,
            "tokenizer_hash": TARGET_FIXTURE_TOKENIZER_HASH,
            "stop_token_id": stop_token_id,
            "stop_sequence": TOKEN_TEXT.get(stop_token_id, str(stop_token_id)),
            "real_frontier_model": False,
        },
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "vocab_size": vocab_size,
            "new_tokens": new_tokens,
            "prompt_tokens": normalized,
            "prompt_token_count": len(normalized),
            "stop_token_id": stop_token_id,
            "device": "cpu",
            "logical_host": logical_host,
            "dtype": "float64-reference",
            "tolerance": tolerance,
        },
        "result": {
            "elapsed_s": elapsed_s,
            "elapsed_ns": elapsed_ns,
            "generated_token_count": token_count,
            "tokens_generated": tokens_generated,
            "tokens_s": tokens_generated / elapsed_s if elapsed_s > 0 else None,
            "generated_token_ids": generated_tokens,
            "reference_generated_token_ids": reference_tokens,
            "generated_text": _text_for_tokens(generated_tokens),
            "reference_generated_text": _text_for_tokens(reference_tokens),
            "finish_reason": finish_reason,
            "stop_token_observed": stop_observed,
            "logit_checksum": _checksum(final_logits),
            "reference_logit_checksum": _checksum(reference_logits),
            "max_abs_error": max_abs_error,
            "sequence_match": sequence_match,
            "correctness_passed": sequence_match and max_abs_error <= tolerance,
            "timing_method": "perf_counter_ns_cpu_target_fixture",
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "hardware": {
            "device_type": "cpu-target-fixture",
            "device": "cpu",
            "name": platform.processor() or platform.machine(),
            "same_physical_host": True,
            "logical_hosts": [logical_host],
        },
        "note": (
            "CPU target-fixture execution reference. Validates deterministic decode, "
            "stop-token handling, and reference parity, but is not accelerator or "
            "real frontier-model evidence."
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
vocab_size = int(sys.argv[3])
new_tokens = int(sys.argv[4])
prompt_tokens = json.loads(sys.argv[5])
stop_token_id = int(sys.argv[6])
device_name = sys.argv[7]
dtype_name = sys.argv[8]
tolerance = float(sys.argv[9])
logical_host = sys.argv[10]
probe_kind = "target-fixture-execution-probe"
tier = "T2-single-node-target-fixture"
model_id = "fornax-local-target-fixture-v1"
template_hash = "sha256:" + "c" * 64
tokenizer_hash = "sha256:" + "d" * 64
token_text = {4: "fixture", 6: "h100", 9: "</final>"}


def emit_unavailable(error):
    print(json.dumps({
        "version": 1,
        "probe_kind": probe_kind,
        "tier": tier,
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.target_fixture_probe.torch_target_fixture_execution",
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

try:
    match = re.fullmatch(r"cuda:(\d+)", device_name)
    if not match:
        raise ValueError(f"device must be cuda:<index>: {device_name}")
    device_index = int(match.group(1))
    device_count = int(torch.cuda.device_count())
    if device_index >= device_count:
        raise ValueError(f"device index out of range for cuda_device_count={device_count}: {device_index}")
    device = torch.device(device_name)
    if not isinstance(prompt_tokens, list) or not prompt_tokens:
        raise ValueError("prompt_tokens must be a non-empty list")
    normalized_prompt = []
    for token in prompt_tokens:
        if not isinstance(token, int) or token < 0 or token >= vocab_size:
            raise ValueError("prompt tokens must be integers in [0, vocab_size)")
        normalized_prompt.append(token)
    if stop_token_id < 0 or stop_token_id >= vocab_size:
        raise ValueError("stop_token_id must be in [0, vocab_size)")
except Exception as exc:
    emit_unavailable(f"configuration failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)


def logits_for(current_token, step):
    vocab = torch.arange(vocab_size, device=device, dtype=torch.float32)
    values = torch.remainder((float(current_token) + 1.0) * (vocab + 3.0), 29.0)
    logits = ((values - 14.0) / 29.0).to(dtype=dtype)
    forced = int((int(current_token) + step + 1) % vocab_size)
    logits[forced] += 100.0
    return logits


def reference_logits_for(current_token, step):
    vals = []
    forced = int((int(current_token) + step + 1) % vocab_size)
    for vocab in range(vocab_size):
        value = (((int(current_token) + 1) * (vocab + 3)) % 29 - 14) / 29.0
        if vocab == forced:
            value += 100.0
        vals.append(value)
    return vals


def run_reference():
    current = int(normalized_prompt[-1])
    generated = []
    finish_reason = "length"
    stop_observed = None
    final_logits = reference_logits_for(current, 0)
    for step in range(new_tokens):
        final_logits = reference_logits_for(current, step)
        next_token = max(range(len(final_logits)), key=lambda idx: final_logits[idx])
        if next_token == stop_token_id:
            finish_reason = "stop"
            stop_observed = stop_token_id
            break
        generated.append(int(next_token))
        current = int(next_token)
    return generated, final_logits, finish_reason, stop_observed


def run_accelerated():
    current = int(normalized_prompt[-1])
    generated = []
    finish_reason = "length"
    stop_observed = None
    final_logits = logits_for(current, 0)
    for step in range(new_tokens):
        final_logits = logits_for(current, step)
        next_token = int(torch.argmax(final_logits.float()).detach().cpu().item())
        if next_token == stop_token_id:
            finish_reason = "stop"
            stop_observed = stop_token_id
            break
        generated.append(next_token)
        current = next_token
    return generated, final_logits.detach().cpu().float(), finish_reason, stop_observed


def text_for(tokens):
    return " ".join(token_text.get(int(token), f"tok{int(token)}") for token in tokens)


def checksum(values):
    total = 0.0
    for index, value in enumerate(values):
        total += float(value) * (1.0 + index * 0.0001)
    return total

try:
    props = torch.cuda.get_device_properties(device)
    with torch.no_grad():
        reference_tokens, reference_logits, reference_finish, reference_stop = run_reference()
        generated_tokens, logits, finish_reason, stop_observed = run_accelerated()
        torch.cuda.synchronize(device)
        accelerated_logits = [float(value) for value in logits.tolist()]
        max_abs_error = max(abs(a - b) for a, b in zip(accelerated_logits, reference_logits))
        sequence_match = bool(generated_tokens == reference_tokens)
        for _ in range(warmup):
            run_accelerated()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        timed_tokens = generated_tokens
        timed_logits = accelerated_logits
        timed_finish = finish_reason
        timed_stop = stop_observed
        for _ in range(iterations):
            timed_tokens, logits, timed_finish, timed_stop = run_accelerated()
            timed_logits = [float(value) for value in logits.tolist()]
        torch.cuda.synchronize(device)
        elapsed_s = time.perf_counter() - started
except Exception as exc:
    emit_unavailable(f"target fixture probe failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

generated_token_count = len(timed_tokens)
tokens_generated = int(iterations * generated_token_count)
print(json.dumps({
    "version": 1,
    "probe_kind": probe_kind,
    "tier": tier,
    "measured": True,
    "accelerator_measured": True,
    "backend": "torch",
    "available": True,
    "source": "fornax.target_fixture_probe.torch_target_fixture_execution",
    "target_fixture": {
        "model_id": model_id,
        "scope": "local-target-fixture",
        "template_hash": template_hash,
        "tokenizer_hash": tokenizer_hash,
        "stop_token_id": stop_token_id,
        "stop_sequence": token_text.get(stop_token_id, str(stop_token_id)),
        "real_frontier_model": False,
    },
    "config": {
        "iterations": iterations,
        "warmup": warmup,
        "vocab_size": vocab_size,
        "new_tokens": new_tokens,
        "prompt_tokens": normalized_prompt,
        "prompt_token_count": len(normalized_prompt),
        "stop_token_id": stop_token_id,
        "device": device_name,
        "logical_host": logical_host,
        "dtype": dtype_name,
        "tolerance": tolerance,
    },
    "result": {
        "elapsed_s": elapsed_s,
        "elapsed_ns": int(elapsed_s * 1000000000),
        "generated_token_count": generated_token_count,
        "tokens_generated": tokens_generated,
        "tokens_s": tokens_generated / elapsed_s if elapsed_s > 0 else None,
        "generated_token_ids": timed_tokens,
        "reference_generated_token_ids": reference_tokens,
        "generated_text": text_for(timed_tokens),
        "reference_generated_text": text_for(reference_tokens),
        "finish_reason": timed_finish,
        "stop_token_observed": timed_stop,
        "logit_checksum": checksum(timed_logits),
        "reference_logit_checksum": checksum(reference_logits),
        "max_abs_error": max_abs_error,
        "sequence_match": sequence_match,
        "correctness_passed": bool(sequence_match and max_abs_error <= tolerance),
        "timing_method": "perf_counter_cuda_synchronize_target_fixture",
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
        "device_type": "cuda-target-fixture",
        "device": device_name,
        "device_index": device_index,
        "name": torch.cuda.get_device_name(device),
        "total_memory_bytes": int(props.total_memory),
        "same_physical_host": True,
        "logical_hosts": [logical_host],
    },
    "note": (
        "Measured single-H100 local target-fixture execution probe. Validates deterministic "
        "fixture decode, stop-token handling, and reference parity on CUDA; not proof of "
        "real frontier target-model loading or G3 heterogeneous closure."
    ),
}))
'''


def run_torch_target_fixture_execution_probe(
    *,
    torch_python: str | None = None,
    device: str = "cuda:0",
    dtype: str = "float32",
    iterations: int = 20,
    warmup: int = 3,
    vocab_size: int = 17,
    new_tokens: int = 4,
    prompt_tokens: list[int] | None = None,
    stop_token_id: int = DEFAULT_STOP_TOKEN_ID,
    tolerance: float = 1e-4,
    logical_host: str = "logical-host-0",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    normalized = _validate_config(
        iterations=iterations,
        warmup=warmup,
        vocab_size=vocab_size,
        new_tokens=new_tokens,
        stop_token_id=stop_token_id,
        prompt_tokens=prompt_tokens,
        tolerance=tolerance,
    )
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if not logical_host:
        raise ValueError("logical_host must be non-empty")
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
                str(new_tokens),
                json.dumps(normalized),
                str(stop_token_id),
                device,
                dtype,
                str(tolerance),
                logical_host,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable_artifact(f"torch target-fixture probe failed to launch: {type(exc).__name__}: {exc}", python)
    stdout = result.stdout.strip()
    if result.returncode != 0:
        data = _unavailable_artifact("torch target-fixture probe exited nonzero", python)
        data.update({"returncode": result.returncode, "stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    try:
        data = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        data = _unavailable_artifact(f"torch target-fixture probe did not emit JSON: {exc}", python)
        data.update({"stdout": stdout[-2000:], "stderr": result.stderr.strip()[-2000:]})
        return data
    if not isinstance(data, dict):
        return _unavailable_artifact("torch target-fixture probe JSON was not an object", python)
    data.setdefault("source", "fornax.target_fixture_probe.torch_target_fixture_execution")
    data.setdefault("backend", "torch")
    data.setdefault("environment", {})
    if isinstance(data["environment"], dict):
        data["environment"].setdefault("python_executable", python)
    return data


def _unavailable_artifact(error: str, python: str) -> dict[str, Any]:
    return {
        "version": 1,
        "probe_kind": PROBE_KIND,
        "tier": "T2-single-node-target-fixture",
        "measured": False,
        "accelerator_measured": False,
        "backend": "torch",
        "available": False,
        "source": "fornax.target_fixture_probe.torch_target_fixture_execution",
        "error": error,
        "environment": {"python_executable": python},
    }


def run_target_fixture_execution_probe(
    *,
    backend: str = "cpu-stdlib",
    torch_python: str | None = None,
    device: str = "cuda:0",
    dtype: str = "float32",
    iterations: int = 5,
    warmup: int = 1,
    vocab_size: int = 17,
    new_tokens: int = 4,
    prompt_tokens: list[int] | None = None,
    stop_token_id: int = DEFAULT_STOP_TOKEN_ID,
    tolerance: float = 0.0,
    logical_host: str = "logical-host-0",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    if backend == "cpu-stdlib":
        return run_cpu_target_fixture_execution_probe(
            iterations=iterations,
            warmup=warmup,
            vocab_size=vocab_size,
            new_tokens=new_tokens,
            prompt_tokens=prompt_tokens,
            stop_token_id=stop_token_id,
            tolerance=tolerance,
            logical_host=logical_host,
        )
    return run_torch_target_fixture_execution_probe(
        torch_python=torch_python,
        device=device,
        dtype=dtype,
        iterations=iterations,
        warmup=warmup,
        vocab_size=vocab_size,
        new_tokens=new_tokens,
        prompt_tokens=prompt_tokens,
        stop_token_id=stop_token_id,
        tolerance=tolerance,
        logical_host=logical_host,
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


def _number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field} must be numeric")
        return None
    return float(value)


def _positive_number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


def _cuda_device_name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"cuda:\d+", value) is not None


def validate_target_fixture_execution_probe_fixture(data: dict[str, Any]) -> dict[str, Any]:
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

    target_fixture = data.get("target_fixture")
    if measured:
        if not isinstance(target_fixture, dict):
            errors.append("target_fixture must be an object for measured probes")
            target_fixture = {}
        if target_fixture.get("model_id") != TARGET_FIXTURE_MODEL_ID:
            errors.append(f"target_fixture.model_id must be {TARGET_FIXTURE_MODEL_ID}")
        if target_fixture.get("scope") != "local-target-fixture":
            errors.append("target_fixture.scope must be local-target-fixture")
        if not _is_sha256(target_fixture.get("template_hash")):
            errors.append("target_fixture.template_hash must be a sha256 hash")
        if not _is_sha256(target_fixture.get("tokenizer_hash")):
            errors.append("target_fixture.tokenizer_hash must be a sha256 hash")
        if target_fixture.get("real_frontier_model") is not False:
            errors.append("target_fixture.real_frontier_model must be false")

    config = data.get("config")
    if not isinstance(config, dict):
        if measured:
            errors.append("config must be an object for measured probes")
        config = {}
    iterations = _positive_int_field(config.get("iterations"), "config.iterations", errors) if measured else None
    _positive_int_field(config.get("vocab_size"), "config.vocab_size", errors) if measured else None
    _positive_int_field(config.get("new_tokens"), "config.new_tokens", errors) if measured else None
    prompt_tokens = config.get("prompt_tokens")
    if measured:
        if not isinstance(prompt_tokens, list) or not prompt_tokens:
            errors.append("config.prompt_tokens must be a non-empty list")
        elif not all(isinstance(token, int) and not isinstance(token, bool) for token in prompt_tokens):
            errors.append("config.prompt_tokens must contain only integers")
        _positive_int_field(config.get("prompt_token_count"), "config.prompt_token_count", errors)
        _positive_int_field(config.get("stop_token_id"), "config.stop_token_id", errors)
        _non_empty_string(config.get("device"), "config.device", errors)
        _non_empty_string(config.get("logical_host"), "config.logical_host", errors)
        dtype = _non_empty_string(config.get("dtype"), "config.dtype", errors)
        if backend == "torch" and dtype is not None and dtype not in DTYPES:
            errors.append(f"config.dtype must be one of {sorted(DTYPES)}")
        _non_negative_number_field(config.get("tolerance"), "config.tolerance", errors)

    result = data.get("result")
    if not isinstance(result, dict):
        if measured:
            errors.append("result must be an object for measured probes")
        result = {}
    if measured:
        generated_count = _positive_int_field(result.get("generated_token_count"), "result.generated_token_count", errors)
        tokens_generated = _positive_int_field(result.get("tokens_generated"), "result.tokens_generated", errors)
        _positive_number_field(result.get("elapsed_s"), "result.elapsed_s", errors)
        _positive_number_field(result.get("tokens_s"), "result.tokens_s", errors)
        _number_field(result.get("logit_checksum"), "result.logit_checksum", errors)
        _number_field(result.get("reference_logit_checksum"), "result.reference_logit_checksum", errors)
        max_abs_error = _non_negative_number_field(result.get("max_abs_error"), "result.max_abs_error", errors)
        _non_empty_string(result.get("timing_method"), "result.timing_method", errors)
        if result.get("sequence_match") is not True:
            errors.append("result.sequence_match must be true")
        if result.get("correctness_passed") is not True:
            errors.append("result.correctness_passed must be true")
        tolerance_value = config.get("tolerance") if isinstance(config, dict) else None
        if isinstance(tolerance_value, (int, float)) and max_abs_error is not None and max_abs_error > tolerance_value:
            errors.append("result.max_abs_error exceeds config.tolerance")
        if result.get("generated_token_ids") != result.get("reference_generated_token_ids"):
            errors.append("result.generated_token_ids must match reference_generated_token_ids")
        if result.get("generated_text") != result.get("reference_generated_text"):
            errors.append("result.generated_text must match reference_generated_text")
        if result.get("finish_reason") != "stop":
            errors.append("result.finish_reason must be stop")
        if result.get("stop_token_observed") != config.get("stop_token_id"):
            errors.append("result.stop_token_observed must match config.stop_token_id")
        if iterations is not None and generated_count is not None and tokens_generated is not None:
            if tokens_generated != iterations * generated_count:
                errors.append("result.tokens_generated must equal iterations * generated_token_count")
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
        hardware_device = _non_empty_string(hardware.get("device"), "hardware.device", errors)
        _non_empty_string(hardware.get("name"), "hardware.name", errors)
        if hardware.get("same_physical_host") is not True:
            errors.append("hardware.same_physical_host must be true")
        logical_hosts = hardware.get("logical_hosts")
        if not isinstance(logical_hosts, list) or len(logical_hosts) != 1:
            errors.append("hardware.logical_hosts must contain one logical host")
        if isinstance(config, dict) and hardware_device != config.get("device"):
            errors.append("hardware.device must match config.device")
        if accelerator_measured:
            if backend != "torch":
                errors.append("accelerator target-fixture probes must use torch backend")
            if tier != "T2-single-node-target-fixture":
                errors.append("accelerator target-fixture probes must use T2-single-node-target-fixture tier")
            if device_type != "cuda-target-fixture":
                errors.append("hardware.device_type must be cuda-target-fixture when accelerator_measured is true")
            if not _cuda_device_name(config.get("device") if isinstance(config, dict) else None):
                errors.append("config.device must be cuda:<index> for accelerator evidence")
            _positive_int_field(hardware.get("total_memory_bytes"), "hardware.total_memory_bytes", errors)
        elif backend == "cpu-stdlib":
            warnings.append("probe is measured but not accelerator evidence")
        elif tier == "T2-single-node-target-fixture":
            errors.append("T2-single-node-target-fixture probes must set accelerator_measured true")
    if backend == "cpu-stdlib" and accelerator_measured:
        errors.append("cpu-stdlib backend cannot be accelerator_measured")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "tier": tier,
            "backend": backend,
            "measured": bool(measured),
            "accelerator_measured": bool(accelerator_measured),
            "device": config.get("device") if isinstance(config, dict) else None,
            "logical_host": config.get("logical_host") if isinstance(config, dict) else None,
            "generated_token_count": result.get("generated_token_count") if isinstance(result, dict) else None,
            "tokens_generated": result.get("tokens_generated") if isinstance(result, dict) else None,
            "tokens_s": result.get("tokens_s") if isinstance(result, dict) else None,
            "generated_text": result.get("generated_text") if isinstance(result, dict) else None,
            "finish_reason": result.get("finish_reason") if isinstance(result, dict) else None,
            "max_abs_error": result.get("max_abs_error") if isinstance(result, dict) else None,
            "target_fixture_model_id": target_fixture.get("model_id") if isinstance(target_fixture, dict) else None,
            "real_frontier_model": target_fixture.get("real_frontier_model") if isinstance(target_fixture, dict) else None,
        },
    }


def validate_target_fixture_execution_probe(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid target fixture execution probe artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["target fixture execution probe artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_target_fixture_execution_probe_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
