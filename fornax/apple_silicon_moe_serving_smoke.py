from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .io import read_json, write_json


RECORD_KIND = "apple-silicon-moe-serving-smoke"
EVIDENCE_SCOPE = "single-mac-max-real-moe-serving-smoke"
DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B"
DEFAULT_PROMPT = "In one short sentence, say what MoE means in AI inference."
ALLOWED_FAMILIES = {"Qwen", "DeepSeek", "Kimi", "GLM"}
RUNTIME_MODES = {"generate", "serve"}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _command_parts(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("command must be non-empty")
    return parts


def _run_text(
    command: list[str], *, timeout_s: float, cwd: str | None = None
) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return None, text or f"exited {result.returncode}"
    return text, None


def _sibling_command(command: list[str], sibling_name: str) -> list[str]:
    if not command:
        return [sibling_name]
    if Path(command[-1]).name == "max":
        candidate = Path(command[-1]).with_name(sibling_name)
        if command[-1] == "max":
            return command[:-1] + [sibling_name]
        return command[:-1] + [str(candidate)]
    return [sibling_name]


def _family_from_model_id(model_id: str) -> str | None:
    lower = model_id.lower()
    if lower.startswith("qwen/") or "qwen" in lower:
        return "Qwen"
    if lower.startswith("deepseek-ai/") or "deepseek" in lower:
        return "DeepSeek"
    if "kimi" in lower or lower.startswith("moonshotai/"):
        return "Kimi"
    if "glm" in lower or lower.startswith("zai-org/"):
        return "GLM"
    return None


def _hub_model_dir(model_id: str, hf_home: str | None) -> Path:
    root = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return root / "hub" / ("models--" + model_id.replace("/", "--"))


def _config_from_cache(
    *, model_id: str, model_path: str | None, hf_home: str | None
) -> tuple[dict[str, Any], str | None]:
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path) / "config.json")

    hub_dir = _hub_model_dir(model_id, hf_home)
    ref = hub_dir / "refs" / "main"
    if ref.exists():
        try:
            sha = ref.read_text(encoding="utf-8").strip()
            if sha:
                candidates.append(hub_dir / "snapshots" / sha / "config.json")
        except OSError:
            pass
    snapshots = hub_dir / "snapshots"
    if snapshots.exists():
        candidates.extend(sorted(snapshots.glob("*/config.json")))

    for path in candidates:
        try:
            data = read_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            return data, str(path)
    return {}, None


def _model_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        architectures = []
    num_experts = (
        text.get("num_experts")
        or text.get("n_routed_experts")
        or text.get("num_local_experts")
        or text.get("n_experts")
    )
    shared_experts = text.get("n_shared_experts") or text.get("num_shared_experts")
    return {
        "architectures": [str(item) for item in architectures],
        "architecture": str(architectures[0]) if architectures else None,
        "model_type": config.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "num_experts": num_experts,
        "n_routed_experts": text.get("n_routed_experts"),
        "n_shared_experts": shared_experts,
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "moe_intermediate_size": text.get("moe_intermediate_size"),
        "max_position_embeddings": text.get("max_position_embeddings"),
    }


def _hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        summary["collection_error"] = f"{type(exc).__name__}: {exc}"
        return summary
    if result.returncode != 0:
        summary["collection_error"] = result.stderr.strip() or f"system_profiler exited {result.returncode}"
        return summary
    fields = {
        "model_name": "Model Name",
        "model_identifier": "Model Identifier",
        "chip": "Chip",
        "core_count": "Total Number of Cores",
        "memory": "Memory",
    }
    for key, label in fields.items():
        match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", result.stdout, re.MULTILINE)
        if match:
            summary[key] = match.group(1).strip()
    return summary


def _generated_text_from_stdout(stdout: str) -> str:
    ignored = (
        "INFO:",
        "WARNING:",
        "Warning:",
        "UserWarning:",
        "Downloading",
        "Loading",
    )
    lines: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d\d:\d\d:\d\d\.\d+\s+\w+:", line):
            continue
        if line.startswith("/private/") or line.startswith("/Users/"):
            continue
        if any(line.startswith(prefix) for prefix in ignored):
            continue
        if "%|" in line:
            continue
        lines.append(line)
    return "\n".join(lines[-8:]).strip()


def _generated_text_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    text = first.get("text")
    return text.strip() if isinstance(text, str) else ""


def _tail(text: str, limit: int = 12000) -> str:
    return text.strip()[-limit:]


def _failure_signature(stdout: str, stderr: str) -> list[str]:
    patterns = (
        "Metal Compiler failed",
        "graph compiler",
        "KGEN",
        "failed to compile",
        "failed to import kernels",
        "failed to resolve built-in kernel package paths",
        "Traceback",
        "RuntimeError:",
        "ValueError:",
    )
    signature: list[str] = []
    for raw in (stdout + "\n" + stderr).splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(pattern in line for pattern in patterns):
            signature.append(line)
    return signature[:12]


def _post_chat_completion(
    *, port: int, model: str, prompt: str, max_new_tokens: int, timeout_s: float
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "stream": False,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return exc.code, None, detail
    except (OSError, urllib.error.URLError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return status, None, f"invalid JSON response: {exc}: {payload[:500]}"
    if not isinstance(data, dict):
        return status, None, "OpenAI-compatible response must be a JSON object"
    return status, data, None


def _run_serve_request(
    *,
    command: list[str],
    env: dict[str, str],
    cwd: str | None,
    port: int,
    served_model_name: str,
    prompt: str,
    max_new_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    http_status: int | None = None
    response: dict[str, Any] | None = None
    http_error: str | None = None
    generated_text = ""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
        )
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                break
            http_status, response, http_error = _post_chat_completion(
                port=port,
                model=served_model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                timeout_s=2.0,
            )
            if response is not None:
                generated_text = _generated_text_from_response(response)
                if generated_text:
                    break
            time.sleep(1.0)
    except (OSError, subprocess.SubprocessError) as exc:
        elapsed_s = time.perf_counter() - started
        return {
            "elapsed_s": elapsed_s,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "generated_text": "",
            "http_status": http_status,
            "http_error": f"MAX serve failed to launch: {type(exc).__name__}: {exc}",
            "response": response,
            "server_terminated_by_probe": False,
        }

    assert process is not None
    server_terminated_by_probe = False
    if generated_text and process.poll() is None:
        process.terminate()
        server_terminated_by_probe = True
    elif process.poll() is None and time.perf_counter() >= deadline:
        http_error = http_error or "MAX serve did not become ready before timeout"
        process.terminate()
        server_terminated_by_probe = True
    try:
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=15)
        server_terminated_by_probe = True
    elapsed_s = time.perf_counter() - started
    return {
        "elapsed_s": elapsed_s,
        "returncode": 0 if generated_text else process.returncode,
        "server_returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "generated_text": generated_text,
        "http_status": http_status,
        "http_error": http_error,
        "response": response,
        "server_terminated_by_probe": server_terminated_by_probe,
    }


def run_apple_silicon_moe_serving_smoke(
    *,
    out: str | Path,
    max_command: str = "max",
    max_cwd: str | None = None,
    max_extra_args: list[str] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    model_path: str | None = None,
    hf_home: str | None = None,
    devices: str = "gpu",
    quantization_encoding: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 16,
    top_k: int = 1,
    max_length: int | None = None,
    runtime_mode: str = "generate",
    serve_port: int = 18080,
    allow_download: bool = True,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    _positive_int("max_new_tokens", max_new_tokens)
    _positive_int("top_k", top_k)
    if max_length is not None:
        _positive_int("max_length", max_length)
    if runtime_mode not in RUNTIME_MODES:
        raise ValueError(f"runtime_mode must be one of {sorted(RUNTIME_MODES)}")
    _positive_int("serve_port", serve_port)
    _positive_number("timeout_s", timeout_s)
    if not prompt:
        raise ValueError("prompt must be non-empty")
    family = _family_from_model_id(model_id)
    if family not in ALLOWED_FAMILIES:
        raise ValueError(
            "model_id must be from one of Qwen, DeepSeek, Kimi, or GLM for this smoke"
        )
    max_cwd_path = Path(max_cwd).expanduser() if max_cwd else None
    if max_cwd_path is not None and not max_cwd_path.is_dir():
        raise ValueError("max_cwd must be an existing directory")
    max_cwd_text = str(max_cwd_path) if max_cwd_path is not None else None
    extra_args = [str(item) for item in (max_extra_args or [])]
    if any(not item for item in extra_args):
        raise ValueError("max_extra_args must not contain empty arguments")

    base_command = _command_parts(max_command)
    max_version, max_version_error = _run_text(
        base_command + ["--version"], timeout_s=60, cwd=max_cwd_text
    )
    mojo_version, mojo_version_error = _run_text(
        _sibling_command(base_command, "mojo") + ["--version"],
        timeout_s=60,
        cwd=max_cwd_text,
    )
    config_before, config_source_before = _config_from_cache(
        model_id=model_id, model_path=model_path, hf_home=hf_home
    )

    model_ref = model_path or model_id
    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home
    if not allow_download:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    if runtime_mode == "generate":
        command = base_command + [
            "generate",
            "--model",
            model_ref,
            "--devices",
            devices,
            "--max-new-tokens",
            str(max_new_tokens),
            "--top-k",
            str(top_k),
            *extra_args,
            "--prompt",
            prompt,
        ]
        if quantization_encoding:
            command.extend(["--quantization-encoding", quantization_encoding])
        if max_length is not None:
            command.extend(["--max-length", str(max_length)])

        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
                cwd=max_cwd_text,
            )
            elapsed_s = time.perf_counter() - started
        except (OSError, subprocess.SubprocessError) as exc:
            elapsed_s = time.perf_counter() - started
            result = None
            error = f"MAX generation failed to launch: {type(exc).__name__}: {exc}"
        else:
            error = None
        stdout = result.stdout if result is not None else ""
        stderr = result.stderr if result is not None else ""
        generated_text = _generated_text_from_stdout(stdout)
        response_payload: dict[str, Any] | None = None
        http_status = None
        http_error = None
        server_returncode = None
        server_terminated_by_probe = False
        returncode = result.returncode if result is not None else None
        ok = bool(result is not None and result.returncode == 0 and generated_text)
        if result is not None and result.returncode != 0:
            error = "MAX generation exited nonzero"
        if result is not None and result.returncode == 0 and not generated_text:
            error = "MAX generation emitted no parseable generated text"
    else:
        served_model_name = model_id
        command = base_command + [
            "serve",
            "--model",
            model_ref,
            "--devices",
            devices,
            "--port",
            str(serve_port),
            "--served-model-name",
            served_model_name,
            *extra_args,
        ]
        if quantization_encoding:
            command.extend(["--quantization-encoding", quantization_encoding])
        if max_length is not None:
            command.extend(["--max-length", str(max_length)])
        serve_result = _run_serve_request(
            command=command,
            env=env,
            cwd=max_cwd_text,
            port=serve_port,
            served_model_name=served_model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            timeout_s=timeout_s,
        )
        elapsed_s = float(serve_result["elapsed_s"])
        stdout = str(serve_result.get("stdout") or "")
        stderr = str(serve_result.get("stderr") or "")
        generated_text = str(serve_result.get("generated_text") or "")
        response_payload = (
            serve_result.get("response")
            if isinstance(serve_result.get("response"), dict)
            else None
        )
        http_status = serve_result.get("http_status")
        http_error = serve_result.get("http_error")
        server_returncode = serve_result.get("server_returncode")
        server_terminated_by_probe = bool(serve_result.get("server_terminated_by_probe"))
        returncode = serve_result.get("returncode")
        ok = bool(response_payload is not None and http_status == 200 and generated_text)
        if ok:
            error = None
        elif returncode not in (None, 0):
            error = "MAX serve exited before HTTP completion"
        elif http_error:
            error = f"MAX serve HTTP probe failed: {http_error}"
        else:
            error = "MAX serve emitted no OpenAI-compatible generated text"

    config_after, config_source_after = _config_from_cache(
        model_id=model_id, model_path=model_path, hf_home=hf_home
    )
    config = config_after or config_before
    config_source = config_source_after or config_source_before
    summary = _model_config_summary(config)

    runtime_mode_label = "max-generate" if runtime_mode == "generate" else "max-serve"
    live_http_endpoint = bool(runtime_mode == "serve" and ok)
    response = response_payload or {
        "id": "fornax-apple-silicon-moe-smoke",
        "object": "chat.completion",
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "length",
            }
        ],
    }

    data: dict[str, Any] = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "ok": ok,
        "error": error,
        "model": {
            "model_id": model_id,
            "model_path": model_path,
            "model_family": family,
            "config_source": config_source,
            "architecture": summary.get("architecture"),
            "architectures": summary.get("architectures", []),
            "model_type": summary.get("model_type"),
            "real_frontier_moe_model": True,
            "synthetic_fixture": False,
            "moe_config": {
                key: summary.get(key)
                for key in [
                    "num_hidden_layers",
                    "hidden_size",
                    "num_experts",
                    "n_routed_experts",
                    "n_shared_experts",
                    "num_experts_per_tok",
                    "moe_intermediate_size",
                    "max_position_embeddings",
                ]
            },
        },
        "runtime": {
            "backend": "max",
            "mode": runtime_mode_label,
            "max_command": base_command,
            "max_cwd": max_cwd_text,
            "max_extra_args": extra_args,
            "max_version": max_version,
            "max_version_error": max_version_error,
            "mojo_version": mojo_version,
            "mojo_version_error": mojo_version_error,
            "devices_requested": devices,
            "quantization_encoding": quantization_encoding,
            "max_length": max_length,
            "serve_port": serve_port if runtime_mode == "serve" else None,
            "allow_download": allow_download,
            "fornax_orchestrated": True,
        },
        "serving": {
            "request": {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_new_tokens": max_new_tokens,
                "stream": False,
            },
            "response": response,
            "generated_text": generated_text,
            "openai_compatible_shape": True,
            "live_http_endpoint": live_http_endpoint,
            "http_status": http_status,
            "http_error": http_error,
        },
        "result": {
            "elapsed_s": elapsed_s,
            "returncode": returncode,
            "server_returncode": server_returncode,
            "server_terminated_by_probe": server_terminated_by_probe,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "failure_signature": _failure_signature(stdout, stderr),
        },
        "hardware": _hardware_summary(),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hf_home": hf_home or os.environ.get("HF_HOME"),
        },
        "claims": {
            "real_frontier_moe_model": True,
            "synthetic_fixture": False,
            "live_http_endpoint": live_http_endpoint,
            "target_model_parity_reference": False,
            "formal_g2_passed": False,
            "formal_g3_passed": False,
            "g2_g3_gate_evidence": False,
            "production_distributed_serving": False,
        },
        "note": (
            "Apple Silicon MAX smoke for a real Qwen/DeepSeek/Kimi/GLM MoE. "
            "This is single-Mac local evidence through MAX, not distributed "
            "Fornax runtime closure, target-model parity, or formal G2/G3 evidence."
        ),
    }
    write_json(out, data)
    return data


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _positive_number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


def validate_apple_silicon_moe_serving_smoke_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "Apple Silicon MoE serving smoke is single-Mac MAX evidence, not production distributed serving or formal G2/G3 closure",
        "use this as bounded MAX/model bring-up evidence, not target-model parity proof",
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must be {EVIDENCE_SCOPE}")
    if data.get("ok") is not True:
        errors.append("ok must be true")

    model = data.get("model")
    if not isinstance(model, dict):
        errors.append("model must be an object")
        model = {}
    model_id = _non_empty_string(model.get("model_id"), "model.model_id", errors)
    family = _non_empty_string(model.get("model_family"), "model.model_family", errors)
    if family is not None and family not in ALLOWED_FAMILIES:
        errors.append(f"model.model_family must be one of {sorted(ALLOWED_FAMILIES)}")
    if model_id is not None and _family_from_model_id(model_id) not in ALLOWED_FAMILIES:
        errors.append("model.model_id must be from Qwen, DeepSeek, Kimi, or GLM")
    architecture = _non_empty_string(model.get("architecture"), "model.architecture", errors)
    model_type = model.get("model_type")
    if model.get("real_frontier_moe_model") is not True:
        errors.append("model.real_frontier_moe_model must be true")
    if model.get("synthetic_fixture") is not False:
        errors.append("model.synthetic_fixture must be false")
    moe_config = model.get("moe_config")
    if not isinstance(moe_config, dict):
        errors.append("model.moe_config must be an object")
        moe_config = {}
    experts = moe_config.get("num_experts")
    active = moe_config.get("num_experts_per_tok")
    has_expert_metadata = not isinstance(experts, bool) and isinstance(experts, int) and experts > 1
    identifies_moe = (
        architecture is not None
        and ("moe" in architecture.lower() or "moe" in str(model_type).lower())
    ) or has_expert_metadata
    if not identifies_moe:
        errors.append("model architecture, model_type, or expert metadata must identify a MoE")
    if isinstance(experts, bool) or not isinstance(experts, int) or experts <= 1:
        errors.append("model.moe_config.num_experts must be an integer greater than 1")
    if isinstance(active, bool) or not isinstance(active, int) or active <= 0:
        errors.append("model.moe_config.num_experts_per_tok must be a positive integer")

    runtime = data.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime must be an object")
        runtime = {}
    if runtime.get("backend") != "max":
        errors.append("runtime.backend must be max")
    runtime_mode = runtime.get("mode")
    if runtime_mode not in {"max-generate", "max-serve"}:
        errors.append("runtime.mode must be max-generate or max-serve")
    max_cwd = runtime.get("max_cwd")
    if max_cwd is not None and not isinstance(max_cwd, str):
        errors.append("runtime.max_cwd must be null or a string")
    serve_port = runtime.get("serve_port")
    if runtime_mode == "max-serve":
        if isinstance(serve_port, bool) or not isinstance(serve_port, int) or serve_port <= 0:
            errors.append("runtime.serve_port must be a positive integer for max-serve mode")
    elif serve_port is not None:
        errors.append("runtime.serve_port must be null for max-generate mode")
    max_extra_args = runtime.get("max_extra_args")
    if not isinstance(max_extra_args, list) or not all(
        isinstance(item, str) and item for item in max_extra_args
    ):
        errors.append("runtime.max_extra_args must be a list of non-empty strings")
    max_version = _non_empty_string(runtime.get("max_version"), "runtime.max_version", errors)
    if max_version is not None and not max_version.startswith("MAX "):
        errors.append("runtime.max_version must start with 'MAX '")
    devices = _non_empty_string(runtime.get("devices_requested"), "runtime.devices_requested", errors)
    if devices is not None and not devices.startswith("gpu"):
        errors.append("runtime.devices_requested must target Apple GPU with gpu/gpu:<id>/gpu:all")
    if runtime.get("fornax_orchestrated") is not True:
        errors.append("runtime.fornax_orchestrated must be true")

    serving = data.get("serving")
    if not isinstance(serving, dict):
        errors.append("serving must be an object")
        serving = {}
    if serving.get("openai_compatible_shape") is not True:
        errors.append("serving.openai_compatible_shape must be true")
    live_http = serving.get("live_http_endpoint")
    expected_live_http = runtime_mode == "max-serve"
    if live_http is not expected_live_http:
        errors.append(
            "serving.live_http_endpoint must be true for max-serve mode and false for max-generate mode"
        )
    http_status = serving.get("http_status")
    if runtime_mode == "max-serve" and http_status != 200:
        errors.append("serving.http_status must be 200 for max-serve mode")
    generated = _non_empty_string(serving.get("generated_text"), "serving.generated_text", errors)
    response = serving.get("response")
    if not isinstance(response, dict):
        errors.append("serving.response must be an object")
    elif response.get("object") != "chat.completion":
        errors.append("serving.response.object must be chat.completion")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    _positive_number_field(result.get("elapsed_s"), "result.elapsed_s", errors)
    if result.get("returncode") != 0:
        errors.append("result.returncode must be 0")

    hardware = data.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("hardware must be an object")
        hardware = {}
    chip = _non_empty_string(hardware.get("chip"), "hardware.chip", errors)
    if chip is not None and "apple" not in chip.lower():
        errors.append("hardware.chip must identify Apple Silicon")
    memory = _non_empty_string(hardware.get("memory"), "hardware.memory", errors)
    if memory is not None and "gb" not in memory.lower():
        errors.append("hardware.memory must record GB unified memory")

    claims = data.get("claims")
    if not isinstance(claims, dict):
        errors.append("claims must be an object")
        claims = {}
    if claims.get("real_frontier_moe_model") is not True:
        errors.append("claims.real_frontier_moe_model must be true")
    if claims.get("synthetic_fixture") is not False:
        errors.append("claims.synthetic_fixture must be false")
    if claims.get("live_http_endpoint") is not expected_live_http:
        errors.append(
            "claims.live_http_endpoint must match runtime mode: true for max-serve and false for max-generate"
        )
    for field in [
        "target_model_parity_reference",
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
            "model_id": model_id,
            "model_family": family,
            "architecture": architecture,
            "num_experts": experts,
            "num_experts_per_tok": active,
            "max_version": runtime.get("max_version"),
            "max_cwd": max_cwd,
            "max_extra_args": max_extra_args if isinstance(max_extra_args, list) else None,
            "runtime_mode": runtime_mode,
            "serve_port": serve_port,
            "devices_requested": devices,
            "chip": chip,
            "memory": memory,
            "generated_text": generated,
            "elapsed_s": result.get("elapsed_s") if isinstance(result, dict) else None,
            "live_http_endpoint": live_http,
            "http_status": http_status,
            "g2_g3_gate_evidence": False,
        },
    }


def validate_apple_silicon_moe_serving_smoke(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid Apple Silicon MoE serving smoke artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["Apple Silicon MoE serving smoke artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_apple_silicon_moe_serving_smoke_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
