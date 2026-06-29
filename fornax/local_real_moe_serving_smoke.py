from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .io import read_json, write_json


RECORD_KIND = "local-real-moe-serving-smoke"
EVIDENCE_SCOPE = "same-host-4gpu-real-moe-serving-proxy"
DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_MODEL_PATH = "/mnt/dataprocessing/cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
DEFAULT_CACHE_DIR = "/mnt/dataprocessing/cache/huggingface"
ALLOWED_MODEL_PREFIXES = ("Qwen/",)
DTYPES = {"bfloat16", "float16", "float32"}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


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


def _validate_args(
    *,
    dtype: str,
    max_new_tokens: int,
    gateway_max_memory_gib: int,
    expert_max_memory_gib: int,
    cpu_max_memory_gib: int,
    timeout_s: float,
) -> None:
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    _positive_int("max_new_tokens", max_new_tokens)
    _positive_int("gateway_max_memory_gib", gateway_max_memory_gib)
    _positive_int("expert_max_memory_gib", expert_max_memory_gib)
    _positive_int("cpu_max_memory_gib", cpu_max_memory_gib)
    _positive_number("timeout_s", timeout_s)


def _external_script() -> str:
    return r'''
import json
import os
import sys
import time
from pathlib import Path

model_id = sys.argv[1]
model_path_arg = sys.argv[2]
cache_dir = sys.argv[3]
devices = [item for item in sys.argv[4].split(",") if item]
dtype_name = sys.argv[5]
system_prompt = sys.argv[6]
user_prompt = sys.argv[7]
max_new_tokens = int(sys.argv[8])
gateway_max_memory_gib = int(sys.argv[9])
expert_max_memory_gib = int(sys.argv[10])
cpu_max_memory_gib = int(sys.argv[11])
local_files_only = sys.argv[12] == "1"
allow_download = sys.argv[13] == "1"
record_kind = "local-real-moe-serving-smoke"
evidence_scope = "same-host-4gpu-real-moe-serving-proxy"


def emit(error, *, ok=False, extra=None):
    payload = {
        "version": 1,
        "record_kind": record_kind,
        "evidence_scope": evidence_scope,
        "ok": ok,
        "error": error,
        "model": {"model_id": model_id, "model_path": model_path_arg or None},
        "environment": {"python_executable": sys.executable},
    }
    if extra:
        payload.update(extra)
    print(json.dumps(payload))

try:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer, Qwen3OmniMoeForConditionalGeneration
except Exception as exc:
    emit(f"import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

if dtype_name == "bfloat16":
    dtype = torch.bfloat16
elif dtype_name == "float16":
    dtype = torch.float16
elif dtype_name == "float32":
    dtype = torch.float32
else:
    emit(f"unsupported dtype: {dtype_name}")
    raise SystemExit(0)

if not torch.cuda.is_available():
    emit("torch.cuda.is_available() is false")
    raise SystemExit(0)
if int(torch.cuda.device_count()) < 4:
    emit(f"expected at least four visible CUDA devices, got {torch.cuda.device_count()}")
    raise SystemExit(0)

try:
    if len(devices) != 4:
        raise ValueError("exactly four devices required")
    for item in devices:
        if not item.startswith("cuda:"):
            raise ValueError(f"device must be cuda:<index>: {item}")
    model_path = model_path_arg
    if not model_path:
        model_path = snapshot_download(
            model_id,
            cache_dir=cache_dir or None,
            local_files_only=local_files_only and not allow_download,
        )
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(model_path)
except Exception as exc:
    emit(f"model path resolution failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)


def memory_snapshot():
    rows = []
    for index in range(min(4, int(torch.cuda.device_count()))):
        with torch.cuda.device(index):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except TypeError:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:
                props = torch.cuda.get_device_properties(index)
                free_bytes, total_bytes = None, int(props.total_memory)
            props = torch.cuda.get_device_properties(index)
            rows.append({
                "device": f"cuda:{index}",
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(props.total_memory),
                "free_bytes": int(free_bytes) if free_bytes is not None else None,
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            })
    return rows


def moe_text_config_summary(config):
    out = {}
    for section_name in ("thinker_config", "talker_config"):
        section = getattr(config, section_name, None)
        text_config = getattr(section, "text_config", None) if section is not None else None
        if text_config is None:
            continue
        out[section_name.replace("_config", "")] = {
            "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
            "hidden_size": getattr(text_config, "hidden_size", None),
            "num_experts": getattr(text_config, "num_experts", None),
            "num_experts_per_tok": getattr(text_config, "num_experts_per_tok", None),
            "moe_intermediate_size": getattr(text_config, "moe_intermediate_size", None),
            "shared_expert_intermediate_size": getattr(text_config, "shared_expert_intermediate_size", None),
        }
    return out


def normalize_device(value):
    text = str(value)
    if text.isdigit():
        return f"cuda:{text}"
    if text.startswith("cuda:"):
        return text
    return text

try:
    memory_before = memory_snapshot()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    architectures = list(getattr(config, "architectures", []) or [])
    moe_summary = moe_text_config_summary(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    template_path = model_path_obj / "chat_template.json"
    if not chat_template and template_path.exists():
        with template_path.open("r", encoding="utf-8") as handle:
            chat_template = json.load(handle).get("chat_template")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    max_memory = {
        0: f"{gateway_max_memory_gib}GiB",
        1: f"{expert_max_memory_gib}GiB",
        2: f"{expert_max_memory_gib}GiB",
        3: f"{expert_max_memory_gib}GiB",
        "cpu": f"{cpu_max_memory_gib}GiB",
    }
    load_started = time.perf_counter()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    load_s = time.perf_counter() - load_started
    model.eval()
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}
    generate_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            thinker_max_new_tokens=max_new_tokens,
            return_audio=False,
            do_sample=False,
        )
    generate_s = time.perf_counter() - generate_started
    sequences = generated[0] if isinstance(generated, tuple) else generated
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    new_token_ids = sequences[:, prompt_tokens:]
    new_tokens = int(new_token_ids.shape[-1])
    generated_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()
    raw_device_map = getattr(model, "hf_device_map", {}) or {}
    device_map_counts = {}
    for _, dev in raw_device_map.items():
        norm = normalize_device(dev)
        device_map_counts[norm] = device_map_counts.get(norm, 0) + 1
    parameter_device_counts = {}
    parameter_device_numel = {}
    for parameter in model.parameters():
        norm = normalize_device(parameter.device)
        parameter_device_counts[norm] = parameter_device_counts.get(norm, 0) + 1
        parameter_device_numel[norm] = parameter_device_numel.get(norm, 0) + int(parameter.numel())
    used_devices = sorted([device for device in parameter_device_counts if device.startswith("cuda:")])
    memory_after = memory_snapshot()
    output = {
        "version": 1,
        "record_kind": record_kind,
        "evidence_scope": evidence_scope,
        "ok": True,
        "model": {
            "model_id": model_id,
            "model_path": str(model_path_obj),
            "model_family": model_id.split("/", 1)[0] if "/" in model_id else model_id,
            "architecture": architectures[0] if architectures else "Qwen3OmniMoeForConditionalGeneration",
            "architectures": architectures,
            "model_type": model_type,
            "dtype": dtype_name,
            "real_frontier_moe_model": True,
            "synthetic_fixture": False,
            "moe_text_configs": moe_summary,
        },
        "runtime": {
            "backend": "transformers",
            "transformers_class": "Qwen3OmniMoeForConditionalGeneration",
            "device_map_strategy": "auto",
            "max_memory": {str(k): v for k, v in max_memory.items()},
            "local_files_only": True,
            "allow_download": allow_download,
            "fornax_orchestrated": True,
        },
        "serving": {
            "request": {"model": model_id, "messages": messages, "max_new_tokens": max_new_tokens, "stream": False},
            "response": {
                "id": "fornax-real-moe-smoke-qwen3-omni",
                "object": "chat.completion",
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": generated_text}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": new_tokens, "total_tokens": prompt_tokens + new_tokens},
            },
            "generated_text": generated_text,
            "openai_compatible_shape": True,
            "live_http_endpoint": False,
        },
        "result": {
            "load_s": load_s,
            "generate_s": generate_s,
            "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens,
            "tokens_s": (new_tokens / generate_s) if generate_s > 0 else None,
            "used_devices": used_devices,
            "all_devices_used": set(used_devices) >= {"cuda:0", "cuda:1", "cuda:2", "cuda:3"},
            "device_map_counts": device_map_counts,
            "parameter_device_counts": parameter_device_counts,
            "parameter_device_numel": parameter_device_numel,
        },
        "hardware": {
            "cuda_device_count": int(torch.cuda.device_count()),
            "devices_requested": devices,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(min(4, int(torch.cuda.device_count())))],
            "memory_before": memory_before,
            "memory_after": memory_after,
            "same_physical_host": True,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch_version": getattr(torch, "__version__", "unknown"),
            "transformers_version": __import__("transformers").__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "claims": {
            "real_frontier_moe_model": True,
            "synthetic_fixture": False,
            "live_http_endpoint": False,
            "target_model_parity_reference": False,
            "formal_g2_passed": False,
            "formal_g3_passed": False,
            "g2_g3_gate_evidence": False,
            "production_distributed_serving": False,
        },
        "note": "Real Qwen3-Omni-30B-A3B MoE text-generation smoke over four same-host H100 GPUs using Transformers device_map=auto. This is real-model local serving evidence, not live HTTP, production distributed transport, or formal G2/G3 closure.",
    }
    print(json.dumps(output))
except Exception as exc:
    emit(f"real MoE generation failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)
'''


def run_local_real_moe_serving_smoke(
    *,
    out: str | Path,
    torch_python: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    model_path: str | None = DEFAULT_MODEL_PATH,
    cache_dir: str = DEFAULT_CACHE_DIR,
    devices: str | Sequence[str] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    dtype: str = "bfloat16",
    system_prompt: str = "You are a concise assistant.",
    prompt: str = "In one short sentence, say what MoE means in AI inference.",
    max_new_tokens: int = 24,
    gateway_max_memory_gib: int = 24,
    expert_max_memory_gib: int = 70,
    cpu_max_memory_gib: int = 160,
    local_files_only: bool = True,
    allow_download: bool = False,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    parsed_devices = parse_cuda_devices(devices)
    _validate_args(
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        gateway_max_memory_gib=gateway_max_memory_gib,
        expert_max_memory_gib=expert_max_memory_gib,
        cpu_max_memory_gib=cpu_max_memory_gib,
        timeout_s=timeout_s,
    )
    python = torch_python or sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(device.split(":", 1)[1] for device in parsed_devices)
    if cache_dir:
        env.setdefault("HF_HOME", cache_dir)
    if local_files_only and not allow_download:
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                _external_script(),
                model_id,
                model_path or "",
                cache_dir,
                ",".join(parsed_devices),
                dtype,
                system_prompt,
                prompt,
                str(max_new_tokens),
                str(gateway_max_memory_gib),
                str(expert_max_memory_gib),
                str(cpu_max_memory_gib),
                "1" if local_files_only else "0",
                "1" if allow_download else "0",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        data = {
            "version": 1,
            "record_kind": RECORD_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "ok": False,
            "error": f"real MoE smoke failed to launch: {type(exc).__name__}: {exc}",
            "model": {"model_id": model_id, "model_path": model_path},
            "environment": {"python_executable": python},
        }
        write_json(out, data)
        return data
    stdout = result.stdout.strip()
    if result.returncode != 0:
        data = {
            "version": 1,
            "record_kind": RECORD_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "ok": False,
            "error": "real MoE smoke exited nonzero",
            "returncode": result.returncode,
            "stdout": stdout[-4000:],
            "stderr": result.stderr.strip()[-4000:],
            "model": {"model_id": model_id, "model_path": model_path},
            "environment": {"python_executable": python},
        }
        write_json(out, data)
        return data
    try:
        data = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        data = {
            "version": 1,
            "record_kind": RECORD_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "ok": False,
            "error": f"real MoE smoke did not emit JSON: {exc}",
            "stdout": stdout[-4000:],
            "stderr": result.stderr.strip()[-4000:],
            "model": {"model_id": model_id, "model_path": model_path},
            "environment": {"python_executable": python},
        }
    if not isinstance(data, dict):
        data = {
            "version": 1,
            "record_kind": RECORD_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "ok": False,
            "error": "real MoE smoke JSON was not an object",
            "model": {"model_id": model_id, "model_path": model_path},
            "environment": {"python_executable": python},
        }
    data.setdefault("artifacts", {})
    if isinstance(data["artifacts"], dict):
        data["artifacts"]["validation"] = str(out)
    write_json(out, data)
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


def _positive_number_field(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


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


def validate_local_real_moe_serving_smoke_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "real MoE serving smoke is same-host local model evidence, not live HTTP, production distributed transport, or formal G2/G3 closure"
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
    if model_id is not None and not model_id.startswith(ALLOWED_MODEL_PREFIXES):
        errors.append("model.model_id must be a Qwen model supported by this smoke")
    _non_empty_string(model.get("model_path"), "model.model_path", errors)
    architecture = _non_empty_string(model.get("architecture"), "model.architecture", errors)
    if architecture is not None and "Moe" not in architecture and "MoE" not in architecture:
        errors.append("model.architecture must identify a MoE architecture")
    if model.get("real_frontier_moe_model") is not True:
        errors.append("model.real_frontier_moe_model must be true")
    if model.get("synthetic_fixture") is not False:
        errors.append("model.synthetic_fixture must be false")
    moe_configs = model.get("moe_text_configs")
    if not isinstance(moe_configs, dict) or not moe_configs:
        errors.append("model.moe_text_configs must be a non-empty object")
        moe_configs = {}
    found_moe = False
    for name, cfg in moe_configs.items():
        if not isinstance(cfg, dict):
            errors.append(f"model.moe_text_configs.{name} must be an object")
            continue
        experts = cfg.get("num_experts")
        experts_per_tok = cfg.get("num_experts_per_tok")
        layers = cfg.get("num_hidden_layers")
        if isinstance(experts, int) and experts > 1 and isinstance(experts_per_tok, int) and experts_per_tok > 1:
            found_moe = True
        if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
            errors.append(f"model.moe_text_configs.{name}.num_hidden_layers must be positive")
    if not found_moe:
        errors.append("model.moe_text_configs must include num_experts > 1 and num_experts_per_tok > 1")

    runtime = data.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime must be an object")
        runtime = {}
    if runtime.get("backend") != "transformers":
        errors.append("runtime.backend must be transformers")
    if runtime.get("transformers_class") != "Qwen3OmniMoeForConditionalGeneration":
        errors.append("runtime.transformers_class must be Qwen3OmniMoeForConditionalGeneration")
    if runtime.get("fornax_orchestrated") is not True:
        errors.append("runtime.fornax_orchestrated must be true")
    if runtime.get("local_files_only") is not True:
        errors.append("runtime.local_files_only must be true for this evidence artifact")

    serving = data.get("serving")
    if not isinstance(serving, dict):
        errors.append("serving must be an object")
        serving = {}
    if serving.get("openai_compatible_shape") is not True:
        errors.append("serving.openai_compatible_shape must be true")
    if serving.get("live_http_endpoint") is not False:
        errors.append("serving.live_http_endpoint must be false")
    _non_empty_string(serving.get("generated_text"), "serving.generated_text", errors)
    response = serving.get("response")
    if not isinstance(response, dict):
        errors.append("serving.response must be an object")
    elif response.get("object") != "chat.completion":
        errors.append("serving.response.object must be chat.completion")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    _positive_number_field(result.get("load_s"), "result.load_s", errors)
    _positive_number_field(result.get("generate_s"), "result.generate_s", errors)
    prompt_tokens = _positive_int_field(result.get("prompt_tokens"), "result.prompt_tokens", errors)
    new_tokens = _positive_int_field(result.get("new_tokens"), "result.new_tokens", errors)
    _positive_number_field(result.get("tokens_s"), "result.tokens_s", errors)
    used_devices = _string_list(result.get("used_devices"), "result.used_devices", errors)
    required = {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}
    if used_devices is not None and not required.issubset(set(used_devices)):
        errors.append("result.used_devices must include cuda:0,cuda:1,cuda:2,cuda:3")
    if result.get("all_devices_used") is not True:
        errors.append("result.all_devices_used must be true")
    device_map_counts = result.get("device_map_counts")
    if not isinstance(device_map_counts, dict):
        errors.append("result.device_map_counts must be an object")
        device_map_counts = {}
    for device in sorted(required):
        count = device_map_counts.get(device)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            errors.append(f"result.device_map_counts[{device}] must be a positive integer")
    parameter_counts = result.get("parameter_device_counts")
    if not isinstance(parameter_counts, dict):
        errors.append("result.parameter_device_counts must be an object")
        parameter_counts = {}
    for device in sorted(required):
        count = parameter_counts.get(device)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            errors.append(f"result.parameter_device_counts[{device}] must be a positive integer")

    hardware = data.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("hardware must be an object")
        hardware = {}
    cuda_count = _positive_int_field(hardware.get("cuda_device_count"), "hardware.cuda_device_count", errors)
    if cuda_count is not None and cuda_count < 4:
        errors.append("hardware.cuda_device_count must be at least 4")
    if hardware.get("same_physical_host") is not True:
        errors.append("hardware.same_physical_host must be true")
    requested = _string_list(hardware.get("devices_requested"), "hardware.devices_requested", errors)
    if requested is not None:
        try:
            parse_cuda_devices(requested)
        except ValueError as exc:
            errors.append(f"hardware.devices_requested invalid: {exc}")

    claims = data.get("claims")
    if not isinstance(claims, dict):
        errors.append("claims must be an object")
        claims = {}
    if claims.get("real_frontier_moe_model") is not True:
        errors.append("claims.real_frontier_moe_model must be true")
    if claims.get("synthetic_fixture") is not False:
        errors.append("claims.synthetic_fixture must be false")
    for field in [
        "live_http_endpoint",
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
            "architecture": architecture,
            "real_frontier_moe_model": model.get("real_frontier_moe_model") is True,
            "synthetic_fixture": model.get("synthetic_fixture") is True,
            "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens,
            "tokens_s": result.get("tokens_s") if isinstance(result, dict) else None,
            "used_devices": used_devices,
            "all_devices_used": result.get("all_devices_used") is True,
            "generated_text": serving.get("generated_text") if isinstance(serving, dict) else None,
            "live_http_endpoint": False,
            "g2_g3_gate_evidence": False,
        },
    }


def validate_local_real_moe_serving_smoke(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid local real MoE serving smoke artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["local real MoE serving smoke artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_local_real_moe_serving_smoke_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
