from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


def summarize_request_trace(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    if isinstance(data, dict):
        requests = data.get("requests")
    else:
        requests = data
    if not isinstance(requests, list):
        raise ValueError("request trace must be a list or an object with a 'requests' list")

    total_prompt = 0
    total_gen = 0
    for idx, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {idx} must be an object")
        prompt_len = int(request.get("prompt_len", request.get("prompt_tokens", 0)))
        gen_len = int(request.get("gen_len", request.get("max_new_tokens", 0)))
        if prompt_len < 0 or gen_len < 0:
            raise ValueError(f"request {idx} has negative token counts")
        total_prompt += prompt_len
        total_gen += gen_len

    return {
        "path": str(path),
        "request_count": len(requests),
        "total_prompt_tokens": total_prompt,
        "total_generation_tokens": total_gen,
    }


def simulation_result(predicted: dict[str, Any], request_trace: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"predicted": predicted}
    if request_trace is not None:
        throughput = float(predicted["throughput_tok_s"])
        total_generation = int(request_trace["total_generation_tokens"])
        result["requests"] = {
            **request_trace,
            "predicted_decode_wall_time_s": (
                total_generation / throughput if throughput > 0 else None
            ),
        }
    return result
