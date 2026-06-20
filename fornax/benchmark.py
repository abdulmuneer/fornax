from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any


DEFAULT_MODE = "tiny-moe-or-expert-mlp"


def _weights(rows: int, cols: int, offset: int) -> list[list[float]]:
    return [
        [(((r + 1) * (c + 3 + offset)) % 17 - 8) / 17.0 for c in range(cols)]
        for r in range(rows)
    ]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


def _relu(values: list[float]) -> list[float]:
    return [value if value > 0.0 else 0.0 for value in values]


def _run_expert(
    vector: list[float], weights: tuple[list[list[float]], list[list[float]]]
) -> list[float]:
    up, down = weights
    return _matvec(down, _relu(_matvec(up, vector)))


def run_tiny_expert_mlp_benchmark(
    *,
    iterations: int = 25,
    batch_tokens: int = 8,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    experts: int = 4,
    top_k: int = 2,
) -> dict[str, Any]:
    """Run a tiny deterministic CPU expert-MLP benchmark.

    This is a Phase-0 smoke benchmark: it proves the benchmark path records a
    real measurement and checksum, not that a target MoE or accelerator is fast.
    """

    for name, value in {
        "iterations": iterations,
        "batch_tokens": batch_tokens,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "experts": experts,
        "top_k": top_k,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if top_k > experts:
        raise ValueError("top_k cannot exceed experts")

    batch = [
        [((token + 1) * (dim + 5) % 23 - 11) / 23.0 for dim in range(hidden_dim)]
        for token in range(batch_tokens)
    ]
    expert_weights = [
        (
            _weights(intermediate_dim, hidden_dim, expert_id),
            _weights(hidden_dim, intermediate_dim, expert_id + 11),
        )
        for expert_id in range(experts)
    ]
    checksum = 0.0
    expert_calls = 0
    started = time.perf_counter_ns()
    for iteration in range(iterations):
        for token_index, vector in enumerate(batch):
            for rank in range(top_k):
                expert_id = (token_index + rank + iteration) % experts
                weight = 1.0 / (rank + 1)
                output = _run_expert(vector, expert_weights[expert_id])
                checksum += weight * sum(output)
                expert_calls += 1
    elapsed_ns = time.perf_counter_ns() - started
    elapsed_s = elapsed_ns / 1_000_000_000.0
    tokens_processed = iterations * batch_tokens
    return {
        "mode": DEFAULT_MODE,
        "measured": True,
        "source": "fornax.benchmark.tiny_expert_mlp.cpu_stdlib",
        "note": (
            "Measured deterministic CPU microbenchmark for Phase-0 benchmark plumbing; "
            "not target-model, MAX, or accelerator throughput evidence."
        ),
        "config": {
            "iterations": iterations,
            "batch_tokens": batch_tokens,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "experts": experts,
            "top_k": top_k,
            "weights_precomputed_before_timing": True,
        },
        "result": {
            "elapsed_s": elapsed_s,
            "elapsed_ns": elapsed_ns,
            "tokens_processed": tokens_processed,
            "expert_calls": expert_calls,
            "tokens_s": tokens_processed / elapsed_s if elapsed_s > 0 else None,
            "expert_calls_s": expert_calls / elapsed_s if elapsed_s > 0 else None,
            "checksum": checksum,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


def benchmark_from_plan(
    plan: dict[str, Any],
    *,
    mode: str = DEFAULT_MODE,
    iterations: int = 25,
) -> dict[str, Any]:
    if mode != DEFAULT_MODE:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    predicted = plan.get("predicted")
    if predicted is None:
        raise ValueError(f"infeasible plan: {plan.get('infeasible_reason')}")
    result = run_tiny_expert_mlp_benchmark(iterations=iterations)
    result["plan_predicted"] = predicted
    return result
