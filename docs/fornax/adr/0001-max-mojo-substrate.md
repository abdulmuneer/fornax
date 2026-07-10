# ADR 0001 — MAX/Mojo Substrate and Fork Policy

Date: 2026-07-10  
Status: Accepted  
Authority: Sponsor + TL  
Plan: `../project-plan-v4.md` §5

## Context

Fornax needs portable per-node graph compilation and kernels across NVIDIA, AMD,
and Apple without building an inference runtime from scratch. MAX supplies the
intended layer, but some Apple MoE/MLA paths currently require local patches and
MAX APIs are evolving.

## Decision

MAX/Mojo is the preferred per-node substrate. Fornax owns cross-node planning,
orchestration, and transport. MAX patches are a managed, root-pinned dependency
with reproducible builds and numerical tests.

Engine v0 uses `ReferenceStageBackend` and `SimulatedMaxStageBackend` so work can
proceed before all physical MAX backends exist. These backends implement the same
Stage ABI as the eventual `MaxStageBackend`.

## Rejected alternatives

| Alternative | Disposition |
|---|---|
| llama.cpp/ggml as primary runtime | Strong baseline, rejected as the primary cross-vendor compiler/kernels substrate |
| MLX as primary runtime | Useful Apple reference, rejected as fleet-wide substrate |
| vLLM/SGLang as primary runtime | Useful homogeneous-serving baselines, rejected for cross-vendor MAX surgery |
| Custom device runtime from scratch | Rejected for v0 scope and schedule |
| Wait for future MAX releases | Rejected; assumptions/simulation keep engine work moving |

## Consequences

- Pin upstream base and patch commit from a tracked Fornax/private build manifest.
- Prefer public MAX graph, model-extension, and custom-op APIs.
- Fork-only internal changes require numerical tests and upstream disposition.
- A token-generation smoke is not correctness evidence.
- Physical platform support remains unclaimed until T2/T3 evidence passes.

## Reversal trigger

Reopen if the required patch surface becomes materially unmaintainable, Stage ABI
integration cannot be expressed without unstable internal runtime hooks, or the
target backends fail correctness/performance gates across two consecutive pinned
MAX lineages.
