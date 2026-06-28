# Quality & Test Governance

Quality is governed through the plan's **T0–T4 test tiers** (plan §6) plus a
correctness-first rule. The PM enforces that *gates cannot pass on hardware demos
alone* — correctness evidence is required.

## Test tiers

| Tier | What | Where | Hardware |
|---|---|---|---|
| T0 | Planner/scheduler unit + golden plans, monotonicity | CI | none |
| T1 | Simulated workers: scheduling, transport contract, backpressure | CI | none |
| T2 | Single-node accelerator | lab | 1 GPU/Mac |
| T3 | 2–3 node pipeline | lab | small fleet |
| T4 | Full heterogeneous lab | lab | `lab-reference` |

T0/T1 run in CI on every change (mirrors Ignis's model-free fixture discipline).
T2–T4 run on the hardware lab and gate G2/G3/G4.

### T0/T1 command contract (plan v3 §6) — must exist before Phase 1

Equivalent commands (names may change) must exist before Phase 1 begins:

```text
fornax test golden-plans
fornax simulate --plan placement.json --requests synthetic_trace.json
fornax test runtime-format --golden golden_vectors/
fornax test network-contract --mode simulated
```

### LLM-seam acceptance tests (plan v3 §5.7) — the Ignis↔Fornax boundary

The `Engine` boundary is tested as a contract, not assumed:
- `EngineRequest`: messages, tools, response format, stop sequences, sampling
  params, max tokens, stream on/off, cancellation, **template/tokenizer version**.
- `EngineResult` / stream events: token chunks, finish reasons, tool/structured
  output, errors, cancellation result.
- The **template/tokenizer hash** used for execution is recorded (honesty).

## Correctness-first rule (non-negotiable)

Per plan §5.6: a **slow-but-obvious reference path** + **golden vectors** +
**per-dtype tolerances** exist *before* an optimized cross-vendor path is trusted.
No gate passes on throughput evidence without the matching correctness evidence.

## Definition of Done (per deliverable)

- Code: tests at the appropriate tier green; reference parity where applicable.
- Spec/doc: reviewed against the lens rubric; IDs/links resolve.
- Metric claim: traceable to a real measurement (honesty invariant) — no
  fabricated numbers, ever.

## Gate ↔ quality mapping

| Gate | Required test evidence |
|---|---|
| G1 | T0 green; v0-contract numbers from the calibrated cost model |
| G2 | T3 pipeline correctness + MoE logit parity (reference path) |
| G3 | T4 heterogeneous serve at predicted throughput + correctness |
| G4 | T4 elasticity: zero dropped in-flight on node loss |

## Benchmark discipline

- One **benchmark of record** on `lab-reference`, reproducible (commands, prompts/
  traces, versions) — productized in Phase 5, methodology stub in Phase 0.
- Benchmark harness is treated as **production code**, not throwaway scripts.
