# Phase-2.5 MoE Runtime Sprint

**Goal:** implement and validate the MoE execution path in simulation/reference
form before relying on heterogeneous hardware.

**Duration:** notional W11-W17.
**Milestone:** M4 MoE expert runtime parity.
**Gate contribution:** feeds G2 distributed correctness.

## Deliverables

| # | Deliverable | Owner | Closes | DoD |
|---|---|---|---|---|
| S25-1 | Router to expert bucketing to weighted gather | RT + DIST | C1 | deterministic routing, top-k weights, gather, and reference parity tests |
| S25-2 | Local/remote expert dispatch tracing | RT + SRE | C2 | expert activations show local/remote placement, timing, and trace IDs |
| S25-3 | Expert placement and migration policy | DIST | C3 | hot-expert migration policy is bounded, explainable, and reversible |
| S25-4 | Layer/logit parity vs reference | QA + RT | C4 | per-dtype tolerance is enforced against slow-correct path |
| S25-5 | Tokenizer/chat-template/model support matrix | API + QA | H2 | template/tokenizer hash and model capability matrix are recorded |

## Sprint Board

| Deliverable | Status |
|---|---|
| S25-1 | Partial: CPU/T1 MoE routing and gather fixtures exist; optimized runtime remains open. |
| S25-2 | Partial: simulated local/remote expert probes exist; real distributed expert dispatch remains open. |
| S25-3 | Partial: migration simulation exists; live migration remains open. |
| S25-4 | Partial: CPU parity fixture exists; Phase-2.5 real parity exit remains open. |
| S25-5 | Partial: model-support fixtures exist; target model support proof remains open. |

## Validation

- `python3 -m fornax test moe-runtime`
- `python3 -m fornax test remote-expert`
- `python3 -m fornax test moe-migration`
- `python3 -m fornax test moe-parity`
- `python3 -m fornax test model-support`

## Exit Criteria

- MoE math and dispatch behavior are correct in reference/simulation form.
- Remote-expert exposure is traceable through the observability ledger.
- G2 gap is explicit: real runtime parity and real multi-node performance.
