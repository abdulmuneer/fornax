# Internal Program Materials

This subtree contains **non-release program history and working knowledge**. It
is tracked so development can move between machines without losing context, but
it is not part of the Fornax runtime, public API, technical contract, or product
documentation set.

Release packaging may exclude this entire directory:

```text
docs/fornax/program_management/internal/
```

## Contents

| Folder | Purpose | Normative? |
|---|---|---|
| `reviews/` | Raw Codex/Claude reviews, review rubrics, and dated review passes | No |
| `research/` | Inputs used to construct review methods or planning analysis | No |
| `journal/` | Chronological development findings and failed-attempt history | No |
| `archive/` | Superseded status overlays and historical working backlogs | No |

## Durable sources outside this folder

- Current technical plan: [`../../project-plan-v4.md`](../../project-plan-v4.md)
- Runtime and wire contract: [`../../stage-runtime-and-wire-abi.md`](../../stage-runtime-and-wire-abi.md)
- Architecture decisions: [`../../adr/`](../../adr/)
- Evidence register: [`../../evidence-register.md`](../../evidence-register.md)
- Current program posture: [`../README.md`](../README.md)
- Gate decisions: [`../gate-reviews/`](../gate-reviews/)

Raw reviews may explain why a decision was considered, but only the current plan,
accepted ADRs, decision log, and gate records authorize implementation or claims.
