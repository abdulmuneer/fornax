# Communications & Cadence

Thin process: enough to surface risk and reach decisions, no more.

## Rituals

| Ritual | Frequency | Owner | Output |
|---|---|---|---|
| Async standup | daily | each WS | blockers only |
| Program weekly | weekly | PM | status report + RAID review |
| Architecture review (lens rubric) | per phase / on plan change | TL | findings → plan version or RAID |
| Gate review | per gate | PM + Sponsor | DEC-\* outcome |
| External watch check | per MAX nightly/release | KER | update [06](06-dependencies-and-external-watch.md) |

The **architecture review** reuses the existing review-lens rubric
(`../review_lenses_by_skill_for_fornax.md`) as the standing rubric — the same
mechanism that produced the reconciled review.

## Weekly status report

Use [templates/status-report.md](templates/status-report.md). Covers: gate
posture, milestone burn, top-3 risks, decisions needed, external-watch delta.

## Escalation path

1. WS owner → 2. PM (unblock / re-sequence) → 3. TL (technical) / Sponsor
(scope, spend, go/no-go). Anything touching **scope, budget, or a gate** goes to
the Sponsor.

## Expectation-setting (People-lens follow-through)

- Maintain a clear **status vocabulary**: *experimental → validated-on-lab →
  supported-target → unsupported*. Never present lab results as product capability
  (mirrors the plan's honesty invariant).
- Separate the **aspirational narrative** (vision) from the **validated-capability
  ledger** (what has actually passed a gate).

## Stakeholder updates

| Audience | What | When |
|---|---|---|
| Sponsor | gate posture, risk burn, spend | weekly + gates |
| Ignis maintainers | `Engine` seam changes | as they arise |
| Design partner | validated capability only | from Phase 3 |
