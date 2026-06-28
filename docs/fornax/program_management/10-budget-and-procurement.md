# Budget & Procurement

Lightweight at this stage (program is pre-G1). The point is to start **long-lead
procurement** early and to attach a rough cost envelope to the go/no-go.

## Cost envelope (rough, pre-G1)

| Category | Phase 0 | Phases 1–3 | Notes |
|---|---|---|---|
| Headcount | small (see [07](07-resourcing-and-skills.md)) | grows | KER/Apple is the scarce hire |
| Hardware | `desktop-minimal` | + `prosumer-rack`, `lab-reference` | bundles in plan §4 |
| Cloud/CI | minimal (T0/T1 are CPU) | CI runners + lab | hardware tiers can't run in normal CI |

The **headline business metric** is cost vs an 8×H100 baseline (plan §8) — the
v0-contract quantifies $/token and $/capacity for the chosen fleet.

## Hardware bundles (procure against)

| Bundle | Contents (set exact SKUs in v0-contract; seed §3.2) | Needed by | Lead time |
|---|---|---|---|
| `desktop-minimal` | 1 multi-GPU box + 1 Apple Silicon node; **Thunderbolt direct link** ok for smoke tests | Phase 0 (D2 probe) | **start now** |
| `prosumer-rack` | **1 Linux high-VRAM two-GPU box + 1–2 high-unified-memory Macs**; **100 GbE preferred**, 25 GbE only if the model/fleet budget still closes | M2 (Phase 1 hw tier) | order by W2 |
| `lab-reference` | controlled heterogeneous lab w/ reproducible thermal conditions (benchmark of record) | calibration (§5.10) | order by W2 |

## Procurement actions (Phase 0)

- [ ] Confirm `desktop-minimal` is on hand for the Apple expert-MLP probe (D2). If
      not, this blocks the most important G1 evidence — escalate.
- [ ] Spec exact SKUs + fabric in `v0-target-contract.md`; place `prosumer-rack`
      and `lab-reference` orders by **W2** (lead time vs M2).
- [ ] Record a **negative hardware list** (plan §4) so procurement stays bounded.

## Rule

No large hardware spend beyond `desktop-minimal` is committed **before G1**
(charter guardrail: no Phase-1 investment pre-gate). Orders may be *placed* early
where lead time demands, but that is itself a Sponsor decision (DEC-\*).
