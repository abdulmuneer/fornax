# Dependencies & External Watch

## External dependency and no-stall policy

**D-1 — Modular / MAX Apple + MoE capability.** Fornax is a *surgery of MAX*; the
Apple/Mac role (WS-D) depends on MAX capabilities Modular ships on its own
schedule. This remains a large exogenous risk (R-4), but it does not block Engine
v0. WS-J provides an Apple-shaped simulated backend under SA-001 while the
physical role remains unvalidated.

### Source precedence ladder (preserved by plan v4 §5) — the gate of record

The preserved source policy makes capability adjudication explicit: when sources disagree, **capability is
unproven until the local probe passes.** Higher rank wins.

| Rank | Source | Authority |
|---|---|---|
| 1 | **Local probe on the pinned build in the target env** | **gate of record** for Fornax role assignment |
| 2 | Package docs + changelog for the pinned build | official support status |
| 3 | Supported-model catalog / model docs | model-level availability |
| 4 | Blog posts / launch announcements | directional signal only — **never a release gate** |
| 5 | Nightly behavior | unblocks only after pinned, probed, recorded; future promises never unblock |

> Live example v3 calls out: the **26.4 blog** announced expanded Apple Silicon
> MAX model support, while **package docs** still caution that large GenAI model
> inference via MAX is not yet available on Apple Silicon. A 2026-07-01 local
> source-built MAX probe showed `DeepSeek-V2-Lite-Chat` can generate short
> output on M3 Max after Apple MLA/MoE/gather backend patches. Per the ladder,
> that result is stronger than upstream commentary but still does not replace the
> **target expert-MLP probe** that decides Apple's v0 role.

### Watch register (update each MAX nightly / release)

| Field | Value |
|---|---|
| Pinned MAX build | Current local lineage: upstream `0735fa29762a5c53d65a0456d0b53eac1472180f`, patch `957aeded5296d6638386409849b60f82c36146dd`; root pin still open (I-11). |
| Capability needed (v0) | target-model **expert-MLP** on the target Mac, within tolerance/throughput bound |
| Adjudicated by | **rank-1 local probe** (ladder above) |
| Last checked | 2026-07-01 |
| Status | partial positive Apple/MAX evidence: source-built DeepSeek-V2-Lite-Chat short `generate` passed on M3 Max; local expert-MLP probe and serving-grade validation still required |
| Reversal trigger armed? | yes — Engine v0 is fixed at T0/T1; failed G2 parity demotes Apple without invalidating the engine contract |
| Owner | KER |

### Latest source snapshot - 2026-07-01

| Source class | Current observation | Gate effect |
|---|---|---|
| Official changelog / release notes | Modular's MAX changelog and 26.4 notes indicate expanded Apple Silicon model serving, especially M3+ support for common Llama/Qwen-family paths that fit memory, with later nightly work extending Apple correctness coverage. | Raises priority for the Apple probe; does not close it. |
| Official package caveat | The MAX packages page still carries the broader caveat that large GenAI model inference via MAX is not generally available on Apple Silicon. | Keeps Apple compute roles unproven until a pinned local probe passes. |
| Social / market signal | Recent X discussion mostly frames Mojo + MAX as a cross-vendor CUDA alternative and interprets the reported Qualcomm/Modular deal as a strategic portability bet. It does not provide usable correctness, throughput, or target-model evidence for Fornax. | Non-gating context only. Do not use it as G2/G3 evidence. |
| Local Fornax evidence | Source-built MAX in `external/modular` generated 1 and 8 tokens from `DeepSeek-V2-Lite-Chat` on M3 Max after local Apple MLA prefill/decode fallback, MoE indices fallback, Metal dispatch scalar, and rank-2 gather work. Evidence: [../deepseek-v2-lite-max-check.md](../deepseek-v2-lite-max-check.md). | Shows Apple can run a full short DeepSeek path in the patched source tree; does not close G2 because numerical parity, serving, long-context/batching, and target stage/expert evidence remain open. |

ADR-ready stance for `adr/0001-max-mojo-substrate.md`:

> Fornax keeps MAX/Mojo as the preferred substrate, but Apple Silicon support is
> a measured capability, not an upstream promise. Official Modular release notes
> are moving in the right direction, while official package caveats remain more
> conservative. Therefore the ADR must pin the exact MAX/Mojo build, record the
> exact Mac SKU and OS/toolchain, and assign Apple's v0 role only from the local
> expert-MLP probe. If the target expert MLP cannot meet the target contract's
> tolerance and throughput bound at G2/G3, Apple is demoted to capacity-only and
> the physical target is narrowed rather than proceeding on a promise.

Upstream anchors:

- Modular 25.6 Apple GPU direction:
  https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple
- Modular 26.4 MoE + Apple note:
  https://www.modular.com/blog/modular-26-4-sota-moe-serving-model-bringup-via-agent-skills-mojo-beta-2-and-more
- MAX changelog:
  https://docs.modular.com/max/changelog/
- MAX package/platform caveats:
  https://docs.modular.com/max/packages/
- MAX custom ops:
  https://docs.modular.com/max/develop/build-custom-ops/

All upstream anchors are rank-2/3/4 signals under the ladder above. They inform
what to probe; they do not gate Apple's Fornax role.

**Policy:** the program never assumes a future MAX capability. It commits only to
the Apple role the *currently measured* build supports, and re-checks per nightly
(R-4 mitigation). Capability changes are logged here and, if they change a
decision, recorded as a `DEC-*` ([08](08-decision-log.md)).

## Internal dependencies

| Dep | Blocks | Owner | Note |
|---|---|---|---|
| D-4 WS-A planner | placement authority and physical phases | DIST | Phase 0.5 repairs complete; measured G2 calibration active |
| D-2 Hardware procurement | G2/G3 physical evidence | PM | Does not block Engine v0 — [10](10-budget-and-procurement.md) |
| D-3 Ignis `Engine` seam | WS-H integration | TL | Keep `generate(...)` stable |
| WS-B format/Stage ABI | WS-C, WS-E, WS-F, WS-J | RT | The load-bearing invariant |
| D-5 physical lab availability | G2/G3 only | PM | Engine v0 uses named assumptions until available |

## Dependency-driven sequencing rules

1. **Contract before backend.** Stage ABI/runtime/network specs bind every
   reference, simulated, and physical backend.
2. **Engine and validation in parallel.** WS-J simulation advances while WS-D and
   RT collect physical MAX evidence opportunistically.
3. **Assumptions are dependencies.** Each SA-* has a validation test and blocked
   gate; it is never silently promoted to fact.
4. **Planner repairs during Engine v0.** A7/A8 are required before the planner can
   authorize physical placement.
5. **Procure deliberately.** Hardware availability blocks G2/G3, not software
   construction; avoid premature broad-fleet purchases.
