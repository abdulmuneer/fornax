# Founder proxy-review follow-up

Date: 2026-07-17  
Baseline reviewed: `founder-review-action-plan-2026-07-13.md`  
Plan of record: `project-plan-v4.md` (unchanged)  
Disposition: **ITERATE — stronger executable specification; not launch-ready**

> This is a proxy review requested through two lenses associated with Chris
> Lattner (platform/compiler/runtime rigor) and Soumith Chintala
> (human-centered framework and developer experience). Neither person
> participated in, approved, or endorsed this review or Fornax. It is not
> permissible evidence for a named-founder, adviser, team, or endorsement claim.

## Joint proxy verdict

The repository is materially more credible than the 2026-07-13 baseline. It now
has a fail-closed physical-backend seam, backend-originated capability
attestation, an explicit public engine facade, a versioned adapter SDK, exact
FNX1 1.0 negotiation, candidate FNX2 2.0 ragged reference execution, bounded
replay/lifecycle state, a provenance-aware planner authority mode, and a
working-tree root-pin mechanism and fail-closed G2 packet builder.

It still is not a heterogeneous inference product. There is no concrete physical
`MaxStageBackend`, real NVIDIA-to-Apple Fornax request, integrated ragged
continuous batching on physical MAX adapters, true
tokenization/sampling/detokenization path, supported serving endpoint,
authenticated physical planner calibration, customer proof, or cleared external
launch package. The correct financing story is a gated
mechanism-and-design-partner proof, after the external-readiness controls permit
circulation.

## Disposition of review comments

| Review concern | Action and evidence now | Proxy status | Remaining satisfaction condition |
|---|---|---|---|
| Physical execution could silently remain simulation | Workers receive an explicit serializable backend spec; a `max` spec requires an importable factory and fails startup rather than falling back. The accepted MAX base/patch/tree has a reconstructable working-tree pin that the G2 runner re-verifies. | **Satisfied at the seam; lineage mechanism verified but uncommitted; physical proof open** | Commit the lineage files, implement the concrete MAX adapter, produce clean per-host builds, pass API v2/FNX2 conformance, then acquire T2/T3 parity and stability evidence. |
| `/capabilities` echoed requested values | Backend, build, device, dtype, ABI, memory, operation, quantization, and frame facts originate from the constructed backend and are recorded beside requested requirements. | **Satisfied at T0/T1** | Demonstrate truthful discovery on each physical adapter and record the accepted inventory. |
| Request/KV/replay state could grow indefinitely | Final release, idle expiry, internal execution leases, same-worker tombstones, count/time/byte caps, high-water health, and unique-request pressure tests now cover reference/loopback state. EV-016 completed 113,718 requests within bounds. | **Mechanism satisfied at T0/T1; EV-016 is non-authoritative** | Rerun the continuity-aware pressure runner from committed source, add restart-durable fences, and validate native KV/buffer/RSS bounds on physical adapters. |
| “At most once” was broader than the implementation | FNX1 remains bounded-window replay only. Candidate FNX2 adds per-request execution leases, KV epochs, semantic replay identity, release/expiry tombstones, and partial-failure isolation in the reference path. | **Satisfied for bounded same-worker T0/T1 claims** | Prove reconnect/restart behavior and replay the lease corpus across physical workers before resilience claims. |
| FNX1 version negotiation accepted ambiguous minors | Codec and channel negotiation require exact ABI 1.0; future minor frames are rejected. The golden corpus is 31 checks. | **Satisfied at T0/T1** | Repeat the same corpus on physical T3 transport. |
| ABI v1 cannot represent real ragged batches | Candidate FNX2 now has an exact codec and frozen golden, stage roles/input kinds, ragged row slices, positions, per-sequence KV/errors, unequal prefill, independent decode, compacted results, release/expiry, leases/tombstones, multi-dimensional credit, a slow oracle, an integrated scheduler, and two separately spawned loopback workers. | **Satisfied at T0/T1; physical conformance open** | Replay the same corpus through concrete MAX adapters, measure the native hot path, and record the compatibility/rollout decision. |
| Public use was not humane or explicit | `Engine` now requires an explicit `str -> str` generator and rejects invalid results; the package exposes `fornax.backends`, quickstart/help/error behavior, adapter conformance, and package-relative assets. No echo/default generator pretends to be a product. | **Satisfied for pre-alpha contracts; product DX open** | Add a real generator, stable serve surface, diagnostics, install/upgrade path, and fresh-operator physical acceptance. |
| Planner capability and calibration claims exceeded code | Plans now carry structured source provenance, confidence, input/prediction error, and an explicit authority result. Deployment mode exactly admits runtime/build/operations/dtype/quantization/role capabilities and fails closed; default fixtures remain exploratory. | **Hardware-independent mechanism satisfied; physical evidence open (I-16)** | Authenticate the declared sources and establish measured stage/link/native-memory calibration within the G2 bound. |
| Python tensor materialization is not a production hot path | The Python oracle remains slow and explicit. A bounded, copy-explicit `TensorBufferAdapter` now separates staged buffers from backend execution and exposes health/high-water state. | **Seam satisfied; zero-copy performance open (I-17)** | Add pooled/zero-copy native buffers, incremental native integrity/finite checks, MAX import, and physical component benchmarks without weakening the oracle. |
| Historical evidence could masquerade as current authority | EV-009 remains immutable and validates its recorded scope, while the validator now reports `current_contract_authority=false` because later attestation/replay/lifecycle requirements are absent. New evidence instructions require a new dated ID/artifact/hash. | **Satisfied as governance** | Generate current sustained and physical artifacts; do not overwrite EV-009. |
| Customer/concurrency fit and business model were assumed | Discovery, traffic-trace, design-partner, paid-pilot, support-burden, pricing, and kill-metric evidence are explicit gates. No traction is claimed. | **Open external work** | Qualify the wedge with observed workloads and paid evidence, or record `NARROW`/`KILL`. |
| Team, MAX rights, financing, and pitch provenance were unsafe | XG-1…XG-8 now block named-person, commercial-rights, ask, capability, traction, security, market, and package claims independently. | **Control satisfied; facts open** | Principal/counsel approval and durable evidence must close or explicitly dispose each blocking gate. |

## Material corrections made in this follow-up

- Reclassified FNX1 as an experimental T0/T1 mechanism contract rather than a
  production ABI and added an immutable terminology erratum.
- Corrected “logits” in Phase 0.5 to mean a hidden-width mechanism tensor, not
  real vocabulary logits or generated text.
- Preserved the frozen FNX1 lockstep path and added candidate FNX2's exact
  ragged codec, slow oracle, integrated scheduler, and independent two-worker
  T0/T1 path without calling it physical batching.
- Added a public `Engine` contract and Stage Backend API v2 rather than exposing
  internal simulation objects as a product API.
- Added exact replay identity, bounded histories, explicit release/expiry,
  execution leases, same-worker tombstones, count/time/byte caps, and pressure
  regressions without changing FNX1 1.0 bytes.
- Hardened failure paths so simulated results become replay-visible only after
  final fault/timing decoration, upstream results replay after a downstream
  stage failure, request-level sequence rejection preserves credit for final
  release, partial release fences execution until cleanup completes, and invalid
  requests cannot consume admission slots.
- Added structured planner provenance/confidence/error and exact capability
  admission. Default fixtures remain exploratory; deployment mode fails closed
  without accepted measured inputs and calibration.
- Root-pinned the accepted MAX base/patch/tree and added deterministic
  reconstruction plus a G2 packet builder whose physical summaries are derived
  from nonce-correlated raw samples. V9 uses runner-observed wall time.
- Added a bounded, copy-explicit native-buffer adapter seam while retaining the
  Python tensor path as the conformance oracle.
- Converted product, blog, fundraising, and market statements from present-tense
  capability claims into target language where physical evidence is absent.
- Preserved historical gate and evidence records instead of rewriting them to
  match the hardened working tree.

## Verification status

Verification refreshed 2026-07-18; the working tree reports:

- Stage ABI v1 golden conformance: **31/31**;
- candidate FNX2 golden conformance: **15/15** through two workers;
- Stage Backend API v2 functional smoke definition: **12 checks**;
- full Python unit suite: **386/386 passed** in the socket-enabled local run;
- EV-016 lifecycle pressure: **113,718 requests within configured bounds**, but
  non-physical, uncommitted-source, and not an uninterrupted-soak claim; and
- compile and whitespace/diff checks: passed.

These are T0/T1 contract results. They do not establish MAX numerical parity,
physical heterogeneous correctness, throughput, supported platforms, or
production stability.

## Pitch and launch decision

| Use | Decision |
|---|---|
| Internal architecture, hiring, lab, or design-partner discussion with the evidence boundary visible | **Proceed** |
| External proof-round narrative | **Blocked until the applicable XG-1/XG-2/XG-3/XG-8 dispositions authorize circulation; present G2 as the financed proof, not an achieved capability** |
| Claim that Chris Lattner or Soumith Chintala founded, reviewed, joined, or endorsed Fornax | **Prohibited without XG-1 evidence and their permission** |
| Product launch, supported heterogeneous serving, performance, production memory stability, or GA pitch | **Do not proceed** |

No repository-only review can truthfully claim “their satisfaction.” Under the
two requested proxy lenses, the immediate contract, truthfulness, and pre-alpha
developer-experience comments have been acted on. The launch threshold remains
unmet until the physical, batching, calibration, customer, legal, and operating
evidence above is real.
