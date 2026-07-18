# RAID Log — Risks, Assumptions, Issues, Dependencies

Living register. Reviewed at the weekly ([09](09-communications-and-cadence.md))
and at every gate. IDs are **stable**; entries are added/updated/retired, never
renumbered. Seeded from plan v4 §11–§12 and the SA-* technical register.

## Risks (R-*)

R-1…R-10 preserve the plan v3 risk lineage; R-11…R-13 are added by plan v4;
R-14…R-16 capture the 2026-07-13 founder-review disposition; R-17…R-18
capture the 2026-07-17 external-readiness review.
Rank numbers remain historical so IDs and prior reporting stay comparable.

| Rank | ID | Risk | Owner | P | I | Mitigation / trigger | Status |
|---|---|---|---|---|---|---|---|
| 1 | R-4 | **Apple/MAX readiness misses target role (critical path)** | KER | H | H | Source precedence (§5.4); pinned probes; staged roles; demotion trigger (§5.5) | Open |
| 2 | R-8 | **Concurrency–market fit** | PM | M | H | §3.3 persona + §3.2 seed sweep; narrow product claim if absent | Open — does not block Engine v0; gates G3/product positioning |
| 3 | R-5 | Remote-expert wait dominates decode | DIST | M | H | Budget remote hits; migrate hot; disable remote if unprofitable; calibrate (§5.10) | Open |
| 4 | R-3 | Heterogeneous numerics divergence | LLM | M | H | Format spec + reference path + golden vectors + per-dtype tol (§5.6) | Open |
| 5 | R-1 | Commodity network caps throughput | TL | M | H | Fabric tiers; measured links; pipeline sizing; replication | Open |
| 6 | R-6 | Surgery vs fast-moving MAX internals | RT | M | M | Pin build; thin seam; ADR source watch (§5.4) | Open |
| 7 | R-7 | Mojo toolchain maturity | RT | M | M | Lean on MAX kernels; minimal custom; fallback tests | Open |
| 8 | R-2 | Pipeline depth ↔ latency | DIST | H | M | Honest positioning; depth penalty in planner | Open |
| 9 | R-9 | **Security/backpressure slips past prototype** | NET | L(gated) | H | Spec before Phase 1a; impl before Phase 3/product (§5.8) | Open |
| 10 | R-10 | **Status drift: planned artifacts look proven** | PM | M | M | §0 gate-status table; owner/checklist per artifact; honesty rule (§12 metrics) | Open |
| 11 | R-11 | **Simulation reality gap**: engine passes assumed profiles but physical backends violate them | WS-J | H | H | SA-* register, scenario envelope, same ABI, physical replacement tests | Open |
| 12 | R-12 | Engine implementation stalls waiting for unavailable hardware | TL | M | H | G1 authorizes reference/simulated backends and loopback engine | Mitigated for M1; monitor physical-resource delay at G2 |
| 13 | R-13 | MAX fork/dependency cannot be reproduced or rebased | RT | M | H | Root pin, fresh-build procedure, numerical/rebase CI | Partially mitigated: exact working-tree pin, patch artifact, and deterministic reconstruction exist; repository commit, fresh clean platform packets, and rebase CI remain open |
| 14 | R-14 | **ABI v1 cannot express ragged multi-sequence MAX batching** | RT/DIST | H | H | ABI v2 sequence descriptors, row offsets, per-request KV/errors, two-worker and physical conformance | Mitigated at T0/T1 by candidate FNX2; physical MAX conformance still blocks a product/G2 batching claim |
| 15 | R-15 | **Customer/concurrency/economic wedge is unvalidated** | PM/SP | H | H | Qualified interviews, real traces, design partners, paid pilot, explicit NARROW trigger | Open — parallel discovery lane |
| 16 | R-16 | **Services become non-repeatable kernel consulting** | PM/TL | M | H | Every engagement must produce reusable profiles, automation, conformance, certified config, or upstream path | Open |
| 17 | R-17 | **Unverified named-person or founding-team claims create diligence, employment, IP, or endorsement exposure** | SP/PM | H | H | XG-1 signed roles/time/IP/conflicts/governance/name permission; functional-role language until closed | Open — blocks named external team claims |
| 18 | R-18 | **MAX commercial-use, device, packaging, branding, or redistribution terms do not support the intended product** | SP/RT | M | H | XG-2 component/license map, counsel memo, packaging design, notices, written clarification/commercial terms | Open — blocks product distribution claim |

P/I = probability/impact (L/M/H). R-4 and R-8 no longer block Engine v0; they can
still force a NARROW/KILL decision for physical platform scope or product
positioning at G2/G3. **R-10 was added in v3** to guard against the exact failure
the v2 review caught (naming a blocker ≠ resolving it).

## Assumptions (A-*) — each has a validation owner & method

| ID | Assumption | Validates via | Owner | Status |
|---|---|---|---|---|
| A-1 | The persona supplies pipeline-filling concurrency (seed goal: saturate ≤ 32 in-flight, §3.2) | B2 sweep in v0-contract | PM | Unvalidated → G3/product |
| A-2 | MAX can run the target expert-MLP on the target Mac acceptably | D2 probe | KER | Unvalidated → G2/G3 |
| A-3 | The named fabric tiers (§4) are procurable/available | procurement ([10](10-budget-and-procurement.md)) | PM | Unvalidated |
| A-4 | One quant format is byte-compatible across MAX backends | B1 format spec | RT | Contract accepted; physical compatibility unvalidated → G2 |
| A-5 | Required skills (esp. KER/Apple) are staffable | [07](07-resourcing-and-skills.md) | PM | Unvalidated |
| A-6 | Partitioner cost model predicts within the provisional bound | calibration (§5.10) | DIST | Unvalidated → G2 |
| A-7 | Production Stage ABI behavior can be developed against reference/simulated backends without architecture divergence | Backend conformance suite + later T3 replay | RT | Validated at T0/T1 by DEC-008; physical equivalence remains open → G2 |
| A-8 | Intended local fabric lies within the named simulation envelope | Fabric probe on acquired/available route | NET | Unvalidated → G2/G3 |
| A-9 | MAX can construct/load useful contiguous layer-range stages | First physical `MaxStageBackend` compile/load | RT | Unvalidated → G2 |
| A-10 | Simulation scenarios bracket enough real behavior to guide scheduling/planning | Compare scenario predictions to each physical profile | WS-J | Unvalidated → G2 |
| A-11 | Qualified buyers have sustained concurrency and a budgeted problem that requires multi-node spanning | Interview record + anonymized traces + pilot commitment | PM/SP | Unvalidated → commercial NARROW decision |
| A-12 | Deployment/model-enablement services can become repeatable and software-leveraged | Pilot delivery/support accounting and reusable-asset review | PM/TL | Unvalidated |
| A-13 | Commercial terms can be secured for every MAX component, supported device, patch, package, and brand use required by the product | Counsel-reviewed component/license map plus written clarification or agreement | SP/RT | Unvalidated → external distribution/G5 |

Detailed hardware assumptions SA-001…SA-010 live in
[`../simulation-and-assumption-contract.md`](../simulation-and-assumption-contract.md).

## Issues (I-*) — known gaps to close now

| ID | Issue | Owner | Due | Status |
|---|---|---|---|---|
| I-1 | `v0-target-contract.md` not written | DIST/PM | G1 | Closed at Engine v0 scope; physical fields carry to G2 |
| I-2 | `runtime-format-and-invariants.md` not written | RT | G1 | Closed: v1 contract and T0/T1 conformance implemented; physical conformance open at G2 |
| I-3 | `networking-security-and-backpressure.md` not written | NET | G1 | Closed: Engine v0 design and loopback implementation; physical/product validation open |
| I-4 | `adr/0001-max-mojo-substrate.md` not written | TL | G1 | Closed; ADR accepted 2026-07-10 |
| I-5 | KER/Apple staffing gap unresolved | PM | W2 | Open |
| I-6 | Phase-0 preflight workflow not written (§3.4) | DIST/SRE | G1 | Closed at workflow/proxy scope |
| I-7 | Planner does not charge remote expert memory/resources to the expert host | DIST | Phase 0.5 | Closed by resource aggregation and regressions; DEC-008 |
| I-8 | Planner >6-node pruning can discard the only feasible high-memory node | DIST | Phase 0.5 | Closed by feasibility-preserving search and seven-node regression; DEC-008 |
| I-9 | No executable two-worker Engine v0 using the shared Stage ABI target | RT/DIST | Phase 0.5 | Closed for recorded T0/T1 scope by experimental FNX1 two-process prefill/decode and 30-minute bundle; DEC-008/terminology erratum |
| I-10 | Stage ABI v1 has no implementation/conformance corpus yet | RT/NET | Phase 0.5 | Closed by FNX1 codec and 24-check golden corpus; DEC-008 |
| I-11 | Root Fornax does not pin the local MAX fork lineage | RT | G2 entry | **Working-tree mechanism implemented; repository closure pending:** exact base/patch/tree, diff, validator, and deterministic reconstruction are present but uncommitted. Commit them before treating the pin as durable; each physical evidence run also requires a clean verified platform build packet. |
| I-12 | Technical source-of-truth/program docs are loose untracked files | SP/PM | Phase 0.5 | Open: choose tracked public/private home |
| I-13 | Stage ABI v1 lacks stage-role/input-kind and ragged per-sequence row/position/KV/error descriptors | RT/DIST | Before G2 batching claim | **Closed at T0/T1 by candidate FNX2:** exact codec, reference oracle, golden vectors, integrated scheduler, and two independent loopback workers. Physical MAX conformance remains open. |
| I-14 | Worker backend is simulator-hardcoded and `/capabilities` echoes requested manifest values instead of discovered backend facts | RT | G2 | **Closed at T0/T1:** serializable fail-closed backend factory and backend-originated pre-load attestation implemented; physical adapter capabilities remain G2 evidence |
| I-15 | No qualified customer interviews, traffic traces, design-partner commitments, priced pilot, or willingness-to-pay evidence | PM/SP | Parallel with G2 | Open |
| I-16 | Planner still lacks runtime/build/operation/quantization/calibration capability admission; KV admission is now fixed | DIST/RT | G2 | **Hardware-independent mechanism closed at T0:** structured provenance/confidence/error, exact capability admission, exploratory labeling, and fail-closed deployment mode are implemented. Authentic physical sources and <=20% measured calibration remain open. |
| I-17 | Python tensor copies/scalar finite checks remain the data-plane reference implementation | RT/NET | Before performance optimization/claim | Partially mitigated by a bounded, copy-explicit `TensorBufferAdapter` seam with health counters; pooled/zero-copy native import, incremental native validation, and physical benchmarks remain open. Preserve the Python oracle. |
| I-18 | Founding-team roles, time, contribution history, equity/governance, IP assignment, conflicts, and name-use permissions are not evidenced | SP/PM | Before named external pitch | Open — XG-1 |
| I-19 | No counsel-reviewed MAX commercial-use, supported-device, branding, patched-runtime packaging, or redistribution disposition | SP/RT | Before product distribution claim | Open — XG-2 |
| I-20 | No bottoms-up monthly cash model, hiring dates/loaded costs, hardware BOM/quotes, cash/burn, or approved financing parameters | SP/PM | Before external ask | Open — XG-3 |
| I-21 | Fundraising package remains an internal working tree with blocked readiness gates and no approved disclosure index | PM/SP | Before external circulation | Open — XG-8 |
| I-22 | Request/KV/idempotency/transform/event state needs a complete production lifecycle | RT/DIST | Before production memory/stability claim | **Mechanism mitigated at T0/T1:** explicit release, idle expiry, internal execution leases, same-worker reconnect tombstones, count/time/byte bounds, high-water health, and unique-request pressure tests are implemented. EV-016 is a non-authoritative 113,718-request candidate whose civil-time gap prevents an uninterrupted-soak claim. Open: committed-source continuity-aware rerun, restart-durable fences, and physical native-KV/buffer/RSS validation. |

## Dependencies (D-*)

| ID | Dependency | Type | Owner | Detail |
|---|---|---|---|---|
| D-1 | Modular/MAX Apple + MoE capability | **External** | KER | [06](06-dependencies-and-external-watch.md) |
| D-2 | Hardware procurement (bundles) | Internal/vendor | PM | [10](10-budget-and-procurement.md) |
| D-3 | Ignis `Engine`-trait seam stability | Internal | TL | §9 plan |
| D-4 | WS-A planner → all downstream phases | Internal | DIST | Phase 0.5 defects closed; physical calibration remains G2 work |
| D-5 | Physical heterogeneous lab availability | Internal/vendor | PM | Engine v0 complete without it; blocks G2/G3 evidence |
| D-6 | Qualified design partners and customer workload traces | External/customer | PM/SP | Blocks commercial validation and frontier workload selection, not Engine v0 |
| D-7 | MAX commercial license, trademark, supported-device, and redistribution terms | External/legal | SP/RT | Blocks product packaging/distribution claims; tracked by XG-2 and [06](06-dependencies-and-external-watch.md) |

> Refresh rule: when the plan version changes, re-seed R-* from §7 and I-* from
> §10, **preserving existing IDs and human-added entries** (see the
> `fornax-program-manager` skill).
