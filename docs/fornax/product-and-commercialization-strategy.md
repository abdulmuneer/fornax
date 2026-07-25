# Fornax product and commercialization strategy

Status: public-repository working brief; hypotheses are not customer evidence

Plan of record: `project-plan-v4.md`

Last updated: 2026-07-25

This file is publicly accessible. Do not put confidential customer, partner,
pricing, topology, security, fundraising, or unannounced product information
here merely because a section is labeled “internal” or “working.”

Non-public commercial and disclosure materials are maintained outside this
public repository. No public document authorizes external claims about
financing, counterparties, commercial terms, team, rights, or traction.
The current platform/developer-experience proxy-review disposition is
[`founder-proxy-review-follow-up-2026-07-17.md`](founder-proxy-review-follow-up-2026-07-17.md).

## Investor one-liner

Fornax aims to build a private inference fabric in which mixed on-prem
accelerators serve one open sparse-MoE model larger than any individual machine,
using MAX for qualified physical node execution and Fornax for cross-node
planning, orchestration, and transport. The integrated physical endpoint is an
open G2/product milestone.

## Fundraising truth

This is a proof-and-design-partner financing story. The repository has
experimental FNX1 v1 contracts, candidate FNX2 integrated ragged T1 execution,
bounded lifecycle mechanisms, a fail-closed planner authority mode, root-pinned
G2 packet automation in the uncommitted working tree, and single-node Apple MAX
bring-up. The root-pin mechanism becomes durable repository lineage only after
commit. It does not yet have a
physical cross-vendor request, restart-durable lifecycle fencing, reviewed
long-running physical memory evidence, authenticated physical calibration, a
bound frontier-capacity target, customer evidence, or product economics.

The financed milestones are:

1. G2 physical cross-vendor correctness and calibration.
2. A qualified design-partner wedge with real traffic evidence.
3. G3 frontier-capacity economics on a customer-representative fleet.
4. A repeatable pilot package and support motion.

## Buyer and wedge hypotheses

The initial buyer hypothesis is a technical small or mid-sized firm that:

- must keep prompts, KV state, and model execution on premises;
- already owns a mixed fleet or has a constrained capital budget;
- needs an open model larger than any one available node;
- has sustained shared/agentic concurrency rather than one bursty user;
- can operate a provisioned high-speed local fabric; and
- values deployment and model-enablement support.

These are hypotheses. If qualified buyers do not supply enough concurrency to
fill a spanned pipeline, Fornax must `NARROW` to capacity-first serving,
homogeneous islands, a different fleet, or a different buyer.

## Product and service offer

| Offer | Customer outcome | Boundary | Current evidence |
|---|---|---|---|
| Community engine/contracts | Inspect targets, fleets, plans, simulations, and evidence | Apache-2.0 Python reference layer | Runnable T0/T1 |
| Enterprise Fornax runtime | One supported endpoint over certified model/fleet combinations | Physical MAX backends, orchestration, security, operations | Not yet available; G2/G3 open |
| Fleet assessment | Determine whether a buyer's model, hardware, fabric, traffic, and constraints can close | Fixed-scope evidence report; no unsupported performance promise | Method exists; customer delivery unproven |
| Deployment service | Install and validate one certified configuration | Repeatable runbook and acceptance packet | Not yet repeatable |
| Model enablement | Add and certify a model/backend combination | Explicit scope, parity corpus, performance profile, upstream policy | Single-node bring-up only |
| Optimization and support | Calibrate placement, monitor regressions, manage upgrades | Productized profiles and supported matrix; avoid indefinite bespoke forks | Hypothesis |

Services should accelerate repeatable product adoption, not hide a consulting
business. Every service engagement must produce reusable profiles, automation,
conformance coverage, or a certified configuration. Bespoke kernel work without
a reuse/upstream path requires an explicit margin and roadmap decision.

## Configuration qualification portfolio

After the core G2/G3 mechanism is physically proven, the Community/Prosumer
product can expand through exact certified configuration profiles. It must not
promise the Cartesian product of every device, model, quantization, context,
runtime, and topology.

Candidate families—not current support commitments—may include high-memory Apple
Silicon; NVIDIA RTX, H100/B100/B200, and DGX-class devices or islands; AMD Radeon
and Instinct-class systems; and other accelerator families only after substrate,
legal, operational, and physical evidence. The planner and current T0/T1 support
fixtures do not yet admit all of these families.

Each profile must bind:

- exact device/island SKU, memory, firmware, OS, driver, runtime, and build;
- exact model, tokenizer/template hashes, quantization, dtype, and license;
- allowed roles such as dense stage, expert worker, KV tier, capacity store, or
  sampler;
- topology and measured sustained links;
- context, request distribution, concurrency, and SLO envelope;
- correctness, capacity, calibration, failure, security, and operator evidence;
- evidence date, expiry, revocation, and unsupported states; and
- support owner, update window, and commercial-rights status.

The profile maturity is the minimum maturity of every required component:

| Level | Meaning | Evidence authority |
|---|---|---|
| C0 Candidate | Desired combination or vendor documentation | Directional only |
| C1 Contracted | Schema, golden vectors, and T0/T1 simulation | No physical claim |
| C2 Device-validated | Exact device/build/model passes T2 operator, stage, and parity tests | Backend-specific |
| C3 Interoperable | Exact pair/island/fabric passes T3 generation, faults, and calibration | G2 scope |
| C4 Configuration-certified | Exact fleet/model/workload passes T4 capacity, throughput, headroom, and security | G3/G4 scope |
| C5 Product-supported | T5 fresh install, upgrade/rollback, monitoring, support window, and rights | G5 |

Side states are `blocked`, `unsupported`, `expired`, `deprecated`, and `revoked`.
Reserve “supported” for C5; fixture-only success is `contract_validated`.

The first executable portfolio is the
[three-model by six-platform consumer MoE recipe cohort](consumer-hardware-recipes.md).
It intentionally stops at C1: deterministic locks and offline preflight checks
reduce future hardware time, while physical compatibility, parity,
interoperability, performance, and support claims remain false until their
named gates produce evidence.

Use capability equivalence classes and pairwise/t-wise coverage for routine
variation, while testing high-risk vendor/runtime, model architecture,
quantization, activation/KV, ownership, and transport boundaries exhaustively.
Rank the first small cohort by observed demand, available hardware, substrate
feasibility, reusable learning, model relevance, fabric requirements, and
support burden. Do not procure a broad device zoo before the profile schema and
current G2 mechanism pass.

## Discovery and design-partner evidence lane

Run this lane in parallel with G2. It does not authorize GA work or physical
claims.

### Qualified interview record

Capture, with consent and anonymization as needed:

- organization and decision-maker role;
- privacy, residency, security, and procurement constraints;
- current models, quantization, context, quality bar, and licenses;
- current hardware, fabric, power/cooling, and operational ownership;
- request arrival distribution, concurrency, prompt/output lengths, and burst
  behavior;
- existing workaround and its cost, failure, or capability gap;
- budget owner, buying process, deployment deadline, and support expectations;
- willingness to run a pilot and definition of a successful pilot.

### Provisional discovery exit targets

These are operating targets, not current traction:

- 10 qualified interviews;
- 3 anonymized real traffic/concurrency traces;
- 2 written design-partner commitments; and
- 1 paid pilot before describing product-market pull.

If the evidence misses, record `ITERATE`, `NARROW`, or `KILL`; do not reinterpret
general enthusiasm as purchase intent.

## Pilot contract and scorecard

Every pilot binds one model, fleet, workload, security posture, owner, time box,
and acceptance packet. Report:

- numerical parity and exact greedy-token behavior;
- model capacity and operational memory headroom;
- aggregate throughput, TTFT, inter-token latency, and saturation concurrency;
- predicted versus measured error and component timing;
- installation time, operator hours, incidents, and support hours;
- hardware/fabric/power and deployment-service cost;
- cost per useful output token at the buyer workload;
- buyer acceptance, paid conversion, and renewal/expansion intent.

Thresholds remain contract-specific until real discovery and G2/G3 measurement
support them.

Before any customer-controlled or sensitive data enters a pilot, XG-6 requires
named identity/authentication, encryption, isolation, logging, retention,
incident-response, and data-processing controls. Earlier technical pilots are
restricted to synthetic or explicitly anonymized workloads with that restriction
in the acceptance contract.

## Business-model hypotheses to test

Do not publish pricing before buyer interviews and cost modeling. Test:

- annual per-cluster subscription with certified-node bands;
- enterprise support and update subscription;
- fixed-scope fleet assessment and installation;
- fixed-price certified model/backend enablement where reusable; and
- reference appliance or partner bundle only if support economics improve.

For each option measure sales friction, perceived value, services attach,
delivery/support hours, software gross-margin path, and the risk of deployments
becoming one-off kernel projects.

## Defensibility hypothesis

The moat cannot be “a planner” or founder reputation alone. It must accumulate as:

- calibrated model/fleet/kernel/link performance profiles;
- a cross-vendor Stage ABI and conformance corpus;
- certified model/fleet configurations and reproducible build lineage;
- hard-won MAX/Apple/NVIDIA kernel and graph-partition knowledge;
- placement, failure, and support data from real deployments;
- fast diagnostics and a humane operator path; and
- upstream relationships that reduce fork burden.

Founder roles, time commitments, IP assignment, MAX patch rights, open-source vs
proprietary boundaries, and dependency/model licenses belong in a private data
room. They are not established by this repository.

## Competitive and economics diligence still required

Before an external pitch claims differentiation or savings, publish a dated,
source-backed comparison and benchmark MAX-native/single-node, naive pipeline,
capacity-offload, homogeneous local cluster, relevant open distributed engines,
and cloud inference alternatives on the same accepted workload.

Bind a three-year customer model including hardware, fabric, power/cooling,
deployment labor, support, upgrades, utilization, and refresh. The economic kill
metric should be agreed before frontier-fleet procurement.

The 2026-07-23 AMD–Cerebras announcement is directional category evidence, not
Fornax validation. The companies describe one disaggregated inference workflow
in which AMD Helios handles high-throughput prompt processing and long contexts,
while Cerebras Wafer-Scale Engine technology handles latency-sensitive decode.
The joint implementation details are not public, availability is expected later
in 2026, and the “up to 5x” efficiency figure is vendor-modeled rather than an
independent deployed-system benchmark. The defensible inference is that
capability-aware workload specialization matters; Fornax's mechanism,
performance, economics, and demand remain open gates.

## Claims ledger

| Claim | Status | Allowed pitch language |
|---|---|---|
| Experimental FNX1 v1 and two-worker lockstep loopback | Proven at T0/T1 | “Implemented and contract-tested in reference/simulation; physical/ragged validation may revise the ABI.” |
| Candidate FNX2 ragged oracle, integrated scheduler, and two-worker loopback | Proven at T0/T1 | “Implemented and contract-tested in reference/loopback; physical MAX conformance and throughput remain open.” |
| Scheduler queues, channel credits/RSS, cancellation, faults, and historical 30-minute soak | Proven at recorded T1 scope | “Recorded loopback/simulation evidence; not a fully memory-bounded engine or hardware throughput.” |
| Release, idle expiry, leases/tombstones, and bounded reference runtime retention | Implemented at T0/T1; I-22 evidence remains open | “Mechanism and configured bounds are contract-tested”; do not claim indefinite-service or physical memory stability before restart durability and reviewed sustained physical evidence. |
| Apple MAX runs short DeepSeek V2 Lite generation on patched source build | Recorded T2 bring-up | “Positive single-Mac bring-up; parity and production role remain open.” |
| AMD and Cerebras announced a workload-specialized disaggregated inference workflow | Directional external signal; planned availability and vendor-modeled benefit | “Independent category evidence for heterogeneous workload specialization”; do not imply it proves Fornax or the reported efficiency. |
| Physical NVIDIA→Apple Fornax generation | Open | “Next G2 milestone.” |
| High aggregate physical throughput | Open | Do not claim. |
| Frontier model larger than every node | Open | “G3 objective,” not current capability. |
| Commodity fleet is cheaper per useful token | Open | Do not claim before TCO and baselines. |
| Customer demand / paid pilots | Open | Do not imply traction. |
| Named founding-team participation, roles, or endorsement | Unverified | Do not use names or biographies until signed role, time, IP, conflict, governance, and name-use evidence exists. |
| Patched MAX commercial use, packaging, branding, and redistribution rights | Open | “Commercial-rights diligence is in progress”; do not imply Modular endorsement or cleared redistribution. |
| Product-ready / GA | False today | “Pre-alpha deep-tech product under validation.” |

## Pitch outline

1. Buyer problem and why current single-node/homogeneous options fail the chosen
   workload.
2. Sparse-MoE capacity/compute insight and the honest network constraint.
3. Product architecture and MAX/Fornax ownership boundary.
4. Evidence ladder: what is proven and what is next.
5. Physical G2 demo and component attribution when available.
6. Qualified customer wedge and traffic evidence.
7. Frontier target, baselines, BOM, and TCO after G3 selection.
8. Product plus repeatable service motion.
9. Defensibility and upstream/fork strategy.
10. Team/IP/data-room evidence, milestones, bottoms-up use of funds, and ask.

Do not insert a fundraise amount, valuation, runway, hiring plan, customer logos,
or named-person credentials until the authorized principals supply and approve
those facts.
