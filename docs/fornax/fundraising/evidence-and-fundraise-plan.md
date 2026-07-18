# Evidence, lab, and fundraise execution plan

Prepared for the current Abu Dhabi program owner with reported access to H100
machines and an M3 MacBook Pro on separate networks. Founding-team participation,
roles, and authority remain gated by
[`external-readiness-gates.md`](external-readiness-gates.md); this document does
not establish them.

## The immediate constraint

The current machines cannot prove Fornax's G2 claim while they remain on
separate networks. The plan of record explicitly excludes WAN federation, and a
WAN run would confound the mechanism with uncontrolled latency, routing,
security, and bandwidth. It would be useful only as a non-gating transport
experiment.

The exact Mac must also be inventoried. “MacBook Pro M3” is insufficient for a
memory or performance claim: record chip variant (M3, M3 Pro, or M3 Max), unified
memory, macOS, MAX build, power mode, thermal state, and available network
interfaces. Existing documentation refers to an M3 Max-class mechanism target;
do not silently treat a base M3 as equivalent.

## Minimum credible G2 lab

Co-locate one NVIDIA machine and the Mac on the same isolated, measured LAN.

Required minimum:

- one H100 or other supported NVIDIA node with reproducible MAX build;
- the inventoried M3 MacBook Pro;
- a direct or switched wired link—10 GbE is a practical mechanism starting
  point, while 25-100 GbE is more representative for later target economics;
- Thunderbolt/Ethernet adapter as required by the Mac;
- synchronized clocks, fixed power/thermal settings, and component telemetry;
- local model snapshot and reproducible build artifacts;
- no Wi-Fi in the benchmark-of-record path; and
- an isolated-network exception recorded under the current security contract.

Lowest-cost options, in order:

1. take the Mac to the H100 site and attach it to the same isolated network;
2. arrange temporary local access through an Abu Dhabi lab, university, cloud
   bare-metal provider, or design partner;
3. acquire or borrow a supported NVIDIA workstation for mechanism proof; or
4. rent both nodes in one physical facility if Apple bare metal is available.

Do not purchase a frontier fleet before the two-node mechanism proof and target
economics define what hardware is actually needed.

## Six-week evidence sprint

| Week | Outcome | Artifact |
|---|---|---|
| 1 | Exact principal/company and hardware inventory | Principal facts sheet; machine manifests; IP and dependency register |
| 1-2 | Same-LAN access secured | Lab agreement, topology, measured link evidence |
| 2-3 | Physical single-stage baselines | NVIDIA and Apple stage/parity profiles on the same model/build |
| 3-4 | First cross-node request | Request/stage manifests, activation frames, logs, failure record |
| 4-5 | Repeated G2 evidence | Numerical tolerances, timings, planner error, physical Stage Backend API v2 request/KV release, native-retention checks, 30-minute current-authority run |
| 1-6 | Five initial buyer interviews | Qualification records and at least one anonymized traffic trace |
| 6 | Investor update package | Two-minute demo, benchmark card, claims ledger, updated deck |

Research accelerator eligibility and prepare internal materials in parallel. Do
not submit a package whose team, rights, financing, or disclosure claims still
fail the external-readiness gates. When those gates permit a bounded
conversation, describe G2 as the financed/active milestone and update interested
parties as evidence closes.

## Eighteen-month operating plan

### Stage A — prove the mechanism (months 0-3)

- reproduce a physical NVIDIA-to-Apple request;
- validate boundary activations and final logits;
- attribute compute, packing, transfer, queue, and exposed wait;
- calibrate planner error to ±20%; and
- decide Apple's measured role.

Kill/narrow trigger: no plausible path to correct stage execution using
supportable MAX extension seams, or exposed communication dominates at every
customer-realistic concurrency and stage size.

### Stage B — prove a buyer (months 0-8)

- interview 10 qualified organizations;
- obtain three anonymized workload/concurrency traces;
- secure two design-partner letters;
- issue two priced pilot proposals; and
- start one paid pilot only after XG-6 passes, or restrict it contractually to
  synthetic/anonymized data while the production security controls remain open.

Kill/narrow trigger: buyers prefer cloud controls or a single-node appliance,
cannot supply concurrency, or will not pay enough to cover repeatable support.

### Stage C — prove product economics (months 6-12)

- bind the frontier-capacity model and fleet;
- compare against single-node, naive pipeline, capacity-offload, homogeneous
  cluster, and relevant cloud baselines;
- calculate three-year TCO and cost per useful output token;
- measure install/operator/support hours; and
- decide whether remote experts are economically useful.

Kill/narrow trigger: Fornax cannot beat or strategically justify itself against
the best practical alternative for the accepted workload.

### Stage D — prove repeatability (months 10-18)

- package installation and upgrades;
- publish the supported configuration matrix;
- harden authentication, encryption, audit, and operational controls;
- pass a fresh-operator deployment; and
- complete a second paid pilot or convert the first to annual support.

## Investor pipeline

### Tier 1 — ecosystem and non-dilutive leverage

- Hub71 Access Programme: direct match for pre-seed to Series A; prepare the
  required deck and Abu Dhabi plan.
- MBZUAI IEC: pursue mentor, builder, technical, and founder ecosystem access;
  grants described on its site are for MBZUAI founders, so verify eligibility.
- startAD: use industry programs and network for customer discovery and pilot
  introductions.
- Khalifa Fund/MZN: only pursue if nationality and program eligibility match;
  do not assume eligibility.
- Hardware, cloud, and semiconductor credits: seek milestone-tied access instead
  of buying all compute.

### Tier 2 — specialist investors

Target deep-tech, AI infrastructure, developer-infrastructure, sovereign AI,
edge/private AI, and MENA seed investors. Prioritize investors who can provide
hardware access, systems recruiting, enterprise design partners, or follow-on
capital. A generalist investor without technical diligence capacity may interpret
the pre-hardware stage as excessive risk.

### Fundraise sequence

1. Build a list of 40 investors/programs: 10 ecosystem, 15 specialist seed, 10
   strategic/industry, and 5 grants.
2. Seek 10 warm introductions and run meetings in concentrated two-week waves.
3. Lead with the evidence ladder and the funded milestone, not a large generic TAM.
4. Send a weekly update containing technical proof, customer proof, hiring, and
   asks.
5. Track objections verbatim and update the diligence FAQ; do not change claims
   merely to make objections disappear.

## Data-room checklist

### Required before serious diligence

- incorporation, cap table, principal identity and residency facts;
- approved principal CVs, relevant technical history, and full-time commitments;
- IP assignment and contributor agreements;
- root Fornax license and dependency/license inventory;
- MAX fork/upstream base, patch rights, redistribution analysis, and build lineage;
- model and dataset license register;
- current plan, stage gates, RAID log, and claims ledger;
- test summary and physical evidence bundles;
- 18-month budget, hiring plan, and monthly cash model;
- customer interview records, design-partner letters, and pilot contracts;
- security architecture, data flows, and compliance counsel memo; and
- competitive benchmark methodology and raw results.

### Principal inputs still missing from the repository

- legal entity and ownership;
- nationality/program eligibility;
- approved biographies and prior achievements;
- amount personally invested and current monthly burn;
- full-time start date and other commitments;
- exact H100 ownership/access terms and location;
- exact Mac SKU and unified memory;
- desired salary/runway and hiring constraints;
- patent/open-source preference; and
- any customer, partner, or investor conversations already completed.

These omissions should be completed in a private principal facts sheet, not
guessed in public documents.
