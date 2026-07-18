# VC diligence questions and direct answers

These are internal model answers. External use is blocked by
[`external-readiness-gates.md`](external-readiness-gates.md); never fill team or
financing facts from assumptions.

## Thesis and customer

### What exactly are you building?

We aim to build a private inference fabric in which a provisioned mix of NVIDIA
and Apple machines serves one open model through a single endpoint. In the target
architecture, MAX executes physical model stages while Fornax plans placement,
orchestrates requests, transports activations, applies backpressure/failure
rules, and validates named configurations. That integrated physical endpoint is
an open G2/product milestone, not a current capability.

### Who is the first buyer?

A technical, privacy-sensitive SME, mid-market firm, AI integrator, or managed
service provider with mixed compute, a workload too capable or too large for one
available node, and concurrent shared/agentic demand. “All SMEs” is not the target.

### What painful use case are you starting with?

The discovery hypothesis is private document and knowledge workloads—legal,
engineering, industrial, financial, Arabic knowledge, and internal agents—where
data control matters and several jobs run concurrently. Interviews and workload
traces will select the first vertical; no vertical traction is claimed today.

### Why will an SMB operate a cluster?

Many will not. The first buyer has an internal technical owner or buys through an
integrator. The product must eventually make installation and diagnostics
repeatable, but the initial motion includes assessment and deployment service.
If the support burden remains bespoke, the model fails or narrows to an OEM/
integrator product.

### Isn't a smaller quantized model good enough?

Often, yes. Fornax should not be sold where a smaller model meets the accepted
quality, context, and throughput bar on one node. Qualification explicitly tests
this alternative. The opportunity exists only where larger capacity creates
measurable business value.

## Technology and evidence

### What works today?

The Python contract layer runs deterministic golden tests. Experimental FNX1 v1
retains the historical lockstep loopback path; candidate FNX2 adds integrated
ragged scheduling, unequal prefill, independent decode, per-sequence failure/KV,
and two separate loopback workers at T0/T1. Release, idle expiry, internal
leases, same-worker tombstones, and bounded reference state are implemented;
restart-durable fencing and a reviewed physical sustained run remain open. A patched
source-built MAX checkout has run short DeepSeek V2 Lite generation on an M3-class
Mac. The latter is single-node bring-up, not distributed Fornax proof.

### What does not work today?

There is no completed physical NVIDIA-to-Apple Fornax request, no cross-vendor
numerical-parity result, no physical aggregate-throughput claim, no bound
frontier-capacity fleet, and no customer evidence or unit economics.

### Why do you need multiple machines if the mechanism model fits one?

The first physical target separates mechanism correctness from product economics.
DeepSeek V2 Lite is small enough to debug reproducibly while exercising stage
partition, transport, and parity. Only after that passes does the program select
a frontier-capacity model that exceeds every individual node.

### What is technically hardest?

Supportable model-stage execution across vendor backends; exact tensor and KV
ownership; numerical parity; hiding cross-node transport behind useful work;
calibrating placement from real measurements; and sustaining the MAX fork without
depending on fragile internal patches.

### Why MAX/Mojo?

It is the chosen portability layer for graph compilation, kernels, device
execution, and extension across accelerator families. This avoids recreating a
full per-node runtime. It is also a material dependency risk: build support,
Apple capability, internal patches, licensing, and upstream alignment require
explicit gates and a reversal trigger.

### What if MAX does not support the required Apple path?

Apple receives only the role that passes measured parity: excluded, capacity
store, expert worker, complete stage, or broader participant. If public extension
seams cannot support the target, the company can narrow to homogeneous islands,
change Apple's role, or reconsider the substrate. The seed plan treats this as a
decision, not an assumption.

### Why not cross-vendor tensor parallelism?

It would require frequent collectives across mismatched devices and networks.
Fornax uses complete contiguous pipeline stages as the default spanning spine so
cross-node communication occurs at bounded stage boundaries.

### What throughput can you promise?

None on physical heterogeneous hardware yet. The design optimizes aggregate
throughput under concurrency; it does not promise single-stream latency parity.
G2 measures mechanism timing, and G3 measures the frontier target against
baselines at a customer-supplied concurrency distribution.

### Your H100s and Mac are on separate networks. Can you test now?

Not the gate-of-record. WAN federation is out of v0 scope and would add
uncontrolled variables. The immediate plan is to bring the Mac to the H100 site,
or secure a local NVIDIA node, and run both on one isolated measured LAN.

## Competition and moat

### Why doesn't vLLM solve this?

vLLM is a strong baseline for distributed serving on supported GPU environments.
Fornax aims at a different boundary: one model across accelerator families,
with hardware-aware planning and cross-vendor contracts, and targets validated
private deployments if the physical and product gates pass. If vLLM or another
incumbent adds equivalent support and operations,
that reduces the differentiation.

### Is exo already doing this?

exo is the closest thesis competitor and validates demand for distributed local
AI. Fornax's intended distinction is MAX-native node execution, a business-grade
Stage/wire contract, evidence governance, measured placement, and certified
private configurations. That distinction must be shown by benchmarks and operator
experience, not asserted from repository descriptions.

### What is the moat?

The moat must accumulate: measured kernel/model/device/link profiles, the
cross-vendor conformance corpus, certified configurations, reproducible build
lineage, deployment/failure data, diagnostics, and upstream expertise. The
planner or open-source code alone is not a durable moat.

### What will be open source?

Proposed answer: keep the contract/reference layer and interoperability tools
open to earn trust and contributors; monetize certified runtime builds,
configuration profiles, security/operations, enterprise support, and model/backend
enablement. Final boundaries require founder, investor, dependency-license, and
IP counsel approval.

## Market and business model

### How large is the market?

No dataset counts the exact qualification intersection. A transparent UAE model
starts with the Ministry's 1.33 million SMEs and assumes only 0.5%-2% qualify, at
AED 30,000-120,000 annual value. That yields AED 199.5 million to AED 3.192
billion, with an AED 798 million base case. These are sensitivity assumptions,
not demand evidence. A capacity-based year-three plan is 20-60 customers and AED
1.2-6.0 million ARR.

### Why start in the UAE?

Abu Dhabi is the proposed operating base. The UAE has a large SME base, active AI
policy and investment, data-protection requirements, local open-model activity,
and a dense government/corporate startup network. It is a discovery and proving
ground hypothesis, not the ceiling or validated demand: the technical product
targets global private AI deployments.

### How will you make money?

Fleet assessments and paid pilots lead to annual per-cluster runtime/support
subscriptions. Reusable model/backend enablement can be separately priced.
Pricing is unvalidated; interviews test willingness to pay and delivery cost
before a public price is set.

### How do you avoid becoming a consultancy?

Every engagement must produce reusable software, a profile, conformance fixture,
installer, diagnostic, or certified configuration. Track support hours and gross
margin per configuration. Reject or explicitly premium-price work without a
reuse or upstream path.

### Is local necessarily cheaper than cloud?

No. It depends on utilization, model, quality, power, networking, support, and
hardware already owned. The pitch is control and extracting value from a qualified
fleet; savings are an open G3 TCO claim.

## Team, fundraising, and risk

### Who is the founding team, and why can it execute?

Not yet an approved external answer. XG-1 requires signed roles, full-time
commitments, contribution history, name/biography permission, equity/governance,
IP assignment, and employer/conflict clearance. The repository cannot establish
those personal facts, and a simulated perspective review is not participation or
endorsement.

### How will the proposed team cover execution risk?

The architecture and gates reduce, but do not remove, key-person and integration
risk. The functional plan adds two senior runtime/distributed-systems engineers,
an accelerator/MAX engineer, and a solutions engineer by the pilot phase. Actual
principal and hiring commitments remain unverified until XG-1/XG-3. Contractors
may fill security, legal, and specialized kernel gaps initially.

### Why USD 1.5 million?

It is an illustrative 18-month allocation for physical feasibility, customer
demand, frontier economics, repeatability, hardware access, pilots, and
security/IP work. It is not an approved ask until a monthly cash model, hiring
dates, loaded costs, BOM/quotes, existing cash/burn, and principal constraints
pass XG-3.

### What does this round de-risk for the next investor?

Technical feasibility across NVIDIA and Apple, an accepted buyer workload and
concurrency distribution, willingness to pay, a measured frontier-capacity
configuration and TCO, support effort, and a repeatable deployment path.

### What are the top risks?

1. Cross-vendor performance is not economically useful.
2. Qualified SMBs cannot supply enough concurrency.
3. Smaller models or new high-memory appliances erase the need to span.
4. MAX/Apple support or fork maintenance is unsustainable.
5. Sales and support become bespoke services.
6. A better-funded open-source or incumbent serving stack adds the capability.
7. Unresolved founding-team, key-person, hiring, and execution risk.

Each has a dated test and a narrow/kill outcome in the evidence plan.

### What is the exit or strategic value?

Do not lead with an exit. If the technology and customer corpus become real,
strategic counterparties could include inference runtimes, accelerator vendors,
private-cloud platforms, OEMs, systems integrators, and sovereign AI operators.
The venture case should rest on recurring software/support revenue and a global
category, not acquisition speculation.

## Questions the authorized principals must answer before external use

- What is the legal entity, cap table, and IP ownership?
- Who is full-time, under what signed role, and what is the available runway?
- What exact H100 access rights and Mac specification do you have?
- What did each principal personally implement, and what external contributions exist?
- Which five customers can you interview this month?
- Which amount, dilution range, and salary are acceptable?
- What is your open-source/proprietary boundary?
- Which investor or program eligibility restrictions apply to your nationality,
  residency, or affiliation?
