# Fornax target: private AI across validated compute you control

Internal seed-financing scenario; not approved for external circulation  
Proposed operating base: Abu Dhabi, United Arab Emirates  
Stage: pre-alpha / technical validation  
Illustrative scenario: USD 1.5 million for 18 months, pending XG-3

> Release is blocked by the team, commercial-rights, financing, technical,
> customer, security, market, and package controls in
> [`external-readiness-gates.md`](external-readiness-gates.md). In particular,
> this draft does not establish participation or endorsement by any proposed
> founder, reviewer, adviser, or public figure.

## Executive Summary

**Businesses want useful AI without surrendering sensitive work or buying a
datacenter.** Many small and mid-sized firms have fragmented compute—an NVIDIA
server, workstations, and Apple Silicon machines—but cannot combine it to serve
one capable open model. Existing local tools usually optimize one machine or one
hardware family. Cloud APIs are easy, but they may be unacceptable for sensitive,
regulated, offline, or sovereignty-driven workflows.

**Fornax aims to become a private inference fabric for that fragmented fleet.**
The target architecture would use Modular MAX for physical model execution on
qualified nodes and Fornax for the cross-node layer: hardware-aware planning,
stage orchestration, versioned tensor transport, admission/batching, failure
semantics, and evidence-led validation. Today only the planner, reference and
simulated mechanism path, and single-node MAX bring-up are evidenced.
The technical goal is to make mixed NVIDIA and Apple machines serve one sparse
mixture-of-experts model larger than any individual node can hold, optimized for
aggregate throughput under concurrent business workloads.

**The modeled round is a proof-and-design-partner financing case, not a finished
product launch.** The repository already contains a contract-tested two-worker engine,
experimental versioned Stage and wire contracts, simulated bounded queues and
failure injection, a
planner, golden vectors, and a patched single-Mac MAX bring-up for DeepSeek V2
Lite. It does not yet contain a physical NVIDIA-to-Apple Fornax request,
frontier-capacity economics, or customer traction. The round converts those open
claims into evidence.

**Abu Dhabi is a strategically strong base.** The UAE Ministry of Economy and
Tourism reports 1.33 million SMEs, representing about 95% of companies, more than
85% of private-sector jobs, and 63% of GDP. The UAE also combines an explicit AI
strategy, personal-data protection obligations, open-model activity, sovereign
AI investment, and startup programs such as Hub71. Fornax can begin with UAE
privacy-sensitive firms and AI solution providers, then—if physical and customer
evidence closes—offer validated private-inference configurations globally.

## 1. Problem

Local AI today forces a bad choice:

- use a small model that fits one available machine but misses the required
  quality or context;
- buy a homogeneous high-end GPU server that is scarce, expensive, and often
  underutilized;
- send sensitive workloads to a cloud provider; or
- assemble experimental open-source components without a supported performance,
  security, or correctness envelope.

This problem is strongest where an organization has sensitive documents,
customer records, source code, industrial knowledge, Arabic/private corpora, or
intermittent connectivity; already owns mixed compute; and can generate several
concurrent AI jobs through staff or agents.

The UAE regulatory and policy context reinforces the need for control. Federal
Decree-Law No. 45 of 2021 establishes obligations around personal-data security,
confidentiality, and cross-border processing. Local execution does not itself
create compliance, but it can reduce data movement and give the operator a
clearer control boundary.

## 2. Solution

The target product would present one private inference endpoint over a
provisioned local fleet. There is no bundled physical text generator or serving
endpoint today. Its intended responsibilities are:

1. inventory the available machines, accelerators, memory, runtime support, and
   network links;
2. determine whether a chosen model and workload can fit with operational
   headroom;
3. partition complete model stages across heterogeneous nodes using measured,
   not advertised, performance;
4. execute and transport activations through a versioned, bounded protocol;
5. keep enough requests in flight to use the pipeline efficiently;
6. validate numerical behavior at vendor boundaries; and
7. expose honest telemetry, diagnostics, and a supported configuration matrix.

The initial architecture uses contiguous pipeline stages. It avoids
cross-vendor tensor-parallel collectives and does not depend on remote expert
all-to-all traffic. Remote expert execution remains an optional measured
optimization.

## 3. Why now

- Open sparse-MoE models separate total model capacity from active computation,
  creating a plausible systems seam for heterogeneous placement.
- Businesses are moving from occasional chat to concurrent agents and shared AI
  services, which better match pipeline throughput economics.
- Privacy, security, and governance concerns remain material barriers to AI
  deployment; on-premises and private-cloud storage continue to coexist with
  public cloud rather than disappearing.
- Hardware diversity is increasing across NVIDIA, AMD, and Apple, while most
  production serving stacks remain optimized for homogeneous accelerator
  islands.
- Abu Dhabi is investing heavily in AI infrastructure and startup formation,
  while the UAE's large SME base supplies local discovery and pilot opportunities.

## 4. Initial customer and use cases

The first customer is **not every small business**. The qualified beachhead is a
technical SME, mid-market firm, managed AI provider, or systems integrator that:

- must keep model execution and sensitive data under its control;
- already owns useful compute or faces a constrained hardware budget;
- needs a model or workload that does not fit comfortably on one available node;
- has shared or agentic demand capable of sustaining concurrent requests;
- can provision a measured local network; and
- will pay for assessment, deployment, certification, and support.

Priority discovery verticals in the UAE are legal and professional services,
health and life-science suppliers, engineering and construction, industrial
operations, financial and insurance service providers, Arabic knowledge
services, and local AI integrators. These are hypotheses to test, not claimed
traction.

Representative workflows include private document intelligence, contract and
tender analysis, engineering knowledge assistants, internal coding agents,
multilingual/Arabic retrieval and drafting, and high-volume back-office agents.

## 5. Product and revenue model

The commercial entry point is a productized service that generates reusable
product assets:

1. **Fleet and workload assessment** — fixed-scope feasibility, security, and
   economics report.
2. **Paid pilot** — one model, fleet, workload, time box, and acceptance packet.
3. **Annual runtime and support subscription** — certified model/fleet profiles,
   updates, monitoring, and support.
4. **Model or backend enablement** — fixed-price work only where it creates a
   reusable certification or upstream contribution.

Pricing must be discovered with buyers. The market model uses a provisional AED
30,000-120,000 annual software-and-support range solely for sensitivity testing.
The business must avoid becoming an unlimited bespoke-kernel consultancy: every
engagement should add a reusable profile, conformance test, installer, diagnostic,
or certified configuration.

## 6. Competition and differentiation

The real alternatives are cloud APIs, a single large local node, homogeneous GPU
clusters using vLLM/SGLang/TensorRT-LLM, desktop tools such as Ollama/llama.cpp/
MLX, and experimental distributed systems such as exo and Petals.

Fornax's intended differentiation is the combination of:

- a single model spanning different accelerator families;
- throughput-aware placement using measured machine and link profiles;
- production-oriented contracts, backpressure, failure semantics, and evidence;
- sparse-MoE-aware stage and future expert placement;
- a portable MAX/Mojo node substrate; and
- certified configurations and operator support for private business deployments.

This is a hypothesis until physical benchmarks establish an advantage. The moat
is not the planner alone. It must become the accumulated performance corpus,
cross-vendor conformance suite, certified configurations, build lineage,
deployment data, diagnostics, and kernel/graph knowledge.

## 7. Evidence today

Proven or recorded:

- deterministic Python contract and simulation layer with golden vectors;
- two independent loopback worker processes using experimental FNX1 v1 framing
  under a lockstep orchestrator;
- framed TCP transport, credits, cancellation, fault injection, and evidence
  ledgers at T0/T1, with bounded admission/continuous batching tested separately;
- reference and simulated MAX backends;
- a planner and regression tests; and
- short DeepSeek-V2-Lite-Chat generation on an M3-class Mac using a patched
  source-built MAX checkout, recorded as bring-up rather than production proof.

Not proven:

- physical NVIDIA-to-Apple distributed generation;
- numerical parity across that boundary;
- aggregate throughput or latency on heterogeneous hardware;
- a frontier model larger than every individual node;
- indefinite-service memory boundedness: T0/T1 release, expiry, leases,
  same-worker tombstones, and configured reference-retention bounds exist, but
  restart durability plus reviewed long-running physical evidence do not;
- cost advantage over cloud or a homogeneous server;
- customer demand, paid pilots, pricing, or retention; and
- production-ready security and operations.

## 8. Eighteen-month milestones

| Period | Milestone | Investor-grade evidence |
|---|---|---|
| 0-3 months | G2 mechanism proof | Repeated NVIDIA-to-Apple request on one LAN; activation/logit parity; component timings; planner error within ±20%; recorded Apple role decision |
| 0-4 months | Discovery proof | 10 qualified interviews, three anonymized concurrency traces, ranked wedge, explicit go/narrow decision |
| 4-8 months | Design-partner proof | Two signed design-partner commitments; security and pilot contracts; first paid pilot launched |
| 6-12 months | G3 decision | Model larger than one node, full fleet fit with headroom, baseline comparison, throughput/concurrency curve, three-year TCO and kill metric |
| 10-15 months | Repeatable pilot | Installer/runbook, supported matrix, monitoring, fresh-operator acceptance, second pilot |
| 15-18 months | Seed exit | One paid reference customer or two completed paid pilots, credible renewal path, repeatable configuration, next-round evidence package |

Each gate can result in `PROCEED`, `ITERATE`, `NARROW`, or `KILL`. If customers
cannot provide pipeline-filling concurrency, the company will narrow toward
capacity-first serving, homogeneous islands, or an integrator tool rather than
misrepresent throughput.

## 9. Illustrative use of funds

Scenario only; replace with a monthly cash model, hiring dates, loaded costs,
hardware BOM/quotes, current cash/burn, and principal approval before external
use:

| Category | Share | USD | Purpose |
|---|---:|---:|---|
| Engineering team | 45% | 675,000 | Principal compensation plus 3-4 systems/runtime hires and selective specialist contracts; exact sequence and loaded costs remain open |
| Hardware and local lab | 20% | 300,000 | Same-LAN NVIDIA/Apple validation, networking, storage, power, spares, and frontier-capacity access |
| Customer discovery and pilots | 12% | 180,000 | Solutions engineering, travel, pilot deployment, partner enablement |
| Security, legal, IP, and compliance | 10% | 150,000 | Company/IP cleanup, licenses, contracts, security architecture and review |
| Cloud/compute and developer infrastructure | 8% | 120,000 | CI, benchmark bursts, artifact hosting, collaboration and observability |
| Contingency | 5% | 75,000 | Hardware, hiring, and schedule variance |
| **Total** | **100%** | **1,500,000** | **18-month proof and design-partner runway** |

The functional coverage scenario is a technical lead, two senior distributed or
runtime engineers, one Mojo/MAX or accelerator engineer, and a customer-facing
solutions engineer by the pilot phase. Names, titles, founder status, time
commitments, contribution history, IP, and governance are withheld until XG-1.
Security, finance, recruiting, and legal can begin as fractional or contracted
functions. Hiring sequence should follow G2 rather than front-load a large team
before feasibility evidence.

## 10. Why Abu Dhabi and what Fornax gives back

Fornax can be built from Abu Dhabi as globally relevant infrastructure with a
local proving ground. It aligns with the UAE's AI strategy, responsible-use and
data-security priorities, and desire to increase SME productivity. Local
ecosystem value includes:

- private AI infrastructure for companies that cannot send all data to external
  services;
- reusable deployment knowledge for Arabic and regional models;
- systems and accelerator engineering roles in Abu Dhabi;
- collaboration opportunities with local AI builders, universities, integrators,
  and government-related industry; and
- a globally exportable software product anchored in the UAE.

The immediate ecosystem targets are Hub71 for company-building, incentives, and
introductions; MBZUAI's entrepreneurship ecosystem for technical and founder
network access where eligible; startAD for industry discovery; and local
integrators for design-partner sourcing. Eligibility must be verified directly;
some grants are restricted to students, affiliated founders, or Emirati founders.

## 11. The modeled financing outcome

If the bottoms-up model validates this scenario, Fornax would seek capital and
partners that can provide more than financing:

- access to a same-site NVIDIA and Apple validation environment;
- introductions to 10 qualified UAE design-partner candidates;
- deep systems, inference, security, and enterprise-sales advisors;
- help recruiting the first runtime and solutions engineers; and
- disciplined milestone governance through G2, design-partner, and G3 decisions.

The round's purpose is simple: prove that heterogeneous private inference works,
prove that a qualified buyer needs it, and turn both proofs into a repeatable
product before scaling spend.
