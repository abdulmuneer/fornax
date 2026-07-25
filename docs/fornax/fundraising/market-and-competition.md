# Market, positioning, and competitive research

Prepared: 2026-07-25

Confidence: directional; customer demand and pricing remain unvalidated

This file is tracked in a public repository. Confidential customer, partner,
topology, pricing, and unannounced product information belongs outside this
repository.

The full MENA, Indian-subcontinent, and global sensitivity model is maintained in
[`tam-mena-india-global.md`](tam-mena-india-global.md). Its numerical cases are
illustrative appendix sensitivities derived from assumed qualification rates and
prices. They are not validated TAM/SAM claims and must not lead an external pitch.

## Market definition

Fornax does not address the whole AI software market or every SME. The relevant
revenue pool is organizations that simultaneously have:

1. a privacy, security, sovereignty, offline, or control reason to run locally;
2. a useful open-model workload that exceeds one available node or benefits from
   fleet-wide capacity;
3. mixed existing compute or a constrained hardware acquisition path;
4. enough concurrent demand to fill a multi-stage pipeline; and
5. an operator or service partner capable of running a provisioned local fabric.

This intersection is the **qualified private heterogeneous inference market**.
No authoritative dataset currently counts it, so a bottom-up sensitivity model
is more honest than quoting a generic edge-AI market forecast.

## UAE anchor facts

- The UAE Ministry of Economy and Tourism reported in January 2026 that the UAE
  has **1.33 million SMEs**, roughly **95% of all companies**, more than **85% of
  private-sector jobs**, and **63% of GDP**.
- The federal Personal Data Protection Law applies controls to electronic
  processing inside and outside the country and establishes company obligations
  to secure personal data and protect confidentiality and privacy.
- The UAE AI strategy and AI charter support private-sector adoption while
  emphasizing responsible use, privacy, data security, transparency, and
  accountability.
- Hub71's Access Programme targets pre-seed to Series A startups and currently
  advertises AED 250,000 of flexible incentives plus AED 250,000 cash via SAFE,
  with a possible AED 250,000 high-performer top-up.

These facts establish a meaningful local discovery base and supportive context.
They do not establish that 1.33 million firms need Fornax.

## Bottom-up UAE sizing

### Formula

`qualified organizations × annual recurring revenue per organization`

| Scenario | Share of UAE SMEs meeting all qualification tests | Qualified organizations | Provisional annual software/support value | Revenue pool |
|---|---:|---:|---:|---:|
| Conservative | 0.5% | 6,650 | AED 30,000 | AED 199.5M |
| Base | 1.0% | 13,300 | AED 60,000 | AED 798.0M |
| Upside | 2.0% | 26,600 | AED 120,000 | AED 3.192B |

All percentages and prices above are explicit assumptions, not survey results.
The base case is about USD 217 million at the AED 3.6725/USD peg. The range is
intentionally wide because the qualification rate and willingness to pay are
unknown.

### Reachable beachhead and SOM

Use Abu Dhabi/UAE AI integrators and privacy-sensitive technical firms as the
first sales network. Without an authoritative emirate-level count of qualified
buyers, planning assumes 20% of the UAE qualified pool is practically reachable
from Abu Dhabi. In the base case this is 2,660 organizations and AED 159.6
million of annual revenue pool.

The operational three-year SOM should be capacity-based, not a percentage of a
large TAM:

| Case | Paying clusters/customers in year 3 | Average ARR | Year-3 ARR |
|---|---:|---:|---:|
| Low | 20 | AED 60,000 | AED 1.2M |
| Base | 40 | AED 80,000 | AED 3.2M |
| High | 60 | AED 100,000 | AED 6.0M |

This SOM is constrained by a small team, enterprise pilot cycles, certified
configuration work, and support load. It becomes credible only after pricing
interviews and paid pilots.

## Sensitivity and what matters most

The model is most sensitive to:

- the fraction of firms whose workload truly requires more than one node;
- the concurrency supplied by real shared/agentic traffic;
- whether a local model meets the buyer's quality bar;
- annual willingness to pay relative to cloud, a new server, or a smaller model;
- installation and ongoing support hours; and
- the number of reusable configurations versus one-off engineering projects.

The fastest confidence improvements are 10 qualified interviews, three traffic
traces, two priced pilot proposals, and one paid acceptance. Published market
reports cannot replace those facts.

## Competitive map

| Alternative | Strength | Where it wins | Fornax opening | Risk to thesis |
|---|---|---|---|---|
| Cloud model APIs | Fastest start, leading models, elastic demand | Low-friction workloads without strict control needs | Local execution, open weights, offline/control boundary | Cloud data controls and falling prices may satisfy most buyers |
| Single large local node | Simple, low network latency | Model fits; modest concurrency | Models/fleets beyond one node; reuse mixed assets | New high-memory systems may make spanning unnecessary |
| vLLM/SGLang/TensorRT-LLM | Mature high-throughput serving on GPU clusters | Homogeneous NVIDIA/AMD islands | Cross-vendor orchestration and certification | They may extend heterogeneous support or remain sufficient via separate islands |
| Ollama/llama.cpp/MLX/LM Studio | Excellent local developer and desktop experience | One machine and smaller models | Multi-node business service, evidence, operations | Model compression may keep most SMB use cases single-node |
| exo | Topology-aware automatic partitioning, pipeline/tensor sharding, Thunderbolt RDMA, multiple APIs, dashboard, and benchmarks | Current mixed-device local/distributed deployments; Linux GPU support is described as under development | Intended MAX-native cross-vendor execution plus certified enterprise operations, if physically proven | Closest thesis competitor; the proposed Fornax distinction is unproven and exo could mature faster |
| Petals | Demonstrated distributed model blocks and private swarms | Collaborative or research inference | Provisioned LAN, business control, support and evidence | Validates concept but also supplies reusable algorithms |
| Vendor-paired disaggregated inference | Co-designed phase-specific acceleration and one integrated workflow | Datacenter/cloud workloads that justify specialized pools | Vendor-neutral owned-fleet composition, only if physically and economically proven | Vendors may close the valuable seams through optimized bilateral integrations |
| Hardware/appliance vendors | Integrated support and predictable deployment | Buyers willing to purchase a certified box | Reuse sunk assets and mix vendors | Appliance price/performance and support may dominate |
| Systems integrator/custom build | Buyer-specific delivery and relationships | One-off regulated deployments | Repeatable engine and certification corpus | Integrators can own the customer and commoditize Fornax |

Official vLLM documentation supports distributed tensor and pipeline parallelism,
but does not document a production cross-vendor pool. At repository commit
[`b5375f8`](https://github.com/exo-explore/exo/commit/b5375f8cee4368d09e1ce96a56b9f81fb0bc81aa)
(checked 2026-07-17), exo advertises materially more than an enthusiast demo:
topology-aware placement, pipeline and tensor parallelism, Thunderbolt RDMA,
several API surfaces, a dashboard, and benchmark tooling. Its README describes
Linux GPU support as under development. Petals supports collaborative
distributed inference and notes that public-swarm data is processed by other
participants; private swarms are possible. Fornax's intended distinction—one
physically proven MAX-native cross-vendor fabric with enterprise certification
and support—remains a hypothesis until G2/G3. Re-pin and recheck this comparison
before each external use.

On 2026-07-23, AMD and Cerebras announced a planned disaggregated inference
workflow: AMD Helios for high-throughput prompt processing and long contexts,
with Cerebras Wafer-Scale Engine technology for low-latency decode/token
generation. This is meaningful directional evidence that capability-aware
heterogeneous specialization is commercially relevant. It does not demonstrate
Fornax's layer-stage/capacity-pooling mechanism or commodity-fabric economics.
The combined topology, KV/state handoff, security, failure behavior, and
absolute results were not disclosed; expected availability is in the second
half of 2026, and the advertised “up to 5x” tokens-per-second-per-watt benefit is
a vendor model rather than an independent production benchmark.

## Positioning

Recommended category: **private heterogeneous inference fabric**.

Recommended one-liner:

> Fornax aims to turn mixed accelerators a business controls into one private AI
> service for open models too large for any individual machine; the physical
> cross-vendor mechanism remains an open proof milestone.

Avoid:

- “AI for every SME” — the qualification intersection is narrow;
- “cheaper than cloud” — no workload-matched TCO evidence exists;
- “combines any device” — the supported matrix will be earned configuration by
  configuration;
- “frontier performance” — aggregate throughput is the goal, not proven fact;
- “data never leaves” without defining the deployment and control boundary; and
- “no new hardware required” — the customer may need networking, storage, or a
  missing compute/memory role.

## Current social and practitioner signal

The last-30-days collector found current X discussion around local/private AI but
timed out before producing a stable, citable report; Reddit, YouTube, HN, and
Polymarket returned no usable items in that run. Accordingly, this document does
not use social counts as market evidence. Recent public repository activity and
discussion around exo, vLLM, local LLM tools, and heterogeneous inference show
technical interest, but customer demand must be established directly.

## Sources

Primary and official sources should control external claims:

- [UAE Ministry of Economy and Tourism: 2026 National Forum for SMEs](https://www.moet.gov.ae/en/-/second-edition-of-the-national-forum-for-smes-government-procurement-2026-offers-government-contracts-and-tenders-worth-aed-2.445-billion)
- [UAE Government: data protection laws](https://u.ae/en/about-the-uae/digital-uae/data/data-protection-laws.)
- [UAE National Strategy for Artificial Intelligence 2031](https://ai.gov.ae/wp-content/uploads/2023/05/AI-Report-EN-v4.pdf)
- [Hub71 Access Programme](https://www.hub71.com/program/access-programme)
- [MBZUAI Incubation and Entrepreneurship Center](https://mbzuai.ac.ae/iec/)
- [startAD](https://startad.ae/)
- [Cisco AI Readiness Index 2025 infrastructure focus](https://www.cisco.com/c/dam/m/en_us/solutions/ai/readiness-index/2025-m12/documents/Cisco-AI-Readiness-Index_Infrastructure-Focus.pdf)
- [Deloitte: challenges in AI data integrity](https://www.deloitte.com/us/en/insights/topics/digital-transformation/data-integrity-in-ai-engineering.html)
- [UK Business Data Survey 2026](https://www.gov.uk/government/statistics/uk-business-data-survey-2026/uk-business-data-survey-2026)
- [vLLM distributed serving documentation](https://docs.vllm.ai/en/v0.8.4/serving/distributed_serving.html)
- [exo repository, pinned comparison commit](https://github.com/exo-explore/exo/tree/b5375f8cee4368d09e1ce96a56b9f81fb0bc81aa)
- [Petals repository](https://github.com/bigscience-workshop/petals)
- [Helix heterogeneous inference paper](https://arxiv.org/abs/2406.01566)
- [AMD/Cerebras disaggregated inference announcement, 2026-07-23](https://investors.cerebras.ai/news-releases/news-release-details/amd-and-cerebras-announce-industry-leading-ultra-low-latency-and)
- [AMD Helios open rack description](https://www.amd.com/en/blogs/2025/amd-helios-ai-rack-built-on-metas-2025-ocp-design.html)
