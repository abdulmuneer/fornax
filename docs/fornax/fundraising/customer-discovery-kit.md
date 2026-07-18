# Customer discovery and design-partner kit

## Interview objective

Determine whether the organization has a costly, urgent private-AI workload that
needs more capability than one available node and enough concurrency to benefit
from Fornax. The interview is not a product demo and should not teach the buyer
to agree with the thesis.

## Target interviewees

- CTO, CIO, CISO, head of data/AI, infrastructure lead, or technical founder;
- managed-service or systems-integration lead serving regulated customers;
- owner/operator of an AI-heavy professional-services firm; and
- budget owner for the workflow, not only an enthusiastic end user.

## Thirty-minute interview script

1. What business workflow are you trying to improve with AI today?
2. What data must the system read, and what are you prohibited or unwilling to
   send to an external model provider?
3. What do you use now? What does it cost, fail to do, or delay?
4. Which model quality, language, context, latency, and reliability requirements
   are non-negotiable?
5. What hardware do you own or control? Capture exact SKU, memory, OS, and
   utilization—not “we have GPUs.”
6. Describe the network between those machines and who operates it.
7. How many users, agents, or jobs run simultaneously? Ask for logs or a one-week
   anonymized trace rather than a recollection.
8. What are typical and p95 prompt/output sizes, burst patterns, and daily volume?
9. Would a slower individual response be acceptable if aggregate private
   throughput and capability increased? Where is the limit?
10. Who approves security, procurement, and budget? What is the timeline?
11. What would a successful 6-8 week pilot prove, in numbers?
12. What would you pay to solve this, and which alternative would that budget
    otherwise buy?

Avoid “Would you use this?” and “Do you care about privacy?” Both invite polite
but non-predictive answers.

## Qualification score

Score each dimension 0-2. A strong design-partner candidate scores at least 12/16
and has no zero in urgency, data/control, or access.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Urgency | exploratory | project this year | funded deadline/problem now |
| Data/control | public/low sensitivity | preference for control | contractual, regulated, offline, or strict policy |
| Model need | small model sufficient | quality gap suspected | accepted workload demonstrably needs larger model/capacity |
| Existing compute | none | one useful node | mixed useful fleet or committed acquisition |
| Network/operator | unmanaged/Wi-Fi | can provision | measured wired fabric and named operator |
| Concurrency | one bursty user | several users/jobs | trace shows sustained pipeline-filling demand |
| Budget/access | no owner | influencer identified | budget owner and pilot authority engaged |
| Reuse potential | unique bespoke stack | partial reuse | representative vertical/configuration |

## Evidence request

With consent and anonymization:

- one-week request timestamps;
- input/output token counts or text-length proxies;
- model and quantization used;
- latency and failure logs;
- hardware and network inventory;
- cloud/API/hardware/support cost; and
- security and data-location requirements.

Never request raw sensitive prompts when timing and size metadata will answer the
concurrency question.

## Design-partner letter outline

The letter should be non-binding unless counsel approves otherwise, but specific:

- named organization and executive sponsor;
- problem/workload and why current alternatives are insufficient;
- fleet and data/control boundary;
- agreed discovery data and security conditions;
- intended pilot window and staff commitment;
- target commercial price range or paid-pilot intent;
- acceptance metrics;
- permission boundaries for logo, quote, and anonymized results; and
- explicit statement that Fornax is pre-production and performance is unproven.

## Paid-pilot scorecard

| Area | Metric |
|---|---|
| Correctness | Boundary activation/logit tolerance and exact greedy tokens where required |
| Capacity | Model fit and operational memory headroom |
| Performance | Aggregate throughput, TTFT, inter-token latency, saturation concurrency |
| Attribution | Compute, packing, transfer, queue, exposed wait |
| Economics | Hardware/fabric/power, operator hours, support hours, useful-output cost |
| Operations | Install time, upgrade, observability, incidents, recovery |
| Security | Identity, authentication, encryption posture, audit trail, data flow |
| Commercial | Buyer acceptance, conversion, renewal/expansion intent |

## Weekly program-owner dashboard

- qualified interviews completed / 10;
- workload traces acquired / 3;
- design-partner letters / 2;
- priced pilot proposals / 2;
- paid pilots / 1;
- median qualification score;
- observed concurrency range and fraction above the Fornax saturation threshold;
- strongest alternative selected by buyers;
- expected ARR and delivery/support hours; and
- decision: `PROCEED`, `ITERATE`, `NARROW`, or `KILL`.
