# Fornax addressable market: MENA, Indian subcontinent, and global

Prepared: 2026-07-13  
Status: internal appendix sensitivity; not validated market evidence

## Executive Summary

This model shows how large numbers can result from assumed qualification rates
and prices; it does **not** establish an investor-grade market size. No dataset
counts the exact Fornax buyer: an organization that needs private AI, needs more
than one available machine, owns or can provision useful mixed compute, supplies
enough concurrent work, and can operate through an internal team or integrator.

The USD 750 million MENA, USD 1.94 billion Indian-subcontinent, and USD 10.8
billion global values below are scenario outputs, not observed demand. The
USD 1.5-3.0 billion SAM and 250-750 deployment cases are also hypotheses. Keep
them in diligence appendices until XG-7 replaces population multipliers with a
named-account denominator, qualification funnel, observed pricing, channel
capacity, and support constraints.

## What is being sized

This sensitivity blends several potential revenue streams for a **private
heterogeneous inference fabric**:

- runtime or per-cluster subscription;
- enterprise support and certified updates;
- fleet assessment and repeatable deployment;
- reusable model/backend enablement; and
- partner or appliance licensing where applicable.

It excludes GPU and server sales, general AI applications, training, public-cloud
inference, and bespoke consulting without reusable product output.

## The model

The formula is:

`business population × Fornax qualification rate × annual contract value`

The business-population figures are sourced but have different dates and
definitions. The qualification rate and annual contract value are assumptions to
be replaced by discovery, pilots, and pricing evidence. Software ARR, support
ARR, and one-time assessment/deployment/enablement services must be modeled
separately before this becomes a finance or market claim.

| Geography | Addressable business population used | Qualification rate, low / base / high | Annual value, low / base / high | TAM, low / base / high |
|---|---:|---:|---:|---:|
| MENA | 10.0M formal MSMEs | 0.10% / 0.25% / 0.50% | $15k / $30k / $60k | **$150M / $750M / $3.0B** |
| Indian subcontinent | 97.0M MSMEs and micro enterprises | 0.03% / 0.10% / 0.25% | $8k / $20k / $40k | **$233M / $1.94B / $9.70B** |
| Global | 450M formal and informal MSMEs | 0.03% / 0.08% / 0.15% | $15k / $30k / $60k | **$2.03B / $10.8B / $40.5B** |

The global population uses the midpoint of the World Bank/IFC historical
estimate of 420-510 million MSMEs. The global result includes MENA and the Indian
subcontinent and must not be summed with them.

## MENA: approximately USD 750 million base TAM

OECD cites IFC analysis showing more than 10 million formal MSMEs in MENA in
2019, over 90% of formal businesses. Older World Bank/IFC work estimates 19-23
million when informal firms are included. The model deliberately uses the lower
formal count because informal and subsistence firms are unlikely direct buyers.

The 0.25% base qualification rate produces 25,000 potential organizations. That
small share is intended to represent firms or service providers with all the
required characteristics—not merely firms using ChatGPT. At USD 30,000 blended
annual value, the base TAM is USD 750 million.

MENA is internally uneven:

- **GCC:** fewer organizations, but higher infrastructure readiness, contract
  values, sovereignty demand, and channel access;
- **Egypt, Morocco, Jordan, Tunisia, and Lebanon:** larger pools of technical and
  professional firms, but lower direct ACVs and greater need for integrator,
  hosted-private, or appliance distribution; and
- **UAE:** a proposed first beachhead because the Ministry of Economy and Tourism
  reports 1.33 million SMEs and the program is evaluating Abu Dhabi channels.

The initial serviceable MENA market is closer to **USD 150-300 million**, assuming
only 20-40% of the qualified base is reachable through supported countries,
hardware configurations, and partners.

## Indian subcontinent: approximately USD 1.94 billion base TAM

The working model captured 86.1 million Udyam and Udyam Assist registrations in
a June 2026 dashboard snapshot, overwhelmingly classified as micro. Because the
dashboard is live and the count changes, an immutable official snapshot must be
archived before external use. The composition is why a raw “millions of
customers” claim would be misleading in any case.

For a conservative regional population, the model combines:

- India: approximately 86.1 million registrations;
- Bangladesh: more than 7.8 million cottage, micro, small, and medium enterprises;
  and
- Pakistan: an older World Bank-reported figure of 3.2 million SMEs.

This totals about 97 million before Sri Lanka, Nepal, Bhutan, and Maldives; the
country definitions and years are not harmonized, so it is a planning population,
not a statistical regional census.

Applying a 0.10% base qualification rate yields 97,000 potential buyers or
partner-served deployments. At USD 20,000 annual value, the base TAM is USD 1.94
billion. The strongest initial segments are likely:

- software and AI service companies managing private client deployments;
- export-oriented IT/BPO firms with concurrent internal agents;
- financial, healthcare, legal, and industrial firms with data-control needs;
- universities and research organizations with fragmented accelerators; and
- OEMs and systems integrators that can distribute a certified appliance or
  managed private cluster.

The initial serviceable regional market is closer to **USD 200-500 million**.
Direct enterprise selling alone is unlikely to capture it efficiently; channel
and OEM distribution are essential.

## Global illustrative case: USD 10.8 billion output

World Bank/IFC work estimated 420-510 million formal and informal MSMEs globally,
including 35-45 million formal non-agricultural SMEs. The wider population is
used because AI integrators and technical micro-firms can operate Fornax on behalf
of many end customers, but the model applies only a 0.08% base qualification
rate.

That produces 360,000 potential direct or channel-served deployments. At USD
30,000 annual value, the base TAM is USD 10.8 billion.

This is directionally consistent with observed AI adoption being material but
far broader than Fornax's niche: OECD reported that 20.2% of firms in countries
with available 2025 data used AI, compared with 17.4% of small firms and 52.0%
of large firms. Fornax assumes only a tiny fraction of the business population
will both adopt AI and require private multi-node heterogeneous inference.

The model also posits a five-to-seven-year **global SAM of USD 1.5-3.0 billion**
after limiting the denominator to supported models, hardware, countries,
partners, and buyers with sufficient operational maturity. That range currently
has no auditable derivation and must not be used in an operating plan or pitch
until those filters are quantified.

## Sensitivity: what can collapse or expand the market

### The TAM collapses if

- smaller quantized models meet most business quality needs on one machine;
- new high-memory appliances become cheap enough to remove the spanning problem;
- cloud privacy, residency, and security controls satisfy nearly all buyers;
- real SMB traffic cannot fill the pipeline;
- heterogeneous configurations cost too much to support; or
- customers want applications and outcomes but will not pay separately for
  infrastructure.

### The TAM expands if

- multi-agent workflows create sustained concurrent inference demand;
- sparse open models continue to improve faster than single-node affordability;
- sovereign, offline, or customer-controlled inference becomes a procurement
  requirement;
- OEMs and integrators embed Fornax into repeatable private-AI offerings; or
- the runtime supports homogeneous islands and capacity pooling in addition to
  the strict cross-vendor spanning use case.

The qualification rate is the largest uncertainty. Doubling it doubles TAM;
there is no model sophistication that compensates for missing customer evidence.

## External-use rule

Do not lead with the USD 10.8 billion output. If an investor asks about this
appendix before XG-7 closes, use wording such as:

> We do not yet have a validated category-size claim. This appendix stress-tests
> possible outcomes by applying explicit but unvalidated qualification and price
> assumptions to broad business populations. We are replacing those assumptions
> with named-account discovery, workload traces, priced pilots, and channel and
> support-capacity evidence before using TAM or SAM externally.

Do not say:

- “There are 450 million potential customers.”
- “Every SME needs local AI.”
- “The market is USD 40 billion” without presenting it as the high sensitivity
  case.
- “Fornax will capture 1% of TAM.” A capacity-based customer plan is more credible.
- “The serviceable market is USD 1.5-3.0 billion.” That range is not yet derived.

## Validation priorities

1. Conduct 10 MENA and 10 Indian-subcontinent interviews, split between end buyers
   and integrators.
2. Measure how many fail each qualification gate: privacy need, model need,
   hardware, concurrency, operator, and budget.
3. Obtain at least three real concurrency traces in each region.
4. Test annual willingness-to-pay bands of USD 10k, 25k, 50k, and 100k with an
   explicit pilot and support scope.
5. Ask integrators how many downstream clusters one partnership could deploy and
   what gross margin/support burden they require.
6. Replace regional qualification rates and ACVs after the first 20 interviews;
   until then, keep the wide ranges visible.

## Sources and population caveats

- [OECD: more than 10 million formal MSMEs in MENA in 2019](https://www.oecd.org/en/publications/informality-and-structural-transformation-in-egypt-iraq-and-jordan_efb16d0b-en/full-report/component-5.html)
- [World Bank/IFC: 19-23 million formal and informal MSMEs in MENA](https://documents1.worldbank.org/curated/en/581841491392213535/pdf/113701-WP-Overcoming-constraints-IFC-Report-PUBLIC.pdf)
- [India Ministry of MSME dashboard](https://dashboard.msme.gov.in/dashboard.aspx)
- [India Press Information Bureau: 7.83 crore registrations as of February 2026](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2246892&lang=1&reg=3)
- [Bangladesh SME Foundation: more than 7.8 million CMSMEs](https://smef.gov.bd/site/news/c4561dec-bdbf-49e6-a9ac-3b792b4822a6/)
- [World Bank: Pakistan had 3.2 million SMEs](https://www.worldbank.org/en/news/feature/2016/02/08/what-will-it-take-for-pakistan-to-achieve-financial-inclusion)
- [World Bank/IFC: 420-510 million global MSMEs and 35-45 million formal SMEs](https://documents1.worldbank.org/curated/en/804871468140039172/pdf/949110WP0Box380p0Report0FinalLatest.pdf)
- [OECD: firm AI adoption reached 20.2% in 2025](https://www.oecd.org/en/about/news/announcements/2026/01/ai-use-by-individuals-surges-across-the-oecd-as-adoption-by-firms-continues-to-expand.html)
- [UAE Ministry of Economy and Tourism: 1.33 million UAE SMEs](https://www.moet.gov.ae/en/-/second-edition-of-the-national-forum-for-smes-government-procurement-2026-offers-government-contracts-and-tenders-worth-aed-2.445-billion)

Population definitions, years, and formality differ materially. The estimates are
therefore scenario models, not additive census totals. No third-party source
measures the number of firms requiring heterogeneous private inference.
