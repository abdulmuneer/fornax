# Fornax external-pitch readiness gates

Status: internal control document  
As of: 2026-07-17  
Decision: `ITERATE` before external circulation

This checklist prevents a technically credible program from losing diligence on
team, rights, financing, customer, or evidence claims. Passing a technical test
does not waive a corporate or commercial gate.

## Current disposition

| Gate | Required evidence | Current state | External effect |
|---|---|---|---|
| XG-1 team and provenance | Principal-approved roles/titles, full-time commitments, contribution history, name/biography permission, equity/governance, IP assignment, employer/conflict clearance | **Blocked** | Do not name unverified founders, reviewers, advisers, or endorsers |
| XG-2 MAX commercial rights | Counsel-reviewed license/redistribution memo; supported-device analysis; patched-build packaging design; NOTICE/attribution; written Modular clarification where required | **Blocked** | Describe MAX as the preferred dependency, not an approved distributable product substrate |
| XG-3 financing | Monthly 18-month cash model, hiring dates and loaded costs, hardware BOM/quotes, existing cash/burn, approved salary/runway/ask | **Blocked** | Any USD 1.5M/18-month case is illustrative, not an approved ask |
| XG-4 technical mechanism | Repeated physical NVIDIA-to-Apple request, parity, component timings, physical validation of release/expiry/lease/tombstone behavior, bounded run, planner error, Apple role, and current-authority evidence | **Blocked at G2; lifecycle and planner authority mechanisms exist only at T0/T1** | Pitch as the financed proof milestone; do not claim heterogeneous serving or production memory stability |
| XG-5 customer wedge | Qualified interviews, anonymized traffic traces, selected ICP/workflow, design-partner commitments, priced proposal, paid pilot | **Blocked** | Do not claim traction, pricing, PMF, or pipeline-filling demand |
| XG-6 pilot security | Named identity/auth/encryption/isolation/logging/retention/incident/DPA controls, or synthetic-only pilot restriction | **Blocked** | No sensitive customer data in an early pilot |
| XG-7 market and competition | Account-based ICP denominator, observed qualification/pricing funnel, dated competitor matrix and benchmark | **Blocked** | Keep MSME TAM as appendix sensitivity; do not lead with USD 10.8B |
| XG-8 package durability | Approved tracked artifacts, no placeholders, current evidence counts, disclosure owner/status/date | **Blocked** | Current package is an internal working draft |

## Named-person claims rule

The current repository history and corporate records available here do not prove
participation by any proposed cofounder, reviewer, adviser, or public figure.
User-supplied intent is not external diligence evidence. Before a name appears in
a deck, data room, website, press statement, or investor email, XG-1 requires:

- signed role, title, time commitment, and authority;
- an accurate separation of past contribution from future responsibility;
- IP assignment and existing-employer/conflict clearance;
- approved biography and permission to use the person's name; and
- a cap-table/governance record consistent with the pitch.

Until then, use functional roles such as `technical lead`, `runtime lead`, and
`product lead`. Never convert a simulated perspective review into endorsement.

## MAX rights checkpoint

The [Modular Community License](https://www.modular.com/legal/community), last
modified 2026-03-20, governs SDK use and permitted redistribution separately
from any source-code license. It includes production/device, distribution,
notice, branding, and supported-hardware conditions. The
[trademark policy](https://www.modular.com/legal/trademark) requires attribution
and prohibits implying endorsement. This document is a diligence flag, not legal
advice.

Before XG-2 passes, record:

- the exact SDK/source components Fornax builds, modifies, bundles, or asks a
  customer to install;
- the applicable license and redistribution status of each component;
- whether every target accelerator is expressly supported;
- production notification, display, attribution, telemetry/privacy, and device
  capacity implications;
- customer support, update, warranty, and indemnity allocation; and
- written clarification or commercial terms for any ambiguous patched runtime.

## Release rule

The external package may be released only when every still-blocking gate has an
owner-approved disposition: `PASS`, or an explicit bounded disclosure accepted
by the Sponsor and counsel where applicable. `ITERATE` is the current outcome.

## Allowed pitch today

> Fornax is a pre-alpha, MAX-oriented heterogeneous inference program with a
> contract-tested Python reference engine and an open G2 physical proof. It is
> seeking evidence partners for same-LAN validation and qualified customer
> discovery. Team, commercial-rights, financing, traction, performance, and GA
> claims remain subject to their named gates.
