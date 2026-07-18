# Planner authority status erratum

Date: 2026-07-17  
Updated: 2026-07-18  
Plan of record: `project-plan-v4.md` (unchanged)

This erratum clarifies implementation status; it does not revise the v4 target
architecture or historical gate records.

Plan v4 says the planner remains advisory until listed correctness and
calibration gaps close, and that uncalibrated placement fails closed. The
hardware-independent I-16 authority mechanism is now implemented; physical
calibration evidence is still absent and cannot be created by this change.

Phase 0.5 closure established at T0/T1:

- remote-expert host resource accounting for the reproduced defect;
- feasibility-preserving search for the reproduced fleet-larger-than-six defect;
- KV-bearing attention rejection on `supports_kv=false`; and
- deterministic planner regressions used by the reference/simulation engine.

Phase 0.5 did not implement or prove:

- exact runtime/build/operation/quantization capability admission;
- measured stage, packing, route, queue, and native-memory profiles;
- measurement source IDs, confidence classes, or prediction-error intervals in
  `PlacementPlan`;
- automatic `exploratory` labeling or fail-closed rejection of unmeasured
  critical operations; or
- physical planner calibration.

The 2026-07-17 I-16 implementation adds:

- structured measurement provenance on node and link inputs, including source
  IDs, confidence, and expected relative error;
- prediction-calibration provenance and emitted numerical error intervals;
- exact target runtime, accepted build, required operation, activation dtype,
  weight-quantization, and role admission against complete node capabilities;
- a separate evidence-registry resolver that content-verifies every referenced
  model, quantization, capability, measurement, calibration, and route artifact
  against its declared SHA-256, evidence type, status, and optional validity
  window;
- automatic `PlacementPlan.authority.status=exploratory` on the backward-
  compatible default path; and
- `authority_mode=deployment`, which returns `feasible=false` and
  `authority.status=rejected` unless all authority checks pass.

CLI callers select the fail-closed path with:

```bash
python3 -m fornax plan ... \
  --authority-mode deployment \
  --evidence-registry planner-evidence-registry.json
```

`authoritative` is accepted as a CLI alias for `deployment`.

Deployment admission requires real source IDs for the model, quantization,
backend capability report, every candidate node/link measurement consumed by
search, and a measured prediction-error calibration within the target bound.
Each ID must resolve through the separate registry to an active, type-correct,
non-stale record whose artifact bytes match its SHA-256. Omitting the registry
rejects deployment even if every declaration says `measured`.

This local resolver closes the arbitrary-string gap; it does not establish the
truth of an artifact's contents. The registry is not a signature or remote
attestation. Physical evidence acquisition, independent review, and registry
governance remain binding.

Current repository fixtures, quickstart inventories, simulations, and static
hardware estimates lack that physical calibration. They therefore continue to
emit **exploratory** plans by default. Merely copying `status=measured` into a
fixture is not G2 evidence and does not authorize an external claim.

I-16's hardware-independent schema, registry resolution, and admission work are
implemented and tested. Physical profile acquisition, evidence governance, and
the <=20% G2 calibration result remain open and still block a truthful
deployment-authoritative G2 plan.
Remote-expert plans remain non-authoritative until contention/route calibration
closes; deployment mode also declines exploratory replica allocation because its
independent route model is still open. The corrected target/current split is
normative in [`cost-model-and-calibration.md`](cost-model-and-calibration.md).
