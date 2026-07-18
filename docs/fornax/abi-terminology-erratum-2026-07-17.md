# Stage ABI terminology erratum

Date: 2026-07-17

Plan of record: `project-plan-v4.md` (unchanged)

This is a maturity-label clarification, not an architecture or historical-record
rewrite. Plan v4 and the 2026-07-10 gate records use “production Stage ABI” for
the intended single contract shared by reference, simulated, and future physical
backends. That phrase describes the **target role**, not the implementation's
current readiness.

The implemented contract is:

- **experimental FNX1 v1 / T0-T1 mechanism contract**;
- dense, single-request `StageRequest` semantics under a lockstep orchestrator;
- reference and simulated backends with framed loopback TCP; and
- no physical MAX, ragged batching, real vocabulary-logit, security, or product
  stability conclusion.

That list describes frozen FNX1. A separate **candidate FNX2 2.0** contract is
now implemented at T0/T1 with exact framing, ragged per-sequence descriptors, a
slow-correct reference oracle, an integrated scheduler, and two independent
loopback workers. It is not physical backend conformance or a supported-product
ABI.

Current documents and external claims must use “experimental FNX1 v1” or
“versioned experimental Stage ABI.” Historical plan, decision, and gate wording
remains immutable and should be read through this erratum. The
[`stage-abi-v2-ragged-design.md`](stage-abi-v2-ragged-design.md) contract now has
reference implementation and golden-vector evidence, but does not supersede
FNX1 for physical claims until MAX adapters pass the same corpus at T2/T3 and
the compatibility decision is recorded.

The same rule applies to historical “bounded Engine” or “bounded memory” wording.
EV-009 observed configured scheduler queues, one-frame channel credit, process
RSS, and cleanup for 1,800 seconds. It did not prove bounded request/KV,
idempotency, transform, or event state for indefinite service. I-22 and the
current lifecycle tests supersede that broad reading; a fresh sustained artifact
is required for current-contract authority.
