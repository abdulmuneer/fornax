# Fornax Repo Review Lens Pass - 2026-06-29

Source lenses: `../research/combined_skill_groups.md` and `review_lenses_by_skill_for_fornax.md`.

## Summary Judgment

Approve with comments at the current proxy-development scope. The repo is strong on model-free contracts, golden vectors, proxy-gate evidence, and explicit formal-vs-proxy boundaries. The highest-priority review issues found in this pass were evidence-honesty and test-governance gaps around the Phase 3 proxy gate and the default deterministic validation command.

## Top Findings Implemented

### Program Governance / Documentation

- Finding: Phase 3 proxy-gate packet validation was weaker than Phase 4/5 validation. It required deferred requirement IDs, but did not enforce `status=deferred`, non-empty reasons, gate date/signature fields, or endpoint-validation metadata. That allowed a malformed packet to carry `phase3_proxy_passed=true` more easily than later gates.
- Change: `fornax.phase3_proxy_gate` now validates gate metadata, endpoint validation shape, deferred requirement statuses/reasons, and ties `phase3_proxy_passed` to a fully valid packet.
- Evidence: focused Phase 3 regression tests and `python3 -m fornax test phase3-proxy-gate` pass.

### Software Engineering / Testing

- Finding: `make golden` only ran the original core fixture subset even though many deterministic CLI contract suites now exist. This made the primary quick validation command weaker than the repo's actual evidence surface.
- Change: `Makefile` now centralizes `GOLDEN_TESTS` and runs the deterministic no-hardware contract/golden suites, including Phase 3, Phase 4, and Phase 5 proxy-gate fixtures.
- Evidence: `make golden` and `make test` pass.

### System / Networking / LLM Evidence Boundaries

- Finding: Phase 3 had no packaged proxy-gate golden fixture comparable to Phase 4/5. That left the endpoint/proxy gate path less reproducible and easier to rely on `/tmp` artifacts from prior sessions.
- Change: Added `fornax/golden_vectors/phase3_proxy_gate/fixture.json` and an endpoint-summary companion that explicitly records two-H100 local proxy scope and `formal_g3_passed=false`.
- Evidence: the new `fornax test phase3-proxy-gate` suite reports 8/8 checks passed with the formal G3 deferral warning.

### Documentation / Status Honesty

- Finding: the root README status described the runtime as simply the next frontier and did not reflect the current Phase 3-5 proxy packet posture tracked in the program ledger.
- Change: README and CLI reference now describe the implemented proxy-gate fixture layer and the expanded deterministic validation suite while preserving formal gate deferrals.

## Remaining High-Priority Follow-Ups

- Formal gate evidence remains deferred: real frontier target-model parity, product auth/mTLS keying, distributed partition proof, real node-loss/add-node lab evidence, installable release, operator acceptance, lab-reference benchmark, and Sponsor decisions.
- Test organization remains a maintainability concern: most coverage still lives in `tests/test_fornax_planner.py`; future work should split high-churn gate/runtime suites into focused test modules once the current untracked program slice is stabilized.
- The local HTTP smoke still embeds local certificate fixtures in code. They are marked non-production, but product security work should move real keying and certificate handling behind explicit operator-managed artifacts.

## Verification

- `python3 -m py_compile fornax/phase3_proxy_gate.py fornax/cli.py tests/test_fornax_planner.py`
- `python3 -m unittest tests.test_fornax_planner.FornaxPlannerTest.test_phase3_proxy_gate_packet_validates_two_h100_proxy tests.test_fornax_planner.FornaxPlannerTest.test_phase3_proxy_gate_rejects_formal_g3_overclaim tests.test_fornax_planner.FornaxPlannerTest.test_phase3_proxy_gate_rejects_missing_gate_signer tests.test_fornax_planner.FornaxPlannerTest.test_phase3_proxy_gate_rejects_non_deferred_requirement_status tests.test_fornax_planner.FornaxPlannerTest.test_phase3_proxy_gate_rejects_bad_endpoint_validation tests.test_fornax_planner.FornaxPlannerTest.test_phase3_proxy_gate_fixture_passes`
- `python3 -m fornax test phase3-proxy-gate`
- `make golden`
- `python3 -m unittest discover -s tests -p 'test_fornax*.py'`
- `make test`
