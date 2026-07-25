PYTHON ?= python3

GOLDEN_TESTS := \
	golden-plans \
	runtime-format \
	network-contract \
	engine-seam \
	stage-host \
	serving-adapter \
	state-ownership \
	engine-simulation \
	observability \
	metrics-ledger \
	trace-ledger \
	worker-contract \
	transport-contract \
	trust-boundary \
	moe-runtime \
	moe-migration \
	remote-expert-probe \
	moe-parity-probe \
	model-support \
	continuous-batching \
	scheduler-contract \
	stage-replication \
	resilience-replay \
	ops-lifecycle \
	onboarding-methodology \
	program-governance \
	backend-coverage \
	phase3-proxy-gate \
	phase4-resilience-gate \
	phase5-ga-gate \
	benchmark-ledger \
	pipeline-correctness-probe \
	throughput-scaling \
	stage-abi-v1 \
	stage-abi-v2

.PHONY: test golden unittest doctor public-boundary help

# Full deterministic suite: contract/golden self-tests + the unittest suites.
# No GPU, no model, no external network.
test: golden unittest

# Golden-vector / contract self-tests driven through the CLI.
golden:
	@set -e; for suite in $(GOLDEN_TESTS); do \
		$(PYTHON) -m fornax test $$suite; \
	done

unittest:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

doctor:
	@if [ -z "$(BUNDLE)" ]; then \
		echo "usage: make doctor BUNDLE=path/to/preflight-bundle"; \
		exit 2; \
	fi
	$(PYTHON) -m fornax doctor --bundle "$(BUNDLE)"

public-boundary:
	$(PYTHON) scripts/check_public_boundary.py

help:
	@echo "make test      - golden self-tests + unittest suite (no hardware)"
	@echo "make golden    - deterministic CLI contract/golden self-tests"
	@echo "make unittest  - unittest suites only"
	@echo "make doctor BUNDLE=... - inspect a phase-0 evidence bundle"
	@echo "make public-boundary - inspect staged files for publication violations"
	@echo "python3 -m fornax --help  - full CLI surface"
