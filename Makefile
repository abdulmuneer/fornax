PYTHON ?= python3

.PHONY: test golden unittest doctor help

# Full deterministic suite: contract/golden self-tests + the unittest suites.
# No GPU, no model, no network.
test: golden unittest

# Golden-vector / contract self-tests driven through the CLI.
golden:
	$(PYTHON) -m fornax test golden-plans
	$(PYTHON) -m fornax test runtime-format
	$(PYTHON) -m fornax test network-contract
	$(PYTHON) -m fornax test engine-seam
	$(PYTHON) -m fornax test stage-host
	$(PYTHON) -m fornax test serving-adapter

unittest:
	$(PYTHON) -m unittest discover -s tests -p 'test_fornax*.py'

doctor:
	$(PYTHON) -m fornax doctor

help:
	@echo "make test      - golden self-tests + unittest suite (no hardware)"
	@echo "make golden    - contract/golden-vector self-tests only"
	@echo "make unittest  - unittest suites only"
	@echo "make doctor    - inspect a phase-0 evidence bundle"
	@echo "python3 -m fornax --help  - full CLI surface"
