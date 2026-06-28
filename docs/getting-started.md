# Getting started

This walks you from a fresh clone to your first placement plan and throughput
prediction in a few minutes. Everything here runs on a laptop — **no GPU, no
model download, no network.** That is deliberate: the part of Fornax you can run
today is the *planner and simulation/contract layer*, which is pure logic.

If you want the mental model first, read [Concepts](concepts.md). If you just
want it running, keep going.

## 1. Prerequisites

- **Python 3.10 or newer** (developed and tested on CPython 3.12).
- Nothing else. Fornax's runnable layer has **zero third-party dependencies** —
  it is pure standard-library Python. There is no `pip install` step and no
  packaging to build.

```bash
python3 --version    # expect 3.10+
```

## 2. Get the code

```bash
git clone git@github.com:abdulmuneer/fornax.git
cd fornax
```

You run Fornax as a module from the repo root — `python3 -m fornax …`. There is
no installed `fornax` command; the repo *is* the tool.

## 3. Verify the install

One command runs the whole deterministic suite — the golden-vector/contract
self-tests and the unit tests. If this is green, your environment is good:

```bash
make test
```

Prefer to run the pieces directly:

```bash
python3 -m fornax --help          # the full CLI surface
make golden                       # contract / golden-vector self-tests only
make unittest                     # unit-test suites only
```

You should see the golden suites report `N/N passed` and the unit tests end in
`OK`.

## 4. Your first plan → simulate

The core loop is two steps:

1. **`plan`** — given a *model* and an *inventory of machines*, decide how to
   place the model across them (which layers live where), and report whether it
   even fits.
2. **`simulate`** — given that plan, predict aggregate throughput, per-request
   latency, and pipeline bubble.

Fornax ships runnable fixtures, so you can do this immediately without writing
any files. First, describe a one-machine "fleet" as an inventory file:

```bash
cat > my_fleet.json <<'JSON'
{
  "nodes": [
    {
      "id": "gpu0",
      "vendor": "nvidia",
      "runtime": "max",
      "mem_free_bytes": 16777216,
      "compute_class": 1000000000000.0,
      "mem_bandwidth_bytes_s": 100000000000.0,
      "supports_stage": true,
      "supports_expert_worker": true,
      "supports_kv": true,
      "supported_dtypes": ["fp16", "fp8"]
    }
  ],
  "links": []
}
JSON
```

Now place the bundled example model (`v0_target_contract_fixture.md`, a small
"target contract" that describes a model and a serving target) onto that fleet,
and simulate the result:

```bash
python3 -m fornax plan \
    --target fornax/golden_plans/v0_target_contract_fixture.md \
    --inventory my_fleet.json \
    --out plan.json

python3 -m fornax simulate --plan plan.json
```

You should see:

```text
wrote placement plan: plan.json
simulate: throughput=89214.343 tok/s latency=0.000359s bubble=0.000
```

That throughput is a **cost-model prediction for this placement**, not a
benchmark of your machine — the toy model and fleet are tiny, which is why the
number is large. The point of the toy run is the *shape* of the workflow, not
the magnitude. See [the honesty note](concepts.md#the-honest-constraint).

Open `plan.json` to see the decision: which node hosts which layers, the
per-stage memory and timing, and a human-readable `reason` for each choice under
`explanations`.

## 5. Check a plan against a target

A *target contract* can also carry acceptance thresholds (a throughput floor,
memory headroom, a concurrency sweep). `target validate` plans the model onto
your fleet and checks it against those thresholds:

```bash
python3 -m fornax target validate \
    fornax/golden_plans/v0_target_contract_fixture.md \
    --inventory my_fleet.json
```

Prints `valid` (exit 0) when the placement clears every threshold, or
`invalid: <failed checks>` (exit 2) otherwise. This is the gate you would put in
CI to keep a target honest as the planner or the fleet changes.

## 6. See an infeasible plan

Feasibility is a first-class outcome, not an error. Shrink the fleet's memory so
the model can't fit and watch `plan` refuse:

```bash
python3 -c "import json; d=json.load(open('my_fleet.json')); \
d['nodes'][0]['mem_free_bytes']=1024; json.dump(d, open('tiny_fleet.json','w'))"

python3 -m fornax plan \
    --target fornax/golden_plans/v0_target_contract_fixture.md \
    --inventory tiny_fleet.json \
    --out infeasible.json ; echo "exit=$?"
```

`plan` exits `2`, and `infeasible.json` records `"feasible": false` with an
`infeasible_reason`. (The bundled golden plan `model_too_big.json` is a
permanent example of this case.)

## 7. Where to go next

- **[Planning and simulation](planning-and-simulation.md)** — the workflow in
  depth: writing your own model/fleet, the request-trace option for `simulate`,
  reading the plan and prediction, and the `preflight` evidence bundle.
- **[Input file reference](input-formats.md)** — the exact JSON schema for
  target contracts, inventories, and links, field by field.
- **[CLI reference](cli-reference.md)** — every command, grouped by purpose.
- **[Concepts](concepts.md)** — the thesis, the vocabulary, and the constraints
  that shape every plan.
