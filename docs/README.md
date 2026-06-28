# Fornax documentation

End-user documentation for Fornax — the heterogeneous frontier-model serving
engine. Start here.

> **What you can do today.** The shipping codebase is the **planner and the
> simulation/contract layer** (pure Python, no GPU or model required). You can
> describe a model and a fleet of machines, ask Fornax to place the model across
> them, and get a *predicted* throughput/latency profile plus a feasibility
> verdict. The heterogeneous Mojo/MAX expert runtime that turns a plan into real
> served tokens is under construction. The docs below are explicit about which
> numbers are **predictions from a cost model** and which would be **measured**.

## Guides

| Guide | Read it when you want to… |
|---|---|
| [Getting started](getting-started.md) | Install Fornax, verify the install, and run your first plan → simulate in a few minutes. |
| [Concepts](concepts.md) | Understand what Fornax is, the MoE/heterogeneous thesis, and the vocabulary (stages, experts, the honest constraint) the other docs assume. |
| [Planning and simulation](planning-and-simulation.md) | Walk the core workflow end to end: describe a model + fleet, place it, predict throughput, validate against a target. |
| [Input file reference](input-formats.md) | Look up the exact JSON schema for a target contract, an inventory, and a links file. |
| [CLI reference](cli-reference.md) | Find a command. The full `python -m fornax` surface, grouped by what it's for. |

## The 60-second version

```bash
make test                              # everything below runs with no GPU and no model
python3 -m fornax plan \
    --target my_target.md \
    --inventory my_fleet.json \
    --out plan.json                    # place the model across the fleet
python3 -m fornax simulate --plan plan.json   # predict throughput / latency / bubble
```

If `plan` prints `wrote placement plan` and exits 0, your model fits the fleet
and `plan.json` holds the placement. If it exits 2, the plan is infeasible and
`plan.json` records the reason.

## A note on honesty

Fornax is built around a hard rule: **a number is either traced to a measurement
or labelled as a prediction — never dressed up as the other.** The simulator
reports cost-model predictions; the gate validators exist to keep those
predictions honest; and the docs follow suit. When you read "throughput" from
`simulate`, that is the cost model's estimate for the placement, not a benchmark
of your hardware. See [Concepts → The honest constraint](concepts.md#the-honest-constraint).
