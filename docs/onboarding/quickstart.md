# Fornax operator quickstart

## Prerequisites

Use Python 3.10 or newer. The default tour needs no GPU, model, or network.

## First successful run

From the repository root:

```bash
python3 -m fornax quickstart --out-dir fornax-quickstart
```

The command writes an executable target, a synthetic NVIDIA/Apple inventory, a
two-stage placement, contract validation, a simulation, and `summary.json`.

## Interpret the result

The tour is a `simulation_fixture`. Its throughput is a cost-model prediction,
not a measurement or supported-hardware claim. Inspect `summary.json`, then run
the two `next_commands` it records.

## Move to your fleet

Replace `target.json` and `inventory.json` with your own explicit inputs, then
use `fornax plan`, `fornax target validate`, `fornax simulate`, and `fornax
preflight`. Physical MAX serving remains gated by the G2 validation plan.
