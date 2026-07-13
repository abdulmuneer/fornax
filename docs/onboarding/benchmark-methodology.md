# Fornax benchmark methodology of record

## Benchmark of record

Only an accepted physical `lab-reference` run can support product performance
claims. Simulation and same-host proxies are development evidence.

## Required inputs

Capture exact commands, prompts or trace hashes, source/model/runtime versions,
hardware and network environment, correctness artifacts, raw logs, and a ledger
record.

## Correctness before throughput

Reject performance evidence unless the same configuration passes numerical
parity and deterministic-token requirements. Do not loosen tolerances to admit a
failing backend.

## Reproducibility

Record warmup, iterations, concurrency, context, thermal state, memory peaks,
and component timings. Preserve raw artifacts and build lineage.

## Gate mapping

T0/T1 informs implementation, T2 validates one backend, T3 closes G2 physical
distribution, and T4 is required for frontier-capacity and later gates.
