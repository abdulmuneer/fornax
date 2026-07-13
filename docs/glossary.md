# Fornax glossary

## Planning terms

- **Plan ID:** immutable identity propagated through placement and execution.
- **Placement plan:** mapping from contiguous layer stages to nodes and links.
- **Logical host:** simulated host used without claiming physical multi-host evidence.

## Runtime terms

- **Stage executable:** versioned function for one contiguous model layer range.
- **T1 simulation:** multi-process reference/simulated execution on loopback.

## Operations terms

- **Drain:** stop admission and finish in-flight work before mutation.
- **Rollback:** restore the prior accepted plan and build.
- **Node replacement:** drain/remove a node, admit a replacement, then restore traffic.

## Benchmark and gate terms

- **Benchmark of record:** reproducible accepted physical benchmark.
- **Lab-reference:** controlled heterogeneous fleet used for gate-grade evidence.
- **Gate:** Sponsor decision with `PROCEED`, `ITERATE`, `NARROW`, or `KILL` outcome.
