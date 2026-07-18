---
title: "Fornax - Heterogeneous MoE Serving"
permalink: /fornax/
header:
  overlay_image: /assets/images/hero-max.svg
  overlay_filter: 0.5
sidebar:
  nav: "fornax"
toc: false
author_profile: false
---

This series explains the design and implementation of **Fornax**, a
Mojo/MAX-native distributed inference engine intended to serve one sparse-MoE
model across heterogeneous commodity nodes when the model exceeds the memory of
any single node.

{: .notice--info}
**Evidence policy.** Model and platform claims link to dated records that name
the model, weights, dtype, shape, hardware, runtime build, command, workload
limits, validation method, and result.

## Series

- [Part 0 - Objective and Constraints](./00-objective-and-constraints.md) - the
  target workload, aggregate-throughput goal, latency boundary, and division of
  responsibility between MAX and Fornax.
- [Part 1 - What Accelerator Support Requires](./01-pytorch-parity-for-new-accelerators.md) -
  device integration, operator coverage, compilation, and the different
  requirements of training and inference.
- [Part 2 - MAX Platform Assessment](./02-max-platform-assessment.md) - what MAX
  provides, how device support is qualified, and which distributed-runtime
  responsibilities belong to Fornax.
- [Part 3 - Fornax Architecture](./03-fornax-architecture.md) - placement, stage
  execution, runtime formats, scheduling, transport, and measurement.
- [Part 4 - Model Bring-Up](./04-model-bring-up.md) - model-support validation
  and an Apple DeepSeek case study spanning MLA, MoE bucketing, gather lowering,
  and dispatch.

## Project references

- [Repository status and quickstart](../../../README.md)
- [Project plan v4](../project-plan-v4.md)
- [MAX operator and platform support](../max-operator-platform-support.md)
- [DeepSeek-V2-Lite Apple MAX runbook](../deepseek-v2-lite-max-check.md)

---

*Back to [Fornax documentation](../README.md).*
