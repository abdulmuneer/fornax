# Apple MLA Prefill: M3 vs M5 Kernel Challenges

Status: live engineering tracker, started 2026-06-30.

This note tracks implementation differences while porting `flare_mla_prefill`
from the upstream MAX NVIDIA/AMD paths to Apple Silicon. It is scoped to the
Apple MLA prefill kernel work in `external/modular`.

## Current Result

- The first Apple MLA prefill kernel builds against upstream MAX, but it is
  M5-only because it uses the 16x16 Apple simdgroup matrix path.
- On this M3 Max host, Metal rejects that pipeline at runtime:
  `simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later`.
- M3 support now has a distinct 8x8 simdgroup matrix kernel path in the local
  upstream MAX checkout.
- The M3 smoke target runs on this M3 Max host with `compute_capability: 3` and
  reports `non-finite outputs: 0`.
- The current M3 check is a launch/finiteness smoke test, not yet a full
  numerical correctness comparison against a host or naive-MHA reference.
- A focused Apple MLA decode smoke test now passes on M3 through a
  correctness-first fallback that routes through Apple `mha_gpu_naive` and then
  copies out the MLA latent/value portion.
- A focused Apple MoE indices smoke test now passes on M3 after adding a serial
  Apple fallback for expert bucketing in `max/kernels/src/nn/moe.mojo`.
- A focused Apple rank-2 gather smoke test now passes on M3 for both `int32`
  and `uint32` axis-0 indices, clearing the earlier Metal f64/floor blocker in
  the DeepSeek MoE permutation path.
- The host-side MLA decode dispatch scalar helper now accepts Metal and returns
  a single-partition key for the Apple fallback.
- `DeepSeek-V2-Lite-Chat` now runs through MAX on this M3 Max for short local
  generation smokes: 1-token generation emitted `!`, and 8-token generation
  emitted `!!!!!!!!`.

## Hardware Primitive Split

| Area | M5 path | M3 path |
| --- | --- | --- |
| Apple GPU primitive | 16x16 `MmaOpApple` / Metal GPUFamily10 | 8x8 `_mma_apple_8x8` |
| Availability | M5 / `compute_capability() == 5` | M1-M4, including M3 |
| Fragment elements per lane | 8 | 2 |
| Fragment row ownership | two rows: `rb`, `rb + 8` | one row |
| Fragment column ownership | four cols per owned row | two cols per owned row |
| Existing MAX abstraction | `MmaOpApple` struct | lower-level intrinsic and matmul helper |
| Direct code reuse from M5 kernel | partial: algorithm only | limited: fragment math must change |

## Implementation Challenges

1. Fragment layout is different.
   The M5 softmax code assumes each lane owns two row fragments packed as
   `SIMD[float32, 8]`. M3 8x8 owns one row and two columns as
   `SIMD[float32, 2]`, so the row max/sum, output rescale, and final store all
   need separate code.

2. The M5 `MmaOpApple` helper cannot be reused as-is.
   It hardcodes 16x16 tiling and 8-element fragments. The M3 path must use
   `_mma_apple_8x8` plus explicit fragment loads and stores, similar to
   `linalg/matmul/gpu/apple/matmul_8x8.mojo`.

3. Softmax reduction is similar in topology but not in shape.
   Both layouts reduce row-sharing lanes with XOR masks `{1, 8}`, but M5 reduces
   two lane-owned rows and M3 reduces one. This argues for a separate
   `AppleSoftmax8` state object rather than over-generalizing `AppleSoftmax`.

4. Paged KV alignment constraints change.
   The M5 kernel resolves pages per 16-row sub-tile and therefore requires
   `page_size % 16 == 0`. An M3 8x8 kernel can resolve pages per 8-row sub-tile,
   so the natural constraint is `page_size % 8 == 0`; a conservative per-row
   pointer path can avoid relying on a whole 8-row subtile being page-contiguous.
   The current M3 implementation uses scalar `block_paged_ptr[1]` loads for
   K/V/K_rope, so it prioritizes correctness and simpler paging semantics over
   bandwidth.

5. Launch geometry and performance expectations differ.
   The M5 kernel uses a wide 16-simdgroup threadgroup to get L2 reuse without
   shared memory. The M3 first version should prioritize correctness with one
   simdgroup per 8 query rows; this is now the implemented first pass.
   Performance tuning can later experiment with wider threadgroups and vector
   fragment loads.

6. Runtime dispatch must be capability-gated.
   `has_apple_gpu_accelerator()` is not enough. M5 must route to the 16x16
   kernel when `ctx.compute_capability() >= 5`; M3/M1-M4 must route to the 8x8
   kernel. Falling through to the M5 kernel on M3 causes Metal pipeline creation
   failure.

7. Tests need separate meanings.
   A compile-smoke target can build the M5 specialization on M3, but runtime
   validation must skip M5. The M3 target must actually run on this host and
   compare output against a reference, otherwise we only know it links.

8. Full DeepSeek-V2 is not only an MLA prefill problem.
   `DeepSeek-V2-Lite-Chat` also exercises MLA decode after prefill and MoE
   routing/index creation in every routed layer. The first Apple MLA decode
   route now exists as a correctness-first fallback, but the optimized split-K
   decode attempt failed Metal pipeline creation on this M3 Max with an XPC
   interruption. With decode unblocked, the next full-model blocker was MoE
   permutation gather lowering:
   `gather_r2_w8_b256_gs_True_b6be364d` and
   `gather_r2_w1_b256_gs_False_c8a36a1d` lowered through double-precision
   conversion/floor operations that Metal rejected. A direct flat GPU rank-2
   axis-0 gather path now clears that blocker for the short DeepSeek smoke.

9. Runtime helper contracts can lag kernel support.
   After the Apple decode fallback and gather path were in place, generation
   still failed because `mla_decode_dispatch_scalars` rejected `ctx.api() ==
   "metal"`. The Apple fallback is non-split-K, so the helper now returns
   `(batch_size, q_max_seq_len, 1)` for Metal. This is a host/runtime contract
   fix rather than a device math kernel.

## Verification Checklist

- Done: build `//max/kernels/src/nn:nn` with both Apple MLA paths exported.
- Done: build and run an M3 smoke target that launches the 8x8 kernel on this
  M3 Max.
- Done: smoke target verifies finite output on M3.
- Done: focused Apple MLA decode smoke test passes on M3 through the generic-MHA
  fallback.
- Done: focused Apple MoE indices smoke test passes on M3.
- Done: focused Apple rank-2 gather smoke test passes on M3 for `int32` and
  `uint32` indices.
- Done: `DeepSeek-V2-Lite-Chat` 1-token and 8-token `generate` smoke runs pass
  on M3 Max through the source-built MAX CLI.
- Pending: add a correctness comparison against the existing naive/reference
  path.
- Pending: benchmark the M3 8x8 kernel against `mha_gpu_naive`.
- Pending: confirm dispatch selects M3 8x8 on `compute_capability() < 5` and
  M5 16x16 on `compute_capability() >= 5` in a full model graph.
- Pending: run longer prompts, larger output counts, serving, batching, and
  memory-pressure checks before treating the Apple path as operational serving.

## Open Decisions

- Whether the first M3 implementation should require `page_size % 8 == 0` or use
  per-row block-paged pointers to support arbitrary page sizes.
- Whether to keep M3 output depth at the same BF16/FP16-only scope as M5 or add
  FP32 output support for easier debugging.
- Whether to implement only paged MLA first, or also support contiguous/ragged
  `LayoutTensorMHAOperand` K_rope inputs immediately.
- Whether the direct graph-compiler rank-2 gather path should stay generic for
  all GPUs or be narrowed further once a reliable Apple-only compile-time gate
  is available in this layer.
