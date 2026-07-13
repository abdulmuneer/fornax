# Apple Silicon `flare_mla_prefill` porting note

Source snapshot date: 2026-06-30; local implementation update: 2026-07-01.

This note originally prepared the shape of an Apple Silicon implementation for
MAX `flare_mla_prefill` / `flareMLA_prefill`, based on the current public
Modular MAX source and docs. The local status has advanced: the vendored
`external/modular` MAX checkout now contains a source-built Apple path that
passes short `DeepSeek-V2-Lite-Chat` generation on an M3 Max. Fornax itself
remains the Python simulation and contract layer; the kernel work lives in the
nested MAX/Mojo checkout.

## Short answer

Yes. The local implementation confirms the correct direction: Apple MLA should
be an Apple-specific kernel/dispatch path, not a line-by-line port of the NVIDIA
path.

The packaged/upstream-public `flare_mla_prefill` path that failed our first
Apple smoke was effectively NVIDIA/AMD-only for the DeepSeek path. The local
source-built checkout now adds Apple branches for MLA prefill/decode and clears
the follow-on MoE/gather blockers needed for a short DeepSeek-V2-Lite-Chat run
on M3 Max. It is still not an upstream-ready or production-quality serving
claim: the active decode and MoE paths are correctness-first fallbacks, and
numerical parity/performance validation remains open.

## 2026-07-01 local result

| Area | Local status |
|---|---|
| Apple MLA prefill | Implemented in `external/modular/max/kernels/src/nn/attention/gpu/apple/mla_prefill.mojo`, with an M3-compatible path and an M5-style path gate. |
| Dispatch | `external/modular/max/kernels/src/nn/attention/gpu/mla.mojo` imports Apple MLA and routes Apple GPU targets through it for the supported BF16/FP16 shape family. |
| Apple MLA decode | Implemented as a fallback in `apple/mla_decode.mojo` using Apple `mha_gpu_naive` over the full DeepSeek MLA cache row, then copying the latent/value prefix. |
| MoE indices | Apple fallback in `nn/moe.mojo` avoids the unsupported Metal `vote` primitive for the smoke path. |
| Rank-2 gather | Direct Apple GPU rank-2 axis-0 gather path in `graph_compiler/builtin_kernels/gather_scatter.mojo` avoids the generic Metal lowering failure seen in the MoE permutation path. |
| Dispatch scalars | `mla_decode_dispatch_scalars` returns `(batch_size, q_max_seq_len, 1)` for Metal, matching the Apple single-partition decode fallback. |
| Evidence | Existing Bazel logs show Apple prefill/decode/MoE/gather smokes passed; fresh source-built `max generate` produced 1-token and 8-token DeepSeek-V2-Lite-Chat outputs on M3 Max. See [deepseek-v2-lite-max-check.md](deepseek-v2-lite-max-check.md). |

## Upstream files to inspect

| Area | Upstream file or doc |
|---|---|
| Public `flare_mla_prefill` API | `https://docs.modular.com/max/api/kernels/nn/attention/gpu/mla/flare_mla_prefill.md` |
| Main MLA dispatch and generic kernel | `https://raw.githubusercontent.com/modular/modular/main/max/kernels/src/nn/attention/gpu/mla.mojo` |
| NVIDIA SM100 MLA prefill | `https://raw.githubusercontent.com/modular/modular/main/max/kernels/src/nn/attention/gpu/nvidia/sm100/mla_prefill.mojo` |
| AMD gfx950 structured MLA prefill | `https://raw.githubusercontent.com/modular/modular/main/max/kernels/src/nn/attention/gpu/amd_structured/mla_prefill.mojo` |
| Apple flash-attention prefill | `https://raw.githubusercontent.com/modular/modular/main/max/kernels/src/nn/attention/gpu/apple/fa_prefill.mojo` |
| Apple naive FA decode | `https://raw.githubusercontent.com/modular/modular/main/max/kernels/src/nn/attention/gpu/apple/naive_fa_decode.mojo` |
| MHA dispatch with Apple branch | `https://raw.githubusercontent.com/modular/modular/main/max/kernels/src/nn/attention/gpu/mha.mojo` |

## What NVIDIA/AMD `flare_mla_prefill` does today

The public API is an optimized compute-graph MLA prefill kernel with ragged
Q/K/V support. The important layout contract is:

| Tensor | Logical shape | Notes |
|---|---|---|
| `q` | `[seq_len, num_heads, q_depth]` | For DeepSeek-style MLA, often `q_depth = 192`. |
| `k` | `[cache_len, kv_heads, depth]` | The non-RoPE key projection, often `depth = 128`. |
| `v` | `[cache_len, kv_heads, depth]` | Output depth follows V depth. |
| `k_rope` | `[cache_len, 1, q_depth - depth]` | Broadcast across query heads. Often the missing 64 dimensions. |
| `output` | `[seq_len, num_heads, depth]` | Accumulates attention over V, not full `q_depth`. |

The implementation has several overloads:

| Path | Purpose |
|---|---|
| Paged `k_rope` with ragged K/V | Real cache path using `KVCacheMHAOperand`. |
| TileTensor `k_rope` variants | Test and non-cache helpers. |
| FP8/block-scale variants | Route through typed scale operands. |
| Per-token scale variants | Route to SM100-specific scale-aware kernels. |

The packaged/public baseline that motivated the local patch had this dispatch
flow:

1. Build `RaggedMHAOperand` objects for K, V, offsets, and optional scales.
2. Build either a `KVCacheMHAOperand` or `LayoutTensorMHAOperand` for `k_rope`.
3. Create an `MHAConfig` with `num_keys_per_block = 64`, `WN = 64`, and FlashAttention v2 algorithm mode.
4. Enter `flare_mla_prefill_dispatch`.
5. Reject non-NVIDIA/non-AMD GPU targets at compile time for the DeepSeek MLA
   path.
6. If the default device is NVIDIA SM100, call the dedicated SM100 MLA prefill path.
7. Otherwise launch the generic `mla_prefill` kernel:
   - NVIDIA grid is ordered as query block, head, batch.
   - AMD grid is ordered as head, query block, batch.
   - NVIDIA allocates shared memory for Q/K/K_rope tiles.
   - AMD uses the gfx950 structured `Attention` implementation.

The core kernel computes two score contributions before softmax:

```text
score = Q_nope @ K_nope.T
score += Q_rope @ K_rope.T
output = softmax(score) @ V
```

That is the algorithm Apple needs to match.

## Apple pieces already available in MAX

MAX already has an Apple attention path, but for MHA rather than MLA. The
relevant reusable pieces are:

| Piece | Why it matters for MLA |
|---|---|
| `has_apple_gpu_accelerator` | Existing compile-time target gate used by MHA dispatch. |
| `fa_prefill_apple` / `fa_prefill_apple_core` | Host launcher and Metal prefill kernel shape for Apple. |
| `MmaOpApple` | Apple simdgroup GEMM primitive for 16x16 attention tiles. |
| Apple online softmax state | Register-resident softmax update used in prefill. |
| `MHAOperand.block_paged_tile` | Existing paged-KV tile accessor used by Apple attention. |
| MHA Apple env-flag pattern | `MODULAR_ENABLE_APPLE_FA_PREFILL` style gate for rollout. |

The Apple FA prefill design avoids threadgroup memory and barriers; it relies on
Apple simdgroups and L2 reuse. That is the main reason the NVIDIA shared-memory
implementation should not be copied directly.

## Proposed upstream implementation shape

This should be implemented in upstream MAX, not under the current Fornax Python
package.

| File | Change |
|---|---|
| `max/kernels/src/nn/attention/gpu/apple/mla_prefill.mojo` | New Apple MLA prefill host launcher and core kernel. |
| `max/kernels/src/nn/attention/gpu/mla.mojo` | Import Apple target gate and Apple MLA launcher; extend dispatch to Apple. |
| Apple attention docs/index | Add Apple MLA prefill entry after validation. |
| Tests | Add deterministic tiny MLA prefill tests and ragged/cache boundary cases. |

### Dispatch sketch

The dispatch should extend the current NVIDIA/AMD-only target check:

```mojo
from std.sys import has_apple_gpu_accelerator
from .apple.mla_prefill import mla_prefill_apple

comptime assert (
    has_nvidia_gpu_accelerator()
    or has_amd_gpu_accelerator()
    or has_apple_gpu_accelerator()
), "flare_mla_prefill supports NVIDIA, AMD, and Apple GPU targets."

...

if batch_size == 0 or max_prompt_len == 0:
    return

comptime if has_apple_gpu_accelerator():
    # First Apple milestone: BF16/FP16, no FP8 scaling, M5/Metal path.
    return mla_prefill_apple[config, q_depth, KVCacheT](
        ctx, q, k, v, k_rope, valid_length, output, batch_size, max_prompt_len
    )
```

The exact argument list must follow the existing `flare_mla_prefill_dispatch`
operands, including ragged offsets and cache offsets. The sketch above shows the
branch shape only.

### Apple kernel sketch

The Apple kernel should mirror `fa_prefill_apple_core`, with MLA-specific score
construction:

```text
for each batch, query-head, query tile:
    load Q_nope = Q[:, :, 0:depth]
    load Q_rope = Q[:, :, depth:q_depth]
    initialize Apple online softmax state

    for each KV tile:
        load K_nope from paged/ragged K
        load K_rope from paged K_rope, broadcasting the single KV head as needed
        scores = MmaOpApple(Q_nope, K_nope.T)
        scores += MmaOpApple(Q_rope, K_rope.T)
        apply valid-length and causal/prompt mask
        update online softmax state
        accumulate softmax(scores) @ V

    normalize and store output[:, :, 0:depth]
```

The first Apple version should be intentionally narrow:

| Constraint | First milestone |
|---|---|
| Apple GPU target | `has_apple_gpu_accelerator()` and compute capability at least 5. |
| Storage dtype | BF16 first; FP16 if already supported by the Apple FA path. |
| Accumulation | FP32. |
| MLA shape | `depth = 128`, `q_depth = 192`, `rope_depth = 64`. |
| Query tile | Start from the Apple FA prefill 16-row tile style. |
| KV tile | 16/32/64 candidates, selected by register pressure and occupancy. |
| Cache mode | Paged K/V and paged/broadcast `k_rope`. |
| Scaling | Use the existing non-scale path first. |

## Not in the first Apple milestone

| Feature | Reason to defer |
|---|---|
| FP8 K/V/K_rope | Needs scale handling and error-budget validation. |
| Per-token scales | Current scale-aware fast path is SM100-specific. |
| Sparse MLA prefill | Separate indexing and validation problem. |
| Decode | Different launch structure; Apple has a separate naive FA decode pattern. |
| Arbitrary `q_depth`/`depth` | DeepSeek MLA shape is the practical first target. |
| Non-M5 Apple GPUs | Current public Apple FA prefill path is M5/Metal-specific. |

## Validation plan

| Step | Check |
|---|---|
| 1. Compile gate | Build in a pinned MAX checkout on Apple Silicon with the Apple target enabled. |
| 2. Tiny numerical test | Compare BF16/FP16 output to a CPU/PyTorch reference for batch 1, head 1, prompt 16, cache 64, `q_depth = 192`, `depth = 128`. |
| 3. Ragged prompts | Vary `valid_length` and offsets across a small batch. |
| 4. Cache boundaries | Exercise page-boundary reads for K, V, and `k_rope`. |
| 5. Broadcast correctness | Verify `k_rope` broadcasts across query heads and KV groups exactly once. |
| 6. Cross-backend parity | Compare against NVIDIA/AMD outputs for the same vectors when hardware is available. |
| 7. Performance probe | Compare against Apple FA prefill and generic MHA fallback for latency, bandwidth, and occupancy. |

Recommended tolerances should start conservative and be dtype-specific:

| Dtype path | Initial tolerance stance |
|---|---|
| BF16 storage, FP32 accumulation | Compare with absolute and relative tolerances around standard BF16 attention error. |
| FP16 storage, FP32 accumulation | Similar to BF16, but validate overflow and softmax stability separately. |
| FP8 paths | Do not enable until a scale-aware reference and per-layer error budget exist. |

## Risks and open questions

| Risk | Porting implication |
|---|---|
| `k_rope` operand mismatch | The Apple MHA accessor must cleanly express MLA's separate RoPE cache tile and broadcast semantics. |
| Register pressure | Apple MLA needs two QK score terms before softmax, increasing live fragments versus MHA. |
| Masking semantics | Ragged prompt lengths and cache offsets must match the existing NVIDIA/AMD contract. |
| Dtype coverage | BF16/FP16 support must be verified on the exact Apple GPU and MAX build. |
| Graph routing | MAX graph lowering must select the Apple MLA branch only for validated shapes/dtypes. |
| Performance portability | The M5/Metal FA prefill assumptions may not hold for older Apple GPUs. |

## Fornax stance

For Fornax planning, treat Apple `flare_mla_prefill` as local positive evidence,
not a closed production capability. The required evidence before it can become a
serving-grade Fornax assumption is:

1. A pinned MAX branch containing the Apple MLA kernel. The local
   `external/modular` branch now satisfies this for the 2026-07-01 smoke.
2. Passing deterministic numerical tests against a CPU reference.
3. Passing ragged/cache boundary tests.
4. Local Apple Silicon performance probes on the target hardware.
5. A documented fallback when shape, dtype, or hardware gates fail.
6. Serving-mode, longer-context, batching, and memory-pressure validation.

Until those gates are met, the operator support chart should mark Apple Silicon
MLA as local evidence / limited (`E/L`) rather than generally supported.
