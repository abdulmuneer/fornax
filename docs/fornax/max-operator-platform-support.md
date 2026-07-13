# MAX operator platform support snapshot

Source snapshot date: 2026-06-30; local Apple DeepSeek evidence updated
2026-07-01.

This is a Fornax-oriented support chart for operators exposed in the current
Modular MAX public docs. Modular does not publish a single official per-operator
hardware matrix, so the support levels below combine:

- the `max.graph.ops` API list;
- MAX AI kernel package/module docs;
- vendor-specific kernel subtrees where present;
- current MAX package/platform requirements.

Use this as a planning map, not as G1/G2 evidence. A Fornax gate still needs a
pinned MAX build, exact hardware, model, dtype, shape, and local probe results.

## Support legend

| Mark | Meaning |
|---|---|
| `S` | Supported by current public docs for this broad platform class. |
| `V` | Vendor-specific kernel subtree or function docs exist. |
| `L` | Limited or hardware-specific support; validate architecture, dtype, model, and env flags. |
| `P` | Public docs expose the API/operator, but do not publish a per-platform guarantee. Probe required. |
| `E` | Local Fornax evidence exists for a pinned source-built checkout; not official packaged support. |
| `-` | No current public support evidence found for this platform. |

Platform columns mean:

- `NVIDIA`: Linux NVIDIA GPUs compatible with MAX.
- `AMD`: Linux AMD GPUs compatible with MAX.
- `Apple Silicon`: macOS on Apple Silicon and Apple GPU/Metal where documented.
- `CPU`: Intel/AMD Linux x86-64-v3 CPU path unless otherwise noted.

## Source links

| Source | Use in this chart |
|---|---|
| [MAX packages and requirements](https://docs.modular.com/max/packages.md) | Baseline platform support and caveats. |
| [MAX quickstart](https://docs.modular.com/max/get-started.md) | Broad NVIDIA/AMD GPU, CPU, and Mac wording. |
| [MAX container](https://docs.modular.com/max/container.md) | NVIDIA/AMD container support and CPU device selection. |
| [max.graph.ops](https://docs.modular.com/max/api/python/graph.ops.md) | High-level graph operator list. |
| [MAX AI kernels](https://docs.modular.com/max/api/kernels.md) | Kernel package index and CPU/GPU scope. |
| [Neural network kernels](https://docs.modular.com/max/api/kernels/nn.md) | NN operator package/module index. |
| [Attention kernels](https://docs.modular.com/max/api/kernels/nn/attention.md) | CPU/GPU attention package split. |
| [GPU attention kernels](https://docs.modular.com/max/api/kernels/nn/attention/gpu.md) | NVIDIA/AMD/Apple attention subtrees. |
| [GPU MLA kernels](https://docs.modular.com/max/api/kernels/nn/attention/gpu/mla.md) | Generic GPU MLA functions such as `flare_mla_prefill`. |
| [NVIDIA SM100 attention](https://docs.modular.com/max/api/kernels/nn/attention/gpu/nvidia/sm100.md) | Blackwell MHA/MLA prefill/decode coverage. |
| [AMD RDNA attention](https://docs.modular.com/max/api/kernels/nn/attention/gpu/amd_rdna.md) | RDNA MHA prefill/decode coverage. |
| [AMD gfx950 attention](https://docs.modular.com/max/api/kernels/nn/attention/gpu/amd_structured.md) | MI355X MHA/MLA coverage. |
| [Apple GPU attention](https://docs.modular.com/max/api/kernels/nn/attention/gpu/apple.md) | Apple Metal FA prefill/decode coverage. |
| [Linalg kernels](https://docs.modular.com/max/api/kernels/linalg.md) | Linear algebra package/module index. |
| [Quantization kernels](https://docs.modular.com/max/api/kernels/quantization.md) | Quantized matmul package/module index. |
| [Communication kernels](https://docs.modular.com/max/api/kernels/comm.md) | GPU collective package/module index. |
| [State-space kernels](https://docs.modular.com/max/api/kernels/state_space.md) | Mamba/Gated Delta kernel package/module index. |

## Platform baseline

| Platform | Baseline support level | Notes for Fornax |
|---|---:|---|
| NVIDIA GPU | `S` | MAX docs list NVIDIA GPUs as compatible and publish NVIDIA containers. Attention has SM90 and SM100 subtrees; SM100 explicitly covers MHA and MLA prefill/decode. |
| AMD GPU | `S` | MAX docs list AMD GPUs as compatible and publish AMD containers. Attention has RDNA and gfx950 subtrees; gfx950 explicitly includes MHA and MLA prefill/decode. |
| Apple Silicon | `L/P/E` | MAX package docs say Apple Silicon GPU support is functional for Mojo GPU programming, but large GenAI inference through packaged MAX remains conservative. Apple Metal attention kernels are documented, including M5 FA prefill and naive FA decode. Fornax local source-built evidence on 2026-07-01 shows `DeepSeek-V2-Lite-Chat` short `generate` can pass on M3 Max after Apple MLA/MoE/gather backend patches. |
| Intel/AMD CPU | `S` | MAX can serve on CPU and docs state CPU support; Linux CPU baseline is x86-64-v3. Some kernel pages also publish explicit CPU reference paths. |

## High-level graph operators

These are the documented Python `max.graph.ops` operators. They are graph API
surface, not a backend-by-backend kernel warranty.

| Category | Operators | NVIDIA | AMD | Apple Silicon | CPU | Notes |
|---|---|---:|---:|---:|---:|---|
| Elementwise math | `abs`, `acos`, `add`, `atanh`, `ceil`, `cos`, `div`, `erf`, `exp`, `floor`, `log`, `log1p`, `mod`, `mul`, `negate`, `pow`, `round`, `rsqrt`, `sin`, `sqrt`, `sub`, `tanh`, `trunc` | `S/P` | `S/P` | `L/P` | `S/P` | Dtype and shape legality still matter. |
| Comparison and boolean | `equal`, `greater`, `greater_equal`, `is_inf`, `is_nan`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `not_equal`, `where` | `S/P` | `S/P` | `L/P` | `S/P` | Basic graph predicates and selects. |
| Activations and normalization | `gelu`, `group_norm`, `layer_norm`, `logsoftmax`, `relu`, `rms_norm`, `sigmoid`, `silu`, `softmax` | `S/P` | `S/P` | `L/P` | `S/P` | NN kernels provide more specialized CPU/GPU implementations for several entries. |
| Reductions, sort, top-k | `argmax`, `argmin`, `argsort`, `bottom_k`, `cumsum`, `max`, `mean`, `min`, `prod`, `sum`, `top_k` | `S/P` | `S/P` | `L/P` | `S/P` | Ranking/top-k often has GPU-specialized kernels. |
| Linear algebra and signal | `as_interleaved_complex`, `dequantize`, `irfft`, `matmul`, `outer`, `qmatmul` | `S/P` | `S/P` | `L/P` | `S/P` | `irfft` docs mention cuFFT in the NN kernel module, so validate non-NVIDIA paths. |
| Shape, layout, and data movement | `broadcast_to`, `chunk`, `concat`, `flatten`, `fold`, `pad`, `permute`, `range`, `repeat_interleave`, `reshape`, `resize`, `resize_bicubic`, `resize_linear`, `resize_nearest`, `shape_to_tensor`, `shard_and_stack`, `slice_tensor`, `split`, `squeeze`, `stack`, `tile`, `transfer_to`, `transpose`, `unsqueeze` | `S/P` | `S/P` | `L/P` | `S/P` | Usually portable graph plumbing, with some GPU-specific fast paths. |
| Indexing, gather, scatter | `gather`, `gather_nd`, `masked_scatter`, `nonzero`, `scatter`, `scatter_add`, `scatter_max`, `scatter_min`, `scatter_mul`, `scatter_nd`, `scatter_nd_add`, `scatter_nd_max`, `scatter_nd_min`, `scatter_nd_mul` | `S/P` | `S/P` | `L/P` | `S/P` | Probe performance for MoE routing paths. |
| Convolution, pooling, vision | `avg_pool2d`, `band_part`, `conv2d`, `conv2d_transpose`, `conv3d`, `max_pool2d`, `non_maximum_suppression`, `roi_align` | `S/P` | `P` | `P` | `S/P` | GPU convolution package currently calls out vendor-specific implementation under NVIDIA. |
| Control, buffers, custom, distributed | `allgather`, `assert_same_device`, `buffer_create`, `buffer_load`, `buffer_store`, `buffer_store_slice`, `call`, `cond`, `constant`, `constant_external`, `custom`, `distributed_broadcast`, `distributed_scatter`, `hann_window`, `inplace_custom`, `parallel`, `print`, `rebind`, `while_loop` | `S/P` | `S/P` | `L/P` | `P` | Collectives are not equivalent to cross-vendor Fornax transport; custom ops require build/probe evidence. |

## MAX AI kernel operator families

| Category | Documented operators / modules | NVIDIA | AMD | Apple Silicon | CPU | Notes |
|---|---|---:|---:|---:|---:|---|
| Activation kernels | `elu`, `gelu`, `gelu_quick`, `gelu_tanh`, `leaky_relu`, `relu`, `relu_n1`, `sigmoid`, `sign`, `silu` | `S/P` | `S/P` | `L/P` | `S/P` | Basic scalar/vector kernels; probe dtype coverage. |
| Softmax kernels | `softmax`, `logsoftmax`, `softmax_2_pass`, `softmax_3_pass`, `softmax_kernel`, `softmax_with_temperature`, helpers `identity`, `mul`, `sub`, `reciprocal`, `reduce_add_simd` | `S/P` | `S/P` | `L/P` | `S/P` | `softmax_with_temperature` is documented as GPU. Apple attention pages also define Apple-specific online softmax state. |
| Normalization kernels | `group_norm`, `layer_norm`, `rms_norm`, `rms_norm_fused_residual_add`, `rms_norm_rope_gpu`, `apply_qk_rms_norm`, `row_mean_of_squares`, plus CPU/GPU block/warp variants | `S/P` | `S/P` | `P` | `S` | Several pages include explicit `_cpu` and `_gpu` variants. |
| Tensor creation, shape, and movement | `arange`, `broadcast`, `concat`, `cumsum`, `fold`, `pad`, `pad_gpu`, `repeat_interleave`, `reshape`, `resize`, `shard_and_stack`, `slice`, `spatial_merge`, `split`, `tile` | `S/P` | `S/P` | `L/P` | `S/P` | Public docs expose modules, but not a per-module platform matrix. |
| Indexing, ranking, sampling | `arg_nonzero`, `argmaxmin`, `argmaxmin_gpu`, `argsort`, `gather_scatter`, `index_tensor`, `index_fp8`, `nms`, `sampling`, `topk`, `topk_bitonic`, `topk_fi`, `toppminp`, `toppminp_gpu` | `S/P` | `S/P` | `P` | `S/P` | Some modules are explicitly GPU (`*_gpu`) or FP8-specialized. |
| Vision and image kernels | `bicubic`, `image`, `learnable_2d_interp_pos_emb`, `pool`, `resize`, `roi_align`, `tpool_patch_merger` | `S/P` | `P` | `P` | `S/P` | `bicubic` docs state CPU and GPU implementations. |
| Convolution kernels | `conv`, `conv_transpose`, `conv_utils`; graph ops `conv2d`, `conv2d_transpose`, `conv3d` | `V/P` | `P` | `P` | `S/P` | GPU convolution package says vendor-specific implementation lives under NVIDIA. |
| CPU attention | `flash_attention`, `flash_attention_kv_cache`, `flash_attention_split_kv` | `-` | `-` | `-` | `S` | CPU flash-attention package is separate from GPU attention. |
| GPU MHA and cross-attention | `flash_attention`, `flash_attention_dispatch`, `flash_attention_ragged`, `mha`, `mha_decoding`, `mha_decoding_single_batch`, `mha_decoding_single_batch_pipelined`, `mha_gpu_naive`, `mha_single_batch`, `mha_single_batch_pipelined`, `mha_splitk_reduce`, `mha_cross_gpu_naive` | `V` | `V` | `L` | `-` | NVIDIA SM90/SM100, AMD RDNA/gfx950, and Apple Metal attention pages exist. Apple docs are specific to M5 FA prefill and naive decode. |
| GPU MLA | `flare_mla_prefill`, `flare_mla_prefill_dispatch`, `flare_mla_decoding`, `flare_mla_decoding_dispatch`, `mla_prefill`, `mla_prefill_plan`, `mla_prefill_single_batch`, `mla_decoding`, `mla_decoding_single_batch`, `mla_splitk_reduce`, `mla_index_fp8`, `mla_decode_dispatch_scalars` | `V/L` | `V/L` | `E/L` | `-` | Generic GPU MLA docs exist. NVIDIA SM100 explicitly covers MLA prefill/decode; AMD gfx950 explicitly covers MLA prefill/decode. Public Apple MLA support is not established by the docs, but the local source-built MAX checkout now has Apple MLA prefill plus decode fallback sufficient for a short DeepSeek-V2-Lite-Chat M3 Max smoke. |
| NVIDIA Blackwell attention | SM100 `mha_1q`, `mla_decode_*`, `mla_prefill_*`, `softmax_warp`, `mma_warp`, `attention_utils` | `V/L` | `-` | `-` | `-` | Blackwell/B200-specific subtree. |
| NVIDIA Hopper attention | SM90 `attention`, `mha`; FlashAttention v3 package | `V/L` | `-` | `-` | `-` | Hopper/H100-specific subtree, MHA-focused in current public index. |
| AMD RDNA attention | `mha_prefill`, `mha_decode`, online `softmax`, RDNA Wave32 helpers | `-` | `V/L` | `-` | `-` | RDNA3+/gfx11xx/gfx12xx subtree. |
| AMD gfx950 attention | `mha_prefill`, `mha_prefill_v2`, `mha_decode`, `mha_decode_streaming`, `mla_prefill`, `mla_prefill_v2`, `mla_decode`, online softmax, MFMA helpers | `-` | `V/L` | `-` | `-` | MI355X/gfx950 subtree. |
| Apple Metal attention | `fa_prefill_apple`, `fa_prefill_apple_core`, `naive_fa_decode`; local patch adds `mla_prefill_apple`, `mla_prefill_apple_m3`, and `mla_decode_apple` | `-` | `-` | `V/L/E` | `-` | Apple page is M5/Metal-specific. Package docs still require conservative treatment for large MAX GenAI inference on Apple Silicon. Local source-built evidence adds M3-compatible MLA smoke coverage, but not production parity/performance evidence. |
| Linear algebra | `matmul`, `bmm`, `gemv`, `grouped_matmul`, `lora`, `matrix_band_part`, `packing`, `qr_factorization`, `transpose`, `accumulate`, `structuring` | `S/P` | `S/P` | `L/P` | `S/P` | Linalg index states CPU and GPU implementations. |
| FP4/FP8/MXFP4 linalg | `fp4_quantization`, `fp8_quantization`, `mxfp4_dequant`, `mxfp4_matmul_sm90`, `grouped_matmul_sm100*` | `L` | `P` | `-` | `-` | Current docs include H100 SM90 MXFP4 and SM100 grouped matmul modules. |
| Quantized matmul | `per_channel_grouped_4bit`, `qmatmul`, `qmatmul_gpu`, `qmatmul_k` | `S/P` | `S/P` | `P` | `S/P` | Split CPU/GPU coverage must be validated for the exact quant format. |
| KV cache | `paged_sparse_kv_index_remap`, `kv_cache`, `kv_cache_ragged`, `rope_split_store`, `types` | `S/P` | `S/P` | `P` | `P` | Used by higher-level attention APIs; sparse MLA remap is documented in the KV cache package. |
| RoPE and fused positional ops | `rope`, `fused_qk_rope`, `rope_split_store`, `rms_norm_rope_gpu` | `S/P` | `S/P` | `P` | `P` | Critical for transformer stage-host probes. |
| MoE routing | `group_limited_router_kernel`, `router_group_limited`, `single_group_router`, `single_group_router_eplb`, `eplb_remap`, `moe_create_indices`, `moe_create_indices_bucket_group_kernel` | `S/P` | `S/P` | `P` | `-` | Docs describe several functions as GPU launches/kernels. Fornax must probe router/expert layout on each target backend. |
| Multi-GPU communication | `allgather`, `allreduce`, `allreduce_residual_rmsnorm`, `broadcast`, `reducescatter`, `scatter`, `device_collective`, `device_query`, `lamport`, `sync`, `rms_norm_fp8` | `S/P` | `S/P` | `-` | `-` | MAX GPU collectives are not a substitute for Fornax cross-vendor activation/KV transport. |
| State-space models | `causal_conv1d`, `varlen_causal_conv1d`, `selective_scan`, `varlen_selective_scan`, `mamba2_ssd_scan`, `gated_delta`, `gated_delta_conv1d`, `rms_norm_fused_residual` and `*_ops` registrations | `P` | `P` | `P` | `P` | Public index lists kernels but does not publish a platform matrix. |
| Custom/extensibility | `custom`, `inplace_custom`, GraphCompiler kernel entry points, `builtin_kernels`, `structured_kernels`, `pipeline`, `shmem`, `nvml` | `P` | `P` | `P` | `P` | `nvml` is NVIDIA-specific. Custom op portability depends on implementation and target compiler path. |

## Fornax planning stance

1. Treat `S/P` rows as candidate reusable MAX operators, not closed evidence.
2. Treat `V/L` rows as promising only for the named vendor architecture.
3. Treat Apple Silicon as gated: Apple GPU kernels exist in the docs, and local
   source-built evidence now proves a short DeepSeek-V2-Lite-Chat smoke on M3
   Max, but packaged MAX support and Fornax production serving remain local-probe
   requirements.
4. For MoE target contracts, probe at least: dense matmul/GEMV, router top-k,
   `softmax`, RMSNorm, RoPE, KV cache write/read, MHA prefill/decode, MLA
   prefill/decode where applicable, and expert MLP throughput.
5. Do not change Fornax golden vectors based on this chart alone; only local
   measured/probed behavior can close planner/runtime gates.
