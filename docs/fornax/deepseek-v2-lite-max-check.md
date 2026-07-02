# DeepSeek-V2-Lite-Chat MAX check runbook

Source snapshot date: 2026-06-30; latest local verification: 2026-07-01.

This note answers whether we can build the planned Apple Silicon
`flare_mla_prefill` kernel and check `deepseek-ai/DeepSeek-V2-Lite-Chat` on
MAX.

## Current answer

Yes for a local short-generation smoke on this M3 Max. It is not yet a
production-quality Apple Silicon serving path.

Update from the upstream MAX checkout in `external/modular` on 2026-07-01:

| Check | Result |
|---|---|
| Source-built MAX CLI | Built with Bazel target `//max/python/max/_entrypoints:pipelines`; reproduced CLI reports `MAX 26.5.0.dev2026063006`. |
| Local model snapshot | Present at `/Users/abdulmuneer/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7`. |
| Standalone Apple MLA prefill smoke | Passed on M3 Max: `compute_capability: 3`, `non-finite outputs: 0`. |
| Standalone Apple MLA decode smoke | Passed on M3 Max with the correctness-first fallback: `compute_capability: 3`, `non-finite outputs: 0`. |
| Standalone Apple MoE indices smoke | Passed on M3 Max after adding a serial Apple fallback for expert bucketing. |
| Standalone Apple rank-2 gather smoke | Passed on M3 Max for both `int32` and `uint32` axis-0 indices. |
| Full `generate`, 1 token | Passed on M3 Max from the local snapshot path. Prompt size 14, output size 1, emitted `!`; model build/compile/init took 104.2s and TTFT was 343.7ms. |
| Full `generate`, 8 tokens | Passed on M3 Max from the local snapshot path. Prompt size 14, output size 8, emitted `!!!!!!!!`; token-generation throughput was about 4.03 tok/s in this tiny smoke. |
| Full serving | Not tested. |

Reproduction notes from the 2026-07-01 run:

- Running the source-built Bazel `pipelines` binary directly requires the Mojo
  runfiles environment that `bazel run` normally injects, especially
  `MODULAR_MOJO_MAX_IMPORT_PATH`, `MODULAR_MOJO_MAX_PACKAGE_ROOT`,
  `MODULAR_MOJO_MAX_DRIVER_PATH`, `MODULAR_MOJO_MAX_LLD_PATH`, and
  `MODULAR_MOJO_MAX_COMPILERRT_PATH`. Without those, graph construction fails
  while importing built-in kernel packages with `MAXG_addKernelPackage: failed
  to import kernels from ''`.
- `HF_HUB_OFFLINE=1` plus the repo id and `--trust-remote-code` failed before
  MAX graph build because the local Hugging Face snapshot contains weights,
  tokenizer, and `config.json`, but not `configuration_deepseek.py` or
  `modeling_deepseek.py`. Using the local snapshot path without
  `--trust-remote-code` let MAX use its built-in `DeepseekV2ForCausalLM`
  architecture and proceed to graph compile/generation.
- A fresh multi-target Bazel test rerun was interrupted after 339s in analysis;
  the evidence for the four standalone Apple smoke targets is from the existing
  Bazel test logs in the workspace-local output base, while the two full
  `generate` rows above were rerun in this session.

The old one-token blocker was the NVIDIA/AMD-only MLA decode gate:

```text
constraint failed: flareMLA_decoding currently only supports Nvidia and AMD GPUs.
```

The local upstream checkout now has a first Apple `flareMLA_decoding` route. It
is not the optimized split-K implementation; the split-K attempt compiled but
failed Metal pipeline creation on this M3 Max with an XPC interruption. The
active fallback calls the existing Apple `mha_gpu_naive` path over the full
DeepSeek MLA cache row and copies the latent/value portion into the MLA output.
A focused smoke test for that path passes on M3.

The next blocker was Apple MoE/gather graph support. The earlier prefill-only
compile failed in the MoE bucketing kernel with:

```text
constraint failed: Current compilation target does not support operation: vote.
```

That was cleared for the focused M3 smoke case by a serial Apple fallback in
`max/kernels/src/nn/moe.mojo`. The following full-model blocker was generic
rank-2 gather lowering in the MoE permutation path, which emitted Metal-invalid
double/floor instructions in kernels named like `gather_r2_w8_*` and
`gather_r2_w1_*`. That is now cleared by a direct flat GPU rank-2 axis-0 gather
path in `max/kernels/src/graph_compiler/builtin_kernels/gather_scatter.mojo`.

The final runtime blocker before the successful smoke was the host-side MLA
decode dispatch scalar helper, which still rejected Metal even though the Apple
decode fallback existed. `mla_decode_dispatch_scalars` now returns
`(batch_size, q_max_seq_len, 1)` for `ctx.api() == "metal"`, matching the
single-partition Apple fallback.

So the current Apple Silicon status is: the local Apple `flare_mla_prefill`
kernel exists and launches in isolation, the first Apple MLA decode fallback
exists and passes a focused smoke test, the first Apple MoE indices fallback
passes a focused smoke test, the rank-2 gather smoke passes, and
`DeepSeek-V2-Lite-Chat` can generate short output through MAX on this M3 Max.
This is still smoke evidence, not numerical parity, performance validation, or
an upstream-ready Apple serving claim.

The model itself is supported by current public MAX docs. The globally installed
environment still does not have a packaged `max` CLI, but today's run used the
source-built MAX CLI from the upstream checkout in `external/modular`:

| Check | Result |
|---|---|
| Repository | Fornax Python simulation/contracts plus upstream MAX checkout under `external/modular`. |
| Local `mojo` | `mojo 24.3.0 (9882e19d)`. |
| Local `modular` CLI | `modular 0.9.3 (020e342b)`. |
| Local `max` CLI | Not found in `PATH`. |
| Local Python `max` package | Not installed in the default `python3`. |
| Local hardware | Apple M3 Max, 40-core GPU, 128 GB unified memory. |
| Current MAX Apple serving status | Public docs say large GenAI model inference via MAX is not yet available on Apple Silicon. |
| DeepSeekV2 CPU fallback | Upstream MAX `DeepseekV2Model` raises if the selected device is CPU. |

So the current local result is no longer blocked by basic environment setup or
the first full-model Apple GPU compile/runtime gates. It remains blocked for
credible serving by correctness, performance, and long-run validation work.

## Model facts

| Field | Value |
|---|---|
| Hugging Face repo | `deepseek-ai/DeepSeek-V2-Lite-Chat` |
| Architecture | `DeepseekV2ForCausalLM` |
| MAX supported architecture | Listed in current MAX supported-models docs. |
| MAX encoding | `bfloat16` |
| MAX multi-GPU | Supported according to the model table. |
| HF parameter count | 15,706,484,224 BF16 parameters. |
| HF stored weights | 31,413,626,576 bytes across 4 safetensor shards. |
| Model-card hardware note | BF16 inference requires one 40 GB GPU. |
| Attention shape | `qk_nope_head_dim = 128`, `qk_rope_head_dim = 64`, `v_head_dim = 128`. |
| MoE shape | 64 routed experts, 2 shared experts, 6 experts per token. |
| Context fields | HF config has `max_position_embeddings = 163840`; model card describes 32k for Lite. |

These dimensions are exactly why `flare_mla_prefill` matters: the attention
score is built from the non-RoPE 128-dim part plus the RoPE 64-dim part, then
accumulates into the 128-dim value head.

## Validation tracks and what changed

There are still two validation tracks, but the Apple track is no longer only a
design task in the local source tree:

| Track | Purpose | Status |
|---|---|---|
| Packaged MAX DeepSeekV2 | Verify `DeepSeek-V2-Lite-Chat` runs on officially supported packaged MAX targets. | NVIDIA/AMD remains the expected supported route; packaged Apple nightly previously failed on MLA/MoE backend gates. |
| Local source-built Apple path | Add the missing Apple MLA/MoE/gather backend pieces in `external/modular` and run the model on M3 Max. | Short `max generate` smoke passed for 1 and 8 output tokens with BF16 weights on `gpu[0]`. |

Current packaged MAX can support the DeepSeekV2 architecture on supported GPU
targets without the local Apple patch. The M3 Max result here depends on the
source-built MAX checkout in `external/modular`; it should not be cited as
official packaged MAX support, production serving support, or distributed
Fornax support.

## Supported MAX check on NVIDIA/AMD

Run this on a Linux host with a current Modular/MAX install and a GPU with enough
memory. The first smoke test should use a short context to control KV cache
memory.

```bash
max list
```

Confirm `DeepseekV2ForCausalLM` appears.

```bash
max generate \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --devices gpu:0 \
  --quantization-encoding bfloat16 \
  --max-length 4096 \
  --max-batch-size 1 \
  --max-batch-total-tokens 4096 \
  --max-new-tokens 64 \
  --top-k 1 \
  --temperature 0 \
  --trust-remote-code \
  --prompt "User: Write a compact Python function that returns the nth Fibonacci number.\n\nAssistant:"
```

If that compiles and generates text, run the same model as a server:

```bash
max serve \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --devices gpu:0 \
  --quantization-encoding bfloat16 \
  --max-length 4096 \
  --max-batch-size 1 \
  --max-batch-total-tokens 4096 \
  --trust-remote-code
```

Then query the OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite-Chat","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":64,"temperature":0}'
```

Expected pass criteria:

| Check | Pass condition |
|---|---|
| Architecture registration | `max list` includes `DeepseekV2ForCausalLM`. |
| Weight load | MAX downloads or resolves all 4 safetensor shards. |
| Compile | Graph builds without unsupported op or dtype failures. |
| Inference | `max generate` returns non-empty text. |
| Serving | `/v1/chat/completions` returns a valid completion. |
| Kernel evidence | Logs or profiler show the MLA path selected for prefill/decode where expected. |

## Apple Silicon path

The local Apple path now has enough upstream MAX patching to run a short
model-level smoke. The status of the backend work is:

| Step | Status |
|---|---|
| Apple MLA prefill source | Implemented locally in `max/kernels/src/nn/attention/gpu/apple/mla_prefill.mojo`; includes an M3-compatible path and an M5-style path selection. |
| MLA dispatch | `flare_mla_prefill_dispatch` accepts Apple GPU targets and routes through the Apple prefill launcher for the supported BF16/FP16 shape family. |
| DeepSeek-Lite shape gate | Local smoke targets the DeepSeek-V2-Lite MLA dimensions: 128 non-RoPE + 64 RoPE -> 128 value output. |
| Apple MLA decode | Implemented as a correctness-first fallback using Apple `mha_gpu_naive` over the DeepSeek MLA cache row, then copying the latent/value prefix. |
| MoE graph support | Apple MoE bucketing has a serial correctness-first fallback to avoid unsupported Metal `vote`; rank-2 axis-0 gather has a direct Apple GPU path. |
| Dispatch scalar helper | `mla_decode_dispatch_scalars` returns one partition for Metal, matching the Apple decode fallback. |
| Standalone smoke tests | Existing Bazel logs show prefill, decode, MoE indices, and rank-2 gather smokes passed on M3 Max. |
| Model run | Fresh local `generate` runs passed for 1-token and 8-token outputs; serving, batching, longer contexts, and numerical parity remain open. |

This local M3 Max result is useful because it proves the full model can compile
and emit tokens through the patched Apple backend. It remains intentionally
narrow: a future production path still needs chip-aware dispatch, hard numerical
tests, optimized decode/MoE implementations, and graceful fallback when shape,
dtype, or hardware gates fail.

## Local blockers to clear

| Blocker | What would clear it |
|---|---|
| Full graph smoke is correctness-only | Add logit/reference checks for prefill, decode, MoE routing, and repeated-token generation. |
| Apple MLA decode fallback is smoke-tested only | Add a numerical correctness comparison and replace the slow generic-MHA fallback with an optimized Apple implementation. |
| Apple MoE bucketing fallback is correctness-first | Replace the one-thread serial fallback with a parallel Apple implementation. |
| Apple MLA prefill is smoke-tested only | Add a numerical correctness test against a reference path. |
| No packaged `max` CLI in PATH | Optional: install a current Modular/MAX package; source-built Bazel CLI works for kernel development. |
| Packaged MAX still lacks this local patch | Rebuild/package from the patched MAX checkout or wait for equivalent upstream support; do not expect `/tmp/fornax-max-smoke` packaged MAX to include these changes. |
| Full Apple serving path incomplete | Test `max serve`, longer contexts, larger generation counts, batching, memory pressure, and failure recovery before claiming serving support. |

## Source links

| Source | Use |
|---|---|
| `https://docs.modular.com/max/models.md` | Confirms `DeepseekV2ForCausalLM` and `DeepSeek-V2-Lite-Chat` are supported in MAX with BF16. |
| `https://docs.modular.com/max/packages.md` | Platform requirements and Apple Silicon serving caveat. |
| `https://docs.modular.com/max/cli/generate.md` | `max generate` command options. |
| `https://docs.modular.com/max/cli/serve.md` | `max serve` command options. |
| `https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat` | Model card, hardware note, license, and local inference guidance. |
| `https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/raw/main/config.json` | Exact model dimensions and architecture fields. |
| `https://raw.githubusercontent.com/modular/modular/main/max/python/max/pipelines/architectures/deepseekV2/arch.py` | MAX DeepSeekV2 architecture registration. |
| `https://raw.githubusercontent.com/modular/modular/main/max/python/max/pipelines/architectures/deepseekV2/model.py` | GPU-only guard in the MAX DeepSeekV2 model path. |
| `docs/fornax/apple-flare-mla-prefill-port.md` | Apple `flare_mla_prefill` porting plan. |
