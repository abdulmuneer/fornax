# Consumer MoE qualification recipes

Status: C1 contracted recipe portfolio; no physical support claim

Evidence snapshot: 2026-07-25

These recipes turn a deliberately small model-and-platform cohort into
repeatable bring-up work. They are meant to make hardware time decisive: an
operator can identify the exact machine, verify an immutable model snapshot,
run the required probes, and collect the evidence needed to promote or reject a
configuration.

They do **not** make Fornax run a distributed physical model today. The physical
`MaxStageBackend`, cross-vendor generation, numerical parity, and G2/G3 evidence
remain open. A rendered recipe is `contract_validated`, not `supported`.

## Selected model cohort

“Popular” is a dated adoption proxy, not a user count. The selection uses
Hugging Face's rolling downloads-last-month field as captured on 2026-07-25,
then avoids spending two of three architecture slots on the two sizes of the
same gpt-oss family.

| Recipe model | Adoption proxy | Why it is in the first cohort |
|---|---:|---|
| [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 3,126,164 downloads last month | First bring-up target: 30.5B total / 3.3B active, conventional text GQA-MoE, BF16 checkpoint, and small enough to isolate correctness on a 128GB Mac or suitable NVIDIA island. |
| [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) | 4,466,763 downloads last month | Native MXFP4 MoE target: about 117B total / 5.1B active and publisher-stated to fit an 80GB-class accelerator. It adds mandatory Harmony formatting and mixed-precision capacity checks. |
| [`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1) | 9,339,928 downloads last month | Frontier-capacity stretch: 671B total / 37B active, FP8 weights, MLA, shared experts, and a representation that exceeds the 512GB Apple target. |

The literal download ranking also includes
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b). It is retained
as a useful boot-smoke companion, but it is not one of the three qualification
families. Hugging Face downloads include automated and provider traffic, so the
numbers only prioritize engineering work. See Hugging Face's
[download-counting methodology](https://huggingface.co/docs/hub/models-download-stats);
the metric is not a unique-user count.

All three are permissively licensed open-weight releases. “Open source model”
has no universally accepted scope; the catalog records the exact license and
does not infer training-data or full-development openness.

## Selected platform cohort

| Recipe platform | Binding hardware identity | Vendor facts used only for C0/C1 planning |
|---|---|---|
| `apple-m3-max-128` | Observed Apple M3 Max chip and at least the marketed 128GB memory threshold | The selected nominal configuration is the 16-inch MacBook Pro with 16-core CPU and 40-core GPU; 400GB/s unified-memory bandwidth; Thunderbolt 4. |
| `apple-m5-max-128` | Observed Apple M5 Max chip and at least the marketed 128GB memory threshold | The selected nominal configuration is the 16-inch MacBook Pro with 18-core CPU and 40-core GPU; 614GB/s unified-memory bandwidth; Thunderbolt 5. |
| `apple-m3-ultra-512` | Observed Apple M3 Ultra chip and at least the marketed 512GB memory threshold | The selected nominal configuration is Mac Studio with 32-core CPU and 80-core GPU; 819GB/s unified-memory bandwidth, built-in 10GbE, Thunderbolt 5. |
| `nvidia-h100-sxm-80gb` | Exact `nvidia-smi` name `NVIDIA H100 80GB HBM3` and at least 80GB nominal memory | 3.35TB/s HBM bandwidth, 900GB/s NVLink, up to 700W configurable TDP; baseboard/topology still requires evidence. |
| `nvidia-rtx-4090-24gb` | Exact `nvidia-smi` name `NVIDIA GeForce RTX 4090` and at least 24GB nominal memory | 1,008GB/s memory bandwidth, PCIe Gen4, no NVLink, 450W TGP. |
| `nvidia-rtx-5090-32gb` | Exact `nvidia-smi` name `NVIDIA GeForce RTX 5090` and at least 32GB nominal memory | 1,792GB/s memory bandwidth, PCIe Gen5, no NVLink, 575W TGP. |

The requested “M5 512” is corrected rather than silently accepted. Apple lists
M5 Max up to 128GB; its marketed 512GB unified-memory Mac is the M3 Ultra Mac
Studio. A machine or profile claiming `M5` plus `512GB unified memory` must fail
identity validation until Apple publishes such a configuration. A 512GB SSD is
storage and does not provide model-residency capacity.

Primary specification sources:

- [M3 Max MacBook Pro](https://support.apple.com/en-us/117737)
- [M5 Max architecture and 128GB ceiling](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- [M3 Ultra and 512GB unified memory](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
- [Mac Studio specifications](https://www.apple.com/mac-studio/specs/)
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)
- [GeForce RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
- [GeForce RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
- [MAX supported model architectures and encodings](https://docs.modular.com/max/models/)
- [`max list --json` registry contract](https://docs.modular.com/max/cli/list/)
- [Modular package/platform support, including the current Apple large-GenAI limitation](https://docs.modular.com/max/packages/)
- [Mojo numeric types](https://docs.modular.com/mojo/reference/mojo-numeric-types/)
- [Thunderbolt 5 data-link modes](https://www.intel.com/content/www/us/en/architecture-and-technology/thunderbolt/overview.html)

`H100` is bound to H100 SXM 80GB because H100 PCIe, H100 SXM, and H100 NVL
have materially different memory and topology. Add separate profiles rather
than weakening the identity rule.

Apple requires an explicit precision gate. Mojo currently documents BF16 as
unavailable on Apple Silicon, so the Apple profiles advertise FP16/FP32
hardware and FP16 as the runtime candidate—not BF16. A BF16, FP8, or MXFP4
checkpoint label describes stored weights; it does not prove that MAX can
decode, convert, or execute that representation on the Apple GPU. Every Apple
recipe therefore starts with `conversion_or_custom_kernel_required=true` until
the exact conversion/kernel path passes physical numerical tests.

Thunderbolt 5 capacity is recorded as the symmetric 80 Gb/s data mode
(10 GB/s theoretical per direction). The marketed 120 Gb/s Bandwidth Boost is
an asymmetric, display-oriented transmit mode and is not used as a Fornax
transport-capacity assumption.

## The 18-recipe matrix

The catalog composes every selected model with every selected platform. It does
not duplicate 18 hand-maintained manifests; a deterministic lock records the
chosen model, platform, quantization representation, unit count, source hashes,
and recipe hash.

| Model \ platform | M3 Max 128 | M5 Max 128 | M3 Ultra 512 | H100 SXM 80 | RTX 4090 24 | RTX 5090 32 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B BF16 | C1 | C1 | C1 | C1 | C1 | C1 |
| gpt-oss-120b MXFP4 | C1 | C1 | C1 | C1 | C1 | C1 |
| DeepSeek-R1 FP8 | C1 | C1 | C1 | C1 | C1 | C1 |

The default capacity-only lower bounds are:

| Model \ platform | M3 Max 128 | M5 Max 128 | M3 Ultra 512 | H100 SXM 80 | RTX 4090 24 | RTX 5090 32 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B BF16 | 1 | 1 | 1 | 1 | 4 | 3 |
| gpt-oss-120b MXFP4 | 1 | 1 | 1 | 1 | 4 | 3 |
| DeepSeek-R1 FP8 | 8 | 8 | 2 | 11 | 35 | 26 |

These counts use the pinned checkpoint bytes, 10% static runtime headroom, and
a conservative 75% Apple / 85% NVIDIA nominal-memory allowance. They are not
recommended fleets: one unit remains one distinct memory pool, and the
arithmetic excludes KV cache, measured workspaces and buffers, topology,
correctness, performance, thermals, and power. `recipe render --units N`
refuses values below the corresponding bound.

Every cell begins with these claims set to false:

- physical device validation;
- numerical parity;
- physical `StageBackend` conformance;
- cross-node interoperability;
- G2 or G3 closure;
- product support.

## Use

List and validate the packaged catalog:

```bash
python3 -m fornax recipe list
python3 -m fornax recipe validate
python3 -m fornax test qualification-recipes
```

Materialize one content-addressed operator packet:

```bash
python3 -m fornax recipe render \
  --model qwen3-30b-a3b \
  --platform apple-m3-max-128 \
  --units 1 \
  --out-dir /tmp/fornax-qwen3-m3max
```

The packet contains:

- `recipe-lock.json`: canonical inputs and SHA-256 recipe identity;
- `commands.json`: argv arrays for the required probes, with no shell
  interpolation;
- `RUNBOOK.md`: ordered bring-up and promotion gates;
- `bundle-manifest.json`: exact byte sizes and SHA-256 digests for those three
  managed files, published last after atomic file replacement.

The bundle manifest is unsigned. It detects incomplete or changed managed files
and can be compared with an out-of-band digest, but it is not publisher
authentication. Verify both packet integrity and exact reproducibility from the
currently installed catalog:

```bash
python3 -m fornax recipe verify \
  --packet-dir /tmp/fornax-qwen3-m3max \
  --expected-bundle-sha256 sha256:<64-lowercase-hex>
```

The default verifier rejects unmanaged entries. After collecting host/model
evidence under the packet directory, add `--allow-unmanaged-evidence`; those
extra entries are reported and explicitly remain outside the managed-file
integrity scope.

On the target machine, collect exact host and local-model evidence before
running a model:

```bash
python3 -m fornax recipe probe-host \
  --platform apple-m3-max-128 \
  --out /tmp/fornax-qwen3-m3max/host-identity.json

python3 -m fornax recipe inspect-model \
  --model qwen3-30b-a3b \
  --model-dir /models/Qwen3-30B-A3B \
  --out /tmp/fornax-qwen3-m3max/model-artifacts.json

python3 -m fornax recipe probe-runtime \
  --model qwen3-30b-a3b \
  --platform apple-m3-max-128 \
  --out /tmp/fornax-qwen3-m3max/max-runtime.json
```

`probe-host` fails closed on the wrong catalog chip/GPU identity or an
insufficient nominal-memory observation. `inspect-model` fails closed on
unproved or wrong local revision provenance, architecture mismatch, missing
tokenizer/template material, incomplete shards, byte mismatches, or unresolved
remote-code review. Neither command downloads a model.

Live collectors label their provenance explicitly. Injected fixture runners are
synthetic test evidence and cannot close a physical bring-up claim. Host and MAX
reports also record content hashes for the resolved `nvidia-smi` and MAX argv
entry-point executables. Those hashes expose drift and PATH substitution; they
remain unauthenticated local evidence and still require operator custody.

`inspect-model` streams and hashes every selected checkpoint file. That is
about 61 GB for Qwen, 65 GB for the selected root-only gpt-oss representation,
and 689 GB for DeepSeek-R1, so operators must budget local I/O time. Reports
carry the current catalog and model-profile hashes and strict inspection
requires complete pinned SHA-256 coverage.

`probe-runtime` safely captures `max --version` and `max list --json`, then
requires the exact catalog pair:

- Qwen: `Qwen3MoeForCausalLM` / `bfloat16`;
- gpt-oss: `GptOssForCausalLM` / `float4_e2m1fnx2`;
- DeepSeek: `DeepseekV3ForCausalLM` / `float8_e4m3fn`.

A registry match is not device compatibility. OS, driver, Apple
decode/conversion, physical kernels, model load, and parity still require
separate evidence.

For a one-unit Apple recipe, `commands.json` invokes
`recipe run-apple-single`. The wrapper requires the prior model-artifact,
host-identity, and MAX-registry reports; freshly re-inspects the local
checkpoint; and freshly observes the exact Apple chip and unified-memory
threshold. It binds the raw report hashes plus the current
`system_profiler`, `sysctl`, `sw_vers`, and MAX executable hashes. It then runs
the exact catalog repository from the resolved local model directory, with
downloads disabled, the catalog encoding, Apple `gpu`, and the exact sentinel
prompt. The smoke artifact is first captured in a private temporary directory,
validated, hashed, and embedded in the final envelope.

This is deliberately a future/custom-runtime recipe. Modular currently
documents Apple M1–M5 Mojo GPU programming, but also states that large GenAI
model inference through MAX is not yet available on Apple silicon. Therefore
`--max-command` is mandatory, stock upstream MAX is expected to fail today, and
a registry match must not be described as publisher-supported Apple inference.
The recipe is code prepared for a future or Fornax-custom capable build, not a
claim that the missing upstream path already exists.

For physical evidence, `--max-command` must identify one direct executable.
The probe and launch canonicalize it to the resolved, hashed absolute path;
multi-part wrappers such as `pixi run max` are rejected. The wrapper re-hashes
that executable, the model manifest, the model-directory inode, the host
identity, and the Apple collector executables immediately before and after
generation. Any drift keeps the bring-up claim false. The final envelope is
fsynced and atomically published without replacement through a retained
no-follow descriptor chain; every parent component must be a real directory
(not a symlink). Choose a new `--out` path for every attempt.

The Apple wrapper does not trust the smoke's hardware summary by itself. It
cross-checks the independently collected chip, unified-memory bytes, machine
name, and model identifier where those fields are available, and requires the
MAX version observed during generation to match the fresh registry probe.
These are unauthenticated local bindings, not proof of custody or support.

For example, after the three preflights pass on a one-M3-Max Qwen recipe:

```bash
python3 -m fornax recipe run-apple-single \
  --model qwen3-30b-a3b \
  --platform apple-m3-max-128 \
  --model-dir /models/Qwen3-30B-A3B \
  --model-artifact-report /evidence/model-artifacts.json \
  --host-report /evidence/hosts/m3-max-0/host-identity.json \
  --runtime-report /evidence/hosts/m3-max-0/max-runtime.json \
  --max-command /opt/fornax/bin/max \
  --out /evidence/qwen-m3-max-single-bringup.json
```

For a one-unit NVIDIA recipe, `commands.json` next invokes
`recipe run-nvidia-single`. That wrapper uses direct argv (never a shell),
requires the prior model-artifact, host-identity, and MAX-registry reports,
freshly re-inspects the local checkpoint, binds all catalog/profile and raw
evidence hashes, and treats the requested `gpu:<index>` strictly as a physical
`nvidia-smi` selector. The wrapper resolves that row's GPU UUID in both the
recorded and fresh host reports, requires the UUID binding to agree, exports
`CUDA_VISIBLE_DEVICES=<GPU-UUID>`, and launches MAX on the resulting visible
ordinal `gpu:0`. It also verifies that the MAX and `nvidia-smi` executable
hashes have not changed, then captures byte-bounded MAX
version/stdout/stderr evidence and validates its own canonical integrity.

The generated-text detector accepts machine-readable or explicit framing. For
the native MAX CLI stream, the recipe uses an exact sentinel prompt and requires
the pre-metrics generated segment to equal that sentinel with a positive
`Output size`; compiler logs, prompt echoes, residual diagnostics, injected
runners, and unframed text fail closed. A pass may set only bounded
single-platform bring-up; parity, distributed execution, G2/G3, and product
support stay false.

For example, after the three preflights pass on a one-H100 Qwen recipe:

```bash
python3 -m fornax recipe run-nvidia-single \
  --model qwen3-30b-a3b \
  --platform nvidia-h100-sxm-80gb \
  --model-dir /models/Qwen3-30B-A3B \
  --model-artifact-report /evidence/model-artifacts.json \
  --host-report /evidence/hosts/h100-0/host-identity.json \
  --runtime-report /evidence/hosts/h100-0/max-runtime.json \
  --device gpu:0 \
  --max-command max \
  --out /evidence/qwen-h100-single-bringup.json
```

More precisely, `probe-host` binds the observed chip/GPU name and nominal memory
threshold. It records, but does not by itself qualify, the exact chassis,
order-code/core configuration, OS, driver, MAX/Mojo build, topology, or
allocatable runtime memory. Use `--units N` only for a same-host homogeneous
NVIDIA count; run it once per Mac or multi-host NVIDIA worker.

Identity fields ending in `_gb` use the vendor-marketed decimal threshold for
matching OS-reported bytes. Capacity planning deliberately normalizes the same
marketed numeral as nominal GiB before applying the conservative usable-memory
allowance. Neither convention substitutes for measured allocatable memory.

For any recipe whose capacity lower bound is greater than one, the generated
packet intentionally emits no full-model `max generate` command. It emits
per-host identity/runtime probes and a fail-closed G2-readiness command only.
The physical topology, host count, transport, placement, and real
`StageBackend` must be bound before Fornax can construct a legitimate
capacity-spanning launch.

DeepSeek-R1's pinned repository includes Python model code. Its inspection
therefore remains blocked until an operator supplies an explicit
model-and-revision-bound review:

```json
{
  "schema_version": 1,
  "record_kind": "fornax_remote_code_review",
  "review_id": "review-ticket-or-evidence-id",
  "model_id": "deepseek-r1",
  "revision": "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
  "decision": "allow_pinned_sha256",
  "files": {
    "configuration_deepseek.py": "sha256:<64-lowercase-hex>",
    "modeling_deepseek.py": "sha256:<64-lowercase-hex>"
  }
}
```

Record the exact bytes through a separately transported digest, then pass both:

```bash
python3 -m fornax recipe inspect-model \
  --model deepseek-r1 \
  --model-dir /models/DeepSeek-R1 \
  --remote-code-review REVIEW.json \
  --expected-remote-code-review-sha256 sha256:<64-lowercase-hex> \
  --out /tmp/deepseek-model-artifacts.json
```

The command checks the review's model, revision, safe paths, code hashes, and
out-of-band file digest. This binds the acknowledged allowlist bytes but is not
a reviewer signature or publisher authentication.

## Promotion sequence

| Level | Required result | Still not established |
|---|---|---|
| C1 Contracted | Catalog validation, deterministic lock, T0/T1 recipe tests | Any physical compatibility or performance |
| C2 Device-validated | Exact identity and artifacts; required MAX architecture/encoding; operator probes; real model bring-up; layer/logit parity on the named build | Cross-node serving |
| C3 Interoperable | Real `StageBackend`; 2–3 node generation; transport, cancellation, faults, and calibration within the G2 bound | Frontier-capacity target |
| C4 Configuration-certified | Exact fleet/model/workload passes capacity, headroom, throughput, security, and sustained T4 checks | Product support |
| C5 Product-supported | Fresh install, upgrade/rollback, monitoring, support window, rights, and G5 approval | Nothing beyond the stated support envelope |

Generated text is bring-up evidence, not parity. Capacity arithmetic is a
planning lower bound, not a run guarantee. Datasheet bandwidth is not measured
payload bandwidth. Apple unified memory is one shared pool, and RTX cards do not
pool memory through NVLink.

## Mandatory per-recipe gates

1. Match the catalog chip/GPU identity and nominal memory, then separately pin
   and qualify the exact chassis/order code, OS, driver, MAX/Mojo build, and
   measured allocatable memory.
2. Verify the immutable model revision, config, tokenizer/template, selected
   representation, every selected file hash, and all local shards.
3. Confirm `max list --json` advertises the exact architecture and encoding on
   the pinned build; then separately prove that device-specific kernels exist.
4. Run FP16/FP32 or the selected, physically supported format for matmul,
   attention prefill/decode, router
   top-k, expert MLP/grouped GEMM, gather/scatter, KV, sampling, and buffer
   probes.
5. Run the model-specific semantics corpus:
   Qwen thinking modes, gpt-oss Harmony channels/tools, or DeepSeek reasoning,
   MLA, shared-expert, and MTP policy.
6. Compare boundary activations, selected layers, logits, and greedy tokens to
   the accepted reference path.
7. Conform a real physical `StageBackend`; simulation fallback is forbidden.
8. Run the actual topology through G2/G3 correctness, fault, calibration,
   concurrency, memory-headroom, and sustained-operation gates.

The first implementation order is Qwen BF16, gpt-oss MXFP4, then DeepSeek-R1
FP8. A failure can demote a role or configuration; it must not be relabeled as
support.
