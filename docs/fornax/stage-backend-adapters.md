# Stage Backend adapter guide

Status: public Stage Backend API v2; functional smoke only  
Physical MAX adapters and G2 evidence: open

This guide is for implementers connecting a real stage executor to Fornax. The
public SDK is `fornax.backends`. It deliberately has no default physical backend
and never falls back from a failed physical adapter to reference or simulation.

## Text engine boundary

The small public facade preserves the harness-agnostic string boundary:

```python
from fornax import Engine

engine = Engine(my_explicit_model_generator)
text = engine.generate("Explain sparse MoE routing.")
```

`my_explicit_model_generator` must implement `str -> str`. Fornax passes the
exact input through and rejects a non-string result. The package does not provide
an echo generator or pretend that the activation-tensor Engine v0 already owns
tokenization, sampling, and detokenization. A bundled physical generator and
`fornax serve` remain product work.

## Adapter factory

A physical worker is selected by an importable top-level `module:factory`:

```python
from typing import Any
from fornax.backends import StageExecutable

def create_backend(options: dict[str, Any]) -> StageExecutable:
    return MyMaxStageBackend(options)
```

The factory receives one JSON-serializable options object. It must return a
`StageExecutable` implementing:

```text
capabilities, load, health, execute, cancel, release, drain, unload
```

The module must be importable in a spawned worker process; closures and
interactive-only factories are unsupported. `health`, `cancel`, and `release`
may race with execution, so adapters must make their own state concurrency-safe.
`release(handle, request_id)` must reject an in-flight request, remove all
request-owned KV/cancellation/execution/idempotency state after terminal work,
and be idempotent when that state is already absent. Transform data that is
shared across requests may remain only within declared count and byte bounds.

The reference and simulated backends additionally implement bounded T0/T1
lifecycle fencing. Idle request state expires opportunistically on backend
operations, each execution has an internal lease, and explicit or automatic
release writes a count- and time-bounded request tombstone. The tombstone lives
in the loaded backend, so it fences request-ID reuse when an FNX1 data channel
disconnects and reconnects to the same worker process. It is not replicated and
does not survive worker-process restart. A full tombstone table fails release or
expiry closed with `TOMBSTONE_CAPACITY` and retains both existing fences and
live request state; it never evicts a live fence to admit another. Callers must
provision the count for the arrival rate and retention window.
`sweep_expired(handle)` is a reference test hook, not a new required
`StageExecutable` method.

Physical adapters do not inherit those semantics merely by implementing the v2
method names. They must bind expiry to native KV/state ownership, cooperatively
cancel or fence expired execution, and produce their own restart policy and
evidence before claiming equivalent behavior.

## Native buffer staging seam

`TensorBufferAdapter`, `ImportedTensorBuffer`, and
`PythonTensorBufferAdapter` make logical-to-native imports explicit without
changing the `StageExecutable` protocol. An import records the expected
descriptor, byte count, owner, validation, and whether a copy occurred. Export
revalidates the descriptor and bytes; release returns ownership to the adapter.

The reference backend stages both input and output through the bounded Python
adapter and then executes the existing deterministic Python oracle. This is a
correctness and accounting seam, not a zero-copy or performance claim. A MAX
adapter may place an opaque runtime/device handle in `ImportedTensorBuffer`, but
must report copies honestly and must not expose a logical `Tensor` until output
descriptor, byte count, and finite-value validation pass.

## Capability truth

`capabilities()` runs before `load(manifest)`. Backend name, exact build, device
identity, memory, dtypes, operations, quantizations, ABI versions, and frame
limits must be facts discovered or bound by the adapter. Do not copy requested
values from the manifest. Fornax records requested and observed values and fails
startup with `CAPABILITY_MISMATCH` when they disagree.

The supported engine constructor accepts explicit specs:

```python
from fornax.backends import StageBackendSpec
from fornax.engine_v0 import start_stage_engine

spec = StageBackendSpec(
    kind="max",
    factory="my_package.fornax_backend:create_backend",
    options={"device": "gpu:0", "build_root": "/opt/my-max-build"},
)

workers, channels, orchestrator = start_stage_engine(manifests, (spec, spec2))
```

An import, protocol, capability, or load failure stops startup. It cannot select
`simulated-max` implicitly.

## Functional conformance smoke

Given a stage manifest JSON and optional factory-options JSON:

```bash
fornax runtime backend-conformance \
  --factory my_package.fornax_backend:create_backend \
  --manifest stage-0.manifest.json \
  --options stage-0.backend-options.json \
  --out stage-0.backend-conformance.json
```

The 12-check command covers pre-load capability attestation, handle and health
identity, one valid execute, output contract, retained-result replay, deadline,
cancellation, explicit request-state release, positive declared retention
limits with post-release counters, drain, and unload. It exits `0` on pass, `1`
on a backend-contract failure, and `2` on invalid command input.

The generic smoke does not pressure every declared cap. Separate reference
regressions exercise live-request, result-count/result-byte, transform-count/
transform-byte, tombstone/expiry, native-staging, and event-history limits. A
fast many-unique-request run is available as:

```bash
python3 -m fornax.lifecycle_pressure \
  --min-iterations 1000 \
  --out /tmp/fornax-lifecycle-pressure.json
```

The runnable 30-minute T0/T1 evidence mode is:

```bash
python3 -m fornax.lifecycle_pressure \
  --wall-seconds 1800 \
  --min-iterations 1800 \
  --out /tmp/fornax-lifecycle-pressure-30m.json
```

It uses a deterministic logical clock for expiry and a monotonic wall clock for
sustained active duration. The runner also fails a sustained result when the
civil-clock gap between progress samples exceeds `--max-pause-seconds` (default
5), so host sleep or process suspension cannot masquerade as an uninterrupted
soak. Its backend counters are contract evidence; its RSS values are diagnostics.
A physical adapter still needs equivalent stress plus native KV/buffer/RSS
evidence before a memory-stability claim.

EV-016 records an earlier 113,718-request candidate under the bounded lifecycle
contract. It is useful churn evidence, but it predates the continuity check and
reports `current_contract_authority=false`; see the
[evidence register](evidence-register.md).

The report is `functional_contract_smoke` evidence with `closes_g2=false`. It
does not establish numerical parity, throughput, latency, multi-node behavior,
cross-vendor correctness, or supported-platform status. Those require the T2/T3
corpus and physical G2 packet.

## Versioning

- `ENGINE_API_VERSION = 1` preserves `Engine.generate(str) -> str`, explicit
  generator selection, exact input pass-through, and error behavior.
- `STAGE_BACKEND_API_VERSION = 2` preserves the synchronous method names,
  arguments, return dataclasses, required `release` operation, retention fields
  in `StageHealth`, and `factory(options: dict)` shape.
- API v1 adapters are intentionally incompatible because they cannot prove that
  request-owned state is released. They must implement `release` and report
  bounded-retention health before they can pass the v2 smoke.
- Adding optional capability fields with defaults may be minor-compatible.
- The lifecycle/native-buffer fields added to `StageHealth` have defaults and
  are observational extensions. `sweep_expired` and `TensorBufferAdapter` are
  optional helpers; they do not add required v2 backend methods.
- Removing fields, adding required methods, changing KV/at-most-once/release semantics,
  or changing the factory shape requires a new Stage Backend API major.
- The Python Stage Backend API version and the `FNX1` wire ABI version are
  independent.

The normative lifecycle and wire rules remain in
[`stage-runtime-and-wire-abi.md`](stage-runtime-and-wire-abi.md).
