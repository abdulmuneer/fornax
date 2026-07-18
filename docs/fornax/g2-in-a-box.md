# G2-in-a-box operator workflow

Status: runnable readiness workflow; physical evidence remains open  
Authority: `project-plan-v4.md` G2 and `two-node-max-validation-plan.md` V1-V10

## Purpose

`fornax program g2-validate` turns the two-node plan into one fail-closed run. It
verifies the current working tree's MAX lineage mechanism, records the exact
Fornax source state, executes
the current T0/T1 prerequisites, and either runs or explicitly marks every
physical V6-V10 step `NOT_RUN` or `BLOCKED`.

The command produces a durable JSON evidence packet, a human summary, raw
stdout/stderr, copies of the immutable inputs, hashes for physical artifacts,
and a bundle-wide SHA-256 manifest. It does not make a physical claim from
simulation, a same-host proxy, a zero exit code, or a hand-written `passed`
field.

## Readiness run without hardware

Run from a Fornax checkout:

```bash
python3 -m fornax program g2-validate \
  --out-dir evidence/g2-readiness-YYYYMMDD-HHMMSS
```

This command deliberately exits `1` while physical evidence is absent. The
bundle is still useful: the MAX root pin and T0/T1 commands are checked, and all
six physical steps have an explicit blocker. Never use an existing output
directory; the runner refuses to overwrite evidence.

The default T0/T1 packet covers:

- V1 planner memory and greater-than-six-node regressions;
- V2 FNX1 malformed/valid frames, FNX2 ragged two-worker golden, and
  runtime-format goldens;
- V3 reference stage execution;
- V4 reference versus simulated-MAX parity and injected failures;
- V5 network contracts and two independent loopback workers.

Loopback V5 opens localhost sockets. A restricted sandbox that denies socket
binding will correctly fail that prerequisite; rerun it in the approved test
environment instead of waiving it.

## Physical run

Physical commands are never inferred or started by default. After preparing and
reviewing a concrete run manifest:

```bash
python3 -m fornax program g2-validate \
  --out-dir /durable/fornax-evidence/g2-YYYYMMDD-HHMMSS \
  --run-manifest /approved/fornax-g2-run.json \
  --run-physical
```

The runner uses `subprocess` argument arrays with no local shell. An `ssh`,
orchestration, copy-back, or lab wrapper may be the executable in that array,
but it must leave the required result JSON in the local step directory. The
runner sets these non-secret environment variables:

| Variable | Meaning |
|---|---|
| `FORNAX_G2_BUNDLE_DIR` | Absolute bundle root |
| `FORNAX_G2_STEP_DIR` | Absolute output directory for the current step |
| `FORNAX_G2_STEP_ID` | V6-V10 step ID |
| `FORNAX_G2_RUN_NONCE` | Runner-issued one-time correlation nonce that must appear in raw measurements |

`{repo_root}` and `{step_dir}` are expanded in `argv` and `cwd`. Do not put
credentials in the manifest or command line; use an approved SSH agent or
external secret mechanism.

Physical execution is blocked unless all of the following are true:

1. the MAX checkout exactly matches the current working-tree root pin and is clean;
2. Fornax execution-affecting source files are committed;
3. every T0/T1 prerequisite passes;
4. the run manifest contains concrete, non-placeholder identities;
5. upstream physical dependencies pass; and
6. `--run-physical` is present.

Fornax source and MAX lineage are checked again after all commands. A command
that changes execution source, the pin, or the MAX checkout invalidates the
packet even if every subprocess returned zero.

The root-pin manifest, reconstruction script, and patch are currently
uncommitted working-tree files. They establish a verifiable mechanism for local
readiness, but become durable repository lineage only after they are committed.

The dependency order is V6 NVIDIA + V6 Apple, then V7 pipeline, then V8 load and
V9 stability. V10 depends on V7. A failed parity step therefore prevents an
unsafe load or stability run.

## Physical run manifest

The manifest is JSON with `schema_version: 1`. Required top-level objects are
`model`, `plan`, `nodes`, `network`, `correctness`, `stability`, and `steps`.
Unknown fields are rejected rather than silently ignored.

### Immutable identity

`model` must contain concrete `model_id` and `snapshot_id` plus SHA-256 values
for the model config, weights manifest, tokenizer, chat template, and prompt
corpus. `plan` must contain a canonical UUID, `plan_artifact`, its SHA-256, an
`evidence_registry_artifact`,
ordered `stage_manifest_artifacts`, and the corresponding ordered SHA-256 list.
Artifact paths are relative to the directory containing the run manifest and
must be contained regular files (not symlinks). The runner recomputes every hash
and copies the actual files into `inputs/`; a string containing a plausible hash
is not sufficient.

The plan artifact is a JSON object containing `schema_version`, `plan_id`, the
exact `model` object, `feasible: true`, `authority`, `stages`, and
`frozen_predictions`. The full authority record must say
`requested_mode: deployment`, `status: deployment_authoritative`, and
`deployment_authorized: true`, and bind its non-empty `source_ids` to
`evidence_registry_sha256`. The runner recomputes the registry hash, resolves
every source ID, rehashes every referenced artifact relative to the registry,
rejects missing/revoked/stale/wrong-type evidence, and requires model,
quantization, expert-trace, capability, measurement, calibration, and route
evidence. The registry and all referenced artifacts must remain contained and
are copied into the bundle. Stages
are ordered from index zero, begin at layer zero, have contiguous non-overlapping
layer ranges, and bind both the NVIDIA and Apple roles to the exact physical host
IDs admitted below. Frozen predictions contain exactly one positive aggregate
tokens/s prediction for inflight 1, 4, and 8.

Each stage manifest repeats the exact model/config/tokenizer/template identity,
plan UUID and recomputed plan hash, stage ID/index/layer range, and a
`node_binding` with role and physical host ID. It also binds the node's pinned
MAX commit and built-binary SHA-256. A mismatched model, cut, host, MAX build, or
exploratory/rejected planner result blocks all physical execution.

`nodes` must contain exactly one `nvidia` and one `apple` node on different
physical hosts. Each records:

- physical host ID and hostname;
- OS build and architecture;
- device identity, memory, and driver/runtime;
- observed MAX CLI, Mojo, Bazel, Bazelisk, Python, compiler, and platform
  toolchain versions;
- build target plus sanitized build-flags and build-environment manifest hashes;
- the root-pinned MAX patch commit; and
- the built MAX binary SHA-256.

`network` records the source/destination physical host IDs, route, interface,
MTU, and declared link rate. These declarations are copied into the bundle and
must be repeated exactly by raw evidence. Raw node identity includes the
declared memory byte count as well as the build/device fields.

### Preapproved correctness and stability policy

The `correctness` object binds three things before the run:

- a concrete reference ID, implementation ID, and reference-artifact SHA-256;
- a concrete tolerance approval ID plus per-dtype (`bf16`, `fp16`, or `fp32`)
  `atol` and `rtol`; and
- a corpus with at least 20 prompts, contexts including 16, 128, 512, and 4096
  tokens, and at least 128 generated tokens per accepted prompt.

The only accepted policies are rejection of non-finite values and exact top-1
and routing identity. Numerical pass/fail is evaluated elementwise as
`abs(observed-reference) <= atol + rtol*abs(reference)`. Physical results cannot
choose a looser tolerance or a different reference after execution.

The `stability` object freezes integer values for duration (at least 1800
seconds), sample interval, target inflight load, minimum completed requests, and
post-drain timeout. These values govern V9; the result cannot substitute its own
sampling/load definition.

All SHA-256 values use `sha256:` followed by 64 lowercase hexadecimal
characters. Values such as `TBD`, `unknown`, `unset`, or `replace-me` are
rejected.

### Step command

Every ID below must have either a runnable command or a concrete blocker:

```json
{
  "steps": {
    "V6_NVIDIA": {
      "status": "READY",
      "argv": ["/approved/bin/run-v6-nvidia", "--out", "{step_dir}"],
      "cwd": "{repo_root}",
      "timeout_seconds": 1800,
      "result_artifact": "result.json"
    },
    "V6_APPLE": {
      "status": "BLOCKED",
      "reason": "Apple lab node is not yet allocated"
    }
  }
}
```

The complete ID set is `V6_NVIDIA`, `V6_APPLE`, `V7_PIPELINE`,
`V8_LOAD_CALIBRATION`, `V9_STABILITY`, and `V10_FAILURES`. The example path and
host reason above are illustrative configuration, not evidence.

## Result contract

Each `READY` command must copy a JSON object to its configured
`result_artifact`. Common fields are:

```json
{
  "schema_version": 1,
  "step_id": "V7_PIPELINE",
  "evidence_class": "T3-physical-multinode",
  "measured": true,
  "physical": true,
  "same_host_proxy": false,
  "passed": true,
  "model": {},
  "plan": {},
  "max_patch_commit": "<root-pinned commit>",
  "observed_nodes": [],
  "physical_host_ids": [],
  "raw_artifacts": ["raw-measurements.json"],
  "raw_measurements_artifact": "raw-measurements.json",
  "checks": {}
}
```

`model`, `plan`, and every observed node must exactly equal the run manifest;
this prevents a successful command for the wrong model, build, device, or plan
from being accepted. The runner additionally enforces:

| Step | Required measured content |
|---|---|
| V6 NVIDIA | operator/stage vectors under the preapproved dtype tolerance, finite values, exact top-1, and exact routing |
| V6 Apple | V6 parity plus integer operator/stage/expert case counts, context coverage, memory high water, and runtime-error count; the Apple role is derived from these criteria |
| V7 | prefill/decode, at least 20 exact-reference prompts covering 16/128/512/4096 contexts and 128 generated tokens, boundary/logit parity, and derived timing percentiles |
| V8 | exactly concurrency 1/4/8; predictions equal the frozen plan inputs; the full correctness corpus passes independently at every point; scaling, attribution, and maximum relative error `<= 0.20` are derived |
| V9 | approved runner-observed duration and exact raw span; contiguous integer sample sequence within cadence; held load; reconciled request and lifecycle counters; integer high-water/bounds; and a clean integer post-drain record |
| V10 | canonical typed cancel, deadline, stale-plan, CRC, and recovered-link-loss codes plus exact cleanup, replay disposition, replay execution count, state hashes, and mutation count |

Physical commands must place a structured `raw_measurements_artifact` beside
`result.json`. It repeats the step/model/plan/MAX identities and runner nonce,
records exact node identities (including memory bytes), includes the configured
link rate and a concrete cross-host transfer sample for multi-node steps, and
carries the numerical vectors, generation outcomes,
timing samples, concurrency samples, stability series/limits, or fault cleanup
observations applicable to the step. The validator derives parity, maximum
error, tolerance/top-1/routing status, corpus correctness, Apple role, timing
percentiles, scaling, planner error, high-water/bounds, request/lifecycle
accounting, and fault/replay/mutation status from those observations; a
hand-written summary boolean cannot override them. Every file below the step
directory is hashed into the bundle, every
declared raw artifact must be a contained regular file, and V9 cannot pass by
merely declaring `duration_seconds: 1800` after exiting immediately.

This correlation and derivation layer makes accidental or summary-only evidence
fail closed. It is not a cryptographic remote-attestation system; Sponsor/TL
review must still verify the lab wrapper, raw host records, and custody of the
durable packet before a formal G2 decision.

## Bundle layout and interpretation

```text
g2-evidence.json             machine-readable decision input
g2-summary.md                operator-readable outcome
inputs/max-lineage.json      exact root pin used
inputs/g2-run-manifest.json  exact physical manifest, when supplied
inputs/physical-plan.json    copied and hash-verified plan artifact
inputs/stage-manifest-*.json copied and hash-verified stage manifests
inputs/planner-evidence-registry.json copied, resolved evidence registry
inputs/planner-evidence-*.json copied, individually rehashed evidence artifacts
logs/*.stdout.log            exact command output
logs/*.stderr.log            exact command errors
physical/<STEP>/**           result and raw physical artifacts
artifact-manifest.json       path, byte count, and SHA-256 for bundle files
artifact-manifest.sha256     integrity hash for the manifest itself
```

Exit `0` means the technical packet satisfied the runner's contract. It does
not close G2 automatically: `gate_decision_authority` remains false and the
Sponsor/TL must review the durable packet. Exit `1` means `BLOCKED` or `FAILED`;
exit `2` means invalid invocation/input.
