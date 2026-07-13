# MAX Fork and Build Reproducibility

Plan: `project-plan-v4.md` §5  
Status: Required procedure; root dependency pin pending

## Accepted lineage for current Apple bring-up

| Field | Value |
|---|---|
| Checkout | `external/modular` |
| Upstream/base commit | `0735fa29762a5c53d65a0456d0b53eac1472180f` |
| Current patch commit | `957aeded5296d6638386409849b60f82c36146dd` |
| Reported CLI | `MAX 26.5.0.dev2026063006` |
| Primary build target | `//max/python/max/_entrypoints:pipelines` |
| Apple toolchain | Metal toolchain path recorded in repository `AGENTS.md` |
| Evidence model snapshot | DeepSeek-V2-Lite-Chat `85864749...` |

This table describes the current local lineage. It is not yet a root Fornax pin.

## Root pin requirement

Before T3 evidence is accepted, Fornax must contain one of:

1. a Git submodule pointer to the accepted MAX fork/commit; or
2. a tracked dependency manifest containing repository URL, upstream base, patch
   commit, patch-series hashes, and verified fetch instructions.

A local untracked checkout is insufficient. If the public Fornax repository may
not expose the fork, the pin belongs in a private tracked build manifest available
to every authorized builder.

## Build manifest

Each accepted build records:

- Fornax commit and dirty-state summary;
- MAX repository URL, base, patch commit, and dirty-state summary;
- Bazel/Bazelisk, Python, Pixi, MAX, Mojo, compiler, and Metal/CUDA versions;
- OS build, CPU architecture, GPU/device identity, and driver/runtime;
- build target, flags, environment variables, output binary hash;
- kernel/test targets executed and their logs/hashes;
- model snapshot and exact invocation.

## Fresh-build procedure

1. Create a clean worktree or clone from the recorded dependency pin.
2. Verify base and patch commits and an empty MAX worktree.
3. Configure the documented workspace-local Bazel output base.
4. Build the source MAX CLI.
5. Run focused Apple MLA prefill/decode, MoE-index, gather, and Stage ABI tests.
6. Run numerical references before any model-generation smoke.
7. Run short and 128-token model generation.
8. Hash the binary, manifests, commands, and logs into an evidence bundle.

## Rebase and upstream policy

- Keep Fornax-required MAX patches as a small reviewable series by subsystem.
- Every rebase reruns compile, numerical, model, and Stage ABI tests.
- A passing token-generation smoke cannot override a failed numerical test.
- Prefer public graph/custom-op/model-extension APIs.
- Internal kernel changes require an upstream issue/PR or a written reason they
  remain fork-only.
- A rebase that expands the patch surface materially triggers ADR-0001 review.

## Cross-node compatibility

Linux and Apple workers may use platform-specific binaries, but they must report:

- compatible Stage ABI major/minor;
- identical model/config/tokenizer/template/plan hashes;
- an accepted MAX build compatibility set recorded by the plan;
- passing logical tensor conformance.

Binary equality across operating systems is neither possible nor required;
reproducible source lineage and ABI/correctness compatibility are required.
