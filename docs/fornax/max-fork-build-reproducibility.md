# MAX Fork and Build Reproducibility

Plan: `project-plan-v4.md` §5  
Status: Working-tree root-pin mechanism verified; repository commit, fresh-build, and physical rebuild evidence pending

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

The same values are encoded in the current working-tree
[`dependencies/max-lineage.json`](../../dependencies/max-lineage.json). The G2
runner verifies the manifest against the local checkout, including remote URL,
base ancestry, HEAD/tree, patch-parent, binary-diff SHA-256, and clean state. A
pin mechanism closes the format/verification gap. It does not become durable
repository lineage until committed, and it does not by itself prove a fresh
rebuild or physical backend correctness.

Because the Apple patch commit is local rather than asserted to exist on the
public Modular remote, the working tree also contains the exact binary Git diff at
`dependencies/max-patches/957aeded5296d6638386409849b60f82c36146dd.diff`
and deterministic commit metadata. Reconstruct a fresh checkout with:

```bash
python3 dependencies/reconstruct_max_lineage.py \
  --checkout /new/path/to/modular
```

The script clones the pinned public repository, checks out the base, verifies
and applies the pinned diff, reconstructs the exact commit, and rejects any
base-tree, patch-tree, commit, or cleanliness mismatch. Its offline local-object
mode reproduced the exact accepted commit on 2026-07-17; that check did not
assert a fresh public-network fetch.

## Root pin requirement

Before T3 evidence is accepted, Fornax must contain one of:

1. a Git submodule pointer to the accepted MAX fork/commit; or
2. a tracked dependency manifest containing repository URL, upstream base, patch
   commit, patch-series hashes, and verified fetch instructions.

A local untracked checkout is insufficient. If the public Fornax repository may
not expose the fork, the pin belongs in a private tracked build manifest available
to every authorized builder.

The working tree implements option 2; check it into the repository before
claiming this G2 entry condition is durable. Run:

```bash
python3 -m fornax program g2-validate --out-dir evidence/g2-readiness-YYYYMMDD-HHMMSS
```

The command verifies the root pin before any physical command is eligible to
run. See [`g2-in-a-box.md`](g2-in-a-box.md).

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
