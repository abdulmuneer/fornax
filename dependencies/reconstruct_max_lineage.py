#!/usr/bin/env python3
"""Reconstruct the root-pinned MAX commit from its public base and tracked diff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _run(
    argv: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    stdin: str | None = None,
) -> str:
    completed = subprocess.run(
        list(argv),
        env=environment,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        command = " ".join(argv)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command}\n{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def reconstruct(
    *,
    repository_root: Path,
    checkout: Path,
    manifest_path: Path,
    source_repository: str | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository = manifest["repository"]
    lineage = manifest["lineage"]
    reconstruction = lineage["reconstruction"]
    base = lineage["upstream_base_commit"]
    expected_base_tree = lineage["upstream_base_tree"]
    expected_commit = lineage["patch_commit"]
    expected_tree = lineage["patch_tree"]
    patch_path = repository_root / lineage["patch_file"]
    if checkout.exists():
        raise ValueError(f"checkout already exists: {checkout}")
    if not patch_path.is_file():
        raise ValueError(f"tracked patch file is missing: {patch_path}")
    observed_patch_hash = _sha256(patch_path)
    if observed_patch_hash != lineage["patch_diff_sha256"]:
        raise ValueError(
            "tracked patch hash mismatch: "
            f"expected {lineage['patch_diff_sha256']}, observed {observed_patch_hash}"
        )

    source = source_repository or repository["url"]
    checkout.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--no-checkout", source, str(checkout)])
    if source_repository is not None:
        _run(
            [
                "git",
                "-C",
                str(checkout),
                "remote",
                "set-url",
                repository.get("remote", "origin"),
                repository["url"],
            ]
        )
    _run(["git", "-C", str(checkout), "checkout", "--detach", base])
    observed_base_tree = _run(["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"])
    if observed_base_tree != expected_base_tree:
        raise ValueError(
            f"base tree mismatch: expected {expected_base_tree}, observed {observed_base_tree}"
        )
    _run(
        [
            "git",
            "-C",
            str(checkout),
            "apply",
            "--index",
            "--binary",
            str(patch_path),
        ]
    )
    observed_tree = _run(["git", "-C", str(checkout), "write-tree"])
    if observed_tree != expected_tree:
        raise ValueError(
            f"patched tree mismatch: expected {expected_tree}, observed {observed_tree}"
        )

    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_NAME": reconstruction["author_name"],
            "GIT_AUTHOR_EMAIL": reconstruction["author_email"],
            "GIT_AUTHOR_DATE": reconstruction["author_date"],
            "GIT_COMMITTER_NAME": reconstruction["committer_name"],
            "GIT_COMMITTER_EMAIL": reconstruction["committer_email"],
            "GIT_COMMITTER_DATE": reconstruction["committer_date"],
        }
    )
    observed_commit = _run(
        ["git", "-C", str(checkout), "commit-tree", observed_tree, "-p", base],
        environment=environment,
        stdin=reconstruction["message"] + "\n",
    )
    if observed_commit != expected_commit:
        raise ValueError(
            f"reconstructed commit mismatch: expected {expected_commit}, observed {observed_commit}"
        )
    _run(["git", "-C", str(checkout), "reset", "--hard", observed_commit])
    status = _run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    if status:
        raise ValueError(f"reconstructed checkout is dirty: {status}")
    return {
        "ok": True,
        "checkout": str(checkout.resolve()),
        "repository_url": repository["url"],
        "base_commit": base,
        "base_tree": observed_base_tree,
        "patch_commit": observed_commit,
        "patch_tree": observed_tree,
        "patch_sha256": observed_patch_hash,
        "used_local_object_source": source_repository is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="reconstruct the exact MAX commit pinned by Fornax"
    )
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--manifest", default="dependencies/max-lineage.json")
    parser.add_argument(
        "--source-repository",
        help="optional local Git object source for offline verification; origin is reset to the pinned URL",
    )
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parent.parent
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = repository_root / manifest
    try:
        result = reconstruct(
            repository_root=repository_root,
            checkout=Path(args.checkout).resolve(),
            manifest_path=manifest.resolve(),
            source_repository=args.source_repository,
        )
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"MAX lineage reconstruction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
