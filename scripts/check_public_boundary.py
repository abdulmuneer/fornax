#!/usr/bin/env python3
"""Fail closed when a public Fornax change crosses the publication boundary."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath


FORBIDDEN_PREFIXES = (
    ".agents/",
    ".claude/",
    "docs/fornax/fundraising/",
    "docs/fornax/program_management/internal/",
    "external/",
)
FORBIDDEN_BASENAMES = {
    ".env",
    ".private-remote-approved",
    "AGENTS.local.md",
}
FORBIDDEN_SUFFIXES = (
    ".age",
    ".jks",
    ".kdbx",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
)
POLICY_FILES = {
    ".github/workflows/public-boundary.yml",
    ".githooks/pre-commit",
    "AGENTS.md",
    "scripts/check_public_boundary.py",
}
FORBIDDEN_PHRASES = (
    "not approved for external circulation",
    "internal data room",
    "fornax confidential",
)
CLASSIFICATION_RE = re.compile(
    r"(?im)^\s*(?:[#>*_-]+\s*)*(?:classification|status)\s*:\s*"
    r"(?:fornax\s+)?(?:confidential|private|internal)\b"
)
REVIEW_TOPIC_RE = re.compile(
    r"(?i)\b(?:fundrais\w*|investor\w*|valuation|runway|cap\s+table|"
    r"term\s+sheet|customer\s+discovery|pricing|data\s+room)\b"
)
PUBLICATION_MARKER = "<!-- fornax-publication: public -->"


def git(*args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def nul_paths(payload: bytes) -> list[str]:
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(b"\0")
        if item
    ]


def changed_paths(range_spec: str | None) -> tuple[list[str], set[str], str]:
    if range_spec:
        paths = nul_paths(
            git("diff", "--name-only", "-z", "--diff-filter=ACMR", range_spec)
        )
        added = set(
            nul_paths(
                git("diff", "--name-only", "-z", "--diff-filter=A", range_spec)
            )
        )
        return paths, added, "HEAD"

    paths = nul_paths(
        git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    )
    added = set(
        nul_paths(git("diff", "--cached", "--name-only", "-z", "--diff-filter=A"))
    )
    return paths, added, ""


def blob(path: str, revision: str) -> str:
    object_name = f"{revision}:{path}" if revision else f":{path}"
    payload = git("show", object_name)
    if b"\0" in payload:
        return ""
    return payload.decode("utf-8", errors="replace")


def added_text(path: str, range_spec: str | None) -> str:
    args = ["diff"]
    if range_spec:
        args.append(range_spec)
    else:
        args.append("--cached")
    args.extend(("--unified=0", "--no-ext-diff", "--", path))
    patch = git(*args).decode("utf-8", errors="replace")
    return "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def path_errors(path: str) -> list[str]:
    lowered = path.casefold()
    basename = PurePosixPath(path).name
    errors: list[str] = []

    if path in FORBIDDEN_BASENAMES or basename in FORBIDDEN_BASENAMES:
        errors.append("machine-local or private-control file")
    if basename.startswith(".env."):
        errors.append("environment file")
    if lowered.endswith(FORBIDDEN_SUFFIXES):
        errors.append("restricted credential/key extension")
    if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        errors.append("forbidden public-repository path")
    return errors


def content_errors(path: str, text: str, review_text: str) -> list[str]:
    if not text or path in POLICY_FILES:
        return []

    folded = text.casefold()
    errors = [
        f"contains restricted phrase: {phrase!r}"
        for phrase in FORBIDDEN_PHRASES
        if phrase in folded
    ]
    if CLASSIFICATION_RE.search(text):
        errors.append("declares an internal/private/confidential classification")
    if (
        REVIEW_TOPIC_RE.search(review_text)
        and PUBLICATION_MARKER not in text
    ):
        errors.append(
            "new publication-sensitive material lacks an explicitly approved "
            f"{PUBLICATION_MARKER} marker"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check staged or range-based changes for public-boundary violations."
    )
    parser.add_argument(
        "--range",
        dest="range_spec",
        help="scan paths changed in a Git range (CI mode); default scans the index",
    )
    args = parser.parse_args()

    try:
        paths, added, revision = changed_paths(args.range_spec)
        violations: list[tuple[str, str]] = []
        for path in paths:
            for error in path_errors(path):
                violations.append((path, error))
            text = blob(path, revision)
            review_text = text if path in added else added_text(path, args.range_spec)
            for error in content_errors(path, text, review_text):
                violations.append((path, error))
    except RuntimeError as error:
        print(f"public-boundary: ERROR: {error}", file=sys.stderr)
        return 2

    if violations:
        print("public-boundary: BLOCKED", file=sys.stderr)
        for path, error in violations:
            print(f"  {path}: {error}", file=sys.stderr)
        print(
            "Move private material to the approved private repository or obtain "
            "an explicit disclosure decision for a sanitized public version.",
            file=sys.stderr,
        )
        return 1

    scope = args.range_spec or "staged changes"
    print(f"public-boundary: PASS ({len(paths)} paths checked; {scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
