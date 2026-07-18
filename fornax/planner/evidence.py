from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVIDENCE_REGISTRY_SCHEMA = "fornax.planner-evidence-registry.v1"
EVIDENCE_TYPES = frozenset(
    {
        "model",
        "quantization",
        "expert_trace",
        "capability",
        "measurement",
        "calibration",
        "route",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_FIELDS = frozenset(
    {
        "source_id",
        "evidence_type",
        "artifact_path",
        "artifact_sha256",
        "status",
        "not_before",
        "expires_at",
    }
)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def _timestamp(value: object, label: str) -> datetime | None:
    if value is None:
        return None
    raw = _required_string(value, label)
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_issue(path: Path, expected_sha256: str) -> str | None:
    try:
        if not path.is_file():
            return f"artifact is not a regular file: {path}"
        observed_sha256 = _sha256_file(path)
    except OSError as exc:
        return f"artifact cannot be read: {path}: {exc}"
    if observed_sha256 != expected_sha256:
        return (
            f"artifact SHA-256 mismatch: declared={expected_sha256} "
            f"observed={observed_sha256}"
        )
    return None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"evidence registry contains duplicate key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class EvidenceRecord:
    """One independently stored, content-verified planner evidence record."""

    source_id: str
    evidence_type: str
    artifact_path: str
    artifact_sha256: str
    status: str = "active"
    not_before: datetime | None = None
    expires_at: datetime | None = None
    artifact_verification_issue: str | None = field(
        default=None, repr=False, compare=False
    )
    resolved_artifact_path: Path | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _required_string(self.source_id, "evidence source_id")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(
                f"unsupported evidence_type {self.evidence_type!r}; "
                f"expected one of {sorted(EVIDENCE_TYPES)}"
            )
        _required_string(self.artifact_path, "evidence artifact_path")
        if not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise ValueError(
                "evidence artifact_sha256 must be exactly 64 lowercase hex characters"
            )
        if self.status not in {"active", "revoked"}:
            raise ValueError("evidence status must be 'active' or 'revoked'")
        if (
            self.not_before is not None
            and self.expires_at is not None
            and self.expires_at <= self.not_before
        ):
            raise ValueError("evidence expires_at must be later than not_before")

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        base_dir: Path,
    ) -> "EvidenceRecord":
        if not isinstance(value, dict):
            raise ValueError("evidence registry records must be objects")
        unknown = sorted(set(value) - _RECORD_FIELDS)
        if unknown:
            raise ValueError(f"evidence record has unknown fields: {unknown}")
        source_id = _required_string(value.get("source_id"), "evidence source_id")
        evidence_type = _required_string(
            value.get("evidence_type"), f"evidence {source_id} evidence_type"
        )
        artifact_path = _required_string(
            value.get("artifact_path"), f"evidence {source_id} artifact_path"
        )
        artifact_sha256 = _required_string(
            value.get("artifact_sha256"), f"evidence {source_id} artifact_sha256"
        )
        status = _required_string(
            value.get("status"), f"evidence {source_id} status"
        )
        declared_path = Path(artifact_path)
        resolved_path = (
            declared_path
            if declared_path.is_absolute()
            else base_dir / declared_path
        )
        verification_issue = _artifact_issue(resolved_path, artifact_sha256)
        return cls(
            source_id=source_id,
            evidence_type=evidence_type,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            status=status,
            not_before=_timestamp(
                value.get("not_before"), f"evidence {source_id} not_before"
            ),
            expires_at=_timestamp(
                value.get("expires_at"), f"evidence {source_id} expires_at"
            ),
            artifact_verification_issue=verification_issue,
            resolved_artifact_path=resolved_path,
        )


@dataclass(frozen=True)
class EvidenceRegistry:
    """Resolver for the deployment planner's separate evidence trust input.

    Loading a registry hashes every referenced artifact. A declaration in a
    model, inventory, link probe, or target is therefore only a reference; it
    cannot authorize deployment unless this resolver has matched it to an
    active, non-stale record whose artifact bytes match the declared SHA-256.
    """

    records: tuple[EvidenceRecord, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        source_ids = [record.source_id for record in self.records]
        if len(source_ids) != len(set(source_ids)):
            duplicates = sorted(
                source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1
            )
            raise ValueError(f"evidence registry source_ids must be unique: {duplicates}")
        if any(record.resolved_artifact_path is None for record in self.records):
            raise ValueError(
                "evidence registry records must be loaded through "
                "EvidenceRegistry.from_file or EvidenceRegistry.from_dict"
            )
        expected_prefix = "sha256:"
        digest = self.manifest_sha256.removeprefix(expected_prefix)
        if not self.manifest_sha256.startswith(expected_prefix) or not _SHA256_RE.fullmatch(
            digest
        ):
            raise ValueError("evidence registry manifest_sha256 must be sha256:<64 hex>")

    @classmethod
    def from_file(cls, path: str | Path) -> "EvidenceRegistry":
        registry_path = Path(path)
        try:
            raw = registry_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"evidence registry cannot be read: {registry_path}: {exc}"
            ) from exc
        try:
            data = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except UnicodeDecodeError as exc:
            raise ValueError("evidence registry must be UTF-8 JSON") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid evidence registry JSON: {exc}") from exc
        return cls._from_parsed(
            data,
            base_dir=registry_path.parent,
            manifest_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        )

    @classmethod
    def from_dict(
        cls, value: object, *, base_dir: str | Path = "."
    ) -> "EvidenceRegistry":
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return cls._from_parsed(
            value,
            base_dir=Path(base_dir),
            manifest_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
        )

    @classmethod
    def _from_parsed(
        cls,
        value: object,
        *,
        base_dir: Path,
        manifest_sha256: str,
    ) -> "EvidenceRegistry":
        if not isinstance(value, dict):
            raise ValueError("evidence registry root must be an object")
        unknown = sorted(set(value) - {"schema_version", "records"})
        if unknown:
            raise ValueError(f"evidence registry has unknown fields: {unknown}")
        if value.get("schema_version") != EVIDENCE_REGISTRY_SCHEMA:
            raise ValueError(
                "evidence registry schema_version must be "
                f"{EVIDENCE_REGISTRY_SCHEMA!r}"
            )
        raw_records = value.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("evidence registry records must be an array")
        records = tuple(
            EvidenceRecord.from_dict(record, base_dir=base_dir)
            for record in raw_records
        )
        return cls(records=records, manifest_sha256=manifest_sha256)

    def record(self, source_id: str) -> EvidenceRecord | None:
        return next(
            (record for record in self.records if record.source_id == source_id),
            None,
        )

    def resolution_issues(
        self,
        source_id: str,
        *,
        evidence_type: str,
        label: str,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        record = self.record(source_id)
        if record is None:
            return (f"{label} source_id={source_id!r} is absent from evidence registry",)
        issues: list[str] = []
        if record.evidence_type != evidence_type:
            issues.append(
                f"{label} source_id={source_id!r} resolves as "
                f"evidence_type={record.evidence_type!r}, expected {evidence_type!r}"
            )
        if record.status != "active":
            issues.append(
                f"{label} source_id={source_id!r} is {record.status}"
            )
        artifact_issue = (
            _artifact_issue(
                record.resolved_artifact_path,
                record.artifact_sha256,
            )
            if record.resolved_artifact_path is not None
            else "artifact path was not resolved by the evidence loader"
        )
        if artifact_issue is not None:
            issues.append(
                f"{label} source_id={source_id!r} has unverified artifact: "
                f"{artifact_issue}"
            )
        observed_at = now or datetime.now(timezone.utc)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("evidence resolution time must include a timezone")
        observed_at = observed_at.astimezone(timezone.utc)
        if record.not_before is not None and observed_at < record.not_before:
            issues.append(
                f"{label} source_id={source_id!r} is not active until "
                f"{record.not_before.isoformat()}"
            )
        if record.expires_at is not None and observed_at >= record.expires_at:
            issues.append(
                f"{label} source_id={source_id!r} is stale; expired at "
                f"{record.expires_at.isoformat()}"
            )
        return tuple(issues)
