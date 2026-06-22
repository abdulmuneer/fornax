from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from .io import read_json

RECORD_KIND = "trust-boundary-simulation-contract"
MODE = "t1-simulation"
SIMULATION_METHOD = "node-identity-endpoint-auth-plan-integrity"
AUTH_SCHEME = "hmac-sha256-simulated-capability-token"
REQUIRED_CLAIMS = {
    "cluster_id",
    "node_id",
    "endpoint_id",
    "role",
    "plan_id",
    "plan_hash",
    "nonce",
    "issued_at_s",
    "expires_at_s",
}
REQUIRED_EVENT_KINDS = {
    "identity_manifest_loaded",
    "node_identity_admitted",
    "endpoint_auth_challenge",
    "endpoint_auth_accepted",
    "auth_reject",
    "plan_tag_verified",
    "plan_integrity_reject",
    "replay_nonce_accepted",
    "replay_nonce_reject",
    "cleanup",
}
REQUIRED_REJECT_REASONS = {
    "anonymous_disallowed",
    "unknown_identity",
    "stale_plan_hash",
    "duplicate_nonce",
}
ALLOWED_ROLES = {"stage_worker", "expert_worker"}


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _positive_number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{field} must be a positive number")
        return None
    return float(value)


def _non_negative_number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{field} must be a non-negative number")
        return None
    return float(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _secret_for_identity(cluster_id: str, node_id: str, endpoint_id: str) -> str:
    return f"{cluster_id}:{node_id}:{endpoint_id}:simulated-shared-secret"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sign_claims(claims: dict[str, Any], secret: str) -> str:
    signature = hmac.new(
        secret.encode("utf-8"), _canonical_bytes(claims), hashlib.sha256
    ).hexdigest()
    return "sha256:" + signature


def _identity(
    *,
    cluster_id: str,
    node_id: str,
    logical_host_id: str,
    endpoint_id: str,
    role: str,
    physical_device: str,
) -> dict[str, Any]:
    secret = _secret_for_identity(cluster_id, node_id, endpoint_id)
    return {
        "node_id": node_id,
        "logical_host_id": logical_host_id,
        "endpoint_id": endpoint_id,
        "role": role,
        "physical_device": physical_device,
        "admitted": True,
        "key_id": f"{node_id}-{endpoint_id}-sim-key",
        "public_key_hash": _hash_text(secret),
        "token_scope": ["runtime:connect", "runtime:send", "runtime:receive"],
    }


def _claims(
    *,
    cluster_id: str,
    node_id: str,
    endpoint_id: str,
    role: str,
    plan_id: str,
    plan_hash: str,
    nonce: str,
    issued_at_s: float,
    token_ttl_s: float,
) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "node_id": node_id,
        "endpoint_id": endpoint_id,
        "role": role,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "nonce": nonce,
        "issued_at_s": issued_at_s,
        "expires_at_s": round(issued_at_s + token_ttl_s, 6),
    }


def _attempt(
    *,
    attempt_id: str,
    cluster_id: str,
    node_id: str,
    endpoint_id: str,
    role: str,
    plan_id: str,
    plan_hash: str,
    root_plan_hash: str,
    nonce: str,
    issued_at_s: float,
    token_ttl_s: float,
    status: str,
    reason: str | None = None,
    anonymous: bool = False,
) -> dict[str, Any]:
    if anonymous:
        claims: dict[str, Any] = {
            "cluster_id": cluster_id,
            "anonymous": True,
            "plan_id": plan_id,
            "plan_hash": plan_hash,
            "nonce": nonce,
            "issued_at_s": issued_at_s,
            "expires_at_s": round(issued_at_s + token_ttl_s, 6),
        }
        signature = ""
    else:
        claims = _claims(
            cluster_id=cluster_id,
            node_id=node_id,
            endpoint_id=endpoint_id,
            role=role,
            plan_id=plan_id,
            plan_hash=plan_hash,
            nonce=nonce,
            issued_at_s=issued_at_s,
            token_ttl_s=token_ttl_s,
        )
        signature = _sign_claims(
            claims, _secret_for_identity(cluster_id, node_id, endpoint_id)
        )
    data = {
        "attempt_id": attempt_id,
        "status": status,
        "claims": claims,
        "signature": signature,
        "expected_plan_hash": root_plan_hash,
    }
    if reason is not None:
        data["reason"] = reason
    return data


def simulate_trust_boundary(
    *,
    plan_id: str = "trust-boundary-plan",
    request_id: str = "req-trust-boundary",
    plan_hash: str = "sha256:trust-boundary-plan",
    cluster_id: str = "fornax-sim-cluster",
    token_ttl_s: float = 30.0,
) -> dict[str, Any]:
    """Build a deterministic T1 trust-boundary artifact over two logical hosts."""

    errors: list[str] = []
    _non_empty_string(plan_id, "plan_id", errors)
    _non_empty_string(request_id, "request_id", errors)
    _non_empty_string(plan_hash, "plan_hash", errors)
    _non_empty_string(cluster_id, "cluster_id", errors)
    _positive_number(token_ttl_s, "token_ttl_s", errors)
    if errors:
        raise ValueError("; ".join(errors))

    identities = [
        _identity(
            cluster_id=cluster_id,
            node_id="sim-gpu0",
            logical_host_id="sim-host-0",
            endpoint_id="stage-0",
            role="stage_worker",
            physical_device="cuda:0",
        ),
        _identity(
            cluster_id=cluster_id,
            node_id="sim-gpu1",
            logical_host_id="sim-host-1",
            endpoint_id="stage-1",
            role="stage_worker",
            physical_device="cuda:1",
        ),
        _identity(
            cluster_id=cluster_id,
            node_id="sim-gpu1",
            logical_host_id="sim-host-1",
            endpoint_id="expert-0",
            role="expert_worker",
            physical_device="cuda:1",
        ),
    ]
    attempts = [
        _attempt(
            attempt_id="auth-stage-0",
            cluster_id=cluster_id,
            node_id="sim-gpu0",
            endpoint_id="stage-0",
            role="stage_worker",
            plan_id=plan_id,
            plan_hash=plan_hash,
            root_plan_hash=plan_hash,
            nonce="nonce-stage-0-1",
            issued_at_s=0.010,
            token_ttl_s=token_ttl_s,
            status="accepted",
        ),
        _attempt(
            attempt_id="auth-stage-1",
            cluster_id=cluster_id,
            node_id="sim-gpu1",
            endpoint_id="stage-1",
            role="stage_worker",
            plan_id=plan_id,
            plan_hash=plan_hash,
            root_plan_hash=plan_hash,
            nonce="nonce-stage-1-1",
            issued_at_s=0.011,
            token_ttl_s=token_ttl_s,
            status="accepted",
        ),
        _attempt(
            attempt_id="reject-stale-plan",
            cluster_id=cluster_id,
            node_id="sim-gpu1",
            endpoint_id="stage-1",
            role="stage_worker",
            plan_id=plan_id,
            plan_hash="sha256:stale-trust-boundary-plan",
            root_plan_hash=plan_hash,
            nonce="nonce-stage-1-stale",
            issued_at_s=0.012,
            token_ttl_s=token_ttl_s,
            status="rejected",
            reason="stale_plan_hash",
        ),
        _attempt(
            attempt_id="reject-duplicate-nonce",
            cluster_id=cluster_id,
            node_id="sim-gpu0",
            endpoint_id="stage-0",
            role="stage_worker",
            plan_id=plan_id,
            plan_hash=plan_hash,
            root_plan_hash=plan_hash,
            nonce="nonce-stage-0-1",
            issued_at_s=0.013,
            token_ttl_s=token_ttl_s,
            status="rejected",
            reason="duplicate_nonce",
        ),
        _attempt(
            attempt_id="reject-unknown-identity",
            cluster_id=cluster_id,
            node_id="sim-gpu9",
            endpoint_id="stage-9",
            role="stage_worker",
            plan_id=plan_id,
            plan_hash=plan_hash,
            root_plan_hash=plan_hash,
            nonce="nonce-stage-9-1",
            issued_at_s=0.014,
            token_ttl_s=token_ttl_s,
            status="rejected",
            reason="unknown_identity",
        ),
        _attempt(
            attempt_id="reject-anonymous",
            cluster_id=cluster_id,
            node_id="",
            endpoint_id="",
            role="",
            plan_id=plan_id,
            plan_hash=plan_hash,
            root_plan_hash=plan_hash,
            nonce="nonce-anonymous-1",
            issued_at_s=0.015,
            token_ttl_s=token_ttl_s,
            status="rejected",
            reason="anonymous_disallowed",
            anonymous=True,
        ),
    ]
    events: list[dict[str, Any]] = [
        {
            "kind": "identity_manifest_loaded",
            "timestamp_s": 0.000,
            "plan_id": plan_id,
            "request_id": request_id,
            "cluster_id": cluster_id,
            "identity_count": len(identities),
        },
    ]
    for identity in identities:
        events.append(
            {
                "kind": "node_identity_admitted",
                "timestamp_s": 0.001,
                "plan_id": plan_id,
                "request_id": request_id,
                "node_id": identity["node_id"],
                "endpoint_id": identity["endpoint_id"],
                "role": identity["role"],
            }
        )
    for attempt in attempts:
        claims = attempt["claims"]
        endpoint_id = claims.get("endpoint_id", "anonymous")
        events.append(
            {
                "kind": "endpoint_auth_challenge",
                "timestamp_s": round(float(claims["issued_at_s"]), 6),
                "plan_id": plan_id,
                "request_id": request_id,
                "attempt_id": attempt["attempt_id"],
                "endpoint_id": endpoint_id,
                "nonce": claims.get("nonce"),
            }
        )
        if attempt["status"] == "accepted":
            events.extend(
                [
                    {
                        "kind": "endpoint_auth_accepted",
                        "timestamp_s": round(float(claims["issued_at_s"]) + 0.001, 6),
                        "plan_id": plan_id,
                        "request_id": request_id,
                        "attempt_id": attempt["attempt_id"],
                        "endpoint_id": endpoint_id,
                    },
                    {
                        "kind": "plan_tag_verified",
                        "timestamp_s": round(float(claims["issued_at_s"]) + 0.002, 6),
                        "plan_id": plan_id,
                        "request_id": request_id,
                        "attempt_id": attempt["attempt_id"],
                        "plan_hash": claims["plan_hash"],
                    },
                    {
                        "kind": "replay_nonce_accepted",
                        "timestamp_s": round(float(claims["issued_at_s"]) + 0.003, 6),
                        "plan_id": plan_id,
                        "request_id": request_id,
                        "attempt_id": attempt["attempt_id"],
                        "endpoint_id": endpoint_id,
                        "nonce": claims["nonce"],
                    },
                ]
            )
        else:
            events.append(
                {
                    "kind": "auth_reject",
                    "timestamp_s": round(float(claims["issued_at_s"]) + 0.001, 6),
                    "plan_id": plan_id,
                    "request_id": request_id,
                    "attempt_id": attempt["attempt_id"],
                    "endpoint_id": endpoint_id,
                    "reason": attempt["reason"],
                }
            )
            if attempt["reason"] == "stale_plan_hash":
                events.append(
                    {
                        "kind": "plan_integrity_reject",
                        "timestamp_s": round(float(claims["issued_at_s"]) + 0.002, 6),
                        "plan_id": plan_id,
                        "request_id": request_id,
                        "attempt_id": attempt["attempt_id"],
                        "expected_plan_hash": plan_hash,
                        "rejected_plan_hash": claims["plan_hash"],
                    }
                )
            if attempt["reason"] == "duplicate_nonce":
                events.append(
                    {
                        "kind": "replay_nonce_reject",
                        "timestamp_s": round(float(claims["issued_at_s"]) + 0.002, 6),
                        "plan_id": plan_id,
                        "request_id": request_id,
                        "attempt_id": attempt["attempt_id"],
                        "endpoint_id": endpoint_id,
                        "nonce": claims["nonce"],
                    }
                )
    events.append(
        {
            "kind": "cleanup",
            "timestamp_s": 0.100,
            "plan_id": plan_id,
            "request_id": request_id,
            "auth_state_released": True,
        }
    )
    accepted_attempts = [attempt for attempt in attempts if attempt["status"] == "accepted"]
    rejected_attempts = [attempt for attempt in attempts if attempt["status"] == "rejected"]
    return {
        "version": 1,
        "record_kind": RECORD_KIND,
        "mode": MODE,
        "plan_id": plan_id,
        "request_id": request_id,
        "plan_hash": plan_hash,
        "cluster_id": cluster_id,
        "simulation_method": SIMULATION_METHOD,
        "target_workstreams": ["WS-E3", "WS-E4"],
        "trust_policy": {
            "auth_scheme": AUTH_SCHEME,
            "identity_source": "simulated-node-manifest",
            "allow_anonymous": False,
            "require_endpoint_auth": True,
            "require_plan_hash": True,
            "require_replay_nonce": True,
            "required_claims": sorted(REQUIRED_CLAIMS),
            "token_ttl_s": token_ttl_s,
        },
        "node_identities": identities,
        "auth_attempts": attempts,
        "authenticated_messages": [
            {
                "message_id": "msg-activation-0",
                "message_kind": "activation",
                "source_endpoint_id": "stage-0",
                "destination_endpoint_id": "stage-1",
                "auth_attempt_id": "auth-stage-0",
                "plan_id": plan_id,
                "plan_hash": plan_hash,
                "plan_tag": _hash_text(f"{plan_id}:{plan_hash}:msg-activation-0"),
                "nonce": "nonce-stage-0-1",
            }
        ],
        "events": events,
        "result": {
            "identities_admitted": True,
            "endpoint_auth_enforced": True,
            "anonymous_rejected": True,
            "unknown_identity_rejected": True,
            "stale_plan_rejected": True,
            "duplicate_nonce_rejected": True,
            "plan_integrity_enforced": True,
            "correctness_passed": True,
        },
        "summary": {
            "identity_count": len(identities),
            "accepted_auth_count": len(accepted_attempts),
            "rejected_auth_count": len(rejected_attempts),
            "required_reject_reason_count": len(REQUIRED_REJECT_REASONS),
            "authenticated_message_count": 1,
            "event_count": len(events),
            "anonymous_rejected": True,
            "stale_plan_rejected": True,
            "duplicate_nonce_rejected": True,
            "unknown_identity_rejected": True,
            "correctness_passed": True,
        },
        "note": (
            "T1 trust-boundary simulation: validates node identity, endpoint auth, "
            "plan-integrity tags, and replay-nonce rejection over two logical GPU hosts. "
            "This is not real TLS, mTLS, production auth, or G3 security evidence."
        ),
    }


def _identity_map(
    identities: Any, *, cluster_id: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(identities, list) or not identities:
        errors.append("node_identities must be a non-empty list")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, identity in enumerate(identities):
        field = f"node_identities[{index}]"
        if not isinstance(identity, dict):
            errors.append(f"{field} must be an object")
            continue
        node_id = _non_empty_string(identity.get("node_id"), f"{field}.node_id", errors)
        _non_empty_string(
            identity.get("logical_host_id"), f"{field}.logical_host_id", errors
        )
        endpoint_id = _non_empty_string(
            identity.get("endpoint_id"), f"{field}.endpoint_id", errors
        )
        role = _non_empty_string(identity.get("role"), f"{field}.role", errors)
        _non_empty_string(
            identity.get("physical_device"), f"{field}.physical_device", errors
        )
        _non_empty_string(identity.get("key_id"), f"{field}.key_id", errors)
        if role is not None and role not in ALLOWED_ROLES:
            errors.append(f"{field}.role must be one of {sorted(ALLOWED_ROLES)}")
        if identity.get("admitted") is not True:
            errors.append(f"{field}.admitted must be true")
        scopes = identity.get("token_scope")
        if not isinstance(scopes, list) or "runtime:connect" not in scopes:
            errors.append(f"{field}.token_scope must include runtime:connect")
        if node_id is not None and endpoint_id is not None:
            expected = _hash_text(_secret_for_identity(cluster_id, node_id, endpoint_id))
            if identity.get("public_key_hash") != expected:
                errors.append(f"{field}.public_key_hash must match simulated identity")
            if endpoint_id in result:
                errors.append(f"duplicate endpoint identity: {endpoint_id}")
            result[endpoint_id] = identity
    return result


def _validate_policy(policy: Any, errors: list[str]) -> float | None:
    if not isinstance(policy, dict):
        errors.append("trust_policy must be an object")
        return None
    if policy.get("auth_scheme") != AUTH_SCHEME:
        errors.append(f"trust_policy.auth_scheme must be {AUTH_SCHEME}")
    if policy.get("allow_anonymous") is not False:
        errors.append("trust_policy.allow_anonymous must be false")
    for field in ("require_endpoint_auth", "require_plan_hash", "require_replay_nonce"):
        if policy.get(field) is not True:
            errors.append(f"trust_policy.{field} must be true")
    required = policy.get("required_claims")
    if not isinstance(required, list) or not REQUIRED_CLAIMS.issubset(set(required)):
        errors.append("trust_policy.required_claims missing required claim names")
    return _positive_number(policy.get("token_ttl_s"), "trust_policy.token_ttl_s", errors)


def _check_signature(
    attempt: dict[str, Any],
    identity: dict[str, Any] | None,
    *,
    cluster_id: str,
    field: str,
    errors: list[str],
) -> None:
    claims = attempt.get("claims")
    if identity is None or not isinstance(claims, dict):
        return
    node_id = str(claims.get("node_id", ""))
    endpoint_id = str(claims.get("endpoint_id", ""))
    expected = _sign_claims(claims, _secret_for_identity(cluster_id, node_id, endpoint_id))
    if attempt.get("signature") != expected:
        errors.append(f"{field}.signature must match claims and endpoint identity")


def _validate_claims(
    claims: Any,
    *,
    field: str,
    required: bool,
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(claims, dict):
        errors.append(f"{field}.claims must be an object")
        return {}
    if required:
        missing = REQUIRED_CLAIMS - set(claims)
        if missing:
            errors.append(f"{field}.claims missing required claims: {sorted(missing)}")
    issued = _non_negative_number(claims.get("issued_at_s"), f"{field}.claims.issued_at_s", errors)
    expires = _positive_number(claims.get("expires_at_s"), f"{field}.claims.expires_at_s", errors)
    if issued is not None and expires is not None and expires <= issued:
        errors.append(f"{field}.claims.expires_at_s must be greater than issued_at_s")
    return claims


def validate_trust_boundary_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("mode") != MODE:
        errors.append(f"mode must be {MODE}")
    if data.get("simulation_method") != SIMULATION_METHOD:
        errors.append(f"simulation_method must be {SIMULATION_METHOD}")
    plan_id = _non_empty_string(data.get("plan_id"), "plan_id", errors)
    _non_empty_string(data.get("request_id"), "request_id", errors)
    plan_hash = _non_empty_string(data.get("plan_hash"), "plan_hash", errors)
    cluster_id = _non_empty_string(data.get("cluster_id"), "cluster_id", errors)
    _validate_policy(data.get("trust_policy"), errors)
    identities = _identity_map(
        data.get("node_identities"), cluster_id=cluster_id or "", errors=errors
    )

    attempts = data.get("auth_attempts")
    if not isinstance(attempts, list) or not attempts:
        errors.append("auth_attempts must be a non-empty list")
        attempts = []
    accepted_attempt_ids: set[str] = set()
    accepted_nonces: set[tuple[str, str]] = set()
    reject_reasons: set[str] = set()
    accepted_count = 0
    rejected_count = 0
    for index, attempt in enumerate(attempts):
        field = f"auth_attempts[{index}]"
        if not isinstance(attempt, dict):
            errors.append(f"{field} must be an object")
            continue
        attempt_id = _non_empty_string(attempt.get("attempt_id"), f"{field}.attempt_id", errors)
        status = _non_empty_string(attempt.get("status"), f"{field}.status", errors)
        if status not in {"accepted", "rejected"}:
            errors.append(f"{field}.status must be accepted or rejected")
        claims = _validate_claims(
            attempt.get("claims"), field=field, required=status == "accepted", errors=errors
        )
        anonymous = claims.get("anonymous") is True
        endpoint_id = claims.get("endpoint_id")
        identity = identities.get(str(endpoint_id)) if endpoint_id is not None else None
        if status == "accepted":
            accepted_count += 1
            if attempt_id is not None:
                accepted_attempt_ids.add(attempt_id)
            if anonymous:
                errors.append(f"{field} accepted anonymous claims")
            if identity is None:
                errors.append(f"{field}.claims.endpoint_id references unknown identity")
            else:
                if claims.get("node_id") != identity.get("node_id"):
                    errors.append(f"{field}.claims.node_id must match identity")
                if claims.get("role") != identity.get("role"):
                    errors.append(f"{field}.claims.role must match identity")
            if plan_id is not None and claims.get("plan_id") != plan_id:
                errors.append(f"{field}.claims.plan_id must match root plan_id")
            if plan_hash is not None and claims.get("plan_hash") != plan_hash:
                errors.append(f"{field}.claims.plan_hash must match root plan_hash")
            nonce = claims.get("nonce")
            if not isinstance(nonce, str) or not nonce:
                errors.append(f"{field}.claims.nonce must be non-empty")
            elif isinstance(endpoint_id, str):
                key = (endpoint_id, nonce)
                if key in accepted_nonces:
                    errors.append(f"{field}.claims.nonce duplicates accepted nonce")
                accepted_nonces.add(key)
            _check_signature(
                attempt,
                identity,
                cluster_id=cluster_id or "",
                field=field,
                errors=errors,
            )
        elif status == "rejected":
            rejected_count += 1
            reason = _non_empty_string(attempt.get("reason"), f"{field}.reason", errors)
            if reason is not None:
                reject_reasons.add(reason)
            if reason == "stale_plan_hash":
                if plan_hash is not None and claims.get("plan_hash") == plan_hash:
                    errors.append(f"{field}.claims.plan_hash must differ for stale_plan_hash")
                if attempt.get("expected_plan_hash") != plan_hash:
                    errors.append(f"{field}.expected_plan_hash must match root plan_hash")
                _check_signature(
                    attempt,
                    identity,
                    cluster_id=cluster_id or "",
                    field=field,
                    errors=errors,
                )
            elif reason == "duplicate_nonce":
                nonce = claims.get("nonce")
                if not isinstance(endpoint_id, str) or not isinstance(nonce, str):
                    errors.append(f"{field} duplicate_nonce must include endpoint_id and nonce")
                elif (endpoint_id, nonce) not in accepted_nonces:
                    errors.append(f"{field} duplicate_nonce must match an accepted nonce")
                _check_signature(
                    attempt,
                    identity,
                    cluster_id=cluster_id or "",
                    field=field,
                    errors=errors,
                )
            elif reason == "unknown_identity":
                if identity is not None:
                    errors.append(f"{field} unknown_identity reason references known identity")
            elif reason == "anonymous_disallowed":
                if not anonymous:
                    errors.append(f"{field} anonymous_disallowed must carry anonymous claims")
            elif reason is not None:
                errors.append(f"{field}.reason is not a required reject reason")
    missing_reasons = REQUIRED_REJECT_REASONS - reject_reasons
    if missing_reasons:
        errors.append(f"auth_attempts missing reject reasons: {sorted(missing_reasons)}")

    messages = data.get("authenticated_messages")
    if not isinstance(messages, list) or not messages:
        errors.append("authenticated_messages must be a non-empty list")
        messages = []
    for index, message in enumerate(messages):
        field = f"authenticated_messages[{index}]"
        if not isinstance(message, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(message.get("message_id"), f"{field}.message_id", errors)
        _non_empty_string(message.get("message_kind"), f"{field}.message_kind", errors)
        source = _non_empty_string(
            message.get("source_endpoint_id"), f"{field}.source_endpoint_id", errors
        )
        destination = _non_empty_string(
            message.get("destination_endpoint_id"),
            f"{field}.destination_endpoint_id",
            errors,
        )
        attempt_id = _non_empty_string(
            message.get("auth_attempt_id"), f"{field}.auth_attempt_id", errors
        )
        if source is not None and source not in identities:
            errors.append(f"{field}.source_endpoint_id references unknown identity")
        if destination is not None and destination not in identities:
            errors.append(f"{field}.destination_endpoint_id references unknown identity")
        if attempt_id is not None and attempt_id not in accepted_attempt_ids:
            errors.append(f"{field}.auth_attempt_id must reference an accepted attempt")
        if plan_id is not None and message.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match root plan_id")
        if plan_hash is not None and message.get("plan_hash") != plan_hash:
            errors.append(f"{field}.plan_hash must match root plan_hash")
        if plan_id is not None and plan_hash is not None and message.get("message_id"):
            expected_tag = _hash_text(f"{plan_id}:{plan_hash}:{message['message_id']}")
            if message.get("plan_tag") != expected_tag:
                errors.append(f"{field}.plan_tag must match message plan tag")

    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        events = []
    event_kinds = {event.get("kind") for event in events if isinstance(event, dict)}
    missing_events = REQUIRED_EVENT_KINDS - event_kinds
    if missing_events:
        errors.append(f"events missing required kinds: {sorted(missing_events)}")
    for index, event in enumerate(events):
        field = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{field} must be an object")
            continue
        _non_empty_string(event.get("kind"), f"{field}.kind", errors)
        _non_negative_number(event.get("timestamp_s"), f"{field}.timestamp_s", errors)
        if plan_id is not None and event.get("plan_id") != plan_id:
            errors.append(f"{field}.plan_id must match root plan_id")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
        result = {}
    expected_result_flags = {
        "identities_admitted",
        "endpoint_auth_enforced",
        "anonymous_rejected",
        "unknown_identity_rejected",
        "stale_plan_rejected",
        "duplicate_nonce_rejected",
        "plan_integrity_enforced",
        "correctness_passed",
    }
    for flag in sorted(expected_result_flags):
        if result.get(flag) is not True:
            errors.append(f"result.{flag} must be true")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("identity_count") != len(identities):
        errors.append("summary.identity_count must match node_identities")
    if summary.get("accepted_auth_count") != accepted_count:
        errors.append("summary.accepted_auth_count must match accepted attempts")
    if summary.get("rejected_auth_count") != rejected_count:
        errors.append("summary.rejected_auth_count must match rejected attempts")
    if summary.get("required_reject_reason_count") != len(REQUIRED_REJECT_REASONS):
        errors.append("summary.required_reject_reason_count must match required reasons")
    if summary.get("authenticated_message_count") != len(messages):
        errors.append("summary.authenticated_message_count must match messages")
    if summary.get("event_count") != len(events):
        errors.append("summary.event_count must equal len(events)")
    for flag in (
        "anonymous_rejected",
        "stale_plan_rejected",
        "duplicate_nonce_rejected",
        "unknown_identity_rejected",
        "correctness_passed",
    ):
        if summary.get(flag) is not True:
            errors.append(f"summary.{flag} must be true")

    warnings.append(
        "trust boundary is simulation evidence, not real TLS/mTLS or product auth evidence"
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "identity_count": summary.get("identity_count"),
            "accepted_auth_count": summary.get("accepted_auth_count"),
            "rejected_auth_count": summary.get("rejected_auth_count"),
            "authenticated_message_count": summary.get("authenticated_message_count"),
            "event_count": summary.get("event_count"),
            "anonymous_rejected": summary.get("anonymous_rejected") is True,
            "stale_plan_rejected": summary.get("stale_plan_rejected") is True,
            "duplicate_nonce_rejected": summary.get("duplicate_nonce_rejected") is True,
            "unknown_identity_rejected": summary.get("unknown_identity_rejected") is True,
            "correctness_passed": summary.get("correctness_passed") is True,
        },
    }


def validate_trust_boundary(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if fixture_path.is_dir():
        fixture_path = fixture_path / "fixture.json"
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid trust boundary artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["trust boundary artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_trust_boundary_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
