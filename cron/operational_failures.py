"""Durable, redacted intake records for owned operational Cron failures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import fcntl

from agent.redact import redact_sensitive_text


INTAKE_FILENAME = "operational-failures.jsonl"
HOST_INTAKE_FILENAME = "operational-failures.jsonl"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_error(value: Any) -> str:
    redacted = redact_sensitive_text(_text(value) or "unspecified failure", force=True)
    # Deterministic command failures commonly carry opaque values under short
    # names that the generic secret redactor cannot classify. Never let a
    # key=value token into the shared control plane.
    redacted = re.sub(
        r"(?i)\b(?:token|secret|password|api[_-]?key|credential)\s*=\s*[^\s,;]+",
        lambda match: match.group(0).split("=", 1)[0] + "=[REDACTED]",
        redacted,
    )
    return redacted[:800]


def profile_failure_event(
    job: dict[str, Any], error: Any, *, outcome: str = "failure", source_scope: str | None = None,
) -> dict[str, Any] | None:
    """Normalize an opted-in Cron failure without selecting a chat target."""
    ownership = job.get("failure_ownership")
    if not isinstance(ownership, dict):
        return None
    workflow_id = _text(job.get("workflow_id") or job.get("workflow_slug") or job.get("runbook_slug"))
    source_id = _text(job.get("id"))
    technical_owner = _text(ownership.get("technical_owner"))
    director = _text(ownership.get("director"))
    missing = [
        name for name, value in (
            ("workflow_id", workflow_id),
            ("source_id", source_id),
            ("technical_owner", technical_owner),
            ("director", director),
        ) if not value
    ]
    status = "invalid_ownership" if missing else outcome
    safe_error = _safe_error(error)
    signature = "\0".join((workflow_id, source_id, status, safe_error))
    return {
        "schema_version": 1,
        "event_id": hashlib.sha256(signature.encode()).hexdigest(),
        "source_kind": "profile_cron",
        "source_id": source_id,
        "source_scope": _text(source_scope) or None,
        "workflow_id": workflow_id,
        "technical_owner": technical_owner or None,
        "director": director or None,
        "severity": _text(ownership.get("severity")) or "warning",
        "attempt": int(job.get("failure_streak") or 0) + 1,
        "dedupe_key": hashlib.sha256(signature.encode()).hexdigest()[:32],
        "sanitized_error": safe_error,
        "evidence_ref": f"cron-output:{source_id}",
        "ack_deadline": ownership.get("ack_deadline"),
        "status": status,
        "missing_fields": missing,
        "recorded_at": int(time.time()),
    }


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append a single intake record before any optional chat delivery."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    # Lock + O_APPEND + fsync gives each intake event a stable source order.
    # The monitor also tolerates a torn final line if a host dies before this
    # block completes.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        # fdopen owns the descriptor after success; only close if it did not.
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def append_profile_failure(
    profile_home: Path, job: dict[str, Any], error: Any, *, outcome: str = "failure",
) -> dict[str, Any] | None:
    event = profile_failure_event(
        job, error, outcome=outcome, source_scope=Path(profile_home).name,
    )
    if event is not None:
        append_event(profile_home / "cron" / INTAKE_FILENAME, event)
    return event


def append_profile_recovery(profile_home: Path, job: dict[str, Any]) -> dict[str, Any] | None:
    """Record a successful opted-in execution after the scheduler ledger."""
    return append_profile_failure(
        profile_home, job, "scheduler execution completed", outcome="recovered",
    )


def host_failure_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a registered deterministic host-job failure.

    Callers supply ownership from the job/runbook registration. This module
    deliberately has no global table of jobs, avoiding a second source of
    scheduling truth beside the existing host configuration.
    """
    required = ("workflow_id", "source_id", "technical_owner", "director")
    values = {name: _text(payload.get(name)) for name in required}
    missing = [name for name, value in values.items() if not value]
    outcome = _text(payload.get("outcome")) or "failure"
    if outcome not in {"failure", "recovered"}:
        outcome = "failure"
    status = "invalid_ownership" if missing else outcome
    safe_error = _safe_error(payload.get("error") or payload.get("sanitized_error"))
    signature = "\0".join(("host_job", values["workflow_id"], values["source_id"], status, safe_error))
    return {
        "schema_version": 1,
        "event_id": hashlib.sha256(signature.encode()).hexdigest(),
        "source_kind": "host_job",
        "source_id": values["source_id"],
        "source_scope": _text(payload.get("source_scope")) or "host",
        "workflow_id": values["workflow_id"],
        "technical_owner": values["technical_owner"] or None,
        "director": values["director"] or None,
        "severity": _text(payload.get("severity")) or "warning",
        "attempt": int(payload.get("attempt") or 1),
        "dedupe_key": hashlib.sha256(signature.encode()).hexdigest()[:32],
        "sanitized_error": safe_error,
        "evidence_ref": _text(payload.get("evidence_ref")) or None,
        "ack_deadline": payload.get("ack_deadline"),
        "status": status,
        "missing_fields": missing,
        "recorded_at": int(time.time()),
    }


def append_host_failure(hermes_home: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Append host failure intake under the existing Hermes state root."""
    event = host_failure_event(payload)
    append_event(Path(hermes_home) / "state" / HOST_INTAKE_FILENAME, event)
    return event
