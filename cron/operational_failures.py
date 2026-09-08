"""Durable, redacted intake records for owned operational Cron failures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows CI
    fcntl = None

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX CI
    msvcrt = None

from agent.redact import redact_sensitive_text


INTAKE_FILENAME = "operational-failures.jsonl"
HOST_INTAKE_FILENAME = "operational-failures.jsonl"
DEFAULT_ACK_TIMEOUT_SECONDS = 15 * 60
DEFAULT_REPAIR_TIMEOUT_SECONDS = 60 * 60
DEFAULT_RECOVERY_SUCCESSES = 2


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


def _positive_int(value: Any, default: int, name: str, errors: list[str]) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(name)
        return default
    if parsed <= 0:
        errors.append(name)
        return default
    return parsed


def _source_order(value: Any = None) -> int:
    """Return a sortable nanosecond timestamp for one execution event."""
    if value not in (None, ""):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            try:
                from datetime import datetime

                numeric = datetime.fromisoformat(
                    str(value).replace("Z", "+00:00")
                ).timestamp()
            except (TypeError, ValueError):
                numeric = 0
        if numeric > 0:
            if numeric >= 1_000_000_000_000_000_000:
                return int(numeric)
            if numeric >= 1_000_000_000_000_000:
                return int(numeric * 1_000)
            if numeric >= 1_000_000_000_000:
                return int(numeric * 1_000_000)
            return int(numeric * 1_000_000_000)
    return time.time_ns()


def _timing(
    config: dict[str, Any], source_order: int, errors: list[str]
) -> tuple[int, int, int]:
    recorded_at = max(1, source_order // 1_000_000_000)
    ack_timeout = _positive_int(
        config.get("ack_timeout_seconds"),
        DEFAULT_ACK_TIMEOUT_SECONDS,
        "ack_timeout_seconds",
        errors,
    )
    repair_timeout = _positive_int(
        config.get("repair_timeout_seconds"),
        DEFAULT_REPAIR_TIMEOUT_SECONDS,
        "repair_timeout_seconds",
        errors,
    )
    if repair_timeout <= ack_timeout:
        errors.append("repair_timeout_seconds")
        repair_timeout = max(DEFAULT_REPAIR_TIMEOUT_SECONDS, ack_timeout + 60)
    recovery_successes = _positive_int(
        config.get("recovery_successes_required"),
        DEFAULT_RECOVERY_SUCCESSES,
        "recovery_successes_required",
        errors,
    )
    return (
        recorded_at + ack_timeout,
        recorded_at + repair_timeout,
        recovery_successes,
    )


def _identity(
    *,
    source_kind: str,
    source_scope: str,
    source_id: str,
    execution_id: Any,
    outcome: str,
) -> tuple[str, str]:
    execution = _text(execution_id)
    if not execution:
        raise ValueError("execution_id is required for operational failure intake")
    value = "\0".join((source_kind, source_scope, source_id, execution, outcome))
    return execution, hashlib.sha256(value.encode()).hexdigest()


def profile_failure_event(
    job: dict[str, Any],
    error: Any,
    *,
    execution_id: str,
    outcome: str = "failure",
    source_scope: str | None = None,
    occurred_at: Any = None,
    failure_type: str = "execution",
    dependency_outcome: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Normalize an opted-in Cron failure without selecting a chat target."""
    ownership = job.get("failure_ownership")
    if not isinstance(ownership, dict):
        return None
    workflow_id = _text(job.get("workflow_id") or job.get("workflow_slug") or job.get("runbook_slug"))
    source_id = _text(job.get("id"))
    technical_owner = _text(ownership.get("technical_owner"))
    director = _text(ownership.get("director"))
    config_errors: list[str] = []
    missing = [
        name for name, value in (
            ("workflow_id", workflow_id),
            ("source_id", source_id),
            ("technical_owner", technical_owner),
            ("director", director),
        ) if not value
    ]
    scope = _text(source_scope) or "default"
    normalized_outcome = outcome if outcome in {"failure", "recovered"} else "failure"
    order = _source_order(occurred_at)
    ack_deadline, checkpoint_at, recovery_successes = _timing(
        ownership, order, config_errors
    )
    missing.extend(config_errors)
    status = "invalid_ownership" if missing else normalized_outcome
    safe_error = _safe_error(error)
    execution, event_id = _identity(
        source_kind="profile_cron",
        source_scope=scope,
        source_id=source_id,
        execution_id=execution_id,
        outcome=normalized_outcome,
    )
    condition = "\0".join((workflow_id, source_id, status, safe_error))
    ownership_value = "\0".join((technical_owner, director))
    event = {
        "schema_version": 1,
        "event_id": event_id,
        "execution_id": execution,
        "source_order": order,
        "source_kind": "profile_cron",
        "source_id": source_id,
        "source_scope": scope,
        "workflow_id": workflow_id,
        "technical_owner": technical_owner or None,
        "director": director or None,
        "ownership_key": hashlib.sha256(ownership_value.encode()).hexdigest()[:32],
        "severity": _text(ownership.get("severity")) or "warning",
        "attempt": int(job.get("failure_streak") or 0) + 1,
        "dedupe_key": hashlib.sha256(condition.encode()).hexdigest()[:32],
        "sanitized_error": safe_error,
        "evidence_ref": f"cron-output:{source_id}",
        "ack_deadline": ack_deadline,
        "checkpoint_at": checkpoint_at,
        "recovery_successes_required": recovery_successes,
        "status": status,
        "failure_type": _text(failure_type) or "execution",
        "missing_fields": missing,
        "recorded_at": order // 1_000_000_000,
    }
    if dependency_outcome is not None:
        # This structure contains only configured tool identities and bounded
        # host classifications. Raw MCP/provider payloads never enter intake.
        event["dependency_outcome"] = dependency_outcome
    return event


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append a single intake record before any optional chat delivery."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    # Lock + O_APPEND + fsync preserves each complete intake record.
    # The monitor also tolerates a torn final line if a host dies before this
    # block completes.
    lock_path = path.with_name(f".{path.name}.lock")
    lock_handle = lock_path.open("a+b")
    try:
        os.chmod(lock_path, 0o600)
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows CI
            lock_handle.seek(0)
            if not lock_handle.read(1):
                lock_handle.write(b"0")
                lock_handle.flush()
            lock_handle.seek(0)
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows CI
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lock_handle.close()


def append_profile_failure(
    profile_home: Path,
    job: dict[str, Any],
    error: Any,
    *,
    execution_id: str,
    outcome: str = "failure",
    occurred_at: Any = None,
    failure_type: str = "execution",
    dependency_outcome: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    event = profile_failure_event(
        job,
        error,
        execution_id=execution_id,
        outcome=outcome,
        source_scope=Path(profile_home).name,
        occurred_at=occurred_at,
        failure_type=failure_type,
        dependency_outcome=dependency_outcome,
    )
    if event is not None:
        if outcome != "recovered":
            try:
                from cron.operational_outcomes import capture_outcome_notice

                notice = capture_outcome_notice(
                    job,
                    source_profile=event["source_scope"],
                    execution_id=event["execution_id"],
                )
                if notice is not None:
                    event["outcome_notice"] = notice
            except Exception:
                # Route capture is subordinate to owned failure intake. The
                # typed failure must persist even when no return route can be
                # resolved safely.
                pass
        append_event(profile_home / "cron" / INTAKE_FILENAME, event)
    return event


def append_profile_recovery(
    profile_home: Path,
    job: dict[str, Any],
    *,
    execution_id: str,
    occurred_at: Any = None,
) -> dict[str, Any] | None:
    """Record a successful opted-in execution after the scheduler ledger."""
    return append_profile_failure(
        profile_home,
        job,
        "scheduler execution completed",
        execution_id=execution_id,
        outcome="recovered",
        occurred_at=occurred_at,
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
    order = _source_order(payload.get("occurred_at"))
    config_errors: list[str] = []
    ack_deadline, checkpoint_at, recovery_successes = _timing(
        payload, order, config_errors
    )
    missing.extend(config_errors)
    status = "invalid_ownership" if missing else outcome
    safe_error = _safe_error(payload.get("error") or payload.get("sanitized_error"))
    scope = _text(payload.get("source_scope")) or "host"
    execution, event_id = _identity(
        source_kind="host_job",
        source_scope=scope,
        source_id=values["source_id"],
        execution_id=payload.get("execution_id"),
        outcome=outcome,
    )
    condition = "\0".join((
        "host_job", values["workflow_id"], values["source_id"], status, safe_error,
    ))
    ownership_value = "\0".join((values["technical_owner"], values["director"]))
    return {
        "schema_version": 1,
        "event_id": event_id,
        "execution_id": execution,
        "source_order": order,
        "source_kind": "host_job",
        "source_id": values["source_id"],
        "source_scope": scope,
        "workflow_id": values["workflow_id"],
        "technical_owner": values["technical_owner"] or None,
        "director": values["director"] or None,
        "ownership_key": hashlib.sha256(ownership_value.encode()).hexdigest()[:32],
        "severity": _text(payload.get("severity")) or "warning",
        "attempt": int(payload.get("attempt") or 1),
        "dedupe_key": hashlib.sha256(condition.encode()).hexdigest()[:32],
        "sanitized_error": safe_error,
        "evidence_ref": _text(payload.get("evidence_ref")) or None,
        "ack_deadline": ack_deadline,
        "checkpoint_at": checkpoint_at,
        "recovery_successes_required": recovery_successes,
        "status": status,
        "missing_fields": missing,
        "recorded_at": order // 1_000_000_000,
    }


def append_host_failure(hermes_home: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Append host failure intake under the existing Hermes state root."""
    event = host_failure_event(payload)
    append_event(Path(hermes_home) / "state" / HOST_INTAKE_FILENAME, event)
    return event
